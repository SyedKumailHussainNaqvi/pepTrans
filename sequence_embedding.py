import os
import re
import h5py
import torch
import numpy as np
import pandas as pd
from transformers import T5Tokenizer, T5EncoderModel


def clean_sequence(seq):
    seq = str(seq).strip().upper()
    seq = re.sub(r"[UZOB]", "X", seq)
    seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWYX]", "", seq)
    return seq


def load_prott5(device="cuda"):
    model_name = "Rostlab/prot_t5_xl_uniref50"
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def embed_sequence(seq, tokenizer, model, device="cuda"):
    seq = clean_sequence(seq)
    spaced_seq = " ".join(list(seq))

    ids = tokenizer(
        spaced_seq,
        return_tensors="pt",
        add_special_tokens=True,
        padding=True
    )

    input_ids = ids["input_ids"].to(device)
    attention_mask = ids["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    emb = outputs.last_hidden_state.squeeze(0)

    # remove special token
    emb = emb[:len(seq)]

    return emb.cpu().numpy().astype(np.float32)


def mean_pool_embedding(emb):
    return emb.mean(axis=0)


def generate_binary_csv_from_sequences(
    input_csv,
    output_csv,
    peptide_col="Peptide",
    protein_col="Protein",
    label_col=None,
    device="cuda"
):
    """
    Creates 2048-dim feature CSV:
    1024 peptide mean embedding + 1024 protein mean embedding.
    Compatible with current pepTrans binary / affinity / PBD / VS pipeline.
    """

    df = pd.read_csv(input_csv)

    tokenizer, model = load_prott5(device=device)

    rows = []

    for idx, row in df.iterrows():
        pep_seq = row[peptide_col]
        prot_seq = row[protein_col]

        pep_emb = embed_sequence(pep_seq, tokenizer, model, device=device)
        prot_emb = embed_sequence(prot_seq, tokenizer, model, device=device)

        pep_vec = mean_pool_embedding(pep_emb)
        prot_vec = mean_pool_embedding(prot_emb)

        feature_vec = np.concatenate([pep_vec, prot_vec], axis=0)

        out_row = {"IDs": row.get("IDs", f"sample_{idx}")}

        for i, val in enumerate(feature_vec):
            out_row[f"F{i}"] = val

        if label_col is not None and label_col in df.columns:
            out_row[label_col] = row[label_col]

        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)

    print(f"Saved embedding CSV: {output_csv}")
    return output_csv


def pad_or_truncate_embedding(emb, max_len):
    dim = emb.shape[1]

    if emb.shape[0] > max_len:
        return emb[:max_len]

    padded = np.zeros((max_len, dim), dtype=np.float32)
    padded[:emb.shape[0]] = emb
    return padded


def generate_binding_site_h5_from_sequences(
    input_csv,
    output_h5,
    peptide_col="Peptide",
    protein_col="Protein",
    peptide_label_col=None,
    max_peptide_len=50,
    max_protein_len=500,
    device="cuda"
):
    """
    Creates H5 file compatible with current binding-site model:
    peptide_data: N x 50 x 1024
    protein_data: N x 500 x 1024
    labels:       N x 50
    """

    df = pd.read_csv(input_csv)

    tokenizer, model = load_prott5(device=device)

    peptide_data = []
    protein_data = []
    labels = []

    for idx, row in df.iterrows():
        pep_seq = clean_sequence(row[peptide_col])
        prot_seq = clean_sequence(row[protein_col])

        pep_emb = embed_sequence(pep_seq, tokenizer, model, device=device)
        prot_emb = embed_sequence(prot_seq, tokenizer, model, device=device)

        pep_pad = pad_or_truncate_embedding(pep_emb, max_peptide_len)
        prot_pad = pad_or_truncate_embedding(prot_emb, max_protein_len)

        peptide_data.append(pep_pad)
        protein_data.append(prot_pad)

        if peptide_label_col is not None and peptide_label_col in df.columns:
            label_str = str(row[peptide_label_col]).replace(" ", "")
            label = np.array([int(x) for x in label_str[:max_peptide_len]], dtype=np.float32)

            padded_label = np.full((max_peptide_len,), -1, dtype=np.float32)
            padded_label[:len(label)] = label
        else:
            padded_label = np.zeros((max_peptide_len,), dtype=np.float32)

        labels.append(padded_label)

    with h5py.File(output_h5, "w") as h5:

        for i in range(len(peptide_data)):

            grp = h5.create_group(f"sample_{i}")

            grp.create_dataset(
                "peptide_embeddings",
                data=peptide_data[i],
                dtype=np.float32
            )

            grp.create_dataset(
                "protein_embeddings",
                data=protein_data[i],
                dtype=np.float32
            )

            grp.create_dataset(
                "labels",
                data=np.expand_dims(labels[i], axis=-1),
                dtype=np.float32
            )

    print(f"Saved binding-site H5: {output_h5}")
    return output_h5


