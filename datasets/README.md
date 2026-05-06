# [pepTrans Datasets](https://)
Datasets for pepTrans: Benchmark Datasets for Multi-Level Peptide–Protein Interaction Prediction

--------------------

## Overview

The `datasets/` directory contains all benchmark datasets, clustering-based cross-validation splits, residue-level annotations, and binding affinity datasets used in the **pepTrans** framework for multi-level peptide–protein interaction (pepPI) prediction.

These datasets are publicly released to improve reproducibility, benchmarking transparency, downstream usability, and independent evaluation of the proposed framework.

The released datasets correspond to the experiments reported in the pepTrans manuscript and supplementary materials.

---

## Dataset Directory Structure

```text
datasets/
│
├── pepPI binary interaction/
│   ├── benchmark dataset/
│   ├── shuffled benchmark dataset/
│   ├── cluster wise dataset pepPI binary interaction/
│   │   ├── novel protein/
│   │   ├── novel peptide/
│   │   └── novel pair/
│
├── peptide binding residue prediction/
│
└── pepPI binding affinity/
```

---

# 1. pepPI Binary Interaction

This directory contains datasets used for binary peptide–protein interaction prediction experiments.

Each entry represents a peptide–protein pair associated with a binary interaction label.

## Dataset Format

| Column | Description |
|---|---|
| `Protein_ID` | Protein receptor identifier |
| `Peptide_ID` | Peptide identifier |
| `Label` | Binary interaction label |

---

## PDB and Chain Identifier Convention

Protein and peptide identifiers follow the format:

```text
PDBID_PROTEINCHAIN_PEPTIDECHAIN
```

### Example Identifiers

```text
6KKB_D_X
6JWV_C_A
6PZC_C_A
```

where:

- `6KKB` = Protein Data Bank (PDB) structure identifier
- `D` = Protein receptor chain
- `X` = Peptide chain

The corresponding receptor and peptide sequences can be directly extracted from the associated experimentally resolved PDB structures using the provided chain identifiers.

---

## Sequence Reconstruction

Some benchmark datasets contain PDB-chain identifiers instead of explicit amino acid sequences. In such cases, the corresponding receptor and peptide sequences can either:

- be directly extracted from the benchmark datasets provided in the `pepPI binary interaction` directory (for sequence-level versions), or
- be reconstructed from experimentally resolved PDB structures using the associated chain identifiers.

For improved reproducibility and downstream usability, the pepTrans repository additionally provides sequence-level datasets wherever applicable.

---

## Binary Interaction Labels

| Label | Meaning |
|---|---|
| `1` | Interacting peptide–protein pair |
| `0` | Non-interacting peptide–protein pair |

---

# 2. Shuffled Benchmark Dataset

This directory contains shuffled peptide–protein datasets generated for negative sampling and robustness evaluation under varying positive-to-negative ratios.

## Example Datasets

```text
1_1_shuffled_Test_dataset.txt
1_5_positive_negative.txt
1_10_shuffled_Test_dataset.txt
1_20_shuffled_Test_dataset.txt
```

These datasets were used for imbalance robustness experiments reported in the manuscript.

---

# 3. Cluster-wise Dataset for pepPI Binary Interaction

This directory contains the complete train/validation/test splits used for cluster-based cross-validation experiments reported in the manuscript.

The clustering strategy follows the same protocol adopted in the CAMP framework using Smith–Waterman similarity-based clustering to reduce sequence redundancy between training and testing sets.

---

## Evaluation Settings

| Evaluation Setting | Description |
|---|---|
| `novel protein` | Protein clusters are separated between training and testing sets |
| `novel peptide` | Peptide clusters are separated between training and testing sets |
| `novel pair` | Neither peptide nor protein clusters overlap between training and testing sets |

---

## Clustering Thresholds

Each evaluation setting contains clustering thresholds:

```text
0.3
0.4
0.5
0.6
```

Higher thresholds correspond to increasingly challenging generalization settings with lower sequence similarity between training and testing samples.

---

## Fold Organization

Each threshold directory contains:

| File | Description |
|---|---|
| `fold_X_train_ids.txt` | Training set identifiers |
| `fold_X_valid_ids.txt` | Validation set identifiers |
| `fold_X_test_ids.txt` | Independent testing set identifiers |

where `X` denotes the fold index.

These split files correspond exactly to the experiments reported in the manuscript and supplementary materials.

---

# 4. Peptide Binding Residue Prediction

This directory contains datasets used for residue-level peptide binding residue prediction.

Each sample includes:

- peptide sequence
- protein receptor sequence
- residue-level peptide binding annotations

---

## Binding Residue Labels

Binding residue labels are represented as binary vectors:

| Value | Meaning |
|---|---|
| `1` | Binding residue |
| `0` | Non-binding residue |

These datasets were used for residue-level binding site prediction experiments.

---

# 5. pepPI Binding Affinity

This directory contains datasets used for peptide–protein binding affinity prediction.

## Dataset Format

| Column | Description |
|---|---|
| `Protein_ID` | Protein receptor identifier |
| `Peptide_ID` | Peptide identifier |
| `Log_Affinity` | Log-transformed experimental binding affinity |

The affinity values correspond to experimentally measured peptide–protein binding strengths collected from curated structural datasets.

These datasets were used for binding affinity prediction experiments reported in the manuscript.

---

# Reproducibility and Data Availability

The datasets released in this repository correspond to the exact datasets, clustering thresholds, and fold assignments used in the pepTrans manuscript.

The repository includes:

- benchmark datasets
- shuffled benchmark datasets
- cluster-wise train/validation/test splits
- residue-level binding annotations
- binding affinity datasets
- fold assignments for all clustering thresholds

This organization enables exact reproduction of all experiments reported in the manuscript and supplementary materials.

---

# Notes

- Clustering was performed using Smith–Waterman similarity-based sequence clustering.
- PDB-chain identifiers correspond to experimentally resolved peptide–protein complex structures deposited in the Protein Data Bank (PDB).
- The datasets are released for academic and research purposes only.

---

# License

This dataset collection is released for academic and non-commercial research purposes only.
