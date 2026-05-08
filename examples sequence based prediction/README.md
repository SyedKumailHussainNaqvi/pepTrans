# Sequence-Based Prediction Examples for pepTrans

Example sequence-based inference files for pepTrans: End-to-End Peptide–Protein Interaction Prediction from Raw Sequences

--------------------

# Overview

The `examples_sequence_based_prediction/` directory contains example input sequence files, automatically generated ProtT5 embedding files, and example outputs for running pepTrans directly from raw peptide and protein sequences.

These examples demonstrate the end-to-end inference workflow of pepTrans:

```text
Peptide and Protein Sequences
                ↓
Automatic ProtT5 Embedding Generation
                ↓
pepTrans Prediction
                ↓
Prediction Output
```

The provided examples demonstrate:

- binary peptide–protein interaction prediction
- peptide binding residue prediction
- peptide–protein binding affinity prediction

---

# Important Note

The sequence examples provided in this directory are dummy demonstration samples created solely for illustrating the input/output format of the sequence-based prediction pipeline.

These examples:

- are NOT part of the original training datasets
- are NOT part of the independent test datasets
- are NOT used during model training or evaluation

They are included only for reproducibility demonstration and usability purposes.

---

# Directory Structure

```text
examples_sequence_based_prediction/
│
├── README.md
│
├── example_sequences.csv
├── generated_binary_embeddings.csv
│
├── example_binding_site_sequences.csv
├── generated_binding_site_embeddings.h5
│
├── example_binding_affinity_sequences.csv
└── generated_binding_affinity_embeddings.csv
```

---

# Quick Start: Create Example CSV Files

For user convenience, example input CSV files can be created directly from the terminal using the following commands.

---

## 1. Create Binary Interaction Example CSV

```bash
cat > "/home/kumail/Bioinformatics/github pepTrans/example_sequences.csv" << 'EOF'
IDs,Peptide,Protein,Label
sample_1,KLVFFAED,MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANLAGK,1
sample_2,ACDEFGHIK,MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSEF,0
EOF
```

Check the file:

```bash
cat "/home/kumail/Bioinformatics/github pepTrans/example_sequences.csv"
```

---

## 2. Create Peptide Binding Residue Example CSV

```bash
cat > "/home/kumail/Bioinformatics/github pepTrans/example_binding_site_sequences.csv" << 'EOF'
IDs,Peptide,Protein,peptide_labels
bs_sample_1,KLVFFAED,MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANLAGK,11100000
bs_sample_2,ACDEFGHIK,MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSEF,001111000
bs_sample_3,RRWEQKLVHIKTMEGEFSVTMWASGIS,MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP,000011111100000000001111000
bs_sample_4,CKRIHIGPGRAFYTTC,EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKG,0111111110000000
bs_sample_5,RVCPRILMECKKDSDCLAECVCLEHGYCG,IVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSGIQVRLGEDNINVVEGNEQF,11111100000011111100000011111
bs_sample_6,ACEDTRRPRVARWTGQIIYCS,MGHHHHHHSSGVDLGTENLYFQSMASMTGGQQMGRGSEFELKSLVQKAKLAEQAERYDDMAAAMK,000001111111000000000
bs_sample_7,GILGFVFTL,MAVMAPRTLLLLLSGALALTQTWAGSHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQR,111111110
bs_sample_8,LLFGYPVYV,MRVTAPRTVLLLLSAALALTETWAGSHSMRYFYTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQR,001111100
bs_sample_9,SIINFEKL,MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVK,11110000
bs_sample_10,KRWIILGLNK,MDYKDDDDKLVPRGSHMASMTGGQQMGRGSEFPMTQYQRESLRIKLEQQL,0111111110
EOF
```

Check the file:

```bash
cat "/home/kumail/Bioinformatics/github pepTrans/example_binding_site_sequences.csv"
```

### Important Note for Binding Residue Labels

For binding-site prediction, the length of `peptide_labels` must match the peptide sequence length.

Each label corresponds to one peptide residue:

```text
1 = binding residue
0 = non-binding residue
```

Example:

```text
Peptide:         KLVFFAED
peptide_labels:  11100000
```

This means:

| Residue | Label |
|---|---|
| K | 1 |
| L | 1 |
| V | 1 |
| F | 0 |
| F | 0 |
| A | 0 |
| E | 0 |
| D | 0 |

---

## 3. Create Binding Affinity Example CSV

```bash
cat > "/home/kumail/Bioinformatics/github pepTrans/example_binding_affinity_sequences.csv" << 'EOF'
IDs,Peptide,Protein,Log_Affinity
ba_sample_1,KLVFFAED,MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANLAGK,7.20
ba_sample_2,ACDEFGHIK,MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSEF,6.85
ba_sample_3,RRWEQKLVHIKTMEGEFSVTMWASGIS,MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP,8.10
ba_sample_4,CKRIHIGPGRAFYTTC,EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKG,7.95
ba_sample_5,RVCPRILMECKKDSDCLAECVCLEHGYCG,IVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSGIQVRLGEDNINVVEGNEQF,8.50
EOF
```

Check the file:

```bash
cat "/home/kumail/Bioinformatics/github pepTrans/example_binding_affinity_sequences.csv"
```

---

# 1. Binary Peptide–Protein Interaction Prediction

This example demonstrates sequence-based binary peptide–protein interaction prediction.

---

## Example Input File

```text
example_sequences.csv
```

---

## Example Format

```csv
IDs,Peptide,Protein,Label
sample_1,KLVFFAED,MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANLAGK,1
sample_2,ACDEFGHIK,MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSEF,0
sample_3,RRWEQKLVHIKTMEGEFSVTMWASGIS,MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP,1
```

---

## Running Sequence-Based Prediction

```bash
python "../main.py" \
--mode b \
--input_sequences "example_sequences.csv" \
--peptide_col Peptide \
--protein_col Protein \
--label Label \
--weights "../pepTrans weight/NA_fold_1_Model_Weights.h5" \
--save_embeddings "generated_binary_embeddings.csv" \
--gpu 0 \
--embedding_device cpu
```

---

## Example Output

```text
=== Peptide–Protein Interaction Predictions ===

sample_1: Prediction = 0 (Non-interaction), Probability = 0.1849

sample_2: Prediction = 1 (Interaction), Probability = 0.7796

sample_3: Prediction = 1 (Interaction), Probability = 0.7563


=== Test Set Evaluation ===

Test Loss: 1.5641
Test AUC: 0.7300
Test AUPR: 0.3333
```

---

## Generated Embedding File

```text
generated_binary_embeddings.csv
```

This file contains automatically generated ProtT5 embeddings extracted from the input sequences.

---

# 2. Peptide Binding Residue Prediction

This example demonstrates sequence-based peptide binding residue prediction.

---

## Example Input File

```text
example_binding_site_sequences.csv
```

---

## Example Format

```csv
IDs,Peptide,Protein,peptide_labels
bs_sample_1,KLVFFAED,MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANLAGK,11100000
bs_sample_2,ACDEFGHIK,MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSEF,001111000
```

---

## Running Sequence-Based Prediction

```bash
python "../main.py" \
--mode bs \
--input_sequences "example_binding_site_sequences.csv" \
--peptide_col Peptide \
--protein_col Protein \
--peptide_label_col peptide_labels \
--weights "../pepTrans weight/PL_500_fold_1_Binding_Residue_Pair_Wise_CNN_Model_Weights.h5" \
--save_embeddings "generated_binding_site_embeddings.h5" \
--gpu 0 \
--embedding_device cpu
```

---

## Example Output

```text
=== Peptide Binding-Site Predictions ===

bs_sample_1
Peptide: KLVFFAED

Overall peptide status:
Interaction / contains predicted binding residues

Residue 01 K: Binding residue | Probability = 0.7071
Residue 02 L: Binding residue | Probability = 0.9206
Residue 03 V: Binding residue | Probability = 0.9320
Residue 04 F: Binding residue | Probability = 0.8483
Residue 05 F: Binding residue | Probability = 0.8194
Residue 06 A: Binding residue | Probability = 0.8500
Residue 07 E: Binding residue | Probability = 0.8426
Residue 08 D: Binding residue | Probability = 0.6698


=== Test Set Evaluation ===

Mean MCC: 0.6437
Mean AUC: 0.3013
```

---

## Generated Embedding File

```text
generated_binding_site_embeddings.h5
```

This HDF5 file contains automatically generated ProtT5 residue-level embeddings for peptide and protein sequences.

---

# 3. Peptide–Protein Binding Affinity Prediction

This example demonstrates sequence-based peptide–protein binding affinity prediction.

---

## Example Input File

```text
example_binding_affinity_sequences.csv
```

---

## Example Format

```csv
IDs,Peptide,Protein,Log_Affinity
ba_sample_1,KLVFFAED,MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANLAGK,7.20
ba_sample_2,ACDEFGHIK,MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSEF,6.85
```

---

## Running Sequence-Based Prediction

```bash
python "../main.py" \
--mode ba \
--input_sequences "example_binding_affinity_sequences.csv" \
--peptide_col Peptide \
--protein_col Protein \
--label Log_Affinity \
--weights "../pepTrans weight/Fold_1_Binding_Affinity_Model_Weights.h5" \
--save_embeddings "generated_binding_affinity_embeddings.csv" \
--gpu 0 \
--embedding_device cpu
```

---

## Example Output

```text
=== Binding Affinity Predictions ===

ba_sample_1:
Predicted Binding Affinity = 3.8778
True Binding Affinity = 7.2000

ba_sample_2:
Predicted Binding Affinity = 3.6363
True Binding Affinity = 6.8500

ba_sample_3:
Predicted Binding Affinity = 2.7740
True Binding Affinity = 8.1000


=== Test Set Evaluation ===

Test Loss: 24.3260
Test RMSE: 4.8735
Test Pearson Correlation: -0.9305
```

---

## Generated Embedding File

```text
generated_binding_affinity_embeddings.csv
```

This file contains automatically generated ProtT5 sequence embeddings used for binding affinity prediction.

---

# Automatic ProtT5 Embedding Generation

The updated version of pepTrans supports direct inference from raw peptide and protein sequences.

During inference, pepTrans automatically:

1. loads the ProtT5-XL-U50 protein language model
2. extracts contextual embeddings
3. generates sequence embeddings internally
4. performs downstream prediction

Users no longer need to manually generate embeddings before prediction.

---

# CPU and GPU Execution

For CPU execution:

```bash
--embedding_device cpu
```

For CUDA-enabled GPU execution:

```bash
--embedding_device cuda
```

CUDA-compatible PyTorch installation is required for GPU-based embedding generation.

---

# Notes

- The generated embedding files are automatically created during sequence-based prediction.
- Both sequence-based inference and precomputed embedding-based inference are supported.
- These examples are provided solely for demonstrating input formatting and reproducibility.
- The examples should not be used for benchmarking or scientific evaluation.
