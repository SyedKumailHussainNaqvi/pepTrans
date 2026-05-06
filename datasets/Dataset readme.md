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
