# PepTrans Model Weights Repository

## Overview

This repository provides access to the pretrained model weights and associated resources for **PepTrans**, a unified deep learning framework for peptide-related prediction tasks, including:

- Peptide Binding Site Prediction
- Protein Binding Site Prediction
- Peptide–Protein Binding Affinity Prediction
- Peptide Virtual Screening
- Peptide PBD Prediction
- Binary Peptide Interaction Prediction
- Generalizability Evaluation Models

## Model Weight Availability

Due to the storage limitations imposed by GitHub for large model files, the complete set of pretrained weights and checkpoints associated with the PepTrans framework are hosted on Hugging Face.

All trained models, checkpoints, and task-specific weight files can be accessed through the official Hugging Face repository:

🔗 **Hugging Face Repository**  
https://huggingface.co/syedkumailhussain/pepTrans

## Repository Structure

The Hugging Face repository contains organized weight files for the following tasks:

| Task | Description |
|--------|------------|
| Binary pepPIs Prediction | Binary peptide–protein interaction prediction models |
| Binding Affinity | Peptide–protein binding affinity prediction models |
| Generalizability | Cross-dataset generalization evaluation models |
| Peptide Binding Residues | Residue-level peptide binding site prediction |
| Peptide PBD Prediction | Peptide binding domain prediction |
| Peptide Virtual Screening | Virtual screening and ranking models |

## Citation

If you use PepTrans, its pretrained models, or any associated resources in your research, please cite the corresponding publication:

```bibtex
@article{PepTrans2026,
  title={pepTrans: Embedding-Based Transformer Framework for Multi-Level Peptide–Protein Interaction Prediction},
  author={Syed Kumail Hussain Naqvi, Hwangeui Cho, Kil To Chong, Hilal Tayara },
  journal={Under Review},
  year={2026}
}
