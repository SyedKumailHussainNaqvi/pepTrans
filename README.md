# pepTrans
pepTrans: Leveraging Protein Language Model Embeddings for Structure-Free Multi-Level Modeling of Peptide–Protein Interactions

## Software Prerequisites

--------------------

**pepTrans** relies on external Python libraries and frameworks to process protein and peptide datasets, extract embedding features, and perform deep learning-based classification and regression tasks.  
The following is the list of required libraries and programs, along with the versions on which the model was tested.

- [ ] [Python](https://www.python.org/) (3.9.16) — Core programming language for pepTrans.
- [ ] [TensorFlow](https://www.tensorflow.org/) (2.9.0) — Main deep learning framework used to build, train, and evaluate the neural networks.
- [ ] [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (11.6.2) — GPU acceleration support for TensorFlow computations.
- [ ] [scikit-learn](https://scikit-learn.org/stable/) (0.24.2) — Machine learning utilities for preprocessing, evaluation, and feature analysis.
- [ ] [NumPy](https://numpy.org/) (1.26.4) — Numerical computation library used for tensor and matrix operations.
- [ ] [Pandas](https://pandas.pydata.org/) (2.2.3) — Data manipulation and analysis.
- [ ] [Matplotlib](https://matplotlib.org/) (3.4.3) — Visualization of performance metrics and results.
- [ ] [h5py](https://www.h5py.org/) (3.9.0) — Reading and writing HDF5 data files containing embeddings and labels.

> **Note:** This environment has been tested with NVIDIA GPUs supporting CUDA 11.6.2 and cuDNN 8.5.0.96. Ensure that your system meets the GPU requirements for optimal performance.
## Installation Guide

The recommended approach for setting up **pepTrans** is via [Anaconda](https://docs.anaconda.com/anaconda/install/index.html), ensuring a reproducible and isolated environment.  

### 1. Clone the repository

```bash
git clone https://github.com/SyedKumailHussainNaqvi/pepTrans.git

```
# License
This software is copyrighted by [Bioinformatics Lab](https://nsclbio.jbnu.ac.kr/) @ Jeonbuk National University.
