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
### Clone the repository
```bash
git clone https://github.com/SyedKumailHussainNaqvi/pepTrans.git
```

### Install Option 1: Install via `pepTrans_env.yml` file (Recommended)

We can easily install the environment by using the provided **`pepTrans_env.yml`** file. This configuration contains all required dependencies with tested versions.
```bash
1. Navigate to the project directory
cd pepTrans

2. Create the conda environment from the .yml file
conda env create -f pepTrans_env.yml

3. Activate the environment
conda activate pepTrans
```
Once the environment is activated, we can proceed to run pepTrans for training, evaluation, or inference as described in the Usage section of this document.
### Install Option 2: Install manually
If the environment installation via the provided .yml file fails, we can manually create it by installing the dependencies step-by-step:
```bash
 1. Create a new conda environment with Python 3.9
conda create -n pepTrans python=3.9 -y

 2. Activate the environment
conda activate pepTrans

 3. Install core dependencies via conda
conda install tensorflow=2.9.0 cudatoolkit=11.6 scikit-learn=0.24.2 numpy=1.26.4 pandas=2.2.3 matplotlib=3.4.3 h5py=3.9.0 -c conda-forge -y

 4. (Optional) Install any additional dependencies via pip if required
 pip install <package-name>
```
This manual installation method gives us more control over versions and sources but is more prone to dependency conflicts compared to using the .yml file
## Dataset and Protein Language Model (pLM) Features
The datasets used for **pepTrans** are adapted from the [CAMP repository](https://github.com/twopin/CAMP).  For detailed instructions, please refer to the CAMP [GitHub repository](https://github.com/twopin/CAMP).  

**pepTrans** uses pre-trained embeddings from the ProtTrans family of protein language models. Specifically, **ProtT5-XL-UniRef50 (ProtT5-XL-U50)** embeddings are employed for feature extraction.  
For more details, consult the [ProtTrans GitHub repository](https://github.com/agemagician/ProtTrans).

* **Input**: Raw protein sequences in FASTA or CSV format.  
* **Feature Extraction**: Performed using pre-trained pLMs to generate embeddings.  
* **Output**: Embedding vectors stored in `.csv` or `.h5` format for model training and evaluation.

## Supported Parameters of pepTrans
For a detailed description of pepTrans parameters and options, run:
  
```bash
python main.py --help
```
This will display all supported parameters of pepTrans:
```bash
usage: main.py [-h] --mode {b,bs,ba,pbd,vs,vas} [--csv CSV] [--vas_csv VAS_CSV] [--label LABEL]
               [--train_h5 TRAIN_H5] [--test_h5 TEST_H5] [--gpu GPU] [--weights WEIGHTS]
               [--test_csv TEST_CSV]

optional arguments:
  -h, --help            show this help message and exit
  --mode {b,bs,ba,pbd,vs,vas}
                        Mode of operation:
                          'b'   - Binary classification (peptide–protein binding)
                          'bs'  - Binding site prediction (residue-level)
                          'ba'  - Binding affinity regression
                          'pbd' - Peptide-PBD (Protein Binding Domain) prediction
                          'vs'  - Virtual screening of peptides
                          'vas' - Virtual alanine scanning mutagenesis
  --csv CSV             Path to input CSV file (for training or evaluation)
  --vas_csv VAS_CSV     Path to alanine-scanning mutated CSV input (required for 'vas' mode)
  --label LABEL         Target column name in CSV (default: 'Label')
  --train_h5 TRAIN_H5   Path to training HDF5 file (for binding site prediction training)
  --test_h5 TEST_H5     Path to test HDF5 file (for binding site prediction evaluation)
  --gpu GPU             GPU ID to use (default: '0')
  --weights WEIGHTS     Path to pretrained model weights file (for evaluation/inference)
  --test_csv TEST_CSV   Path to test CSV file (for evaluation of binary classification or regression)
```
## Running pepTrans
pepTrans supports multiple modes for multi-level of Peptide–Protein Interactions predictions, enabling training, evaluation with pretrained models, and application to custom datasets.  
# License
This software is copyrighted by [Bioinformatics Lab](https://nsclbio.jbnu.ac.kr/) @ Jeonbuk National University.
