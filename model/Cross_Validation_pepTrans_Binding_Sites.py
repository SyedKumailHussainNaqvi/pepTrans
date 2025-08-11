import h5py
import numpy as np
from sklearn.model_selection import KFold

def read_hdf5_file(file_path):
    peptide_data = []
    protein_data = []
    labels_list = []
    with h5py.File(file_path, 'r') as h5file:
        for combined_id in h5file:
            group = h5file[combined_id]
            peptide_embeddings = group['peptide_embeddings'][:]
            labels = group['labels'][:]
            if 'protein_embeddings' in group:
                protein_embeddings = group['protein_embeddings'][:]
            else:
                protein_embeddings = np.zeros((500, 1024))  # Fallback
            peptide_data.append(peptide_embeddings)
            protein_data.append(protein_embeddings)
            labels_list.append(labels)
    peptide_data = np.array(peptide_data)
    protein_data = np.array(protein_data)
    labels = np.squeeze(np.array(labels_list), axis=-1)
    return peptide_data, protein_data, labels

def filter_data_by_indices(peptide_data, protein_data, labels, indices):
    return peptide_data[indices], protein_data[indices], labels[indices]

benchmark_dataset = "/home/kumail/Bioinformatics/Dataset/Bindingsites/Output/Protein_Length_500/fold_1_test_data.h5"
peptide_data, protein_data, labels = read_hdf5_file(benchmark_dataset)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
batch_size = 128
fold_no = 1

for fold, (train_index, test_index) in enumerate(kf.split(peptide_data)):
    #print(f"\n=== Fold {fold_no} ===")

    # Split the data in memory without saving/loading indices
    pep_train, prot_train, lbl_train = filter_data_by_indices(peptide_data, protein_data, labels, train_index)
    pep_test, prot_test, lbl_test = filter_data_by_indices(peptide_data, protein_data, labels, test_index)

    #print("Train Peptide Shape:", pep_train.shape)
    #print("Train Protein Shape:", prot_train.shape)
    #print("Train Labels Shape :", lbl_train.shape)

    #print("Test Peptide Shape :", pep_test.shape)
    #print("Test Protein Shape :", prot_test.shape)
    #print("Test Labels Shape  :", lbl_test.shape)

    fold_no += 1
