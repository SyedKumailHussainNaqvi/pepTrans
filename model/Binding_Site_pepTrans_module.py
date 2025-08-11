from imports import *
from Binary_Classification_pepTrans_module import  wd_schedule

import tensorflow as tf
tf.config.run_functions_eagerly(True)

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
                protein_embeddings = np.zeros((500, 1024))  
            
            peptide_data.append(peptide_embeddings)
            protein_data.append(protein_embeddings)
            labels_list.append(labels)
    
    peptide_data = np.array(peptide_data)
    protein_data = np.array(protein_data)
    labels = np.array(labels_list)
    
    labels = np.squeeze(labels, axis=-1)
    
    return peptide_data, protein_data, labels

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, peptide_sequences, protein_sequences, labels, batch_size):
        self.peptide_sequences = peptide_sequences
        self.protein_sequences = protein_sequences
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.peptide_sequences) / self.batch_size))

    def __getitem__(self, index):
        batch_peptide_sequences = self.peptide_sequences[index * self.batch_size:(index + 1) * self.batch_size]
        batch_protein_sequences = self.protein_sequences[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        
        return (tf.convert_to_tensor(batch_peptide_sequences, dtype=tf.float32), 
                tf.convert_to_tensor(batch_protein_sequences, dtype=tf.float32)), \
               tf.convert_to_tensor(batch_labels, dtype=tf.float32)

    def on_epoch_end(self):
        pass

def custom_loss_function(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    y_true_reshaped = K.reshape(y_true, (-1, 50))
    y_true_reshaped = tf.cast(y_true_reshaped, tf.float32) 
    loss = K.binary_crossentropy(y_true_reshaped, y_pred)
    masked_loss = loss * mask
    return tf.reduce_sum(masked_loss) / (tf.reduce_sum(mask) + tf.keras.backend.epsilon())

def custom_metrics_MCC(y_true, y_pred):
    # Mask out -1 values
    mask = tf.not_equal(y_true, -1.0)
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)

    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred_binary)
    tn = tf.reduce_sum((1 - y_true) * (1 - y_pred_binary))
    fp = tf.reduce_sum((1 - y_true) * y_pred_binary)
    fn = tf.reduce_sum(y_true * (1 - y_pred_binary))

    numerator = tp * tn - fp * fn
    denominator = tf.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    ) + tf.keras.backend.epsilon()

    mcc = numerator / denominator
    return mcc

def pepTrans_Binding_Sites(peptide_length, protein_length, embedding_dim):
        lr_schedule = CosineDecay(
            initial_learning_rate=0.01,
            decay_steps=150,
            alpha=0.01)  
        
        peptide_input = Input(shape=(peptide_length, embedding_dim), name='peptide_input')
        protein_input = Input(shape=(protein_length, embedding_dim), name='protein_input')
        
        masked_peptide_input = Masking(mask_value=0)(peptide_input)
        masked_protein_input = Masking(mask_value=0)(protein_input)
        
        cnn_peptide = Conv1D(170, 3, strides=1, activation='relu', padding='same',  kernel_regularizer=l2(0.00003), kernel_initializer=he_normal())(masked_peptide_input)
        cnn_peptide = Dropout(0.3)(cnn_peptide)
        cnn_peptide = MaxPooling1D(pool_size=2)(cnn_peptide)

        cnn_peptide = Conv1D(140, 3, strides=1, activation='relu', padding='same', kernel_regularizer=l2(0.00003), kernel_initializer=he_normal())(cnn_peptide)
        cnn_peptide = Dropout(0.4)(cnn_peptide)
        cnn_peptide = MaxPooling1D(pool_size=2)(cnn_peptide)
        
        cnn_peptide = Conv1D(50, 3, strides=1, activation='relu', padding='same',  kernel_regularizer=l2(0.00003), kernel_initializer=he_normal())(cnn_peptide)
        cnn_peptide = Dropout(0.3)(cnn_peptide)
        cnn_peptide = MaxPooling1D(pool_size=2)(cnn_peptide)
        cnn_peptide = Flatten()(cnn_peptide)
        
        cnn_protein = Conv1D(150, 3, strides=1, activation='relu', padding='same',  kernel_regularizer=l2(0.00003), kernel_initializer=he_normal())(masked_protein_input)
        cnn_protein = Dropout(0.3)(cnn_protein)
        cnn_protein = MaxPooling1D(pool_size=2)(cnn_protein)

        cnn_protein = Conv1D(170, 3, strides=1, activation='relu', padding='same',  kernel_regularizer=l2(0.00003), kernel_initializer=he_normal())(cnn_protein)
        cnn_protein = Dropout(0.4)(cnn_protein)
        cnn_protein = MaxPooling1D(pool_size=2)(cnn_protein)
        
        cnn_protein = Conv1D(50, 3, strides=1, activation='relu', padding='same', kernel_regularizer=l2(0.00003), kernel_initializer=he_normal())(cnn_protein)
        cnn_protein = Dropout(0.3)(cnn_protein)
        cnn_protein = MaxPooling1D(pool_size=2)(cnn_protein)
        
        cnn_protein = Flatten()(cnn_protein)

        concatenated = concatenate([cnn_peptide, cnn_protein])
        combined_output = Dense(2500, activation='relu')(concatenated)
        combined_output = Dropout(0.4)(combined_output)

        output = Dense(peptide_length, activation='sigmoid')(combined_output)
        optimizer = AdamW(learning_rate=lr_schedule, weight_decay=wd_schedule(0))
        model = Model(inputs=[peptide_input, protein_input], outputs=output)
        model.compile(optimizer=optimizer, loss=custom_loss_function, metrics=[custom_metrics_MCC])

        return model
    
def custom_metrics(y_true, y_pred):
    mask = tf.not_equal(y_true, -1)
    
    y_true_filtered = tf.boolean_mask(y_true, mask)
    y_pred_filtered = tf.boolean_mask(y_pred, mask)
    
    y_pred_binary = tf.cast(y_pred_filtered > 0.5, tf.float32)
    
    accuracy = tf.keras.metrics.binary_accuracy(y_true_filtered, y_pred_binary)
    accuracy = tf.reduce_mean(accuracy).numpy()
    
    y_true_filtered_np = y_true_filtered.numpy()
    y_pred_filtered_np = y_pred_binary.numpy()
    y_pred = [1 if prob >= 0.5 else 0 for prob in y_pred]

    def calculate_roc_auc_mcc(y_true, pred):

        y_true_combined = list(y_true)
        pred_combined = list(pred)

        if len(set(y_true)) == 1 & len(set(pred)) == 1:
            y_true_complement = [1 - val for val in y_true]  
            y_pred_complement = [1 - p for p in pred] 
            
            y_true_combined.extend(y_true_complement)
            pred_combined.extend(y_pred_complement)

        elif len(set(y_true)) == 1:
            y_true_complement = [1 - val for val in y_true]  
            y_pred_complement = [1 - p for p in pred] 
            
            y_true_combined.extend(y_true_complement)
            pred_combined.extend(y_pred_complement)
            
        elif len(set(pred)) == 1:
            y_true_complement = [1 - val for val in y_true]  
            y_pred_complement = [1 - p for p in pred] 
            
            y_true_combined.extend(y_true_complement)
            pred_combined.extend(y_pred_complement)
            
        roc_auc = roc_auc_score(y_true_combined, pred_combined)
        mcc_value = matthews_corrcoef(y_true_combined, pred_combined)
        return roc_auc,mcc_value

    auc_value, mcc_value = calculate_roc_auc_mcc(y_true_filtered_np, y_pred_filtered_np)
    
    return  auc_value, mcc_value
