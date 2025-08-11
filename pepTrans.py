import sys
import os

sys.path.append("/model")

from imports import *
from Binary_Classification_pepTrans_module import pepTrans_Binary_Classification, wd_schedule
from Binding_Site_pepTrans_module import *
from Binding_Affinity_pepTrans_Module import *
from Cross_Validation_pepTrans_Binding_Sites import *

def configure_device(gpu_id='0'):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' 
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPU found. Defaulting to CPU.")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  
    else:
        print(f"Using GPU {gpu_id}")

def load_csv_dataset(path, label_col):
    df = pd.read_csv(path, index_col='IDs')
    X = df.drop(columns=[label_col]).to_numpy(dtype=np.float32)
    y = df[[label_col]].to_numpy(dtype=np.float32).reshape(-1, 1)
    return X, y

def pepTrans_binary_classification(X, y, label_col, input_dim=2048, task='binary', pretrained_weights=None, test_csv_path=None):
    
    if pretrained_weights:
        if not test_csv_path:
            raise ValueError("Test CSV path required when using pretrained weights for evaluation.")
        print(f"Loading pretrained weights from: {pretrained_weights}")
        print(f"Loading test dataset from: {test_csv_path}")

        X, y = load_csv_dataset(test_csv_path, label_col )
        X_test = X
        y_test = y

        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)  

        with tf.device('/GPU:1'):
            model = pepTrans_Binary_Classification(input_dim, X_test_scaled)
            model.load_weights(pretrained_weights)
            print("Pretrained model weights loaded.")

            test_preds = model.predict(X_test_scaled).flatten()
            test_loss, test_acc, test_model_auc = model.evaluate(X_test_scaled, y_test, verbose=0)

        print('\n=== Test Set Evaluation ===')
        print(f'Test Loss: {test_loss:.4f}')
        print(f'Test Model AUC: {test_model_auc:.4f}')

        if task == 'binary':
            test_roc_auc = roc_auc_score(y_test, test_preds)
            test_aupr = average_precision_score(y_test, test_preds)
            print(f'Test AUC: {test_roc_auc:.4f}')
            print(f'Test AUPR: {test_aupr:.4f}')
            
        if task == 'pbd':
            test_roc_auc = roc_auc_score(y_test, test_preds)
            test_aupr = average_precision_score(y_test, test_preds)
            print(f'Test AUC: {test_roc_auc:.4f}')

        return  
    
    if task == 'pbd':
        n_splits = 8
        print("peptide–PBD prediction: pepTrans loading for training with 8-fold cross-validation...\n")
    elif task == 'binary':
        n_splits = 5
        print("Peptide–Protein Binary Interaction: pepTrans loading for training with 5-fold cross-validation...\n")
    elif task == 'vs':
        n_splits = 5
        print("Virtual screening of peptides: pepTrans loading for training with 5-fold cross-validation...\n")
        
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    acc_per_fold, loss_per_fold = [], []
    model_auc_per_fold, roc_auc_per_fold, aupr_per_fold = [], [], []
    fold_no = 1

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        labels = y_train[:, 0].astype(int)
        weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
        class_weight = {i: w for i, w in enumerate(weights)}

        with tf.device('/GPU:1'):
            print(f"Training pepTrans model on fold {fold_no}")
            model = pepTrans_Binary_Classification(input_dim, X_train)
            early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            history = model.fit(X_train, y_train, epochs=5000, batch_size=128,
                                validation_data=(X_test, y_test),
                                callbacks=[early_stopping],
                                class_weight=class_weight)

        scores = model.evaluate(X_test, y_test)
        predictions = model.predict(X_test).flatten()

        acc_per_fold.append(scores[1])
        loss_per_fold.append(scores[0])
        model_auc_per_fold.append(scores[2])
        
        if task == 'binary':
            roc_auc_per_fold.append(roc_auc_score(y_test, predictions))
            aupr_per_fold.append(average_precision_score(y_test, predictions))

            print(f'Fold {fold_no} - Loss: {scores[0]:.4f} '
                f'Model AUC: {scores[2]:.4f}, ROC AUC: {roc_auc_per_fold[-1]:.4f}, AUPR: {aupr_per_fold[-1]:.4f}')
        elif task == 'pbd' or task == 'vs':
            print(f'Fold {fold_no} - Loss: {scores[0]:.4f}'
                    f'Model AUC: {scores[2]:.4f}')
            
            print(f'> Model AUC: {np.mean(model_auc_per_fold)} ± {np.std(model_auc_per_fold)}')
        print(f"pepTrans trained successfully for {'Peptide–PDB Prediction' if task == 'pbd' else 'Virtual Screening of Peptides' if task == 'vs' else 'Peptide–Protein Binary Interaction'} on fold {fold_no}.\n")
        fold_no += 1

    print('PepTrans Predictions:')
    print(f'> Loss: {np.mean(loss_per_fold):.4f}')

    if task == 'binary':
        print(f'> Model AUC: {np.mean(model_auc_per_fold):.4f} ± {np.std(model_auc_per_fold):.4f}')
        print(f'> ROC AUC: {np.mean(roc_auc_per_fold):.4f} ± {np.std(roc_auc_per_fold):.4f}')
        print(f'> AUPR: {np.mean(aupr_per_fold):.4f} ± {np.std(aupr_per_fold):.4f}')
        print("pepTrans trained successfully for peptide-protein interaction prediction.")
    elif task == 'pbd':
        print(f'> Model AUC: {np.mean(model_auc_per_fold):.4f} ± {np.std(model_auc_per_fold):.4f}')
        print("pepTrans trained successfully for peptide-PDB prediction.")
    elif task == 'vs':
        print(f'> Model AUC: {np.mean(model_auc_per_fold):.4f} ± {np.std(model_auc_per_fold):.4f}')
        print("pepTrans trained successfully for virtual screening of peptides.")
        
def pepTrans_binding_site_prediction(h5_file_path, h5_test_file_path=None, pretrained_weights_path=None, n_splits=5):

    # ---------- 1. Pretrained Weights Evaluation ----------
    if pretrained_weights_path:
        print(f"Loading pretrained weights from: {pretrained_weights_path}")
        print(f"Loading test/train dataset from: {h5_test_file_path}")
        
        peptide_data, protein_data, labels = read_hdf5_file(h5_test_file_path)
        batch_size = 128

        peptide_length = peptide_data.shape[1]
        protein_length = protein_data.shape[1]
        embedding_dim = peptide_data.shape[2]

        # Create test generator
        test_gen = DataGenerator(peptide_data, protein_data, labels, batch_size)

        with tf.device('/GPU:1'):
            model = pepTrans_Binding_Sites(peptide_length, protein_length, embedding_dim)
            model.load_weights(pretrained_weights_path)
            print("Pretrained model weights loaded.")

            predictions = model.predict(test_gen, verbose=1)

        # Flatten predictions and labels for metrics
        #predictions_flat = predictions.flatten()
        #labels_flat = labels.flatten()
        
        def Avg_mcc_auc(model, test_peptide_data, test_protein_data, test_labels, custom_metrics):

            sample_mcc_values = []
            sample_auc_values = []

            for i in range(len(test_peptide_data)):
                test_pep = np.expand_dims(test_peptide_data[i], axis=0)
                test_pro = np.expand_dims(test_protein_data[i], axis=0)
                true_label = test_labels[i].reshape(-1)

                prediction = model.predict([test_pep, test_pro], verbose=0).flatten()

                y_test_tensor = tf.convert_to_tensor(true_label, dtype=tf.float32)
                predictions_tensor = tf.convert_to_tensor(prediction, dtype=tf.float32)

                auc_value, mcc_value = custom_metrics(y_test_tensor, predictions_tensor)

                sample_mcc_values.append(mcc_value)
                sample_auc_values.append(auc_value)

            # Calculate mean MCC and AUC
            mean_mcc = np.mean(sample_mcc_values)
            mean_auc = np.mean(sample_auc_values)

            return mean_mcc, mean_auc
        
        auc, mcc = Avg_mcc_auc(model, peptide_data, protein_data, labels, custom_metrics)

        #auc, mcc = custom_metrics(
           # tf.convert_to_tensor(labels_flat, dtype=tf.float32),
           # tf.convert_to_tensor(predictions_flat, dtype=tf.float32)
        #)

        print("\n=== Test Set Evaluation ===")
        print(f"Mean MCC: {mcc:.4f}")
        print(f"Mean AUC: {auc:.4f}")
        return predictions, auc, mcc
    
    # ---------- 2. K-Fold Cross Validation ----------
    peptide_data, protein_data, labels = read_hdf5_file(h5_file_path)
    print(f"Training pepTrans model for Binding Site Prediction with {n_splits}-fold cross-validation...\n")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    batch_size = 128
    loss_per_fold = [],
    model_auc_per_fold,  mcc_per_fold = [], []

    fold_no = 1
    for train_idx, test_idx in kf.split(peptide_data):
        print(f"\n Training pepTrans model for Peptide Binding Residues on fold {fold_no}")

        pep_train, pro_train, lbl_train = peptide_data[train_idx], protein_data[train_idx], labels[train_idx]
        pep_test, pro_test, lbl_test = peptide_data[test_idx], protein_data[test_idx], labels[test_idx]

        train_gen = DataGenerator(pep_train, pro_train, lbl_train, batch_size)
        test_gen = DataGenerator(pep_test, pro_test, lbl_test, batch_size)

        peptide_length = pep_train.shape[1]
        protein_length = pro_train.shape[1]
        embedding_dim = pep_train.shape[2]

        with tf.device('/GPU:1'):
            model = pepTrans_Binding_Sites(peptide_length, protein_length, embedding_dim)
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

            history = model.fit(train_gen,
                                validation_data=test_gen,
                                epochs=5000,
                                callbacks=[early_stop],
                                verbose=1)

        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        plt.figure(figsize=(5, 2))
        plt.plot(train_loss, label="Training Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.axvline(np.argmin(val_loss), color='red', linestyle='--', label='Early Stop')
        plt.title(f'Fold {fold_no} Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()

        mcc_values, auc_values = [], []
        for i in range(len(pep_test)):
            pred = model.predict([np.expand_dims(pep_test[i], axis=0), 
                                  np.expand_dims(pro_test[i], axis=0)], verbose=0).flatten()
            y_true = lbl_test[i].reshape(-1)
            auc, mcc = custom_metrics(tf.convert_to_tensor(y_true, dtype=tf.float32),
                                      tf.convert_to_tensor(pred, dtype=tf.float32))
            mcc_values.append(mcc)
            auc_values.append(auc)

        mean_mcc, mean_auc = np.mean(mcc_values), np.mean(auc_values)
        mcc_per_fold.append(mean_mcc)
        model_auc_per_fold.append(mean_auc)

        unique_vals = sorted(set(mcc_values))
        counts = [mcc_values.count(val) for val in unique_vals]
        plt.figure(figsize=(10, 6))
        plt.bar(unique_vals, counts, width=0.02, color='skyblue', edgecolor='black')
        plt.title(f'MCC Distribution - Fold {fold_no}')
        plt.xlabel('MCC')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', axis='y', alpha=0.7)
        plt.tight_layout()
        plt.show()

        print(f"Fold {fold_no} - Mean MCC: {mean_mcc:.4f}, Mean AUC: {mean_auc:.4f}")
        fold_no += 1

    print("\n=== Cross-Validation Results ===")
    print(f"> Avg MCC: {np.mean(mcc_per_fold):.4f}")
    print(f"> Avg AUC: {np.mean(model_auc_per_fold):.4f}")
    
def load_regression_dataset(path, target_col='Log_Affinity'):
    df = pd.read_csv(path, index_col='IDs')
    X = df.drop(columns=[target_col]).to_numpy(dtype=np.float32)
    y = df[[target_col]].to_numpy(dtype=np.float32).reshape(-1, 1)
    return X, y

def pepTrans_binding_affinity_regression(X=None, y=None, label_col=None, input_dim=2048, pretrained_weights=None, test_csv_path=None):
    
    # === Pretrained Weights Evaluation Mode ===
    if pretrained_weights:
        if not test_csv_path:
            raise ValueError("Test CSV path is required when using pretrained weights.")
        print(f"Binding Affinity")
        print(f"Loading pretrained weights from: {pretrained_weights}")
        print(f"Loading test dataset from: {test_csv_path}")
        def load_regression_dataset(path, target_col='Log_Affinity'):
            df = pd.read_csv(path)
            X = df.drop(columns=[target_col]).to_numpy(dtype=np.float32)
            y = df[[target_col]].to_numpy(dtype=np.float32).reshape(-1, 1)
            return X, y

        X_test, y_test = load_regression_dataset(test_csv_path, label_col)

        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)

        with tf.device('/GPU:1'):
            model = pepTrans_Binding_Affinity(input_dim, X_test_scaled)
            model.load_weights(pretrained_weights)
            print("Pretrained model weights loaded.")

            predictions = model.predict(X_test_scaled).flatten()
            test_labels = y_test.flatten()

            rmse = mean_squared_error(test_labels, predictions, squared=False)
            pearson_corr, _ = pearsonr(test_labels, predictions)
            test_loss = model.evaluate(X_test_scaled, y_test, verbose=0)

        print('\n=== Test Set Evaluation ===')
        print(f'Test Loss: {test_loss[0]:.4f}')
        print(f'Test RMSE: {rmse:.4f}')
        print(f'Test Pearson Correlation: {pearson_corr:.4f}')
        return

    # === Training with K-Fold Cross-Validation ===
    
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    rmse_per_fold, pearson_per_fold = [], []
    fold_no = 1

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(f"\n Training pepTrans model for Binding Affinity on fold {fold_no}")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        with tf.device('/GPU:1'):
            model = pepTrans_Binding_Affinity(input_dim, X_train)
            early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            history = model.fit(X_train, y_train, epochs=5000, batch_size=128,
                                validation_data=(X_test, y_test),
                                callbacks=[early_stopping])

        scores = model.evaluate(X_test, y_test)
        predictions = model.predict(X_test).flatten()
        test_labels = y_test.flatten()

        rmse = mean_squared_error(test_labels, predictions, squared=False)
        pearson_corr, _ = pearsonr(test_labels, predictions)

        rmse_per_fold.append(rmse)
        pearson_per_fold.append(pearson_corr)

        plt.figure(figsize=(5, 2))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        min_epoch = np.argmin(history.history['val_loss']) + 1
        plt.axvline(min_epoch, linestyle='--', color='r', label='Early Stopping Checkpoint')
        plt.title(f'Fold {fold_no}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        print(f'Fold {fold_no} - Loss: {scores[0]:.4f}, RMSE: {rmse:.4f}, '
              f'Pearson Correlation: {pearson_corr:.4f}')
        fold_no += 1

    print('Binding Affinity Regression Summary:')
    print(f'> RMSE: {np.mean(rmse_per_fold):.4f} ± {np.std(rmse_per_fold):.4f}')
    print(f'> Pearson Correlation: {np.mean(pearson_per_fold):.4f} ± {np.std(pearson_per_fold):.4f}')
    
    print("pepTrans trained successfully for Binding Affinity Regression.")

def pepTrans_alanine_scanning_regression(X, X_VAS, y_train, y_VAS, input_dim=2048):

    scaler = StandardScaler()
    train = scaler.fit_transform(X)
    X_VAS = scaler.transform(X_VAS)

    with tf.device('/GPU:1'):
        model = pepTrans_Binding_Affinity(input_dim, train)
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        history = model.fit(train, y_train, epochs=50, batch_size=128,
                            validation_data=(X_VAS, y_VAS),
                            callbacks=[early_stopping])
        
    train_loss = history.history['loss']
    valid_loss = history.history['val_loss']
    fig = plt.figure(figsize=(5, 2))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')
    minposs = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
    plt.title(f'Model Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    scores = model.evaluate(X_VAS, y_VAS)
    predictions = model.predict(X_VAS).flatten()
    y_VAS_labels = np.array(y_VAS).flatten()
    VAS_predictions = np.array(predictions).flatten()
    pearson_corr, _ = pearsonr(y_VAS_labels, VAS_predictions)
    print(f'Loss: {scores[0]:.4f}, Pearson Correlation: {pearson_corr:.4f}')
    
    print("pepTrans trained successfully for Virtual Alanine Scanning.")