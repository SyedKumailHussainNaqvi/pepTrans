import sys
import os

sys.path.append("/model")

from imports import *
from Binary_Classification_pepTrans_module import pepTrans_Binary_Classification, wd_schedule
from Binding_Site_pepTrans_module import *
from Binding_Affinity_pepTrans_Module import *
from Cross_Validation_pepTrans_Binding_Sites import *
from pepTrans import *

if __name__ == '__main__':
    import argparse
    import timeit as ti
    import sys

    parser = argparse.ArgumentParser(description='pepTrans - Prediction Modes')
    parser.add_argument('--mode', choices=['b', 'bs', 'ba', 'pbd', 'vs', 'vas'], required=True,
                        help="'b' for binary classification, 'bs' for binding site prediction, 'ba' for binding affinity regression, 'pbd' for peptide–PBD prediction, 'vs' for virtual screening of peptides, 'vas' for virtual alanine scanning")
    parser.add_argument('--csv', type=str, help='CSV path for binary classification or regression')
    parser.add_argument('--vas_csv', type=str, help='CSV path for alanine-scanning mutated input')
    parser.add_argument('--label', type=str, default='Label', help='Target column in CSV (for classification or regression)')
    parser.add_argument('--train_h5', type=str, help='Training HDF5 path for binding site')
    parser.add_argument('--test_h5', type=str, help='Test HDF5 path for binding site')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use (default: 0)')
    parser.add_argument('--weights', type=str, help='Path to pretrained model weights')
    parser.add_argument('--test_csv', type=str, help='Path to test CSV file')

    args = parser.parse_args()
    
    configure_device(args.gpu)

    t1 = ti.default_timer()

    if args.mode == 'b':
        if args.csv:
            X, y = load_csv_dataset(args.csv, args.label)
            pepTrans_binary_classification(
                X, y, args.label,
                pretrained_weights=None,
                test_csv_path=None
            )
        
        elif args.weights and args.test_csv:
            pepTrans_binary_classification(
                None,None, args.label,
                pretrained_weights=args.weights,
                test_csv_path=args.test_csv
            )
        
        else:
            print("Provide either a CSV path for training or both weights and test CSV for evaluation.")

    elif args.mode == 'bs':

        if args.weights and args.test_h5:
            pepTrans_binding_site_prediction(
                h5_file_path=None,  
                h5_test_file_path=args.test_h5,
                pretrained_weights_path=args.weights,
                n_splits=5
            )

        elif args.train_h5:
            pepTrans_binding_site_prediction(
                h5_file_path=args.train_h5,
                h5_test_file_path=None,  
                pretrained_weights_path=None,
                n_splits=5
            )

        else:
            print("Provide either (train_h5) for training or (weights + test_h5) for pretrained evaluation.")
            sys.exit(1)
    
    elif args.mode == 'ba':

        if args.weights and args.test_csv:
            pepTrans_binding_affinity_regression(
                X=None,
                y=None,
                label_col=args.label,
                input_dim=2048,
                pretrained_weights=args.weights,
                test_csv_path=args.test_csv
            )

        elif args.csv:
            X, y = load_regression_dataset(args.csv, target_col=args.label)
            pepTrans_binding_affinity_regression(
                X=X,
                y=y,
                label_col=args.label,
                input_dim=2048,
                pretrained_weights=None,
                test_csv_path=None
            )

        else:
            print("Provide either (csv) for training or (weights + test_csv) for pretrained evaluation.")
            sys.exit(1)
    
    elif args.mode == 'pbd':
        if args.csv:
            X, y = load_csv_dataset(args.csv, args.label)
            pepTrans_binary_classification(
                X, y, args.label,  
                task='pbd', 
                pretrained_weights=None,
                test_csv_path=None
            )

        elif args.weights and args.test_csv:
            pepTrans_binary_classification(
                None, None, args.label, 
                task='pbd',
                pretrained_weights=args.weights,
                test_csv_path=args.test_csv
            )

        else:
            print("Provide either a CSV path for training or both weights and test CSV for evaluation.")
            sys.exit(1)
    
    elif args.mode == 'vs':
        if not args.csv:
            print("CSV path required for virtual screening of peptides")
            sys.exit(1)
        X, y = load_csv_dataset(args.csv, args.label)
        pepTrans_binary_classification(
                X, y, args.label,  
                task='vs', 
                pretrained_weights=None,
                test_csv_path=None)
        
    elif args.mode == 'vas':
        if not args.csv or not args.vas_csv:
            print("Both --csv (WT) and --vas_csv (Mutant) paths are required for virtual alanine scanning")
            sys.exit(1)
        X, y = load_regression_dataset(args.csv, target_col=args.label) # WT
        X_VAS, y_VAS = load_regression_dataset(args.csv, target_col=args.label) # Mutant
        pepTrans_alanine_scanning_regression(X, X_VAS, y, y_VAS,)
        
    else:
            print(f"Mode {args.mode} not implemented.")
            sys.exit(1)

    t2 = ti.default_timer()
    print(f"\nTotal execution time: {t2 - t1:.2f} seconds")