import sys
import os

sys.path.append("/model")

from imports import *
from Binary_Classification_pepTrans_module import pepTrans_Binary_Classification, wd_schedule
from Binding_Site_pepTrans_module import *
from Binding_Affinity_pepTrans_Module import *
from Cross_Validation_pepTrans_Binding_Sites import *
from Cross_Validation_pepTrans_Binding_Sites import *
from pepTrans import *
import timeit as ti
import sys
from types import SimpleNamespace 

def run_pepTrans(args):
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
                    X=None, y=None, task=args.label,
                    pretrained_weights=args.weights,
                    test_csv_path=args.test_csv
                )
            
        else:
                print("Provide either a CSV path for training or both weights and test CSV for evaluation.")
                return

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
                h5_test_file_path=args.test_h5,  
                pretrained_weights_path=None,
                n_splits=5
            )
        else:
            print("Provide either (train_h5) for training or (weights + test_h5) for pretrained evaluation.")
            return
        
    elif args.mode == 'ba':
        
        if args.pretrained_weights and args.test_csv:
            pepTrans_binding_affinity_regression(
                X=None,  
                y=None,
                input_dim=2048,
                pretrained_weights=args.pretrained_weights,
                test_csv_path=args.test_csv,
                label_col=args.label  
            )

        elif args.csv:

            X, y = load_regression_dataset(args.csv, target_col=args.label)
            pepTrans_binding_affinity_regression(
                X=X,
                y=y,
                input_dim=2048,
                pretrained_weights=None,
                test_csv_path=None
            )
        else:
            print("Provide either (csv) for training or (pretrained_weights + test_csv) for pretrained evaluation.")
            return

    elif args.mode == 'pbd':
        if not args.csv:
            print("CSV path required for peptide–PBD classification")
            return
        X, y = load_csv_dataset(args.csv, args.label)
        pepTrans_binary_classification(X, y, args.label, task='pbd', 
            pretrained_weights=getattr(args, 'weights', None),
            test_csv_path=getattr(args, 'test_csv', None))

    elif args.mode == 'vs':
        if not args.csv:
            print("CSV path required for virtual screening of peptides")
            return
        X, y = load_csv_dataset(args.csv, args.label)
        pepTrans_binary_classification(X, y,args.label, task='vs')

    elif args.mode == 'vas':
        if not args.csv or not args.vas_csv:
            print("Both --csv (WT) and --vas_csv (Mutant) paths are required for virtual alanine scanning")
            return
        X, y = load_regression_dataset(args.csv, target_col=args.label)  # WT
        X_VAS, y_VAS = load_regression_dataset(args.vas_csv, target_col=args.label)  # Mutant
        pepTrans_alanine_scanning_regression(X, X_VAS, y, y_VAS)

    else:
        print(f"Invalid mode: {args.mode}")

    t2 = ti.default_timer()
    print(f"\nTotal execution time: {t2 - t1:.2f} seconds")