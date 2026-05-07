import sys
import os

sys.path.append("/home/kumail/Bioinformatics/github pepTrans/model/")

from sequence_embedding import (
    generate_binary_csv_from_sequences,
    generate_binding_site_h5_from_sequences
)

from imports import *
from Binary_Classification_pepTrans_module import pepTrans_Binary_Classification, wd_schedule
from Binding_Site_pepTrans_module import *
from Binding_Affinity_pepTrans_Module import *
from Cross_Validation_pepTrans_Binding_Sites import *
from pepTrans import *


if __name__ == '__main__':
    import argparse
    import timeit as ti

    parser = argparse.ArgumentParser(description='pepTrans - Prediction Modes')

    parser.add_argument(
        '--mode',
        choices=['b', 'bs', 'ba', 'pbd', 'vs', 'vas'],
        required=True,
        help="'b' binary classification, 'bs' binding site prediction, "
             "'ba' binding affinity regression, 'pbd' peptide-PBD prediction, "
             "'vs' virtual screening, 'vas' virtual alanine scanning"
    )

    parser.add_argument('--csv', type=str, help='CSV path for training or evaluation using precomputed embeddings')
    parser.add_argument('--vas_csv', type=str, help='CSV path for alanine-scanning mutated input')
    parser.add_argument('--label', type=str, default='Label', help='Target column in CSV')
    parser.add_argument('--train_h5', type=str, help='Training HDF5 path for binding site prediction')
    parser.add_argument('--test_h5', type=str, help='Test HDF5 path for binding site prediction')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    parser.add_argument('--weights', type=str, help='Path to pretrained model weights')
    parser.add_argument('--test_csv', type=str, help='Path to test CSV file using precomputed embeddings')

    # ============================
    # New sequence-input arguments
    # ============================
    parser.add_argument(
        '--input_sequences',
        type=str,
        help='CSV file containing raw peptide and protein sequences'
    )

    parser.add_argument(
        '--peptide_col',
        type=str,
        default='Peptide',
        help='Column name for peptide sequence'
    )

    parser.add_argument(
        '--protein_col',
        type=str,
        default='Protein',
        help='Column name for protein/receptor sequence'
    )

    parser.add_argument(
        '--peptide_label_col',
        type=str,
        default=None,
        help='Column name for peptide residue labels for binding-site task'
    )

    parser.add_argument(
        '--save_embeddings',
        type=str,
        default='pepTrans_generated_embeddings.csv',
        help='Path to save generated embedding CSV/H5'
    )

    parser.add_argument(
        '--embedding_device',
        type=str,
        default='cuda',
        help='Device for ProtT5 embedding generation: cuda or cpu'
    )

    args = parser.parse_args()

    configure_device(args.gpu)

    t1 = ti.default_timer()

    # ============================================================
    # 1. Binary peptide-protein interaction prediction
    # ============================================================
    if args.mode == 'b':

        # New: raw sequence input -> ProtT5 embeddings -> pepTrans prediction
        if args.input_sequences and args.weights:
            generated_csv = generate_binary_csv_from_sequences(
                input_csv=args.input_sequences,
                output_csv=args.save_embeddings,
                peptide_col=args.peptide_col,
                protein_col=args.protein_col,
                label_col=args.label if args.label else None,
                device=args.embedding_device
            )

            pepTrans_binary_classification(
                None,
                None,
                args.label,
                pretrained_weights=args.weights,
                test_csv_path=generated_csv
            )

        # Existing: train from precomputed embedding CSV
        elif args.csv:
            X, y = load_csv_dataset(args.csv, args.label)

            pepTrans_binary_classification(
                X,
                y,
                args.label,
                pretrained_weights=None,
                test_csv_path=None
            )

        # Existing: evaluate from precomputed embedding CSV
        elif args.weights and args.test_csv:
            pepTrans_binary_classification(
                None,
                None,
                args.label,
                pretrained_weights=args.weights,
                test_csv_path=args.test_csv
            )

        else:
            print(
                "Provide one of the following:\n"
                "1) --csv for training using precomputed embeddings\n"
                "2) --weights + --test_csv for evaluation using precomputed embeddings\n"
                "3) --input_sequences + --weights for sequence-based prediction"
            )
            sys.exit(1)

    # ============================================================
    # 2. Peptide binding residue prediction
    # ============================================================
    elif args.mode == 'bs':

        # New: raw sequence input -> ProtT5 embeddings -> H5 -> pepTrans prediction
        if args.input_sequences and args.weights:
            generated_h5 = args.save_embeddings

            if not generated_h5.endswith(".h5"):
                generated_h5 = generated_h5.replace(".csv", ".h5")

            generate_binding_site_h5_from_sequences(
                input_csv=args.input_sequences,
                output_h5=generated_h5,
                peptide_col=args.peptide_col,
                protein_col=args.protein_col,
                peptide_label_col=args.peptide_label_col,
                max_peptide_len=50,
                max_protein_len=500,
                device=args.embedding_device
            )

            pepTrans_binding_site_prediction(
                h5_file_path=None,
                h5_test_file_path=generated_h5,
                pretrained_weights_path=args.weights,
                n_splits=5
            )

        # Existing: evaluate using precomputed H5
        elif args.weights and args.test_h5:
            pepTrans_binding_site_prediction(
                h5_file_path=None,
                h5_test_file_path=args.test_h5,
                pretrained_weights_path=args.weights,
                n_splits=5
            )

        # Existing: train using precomputed H5
        elif args.train_h5:
            pepTrans_binding_site_prediction(
                h5_file_path=args.train_h5,
                h5_test_file_path=None,
                pretrained_weights_path=None,
                n_splits=5
            )

        else:
            print(
                "Provide one of the following:\n"
                "1) --train_h5 for training using precomputed H5 embeddings\n"
                "2) --weights + --test_h5 for evaluation using precomputed H5 embeddings\n"
                "3) --input_sequences + --weights for sequence-based binding-site prediction"
            )
            sys.exit(1)

    # ============================================================
    # 3. Binding affinity regression
    # ============================================================
    elif args.mode == 'ba':

        # New: raw sequence input -> ProtT5 embeddings -> pepTrans affinity prediction
        if args.input_sequences and args.weights:
            generated_csv = generate_binary_csv_from_sequences(
                input_csv=args.input_sequences,
                output_csv=args.save_embeddings,
                peptide_col=args.peptide_col,
                protein_col=args.protein_col,
                label_col=args.label if args.label else None,
                device=args.embedding_device
            )

            pepTrans_binding_affinity_regression(
                X=None,
                y=None,
                label_col=args.label,
                input_dim=2048,
                pretrained_weights=args.weights,
                test_csv_path=generated_csv
            )

        elif args.weights and args.test_csv:
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
            print(
                "Provide one of the following:\n"
                "1) --csv for training using precomputed embeddings\n"
                "2) --weights + --test_csv for evaluation using precomputed embeddings\n"
                "3) --input_sequences + --weights for sequence-based affinity prediction"
            )
            sys.exit(1)

    # ============================================================
    # 4. Peptide-PBD prediction
    # ============================================================
    elif args.mode == 'pbd':

        # New: raw sequence input -> ProtT5 embeddings -> pepTrans PBD prediction
        if args.input_sequences and args.weights:
            generated_csv = generate_binary_csv_from_sequences(
                input_csv=args.input_sequences,
                output_csv=args.save_embeddings,
                peptide_col=args.peptide_col,
                protein_col=args.protein_col,
                label_col=args.label if args.label else None,
                device=args.embedding_device
            )

            pepTrans_binary_classification(
                None,
                None,
                args.label,
                task='pbd',
                pretrained_weights=args.weights,
                test_csv_path=generated_csv
            )

        elif args.csv:
            X, y = load_csv_dataset(args.csv, args.label)

            pepTrans_binary_classification(
                X,
                y,
                args.label,
                task='pbd',
                pretrained_weights=None,
                test_csv_path=None
            )

        elif args.weights and args.test_csv:
            pepTrans_binary_classification(
                None,
                None,
                args.label,
                task='pbd',
                pretrained_weights=args.weights,
                test_csv_path=args.test_csv
            )

        else:
            print(
                "Provide one of the following:\n"
                "1) --csv for training using precomputed embeddings\n"
                "2) --weights + --test_csv for evaluation using precomputed embeddings\n"
                "3) --input_sequences + --weights for sequence-based PBD prediction"
            )
            sys.exit(1)

    # ============================================================
    # 5. Virtual screening
    # ============================================================
    elif args.mode == 'vs':

        # New: raw sequence input -> ProtT5 embeddings -> pepTrans virtual screening
        if args.input_sequences and args.weights:
            generated_csv = generate_binary_csv_from_sequences(
                input_csv=args.input_sequences,
                output_csv=args.save_embeddings,
                peptide_col=args.peptide_col,
                protein_col=args.protein_col,
                label_col=args.label if args.label else None,
                device=args.embedding_device
            )

            pepTrans_binary_classification(
                None,
                None,
                args.label,
                task='vs',
                pretrained_weights=args.weights,
                test_csv_path=generated_csv
            )

        elif args.csv:
            X, y = load_csv_dataset(args.csv, args.label)

            pepTrans_binary_classification(
                X,
                y,
                args.label,
                task='vs',
                pretrained_weights=None,
                test_csv_path=None
            )

        else:
            print(
                "Provide one of the following:\n"
                "1) --csv for virtual screening training/evaluation using precomputed embeddings\n"
                "2) --input_sequences + --weights for sequence-based virtual screening"
            )
            sys.exit(1)

    # ============================================================
    # 6. Virtual alanine scanning
    # ============================================================
    elif args.mode == 'vas':

        if not args.csv or not args.vas_csv:
            print("Both --csv WT dataset and --vas_csv mutant dataset are required for virtual alanine scanning.")
            sys.exit(1)

        X, y = load_regression_dataset(args.csv, target_col=args.label)
        X_VAS, y_VAS = load_regression_dataset(args.vas_csv, target_col=args.label)

        pepTrans_alanine_scanning_regression(
            X,
            X_VAS,
            y,
            y_VAS
        )

    else:
        print(f"Mode {args.mode} not implemented.")
        sys.exit(1)

    t2 = ti.default_timer()
    print(f"\nTotal execution time: {t2 - t1:.2f} seconds")