# coding=utf-8
# MIT License

# Copyright (c) 2020 Carnegie Mellon University, Auton Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys

sys.path.append('../keyclass_multilabel/')

import utils
import models
import create_lfs
import numpy as np
import pickle
import argparse
import torch
import os
from os.path import join, exists


def run(args_cmd, use_wandb = False, run = None, experiment_name = ''):
    """
    Generates weak labels for training data using specified labeling functions
    and evaluates the label model if ground truth is available.
    Handles both single-label and multi-label classification tasks.
    """

    args = utils.Parser(config_file_path=args_cmd.config).parse()
    print("Configuration Arguments:")
    print(args)

    final_preds_path = args['results_path'] + experiment_name + '/predictions'
    final_results_path = args['results_path'] + experiment_name + '/metrics'
    # final_embeddings_path = args['results_path'] + experiment_name + '/data_embeddings'

    # --- Load Training Data Text ---
    print(f"Fetching training text data for dataset: {args['dataset']}...")
    train_text = utils.fetch_data(dataset=args['dataset'], path=args['data_path'], split='train')
    print(f"Loaded {len(train_text)} training text samples.")

    # --- Load Ground Truth Training Labels (Conditional) ---
    y_train = None
    training_labels_present = False
    train_labels_filepath = join(args['data_path'], args['dataset'], 'train_labels.txt')

    if exists(train_labels_filepath):
        print(f"Found training labels file: {train_labels_filepath}. Loading labels...")
        with open(train_labels_filepath, 'r') as f:
            label_lines = f.readlines()

        if args['problem_type'] == 'multi_label':
            # Load multi-hot encoded labels (e.g., "0101...")
            print("Processing multi-label ground truth labels...")
            y_train_list = []
            for line in label_lines:
                # Remove newline and convert string of '0'/'1' to list of ints
                cleaned_line = line.strip()
                if cleaned_line: # Avoid processing empty lines
                    try:
                        label_vector = [int(char) for char in cleaned_line]
                        y_train_list.append(label_vector)
                    except ValueError:
                        print(f"Warning: Could not parse multi-label line: '{cleaned_line}'. Skipping.")
            if y_train_list:
                y_train = np.array(y_train_list, dtype=np.int64) # Use int64 or float32 depending on downstream needs
                print(f"Loaded {y_train.shape[0]} multi-label ground truth vectors with {y_train.shape[1]} classes.")
            else:
                 print("Warning: Training labels file exists but no valid multi-label lines found.")

        elif args['problem_type'] == 'single_label':
             # Load single integer labels per line
            print("Processing single-label ground truth labels...")
            try:
                y_train = np.array([int(i.strip()) for i in label_lines if i.strip()], dtype=np.int64)
                print(f"Loaded {len(y_train)} single-label ground truth labels.")
            except ValueError:
                 print(f"Warning: Could not parse single-label lines in {train_labels_filepath}. Check file format.")
        else:
            print(f"Warning: Unknown problem_type '{args['problem_type']}' specified in config.")

        if y_train is not None and len(y_train) > 0:
            if len(y_train) != len(train_text):
                 print(f"ERROR: Mismatch between number of text samples ({len(train_text)}) and loaded labels ({len(y_train)}). Check data files.")
                 # Decide how to handle mismatch - potentially exit or proceed with caution
                 # For now, proceed but acknowledge the potential issue
                 sys.exit(1)
            training_labels_present = True
        else:
            print("ERROR: Training labels file exists but loading resulted in empty or invalid data.")
            y_train = None # Ensure y_train is None if loading failed
            training_labels_present = False
            sys.exit(1)

    else:
        print(f"No training labels file found at: {train_labels_filepath}")
        training_labels_present = False

    # if exists(join(args['data_path'], args['dataset'], 'train_labels.txt')):
    #     with open(join(args['data_path'], args['dataset'], 'train_labels.txt'), 'r') as f:
    #         y_train = f.readlines()
    #     y_train = np.array([int(i.replace('\n', '')) for i in y_train])
    #     training_labels_present = True
    # else:
    #     y_train = None
    #     training_labels_present = False
    #     print('No training labels found!')

    # with open(join(final_embeddings_path, 'train_embeddings.pkl'), 'rb') as f:
    #     X_train = pickle.load(f)

    # Print dataset statistics
    # print(f"Getting labels for the {args['dataset']} data...")
    # print(f'Size of the data: {len(train_text)}')
    # if training_labels_present:
    #     print('Class distribution', np.unique(y_train, return_counts=True))

    # --- Print Dataset Statistics ---
    print(f"\n--- Dataset Statistics ---")
    print(f"Dataset: {args['dataset']}")
    print(f"Problem Type: {args['problem_type']}")
    print(f'Total training samples: {len(train_text)}')
    if training_labels_present:
        print(f'Ground truth training labels available: Yes')
        if args['problem_type'] == 'single_label':
            unique_labels, counts = np.unique(y_train, return_counts=True)
            print(f'Ground truth class distribution: {dict(zip(unique_labels, counts))}')
            print(f'Ground truth class proportions: {dict(zip(unique_labels, counts/len(y_train)))}')
        elif args['problem_type'] == 'multi_label':
             # Calculate prevalence per class for multi-label
             class_prevalence = np.mean(y_train, axis=0)
             print(f'Ground truth class prevalence (proportion of samples per class):')
             for i, prev in enumerate(class_prevalence):
                 print(f'  Class {i}: {prev:.4f}')
    else:
        print('Ground truth training labels available: No')
    print(f"--- End Statistics ---")


    # Load label names/descriptions
    label_names = []
    for a in args:
        if 'target' in a: label_names.append(args[a])
    print(f"\nLoaded {len(label_names)} class descriptions.")
    if len(label_names) != args['n_classes']:
         print(f"ERROR: Number of target descriptions ({len(label_names)}) does not match n_classes ({args['n_classes']}) in config.")
         sys.exit(1)

    # --- Create Labeling Functions & Generate Probabilistic Labels ---
    print("\n--- Generating Weak Labels using Label Model ---")
    labeler = create_lfs.CreateLabellingFunctions(
        base_encoder=args['base_encoder'],
        device=torch.device(args['device']),
        label_model=args['label_model'])
    
    proba_preds = labeler.get_labels(
        text_corpus=train_text,
        label_names=label_names,
        min_df=args['min_df'],
        ngram_range=args['ngram_range'],
        topk=args['topk'],
        y_train=y_train,
        label_model_lr=args['label_model_lr'],
        label_model_n_epochs=args['label_model_n_epochs'],
        verbose=True,
        n_classes=args['n_classes'])
    print("Generated probabilistic labels (proba_preds) with shape:", proba_preds.shape)

    # y_train_pred = np.argmax(proba_preds, axis=1)

    # --- Save Probabilistic Predictions ---
    print("\n--- Saving Label Model Outputs ---")
    if not os.path.exists(final_preds_path):
        os.makedirs(final_preds_path)
        print(f"Created directory: {final_preds_path}")
    proba_preds_filename = f"{args['label_model']}_proba_preds.pkl"
    proba_preds_filepath = join(final_preds_path, proba_preds_filename)
    with open(proba_preds_filepath, 'wb') as f:
        pickle.dump(proba_preds, f)
    print(f"Saved probabilistic predictions to: {proba_preds_filepath}")


    # --- Evaluate Label Model Against Ground Truth (if available) ---
    print("\n--- Evaluating Label Model Performance (if ground truth available) ---")
    if training_labels_present:
        print("Ground truth training labels found. Evaluating...")

        if args['problem_type'] == 'multi_label':
            print("Evaluating multi-label predictions...")
            # Generate binary predictions from probabilities using a 0.5 threshold
            y_train_pred_binary = (proba_preds >= 0.5).astype(int)

            print('Label Model Binary Predictions (Top 5 samples):\n', y_train_pred_binary[:5])
            print('Ground Truth Labels (Top 5 samples):\n', y_train[:5])

            # Use the modified utils.compute_metrics for multi-label
            print(f"Calculating metrics with average='{args['average']}'...") # Should be 'samples' for MIMIC
            # Assuming compute_metrics returns [f1, precision, recall] for multi-label
            # Pass problem_type explicitly if needed by compute_metrics implementation
            training_metrics_with_gt = utils.compute_metrics(
                y_preds=y_train_pred_binary,
                y_true=y_train,
                average=args['average'],
                verbose=False,
                problem_type=args['problem_type'])

            # Ensure we have 3 metrics returned
            if len(training_metrics_with_gt) == 3:
                 f1_val, precision_val, recall_val = training_metrics_with_gt
                 print(f"Label Model Training F1 ({args['average']}): {f1_val:.4f}")
                 print(f"Label Model Training Precision ({args['average']}): {precision_val:.4f}")
                 print(f"Label Model Training Recall ({args['average']}): {recall_val:.4f}")

                 # Log metrics to file
                 utils.log(metrics=training_metrics_with_gt,
                           filename='label_model_with_ground_truth',
                           results_dir=final_results_path,
                           split='train',
                           problem_type=args['problem_type'])

                 # Log to Weights & Biases if enabled
                 if use_wandb:
                     # Include the averaging method in the metric name for clarity
                     wandb_metrics = {
                         f"label_model/f1_{args['average']}": f1_val,
                         f"label_model/precision_{args['average']}": precision_val,
                         f"label_model/recall_{args['average']}": recall_val
                     }
                     run.log(wandb_metrics)
                     print("Logged multi-label metrics to W&B.")
            else:
                 print(f"Warning: Expected 3 metrics (F1, P, R) from compute_metrics for multi-label, but received {len(training_metrics_with_gt)}.")

        elif args['problem_type'] == 'single_label':
            print("Evaluating single-label predictions...")
            # Generate discrete predictions using argmax
            y_train_pred = np.argmax(proba_preds, axis=1)

            print('Label Model Discrete Predictions (Top 5):', y_train_pred[:5])
            print('Ground Truth Labels (Top 5):', y_train[:5])
            print('Label Model Predictions: Unique values and counts:',
                  np.unique(y_train_pred, return_counts=True))

            # Calculate simple accuracy (for single-label)
            training_accuracy = np.mean(y_train_pred == y_train)
            print(f'Label Model Training Accuracy: {training_accuracy:.4f}')

             # Log simple accuracy to Weights & Biases if enabled
            if use_wandb:
                 run.log({"label_model/training_accuracy": training_accuracy})
                 print("Logged single-label accuracy to W&B.")

            # Use the modified utils.compute_metrics for single-label
            print(f"Calculating metrics with average='{args['average']}'...") # Likely 'weighted' for benchmarks
            # Assuming compute_metrics returns [accuracy, f1, precision, recall] for single-label
             # Pass problem_type explicitly if needed by compute_metrics implementation
            training_metrics_with_gt = utils.compute_metrics(
                y_preds=y_train_pred,
                y_true=y_train,
                average=args['average'],
                verbose=False,
                problem_type=args['problem_type'])

             # Ensure we have 4 metrics returned
            if len(training_metrics_with_gt) == 4:
                 acc_val, f1_val, precision_val, recall_val = training_metrics_with_gt
                 print(f"Label Model Training Accuracy (from metrics func): {acc_val:.4f}")
                 print(f"Label Model Training F1 ({args['average']}): {f1_val:.4f}")
                 print(f"Label Model Training Precision ({args['average']}): {precision_val:.4f}")
                 print(f"Label Model Training Recall ({args['average']}): {recall_val:.4f}")

                 # Log detailed metrics to file
                 metrics_dict = {'Accuracy': acc_val, 'F1': f1_val, 'Precision': precision_val, 'Recall': recall_val}
                 utils.log(metrics=metrics_dict,
                           filename='label_model_with_ground_truth',
                           results_dir=final_results_path,
                           split='train',
                           problem_type=args['problem_type'])
                 # Log detailed metrics to W&B if enabled (optional, accuracy already logged)
                 if use_wandb:
                     wandb_metrics = {
                          f"label_model/f1_{args['average']}": f1_val,
                          f"label_model/precision_{args['average']}": precision_val,
                          f"label_model/recall_{args['average']}": recall_val
                     }
                     run.log(wandb_metrics)
                     print("Logged detailed single-label metrics to W&B.")
            else:
                 print(f"Warning: Expected 4 metrics (Acc, F1, P, R) from compute_metrics for single-label, but received {len(training_metrics_with_gt)}.")

        else:
            print(f"Skipping evaluation for unknown problem_type '{args['problem_type']}'.")

    else:
        print("No ground truth training labels available. Skipping label model evaluation.")

    print("\n--- Labeling Data Script Finished ---")

    # # Print statistics
    # print('Label Model Predictions: Unique value and counts',
    #       np.unique(y_train_pred, return_counts=True))
    
    # if training_labels_present:

    #     # training_accuracy = np.mean(y_train_pred == y_train)

    #     # # Log the label model training accuracy to Weights & Biases
    #     # if use_wandb:
    #     #     run.log({"label_model/training_accuracy": training_accuracy})

    #     # print('Label Model Training Accuracy', training_accuracy)

    #     # Log the metrics
    #     training_metrics_with_gt = utils.compute_metrics(
    #         y_preds=y_train_pred, y_true=y_train, average=args['average'], problem_type=args['problem_type'])
        
    #     if args['problem-type'] == 'multi_label':
    #         print(f"Label Model Training F1 Score: {training_metrics_with_gt[0]}")
    #     else: # single-label
    #         print(f"Label Model Training Accuracy: {training_metrics_with_gt[0]}")
    #         if use_wandb:
    #             run.log({"label_model/training_accuracy": training_metrics_with_gt[0]})
        
    #     utils.log(metrics=training_metrics_with_gt,
    #               filename='label_model_with_ground_truth',
    #               results_dir=final_results_path,
    #               split='train',
    #               problem_type=args['problem_type'])


# if __name__ == "__main__":
#     parser_cmd = argparse.ArgumentParser()
#     parser_cmd.add_argument('--config', default='../default_config.yml', help='Configuration file')
#     args_cmd = parser_cmd.parse_args()

#     run(args_cmd)
