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

import argparse
import numpy as np
import torch
import os
from os.path import join, exists
import models
import utils
import train_classifier
import pickle
from datetime import datetime


def load_data(args, experiment_name):
    """
    Loads necessary data including embeddings, probabilistic labels,
    and ground truth labels (handling both single and multi-label formats).

    Args:
        args (dict): Dictionary containing configuration parameters.
        experiment_name (str): Identifier for the current experiment run.

    Returns:
        tuple: Contains loaded data arrays and flags:
               (X_train_embed_masked, y_train_lm_masked, y_train_masked,
                X_test_embed, y_test, training_labels_present,
                sample_weights_masked, proba_preds_masked)
    """

    final_preds_path = args['results_path'] + experiment_name + '/predictions'
    final_embeddings_path = args['results_path'] + experiment_name + '/data_embeddings'

    # Load probabilistic predictions from the label model
    with open(join(final_preds_path, f"{args['label_model']}_proba_preds.pkl"), 'rb') as f:
        proba_preds = pickle.load(f)

    # Derive integer labels and sample weights from label model probabilities
    y_train_lm = np.argmax(proba_preds, axis=1)
    sample_weights = np.max(proba_preds, axis=1)  # Sample weights for noise aware loss

    # Keep only very confident predictions using mask
    mask = utils.get_balanced_data_mask(proba_preds, max_num=args['max_num'], class_balance=None)

    # Load training and testing data
    # We have already encode the dataset, so we'll just load the embeddings
    with open(join(final_embeddings_path, f'train_embeddings.pkl'), 'rb') as f:
        X_train_embed = pickle.load(f)
    with open(join(final_embeddings_path, f'test_embeddings.pkl'), 'rb') as f:
        X_test_embed = pickle.load(f)

    # Load training ground truth labels
    training_labels_present = False
    y_train = None 
    train_labels_path = join(args['data_path'], args['dataset'], 'train_labels.txt')

    if exists(train_labels_path):
        
        print(f"Found training labels at: {train_labels_path}")

        with open(train_labels_path, 'r') as f:
            train_label_lines = f.readlines()

            if args['problem_type'] == 'multi_label':
                print("Processing multi-label training labels...")
                parsed_labels = []
                expected_len = -1 # To check consistency
                for i, line in enumerate(train_label_lines):
                    line = line.strip()
                    if not line: continue # Skip empty lines
                    if expected_len == -1:
                         expected_len = len(line)
                    elif len(line) != expected_len:
                         print(f"Warning: Inconsistent line length in {train_labels_path}. Line {i+1} has length {len(line)}, expected {expected_len}. Skipping line.")
                         continue

                    try:
                        # Convert the string of '0'/'1' into a list of integers
                        label_vector = [int(digit) for digit in line]
                        parsed_labels.append(label_vector)
                    except ValueError as e:
                        print(f"Warning: Could not parse line {i+1} in {train_labels_path}: '{line}'. Error: {e}. Skipping line.")
                        continue

                if parsed_labels:
                    # Convert list of lists to NumPy array with float32 dtype for BCEWithLogitsLoss compatibility
                    y_train = np.array(parsed_labels, dtype=np.float32)
                    training_labels_present = True
                    print(f"Loaded multi-label training labels. Shape: {y_train.shape}, dtype: {y_train.dtype}")
                    # Check if number of classes matches config
                    if y_train.shape[1] != args['n_classes']:
                        print(f"Warning: Number of columns in parsed training labels ({y_train.shape[1]}) does not match n_classes in config ({args['n_classes']}).")

                else:
                    print("Warning: No valid multi-label training labels found after parsing.")
                    training_labels_present = False

            elif args['problem_type'] == 'single_label':
                print("Processing single-label training labels...")
                try:
                    # Original logic for single integer labels
                    y_train = np.array([int(i.strip()) for i in train_label_lines if i.strip()])
                    training_labels_present = True
                    print(f"Loaded single-label training labels. Shape: {y_train.shape}, dtype: {y_train.dtype}")
                except ValueError as e:
                    print(f"Error parsing single-label training labels: {e}")
                    y_train = None
                    training_labels_present = False
            else:
                raise ValueError(f"Unsupported problem_type in config: {args['problem_type']}")
    else:
        print('No training labels file found.')
        y_train = None
    
    # Load test ground truth labels
    test_labels_path = join(args['data_path'], args['dataset'], 'test_labels.txt')
    if not exists(test_labels_path):
        raise FileNotFoundError(f"Test labels file not found at: {test_labels_path}")
    
    with open(test_labels_path, 'r') as f:
        test_label_lines = f.readlines()

    if not test_label_lines:
         raise ValueError("Test labels file is empty.")
    
    if args['problem_type'] == 'multi_label':
        print("Processing multi-label test labels...")
        parsed_labels = []
        expected_len = -1
        for i, line in enumerate(test_label_lines):
            line = line.strip()
            if not line: continue # Skip empty lines
            if expected_len == -1:
                 expected_len = len(line)
            elif len(line) != expected_len:
                 print(f"Warning: Inconsistent line length in {test_labels_path}. Line {i+1} has length {len(line)}, expected {expected_len}. Skipping line.")
                 continue

            try:
                label_vector = [int(digit) for digit in line]
                parsed_labels.append(label_vector)
            except ValueError as e:
                 print(f"Warning: Could not parse line {i+1} in {test_labels_path}: '{line}'. Error: {e}. Skipping line.")
                 continue

        if not parsed_labels:
             raise ValueError("Could not parse any valid multi-label test labels.")

        y_test = np.array(parsed_labels, dtype=np.float32)
        print(f"Loaded multi-label test labels. Shape: {y_test.shape}, dtype: {y_test.dtype}")
        # Check if number of classes matches config
        if y_test.shape[1] != args['n_classes']:
             print(f"Warning: Number of columns in parsed test labels ({y_test.shape[1]}) does not match n_classes in config ({args['n_classes']}).")

    elif args['problem_type'] == 'single_label':
        print("Processing single-label test labels...")
        try:
            y_test = np.array([int(i.strip()) for i in test_label_lines if i.strip()])
            print(f"Loaded single-label test labels. Shape: {y_test.shape}, dtype: {y_test.dtype}")
        except ValueError as e:
             raise ValueError(f"Error parsing single-label test labels: {e}")
    else:
         raise ValueError(f"Unsupported problem_type in config: {args['problem_type']}")

    # Print data statistics
    print('\n==== Data statistics ====')
    print(f'Size of training data: {X_train_embed.shape}, testing data: {X_test_embed.shape}')
    print(f'Size of testing labels: {y_test.shape}')

    if training_labels_present:
        print(f'Size of training labels: {y_train.shape}') # Shape depends on problem_type now
        if args['problem_type'] == 'single_label':
            print(f'Training class distribution (ground truth): {np.unique(y_train, return_counts=True)[1]/len(y_train)}')
    
    print(f'Training class distribution (label model predictions): {np.unique(y_train_lm, return_counts=True)[1]/len(y_train_lm)}')

    print('\nKeyClass only trains on the most confidently labeled data points! Applying mask...')
    print(f'Mask selects {np.sum(mask)} out of {len(X_train_embed)} training samples.')
    
    y_train_masked = y_train[mask] if training_labels_present else None  
    y_train_lm_masked = y_train_lm[mask]
    X_train_embed_masked = X_train_embed[mask]
    sample_weights_masked = sample_weights[mask]
    proba_preds_masked = proba_preds[mask] # Mask the original probabilities too
    
    print('\n==== Data statistics (after applying mask) ====')
    
    print(f'Size of masked training data: {X_train_embed_masked.shape}')
    
    if training_labels_present:
        print(f'Size of masked training labels (ground truth): {y_train_masked.shape}') # Shape depends on problem_type
        if args['problem_type'] == 'single_label':
             print(f'Masked Training class distribution (ground truth): {np.unique(y_train_masked, return_counts=True)[1]/len(y_train_masked)}')

    print(f'Masked Training class distribution (label model predictions): {np.unique(y_train_lm_masked, return_counts=True)[1]/len(y_train_lm_masked)}')

    return X_train_embed_masked, y_train_lm_masked, y_train_masked, \
     X_test_embed, y_test, \
     training_labels_present, sample_weights_masked, proba_preds_masked


def train(args_cmd, use_wandb = False, run = None, experiment_name = '', skip_self_training = False):

    args = utils.Parser(config_file_path=args_cmd.config).parse()

    # Set random seeds
    random_seed = args_cmd.random_seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Final paths
    final_preds_path = args['results_path'] + experiment_name + '/predictions'
    final_results_path = args['results_path'] + experiment_name + '/metrics'
    final_model_path = args['results_path'] + experiment_name + '/models'
    category_labels_path = None

    # Get the labels location for calculating category-specific F1 scores later on
    if args['problem_type'] == 'multi_label':
        category_labels_path = join(args['data_path'], args['dataset'], 'labels.txt')

    X_train_embed_masked, y_train_lm_masked, y_train_masked, \
     X_test_embed, y_test, training_labels_present, \
     sample_weights_masked, proba_preds_masked = load_data(args, experiment_name)

    # Train downstream classifier
    if args['use_custom_encoder']:
        encoder = models.CustomEncoder(
            pretrained_model_name_or_path=args['base_encoder'],
            device=args['device'])
    else:
        encoder = models.Encoder(model_name=args['base_encoder'],
                                 device=args['device'])

    classifier = models.FeedForwardFlexible(
        encoder_model=encoder,
        h_sizes=args['h_sizes'],
        activation=eval(args['activation']),
        device=torch.device(args['device']))
    
    criterion_instance = eval(args['criterion'])
    is_multi_label_loss = isinstance(criterion_instance, torch.nn.BCEWithLogitsLoss)

    if is_multi_label_loss:
        print("Using probabilistic labels (proba_preds_masked) as target for BCEWithLogitsLoss.")
        y_train_target = proba_preds_masked # Use probabilities directly
    else: # Single-label CrossEntropyLoss
        print("Using discrete label model predictions (y_train_lm_masked) as target for CrossEntropyLoss.")
        y_train_target = y_train_lm_masked # Use argmax labels
    
    print('\n===== Training the downstream classifier =====\n')

    model = train_classifier.train(model=classifier,
                                   device=torch.device(args['device']),
                                   X_train=X_train_embed_masked,
                                   y_train=y_train_target,
                                   sample_weights=sample_weights_masked
                                   if args['use_noise_aware_loss'] else None,
                                   epochs=args['end_model_epochs'],
                                   batch_size=args['end_model_batch_size'],
                                   criterion=criterion_instance,
                                   raw_text=False,
                                   lr=eval(args['end_model_lr']),
                                   weight_decay=eval(args['end_model_weight_decay']),
                                   patience=args['end_model_patience'],
                                   use_noise_aware_loss=args['use_noise_aware_loss'], 
                                   use_wandb = use_wandb,
                                   run = run)

    # Saving the end model
    if not os.path.exists(final_model_path): os.makedirs(final_model_path)
    model_name = f'end_model.pth'
    print(f'Saving model {model_name}...')
    with open(join(final_model_path, model_name), 'wb') as f:
        torch.save(model, f)

    # Running predictions on end model
    end_model_preds_train_proba = model.predict_proba(torch.from_numpy(X_train_embed_masked), batch_size=512, raw_text=False, problem_type=args['problem_type'])
    end_model_preds_test_proba = model.predict_proba(torch.from_numpy(X_test_embed), batch_size=512, raw_text=False, problem_type=args['problem_type'])

    # Generate discrete predictions
    if args['problem_type'] == 'multi_label':
         end_model_preds_train_discrete = (end_model_preds_train_proba >= 0.5).astype(int)
         end_model_preds_test_discrete = (end_model_preds_test_proba >= 0.5).astype(int)
    else:
         end_model_preds_train_discrete = np.argmax(end_model_preds_train_proba, axis=1)
         end_model_preds_test_discrete = np.argmax(end_model_preds_test_proba, axis=1)

    # Save probabilities
    with open(join(final_preds_path, 'end_model_preds_train_proba.pkl'), 'wb') as f:
        pickle.dump(end_model_preds_train_proba, f)
    with open(join(final_preds_path, 'end_model_preds_test_proba.pkl'), 'wb') as f:
        pickle.dump(end_model_preds_test_proba, f)
     # Save discrete predictions
    with open(join(final_preds_path, 'end_model_preds_train_discrete.pkl'), 'wb') as f:
        pickle.dump(end_model_preds_train_discrete, f)
    with open(join(final_preds_path, 'end_model_preds_test_discrete.pkl'), 'wb') as f:
        pickle.dump(end_model_preds_test_discrete, f)

     # Evaluate end model performance
    print("\n--- Evaluating End Model ---")

    # Print statistics
    if training_labels_present:
        training_metrics_with_gt = utils.compute_metrics(
            y_preds=end_model_preds_train_discrete,
            y_true=y_train_masked,
            average=args['average'],
            problem_type=args['problem_type'])
        
        # Log statistics to file
        utils.log(metrics=training_metrics_with_gt,
                  filename='end_model_with_ground_truth',
                  results_dir=final_results_path,
                  split='train',
                  problem_type=args['problem_type'])

    # Evaluate training performance against the label model's discrete predictions
    # This checks how well the model learned the weak labels it was trained on (for single-label)
    if args['problem_type'] == 'single_label':
        print("Evaluating on masked training data (vs label model discrete preds)...")
        training_metrics_with_lm = utils.compute_metrics(y_preds=end_model_preds_train_discrete, y_true=y_train_lm_masked, average=args['average'], problem_type=args['problem_type'])
        
        utils.log(metrics=training_metrics_with_lm,
                filename='end_model_with_label_model',
                results_dir=final_results_path,
                split='train',
                problem_type=args['problem_type'])

    # Evaluate on Test Set
    print("Evaluating on test data (vs ground truth)...")
    testing_metrics = utils.compute_metrics_bootstrap(
        y_preds=end_model_preds_test_discrete,
        y_true=y_test,
        average=args['average'],
        n_bootstrap=args['n_bootstrap'],
        n_jobs=args['n_jobs'],
        model_name='end_model',
        use_wandb=use_wandb,
        run=run,
        problem_type=args['problem_type'])
    
    # Log metrics to file
    utils.log(metrics=testing_metrics,
              filename='end_model_with_ground_truth',
              results_dir=final_results_path,
              split='test',
              problem_type=args['problem_type'])
    
    # Calculate category-specific F1 Scores for multi-label problems
    if args['problem_type'] == 'multi_label':
        category_f1_end_model = utils.compute_category_f1_scores(
            y_true=y_test,
            y_pred=end_model_preds_test_discrete,
            labels_file_path=category_labels_path
        )
        utils.log_category_metrics(
            metrics_dict=category_f1_end_model,
            filename='end_model_with_ground_truth_category_specific', 
            results_dir=final_results_path,
            split='test'
        )
    
    # Self-Training Section
    if skip_self_training:
        print('\n===== SKIPPING self-training of the downstream classifier ======\n')
    else:
        print('\n===== Self-training the downstream classifier =====\n')

        # Fetching the raw text data for self-training
        X_train_text = utils.fetch_data(dataset=args['dataset'], path=args['data_path'], split='train')
        X_test_text = utils.fetch_data(dataset=args['dataset'], path=args['data_path'], split='test')

        # Self-train the end model
        model_self_trained = train_classifier.self_train(
            model=model, # Start from the already trained model
            X_train=X_train_text,
            X_val=X_test_text,
            y_val=y_test,
            problem_type=args['problem_type'],
            device=torch.device(args['device']),
            lr=eval(args['self_train_lr']),
            weight_decay=eval(args['self_train_weight_decay']),
            patience=args['self_train_patience'],
            batch_size=args['self_train_batch_size'],
            q_update_interval=args['q_update_interval'],
            self_train_thresh=eval(args['self_train_thresh']),
            print_eval=True,
            use_wandb = use_wandb,
            run = run)

        # Save self-trained end model to file
        model_name = f'end_model_self_trained.pth'
        print(f'Saving model {model_name}...')
        with open(join(final_model_path, model_name), 'wb') as f:
            torch.save(model, f)

        # Run predictions
        end_model_preds_test = model_self_trained.predict_proba(X_test_text, batch_size=args['self_train_batch_size'], raw_text=True, problem_type=args['problem_type'])

        # Generate discrete predictions
        if args['problem_type'] == 'multi_label':
             st_model_preds_test_discrete = (end_model_preds_test >= 0.5).astype(int)
        else:
             st_model_preds_test_discrete = np.argmax(end_model_preds_test, axis=1)

        # Save probabilities
        with open(join(final_preds_path, 'end_model_self_trained_preds_test_proba.pkl'), 'wb') as f:
             pickle.dump(end_model_preds_test, f)
        # Save discrete predictions
        with open(join(final_preds_path, 'end_model_self_trained_preds_test_discrete.pkl'), 'wb') as f:
             pickle.dump(st_model_preds_test_discrete, f)

        # Print statistics
        print("Evaluating self-trained model on test data (vs ground truth)...")
        testing_metrics = utils.compute_metrics_bootstrap(
            y_preds=st_model_preds_test_discrete,
            y_true=y_test,
            average=args['average'],
            n_bootstrap=args['n_bootstrap'],
            n_jobs=args['n_jobs'],
            model_name='self_trained_end_model',
            use_wandb=use_wandb,
            run=run,
            problem_type=args['problem_type'])
        
        # Log statistics to file
        utils.log(metrics=testing_metrics,
                filename='end_model_with_ground_truth_self_trained',
                results_dir=final_results_path,
                split='test',
                problem_type=args['problem_type'])
        
        # Calculate category-specific F1 Scores for multi-label problems
        if args['problem_type'] == 'multi_label':
            category_f1_end_model_st = utils.compute_category_f1_scores(
                y_true=y_test,
                y_pred=st_model_preds_test_discrete,
                labels_file_path=category_labels_path
            )
            utils.log_category_metrics(
                metrics_dict=category_f1_end_model_st,
                filename='end_model_with_ground_truth_self_trained_category_specific', 
                results_dir=final_results_path,
                split='test'
            )
    
    # Return metrics 
    return testing_metrics


def test(args_cmd, end_model_path, end_model_self_trained_path, experiment_name):

    args = utils.Parser(config_file_path=args_cmd.config).parse()

    # Set random seeds
    random_seed = args_cmd.random_seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print('\n===== Testing the end model downstream classifier =====\n')

    X_train_embed_masked, y_train_lm_masked, y_train_masked, \
     X_test_embed, y_test, training_labels_present, \
     sample_weights_masked, proba_preds_masked = load_data(args, experiment_name)
    
    category_labels_path = None

    # Get the labels location for calculating category-specific F1 scores later on
    if args['problem_type'] == 'multi_label':
        category_labels_path = join(args['data_path'], args['dataset'], 'labels.txt')

    model = torch.load(end_model_path)

    end_model_preds_train_proba = model.predict_proba(torch.from_numpy(X_train_embed_masked), batch_size=512, raw_text=False, problem_type=args['problem_type'])
    end_model_preds_test_proba = model.predict_proba(torch.from_numpy(X_test_embed), batch_size=512, raw_text=Fals,problem_type=args['problem_type'])

    # Generate discrete predictions
    if args['problem_type'] == 'multi_label':
         end_model_preds_train_discrete = (end_model_preds_train_proba >= 0.5).astype(int)
         end_model_preds_test_discrete = (end_model_preds_test_proba >= 0.5).astype(int)
    else:
         end_model_preds_train_discrete = np.argmax(end_model_preds_train_proba, axis=1)
         end_model_preds_test_discrete = np.argmax(end_model_preds_test_proba, axis=1)

    # Evaluate end model
    print("--- Non-Self-Trained Model Evaluation ---")
    if training_labels_present and y_train_masked is not None:
        print("On Masked Training Set (vs Ground Truth):")
        training_metrics_with_gt = utils.compute_metrics(
            y_preds=end_model_preds_train_discrete,
            y_true=y_train_masked,
            average=args['average'],
            problem_type=args['problem_type'],
            verbose=True)
        # print('training_metrics_with_gt', training_metrics_with_gt)

    if args['problem_type'] == 'single_label':
        print("On Masked Training Set (vs Label Model Discrete Preds):")
        training_metrics_with_lm = utils.compute_metrics(
            y_preds=end_model_preds_train_discrete,
            y_true=y_train_lm_masked,
            average=args['average'],
            problem_type=args['problem_type'],
            verbose=True)
        # print('training_metrics_with_lm', training_metrics_with_lm)

    print("On Test Set (vs Ground Truth):")
    testing_metrics = utils.compute_metrics_bootstrap(
        y_preds=end_model_preds_test_discrete,
        y_true=y_test,
        average=args['average'],
        n_bootstrap=args['n_bootstrap'],
        n_jobs=args['n_jobs'],
        problem_type=args['problem_type'],
        verbose=True)
    
     # Calculate category-specific F1 Scores for multi-label problems
    if args['problem_type'] == 'multi_label':
        category_f1_end_model = utils.compute_category_f1_scores(
            y_true=y_test,
            y_pred=end_model_preds_test_discrete,
            labels_file_path=category_labels_path
        )

    print('\n===== Testing the end model self-trained downstream classifier =====\n')

    X_test_text = utils.fetch_data(dataset=args['dataset'], path=args['data_path'], split='test')

    model = torch.load(end_model_self_trained_path)

    st_model_preds_test_proba = model.predict_proba(
        X_test_text, batch_size=args['self_train_batch_size'], raw_text=True, problem_type=args['problem_type'])
    
    # Generate discrete predictions
    if args['problem_type'] == 'multi_label':
        st_model_preds_test_discrete = (st_model_preds_test_proba >= 0.5).astype(int)
    else:
        st_model_preds_test_discrete = np.argmax(st_model_preds_test_proba, axis=1)

    # Evaluate Self-Trained Model
    print("--- Self-Trained Model Evaluation ---")
    print("On Test Set (vs Ground Truth):")
    final_self_trained_metrics = utils.compute_metrics_bootstrap(
        y_preds=st_model_preds_test_discrete,
        y_true=y_test,
        average=args['average'],
        n_bootstrap=args['n_bootstrap'],
        n_jobs=args['n_jobs'],
        problem_type=args['problem_type'],
        verbose=True)

    # Calculate category-specific F1 Scores for multi-label problems
    if args['problem_type'] == 'multi_label':
        category_f1_end_model_st = utils.compute_category_f1_scores(
            y_true=y_test,
            y_pred=st_model_preds_test_discrete,
            labels_file_path=category_labels_path
        )
    
    # Return the metrics from the self-trained model if available, otherwise the non-self-trained ones
    return final_self_trained_metrics if final_self_trained_metrics is not None else testing_metrics