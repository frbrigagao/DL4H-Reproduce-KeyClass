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

from curses import raw
import torch
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
import numpy as np
from tqdm import tqdm, trange
from sklearn.metrics import f1_score, accuracy_score
import copy


def get_q_soft(p: np.ndarray):
    """Get target distribution for model refinement via self-training. 

    Soft labeling (Xie et al., 2016) derives Q by enhancing high-confidence predictions while
    demoting low-confidence ones via squaring and normalizing the current predictions.

    Parameters
    ----------
    p: Current predictions of the model.

    References
    ----------
    Junyuan Xie, Ross B. Girshick, and Ali Farhadi. 2016. Unsupervised deep embedding for clustering analysis. In ICML.
    """
    q = np.square(p) / np.sum(p, axis=0, keepdims=True)
    q = q / np.sum(q, axis=1, keepdims=True)
    return q


def train(model: torch.nn.Module,
          X_train: Union[Union[str, List[str]], np.ndarray],
          y_train: Union[torch.Tensor, np.ndarray],       # IMPORTANT: For multi-label (BCE), this should be the probabilistic labels (n_samples, n_classes). 
          device: torch.device = torch.device("cuda"),    # For single-label (CE), this should be integer labels (n_samples,).
          sample_weights: Optional[np.array] = None,      # Used only for single-label CE loss if use_noise_aware_loss is True
          epochs: int = 200,
          batch_size: int = 128,
          criterion: Callable = torch.nn.CrossEntropyLoss(reduction='none'),
          raw_text: bool = False,
          lr: float = 1e-3,
          weight_decay: float = 1e-4,
          patience: int = 2,
          use_noise_aware_loss: bool = True, # From config, relevant only for CE loss
          use_wandb: bool = False,
          run: object = None):
    """Function to train the encoder along with fully connected layers. 

    Parameters
    ----------
    model: ML/DL Model to train
    X_train: Training Data Features
    y_train: Training Data Targets.
             - For multi-label (BCE criterion): Probabilistic labels (n_samples, n_classes), dtype float32.
             - For single-label (CE criterion): Integer class labels (n_samples,), dtype int64
    device: Device to use for training. 'cuda' by default
    sample_weights: Array of weights assigned to individual samples
    epochs: Number of complete passes of the training data through the model
    batch_size: Number of samples to feed into the model before updating hyperparameters
    criterion: Loss function (or Optimizer)
    raw_text: Boolean Flag describing if raw text is to be processed (True if processing raw text, else False)
    lr: Learning Rate
    weight_decay: Weight decay parameter (for regularization/to prevent overfitting) 
    patience: Number of consecutive epochs of no performance improvement before terminating training (for early stopping)
    use_noise_aware_loss: Relevant only for single-label (CE). If True, uses sample_weights.
    use_wandb: True if Weights & Biases is set up to log the training run
    run: the Weights & Biases run object
    """
    if isinstance(y_train, np.ndarray):

        # Convert based on expected dtype for the loss function
        if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
             # Expecting probabilities for BCE
            y_train = torch.from_numpy(y_train).float()
        else:
            # Expecting integer labels for CrossEntropy
            y_train = torch.from_numpy(y_train).long()
            # y_train = torch.from_numpy(y_train)

    if raw_text == False and isinstance(X_train, np.ndarray):
        X_train = torch.from_numpy(X_train)

     # Prepare sample weights only if needed (single-label, noise-aware)
    is_bce_loss = isinstance(criterion, torch.nn.BCEWithLogitsLoss)
    use_sample_weights_for_ce = (not is_bce_loss) and use_noise_aware_loss and (sample_weights is not None)

    if use_sample_weights_for_ce:
        print("Using sample weights for CrossEntropyLoss.")
        sample_weights_tensor = torch.from_numpy(sample_weights.reshape(-1, 1)).to(device).float()
    else:
        sample_weights_tensor = None
        if is_bce_loss:
            print("Using probabilistic targets with BCEWithLogitsLoss. Sample weights (max prob) will not be applied.")
        elif not is_bce_loss and not use_noise_aware_loss:
            print("Using CrossEntropyLoss without noise-aware sample weights.")
        elif not is_bce_loss and use_noise_aware_loss and sample_weights is None:
            print("Warning: Noise-aware loss requested for CrossEntropyLoss, but no sample_weights provided.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    model = model.train()

    best_loss = np.inf
    tolcount = 0
    best_state_dict = None

    N = len(X_train)
    pbar = trange(epochs, unit="batch")

    for nep in pbar:
        pbar.set_description(f"Epoch {nep}")
        permutation = torch.randperm(N)
        running_loss = 0

        for i in range(0, N, batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i + batch_size]

            batch_x = X_train[indices] # Embeddings are already numpy/torch tensors
            batch_y = y_train[indices].to(device) # Target labels (int or float based on loss)

            # Since raw text is a list of strings, it cannot be trivially moved to the GPU using the
            # .to() method. The base encoder model takes care of this.
            if raw_text == False: 
                # If X_train is numpy array, convert batch to tensor
                if isinstance(batch_x, np.ndarray):
                    batch_x = torch.from_numpy(batch_x).to(device).float()
                else: # assume it's already a tensor slice (original implementation)
                    batch_x = batch_x.to(device)

            # model.forward should return raw logits
            out_logits = model.forward(batch_x, raw_text=raw_text)

            # Calculate loss based on criterion type
            if is_bce_loss:
                # Target batch_y is now [batch_size, n_classes], dtype float32
                # Criterion reduction='none' was set in config, so we average manually
                loss = criterion(out_logits, batch_y).mean() 
            else: # CrossEntropyLoss case
                # Target batch_y is [batch_size], dtype long
                loss_unreduced = criterion(out_logits, batch_y)
                if use_sample_weights_for_ce:
                    batch_weight = sample_weights_tensor[indices]
                    # Ensure batch_weight aligns with loss_unreduced shape if necessary (e.g., broadcasting)
                    loss = torch.mul(loss_unreduced, batch_weight.squeeze()).mean() # Squeeze if weight is (N, 1)
                else:
                    loss = loss_unreduced.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            #running_loss = running_loss + (loss.cpu().detach().numpy() * batch_size / N)
            running_loss = running_loss + (loss.item() * len(indices) / N) # Use loss.item()

        scheduler.step()

        with torch.no_grad():  # Early stopping

            pbar.set_postfix(tolerance_count=tolcount,
                             running_loss=running_loss,
                             best_loss=best_loss)
            
            # Log training to Weights & Balances (if set)
            if use_wandb:
                run.log({
                    "end_model/epoch": nep,
                    "end_model/tolerance_count": tolcount,
                    "end_model/running_loss": running_loss,
                    "end_model/best_loss": best_loss
                })

            if running_loss <= best_loss:
                best_loss = running_loss
                tolcount = 0
                best_state_dict = copy.deepcopy(model.state_dict())

            else:  # loss.cpu().detach().numpy() > best_loss:
                tolcount += 1

            if tolcount > patience:
                print('Stopping early...')
                model.load_state_dict(best_state_dict)  # Return the best model
                return model
    
    # Load best state if training finished normally (Gemini correction)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model


def self_train(model: torch.nn.Module,
               X_train: Union[str, List[str]], # Expecting raw text
               X_val: Union[str, List[str]],   # Expecting raw text for validation preds
               y_val: np.ndarray,              # Ground truth for validation (int or multi-hot)
               problem_type: str,              # 'single_label' or 'multi_label'
               device: torch.device = torch.device("cuda"),
               lr: float = 1e-5,
               weight_decay: float = 1e-4,
               batch_size: int = 32,
               q_update_interval: int = 50,
               patience: int = 3,
               self_train_thresh: float = 1 - 2e-3,
               print_eval: bool = True,
               use_wandb: bool = False,
               run: object = None):
    """Function to self train a model using KL divergence loss.

    Assumes model.forward returns logits. Applies log_softmax before KLDivLoss.
    Uses model.predict_proba (which applies sigmoid/softmax) to get P for target Q calculation.

    Parameters
    ----------
    model: ML/DL model to self train on
    X_train: Feature vectors for training dataset
    X_val: Feature vectors for validation
    y_val: Ground Truths for validation
    problem_type: str ('single_label' or 'multi_label') - needed for validation metric calculation.
    device: Device to use for self training. 'cuda' by default
    lr: Learning Rate for self training
    weight_decay: Weight decay parameter (for regularization/to prevent overfitting) for self training
    batch_size: Number of samples to feed into the model before updating hyperparameters for self training
    q_update_interval: Number of steps before q is updated for self training
    patience: Number of consecutive epochs of no performance improvement before terminating training (for early stopping) for self training
    self_train_thresh: If p matches q at a rate above this threshold for "patience" number of epochs, then self training will stop early (if predictions p are not flipping, stop early)
    print_eval: Boolean - prints validation metrics if True, and does not if False
    use_wandb: True if Weights & Biases is set up to log the training run
    run: the Weights & Biases run object
    """
    model.train()
    model.zero_grad()
    model.to(device)

    criterion = torch.nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    tolcount = 0

    # Update P every batch and Q every epoch
    N = len(X_train)

    # Convert X_train to numpy array only if it's not already (e.g., if it's a list)
    if not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train, dtype=object) # Use dtype=object for lists of strings

    pbar = trange(N // (batch_size * q_update_interval), unit="epochs")
    
    for epoch in pbar:
        pbar.set_description(f"Self-Train Epoch {epoch}")

        # Get indices for the full target update pass
        update_indices = np.random.permutation(N)
        num_updates_per_epoch = N // (batch_size * q_update_interval)

        # Predict probabilities on the entire training set (or large random subset for efficiency if N is huge)
        # This uses model.predict_proba which applies sigmoid/softmax internally based on problem type
        with torch.no_grad():

            pred_proba = model.predict_proba(X_train, batch_size=batch_size, raw_text=True, problem_type=problem_type)
            target_dist_q = get_q_soft(pred_proba) # should be of size (N, num_categories)

            # Check agreement for early stopping
            pred_labels_p = (pred_proba >= 0.5).astype(int) if problem_type == 'multi_label' else np.argmax(pred_proba, axis=1)
            target_labels_q = (target_dist_q >= 0.5).astype(int) if problem_type == 'multi_label' else np.argmax(target_dist_q, axis=1)

            # Calculate agreement based on label comparison
            if problem_type == 'multi_label':
                 # Agreement for multi-label: fraction of samples where predicted labels match target labels
                 self_train_agreement = np.mean(np.all(pred_labels_p == target_labels_q, axis=1))
            else: # single-label
                 self_train_agreement = np.mean(pred_labels_p == target_labels_q)

            if self_train_agreement > self_train_thresh:
                tolcount += 1
            else:
                tolcount = 0

            if tolcount >= patience:
                print(f"Stopping self-training early due to high agreement ({self_train_agreement:.4f} > {self_train_thresh:.4f}) for {patience} epochs.")
                break

        # Model Training Steps
        current_step_in_epoch = 0
        # Use update_indices to shuffle data access within the epoch
        shuffled_indices = np.random.permutation(N)

        for i in range(0, N, batch_size):
            if current_step_in_epoch >= q_update_interval: # Enough steps for this Q update interval
                 break

            batch_indices = shuffled_indices[i:min(i + batch_size, N)]
            if len(batch_indices) == 0: continue

            batch_x = X_train[batch_indices] # Raw text batch
            # Get corresponding Q targets for this batch
            batch_q = torch.from_numpy(target_dist_q[batch_indices]).to(device).float()

            # Forward pass: get logits
            out_logits = model.forward(batch_x, raw_text=True)
            # Apply log_softmax for KLDivLoss
            log_probs = torch.nn.functional.log_softmax(out_logits, dim=-1)

            # Calculate KL divergence loss
            loss = criterion(log_probs, batch_q)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            current_step_in_epoch += 1
            del batch_x, batch_q, out_logits, log_probs, loss # Free memory

        validation_accuracy_or_f1 = None # Use F1 for multi-label, Accuracy for single-label
        if print_eval:
            val_preds_binary = model.predict(X_val, batch_size=batch_size, raw_text=True, problem_type=problem_type) # Get binary predictions

            if problem_type == 'multi_label':
                # For multi-label validation, compute sample-averaged F1 score
                validation_accuracy_or_f1 = f1_score(y_val, val_preds_binary, average='samples', zero_division=0)
                val_metric_name = "validation_f1_samples"
            else:
                 # For single-label, compute accuracy
                validation_accuracy_or_f1 = accuracy_score(y_val, val_preds_binary)
                val_metric_name = "validation_accuracy"
        
        
        postfix_dict = {
            "tolerance_count": tolcount,
            "self_train_agreement": f"{self_train_agreement:.4f}",
        }

        if print_eval:
            postfix_dict[val_metric_name] = f"{validation_accuracy_or_f1:.4f}"

        pbar.set_postfix(postfix_dict)

        # Log training to Weights & Balances (if set)
        if use_wandb:
            log_dict = {
                "self_trained_end_model/epoch": epoch,
                "self_trained_end_model/tolerance_count": tolcount,
                "self_trained_end_model/self_train_agreement": self_train_agreement,
            }
            if print_eval and validation_accuracy_or_f1 is not None:
                 log_dict[f"self_trained_end_model/{val_metric_name}"] = validation_accuracy_or_f1
            run.log(log_dict)

    return model
