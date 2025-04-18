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

import json
from os.path import join, exists
import re
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from datetime import datetime
import torch
from yaml import load, dump
from yaml import CLoader as Loader, CDumper as Dumper
import logging
import sys
import os

def setup_logging(log_file= None):
    """Set up logging to file and console
    
    Parameters
    ----------
    log_file : str
        Path to log file. If None, logging to file is disabled.
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logging.info(f"Log location: {log_file}")
    
    # Replace print with logging.info
    def print_to_log(*args, **kwargs):
        text = " ".join(map(str, args))
        if 'end' in kwargs:
            text += kwargs['end']
        else:
            text += '\n'
        logging.info(text.rstrip())
    
    # Replace the built-in print with a custom function
    __builtins__['print'] = print_to_log
    
    return logger

def log(metrics: Union[List, Dict, np.ndarray], filename: str, results_dir: str, split: str, problem_type: str):
    """Logging function

        Parameters
        ----------
        metrics: Union[List, Dict, np.ndarray]
            The metrics to log and save in a file. Can be a list/dict or a numpy array (from bootstrap).
        filename: str
            Name of the file
        results_dir: str
            Path to results directory
        split: str
            Train/test split
        problem_type: str (single_label, multi_label)
            Type of classification problem
    """
    results = dict()

    if isinstance(metrics, list):
        # Infer problem type based on length of the list (heuristic)
        if problem_type == 'multi_label':
             results['F1'] = metrics[0]
             results['Precision'] = metrics[1]
             results['Recall'] = metrics[2]
        elif problem_type == 'single_label':
             results['Accuracy'] = metrics[0]
             results['F1'] = metrics[1]
             results['Precision'] = metrics[2]
             results['Recall'] = metrics[3]
        else:
            # Fallback or raise error if format is unexpected
            print(f"Warning: Unexpected metrics list length {len(metrics)} in log function.")
            results['metrics'] = metrics # Log raw list
        # assert len(metrics) == 3, "Metrics must be of length 3!"
        # results = dict()
        # results['Accuracy'] = metrics[0]
        # results['Precision'] = metrics[1]
        # results['Recall'] = metrics[2]
    elif isinstance(metrics, np.ndarray):

         # Infer problem type based on shape (rows = metrics, cols = mean, std)
        if problem_type == 'multi_label': # multi-label [f1, precision, recall]
            results['F1 (mean, std)'] = metrics[0].tolist()
            results['Precision (mean, std)'] = metrics[1].tolist()
            results['Recall (mean, std)'] = metrics[2].tolist()
        elif problem_type == 'single_label': # single-label [accuracy, f1, precision, recall]
            results['Accuracy (mean, std)'] = metrics[0].tolist()
            results['F1 (mean, std)'] = metrics[1].tolist()
            results['Precision (mean, std)'] = metrics[2].tolist()
            results['Recall (mean, std)'] = metrics[3].tolist()
        else:
             print(f"Warning: Unexpected metrics array shape {metrics.shape} in log function.")
             results['metrics_array'] = metrics.tolist() # Log raw array

        # assert len(metrics) == 3, "Metrics must be of length 3!"
        # results = dict()
        # results['Accuracy (mean, std)'] = metrics[0].tolist()
        # results['Precision (mean, std)'] = metrics[1].tolist()
        # results['Recall (mean, std)'] = metrics[2].tolist()
    elif isinstance(metrics, dict):
        results = metrics # Assume dict is already formatted correctly
    else:
        raise TypeError(f"Unsupported type for metrics: {type(metrics)}")
    # else:
    #     results = metrics

    # Create directory if it doesn´t exist
    if not os.path.exists(results_dir): os.makedirs(results_dir)

    filename_complete = join(results_dir, f'{split}_{filename}.txt')

    print(f'Saving results in {filename_complete}...')

    with open(filename_complete, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        #f.write(json.dumps(results))

def log_category_metrics(metrics_dict: Dict[str, float], filename: str, results_dir: str, split: str):
    """
    Logs the category-specific metrics dictionary to a file.

    Args:
        metrics_dict: Dictionary mapping category names to scores (e.g., F1).
        filename: Base name for the output file.
        results_dir: Directory to save the results file.
        split: Data split identifier ('train' or 'test').
    """
    # Create directory if it doesn't exist
    if not exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory: {results_dir}")

    filename_complete = join(results_dir, f'{split}_{filename}_by_category.json') # Save as JSON

    print(f'Saving category-specific results in {filename_complete}...')

    try:
        with open(filename_complete, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=4) # Use indent for readability
    except Exception as e:
        print(f"Error saving category metrics to {filename_complete}: {e}")

def compute_category_f1_scores(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               labels_file_path: str) -> Dict[str, float]:
    """
    Computes the F1 score for each individual category in a multi-label setting,
    reading category names from a specified file.

    Args:
        y_true: Ground truth labels (binary multi-hot format, shape [n_samples, n_classes]).
        y_pred: Predicted labels (binary multi-hot format, shape [n_samples, n_classes]).
        labels_file_path: Path to a text file containing category names, one per line,
                          in the order corresponding to the columns of y_true/y_pred.

    Returns:
        A dictionary mapping category names to their individual F1 scores. Returns None if
        category names cannot be loaded or if validation fails.
    """
    # --- Load Category Names from File ---
    category_names = []
    try:
        print(f"Loading category names from: {labels_file_path}")
        with open(labels_file_path, 'r', encoding='utf-8') as f:
            category_names = [line.strip() for line in f if line.strip()]
        if not category_names:
            print(f"Error: No category names found in {labels_file_path}.")
            return None
        print(f"Loaded {len(category_names)} category names.")
    except FileNotFoundError:
        print(f"Error: Labels file not found at {labels_file_path}.")
        return None
    except Exception as e:
        print(f"Error reading labels file {labels_file_path}: {e}")
        return None

    # --- Input Validation ---
    if not isinstance(y_true, np.ndarray) or y_true.ndim != 2:
        print("Error: y_true must be a 2D NumPy array (n_samples, n_classes).")
        return None
    if not isinstance(y_pred, np.ndarray) or y_pred.ndim != 2:
        print("Error: y_pred must be a 2D NumPy array (n_samples, n_classes).")
        return None
    if y_true.shape != y_pred.shape:
        print(f"Error: Shape mismatch: y_true {y_true.shape} and y_pred {y_pred.shape} must have the same shape.")
        return None
    if y_true.shape[1] != len(category_names):
        print(f"Error: Number of columns in y_true/y_pred ({y_true.shape[1]}) must match the number of loaded category names ({len(category_names)}).")
        return None

    # --- Compute Report ---
    print("Computing classification report for category-specific F1 scores...")
    # Use classification_report to get metrics per class
    # Set zero_division=0 to handle cases where a class might have no true/predicted samples
    try:
        report = classification_report(y_true, y_pred, target_names=category_names, output_dict=True, zero_division=0)
    except Exception as e:
        print(f"Error generating classification report: {e}")
        return None

    # --- Extract F1 Scores ---
    category_f1_scores = {}
    print("\n--- Category-Specific F1 Scores ---")
    for name in category_names:
        if name in report:
            # Check if the category dictionary and 'f1-score' key exist
            if isinstance(report[name], dict) and 'f1-score' in report[name]:
                f1 = report[name]['f1-score']
                category_f1_scores[name] = f1
                print(f"{name}: {f1:.4f}")
            else:
                 print(f"Warning: F1 score not found for category '{name}' in the report. Assigning 0.0.")
                 category_f1_scores[name] = 0.0
        else:
            print(f"Warning: Category '{name}' not found in classification report keys. Assigning 0.0.")
            category_f1_scores[name] = 0.0 # Assign 0 if category name itself is missing from report keys

    # Also calculate and print sample-averaged metrics for comparison
    try:
        f1_samples = f1_score(y_true, y_pred, average='samples', zero_division=0)
        print(f"\nSample-Averaged F1 (for comparison): {f1_samples:.4f}")
    except Exception as e:
         print(f"Could not compute sample-averaged F1: {e}")
    print("--- End Category F1 Scores ---")

    return category_f1_scores

def compute_metrics(y_preds: np.array,
                    y_true: np.array,
                    average: str = 'weighted',
                    verbose: bool = False,
                    problem_type: str = 'single_label'):
    """Compute appropriate metrics based on input shapes.

        Handles both single-label (1D arrays) and multi-label (2D arrays).
        For multi-label, returns [F1, Precision, Recall].
        For single-label, returns [Accuracy, F1, Precision, Recall].

        Parameters
        ----------
        y_preds: np.array
            Predictions (1D for single-label, 2D for multi-label)
        y_true: np.array
            Ground truth labels (1D for single-label, 2D for multi-label)
        average: str
            Averaging method for precision, recall, F1. Crucial for multi-label/multi-class.
            Use 'samples' for MIMIC as per paper footnote 4/config.
            Use 'weighted' or 'macro' for single-label benchmarks.
        verbose: bool
            If True, print computed metrics.
        problem_type: str (single_label or multi_label)

        Returns
        -------
        List: Computed metrics based on problem type.
    """

    # Infer problem type based on the dimensions of y_true
    # if y_true.ndim > 1 and y_true.shape[1] > 1:
    #     problem_type = 'multi_label'
    #     # Ensure predictions are also 2D for multi-label
    #     if y_preds.ndim == 1:
    #          # This case shouldn't happen if predict function is correct, but handle defensively
    #          print("Warning: y_preds is 1D for multi-label task. This might indicate an issue.")
    #          # Attempt to reshape or handle appropriately if possible, otherwise raise error
    #          # For now, we assume y_preds is correctly shaped as 2D by the caller
    #          pass
    # else:
    #     problem_type = 'single_label'

    if problem_type == 'multi_label':
        # Multi-label metrics: F1, Precision, Recall (using specified average, e.g., 'samples')
        precision = precision_score(y_true, y_preds, average=average, zero_division=0)
        recall = recall_score(y_true, y_preds, average=average, zero_division=0)
        f1 = f1_score(y_true, y_preds, average=average, zero_division=0)

        if verbose:
            print("\n===== Metrics (Multi-Label) =====")
            print(f"Average Method: {average}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")

        return [f1, precision, recall]

    else: # Single-label metrics: Accuracy, F1, Precision, Recall
        accuracy = accuracy_score(y_true, y_preds)
        precision = precision_score(y_true, y_preds, average=average, zero_division=0)
        recall = recall_score(y_true, y_preds, average=average, zero_division=0)
        f1 = f1_score(y_true, y_preds, average=average, zero_division=0)

        if verbose:
            print("\n===== Metrics (Single-Label) =====")
            print(f"Average Method: {average}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")

        return [accuracy, f1, precision, recall]


    # accuracy = np.mean(y_preds == y_true)
    # precision = precision_score(y_true, y_preds, average=average)
    # recall = recall_score(y_true, y_preds, average=average)
    #f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # # Add optional verbose printing without changing return values
    # if verbose:
    #     print("\n===== Metrics =====")
    #     print(f"Accuracy: {accuracy:.4f}")
    #     print(f"Precision: {precision:.4f}")
    #     print(f"Recall: {recall:.4f}")
    #    print(f"F1 Score: {f1:.4f}")
    
    # Return original format as expected by existing code
    #return [accuracy, precision, recall]


def compute_metrics_bootstrap(y_preds: np.array,
                              y_true: np.array,
                              average: str = 'weighted',
                              n_bootstrap: int = 100,
                              n_jobs: int = 10,
                              verbose: bool = True,
                              model_name: str = '',
                              use_wandb: bool = False,
                              run: object = None,
                              problem_type: str = 'single_label'):
    """Compute bootstrapped confidence intervals (CIs) around metrics of interest. 

        Handles both single-label and multi-label based on input shapes.

        Parameters
        ----------
        y_preds: np.array
            Predictions (1D or 2D)
        y_true: np.array
            Ground truth labels (1D or 2D)
        average: str
            Averaging method passed to compute_metrics.
        n_bootstrap: int
            Number of boostrap samples to compute CI.
        n_jobs: int
            Number of jobs to run in parallel.
        verbose: bool
            If True, print the metrics with names, means, and standard deviations
        model_name: str
            The name of the model being logged (eg: end_model, self_trained_end_model)
        use_wandb: bool
            True if Weights & Biases logging is enabled for the run
        run: object
            Weights & Biases logging object
        problem_type: str 
            Problem type (single_label or multi_label)

        Returns
        -------
        np.ndarray: Array of shape (num_metrics, 2) containing mean and std dev for each metric.
                    Shape depends on single-label (4 metrics) vs multi-label (3 metrics).
    """

    # Infer problem type based on input dimension AFTER bootstrap sampling
    # We determine based on the original y_true before passing to parallel jobs
    if problem_type == 'multi_label':
        metric_names = ['F1', 'Precision', 'Recall']
    else:
        metric_names = ['Accuracy', 'F1', 'Precision', 'Recall']

     # Run compute_metrics in parallel for each bootstrap sample
    output_ = joblib.Parallel(n_jobs=n_jobs, verbose=1)(
                                joblib.delayed(compute_metrics)(
                                    y_preds[boostrap_inds],
                                    y_true[boostrap_inds],
                                    average=average, # Pass the averaging method,
                                    problem_type=problem_type
                                ) for boostrap_inds in [ \
                                    np.random.choice(a=len(y_true), size=len(y_true), replace=True) \
                                    for _ in range(n_bootstrap) # Use _ for loop variable
                                ])

    # output_ =  joblib.Parallel(n_jobs=n_jobs, verbose=1)(
    #                             joblib.delayed(compute_metrics)
    #                                 (y_preds[boostrap_inds], y_true[boostrap_inds]) \
    #                                 for boostrap_inds in [\
    #                                 np.random.choice(a=len(y_true), size=len(y_true)) for k in range(n_bootstrap)])

    output_ = np.array(output_)
    means = np.mean(output_, axis=0)
    stds = np.std(output_, axis=0)

    # Ensure means and stds have the expected number of elements
    if len(means) != len(metric_names) or len(stds) != len(metric_names):
         print(f"Warning: Mismatch between expected metrics ({len(metric_names)}) and computed ({len(means)}). Check compute_metrics output.")
         # Adjust metric_names or handle error as appropriate
         # For now, proceed but logging might be incorrect
         pass

    if verbose:
        print(f"\n===== Bootstrap Metrics for {model_name} ({problem_type}, average='{average}') =====")
        for i, name in enumerate(metric_names):
            if i < len(means): # Check index bounds
                 print(f"{name}: {means[i]:.4f} ± {stds[i]:.4f}")
            else:
                 print(f"{name}: Not computed")

        # print(f"\n===== Bootstrap Metrics for {model_name} =====")
        # print(f"Accuracy: {means[0]:.4f} ± {stds[0]:.4f}")
        # print(f"Precision: {means[1]:.4f} ± {stds[1]:.4f}")
        # print(f"Recall: {means[2]:.4f} ± {stds[2]:.4f}")

    # Log to Weights & Biases (if set)
    if use_wandb and run:
        log_dict = {}
        for i, name in enumerate(metric_names):
            if i < len(means): # Check index bounds before logging
                log_dict[f"{model_name}/{name.lower()}"] = means[i]
                log_dict[f"{model_name}/{name.lower()}_std"] = stds[i]
        if log_dict: # Only log if we have metrics
            run.log(log_dict)
        else:
            print("Warning: No metrics computed to log to W&B.")

    # if use_wandb:
    #     run.log({
    #         f"{model_name}/accuracy": means[0],
    #         f"{model_name}/accuracy_std": stds[0],
    #         f"{model_name}/precision": means[1],
    #         f"{model_name}/precision_std": stds[1],
    #         f"{model_name}/recall": means[2],
    #         f"{model_name}/recall_std": stds[2],
    #     })

    return np.stack([means, stds], axis=1)


def get_balanced_data_mask(proba_preds: np.array,
                           max_num: int = 7000,
                           class_balance: Optional[np.array] = None):
    """Utility function to keep only the most confident predictions, while maintaining class balance

        Parameters
        ---------- 
        proba_preds: Probabilistic labels of data points
        max_num: Maximum number of data points per class
        class_balance: Prevalence of each class

    """
    if class_balance is None:  # Assume all classes are equally likely
        class_balance = np.ones(proba_preds.shape[1]) / proba_preds.shape[1]

    # Allow for slight floating point inaccuracies in sum
    if not np.isclose(np.sum(class_balance), 1.0):
         raise ValueError(f"Class balance must sum to 1, but sums to {np.sum(class_balance)}")
    #assert np.sum(class_balance) - 1 < 1e-3, "Class balance must be a probability, and hence sum to 1"

    assert len(class_balance) == proba_preds.shape[1], f"Class balance length ({len(class_balance)}) does not match number of classes ({proba_preds.shape[1]})"

    # Get integer of max number of elements per class
    class_max_inds = [int(max_num * c) for c in class_balance]
    train_idxs = np.array([], dtype=int)

    for i in range(proba_preds.shape[1]):

        if class_max_inds[i] > 0: # Only process if we need samples from this class
            sorted_idxs = np.argsort(proba_preds[:, i])[::-1]  # gets highest probas for class
            # Ensure we don't take more indices than available or needed
            num_to_take = min(class_max_inds[i], len(sorted_idxs))
            if num_to_take > 0:
                selected_idxs = sorted_idxs[:num_to_take]
                print(f'Confidence of least confident data point of class {i}: {proba_preds[selected_idxs[-1], i]} (Selected {len(selected_idxs)} points)')
                train_idxs = np.union1d(train_idxs, selected_idxs)

        # sorted_idxs = np.argsort(proba_preds[:, i])[::-1]  # gets highest probas for class
        # sorted_idxs = sorted_idxs[:class_max_inds[i]]
        # print(f'Confidence of least confident data point of class {i}: {proba_preds[sorted_idxs[-1], i]}')
        # train_idxs = np.union1d(train_idxs, sorted_idxs)

    mask = np.zeros(len(proba_preds), dtype=bool)
    if len(train_idxs) > 0:
        mask[train_idxs] = True
    return mask


def clean_text(sentences: Union[str, List[str]]):
    """Utility function to clean sentences
    """
    if isinstance(sentences, str):
        sentences = [sentences]

    for i, text in enumerate(sentences):
        text = text.lower()
        text = re.sub(r'<.*?>|[\.`\',;\?\*\[\]\(\)-:_]*|[0-9]*', '', text)
        text = re.sub(r'[\r\n]+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with a single space
        sentences[i] = text

    return sentences


def fetch_data(dataset='imdb', path='~/', split='train'):
    """Fetches a dataset by its name

	    Parameters
	    ---------- 
	    dataset: str
	        List of text to be encoded. 

	    path: str
	        Path to the stored data. 

	    split: str
	        Whether to fetch the train or test dataset. Options are one of 'train' or 'test'. 
    """
    #_dataset_names = ['agnews', 'amazon', 'dbpedia', 'imdb', 'mimic']
   
    _VALID_SPLITS = ['train', 'test']
    if split not in _VALID_SPLITS:
         raise ValueError(f'split must be one of {_VALID_SPLITS}, but received {split}.')
    
    filepath = join(path, dataset, f"{split}.txt")

    if not exists(filepath):
        raise ValueError(f'File {split}.txt does not exist in {join(path, dataset)}')

    text = open(f'{join(path, dataset, split)}.txt').readlines()

    # Try reading with utf-8 first, fallback to latin-1 if needed
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.readlines()
    except UnicodeDecodeError:
        print(f"Warning: UTF-8 decoding failed for {filepath}. Trying latin-1.")
        with open(filepath, 'r', encoding='latin-1') as f:
            text = f.readlines()

    # Perform text cleaning on mimic datasets
    if 'mimic' in dataset:
        text = [clean_text(line)[0] for line in text]

    return text


def mean_pooling(model_output, attention_mask):
    """
    Performs mean pooling on the token embeddings using the attention mask.
    Handles Hugging Face model output objects.
    """
    # --- CHANGE START ---
    # Check if the model_output object has the 'last_hidden_state' attribute
    if hasattr(model_output, 'last_hidden_state'):
        token_embeddings = model_output.last_hidden_state
    # Optional: Add back a check for tuple/list if needed for other encoder types
    elif isinstance(model_output, (tuple, list)) and model_output:
        token_embeddings = model_output[0]
    else:
        # If the expected attribute is missing, raise a more informative error
        raise TypeError(f"model_output of type {type(model_output)} does not have 'last_hidden_state' attribute needed for mean pooling.")
    # --- CHANGE END ---

    # Check if model_output is a tuple/list and has at least one element
    # if not isinstance(model_output, (tuple, list)) or not model_output:
    #     print(f"model_output: {model_output}")
    #     print(f"Model output type: {type(model_output)}")
    #     raise ValueError("model_output must be a non-empty tuple or list")
    

    #token_embeddings = model_output[0]  #First element of model_output contains all token embeddings

    # Ensure attention_mask has the correct dimensions
    if attention_mask.ndim == token_embeddings.ndim -1:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    elif attention_mask.ndim == token_embeddings.ndim:
         input_mask_expanded = attention_mask.float() # Assume mask is already expanded if dims match
    else:
         raise ValueError(f"Attention mask dimension ({attention_mask.ndim}) incompatible with token embeddings dimension ({token_embeddings.ndim})")

    # Perform pooling
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    return sum_embeddings / sum_mask


def _text_length(text: Union[str, List[str], Dict[str, List[int]]]): # Adjusted type hint
    """

    Help function to get the length for the input text. Text can be either
    a string, a list of strings, or a dictionary mapping feature names to list of ints (token IDs).

    PREVIOUS COMMENT: Help function to get the length for the input text. Text can be either a list of ints (which means a single text as input), or a tuple of list of ints (representing several text inputs to the model).

    Adapted from https://github.com/UKPLab/sentence-transformers/blob/40af04ed70e16408f466faaa5243bee6f476b96e/sentence_transformers/SentenceTransformer.py#L548
    """

    if isinstance(text, dict):  #{key: value} case, typical for tokenized inputs
        # Return the length of the first list of token IDs found in the dictionary values
        for value in text.values():
            if isinstance(value, (list, torch.Tensor)) and hasattr(value, '__len__'):
                 # If it's a tensor, get its first dimension size
                 return value.shape[0] if isinstance(value, torch.Tensor) else len(value)
        return 0 # Return 0 if no suitable list/tensor found
    elif isinstance(text, str): # Single string
        return len(text)
    elif isinstance(text, list): # List of strings or list of token IDs
        if not text: # Empty list
            return 0
        elif isinstance(text[0], int): # List of token IDs
            return len(text)
        elif isinstance(text[0], str): # List of strings
            return sum(len(t) for t in text) # Sum lengths if list of strings
        else:
             # Fallback for unexpected list content
             return len(text)
    elif not hasattr(text, '__len__'):  #Object has no len() method (e.g., int, float)
        return 1 # Treat as single item
    else: # General case for other iterable types
        try:
             return len(text)
        except TypeError:
             return 1 # Fallback if len() is not supported

    # if isinstance(text, dict):  #{key: value} case
    #     return len(next(iter(text.values())))
    # elif not hasattr(text, '__len__'):  #Object has no len() method
    #     return 1
    # elif len(text) == 0 or isinstance(text[0], int):  #Empty string or list of ints
    #     return len(text)
    # else:
    #     return sum([len(t) for t in text])  #Sum of length of individual strings


class Parser:

    def __init__(
            self,
            config_file_path='../config_files/default_config.yml',
            default_config_file_path='../config_files/default_config.yml'):
        """Class to read and parse the config.yml file
		"""
        self.config_file_path = config_file_path
        with open(default_config_file_path, 'rb') as f:
            self.default_config = load(f, Loader=Loader)

    def parse(self):
        with open(self.config_file_path, 'rb') as f:
            self.config = load(f, Loader=Loader)

        for key, value in self.default_config.items():
            if ('target' not in key) and ((key not in list(self.config.keys()))
                                          or (self.config[key] is None)):
                self.config[key] = self.default_config[key]
                print(f'KEY NOT FOUND! Setting the value of {key} to {self.default_config[key]}!')

        target_present = False
        for key in self.config.keys():
            if 'target' in key:
                target_present = True
                break
        if not target_present: 
            raise ValueError("Target must be present.")
        
        # This was overwriting the .yml configuration file. Disabled. 
        #self.save_config()
        return self.config

    def save_config(self):
        with open(self.config_file_path, 'w') as f:
            dump(self.config, f)
