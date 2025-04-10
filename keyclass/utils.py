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
from sklearn.metrics import precision_score, recall_score
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

def log(metrics: Union[List, Dict], filename: str, results_dir: str, split: str):
    """Logging function
        
        Parameters
        ----------
        metrics: Union[List, Dict]
            The metrics to log and save in a file
        filename: str
            Name of the file
        results_dir: str
            Path to results directory
        split: str
            Train/test split
    """
    if isinstance(metrics, list):
        assert len(metrics) == 3, "Metrics must be of length 3!"
        results = dict()
        results['Accuracy'] = metrics[0]
        results['Precision'] = metrics[1]
        results['Recall'] = metrics[2]
    elif isinstance(metrics, np.ndarray):
        assert len(metrics) == 3, "Metrics must be of length 3!"
        results = dict()
        results['Accuracy (mean, std)'] = metrics[0].tolist()
        results['Precision (mean, std)'] = metrics[1].tolist()
        results['Recall (mean, std)'] = metrics[2].tolist()
    else:
        results = metrics

    # Create directory if it doesn´t exist
    if not os.path.exists(results_dir): os.makedirs(results_dir)

    filename_complete = join(results_dir, f'{split}_{filename}.txt')

    print(f'Saving results in {filename_complete}...')

    with open(filename_complete, 'w', encoding='utf-8') as f:
        f.write(json.dumps(results))


def compute_metrics(y_preds: np.array,
                    y_true: np.array,
                    average: str = 'weighted',
                    verbose: bool = False):
    """Compute accuracy, recall and precision

        Parameters
        ----------
        y_preds: np.array
            Predictions
        
        y_true: np.array
            Ground truth labels
        
        average: str
            This parameter is required for multiclass/multilabel targets. If None, 
            the scores for each class are returned. Otherwise, this determines the 
            type of averaging performed on the data.
    """

    accuracy = np.mean(y_preds == y_true)
    precision = precision_score(y_true, y_preds, average=average)
    recall = recall_score(y_true, y_preds, average=average)
    #f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Add optional verbose printing without changing return values
    if verbose:
        print("\n===== Metrics =====")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
    #    print(f"F1 Score: {f1:.4f}")
    
    # Return original format as expected by existing code
    return [accuracy, precision, recall]


def compute_metrics_bootstrap(y_preds: np.array,
                              y_true: np.array,
                              average: str = 'weighted',
                              n_bootstrap: int = 100,
                              n_jobs: int = 10,
                              verbose: bool = True,
                              model_name: str = '',
                              use_wandb: bool = False,
                              run: object = None):
    """Compute bootstrapped confidence intervals (CIs) around metrics of interest. 

        Parameters
        ----------
        y_preds: np.array
            Predictions
        
        y_true: np.array
            Ground truth labels
        
        average: str
            This parameter is required for multiclass/multilabel targets. If None, 
            the scores for each class are returned. Otherwise, this determines the 
            type of averaging performed on the data.

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
    """
    output_ =  joblib.Parallel(n_jobs=n_jobs, verbose=1)(
                                joblib.delayed(compute_metrics)
                                    (y_preds[boostrap_inds], y_true[boostrap_inds]) \
                                    for boostrap_inds in [\
                                    np.random.choice(a=len(y_true), size=len(y_true)) for k in range(n_bootstrap)])
    output_ = np.array(output_)
    means = np.mean(output_, axis=0)
    stds = np.std(output_, axis=0)

    if verbose:
        print(f"\n===== Bootstrap Metrics for {model_name} =====")
        print(f"Accuracy: {means[0]:.4f} ± {stds[0]:.4f}")
        print(f"Precision: {means[1]:.4f} ± {stds[1]:.4f}")
        print(f"Recall: {means[2]:.4f} ± {stds[2]:.4f}")

    # Log to Weights & Biases (if set)
    if use_wandb:
        run.log({
            f"{model_name}/accuracy": means[0],
            f"{model_name}/accuracy_std": stds[0],
            f"{model_name}/precision": means[1],
            f"{model_name}/precision_std": stds[1],
            f"{model_name}/recall": means[2],
            f"{model_name}/recall_std": stds[2],
        })

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

    assert np.sum(class_balance) - 1 < 1e-3, "Class balance must be a probability, and hence sum to 1"
    assert len(class_balance) == proba_preds.shape[1], f"Only {proba_preds.shape[1]} classes in the data"

    # Get integer of max number of elements per class
    class_max_inds = [int(max_num * c) for c in class_balance]
    train_idxs = np.array([], dtype=int)

    for i in range(proba_preds.shape[1]):
        sorted_idxs = np.argsort(proba_preds[:, i])[::-1]  # gets highest probas for class
        sorted_idxs = sorted_idxs[:class_max_inds[i]]
        print(f'Confidence of least confident data point of class {i}: {proba_preds[sorted_idxs[-1], i]}')
        train_idxs = np.union1d(train_idxs, sorted_idxs)

    mask = np.zeros(len(proba_preds), dtype=bool)
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
    #if dataset not in _dataset_names:
    #    raise ValueError(f'Dataset must be one of {_dataset_names}, but received {dataset}.')
    #if split not in ['train', 'test']:
    #    raise ValueError(f'split must be one of \'train\' or \'test\', but received {split}.')

    if not exists(f"{join(path, dataset, split)}.txt"):
        raise ValueError(f'File {split}.txt does not exists in {join(path, dataset)}')

    text = open(f'{join(path, dataset, split)}.txt').readlines()

    if dataset == 'mimic':
        text = [clean_text(line) for line in text]

    return text


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).float()

    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9)


def _text_length(text: Union[List[int], List[List[int]]]):
    """
    Help function to get the length for the input text. Text can be either
    a list of ints (which means a single text as input), or a tuple of list of ints
    (representing several text inputs to the model).

    Adapted from https://github.com/UKPLab/sentence-transformers/blob/40af04ed70e16408f466faaa5243bee6f476b96e/sentence_transformers/SentenceTransformer.py#L548
    """

    if isinstance(text, dict):  #{key: value} case
        return len(next(iter(text.values())))
    elif not hasattr(text, '__len__'):  #Object has no len() method
        return 1
    elif len(text) == 0 or isinstance(text[0], int):  #Empty string or list of ints
        return len(text)
    else:
        return sum([len(t) for t in text])  #Sum of length of individual strings


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
