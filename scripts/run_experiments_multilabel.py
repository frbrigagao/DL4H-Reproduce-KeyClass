import argparse
import os
import yaml
import json
import pandas as pd
import sys
import subprocess
from datetime import datetime
import glob
import time
import re

# Add keyclass directory to path for imports
sys.path.append('../keyclass/')
import utils
from dropbox_upload import upload_to_dropbox

# Experiment results csv prefix to use (may be changed as needed)
EXPERIMENT_CSV_PREFIX = 'experiment_results_mimic'

# Define datasets
DATASETS = ['mimic'] 
DATASET_DETAIL = ['filtered_descriptions', 'unfiltered_descriptions']

# Define dataset-specific parameter combinations
LEARNING_RATES = {
    'mimic': ['1e-3', '1e-4']
}

BATCH_SIZES = {
    'mimic': [128 , 64],
}

LABEL_MODELS = {
    'mimic': ['data_programming'],
}

LABELING_FUNCTIONS = {
    'mimic': [30, 40, 50, 80, 100, 150, 200, 250, 300],  
}
NGRAM_RANGE = {
    'mimic': [(1,1), (1,2), (1,3)]
}

def find_latest_results_file():
    """Find the most recent experiment results file"""
    result_files = glob.glob(f"../results_csv/{EXPERIMENT_CSV_PREFIX}_*.csv")
    if not result_files:
        return None
    
    # Sort by modification time, newest first
    result_files.sort(key=os.path.getmtime, reverse=True)
    return result_files[0]

def check_log_for_memory_error(log_file):
    """Check if the log file contains memory error messages"""
    if not os.path.exists(log_file):
        return False
        
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            return ("Unable to allocate" in content or 
                    "_ArrayMemoryError" in content or
                    "MemoryError" in content or 
                    "")
    except Exception as e:
        print(f"Error checking log file for memory errors: {e}")
        return False

def run_experiment(dataset, dataset_detail, learning_rate, batch_size, label_model, labeling_functions, ngram_range, use_wandb, keep_configs=False, skip_self_training = False):
    """Run a single experiment with the given parameters"""
    try:
        # Load the base config using utils.Parser
        config_path = f"../config_files/config_{dataset}_{dataset_detail}.yml"
        config = utils.Parser(config_file_path=config_path).parse()
        
        # Modify the config
        config['end_model_lr'] = learning_rate
        config['end_model_batch_size'] = batch_size
        config['label_model'] = label_model
        config['ngram_range'] = ngram_range
        ngmin, ngmax = ngram_range
        ngram_desc = f"{ngmin}_{ngmax}"
        
        # Set number of labeling functions if provided
        if labeling_functions is not None:
            config['topk'] = labeling_functions
        else:
            labeling_functions = config['topk']
        
        # Create a temp config file with descriptive name
        temp_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lm_prefix = "dp" if label_model == "data_programming" else "mv"
        temp_config_path = f"../config_files/temp_config_{dataset}_{dataset_detail}_lr_{learning_rate}_b_{batch_size}_lf_{labeling_functions}_ngram_{ngram_desc}_{lm_prefix}_{temp_timestamp}.yml"  
        
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Get the results directory
        results_base_dir = config['results_path']
        
        # Run the experiment
        wandb_flag = "1" if use_wandb else "0"
        skip_self_training_flag = "1" if skip_self_training else "0"
        cmd = f"python run_all_multilabel.py --config {temp_config_path} --use_wandb {wandb_flag} --skip_self_training {skip_self_training_flag}"
        
        # Record the start time
        start_time = time.time()
        
        # Run the command, allowing output to be displayed in real-time
        print(f"Executing: {cmd}")
        process = subprocess.run(cmd, shell=True, check=False)
        
        # Check if the command was successful
        if process.returncode != 0:
            print(f"Command failed with return code {process.returncode}")
            
            # Construct a pattern to find log files created during this run
            pattern = f"{dataset}_{dataset_detail}_lr_{learning_rate}_b_{batch_size}_lf_{labeling_functions}_ngram_{ngram_desc}_{lm_prefix}_*"
            log_files = glob.glob(f"../logs/{pattern}*.log")
            log_files = [f for f in log_files if os.path.getctime(f) > start_time]
            
            if log_files:
                newest_log = max(log_files, key=os.path.getctime)
                if check_log_for_memory_error(newest_log):
                    print(f"Memory error detected in log file {newest_log} for {dataset} with batch_size={batch_size}. Skipping this configuration.")
                else:
                    print(f"Process failed but no memory error detected. Check log file: {newest_log}")
            return None
        
        # Only delete the config file if keep_configs is False
        if not keep_configs:
            os.remove(temp_config_path)
            print(f"Deleted temporary config file: {temp_config_path}")
        else:
            print(f"Keeping temporary config file: {temp_config_path}")
        
        # Wait to ensure files are written
        time.sleep(5)
        
        # Construct a glob pattern to find the experiment folder
        pattern = f"{dataset}_{dataset_detail}_lr_{learning_rate}_b_{batch_size}_lf_{labeling_functions}_ngram_{ngram_desc}_{lm_prefix}_*"
        
        # Find experiment folders matching the pattern, created after start_time
        experiment_folders = glob.glob(os.path.join(results_base_dir, pattern))
        # Filter to those created after start_time
        experiment_folders = [f for f in experiment_folders if os.path.getctime(f) > start_time]
        # Sort by creation time
        experiment_folders.sort(key=os.path.getctime, reverse=True)
        
        if experiment_folders:
            experiment_folder = experiment_folders[0]
            experiment_id = os.path.basename(experiment_folder)
            log_file = f"../logs/{experiment_id}.log"
            
            return {
                'experiment_id': experiment_id,
                'results_folder': experiment_folder,
                'log_file': log_file
            }
        else:
            print(f"WARNING: Could not find experiment folder for {pattern}")
            return None
    except Exception as e:
        print(f"Error running experiment: {e}")
        return None

def extract_metrics(experiment_folder, skipped_self_training = False):
    """Extract metrics from experiment results"""
    metrics = {}
    
    # Define paths with correct prefixes
    metrics_dir = os.path.join(experiment_folder, 'metrics')
    label_model_path = os.path.join(metrics_dir, 'train_label_model_with_ground_truth.txt')
    end_model_path = os.path.join(metrics_dir, 'test_end_model_with_ground_truth.txt')
    self_trained_path = os.path.join(metrics_dir, 'test_end_model_with_ground_truth_self_trained.txt')
    
    # Extract label model metrics if available
    if os.path.exists(label_model_path):
        try:
            with open(label_model_path, 'r') as f:
                data = json.load(f)
                metrics['label_model_F1'] = data.get('F1', None)
        except Exception as e:
            print(f"Error reading label model metrics: {e}")
            metrics['label_model_F1'] = None
    else:
        metrics['label_model_F1'] = None
    
    # Extract end model metrics if available
    if os.path.exists(end_model_path):
        try:
            with open(end_model_path, 'r') as f:
                data = json.load(f)
                accuracy = data.get('F1 (mean, std)', [None, None])
                precision = data.get('Precision (mean, std)', [None, None])
                recall = data.get('Recall (mean, std)', [None, None])
                
                # Extract both mean and std values
                metrics['end_model_F1'] = accuracy[0] if isinstance(accuracy, list) and len(accuracy) > 0 else None
                metrics['end_model_F1_std'] = accuracy[1] if isinstance(accuracy, list) and len(accuracy) > 1 else None
                
                metrics['end_model_precision'] = precision[0] if isinstance(precision, list) and len(precision) > 0 else None
                metrics['end_model_precision_std'] = precision[1] if isinstance(precision, list) and len(precision) > 1 else None
                
                metrics['end_model_recall'] = recall[0] if isinstance(recall, list) and len(recall) > 0 else None
                metrics['end_model_recall_std'] = recall[1] if isinstance(recall, list) and len(recall) > 1 else None
        except Exception as e:
            print(f"Error reading end model metrics: {e}")
            metrics['end_model_F1'] = None
            metrics['end_model_F1_std'] = None
            metrics['end_model_precision'] = None
            metrics['end_model_precision_std'] = None
            metrics['end_model_recall'] = None
            metrics['end_model_recall_std'] = None
    else:
        metrics['end_model_F1'] = None
        metrics['end_model_F1_std'] = None
        metrics['end_model_precision'] = None
        metrics['end_model_precision_std'] = None
        metrics['end_model_recall'] = None
        metrics['end_model_recall_std'] = None
    
    # Extract self-trained end model metrics if available and self-training wasn´t skipped
    if skipped_self_training == False and os.path.exists(self_trained_path):
        try:
            with open(self_trained_path, 'r') as f:
                data = json.load(f)
                accuracy = data.get('F1 (mean, std)', [None, None])
                precision = data.get('Precision (mean, std)', [None, None])
                recall = data.get('Recall (mean, std)', [None, None])
                
                # Extract both mean and std values
                metrics['self_trained_F1'] = accuracy[0] if isinstance(accuracy, list) and len(accuracy) > 0 else None
                metrics['self_trained_F1_std'] = accuracy[1] if isinstance(accuracy, list) and len(accuracy) > 1 else None
                
                metrics['self_trained_precision'] = precision[0] if isinstance(precision, list) and len(precision) > 0 else None
                metrics['self_trained_precision_std'] = precision[1] if isinstance(precision, list) and len(precision) > 1 else None
                
                metrics['self_trained_recall'] = recall[0] if isinstance(recall, list) and len(recall) > 0 else None
                metrics['self_trained_recall_std'] = recall[1] if isinstance(recall, list) and len(recall) > 1 else None
        except Exception as e:
            print(f"Error reading self-trained model metrics: {e}")
            metrics['self_trained_F1'] = None
            metrics['self_trained_F1_std'] = None
            metrics['self_trained_precision'] = None
            metrics['self_trained_precision_std'] = None
            metrics['self_trained_recall'] = None
            metrics['self_trained_recall_std'] = None
    else:
        metrics['self_trained_F1'] = None
        metrics['self_trained_F1_std'] = None
        metrics['self_trained_precision'] = None
        metrics['self_trained_precision_std'] = None
        metrics['self_trained_recall'] = None
        metrics['self_trained_recall_std'] = None
    
    return metrics

def upload_experiment_files(experiment_info, results_csv_path):
    """Upload experiment files to Dropbox"""
    try:
        print("\n=== Uploading experiment files to Dropbox ===")
        
        # Upload the results folder for this experiment
        if experiment_info and 'results_folder' in experiment_info:
            results_folder_path = experiment_info['results_folder']
            print(f"Uploading results folder: {results_folder_path}")
            result = upload_to_dropbox(path=results_folder_path, is_file=False)
            if result != 0:
                print(f"ERROR: Failed to upload results folder to Dropbox (error code {result})")
            else:
                print("Successfully uploaded results folder to Dropbox")
        
        # Upload the log file for this experiment
        if experiment_info and 'log_file' in experiment_info:
            log_file_path = experiment_info['log_file']
            if os.path.exists(log_file_path):
                print(f"Uploading log file: {log_file_path}")
                result = upload_to_dropbox(path=log_file_path, is_file=True)
                if result != 0:
                    print(f"ERROR: Failed to upload log file to Dropbox (error code {result})")
                else:
                    print("Successfully uploaded log file to Dropbox")
        
        # Upload the latest results CSV
        if results_csv_path and os.path.exists(results_csv_path):
            print(f"Uploading results CSV: {results_csv_path}")
            result = upload_to_dropbox(path=results_csv_path, dest_path='/results_csv/', is_file=True)
            if result != 0:
                print(f"ERROR: Failed to upload results CSV to Dropbox (error code {result})")
            else:
                print("Successfully uploaded results CSV to Dropbox")
        
        return True
    except Exception as e:
        print(f"ERROR: Exception during Dropbox upload: {e}")
        return False

def get_default_parameters(config, dataset):
    """Get the default parameters for a dataset from its config file"""
    config_path = f"../config_files/config_{dataset}_{config}.yml"
    config = utils.Parser(config_file_path=config_path).parse()
    return {
        'topk': config['topk'],
        'end_model_lr': config['end_model_lr'],
        'end_model_batch_size': config['end_model_batch_size'],
        'label_model': config['label_model'],
        'ngram_range': tuple(config.get('ngram_range', (1, 1)))
    }

def run_all_experiments(use_wandb=False, keep_configs=False, use_dropbox=False):
    """Run all experiments and collect results"""
    
    # Initialize results DataFrame with empty data and include std columns
    latest_file = find_latest_results_file()
    if latest_file and os.path.exists(latest_file):
        try:
            results = pd.read_csv(latest_file)
            print(f"Loaded previous results from {latest_file} with {len(results)} experiments")
        except Exception as e:
            print(f"Error loading previous results: {e}. Starting fresh.")
            results = pd.DataFrame(columns=[
                'dataset', 'config', 'log_file', 'learning_rate', 'batch_size', 'label_model', 'labeling_functions', 'ngram_range',
                'label_model_F1', 
                'end_model_F1', 'end_model_F1_std',
                'end_model_precision', 'end_model_precision_std',
                'end_model_recall', 'end_model_recall_std',
                'self_trained_end_model_F1', 'self_trained_end_model_F1_std',
                'self_trained_end_model_precision', 'self_trained_end_model_precision_std',
                'self_trained_end_model_recall', 'self_trained_end_model_recall_std'
            ])
    else:
        results = pd.DataFrame(columns=[
            'dataset', 'config', 'log_file', 'learning_rate', 'batch_size', 'label_model', 'labeling_functions', 'ngram_range',
            'label_model_F1', 
            'end_model_F1', 'end_model_F1_std',
            'end_model_precision', 'end_model_precision_std',
            'end_model_recall', 'end_model_recall_std',
            'self_trained_end_model_F1', 'self_trained_end_model_F1_std',
            'self_trained_end_model_precision', 'self_trained_end_model_precision_std',
            'self_trained_end_model_recall', 'self_trained_end_model_recall_std'
        ])
    
    print(f'\n Current results file content:')
    print(results)

    # Run all experiments
    print("Starting experiments loop...")

    for dataset in DATASETS:
        for dataset_detail in DATASET_DETAIL:
            # Get parameters for this dataset
            default_params = get_default_parameters(dataset_detail, dataset)
            
            # Get available parameter options for this dataset
            lr_options = LEARNING_RATES.get(dataset, [default_params['end_model_lr']])
            batch_size_options = BATCH_SIZES.get(dataset, [default_params['end_model_batch_size']])
            label_model_options = LABEL_MODELS.get(dataset, [default_params['label_model']])
            lf_options = LABELING_FUNCTIONS.get(dataset, [default_params['topk']])
            ngram_options = NGRAM_RANGE.get(dataset, [default_params['ngram_range']])
            
            for lr in lr_options:
                for batch_size in batch_size_options:
                    for label_model in label_model_options:
                        for lf_count in lf_options:
                            for ngram in ngram_options:
                                ngmin, ngmax = ngram
                                ngram_desc = f"({ngmin},{ngmax})"
                                # Create a unique experiment identifier for display
                                exp_id = f"{dataset}_{dataset_detail}_lr_{lr}_b_{batch_size}_lf_{lf_count}_ngram_{ngram_desc}_{label_model}_*"

                                # Default behavior is to also execute the self-training run
                                skip_self_training = False
                                
                                # Check if this experiment has already been run by checking the DataFrame
                                experiment_mask = ((results['dataset'] == dataset) & 
                                                (results['config'] == dataset_detail) &
                                                (results['learning_rate'] == float(lr)) &
                                                (results['batch_size'] == batch_size) & 
                                                (results['label_model'] == label_model) &
                                                (results['labeling_functions'] == lf_count) &
                                                (results['ngram_range'] == ngram_desc)
                                                )
                                
                                # Skip if already run
                                if experiment_mask.any():
                                    print(f"Skipping already completed experiment: {exp_id}")
                                    continue
                                
                                print(f"\n=== Running experiment: dataset={dataset}, config={dataset_detail}, lr={lr}, batch_size={batch_size}, label_model={label_model}, lf_count={lf_count}, ngram_range={ngram_desc} ===\n")
                                if skip_self_training:
                                    print(f"\n=== Experiment will SKIP self-training ===\n")
                                
                                # Run experiment
                                experiment_info = run_experiment(dataset, dataset_detail, lr, batch_size, label_model, lf_count, ngram, use_wandb, keep_configs, skip_self_training)

                                if experiment_info:
                                    # Extract metrics
                                    metrics = extract_metrics(experiment_info['results_folder'], skip_self_training)
                                    
                                    # Add to results DataFrame
                                    result = pd.DataFrame([{
                                        'dataset': dataset,
                                        'config': dataset_detail,
                                        'log_file': experiment_info['log_file'],
                                        'learning_rate': float(lr),  # Store as float
                                        'batch_size': batch_size,
                                        'label_model': label_model,
                                        'labeling_functions': lf_count,
                                        'ngram_range': ngram_desc,
                                        'label_model_F1': metrics.get('label_model_F1', None),
                                        'end_model_F1': metrics.get('end_model_F1', None),
                                        'end_model_F1_std': metrics.get('end_model_F1_std', None),
                                        'end_model_precision': metrics.get('end_model_precision', None),
                                        'end_model_precision_std': metrics.get('end_model_precision_std', None),
                                        'end_model_recall': metrics.get('end_model_recall', None),
                                        'end_model_recall_std': metrics.get('end_model_recall_std', None),
                                        'self_trained_end_model_F1': metrics.get('self_trained_F1', None),
                                        'self_trained_end_model_F1_std': metrics.get('self_trained_F1_std', None),
                                        'self_trained_end_model_precision': metrics.get('self_trained_precision', None),
                                        'self_trained_end_model_precision_std': metrics.get('self_trained_precision_std', None),
                                        'self_trained_end_model_recall': metrics.get('self_trained_recall', None),
                                        'self_trained_end_model_recall_std': metrics.get('self_trained_recall_std', None)
                                    }])
                                    
                                    results = pd.concat([results, result], ignore_index=True)
                                    
                                    # Save intermediate results
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    results_csv_path = f"../results_csv/{EXPERIMENT_CSV_PREFIX}_{timestamp}.csv"
                                    results.to_csv(results_csv_path, index=False)
                                    
                                    print(f"Updated results table. Current shape: {results.shape}")
                                    
                                    # Upload experiment files to Dropbox if selected
                                    if use_dropbox:
                                        upload_experiment_files(experiment_info, results_csv_path)
                                else:
                                    print(f"Experiment failed or could not find results folder for {exp_id}. Moving to next experiment.")
        
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results_path = f"../results_csv/{EXPERIMENT_CSV_PREFIX}_final_{timestamp}.csv"
    results.to_csv(final_results_path, index=False)
    print(f"Saved final results to {final_results_path}")
    
    # Upload final results to Dropbox if selected
    if use_dropbox:
        result = upload_to_dropbox(path=final_results_path, dest_path='/results_csv', is_file=True)
        if result != 0:
            print(f"ERROR: Failed to upload final results CSV to Dropbox (error code {result})")
        else:
            print("Successfully uploaded final results CSV to Dropbox")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run KeyClass experiments with different hyperparameters for MIMIC dataset')
    parser.add_argument('--use_wandb', action='store_true', default=False, help='Use Weights & Biases for experiment tracking')
    parser.add_argument('--keep_configs', action='store_true', default=False, help='Keep temporary config files after experiments')
    parser.add_argument('--use_dropbox', action='store_true', default=False, help='Uploads files to dropbox.')
    args = parser.parse_args()
    
    # Make sure the results directory exists
    os.makedirs("../results", exist_ok=True)
    # Make sure the results_csv directory exists
    os.makedirs("../results_csv", exist_ok=True)
    
    print(f"Running all experiments with use_wandb={args.use_wandb}, keep_configs={args.keep_configs}, use_dropbox={args.use_dropbox}")
    results = run_all_experiments(use_wandb=args.use_wandb, keep_configs=args.keep_configs, use_dropbox=args.use_dropbox)
    
    # Display summary
    print("\n=== Experiment Results Summary ===")
    print(results)