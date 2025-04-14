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

# Define datasets
#DATASETS = ['imdb', 'agnews', 'dbpedia', 'amazon']

DATASETS = ['agnews', 'amazon', 'dbpedia'] # Skipping imdb and putting dbpedia last

# Define dataset-specific parameter combinations
LEARNING_RATES = {
 #   'imdb': ['1e-3', '1e-4'],
    'agnews': ['1e-3', '1e-4'],
    'dbpedia': ['1e-3'], # Will only test default learning rate of 1e-3 due to time/hw constraints
    'amazon': ['1e-3', '1e-4']
}

BATCH_SIZES = {
  #  'imdb': [128, 64, 32],
    'agnews': [128, 64, 32],
    'dbpedia': [128], # Will only test default batch size due to time/hw contraints
    'amazon': [128, 64, 32]
}

LABEL_MODELS = {
#    'imdb': ['data_programming', 'majority_vote'],
    'agnews': ['data_programming', 'majority_vote'],
    'dbpedia': ['data_programming', 'majority_vote'],
    'amazon': ['data_programming', 'majority_vote']
}

LABELING_FUNCTIONS = {
 #   'imdb': [50, 100, 200, 300],
    'agnews': [300], # Default number of LFs
    'dbpedia': [30, 50, 100, 250, 300],  # 15 is already being tested by first script
    'amazon': [300] # Default number of LFs
}

def find_latest_results_file():
    """Find the most recent experiment results file"""
    result_files = glob.glob("../results/experiment_results_2_*.csv")
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
                    "MemoryError" in content)
    except Exception as e:
        print(f"Error checking log file for memory errors: {e}")
        return False

def run_experiment(dataset, learning_rate, batch_size, label_model, labeling_functions, use_wandb, keep_configs=False, skip_self_training = False):
    """Run a single experiment with the given parameters"""
    try:
        # Load the base config using utils.Parser
        config_path = f"../config_files/config_{dataset}.yml"
        config = utils.Parser(config_file_path=config_path).parse()
        
        # Modify the config
        config['end_model_lr'] = learning_rate
        config['end_model_batch_size'] = batch_size
        config['label_model'] = label_model
        
        # Set number of labeling functions if provided
        if labeling_functions is not None:
            config['topk'] = labeling_functions
        else:
            labeling_functions = config['topk']
        
        # Create a temp config file with descriptive name
        temp_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_config_path = f"../config_files/temp_config_{dataset}_lr_{learning_rate}_b_{batch_size}_lf_{labeling_functions}_{label_model}_{temp_timestamp}.yml"  
        
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Get the results directory
        results_base_dir = config['results_path']
        
        # Run the experiment
        wandb_flag = "1" if use_wandb else "0"
        skip_self_training_flag = "1" if skip_self_training else "0"
        cmd = f"python run_all.py --config {temp_config_path} --use_wandb {wandb_flag} --skip_self_training {skip_self_training_flag}"
        
        # Record the start time
        start_time = time.time()
        
        # Run the command, allowing output to be displayed in real-time
        print(f"Executing: {cmd}")
        process = subprocess.run(cmd, shell=True, check=False)
        
        # Check if the command was successful
        if process.returncode != 0:
            print(f"Command failed with return code {process.returncode}")
            
            # Construct a pattern to find log files created during this run
            pattern = f"{dataset}_lr_{learning_rate}_b_{batch_size}_lf_*"
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
        lm_prefix = "dp" if label_model == "data_programming" else "mv"
        pattern = f"{dataset}_lr_{learning_rate}_b_{batch_size}_lf_{labeling_functions}_{lm_prefix}*"
        
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
                metrics['label_model_accuracy'] = data.get('Accuracy', None)
        except Exception as e:
            print(f"Error reading label model metrics: {e}")
            metrics['label_model_accuracy'] = None
    else:
        metrics['label_model_accuracy'] = None
    
    # Extract end model metrics if available
    if os.path.exists(end_model_path):
        try:
            with open(end_model_path, 'r') as f:
                data = json.load(f)
                accuracy = data.get('Accuracy (mean, std)', [None, None])
                precision = data.get('Precision (mean, std)', [None, None])
                recall = data.get('Recall (mean, std)', [None, None])
                
                # Extract both mean and std values
                metrics['end_model_accuracy'] = accuracy[0] if isinstance(accuracy, list) and len(accuracy) > 0 else None
                metrics['end_model_accuracy_std'] = accuracy[1] if isinstance(accuracy, list) and len(accuracy) > 1 else None
                
                metrics['end_model_precision'] = precision[0] if isinstance(precision, list) and len(precision) > 0 else None
                metrics['end_model_precision_std'] = precision[1] if isinstance(precision, list) and len(precision) > 1 else None
                
                metrics['end_model_recall'] = recall[0] if isinstance(recall, list) and len(recall) > 0 else None
                metrics['end_model_recall_std'] = recall[1] if isinstance(recall, list) and len(recall) > 1 else None
        except Exception as e:
            print(f"Error reading end model metrics: {e}")
            metrics['end_model_accuracy'] = None
            metrics['end_model_accuracy_std'] = None
            metrics['end_model_precision'] = None
            metrics['end_model_precision_std'] = None
            metrics['end_model_recall'] = None
            metrics['end_model_recall_std'] = None
    else:
        metrics['end_model_accuracy'] = None
        metrics['end_model_accuracy_std'] = None
        metrics['end_model_precision'] = None
        metrics['end_model_precision_std'] = None
        metrics['end_model_recall'] = None
        metrics['end_model_recall_std'] = None
    
    # Extract self-trained end model metrics if available and self-training wasn´t skipped
    if skipped_self_training == False and os.path.exists(self_trained_path):
        try:
            with open(self_trained_path, 'r') as f:
                data = json.load(f)
                accuracy = data.get('Accuracy (mean, std)', [None, None])
                precision = data.get('Precision (mean, std)', [None, None])
                recall = data.get('Recall (mean, std)', [None, None])
                
                # Extract both mean and std values
                metrics['self_trained_accuracy'] = accuracy[0] if isinstance(accuracy, list) and len(accuracy) > 0 else None
                metrics['self_trained_accuracy_std'] = accuracy[1] if isinstance(accuracy, list) and len(accuracy) > 1 else None
                
                metrics['self_trained_precision'] = precision[0] if isinstance(precision, list) and len(precision) > 0 else None
                metrics['self_trained_precision_std'] = precision[1] if isinstance(precision, list) and len(precision) > 1 else None
                
                metrics['self_trained_recall'] = recall[0] if isinstance(recall, list) and len(recall) > 0 else None
                metrics['self_trained_recall_std'] = recall[1] if isinstance(recall, list) and len(recall) > 1 else None
        except Exception as e:
            print(f"Error reading self-trained model metrics: {e}")
            metrics['self_trained_accuracy'] = None
            metrics['self_trained_accuracy_std'] = None
            metrics['self_trained_precision'] = None
            metrics['self_trained_precision_std'] = None
            metrics['self_trained_recall'] = None
            metrics['self_trained_recall_std'] = None
    else:
        metrics['self_trained_accuracy'] = None
        metrics['self_trained_accuracy_std'] = None
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

def get_default_parameters(dataset):
    """Get the default parameters for a dataset from its config file"""
    config_path = f"../config_files/config_{dataset}.yml"
    config = utils.Parser(config_file_path=config_path).parse()
    return {
        'topk': config['topk'],
        'end_model_lr': config['end_model_lr'],
        'end_model_batch_size': config['end_model_batch_size'],
        'label_model': config['label_model']
    }

def run_all_experiments(use_wandb=False, keep_configs=False):
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
                'dataset', 'log_file', 'learning_rate', 'batch_size', 'label_model', 'labeling_functions',
                'label_model_accuracy', 
                'end_model_accuracy', 'end_model_accuracy_std',
                'end_model_precision', 'end_model_precision_std',
                'end_model_recall', 'end_model_recall_std',
                'self_trained_end_model_accuracy', 'self_trained_end_model_accuracy_std',
                'self_trained_end_model_precision', 'self_trained_end_model_precision_std',
                'self_trained_end_model_recall', 'self_trained_end_model_recall_std'
            ])
    else:
        results = pd.DataFrame(columns=[
            'dataset', 'log_file', 'learning_rate', 'batch_size', 'label_model', 'labeling_functions',
            'label_model_accuracy', 
            'end_model_accuracy', 'end_model_accuracy_std',
            'end_model_precision', 'end_model_precision_std',
            'end_model_recall', 'end_model_recall_std',
            'self_trained_end_model_accuracy', 'self_trained_end_model_accuracy_std',
            'self_trained_end_model_precision', 'self_trained_end_model_precision_std',
            'self_trained_end_model_recall', 'self_trained_end_model_recall_std'
        ])
    
    print(f'\n Current results file content:')
    print(results)

    # Run all experiments
    for dataset in DATASETS:
        # Get parameters for this dataset
        default_params = get_default_parameters(dataset)
        
        # Get available parameter options for this dataset
        lr_options = LEARNING_RATES.get(dataset, [default_params['end_model_lr']])
        batch_size_options = BATCH_SIZES.get(dataset, [default_params['end_model_batch_size']])
        label_model_options = LABEL_MODELS.get(dataset, [default_params['label_model']])
        lf_options = LABELING_FUNCTIONS.get(dataset, [default_params['topk']])
        
        for lr in lr_options:
            for batch_size in batch_size_options:
                for label_model in label_model_options:
                    for lf_count in lf_options:
                        # Create a unique experiment identifier for display
                        exp_id = f"{dataset}_{lr}_{batch_size}_{label_model}_{lf_count}"

                        # Default behavior is to also execute the self-training run
                        skip_self_training = False
                        
                        # Check if this experiment has already been run by checking the DataFrame
                        experiment_mask = ((results['dataset'] == dataset) & 
                                           (results['learning_rate'] == float(lr)) &  # Compare as float
                                           (results['batch_size'] == batch_size) & 
                                           (results['label_model'] == label_model) &
                                           (results['labeling_functions'] == lf_count))
                        
                        # Skip if already run
                        if experiment_mask.any():
                            print(f"Skipping already completed experiment: {exp_id}")
                            continue

                        # Skip self-training for amazon dataset with majority_vote due to hardware/time constraints
                        if dataset == 'amazon' and label_model == 'majority_vote':
                            skip_self_training = True 
                        
                        print(f"\n=== Running experiment: dataset={dataset}, lr={lr}, batch_size={batch_size}, label_model={label_model}, lf_count={lf_count} ===\n")
                        if skip_self_training:
                            print(f"\n=== Experiment will SKIP self-training ===\n")
                        
                        # Run experiment
                        experiment_info = run_experiment(dataset, lr, batch_size, label_model, lf_count, use_wandb, keep_configs, skip_self_training)
                        
                        if experiment_info:
                            # Extract metrics
                            metrics = extract_metrics(experiment_info['results_folder'], skip_self_training)
                            
                            # Add to results DataFrame
                            result = pd.DataFrame([{
                                'dataset': dataset,
                                'log_file': experiment_info['log_file'],
                                'learning_rate': float(lr),  # Store as float
                                'batch_size': batch_size,
                                'label_model': label_model,
                                'labeling_functions': lf_count,
                                'label_model_accuracy': metrics.get('label_model_accuracy', None),
                                'end_model_accuracy': metrics.get('end_model_accuracy', None),
                                'end_model_accuracy_std': metrics.get('end_model_accuracy_std', None),
                                'end_model_precision': metrics.get('end_model_precision', None),
                                'end_model_precision_std': metrics.get('end_model_precision_std', None),
                                'end_model_recall': metrics.get('end_model_recall', None),
                                'end_model_recall_std': metrics.get('end_model_recall_std', None),
                                'self_trained_end_model_accuracy': metrics.get('self_trained_accuracy', None),
                                'self_trained_end_model_accuracy_std': metrics.get('self_trained_accuracy_std', None),
                                'self_trained_end_model_precision': metrics.get('self_trained_precision', None),
                                'self_trained_end_model_precision_std': metrics.get('self_trained_precision_std', None),
                                'self_trained_end_model_recall': metrics.get('self_trained_recall', None),
                                'self_trained_end_model_recall_std': metrics.get('self_trained_recall_std', None)
                            }])
                            
                            results = pd.concat([results, result], ignore_index=True)
                            
                            # Save intermediate results
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            results_csv_path = f"../results/experiment_results_2_{timestamp}.csv"
                            results.to_csv(results_csv_path, index=False)
                            
                            print(f"Updated results table. Current shape: {results.shape}")
                            
                            # Upload experiment files to Dropbox
                            upload_experiment_files(experiment_info, results_csv_path)
                        else:
                            print(f"Experiment failed or could not find results folder for {exp_id}. Moving to next experiment.")
        
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results_path = f"../results/experiment_results_2_final_{timestamp}.csv"
    results.to_csv(final_results_path, index=False)
    print(f"Saved final results to {final_results_path}")
    
    # Upload final results to Dropbox
    result = upload_to_dropbox(path=final_results_path, dest_path='/results_csv', is_file=True)
    if result != 0:
        print(f"ERROR: Failed to upload final results CSV to Dropbox (error code {result})")
    else:
        print("Successfully uploaded final results CSV to Dropbox")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run KeyClass experiments with different hyperparameters')
    parser.add_argument('--use_wandb', action='store_true', default=False, help='Use Weights & Biases for experiment tracking')
    parser.add_argument('--keep_configs', action='store_true', default=False, help='Keep temporary config files after experiments')
    args = parser.parse_args()
    
    # Make sure the results directory exists
    os.makedirs("../results", exist_ok=True)
    
    print(f"Running all experiments with use_wandb={args.use_wandb}, keep_configs={args.keep_configs}")
    results = run_all_experiments(use_wandb=args.use_wandb, keep_configs=args.keep_configs)
    
    # Display summary
    print("\n=== Experiment Results Summary ===")
    print(results)