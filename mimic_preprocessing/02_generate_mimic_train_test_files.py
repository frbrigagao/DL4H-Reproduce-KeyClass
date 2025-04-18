#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIMIC-3 Dataset Preprocessing Script

This script processes the MIMIC-3 dataset, extracting clinical notes and their
associated ICD-9 codes, and saves them to disk in the requested format.
"""

import pandas as pd
import numpy as np
import os
import csv
from collections import Counter

def read_clinical_note(path, code_idx, text_idx):
    """
    Reads in clinical notes and returns tokens and ICD9 codes.
    
    Args:
        path (str): Path to the clinical note CSV
        code_idx (int): Column index for ICD codes
        text_idx (int): Column index for text
    
    Returns:
        list: List of tuples containing (tokens, ICD9 codes)
        list: List of unique ICD9 codes
    """
    ret = []
    icd_code_list = []
    
    with open(path, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"', skipinitialspace=True)
        # Skip header row
        next(csvreader)
        for row in csvreader:
            # Split text into tokens and ICD9 codes by dash
            ret.append((row[text_idx].split(), row[code_idx].split('-')))
            icd_code_list.extend(row[code_idx].split('-'))
    
    return ret, list(set(icd_code_list))

def process_dataset(data_train_path, data_test_path, output_dir, code_idx=9, text_idx=6, max_note_length=1000):
    """
    Process MIMIC-3 dataset and save preprocessed files.
    
    Args:
        data_train_path (str): Path to training data CSV
        data_test_path (str): Path to test data CSV
        output_dir (str): Directory to save processed files
        code_idx (int): Column index containing ICD-9 codes
        text_idx (int): Column index containing notes text
        max_note_length (int): Maximum note length to keep
    """
    print("Reading training data...")
    train_data, icd_code_list = read_clinical_note(data_train_path, code_idx, text_idx)
    
    print("Reading test data...")
    test_data, more_icd_codes = read_clinical_note(data_test_path, code_idx, text_idx)
    icd_code_list.extend(more_icd_codes)
    
    # Get unique top-level ICD9 categories
    top_level_categories = [code for code in icd_code_list if code.startswith('cat:')]
    top_level_categories = list(set(top_level_categories))
    
    # Sort categories numerically by extracting the number from 'cat:X'
    top_level_categories.sort(key=lambda x: int(x.split(':')[1]))
    
    print(f"Found {len(top_level_categories)} top-level ICD9 categories")
    
    # Create mapping from categories to indices
    # Categories should be mapped in numerical order (cat:1 -> 0, cat:2 -> 1, etc.)
    category_to_idx = {cat: i for i, cat in enumerate(top_level_categories)}

    # print(category_to_idx)
    # OUTPUT: {'cat:1': 0, 'cat:2': 1, 'cat:3': 2, 'cat:4': 3, 'cat:5': 4, 'cat:6': 5, 'cat:7': 6, 'cat:8': 7, 'cat:9': 8, 'cat:10': 9, 'cat:11': 10, 'cat:12': 11, 'cat:13': 12, 'cat:14': 13, 'cat:15': 14, 'cat:16': 15, 'cat:17': 16, 'cat:18': 17, 'cat:19': 18}
    
    # Prepare output files
    os.makedirs(output_dir, exist_ok=True)
    
    # Write labels file
    with open(os.path.join(output_dir, 'labels.txt'), 'w') as f:
        labels = [
            "Infectious & parasitic",
            "Neoplasms",
            "Endocrine, nutritional and metabolic",
            "Blood & blood-forming organs",
            "Mental disorders",
            "Nervous system",
            "Sense organs",
            "Circulatory system",
            "Respiratory system",
            "Digestive system",
            "Genitourinary system",
            "Pregnancy & childbirth complications",
            "Skin & subcutaneous tissue",
            "Musculoskeletal system & connective tissue",
            "Congenital anomalies",
            "Perinatal period conditions",
            "Injury and poisoning",
            "External causes of injury",
            "Supplementary"
        ]
        for label in labels:
            f.write(f"{label}\n")
    
    # Process training data
    print("Processing training data...")
    process_split(train_data, output_dir, 'train', category_to_idx, max_note_length)
    
    # Process test data
    print("Processing test data...")
    process_split(test_data, output_dir, 'test', category_to_idx, max_note_length)
    
    print("Dataset processing complete!")

def process_split(data, output_dir, split_name, category_to_idx, max_note_length):
    """
    Process a data split and save to disk.
    
    Args:
        data (list): List of (tokens, labels) tuples
        output_dir (str): Output directory
        split_name (str): Name of split (train/test)
        category_to_idx (dict): Mapping from category to index
        max_note_length (int): Maximum number of tokens to keep per note
    """
    text_file = open(os.path.join(output_dir, f'{split_name}.txt'), 'w')
    labels_file = open(os.path.join(output_dir, f'{split_name}_labels.txt'), 'w')
    
    num_categories = len(category_to_idx)
    
    num_records = 0

    for tokens, categories in data:
        # Truncate tokens to max_note_length
        tokens = tokens[:max_note_length]
        
        # Write tokens to text file
        text_file.write(' '.join(tokens) + '\n')
        
        # Create n-hot vector for categories
        n_hot = np.zeros(num_categories, dtype=int)
        for cat in categories:
            if cat.startswith('cat:') and cat in category_to_idx:
                n_hot[category_to_idx[cat]] = 1
        
        # Write n-hot vector to labels file
        # labels_file.write(' '.join(map(str, n_hot)) + '\n')

        labels_file.write(''.join(map(str, n_hot)) + '\n')

        num_records = num_records + 1
    
    text_file.close()
    labels_file.close()

    print(f"Wrote {num_records} records to {split_name}.txt and {split_name}_labels.txt")
          

if __name__ == "__main__":
    # Set paths
    data_train_path = "intermediate_files/icd9NotesDataTable_train.csv"
    data_test_path = "intermediate_files/icd9NotesDataTable_test.csv"
    output_dir = "output_mimic_files/mimic"
    
    # Check if paths exist
    if not os.path.exists(data_train_path):
        print(f"Warning: {data_train_path} not found. Update path to your MIMIC training data.")
    
    if not os.path.exists(data_test_path):
        print(f"Warning: {data_test_path} not found. Update path to your MIMIC test data.")
    
    # Process dataset
    process_dataset(
        data_train_path=data_train_path,
        data_test_path=data_test_path,
        output_dir=output_dir,
        code_idx=8,  # Column with the top-level ICD9 Codes
        text_idx=6,  # Column with the text of the report
        max_note_length=4000  # From Fastag's preprocessing steps
    )