#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced ICD-9 Category Analysis Script with Missing Categories Check

This script analyzes the distribution of top-level ICD-9 categories in the MIMIC-3 dataset,
calculating the number of records, patients, and admissions for each category.
It also checks for records that don't have any assigned top-level categories.
"""

import pandas as pd
import numpy as np
import csv
import os
from collections import Counter, defaultdict

def load_icd9_category_names():
    """Load ICD-9 category names from the labels file or define them directly."""
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
    
    # Create mapping from category ID to name
    cat_id_to_name = {f"cat:{i+1}": name for i, name in enumerate(labels)}
    return cat_id_to_name

def analyze_categories(data_train_path, data_valid_path, code_idx=8, output_path="category_analysis.csv"):
    """
    Analyze the distribution of ICD-9 categories across records, patients, and admissions.
    
    Args:
        data_train_path (str): Path to training data CSV
        data_valid_path (str): Path to validation data CSV
        code_idx (int): Column index for ICD codes
        output_path (str): Path to save analysis results
    """
    print("Loading category names...")
    cat_id_to_name = load_icd9_category_names()
    
    # Initialize counters
    record_counts = Counter()
    patient_counts = defaultdict(set)
    admission_counts = defaultdict(set)
    
    # Track overall dataset statistics
    total_records = 0
    all_patient_ids = set()
    all_admission_ids = set()
    
    # Track records with no assigned categories
    records_without_categories = []
    
    # Process both training and validation data
    for data_path in [data_train_path, data_valid_path]:
        print(f"Processing {data_path}...")
        
        with open(data_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"', skipinitialspace=True)
            header = next(csvreader)
            
            # Find column indices
            subject_id_idx = header.index("SUBJECT_ID") if "SUBJECT_ID" in header else 2
            hadm_id_idx = header.index("HADM_ID") if "HADM_ID" in header else 1
            
            for row_index, row in enumerate(csvreader, start=1):
                if len(row) <= max(code_idx, subject_id_idx, hadm_id_idx):
                    continue  # Skip rows with insufficient columns
                
                # Count total records
                total_records += 1
                
                # Get patient and admission IDs
                subject_id = row[subject_id_idx]
                hadm_id = row[hadm_id_idx]
                
                # Add to overall sets
                all_patient_ids.add(subject_id)
                all_admission_ids.add(hadm_id)
                
                # Check for records with no categories
                has_categories = False
                
                # Get ICD-9 categories
                if row[code_idx]:
                    categories = row[code_idx].split('-')
                    
                    for category in categories:
                        if category.startswith("cat:"):
                            has_categories = True
                            # Update record count
                            record_counts[category] += 1
                            
                            # Update patient count
                            patient_counts[category].add(subject_id)
                            
                            # Update admission count
                            admission_counts[category].add(hadm_id)
                
                # If no categories found, add to the list of records without categories
                if not has_categories:
                    file_name = os.path.basename(data_path)
                    records_without_categories.append({
                        'File': file_name,
                        'Row': row_index,
                        'SUBJECT_ID': subject_id,
                        'HADM_ID': hadm_id
                    })
    
    # Print overall dataset statistics
    print("\n============== OVERALL DATASET STATISTICS ==============")
    print(f"Total number of records: {total_records}")
    print(f"Total number of unique patients (SUBJECT_ID): {len(all_patient_ids)}")
    print(f"Total number of unique admissions (HADM_ID): {len(all_admission_ids)}")
    print("=======================================================")
    
    # Report on records without categories
    print("\n============== RECORDS WITHOUT CATEGORIES ==============")
    if records_without_categories:
        print(f"Found {len(records_without_categories)} records without any assigned top-level ICD-9 categories")
        
        # Save details of these records to a file
        pd.DataFrame(records_without_categories).to_csv('records_without_categories.csv', index=False)
        print("Details saved to records_without_categories.csv")
    else:
        print("All records have at least one assigned top-level ICD-9 category")
    print("=======================================================")
    
    # Prepare results for output
    results = []
    
    # Create a custom sort key function that extracts the numeric part of "cat:X"
    def extract_category_number(category):
        return int(category.split(':')[1])
    
    # Sort categories numerically based on their numbers
    for category in sorted(record_counts.keys(), key=extract_category_number):
        category_name = cat_id_to_name.get(category, category)
        results.append({
            'Category ID': category,
            'Category Name': category_name,
            'Record Count': record_counts[category],
            'Patient Count': len(patient_counts[category]),
            'Admission Count': len(admission_counts[category]),
            'Record Percentage': round(record_counts[category] / total_records * 100, 2),
            'Patient Percentage': round(len(patient_counts[category]) / len(all_patient_ids) * 100, 2),
            'Admission Percentage': round(len(admission_counts[category]) / len(all_admission_ids) * 100, 2)
        })
    # Append any records without categories
    if len(records_without_categories) > 0:

        # Count unique SUBJECT_IDs
        unique_subject_ids_without_cat = {record['SUBJECT_ID'] for record in records_without_categories}
        count_unique_subject_ids_without_cat = len(unique_subject_ids_without_cat)

        # Count unique HADM_IDs
        unique_hadm_ids_without_cat = {record['HADM_ID'] for record in records_without_categories}
        count_unique_hadm_ids_without_cat = len(unique_hadm_ids_without_cat)

        results.append({
            'Category ID': "N/A",
            'Category Name': "Records Without Assigned Categories",
            'Record Count': len(records_without_categories),
            'Patient Count': len(unique_subject_ids_without_cat),
            'Admission Count': len(unique_hadm_ids_without_cat),
            'Record Percentage': round(len(records_without_categories) / total_records * 100, 2),
            'Patient Percentage': round(len(unique_subject_ids_without_cat) / len(all_patient_ids) * 100, 2),
            'Admission Percentage': round(len(unique_hadm_ids_without_cat) / len(all_admission_ids) * 100, 2)
        })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    
    # Add overall dataset statistics to a separate file
    overall_stats = {
        'Metric': ['Total Records', 'Unique Patients', 'Unique Admissions', 'Records Without Categories'],
        'Count': [total_records, len(all_patient_ids), len(all_admission_ids), len(records_without_categories)],
        'Percentage': [100.0, 
                      len(all_patient_ids) / total_records * 100,
                      len(all_admission_ids) / total_records * 100,
                      len(records_without_categories) / total_records * 100]
    }
    pd.DataFrame(overall_stats).to_csv('overall_dataset_statistics.csv', index=False)
    
    # Display results
    print("\nICD-9 Category Analysis Results:")
    print("=" * 120)
    print(f"{'Category ID':<10} {'Category Name':<40} {'Records':<10} {'%':<6} {'Patients':<10} {'%':<6} {'Admissions':<10} {'%':<6}")
    print("-" * 120)
    
    for row in results:
        print(f"{row['Category ID']:<10} {row['Category Name']:<40} {row['Record Count']:<10} "
              f"{row['Record Percentage']:>5.2f}% {row['Patient Count']:<10} "
              f"{row['Patient Percentage']:>5.2f}% {row['Admission Count']:<10} "
              f"{row['Admission Percentage']:>5.2f}%")
    
    print("\nAnalysis saved to", output_path)
    print("Overall statistics saved to overall_dataset_statistics.csv")
    
    return df, {'total_records': total_records, 
                'unique_patients': len(all_patient_ids), 
                'unique_admissions': len(all_admission_ids),
                'records_without_categories': len(records_without_categories)}

def main():
    # Set paths
    data_train_path = "intermediate_files/icd9NotesDataTable_train.csv"
    data_valid_path = "intermediate_files/icd9NotesDataTable_test.csv"
    output_path = "stats/icd9_category_analysis.csv"
    
    # Check if paths exist
    if not os.path.exists(data_train_path):
        print(f"Warning: {data_train_path} not found. Update path to your MIMIC training data.")
        return
    
    if not os.path.exists(data_valid_path):
        print(f"Warning: {data_valid_path} not found. Update path to your MIMIC validation data.")
        return
    
    # Run analysis
    df, overall_stats = analyze_categories(
        data_train_path=data_train_path,
        data_valid_path=data_valid_path,
        code_idx=8,  # Based on the column containing top-level ICD codes
        output_path=output_path
    )
    
    # # Create visualizations
    # try:
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
        
    #     # Set plot style
    #     plt.style.use('ggplot')
    #     sns.set(font_scale=1.2)
        
    #     # Create figure with four subplots (including overall stats)
    #     fig, axes = plt.subplots(4, 1, figsize=(14, 24))
        
    #     # Plot overall statistics
    #     overall_df = pd.DataFrame({
    #         'Metric': ['Total Records', 'Unique Patients', 'Unique Admissions', 'Records Without Categories'],
    #         'Count': [overall_stats['total_records'], 
    #                  overall_stats['unique_patients'], 
    #                  overall_stats['unique_admissions'],
    #                  overall_stats['records_without_categories']]
    #     })
        
    #     sns.barplot(x='Count', y='Metric', data=overall_df, ax=axes[0], palette='Blues_d')
    #     axes[0].set_title('Overall Dataset Statistics', fontsize=16)
    #     axes[0].set_xlabel('Count')
    #     axes[0].set_ylabel('')
        
    #     for i, v in enumerate(overall_df['Count']):
    #         axes[0].text(v + 0.1, i, f"{v:,}", va='center')
        
    #     # For visualization, create a custom order based on Category ID numeric value
    #     df['Category Number'] = df['Category ID'].apply(lambda x: int(x.split(':')[1]))
        
    #     # Plot record counts (in category order)
    #     ordered_df = df.sort_values('Category Number')
    #     sns.barplot(x='Record Count', y='Category Name', data=ordered_df, 
    #                ax=axes[1], palette='viridis')
    #     axes[1].set_title('Number of Records per ICD-9 Category')
    #     axes[1].set_xlabel('Record Count')
    #     axes[1].set_ylabel('Category')
        
    #     # Plot patient counts (in category order)
    #     sns.barplot(x='Patient Count', y='Category Name', data=ordered_df, 
    #                ax=axes[2], palette='magma')
    #     axes[2].set_title('Number of Patients per ICD-9 Category')
    #     axes[2].set_xlabel('Patient Count')
    #     axes[2].set_ylabel('Category')
        
    #     # Plot admission counts (in category order)
    #     sns.barplot(x='Admission Count', y='Category Name', data=ordered_df, 
    #                ax=axes[3], palette='plasma')
    #     axes[3].set_title('Number of Admissions per ICD-9 Category')
    #     axes[3].set_xlabel('Admission Count')
    #     axes[3].set_ylabel('Category')
        
    #     plt.tight_layout()
    #     plt.savefig('stats/icd9_category_analysis.png', dpi=300, bbox_inches='tight')
    #     print("Visualization saved to stats/icd9_category_analysis.png")
        
    #     # Create an additional pie chart for records with/without categories
    #     plt.figure(figsize=(10, 8))
    #     labels = ['Records with categories', 'Records without categories']
    #     sizes = [overall_stats['total_records'] - overall_stats['records_without_categories'], 
    #             overall_stats['records_without_categories']]
    #     colors = ['#66b3ff', '#ff9999']
        
    #     plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, shadow=True)
    #     plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    #     plt.title('Proportion of Records With/Without ICD-9 Categories', fontsize=16)
    #     plt.tight_layout()
    #     plt.savefig('stats/records_with_without_categories.png', dpi=300)
    #     print("Category presence visualization saved to stats/records_with_without_categories.png")
        
    # except ImportError:
    #     print("Matplotlib and/or seaborn not installed. Skipping visualization.")

if __name__ == "__main__":
    main()