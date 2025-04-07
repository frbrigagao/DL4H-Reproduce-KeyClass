import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

# Load Data
#notes = pd.read_csv('data/NOTEEVENTS.csv', na_filter=False, dtype={'HADM_ID': int})


# First, read HADM_ID as string to identify problematic rows
notes = pd.read_csv('data/NOTEEVENTS.csv', na_filter=False, dtype={'HADM_ID': str})

# Identify records where HADM_ID is missing (empty string)
# missing_hadm_id = notes[notes['HADM_ID'] == '']

# if not missing_hadm_id.empty:
#     print("Warning: Found records with missing HADM_ID:")
#     print(missing_hadm_id)
# else:
#     print("No records found with missing HADM_ID. Proceeding as normal.")


# Filter Notes *before* checking for non-numeric HADM_ID
notes = notes[notes['CATEGORY'] == 'Discharge summary']


# Identify records where HADM_ID is not a number (i.e., contains a string) AFTER filtering
non_numeric_hadm_id = notes[~notes['HADM_ID'].str.isnumeric()]

if not non_numeric_hadm_id.empty:
    print("Warning: Found records with non-numeric HADM_ID (after filtering):")
    print(non_numeric_hadm_id)
else:
    print("No records found with non-numeric HADM_ID after filtering.")

diagnoses = pd.read_csv('data/DIAGNOSES_ICD.csv', dtype={'HADM_ID': int})

# Data Cleaning and Preprocessing for notes
notes = notes[[
    'ROW_ID',
    'SUBJECT_ID',
    'HADM_ID',
    'CHARTDATE',
    'CHARTTIME',
    'CATEGORY',
    'DESCRIPTION',
    'TEXT',
]]

# Handle empty CHARTTIME values
notes['CHARTTIME'] = notes['CHARTTIME'].apply(
    lambda x: None if x == '' else x
)  # Replacing empty strings with None

# Convert HADM_ID to integer
notes['HADM_ID'] = notes['HADM_ID'].astype('Int64')

# Collapse ICD9 codes
diagnoses = diagnoses[['HADM_ID', 'SUBJECT_ID', 'ICD9_CODE']]
diagnoses = (
    diagnoses.groupby(['HADM_ID', 'SUBJECT_ID'])['ICD9_CODE']
    .apply(lambda x: '-'.join(x.astype(str)))
    .reset_index()
)

# Filter Notes
notes = notes[notes['CATEGORY'] == 'Discharge summary']
notes = notes[['HADM_ID', 'CHARTDATE', 'DESCRIPTION', 'TEXT']]
notes['CHARTDATE'] = pd.to_datetime(notes['CHARTDATE'])

notes = notes.sort_values('CHARTDATE').groupby('HADM_ID').first().reset_index()

# Create note and ICD9 code matrix
icd9NotesDataTable = pd.merge(
    diagnoses, notes, on='HADM_ID', how='right'
)  # Right Join

# Extra filtering
icd9NotesDataTable = icd9NotesDataTable[icd9NotesDataTable['ICD9_CODE'] != '']


# Add ICD-9 Level 2 and Top Level Categories
def process_icd9(icd9_code):
    level2_list = []
    top_list = []
    if isinstance(icd9_code, str):
        icd9_row = icd9_code.split('-')
        for icd9 in icd9_row:
            if not icd9:
                continue

            if icd9.startswith('E'):
                icd_level2 = icd9[:4]
                icd9_top = 'cat:19'
            elif icd9.startswith('V'):
                icd_level2 = icd9[:3]
                icd9_top = 'cat:20'
            else:
                try:
                    icd9_num = int(float(icd9[:3]))
                    icd_level2 = icd9[:3]

                    if 1 <= icd9_num <= 139:     # Infectious and parasitic diseases
                        icd9_top = 'cat:1'
                    elif 140 <= icd9_num <= 239: # Neoplasms
                        icd9_top = 'cat:2'
                    elif 240 <= icd9_num <= 279: # Endocrine, nutritional and metabolic diseases, and immunity disorders
                        icd9_top = 'cat:3'
                    elif 280 <= icd9_num <= 289: # Diseases of blood and blood-forming organs
                        icd9_top = 'cat:4'
                    elif 290 <= icd9_num <= 319: # Mental disorders
                        icd9_top = 'cat:5'
                    elif 320 <= icd9_num <= 359: # Diseases of the nervous system
                        icd9_top = 'cat:6'
                    elif 360 <= icd9_num <= 389: # Diseases of sense organs
                        icd9_top = 'cat:7'
                    elif 390 <= icd9_num <= 459: # Diseases of the circulatory system
                        icd9_top = 'cat:8'
                    elif 460 <= icd9_num <= 519: # Diseases of the respiratory system
                        icd9_top = 'cat:9'
                    elif 520 <= icd9_num <= 579: # Diseases of the digestive system
                        icd9_top = 'cat:10'
                    elif 580 <= icd9_num <= 629: # Diseases of the genitourinary system
                        icd9_top = 'cat:11'
                    elif 630 <= icd9_num <= 679: # Complications of pregnancy, childbirth, and the puerperium
                        icd9_top = 'cat:12'
                    elif 680 <= icd9_num <= 709: # Diseases of the skin and subcutaneous tissue
                        icd9_top = 'cat:13'
                    elif 710 <= icd9_num <= 739: # Diseases of the musculoskeletal system and connective tissue
                        icd9_top = 'cat:14'
                    elif 740 <= icd9_num <= 759: # Congenital anomalies
                        icd9_top = 'cat:15'
                    elif 760 <= icd9_num <= 779: # Certain conditions originating in the perinatal period
                        icd9_top = 'cat:16'
                    elif 780 <= icd9_num <= 799: # Symptoms, signs, and ill-defined conditions
                        icd9_top = 'cat:17'
                    elif 800 <= icd9_num <= 999: # Injury and poisoning
                        icd9_top = 'cat:18'
                    else:
                        icd9_top = None
                except ValueError:
                    icd9_top = None
                    icd_level2 = None
            top_list.append(icd9_top)
            level2_list.append(icd_level2)

    return (
        '-'.join(sorted(list(set(filter(None, level2_list))))),
        '-'.join(sorted(list(set(filter(None, top_list))))),
    )  # Remove None values


icd9NotesDataTable['Level2ICD'], icd9NotesDataTable['TopLevelICD'] = zip(
    *icd9NotesDataTable['ICD9_CODE'].apply(process_icd9)
)

# Check for empty values in TopLevelICD and Level2ICD and remove those records
empty_toplevel = icd9NotesDataTable['TopLevelICD'] == ''
empty_level2 = icd9NotesDataTable['Level2ICD'] == ''

# Combine the conditions to remove rows where either TopLevelICD or Level2ICD is empty
icd9NotesDataTable = icd9NotesDataTable[~(empty_toplevel | empty_level2)]

# Print the number of remaining records
print(f"Number of records remaining after filtering empty ICD9 categories: {len(icd9NotesDataTable)}")

# Split into Training and Validation
trainingFrac = 0.70 #0.75
icd9NotesDataTable_train, icd9NotesDataTable_valid = train_test_split(
    icd9NotesDataTable, train_size=trainingFrac, random_state=42
)  # Added random_state for reproducibility

# Write to file
icd9NotesDataTable.to_csv('data/p_icd9NotesDataTable.csv', index=False)
icd9NotesDataTable_train.to_csv('data/p_icd9NotesDataTable_train.csv', index=False)
icd9NotesDataTable_valid.to_csv('data/p_icd9NotesDataTable_valid.csv', index=False)