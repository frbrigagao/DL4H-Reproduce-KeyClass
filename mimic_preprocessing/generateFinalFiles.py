import pandas as pd
import numpy as np

def process_data(input_file, text_output_file, labels_output_file):
    """
    Processes a CSV file, extracts text and generates one-hot encoded labels
    based on TopLevelICD. Collapses multi-line text into a single line.

    Args:
        input_file (str): Path to the input CSV file (e.g., train.csv or valid.csv).
        text_output_file (str): Path to the output text file containing the TEXT field.
        labels_output_file (str): Path to the output file containing one-hot encoded labels.
    """

    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(input_file)

        # Initialize lists to store text and labels
        text_data = []
        label_data = []

        # Define the 19 ICD-9 top-level categories (consistent with your original R script)
        icd9_categories = [f"cat:{i}" for i in range(1, 21)]

        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            # Extract text from the TEXT column and collapse into a single line
            text = row['TEXT'].replace('\n', ' ').replace('\r', ' ')  # Replace newlines and carriage returns with spaces
            text_data.append(text)

            # Create one-hot encoded labels based on TopLevelICD
            top_level_icds = str(row['TopLevelICD']).split('-') # Handle NaNs by converting to string
            labels = [1 if category in top_level_icds else 0 for category in icd9_categories]
            label_data.append(labels)

        # Write text data to the text output file
        with open(text_output_file, 'w', encoding='utf-8') as f:  # Specify encoding
            for text in text_data:
                f.write(text + '\n')  # Write each text on a new line

        # Write label data to the labels output file
        with open(labels_output_file, 'w', encoding='utf-8') as f:  # Specify encoding
            for labels in label_data:
                f.write(' '.join(map(str, labels)) + '\n')  # Write comma-separated labels

        print(f"Successfully processed {input_file}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except KeyError as e:
        print(f"Error: Column '{e}' not found in input file '{input_file}'.")
    except Exception as e:
        print(f"An error occurred while processing '{input_file}': {e}")


# Define the input and output file paths for train data
train_input_file = 'data/p_icd9NotesDataTable_train.csv'
train_text_output_file = 'data/train.txt'
train_labels_output_file = 'data/train_labels.txt'

# Process the train data
process_data(train_input_file, train_text_output_file, train_labels_output_file)

# Define the input and output file paths for validation data
valid_input_file = 'data/p_icd9NotesDataTable_valid.csv'
valid_text_output_file = 'data/valid.txt'
valid_labels_output_file = 'data/valid_labels.txt'

# Process the validation data
process_data(valid_input_file, valid_text_output_file, valid_labels_output_file)

print("Text and labels files generated successfully.")