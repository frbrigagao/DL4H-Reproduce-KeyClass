import re
import sys
import argparse
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract frequent keywords from ICD-9 descriptions by category.')
    parser.add_argument('input_file', help='Path to the ICD-9 description file')
    parser.add_argument('--output_file', help='Path to save output file (if not specified, prints to console)')
    parser.add_argument('--top_n', type=int, default=15, help='Number of top keywords to return per category')
    parser.add_argument('--min_freq', type=float, default=0.001, help='Minimum document frequency for keywords')
    return parser.parse_args()

def read_icd9_descriptions(file_path):
    """Read ICD-9 code descriptions from file, trying multiple encodings."""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            descriptions = {}
            with open(file_path, 'r', encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Extract code and description
                    code = line[:5].strip()
                    description = line[5:].strip()
                    descriptions[code] = description
            
            # If we get here without an exception, the encoding worked
            print(f"Successfully read file with {encoding} encoding")
            return descriptions
            
        except UnicodeDecodeError:
            print(f"Failed to read with {encoding} encoding, trying next...")
            continue
    
    # If we get here, none of the encodings worked
    raise ValueError(f"Could not read file {file_path} with any of the attempted encodings")

def categorize_icd9_code(code):
    """Assign the ICD-9 code to its top-level category (1-19) based on the R script logic."""
    if code.startswith('E'):
        return 18  # cat:18
    elif code.startswith('V'):
        return 19  # cat:19
    else:
        try:
            numeric_code = int(code[:3])
            
            if 1 <= numeric_code <= 139:
                return 1  # cat:1 - Infectious & parasitic diseases
            elif 140 <= numeric_code <= 239:
                return 2  # cat:2 - Neoplasms
            elif 240 <= numeric_code <= 279:
                return 3  # cat:3 - Endocrine, nutritional and metabolic diseases
            elif 280 <= numeric_code <= 289:
                return 4  # cat:4 - Blood & blood-forming organs
            elif 290 <= numeric_code <= 319:
                return 5  # cat:5 - Mental disorders
            elif 320 <= numeric_code <= 359:
                return 6  # cat:6 - Nervous system
            elif 360 <= numeric_code <= 389:
                return 7  # cat:7 - Sense organs
            elif 390 <= numeric_code <= 459:
                return 8  # cat:8 - Circulatory system
            elif 460 <= numeric_code <= 519:
                return 9  # cat:9 - Respiratory system
            elif 520 <= numeric_code <= 579:
                return 10  # cat:10 - Digestive system
            elif 580 <= numeric_code <= 629:
                return 11  # cat:11 - Genitourinary system
            elif 630 <= numeric_code <= 679:
                return 12  # cat:12 - Pregnancy & childbirth complications
            elif 680 <= numeric_code <= 709:
                return 13  # cat:13 - Skin & subcutaneous tissue
            elif 710 <= numeric_code <= 739:
                return 14  # cat:14 - Musculoskeletal system & connective tissue
            elif 740 <= numeric_code <= 759:
                return 15  # cat:15 - Congenital anomalies
            elif 760 <= numeric_code <= 779:
                return 16  # cat:16 - Perinatal period conditions
            elif 800 <= numeric_code <= 999:
                return 17  # cat:17 - Injury and poisoning
        except ValueError:
            # If code cannot be parsed as a number
            return None
    
    return None

def extract_keywords(descriptions):
    """Extract keywords from descriptions by category."""
    category_descriptions = {i: [] for i in range(1, 20)}  # Categories 1-19
    
    # Group descriptions by category
    for code, description in descriptions.items():
        category = categorize_icd9_code(code)
        if category:
            category_descriptions[category].append(description)
    
    # Get stop words
    stop_words = set(stopwords.words('english'))
    
    # Extract keywords for each category
    category_keywords = {}
    for category, desc_list in category_descriptions.items():
        # Combine all descriptions for this category
        combined_text = ' '.join(desc_list).lower()
        
        # Tokenize and clean words
        words = re.findall(r'\b[a-z]+\b', combined_text)
        
        # Remove stop words
        filtered_words = [word for word in words if word not in stop_words]
        
        # Count word frequencies
        word_counts = Counter(filtered_words)
        
        # Store in the dictionary
        category_keywords[category] = word_counts
    
    return category_keywords

def main():
    args = parse_arguments()
    
    # Read ICD-9 descriptions
    descriptions = read_icd9_descriptions(args.input_file)
    
    # Extract keywords by category
    category_keywords = extract_keywords(descriptions)
    
    # Create a list to store output lines
    output_lines = []
    
    # Generate output lines for each category
    for category in range(1, 20):
        if category in category_keywords:
            # Format the category number to be zero-padded
            category_idx = str(category - 1).zfill(2)
            
            # Get the most common words for this category
            most_common = category_keywords[category].most_common(args.top_n)
            
            # Format the output with commas and spaces between words
            keywords = ", ".join([word for word, count in most_common])
            output_lines.append(f"target_{category_idx}: {keywords}")
    
    # Output the results
    if args.output_file:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for line in output_lines:
                    f.write(line + '\n')
            print(f"Results saved to {args.output_file}")
        except Exception as e:
            print(f"Error writing to output file: {e}")
            # Fall back to console output
            for line in output_lines:
                print(line)
    else:
        # Print to console
        for line in output_lines:
            print(line)

if __name__ == "__main__":
    main()