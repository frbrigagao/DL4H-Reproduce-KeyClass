#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Script: 00_generate_icd9_descriptors.py
#
# Description:
#   Reads an ICD‑9 description file, groups descriptions by their top‑level
#   category, extracts the top N most frequent keywords per category (excluding
#   stopwords), optionally filters out terms common across many categories, and
#   writes out a “target_XX: keyword, keyword…” list for each of the 19 categories.
#
# Usage:
#   uv run 00_generate_icd9_descriptors.py \
#       [--input_file PATH] \
#       [--output_file PATH] \
#       [--num_keywords_per_cat N] \
#       [--shared_keyword_threshold PERCENT]
#
#   --input_file                Path to ICD‑9 description file. Default: icd9_descriptions/CMS32_DESC_LONG_DX.txt
#   --output_file               Where to save the resulting targets. Default: saves to target_icd9_descriptors.txt)
#   --num_keywords_per_cat      Number of top keywords per category. Default: 30
#   --shared_keyword_threshold  Remove cmmon keywords appearing in >PERCENT% of categories. Default: 0 (preservers all common keywords).
#
# Dependencies:
#   • Python 3.x
#   • nltk (with ‘stopwords’ corpus)
#   • re, argparse, collections
#
# Author:     Fabricio Brigagao

import re
import sys
import argparse
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract frequent keywords from ICD-9 descriptions by category.')
    parser.add_argument('--input_file', default='icd9_descriptions/CMS32_DESC_LONG_DX.txt', help='Path to the ICD-9 description file. Default is "icd9_descriptions/CMS32_DESC_LONG_DX.txt"')
    parser.add_argument('--output_file', default='target_icd9_descriptors.txt', help='Path to save output file. Default is "target_icd9_descriptors.txt"')
    parser.add_argument('--num_keywords_per_cat', type=int, default=30, help='Number of top keywords to return per category. Default is 30 keywords.')
    parser.add_argument('--shared_keyword_threshold', type=int, default=0, help='If set, remove common keywords appearing over the established percentage of categories. Default is 0 (does not remove any common keywords).')
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
                    # Handle potential variations in spacing/format
                    match = re.match(r'^(\S+)\s+(.*)$', line)
                    if match:
                        code = match.group(1).strip()
                        description = match.group(2).strip()
                        # Ensure code looks like an ICD9 code (e.g., starts with digit, E, or V)
                        if re.match(r'^[0-9EV]', code):
                             descriptions[code] = description
                        else:
                             print(f"Skipping line with potentially invalid code format: {line}", file=sys.stderr)
                    else:
                        print(f"Skipping line, could not parse code/description: {line}", file=sys.stderr)

            # If we get here without an exception, the encoding worked
            if descriptions: # Check if we actually read anything
                print(f"Successfully read file {file_path} with {encoding} encoding. Found {len(descriptions)} descriptions.")
                return descriptions
            else:
                print(f"Read file {file_path} with {encoding}, but found no valid descriptions. Trying next encoding...")
                continue

        except UnicodeDecodeError:
            print(f"Failed to read {file_path} with {encoding} encoding, trying next...", file=sys.stderr)
            continue
        except FileNotFoundError:
            print(f"Error: Input file not found at {file_path}", file=sys.stderr)
            sys.exit(1) # Exit if the file doesn't exist
        except Exception as e:
             print(f"An unexpected error occurred while reading {file_path} with {encoding}: {e}", file=sys.stderr)
             continue


    # If we get here, none of the encodings worked or the file was empty/unparseable
    raise ValueError(f"Could not read valid descriptions from file {file_path} with any attempted encoding.")


def categorize_icd9_code(code):
    """Assign the ICD-9 code to its top-level category (1-19) based on the R script logic."""
    if not isinstance(code, str): # Add check for code type
        return None

    code = code.strip().upper() # Normalize code

    if code.startswith('E'):
        return 18  # cat:18 - Injury and poisoning (External causes)
    elif code.startswith('V'):
        return 19  # cat:19 - Supplementary classification (V codes)
    else:
        # Standard numeric codes
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
                return 17  # cat:17 - Injury and poisoning (Main codes)
        except ValueError:
            # If code's numeric part cannot be parsed
            print(f"Warning: Could not parse numeric part of code: {code}", file=sys.stderr)

    return None # Return None if no category is matched


def extract_keywords(descriptions):
    """Extract keywords from descriptions by category."""
    category_descriptions = defaultdict(list) # Use defaultdict for cleaner code

    # Group descriptions by category
    categorized_count = 0
    uncategorized_count = 0
    for code, description in descriptions.items():
        category = categorize_icd9_code(code)
        if category is not None: # Check if categorization was successful
            category_descriptions[category].append(description)
            categorized_count += 1
        else:
            # print(f"Warning: Could not categorize code: {code}", file=sys.stderr)
            uncategorized_count += 1

    print(f"Processed {len(descriptions)} codes: {categorized_count} categorized, {uncategorized_count} uncategorized.")
    if categorized_count == 0:
        print("Warning: No codes were successfully categorized. Check input file format and categorization logic.", file=sys.stderr)
        return {} # Return empty dict if nothing was categorized


    # Get stop words
    stop_words = set(stopwords.words('english'))

    # Extract keywords for each category
    category_keywords = {}
    for category, desc_list in category_descriptions.items():
        if not desc_list: # Skip if a category ended up with no descriptions
             continue

        # Combine all descriptions for this category
        combined_text = ' '.join(desc_list).lower()

        # Tokenize and clean words
        # Removes punctuation, keeps only alphabetic words
        words = re.findall(r'\b[a-z]+\b', combined_text)

        # Remove stop words and short words (e.g., length 1 or 2)
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]

        # Count word frequencies
        word_counts = Counter(filtered_words)

        # Store in the dictionary if there are any counts
        if word_counts:
            category_keywords[category] = word_counts

    print(f"Extracted keywords for {len(category_keywords)} categories.")
    return category_keywords

def filter_common_keywords(category_top_words, num_categories, max_percentage):
    """
    Identifies and returns a set of keywords that appear in the top list
    of more than max_percentage of the categories.
    """
    if num_categories == 0:
        return set()

    # Count how many categories each keyword appears in (among the top N)
    keyword_category_count = Counter()
    for category, words in category_top_words.items():
        keyword_category_count.update(set(words)) # Use set to count each word once per category

    # Determine the threshold
    threshold = num_categories * max_percentage / 100

    # Identify keywords exceeding the threshold
    common_keywords_to_remove = {
        word for word, count in keyword_category_count.items() if count > threshold
    }

    print(f"Found {len(common_keywords_to_remove)} keywords appearing in > {max_percentage}% ({threshold:.1f}) of the {num_categories} active categories: {sorted(list(common_keywords_to_remove))}")
    return common_keywords_to_remove


def main():
    args = parse_arguments()

    # Read ICD-9 descriptions
    try:
        descriptions = read_icd9_descriptions(args.input_file)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during file reading: {e}", file=sys.stderr)
        sys.exit(1)


    if not descriptions:
         print("Error: No descriptions were read from the input file.", file=sys.stderr)
         sys.exit(1)

    # Extract keywords by category
    category_keywords = extract_keywords(descriptions)

    if not category_keywords:
        print("Warning: No keywords extracted. Check categorization and input data.", file=sys.stderr)
        sys.exit(1)

    # --- Filtering Logic ---
    words_to_remove = set()
    category_top_words = {} # Store the initial top N keywords per category {cat_id: [word1, word2,...]}

    # First pass: Get the top N keywords for each category
    for category in range(1, 20):
        if category in category_keywords:
            most_common_pairs = category_keywords[category].most_common(args.num_keywords_per_cat)
            if most_common_pairs: # Ensure there are words
                 category_top_words[category] = [word for word, count in most_common_pairs]

    num_active_categories = len(category_top_words)

    # Apply filtering if the flag is set and there are categories to compare
    if args.shared_keyword_threshold > 0 and num_active_categories > 0:
        words_to_remove = filter_common_keywords(category_top_words, num_active_categories, args.shared_keyword_threshold)

    # --- Generate Output ---
    output_lines = []
    # Generate output lines for each category (1 to 19)
    for category in range(1, 20):
         category_idx_str = str(category - 1).zfill(2) # 00 to 18

         # Get the original top N words for this category (if they exist)
         original_top_words = category_top_words.get(category, [])

         # Filter out the common words if removal is enabled
         if args.shared_keyword_threshold > 0:
             filtered_keywords = [word for word in original_top_words if word not in words_to_remove]
         else:
             filtered_keywords = original_top_words # Use original list if not filtering

         # Format the output line only if there are keywords left for the category
         if filtered_keywords:
             keywords_str = ", ".join(filtered_keywords)
             output_lines.append(f"target_{category_idx_str}: {keywords_str}")
         else:
             print(f"Category {category} (target_{category_idx_str}) has no keywords after filtering (or initially).")
             sys.exit(1)


    # Output the results
    if args.output_file:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for line in output_lines:
                    f.write(line + '\n')
            print(f"Results saved to {args.output_file}")
        except Exception as e:
            print(f"Error writing to output file '{args.output_file}': {e}", file=sys.stderr)
            # Fall back to console output if writing fails
            print("\n--- Outputting to Console Instead ---")
            for line in output_lines:
                print(line)
    else:
        # Print to console
        for line in output_lines:
            print(line)

if __name__ == "__main__":
    main()