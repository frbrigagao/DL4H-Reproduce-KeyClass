import re
from collections import Counter
from nltk.corpus import stopwords
import nltk
import csv
import sys

# Download stopwords if you haven't already
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


def analyze_icd9_descriptions(filename="CMS32_DESC_LONG_DX.csv"):
    """
    Analyzes ICD-9 descriptions to extract the most frequent words for each category,
    using NLTK's stop words.  Now reads a CSV file.

    Args:
        filename (str): The name of the CSV file containing the ICD-9 descriptions.

    Returns:
        dict: A dictionary where keys are category labels ('target_00', 'target_01', ...)
              and values are strings containing the most frequent words for each category.
    """

    category_descriptions = {f'target_{i:02d}': [] for i in range(20)}
    icd9_to_description = {}

    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        next(reader, None)  # Skip the header row

        for row in reader:
            code, description, trailing = row
            icd9_to_description[code] = description.lower()


    for icd9, description in icd9_to_description.items():
            if icd9.startswith('E'):
                icd9_top = 'target_18'
            elif icd9.startswith('V'):
                icd9_top = 'target_19'
            else:
                try:
                    icd9_num = int(float(icd9[:3]))

                    if 1 <= icd9_num <= 139:     # Infectious and parasitic diseases
                        icd9_top = 'target_00'
                    elif 140 <= icd9_num <= 239: # Neoplasms
                        icd9_top = 'target_01'
                    elif 240 <= icd9_num <= 279: # Endocrine, nutritional and metabolic diseases, and immunity disorders
                        icd9_top = 'target_02'
                    elif 280 <= icd9_num <= 289: # Diseases of blood and blood-forming organs
                        icd9_top = 'target_03'
                    elif 290 <= icd9_num <= 319: # Mental disorders
                        icd9_top = 'target_04'
                    elif 320 <= icd9_num <= 359: # Diseases of the nervous system
                        icd9_top = 'target_05'
                    elif 360 <= icd9_num <= 389: # Diseases of sense organs
                        icd9_top = 'target_06'
                    elif 390 <= icd9_num <= 459: # Diseases of the circulatory system
                        icd9_top = 'target_07'
                    elif 460 <= icd9_num <= 519: # Diseases of the respiratory system
                        icd9_top = 'target_08'
                    elif 520 <= icd9_num <= 579: # Diseases of the digestive system
                        icd9_top = 'target_09'
                    elif 580 <= icd9_num <= 629: # Diseases of the genitourinary system
                        icd9_top = 'target_10'
                    elif 630 <= icd9_num <= 679: # Complications of pregnancy, childbirth, and the puerperium
                        icd9_top = 'target_11'
                    elif 680 <= icd9_num <= 709: # Diseases of the skin and subcutaneous tissue
                        icd9_top = 'target_12'
                    elif 710 <= icd9_num <= 739: # Diseases of the musculoskeletal system and connective tissue
                        icd9_top = 'target_13'
                    elif 740 <= icd9_num <= 759: # Congenital anomalies
                        icd9_top = 'target_14'
                    elif 760 <= icd9_num <= 779: # Certain conditions originating in the perinatal period
                        icd9_top = 'target_15'
                    elif 780 <= icd9_num <= 799: # Symptoms, signs, and ill-defined conditions
                        icd9_top = 'target_16'
                    elif 800 <= icd9_num <= 999: # Injury and poisoning
                        icd9_top = 'target_17'
                except ValueError:
                    continue

            category_descriptions[icd9_top].append(description)


    category_keywords = {}
    for category, descriptions in category_descriptions.items():
        text = ' '.join(descriptions)
        words = re.findall(r'\b\w+\b', text)  # Extract words
        words = [word for word in words if word not in stop_words and len(word) > 3]  # Remove stop words and short words
        word_counts = Counter(words)
        most_common_words = [word for word, count in word_counts.most_common(50)]  # Get top 50
        category_keywords[category] = ', '.join(most_common_words)
    
    return category_keywords


if __name__ == "__main__":
    category_keywords = analyze_icd9_descriptions()
    for category, keywords in category_keywords.items():
        print(f"{category}: {keywords}")
        
    output_file = "output.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        sys.stdout = f  # Redirect standard output to the file

        for category, keywords in category_keywords.items():
            print(f"{category}: {keywords}")

    sys.stdout = sys.__stdout__  # Restore standard output to the console
    print(f"Output written to {output_file}") #inform user to redirect was successful