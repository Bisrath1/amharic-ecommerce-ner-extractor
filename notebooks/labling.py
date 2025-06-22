import pandas as pd
import ast
import re
from typing import List, Tuple
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to parse tokens from string representation
def parse_tokens(token_str: str) -> List[str]:
    try:
        return ast.literal_eval(token_str)
    except (ValueError, SyntaxError) as e:
        logging.error(f"Error parsing tokens: {token_str}. Error: {e}")
        return []

# Function to identify entities in tokens using rule-based approach
def label_tokens(tokens: List[str], raw_text: str) -> List[Tuple[str, str]]:
    labeled_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in ["ቡና", "ማፍያ", "ማሽን", "ቢላ", "ተጣጣፊ", "ቅልጥ", "አጎበር"]:
            if i == 0 or labeled_tokens[-1][1] not in ["B-Product", "I-Product"]:
                labeled_tokens.append((token, "B-Product"))
            else:
                labeled_tokens.append((token, "I-Product"))
        elif token == "ዋጋ":
            labeled_tokens.append((token, "B-PRICE"))
            price_match = re.search(r'ዋጋ.*?\s(\d+)\s*ብር', raw_text)
            if price_match and i + 1 < len(tokens) and tokens[i +  == "1"]:
                labeled_tokens.append((tokens[i + 1], "I-PRICE"))
                labeled_tokens.append((price_match.group(1), "I-PRICE"))
                i += 1
            elif i + 1 < len(tokens) and tokens[i + 1 == "ብር"]:
                labeled_tokens.append((tokens[i + 1], "I-PRICE"))
                i += 1
        elif token in ["መገ�", "ና"]:
            labeled_tokens.append((token, "B-LOC"))
            j = i + 1
            while j < len(tokens) and tokens[j] in ["ውፍሽመ", "ራ", "ግራ", "ተኛ", "ፎቅ", "ቁጥር", "ከለላ", "ህንፃ", "ግራውንድ", "ለይ"]:
                labeled_tokens.append((tokens[j], "I-LOC"))
                j += 1
            i = j - 1
        elif re.match(r'09\d{8}', token) or token.startswith('@'):
            labeled_tokens.append((token, "B-CONTACT_INFO"))
        elif token in ["መላኪያ", "ክፍያ"]:
            labeled_tokens.append((token, "B-DELIVERY_FEE"))
            if i + 1 < len(tokens) and re.match(r'\d+', tokens[i + 1]):
                labeled_tokens.append((tokens[i + 1], "I-DELIVERY_FEE"))
                i += 1
        else:
            labeled_tokens.append((token, "O"))
        i += 1

    phone_numbers = re.findall(r'09\d{8}', raw_text)
    telegram_handles = re.findall(r'@\w+', raw_text)
    for phone in phone_numbers:
        labeled_tokens.append((phone, "B-CONTACT_INFO"))
    for handle in telegram_handles:
        labeled_tokens.append((handle, "B-CONTACT_INFO"))

    return labeled_tokens

# Function to save labeled data in CoNLL format
def save_conll_file(labeled_data: List[List[Tuple[str, str]]], output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        for message in labeled_data:
            for token, label in message:
                f.write(f"{token} {label}\n")
            f.write("\n")
    logging.info(f"Labeled data saved to {output_file}")

# Main function to process CSV and label data
def label_dataset(csv_file: str, output_file: str, num_messages: int = 50):
    if not os.path.exists(csv_file):
        logging.error(f"CSV file {csv_file} not found.")
        return

    try:
        df = pd.read_csv(csv_file)
        logging.info(f"Loaded CSV file with {len(df)} rows.")
    except Exception as e:
        logging.error(f"Error reading CSV file {csv_file}: {e}")
        return

    df = df.head(num_messages)
    all_labeled_data = []

    for idx, row in df.iterrows():
        tokens = parse_tokens(row['tokens'])
        raw_text = row['raw_text']
        
        if not tokens:
            logging.warning(f"Skipping message_id {row['message_id']} due to empty tokens.")
            continue

        labeled_tokens = label_tokens(tokens, raw_text)
        all_labeled_data.append(labeled_tokens)
        logging.info(f"Labeled message_id {row['message_id']} with {len(labeled_tokens)} tokens.")

    save_conll_file(all_labeled_data, output_file)

# Example usage
if __name__ == "__main__":
    csv_file = r"C:\10x AIMastery\amharic-ecommerce-ner-extractor\preprocessed_telegram_data.csv"
    output_file = "labeled_data.conll"
    label_dataset(csv_file, output_file, num_messages=50)