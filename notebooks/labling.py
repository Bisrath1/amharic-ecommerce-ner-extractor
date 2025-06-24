import pandas as pd
import re
import ast
from typing import List, Tuple
import numpy as np

# Valid entity labels
VALID_LABELS = {
    'O', 'B-Product', 'I-Product', 'B-PRICE', 'I-PRICE', 'B-LOC', 'I-LOC',
    'B-DELIVERY_FEE', 'I-DELIVERY_FEE', 'B-CONTACT_INFO', 'I-CONTACT_INFO'
}

def extract_entities(raw_text: str, normalized_text: str, tokens: List[str]) -> List[Tuple[str, str]]:
    """
    Extract entities from raw_text or normalized_text and tokens, assigning BIO tags.
    Returns a list of (token, label) tuples.
    """
    labels = ['O'] * len(tokens)
    
    # Use raw_text if available, otherwise normalized_text
    text = ""
    if isinstance(raw_text, str):
        text = raw_text.lower()
    elif isinstance(normalized_text, str):
        text = normalized_text.lower()
    else:
        print(f"Warning: No valid text for tokens {tokens[:5]}...; using empty string")
        text = ""

    # Rule-based entity detection
    for i in range(len(tokens)):
        token = tokens[i]

        # Product: Detect product names
        if i < len(tokens) - 2 and tokens[i:i+3] == ['ቡና', 'ማፍያ', 'ማሽን']:
            labels[i] = 'B-Product'
            labels[i+1] = 'I-Product'
            labels[i+2] = 'I-Product'
        elif 'silicone gel bath brush' in text and token in ['silicone', 'gel', 'bath', 'brush']:
            if token == 'silicone':
                labels[i] = 'B-Product'
            else:
                labels[i] = 'I-Product'
        elif token == 'ግስላው':  # Handle single-token product like 'ግስላው'
            labels[i] = 'B-Product'

        # Price: Detect price (e.g., "3400 ብር", "350 ብር")
        if re.match(r'^\d+$', token) and i + 1 < len(tokens) and tokens[i+1] == 'ብር':
            labels[i] = 'B-PRICE'
            labels[i+1] = 'I-PRICE'

        # Location: Detect location (e.g., "መገናኛ ዘፍመሽ ግራንድ ሞል", "ጀሞ 1")
        if token in ['መገናኛ', 'ጀሞ']:
            labels[i] = 'B-LOC'
            if i + 1 < len(tokens) and tokens[i+1] in ['ዘፍመሽ', '1']:
                labels[i+1] = 'I-LOC'
            if i + 2 < len(tokens) and tokens[i+2] in ['ግራንድ']:
                labels[i+2] = 'I-LOC'
            if i + 3 < len(tokens) and tokens[i+3] in ['ሞል']:
                labels[i+3] = 'I-LOC'

        # Delivery Fee: Detect delivery-related terms (e.g., "ነፃ መላኪያ")
        if i < len(tokens) - 1 and tokens[i:i+2] == ['ነፃ', 'መላኪያ']:
            labels[i] = 'B-DELIVERY_FEE'
            labels[i+1] = 'I-DELIVERY_FEE'

        # Contact Info: Detect phone numbers and Telegram handles
        if re.match(r'^09\d{8}$', token) or token.startswith('@'):
            labels[i] = 'B-CONTACT_INFO'

    return list(zip(tokens, labels))

def validate_conll_format(token_label_pairs: List[Tuple[str, str]]) -> bool:
    """
    Validate the CoNLL format for token-label pairs.
    Returns True if valid, False otherwise.
    """
    if not token_label_pairs:  # Skip empty token lists
        return False
    for i, (token, label) in enumerate(token_label_pairs):
        if not token or not label:
            print(f"Error: Empty token or label at index {i} - {token}, {label}")
            return False
        if label not in VALID_LABELS:
            print(f"Error: Invalid label at index {i} - {label}")
            return False
        if label.startswith('I-'):
            entity_type = label[2:]
            if i == 0:
                print(f"Error: {label} at start of sequence")
                return False
            prev_label = token_label_pairs[i-1][1]
            if not (prev_label == f'B-{entity_type}' or prev_label == f'I-{entity_type}'):
                print(f"Error: {label} follows invalid label {prev_label} at index {i}")
                return False
    return True

def save_to_conll(data: List[dict], output_file: str):
    """
    Process dataset and save labeled data in CoNLL format.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        messages_processed = 0
        for entry in data:
            # Ensure entry is a dictionary
            if not isinstance(entry, dict):
                print(f"Error: Entry is not a dictionary - {entry}")
                continue

            # Extract tokens, raw_text, and normalized_text
            try:
                tokens = entry['tokens']
                raw_text = entry['raw_text']
                normalized_text = entry['normalized_text']
                message_id = entry.get('message_id', 'unknown')
            except KeyError as e:
                print(f"Error: Missing key {e} in entry {message_id}")
                continue

            # Handle stringified tokens
            if isinstance(tokens, str):
                try:
                    tokens = ast.literal_eval(tokens)
                except (ValueError, SyntaxError) as e:
                    print(f"Error: Failed to parse tokens for message {message_id} - {tokens}")
                    continue

            # Ensure tokens is a list and not empty
            if not isinstance(tokens, list):
                print(f"Error: Tokens is not a list for警方")
                continue
            if not tokens:
                print(f"Warning: Empty tokens list for message {message_id}")
                continue

            # Extract entities and assign labels
            token_label_pairs = extract_entities(raw_text, normalized_text, tokens)
            
            # Validate format
            if not validate_conll_format(token_label_pairs):
                print(f"Skipping message {message_id} due to invalid format")
                continue
            
            # Write to CoNLL file
            for token, label in token_label_pairs:
                f.write(f"{token} {label}\n")
            f.write("\n")  # Blank line to separate messages
            messages_processed += 1

        print(f"Processed {messages_processed} messages")

def main():
    # Load CSV file
    csv_file = r"C:\10x AIMastery\amharic-ecommerce-ner-extractor\preprocessed_telegram_data.csv"  # Update with your actual CSV path
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file is empty at {csv_file}")
        return

    # Handle NaN values
    df['raw_text'] = df['raw_text'].fillna('')
    df['normalized_text'] = df['normalized_text'].fillna('')

    # Convert DataFrame to list of dictionaries
    data = df.to_dict('records')
    
    # Print first few entries to verify structure
    print("First few entries of data:")
    for i, entry in enumerate(data[:3]):
        print(f"Entry {i}: {entry.get('message_id', 'unknown')} - tokens: {entry.get('tokens', 'missing')}")
        print(f"  raw_text: {entry.get('raw_text', 'missing')[:50]}...")
        print(f"  normalized_text: {entry.get('normalized_text', 'missing')[:50]}...")

    # Save to CoNLL format
    output_file = 'C:/10x AIMastery/amharic-ecommerce-ner-extractor/amharic_ner.conll'
    save_to_conll(data, output_file)
    print(f"CoNLL file saved as {output_file}")

if __name__ == "__main__":
    main()
    
    
   