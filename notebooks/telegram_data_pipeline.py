import asyncio
import platform
from telethon import TelegramClient
import pandas as pd
import os
import re
from datetime import datetime
try:
    from etnltk import Amharic
    etnltk_available = True
except ImportError:
    etnltk_available = False
    raise ImportError("Please install etnltk: `pip install etnltk` or `git clone https://github.com/robeleq/etnltk.git; cd etnltk; pip install -e .`")

# Telegram API credentials
api_id = '.......'
api_hash = '.....'
phone = '....'

# List of Telegram channels
channels = [
    '@qnashcom',
    '@AwasMart',
    '@aradabrand2',
    '@marakibrand',
    '@classybrands'
]

# Initialize Telegram client
client = TelegramClient('session_name', api_id, api_hash)

# Directory to save images
image_dir = 'telegram_images'
os.makedirs(image_dir, exist_ok=True)

async def fetch_messages(channel, limit=100):
    """Fetch messages and images from a Telegram channel."""
    messages_data = []
    async for message in client.iter_messages(channel, limit=limit):
        # Extract metadata
        msg_id = message.id
        timestamp = message.date
        sender = message.sender_id
        
        # Extract text content
        text = message.text if message.text else ''
        
        # Handle images
        image_path = None
        if message.photo:
            image_filename = f"{image_dir}/{channel.replace('@', '')}_{msg_id}.jpg"
            await message.download_media(file=image_filename)
            image_path = image_filename
        
        # Append data
        messages_data.append({
            'channel': channel,
            'message_id': msg_id,
            'timestamp': timestamp,
            'sender_id': sender,
            'text': text,
            'image_path': image_path
        })
    
    return messages_data

def preprocess_amharic_text(text):
    """Preprocess Amharic text using ETNLTK's Amharic class with custom cleaning."""
    if not text or not isinstance(text, str):
        return {'normalized_text': '', 'tokens': []}
    
    if etnltk_available:
        # Step 1: Custom cleaning pipeline
        # Remove emojis (Unicode ranges for emojis)
        cleaned_text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]', '', text)
        # Remove digits (Arabic and Ethiopic)
        cleaned_text = re.sub(r'[0-9፩፪፫፬፭፮፯፰፱፲]', '', cleaned_text)
        # Remove English characters
        cleaned_text = re.sub(r'[a-zA-Z]', '', cleaned_text)
        # Remove Ethiopic punctuation
        cleaned_text = re.sub(r'[፠፡።፣፤፥፦፧፨]', '', cleaned_text)
        # Remove extra spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Step 2: Handle empty cleaned text
        if not cleaned_text:
            return {'normalized_text': '', 'tokens': []}
        
        # Step 3: Create Amharic document for normalization and tokenization
        try:
            doc = Amharic(cleaned_text)
            normalized_text = str(doc)  # Normalized text
            tokens = [str(word) for word in doc.words if str(word).strip()]  # Word tokens
        except Exception as e:
            print(f"ETNLTK processing failed for text '{text}': {e}")
            normalized_text = cleaned_text
            tokens = cleaned_text.split()
        
        return {
            'normalized_text': normalized_text,
            'tokens': tokens
        }
    else:
        raise ImportError("ETNLTK is required for preprocessing. Install it using `pip install etnltk`.")

def structure_data(messages_data, output_file='preprocessed_telegram_data.csv'):
    """Clean and structure data into a unified CSV format."""
    # Convert to DataFrame
    df = pd.DataFrame(messages_data)
    
    # Apply preprocessing to text column
    preprocess_results = df['text'].apply(preprocess_amharic_text)
    
    # Extract normalized text and tokens
    df['normalized_text'] = preprocess_results.apply(lambda x: x['normalized_text'])
    df['tokens'] = preprocess_results.apply(lambda x: x['tokens'])
    
    # Ensure metadata columns are present
    required_columns = ['channel', 'message_id', 'timestamp', 'sender_id', 'raw_text', 'normalized_text', 'tokens', 'image_path']
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
    
    # Rename 'text' to 'raw_text' for clarity
    df = df.rename(columns={'text': 'raw_text'})
    
    # Reorder columns
    df = df[required_columns]
    
    # Save to CSV
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Preprocessed data saved to {output_file}")
    
    return df

async def main():
    """Main function to fetch, preprocess, and structure data from Telegram channels."""
    all_messages = []
    
    # Connect to Telegram
    await client.start(phone)
    
    # Fetch messages from each channel
    for channel in channels:
        print(f"Fetching messages from {channel}")
        messages = await fetch_messages(channel, limit=100)
        all_messages.extend(messages)
    
    # Structure and save preprocessed data
    df = structure_data(all_messages)
    
    # Display sample output
    print("\nSample Preprocessed Data:")
    print(df[['channel', 'raw_text', 'normalized_text', 'tokens']].head())
    
    # Disconnect client
    await client.disconnect()

# Run the script
if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())