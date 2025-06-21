from telethon.sync import TelegramClient
from telethon.tl.functions.channels import JoinChannelRequest
import os

# Telegram API credentials
api_id = '26139296'
api_hash = 'b54335ce5928cc0f0fbdda8d31abd79e'
phone = '+251994180777'

# List of Telegram channels
channels = [
    '@qnashcom',
    '@AwasMart',
    '@aradabrand2',
    '@marakibrand',
    '@classybrands'
]

async def fetch_messages():
    async with TelegramClient('session_name', api_id, api_hash) as client:
        await client.start(phone=phone)
        os.makedirs('data/text', exist_ok=True)
        os.makedirs('data/images', exist_ok=True)
        os.makedirs('data/documents', exist_ok=True)

        for channel in channels:
            try:
                async for message in client.iter_messages(channel, limit=100):  # Adjust limit as needed
                    # Save text messages
                    if message.text:
                        with open(f'data/text/{channel[1:]}_{message.id}.txt', 'w', encoding='utf-8') as f:
                            f.write(f"Timestamp: {message.date}\nSender: {message.sender_id}\nText: {message.text}\n")
                        print(f"Saved text message {message.id} from {channel}")

                    # Save images
                    if message.photo:
                        path = await message.download_media(f'data/images/{channel[1:]}_{message.id}.jpg')
                        print(f"Saved image {message.id} from {channel} to {path}")

                    # Save documents
                    if message.document:
                        path = await message.download_media(f'data/documents/{channel[1:]}_{message.id}')
                        print(f"Saved document {message.id} from {channel} to {path}")

            except Exception as e:
                print(f"Error fetching from {channel}: {e}")

if __name__ == '__main__':
    import asyncio
    asyncio.run(fetch_messages())
