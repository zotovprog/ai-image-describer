import os
import logging
import requests
from pymongo import MongoClient
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("image_description_service.log"),
        logging.StreamHandler()
    ]
)

# MongoDB connection
mongo_uri = "mongodb://gen_user:%25EtbJ%23lMUzNTL0@89.223.69.220:27017/default_db?authSource=admin&directConnection=true"
client = MongoClient(mongo_uri)
db = client["pumpFunTokens"]
collection = db["tokens"]

# Constants
BATCH_SIZE = 50  # Adjust batch size based on your requirements
TIMEOUT = 10  # Request timeout in seconds
SLEEP_INTERVAL = 30  # Time to wait before checking for new tokens

# AI Model for Image Description
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def fetch_image_description(image_url):
    """
    Fetches an image from the URL and generates a description using the AI model.
    """
    try:
        response = requests.get(image_url, timeout=TIMEOUT)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))

        inputs = processor(images=image, return_tensors="pt")
        caption_ids = model.generate(**inputs)
        caption = processor.decode(caption_ids[0], skip_special_tokens=True)

        logging.info(f"Generated description: {caption}")
        return caption
    except Exception as e:
        logging.error(f"Failed to generate description for image: {image_url}, Error: {e}")
        return None

def process_batch(docs):
    """
    Processes a batch of tokens, fetches the image from the Uri, generates descriptions,
    and updates the database.
    """
    for doc in docs:
        token_id = doc["_id"]
        uri = doc.get("Uri")

        if not uri:
            logging.warning(f"Token {token_id} has no Uri. Skipping.")
            collection.update_one(
                {"_id": token_id},
                {"$set": {"image_status": "failed"}}
            )
            continue

        try:
            # Fetch the JSON data from the Uri
            response = requests.get(uri, timeout=TIMEOUT)
            response.raise_for_status()
            json_data = response.json()

            # Extract image URL from the JSON
            image_url = json_data.get("image")
            if not image_url:
                logging.warning(f"No 'image' field in JSON from {uri}. Skipping token {token_id}.")
                collection.update_one(
                    {"_id": token_id},
                    {"$set": {"image_status": "failed"}}
                )
                continue

            # Generate image description
            description = fetch_image_description(image_url)
            if description:
                # Update the database with the generated description
                collection.update_one(
                    {"_id": token_id},
                    {"$set": {"image_description": description, "image_status": "success"}}
                )
                logging.info(f"Token {token_id} updated with image description.")
            else:
                logging.warning(f"Failed to generate description for token {token_id}.")
                collection.update_one(
                    {"_id": token_id},
                    {"$set": {"image_status": "failed"}}
                )

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch JSON from {uri}. Error: {e}")
            collection.update_one(
                {"_id": token_id},
                {"$set": {"image_status": "failed"}}
            )
        except Exception as e:
            logging.error(f"Unexpected error for token {token_id}. Error: {e}")
            collection.update_one(
                {"_id": token_id},
                {"$set": {"image_status": "failed"}}
            )

def fetch_tokens_without_descriptions(batch_size):
    """
    Fetches tokens from the database that don't have an image description and are not marked as failed.
    """
    return list(collection.find(
        {"image_description": {"$exists": False}, "image_status": {"$ne": "failed"}},
        {"Uri": 1, "_id": 1}
    ).limit(batch_size))

def main():
    logging.info("Starting image description service...")
    while True:
        try:
            docs = fetch_tokens_without_descriptions(BATCH_SIZE)
            if not docs:
                logging.info("No tokens left to process. Waiting for new entries...")
                time.sleep(SLEEP_INTERVAL)
                continue
            process_batch(docs)
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {e}")
            time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    main()