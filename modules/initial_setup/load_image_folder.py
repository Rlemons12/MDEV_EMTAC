#!/usr/bin/env python3
import os
import sys
import time
from PIL import Image as PILImage, ImageFile
from logging import Logger  # Ensure you have the Logger imported if needed
from sqlalchemy.orm import scoped_session

from modules.configuration.config import DATABASE_PATH_IMAGES_FOLDER
from modules.initial_setup.initializer_logger import (
    LOG_DIRECTORY, initializer_logger, close_initializer_logger
)
from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import load_image_model_config_from_db
from plugins.image_modules import CLIPModelHandler, NoImageModel

# Make sure truncated images won't crash PIL
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Initialize Logging
logger = initializer_logger
logger.info(f"Using logs in directory: {LOG_DIRECTORY}")

# Database Setup
db_config = DatabaseConfig()
session_factory = db_config.get_main_session

# Load the current image model
CURRENT_IMAGE_MODEL = load_image_model_config_from_db()
image_handler = CLIPModelHandler() if CURRENT_IMAGE_MODEL != "no_model" else NoImageModel()

def prompt_model_selection():
    """
    Prompt the user to select the image model to use.
    """
    logger.info("Prompting user for model selection.")
    print("Select an image model to use:")
    print("1. CLIPModelHandler")
    print("2. NoImageModel (Skip embedding generation)")
    print("3. Custom (Enter the name of another model)")

    while True:
        choice = input("> ").strip()
        if choice == "1":
            return "CLIPModelHandler"
        elif choice == "2":
            return "NoImageModel"
        elif choice == "3":
            custom_model = input("Enter the name of the custom model: ").strip()
            return custom_model
        else:
            print("Invalid choice. Please select 1, 2, or 3.")

def set_models():
    """
    Allow the admin to set the AI, embedding, and image models.
    """
    global CURRENT_IMAGE_MODEL, image_handler

    logger.info("Setting models based on user input.")
    CURRENT_IMAGE_MODEL = prompt_model_selection()

    if CURRENT_IMAGE_MODEL == "CLIPModelHandler":
        image_handler = CLIPModelHandler()
    elif CURRENT_IMAGE_MODEL == "NoImageModel":
        image_handler = NoImageModel()
    else:
        # Here you can integrate additional handlers for other models
        logger.warning(f"Custom model '{CURRENT_IMAGE_MODEL}' not fully implemented. Defaulting to NoImageModel.")
        image_handler = NoImageModel()

    logger.info(f"Current Image Model set to: {CURRENT_IMAGE_MODEL}")

def process_and_store_images(folder_path: str):
    """
    Scans the given folder_path for images and processes them.
    """
    logger.debug(f"Starting processing for folder: {folder_path}")
    start_time = time.time()

    session = session_factory()
    logger.debug("Database session created.")

    try:
        os.makedirs(DATABASE_PATH_IMAGES_FOLDER, exist_ok=True)
        logger.debug(f"Ensured that the images folder exists at: {DATABASE_PATH_IMAGES_FOLDER}")

        filenames = os.listdir(folder_path)
        logger.info(f"Found {len(filenames)} files in '{folder_path}'.")

        for index, filename in enumerate(filenames, start=1):
            source_file_path = os.path.join(folder_path, filename)
            logger.debug(f"Processing file {index}/{len(filenames)}: {source_file_path}")

            if os.path.isdir(source_file_path):
                logger.debug(f"Skipping subdirectory: {source_file_path}")
                continue

            if image_handler.allowed_file(filename):
                logger.debug(f"File '{filename}' is allowed for processing.")
                try:
                    dest_file_path = os.path.join(DATABASE_PATH_IMAGES_FOLDER, filename)
                    logger.info(f"Processing image: {filename}")

                    image = PILImage.open(source_file_path).convert("RGB")
                    logger.debug(f"Opened and converted image '{filename}' to RGB.")

                    if not image_handler.is_valid_image(image):
                        logger.warning(f"Skipping '{filename}': Invalid dimension/aspect ratio.")
                        continue

                    embedding = image_handler.get_image_embedding(image)
                    embedding_preview = embedding[:10] if hasattr(embedding, '__iter__') else 'N/A'
                    logger.debug(f"Generated embedding for image '{filename}': {embedding_preview}...")

                    image.save(dest_file_path)
                    logger.debug(f"Saved image '{filename}' to '{dest_file_path}'.")

                    image_handler.store_image_metadata(
                        session=session,
                        title=filename,
                        description="Auto-generated description",
                        file_path=dest_file_path,
                        embedding=embedding,
                        model_name=CURRENT_IMAGE_MODEL
                    )
                    logger.info(f"Successfully stored '{filename}' in DB and saved to '{DATABASE_PATH_IMAGES_FOLDER}'.")

                except Exception as e:
                    logger.error(f"Failed to process '{filename}': {e}", exc_info=True)
            else:
                logger.info(f"Skipping non-image file: '{filename}'")

    except Exception as e:
        logger.critical(f"An unexpected error occurred while processing folder '{folder_path}': {e}", exc_info=True)
    finally:
        session.close()
        logger.debug("Database session closed.")
        elapsed_time = time.time() - start_time
        logger.info(f"Finished processing folder '{folder_path}' in {elapsed_time:.2f} seconds.")

def main():
    """
    Main function for image processing setup.
    """
    logger.info("=== Starting EMTACDB Image Setup ===")
    set_models()
    logger.debug(f"Selected Image Model: {CURRENT_IMAGE_MODEL}")

    if len(sys.argv) > 1:
        folders = sys.argv[1:]
        logger.debug(f"Received CLI arguments for folders: {folders}")
    else:
        folders = []
        logger.info("Enter folder paths containing images. Blank line finishes input.")
        while True:
            folder_path = input("> ").strip().strip('"').strip("'")
            if not folder_path:
                logger.debug("No more folders entered by the user.")
                break
            if not os.path.isdir(folder_path):
                logger.warning(f"Invalid directory: {folder_path}")
            else:
                folders.append(folder_path)
                logger.debug(f"Added folder to processing list: {folder_path}")

    if not folders:
        logger.error("No valid folders provided. Exiting setup.")
        return

    for folder in folders:
        logger.info(f"\n--- Processing folder: {folder} ---")
        process_and_store_images(folder)

    logger.info("=== EMTACDB Image Setup Complete! ===")

if __name__ == "__main__":
    try:
        main()
    finally:
        close_initializer_logger()
