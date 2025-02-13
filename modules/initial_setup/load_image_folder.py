import os
import sys
import time
from PIL import Image as PILImage, ImageFile
from modules.configuration.log_config import logger  # Ensure you have the Logger imported if needed
from sqlalchemy.orm import scoped_session

# Additional imports needed for multithreading and numerical computations
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from modules.configuration.config import DATABASE_PATH_IMAGES_FOLDER
from modules.initial_setup.initializer_logger import (
    LOG_DIRECTORY, initializer_logger, close_initializer_logger
)
from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import load_image_model_config_from_db
from plugins.image_modules import CLIPModelHandler, NoImageModel, BaseImageModelHandler

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


def is_duplicate_embedding(new_embedding, session, threshold=0.9):
    """
    Checks if the new_embedding is too similar to any stored embeddings.
    This function queries the database for all stored image embeddings and computes
    cosine similarity between the new embedding and each stored embedding.

    Returns True if a duplicate is found, otherwise False.
    """
    # Import the ImageEmbedding model (adjust the import path to your project structure)
    from modules.emtacdb.emtacdb_fts import ImageEmbedding

    stored_embeddings = session.query(ImageEmbedding).all()
    new_vec = np.array(new_embedding, dtype=np.float32)

    for record in stored_embeddings:
        # Convert the stored bytes back to a NumPy array.
        # Adjust the dtype if needed to match your embeddings.
        stored_vec = np.frombuffer(record.model_embedding, dtype=np.float32)
        # Compute cosine similarity
        similarity = np.dot(new_vec, stored_vec) / (np.linalg.norm(new_vec) * np.linalg.norm(stored_vec))
        if similarity > threshold:
            logger.debug(
                f"Found duplicate with similarity {similarity:.2f} for image embedding record with image id {record.image_id}")
            return True
    return False


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
        logger.warning(f"Custom model '{CURRENT_IMAGE_MODEL}' not fully implemented. Defaulting to NoImageModel.")
        image_handler = NoImageModel()

    logger.info(f"Current Image Model set to: {CURRENT_IMAGE_MODEL}")


def process_single_image(folder_path: str, filename: str):
    """
    Process a single image: validate, generate embedding, optionally check for duplicates,
    save the processed image, and store metadata.

    This function measures how long it takes to process the image.

    Returns:
        tuple: (result_message, processing_time_in_seconds)
    """
    start_time_image = time.time()
    session = session_factory()  # Create a session for this thread
    try:
        source_file_path = os.path.join(folder_path, filename)
        if os.path.isdir(source_file_path):
            logger.debug(f"Skipping subdirectory: {source_file_path}")
            result_message = f"Skipped directory: {filename}"
            return (result_message, time.time() - start_time_image)

        if image_handler.allowed_file(filename):
            # Extract the base name without the extension
            file_base, ext = os.path.splitext(filename)
            # Determine file format based on the original extension
            if ext.lower() in [".jpg", ".jpeg"]:
                file_format = "JPEG"
            elif ext.lower() == ".png":
                file_format = "PNG"
            else:
                file_format = "JPEG"  # default fallback if extension is unknown

            # Create destination file path without the file extension in the filename
            dest_file_path = os.path.join(DATABASE_PATH_IMAGES_FOLDER, file_base)
            logger.info(f"Processing image: {filename}")

            # Open and convert image to RGB
            image = PILImage.open(source_file_path)
            image = image.convert("RGB")
            logger.debug(f"Opened and converted image '{filename}' to RGB.")

            if not image_handler.is_valid_image(image):
                logger.warning(f"Skipping '{filename}': Invalid dimension/aspect ratio.")
                result_message = f"Skipped invalid image: {filename}"
                return (result_message, time.time() - start_time_image)

            # Generate image embedding
            embedding = image_handler.get_image_embedding(image)
            embedding_preview = embedding[:10] if hasattr(embedding, '__iter__') else 'N/A'
            logger.debug(f"Generated embedding for image '{filename}': {embedding_preview}...")

            # Optional: Check for duplicate using the embedding
            if is_duplicate_embedding(embedding, session, threshold=0.9):
                logger.info(f"Skipping '{filename}': duplicate image based on embedding.")
                result_message = f"Skipped duplicate image: {filename}"
                return (result_message, time.time() - start_time_image)

            # Save the processed image without the file extension in its name.
            image.save(dest_file_path, format=file_format, optimize=True, quality=85)
            logger.debug(f"Saved image '{filename}' as '{file_base}' in '{DATABASE_PATH_IMAGES_FOLDER}'.")

            # Store metadata in the database using the file base (name without extension) as the title.
            image_handler.store_image_metadata(
                session=session,
                title=file_base,
                description="Auto-generated description",
                file_path=dest_file_path,
                embedding=embedding,
                model_name=CURRENT_IMAGE_MODEL
            )
            logger.info(f"Successfully processed and stored '{filename}' as '{file_base}'.")
            result_message = f"Processed: {file_base}"
        else:
            logger.info(f"Skipping non-image file: '{filename}'")
            result_message = f"Skipped non-image: {filename}"
    except Exception as e:
        logger.error(f"Failed to process '{filename}': {e}", exc_info=True)
        result_message = f"Error processing: {filename}"
    finally:
        session.close()

    processing_time = time.time() - start_time_image
    return (result_message, processing_time)


def process_and_store_images(folder_path: str):
    """
    Scans the given folder_path for images and processes them concurrently using multithreading.
    Logs the total processing time for the folder and the average processing time per image.
    Dynamically sets the number of worker threads.
    """
    logger.debug(f"Starting processing for folder: {folder_path}")
    folder_start_time = time.time()

    # Ensure the destination folder exists
    os.makedirs(DATABASE_PATH_IMAGES_FOLDER, exist_ok=True)
    filenames = os.listdir(folder_path)
    num_files = len(filenames)
    logger.info(f"Found {num_files} files in '{folder_path}'.")

    # Ask the user if they want to proceed
    proceed = input(f"There are {num_files} files in folder '{folder_path}'. Do you want to proceed with processing? (y/n): ").strip().lower()
    if proceed != "y":
        logger.info("User chose not to proceed with processing this folder.")
        return

    # Dynamically determine the number of worker threads.
    default_workers = os.cpu_count() or 1  # For example, 4 cores return 4.
    dynamic_workers = default_workers * 5   # Multiply for I/O-bound tasks.
    max_workers = min(num_files, dynamic_workers)  # Limit workers to the number of files if necessary.
    logger.info(f"Using {max_workers} worker threads for processing.")

    individual_times = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_image, folder_path, filename): filename
            for filename in filenames
        }
        for future in as_completed(futures):
            filename = futures[future]
            try:
                result_message, process_time = future.result()
                individual_times.append(process_time)
                logger.info(f"{result_message} (Time: {process_time:.2f} sec)")
            except Exception as e:
                logger.error(f"Error processing file '{filename}': {e}", exc_info=True)

    folder_elapsed_time = time.time() - folder_start_time
    avg_image_time = (sum(individual_times) / len(individual_times)) if individual_times else 0

    logger.info(f"Finished processing folder '{folder_path}' in {folder_elapsed_time:.2f} seconds.")
    logger.info(f"Average processing time per image: {avg_image_time:.2f} seconds.")


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
