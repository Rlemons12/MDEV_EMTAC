import os
import sys
import time
import random
from PIL import Image as PILImage, ImageFile
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from modules.configuration.log_config import logger
from modules.configuration.config import DATABASE_PATH_IMAGES_FOLDER
from modules.initial_setup.initializer_logger import (
    LOG_DIRECTORY, initializer_logger, close_initializer_logger
)
from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import load_image_model_config_from_db, Image
from plugins.image_modules import CLIPModelHandler, NoImageModel

# Make sure truncated images won't crash PIL
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Initialize Logging
logger = initializer_logger
logger.info(f"Using logs directory: {LOG_DIRECTORY}")

# Database Setup - Enable connection limiting and set lower max connections
os.environ['DB_CONNECTION_LIMITING'] = 'True'
os.environ['MAX_DB_CONNECTIONS'] = '4'  # Reduce from default 8 to 4
os.environ['DB_CONNECTION_TIMEOUT'] = '60'  # Set a longer timeout of 60 seconds

# Initialize database configuration
db_config = DatabaseConfig()

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
        else:
            print("Invalid choice. Please select 1, 2, or 3.")


def set_models():
    """
    Allow the admin to set the image model.
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
    Process a single image using the Image class methods.
    Now with improved error handling and retry logic.
    """
    start_time_image = time.time()
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        # Get a session - connection limiting is now built-in to this method
        session = db_config.get_main_session()

        try:
            source_file_path = os.path.join(folder_path, filename)
            if os.path.isdir(source_file_path):
                logger.debug(f"Skipping subdirectory: {source_file_path}")
                result_message = f"Skipped directory: {filename}"
                session.close()
                return (result_message, time.time() - start_time_image)

            if image_handler.allowed_file(filename):
                # Extract the base name without the extension
                file_base, ext = os.path.splitext(filename)

                logger.info(f"Processing image: {filename}")

                # Use the Image.add_to_db class method to add the image to the database
                # It will now clean the title internally (clean_title=True by default)
                new_image = Image.add_to_db(
                    session=session,
                    title=file_base,
                    file_path=source_file_path,
                    description="Auto-generated description"
                )

                # Immediately commit changes to reduce lock time
                session.commit()

                logger.info(f"Successfully processed and stored '{filename}' with ID: {new_image.id}")
                result_message = f"Processed: {new_image.title}"  # Use the possibly cleaned title from the returned image
            else:
                logger.info(f"Skipping non-image file: '{filename}'")
                result_message = f"Skipped non-image: {filename}"

            # Successfully processed, break retry loop
            break

        except Exception as e:
            # Check if it's a database lock error
            if "database is locked" in str(e).lower():
                retry_count += 1
                backoff_time = random.uniform(0.5, 2.0) * retry_count  # Exponential backoff with jitter
                logger.warning(
                    f"Database locked while processing '{filename}'. Retry {retry_count}/{max_retries} after {backoff_time:.2f}s")

                # Always rollback on error
                try:
                    session.rollback()
                except:
                    pass

                # Close session to release connection
                try:
                    session.close()
                except:
                    pass

                # Wait before retrying
                time.sleep(backoff_time)

                # If it's not the last retry, continue to next iteration
                if retry_count < max_retries:
                    continue

            # For non-lock errors or final retry failure
            logger.error(f"Failed to process '{filename}': {e}", exc_info=True)
            try:
                session.rollback()
            except:
                pass

            result_message = f"Error processing: {filename}"
            break

        finally:
            # Always close the session in the finally block
            try:
                session.close()
            except:
                pass

    processing_time = time.time() - start_time_image
    return (result_message, processing_time)


def process_and_store_images(folder_path: str):
    """
    Scans the given folder_path for images and processes them concurrently using multithreading.
    Now with improved batch processing and worker management.
    """
    logger.debug(f"Starting processing for folder: {folder_path}")
    folder_start_time = time.time()

    # Ensure the destination folder exists (for database images)
    os.makedirs(DATABASE_PATH_IMAGES_FOLDER, exist_ok=True)

    filenames = os.listdir(folder_path)
    num_files = len(filenames)
    logger.info(f"Found {num_files} files in '{folder_path}'.")

    # Ask the user if they want to proceed
    proceed = input(
        f"There are {num_files} files in folder '{folder_path}'. Do you want to proceed with processing? (y/n): ").strip().lower()
    if proceed != "y":
        logger.info("User chose not to proceed with processing this folder.")
        return

    # More conservative worker settings - don't let CPU count dictate this
    # since SQLite is the bottleneck, not CPU
    max_workers = min(num_files, os.cpu_count() or 1, 8)  # No more than 8 workers
    logger.info(f"Using {max_workers} worker threads for processing.")

    # Log connection stats for monitoring
    logger.info(f"Database connection stats: {db_config.get_connection_stats()}")

    individual_times = []
    processed_count = 0
    total_count = len(filenames)

    # Use smaller batches to reduce concurrent database load
    batch_size = max(1, min(20, num_files // 20))  # Smaller batches

    # Shuffle the filenames to distribute large/small images more evenly
    random.shuffle(filenames)

    for batch_start in range(0, num_files, batch_size):
        batch_end = min(batch_start + batch_size, num_files)
        batch_files = filenames[batch_start:batch_end]

        # Log batch info
        logger.info(
            f"Processing batch {batch_start // batch_size + 1} of {(num_files + batch_size - 1) // batch_size} ({len(batch_files)} files)")

        # Use a new ThreadPoolExecutor for each batch to ensure clean resources
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_image, folder_path, filename): filename
                for filename in batch_files
            }
            for future in as_completed(futures):
                filename = futures[future]
                try:
                    result_message, process_time = future.result()
                    individual_times.append(process_time)
                    processed_count += 1

                    # Display progress
                    progress = processed_count / total_count * 100
                    logger.info(f"{result_message} (Time: {process_time:.2f} sec) - Progress: {progress:.1f}%")

                except Exception as e:
                    logger.error(f"Error processing file '{filename}': {e}", exc_info=True)

        # Wait a short time between batches to let any lingering transactions complete
        time.sleep(1.0)

        # Log connection stats after each batch for monitoring
        logger.info(f"Database connection stats after batch: {db_config.get_connection_stats()}")

    # Calculate stats
    folder_elapsed_time = time.time() - folder_start_time
    avg_image_time = (sum(individual_times) / len(individual_times)) if individual_times else 0
    success_rate = processed_count / total_count * 100 if total_count else 0

    logger.info(f"Finished processing folder '{folder_path}' in {folder_elapsed_time:.2f} seconds.")
    logger.info(f"Average processing time per image: {avg_image_time:.2f} seconds.")
    logger.info(f"Successfully processed {processed_count} of {total_count} files ({success_rate:.1f}%).")


def main():
    """
    Main function for image processing setup.
    """
    logger.info("=== Starting EMTACDB Image Setup ===")
    logger.info(f"DatabaseConfig settings: {db_config.get_connection_stats()}")

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
        # Ensure all sessions are closed
        db_config.get_main_session_registry().remove()
        close_initializer_logger()