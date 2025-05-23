import os
import sys
import time
import random
from PIL import Image as PILImage, ImageFile
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from datetime import datetime, timedelta
import threading

from modules.configuration.log_config import (
    logger, with_request_id, get_request_id, set_request_id,
    info_id, debug_id, warning_id, error_id, log_timed_operation
)
from modules.configuration.config import DATABASE_PATH_IMAGES_FOLDER, ALLOWED_EXTENSIONS
from modules.initial_setup.initializer_logger import (
    LOG_DIRECTORY, initializer_logger, close_initializer_logger
)
from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import Image
from plugins.ai_modules.ai_models import ModelsConfig

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


class ProgressTracker:
    """
    A thread-safe progress tracker for image processing with time estimation.
    """

    def __init__(self, total_files, request_id=None):
        self.total_files = total_files
        self.processed_files = 0
        self.successful_files = 0
        self.skipped_files = 0
        self.error_files = 0
        self.start_time = time.time()
        self.request_id = request_id or get_request_id()
        self.lock = threading.Lock()
        self.processing_times = []
        self.last_update_time = time.time()
        self.update_interval = 2.0  # Update every 2 seconds minimum

    def update(self, processing_time, status="processed"):
        """Update progress with thread safety."""
        with self.lock:
            self.processed_files += 1
            self.processing_times.append(processing_time)

            if status == "processed":
                self.successful_files += 1
            elif status == "skipped":
                self.skipped_files += 1
            elif status == "error":
                self.error_files += 1

            # Only show progress updates every few seconds to avoid spam
            current_time = time.time()
            if current_time - self.last_update_time >= self.update_interval or self.processed_files == self.total_files:
                self._show_progress()
                self.last_update_time = current_time

    def _show_progress(self):
        """Display current progress with time estimates."""
        if self.processed_files == 0:
            return

        # Calculate progress percentage
        progress_pct = (self.processed_files / self.total_files) * 100

        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time

        # Calculate average processing time
        avg_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0

        # Calculate ETA
        remaining_files = self.total_files - self.processed_files
        eta_seconds = remaining_files * avg_time if avg_time > 0 else 0
        eta_str = self._format_time(eta_seconds)

        # Calculate processing speed
        files_per_second = self.processed_files / elapsed_time if elapsed_time > 0 else 0

        # Create progress bar
        bar_width = 30
        filled_width = int((progress_pct / 100) * bar_width)
        bar = "█" * filled_width + "░" * (bar_width - filled_width)

        # Format the progress message
        progress_msg = (
            f"Progress: [{bar}] {progress_pct:.1f}% "
            f"({self.processed_files:,}/{self.total_files:,}) | "
            f"Speed: {files_per_second:.1f} files/sec | "
            f"ETA: {eta_str} | "
            f"Success: {self.successful_files} | "
            f"Skipped: {self.skipped_files} | "
            f"Errors: {self.error_files}"
        )

        print(f"\r{progress_msg}", end="", flush=True)

        # Also log to file periodically
        if self.processed_files % 100 == 0 or self.processed_files == self.total_files:
            info_id(progress_msg, self.request_id)

    def _format_time(self, seconds):
        """Format seconds into a readable time string."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def get_summary(self):
        """Get final processing summary."""
        elapsed_time = time.time() - self.start_time
        avg_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0

        return {
            'total_files': self.total_files,
            'processed_files': self.processed_files,
            'successful_files': self.successful_files,
            'skipped_files': self.skipped_files,
            'error_files': self.error_files,
            'elapsed_time': elapsed_time,
            'average_time_per_file': avg_time,
            'files_per_second': self.processed_files / elapsed_time if elapsed_time > 0 else 0
        }


@with_request_id
def estimate_processing_time(image_files, sample_size=20, request_id=None):
    """
    Process a small sample of images to estimate total processing time.
    """
    rid = request_id or get_request_id()

    if len(image_files) == 0:
        return 0, 0

    # Take a random sample for estimation
    sample_size = min(sample_size, len(image_files))
    sample_files = random.sample(image_files, sample_size)

    info_id(f"Processing {sample_size} sample files to estimate total time...", rid)

    sample_times = []
    sample_tracker = ProgressTracker(sample_size, rid)

    print(f"\nEstimating processing time with {sample_size} sample files...")

    # Process sample files
    for i, (root, filename, full_path) in enumerate(sample_files):
        start_time = time.time()

        try:
            # Use a temporary request ID for sample processing
            sample_rid = f"{rid}-sample-{i}"
            result_message, process_time = process_single_image(root, filename, sample_rid)
            sample_times.append(process_time)

            status = "processed" if "Processed:" in result_message else "skipped"
            sample_tracker.update(process_time, status)

        except Exception as e:
            error_time = time.time() - start_time
            sample_times.append(error_time)
            sample_tracker.update(error_time, "error")
            warning_id(f"Sample processing error for {filename}: {e}", rid)

    print()  # New line after progress bar

    # Calculate statistics
    if sample_times:
        avg_time = sum(sample_times) / len(sample_times)
        total_estimated_time = avg_time * len(image_files)

        # Add some buffer for overhead (10%)
        total_estimated_time *= 1.1

        summary = sample_tracker.get_summary()

        info_id(f"Sample processing complete. Average time per file: {avg_time:.2f}s", rid)
        info_id(f"Estimated total time: {sample_tracker._format_time(total_estimated_time)}", rid)
        info_id(
            f"Sample stats - Success: {summary['successful_files']}, Skipped: {summary['skipped_files']}, Errors: {summary['error_files']}",
            rid)

        return avg_time, total_estimated_time
    else:
        warning_id("No valid processing times from sample", rid)
        return 0, 0


@with_request_id
def show_processing_estimate(total_files, avg_time_per_file, request_id=None):
    """
    Display a formatted processing time estimate.
    """
    rid = request_id or get_request_id()

    total_time = avg_time_per_file * total_files

    # Format time estimates
    def format_time_range(seconds):
        # Add some variance for realistic estimates
        min_time = seconds * 0.8
        max_time = seconds * 1.3

        min_str = ProgressTracker(0)._format_time(min_time)
        max_str = ProgressTracker(0)._format_time(max_time)

        return min_str, max_str

    min_time, max_time = format_time_range(total_time)

    print(f"\n{'=' * 60}")
    print(f"📊 PROCESSING TIME ESTIMATE")
    print(f"{'=' * 60}")
    print(f"Total files to process: {total_files:,}")
    print(f"Average time per file: {avg_time_per_file:.2f} seconds")
    print(f"Estimated total time: {min_time} - {max_time}")
    print(f"{'=' * 60}")

    # Show time breakdown
    current_time = datetime.now()
    estimated_completion = current_time + timedelta(seconds=total_time)
    print(f"Started at: {current_time.strftime('%H:%M:%S')}")
    print(f"Estimated completion: {estimated_completion.strftime('%H:%M:%S')}")
    print(f"{'=' * 60}\n")

    info_id(
        f"Processing estimate: {total_files:,} files, {min_time}-{max_time}, completion ~{estimated_completion.strftime('%H:%M:%S')}",
        rid)


@with_request_id
def prompt_model_selection(request_id=None):
    """
    Prompt the user to select the image model to use.
    """
    rid = request_id or get_request_id()
    info_id("Prompting user for model selection", rid)

    print("Select an image model to use:")
    print("1. CLIPModelHandler")
    print("2. NoImageModel (Skip embedding generation)")
    print("3. Custom (Enter the name of another model)")

    while True:
        choice = input("> ").strip()
        if choice == "1":
            debug_id("User selected CLIPModelHandler", rid)
            return "CLIPModelHandler"
        elif choice == "2":
            debug_id("User selected NoImageModel", rid)
            return "NoImageModel"
        else:
            print("Invalid choice. Please select 1, 2, or 3.")


@with_request_id
def set_models(request_id=None):
    """
    Allow the admin to set the image model using the modern ModelsConfig system.
    """
    rid = request_id or get_request_id()
    info_id("Setting models based on user input", rid)

    selected_model = prompt_model_selection(request_id=rid)

    # Use ModelsConfig to set the current image model
    success = ModelsConfig.set_current_image_model(selected_model)

    if success:
        info_id(f"Successfully set current image model to: {selected_model}", rid)
    else:
        error_id(f"Failed to set current image model to: {selected_model}", rid)
        warning_id("Falling back to NoImageModel", rid)
        ModelsConfig.set_current_image_model("NoImageModel")

    # Verify the setting
    current_model = ModelsConfig.get_current_image_model_name()
    info_id(f"Current image model confirmed as: {current_model}", rid)


@with_request_id
def process_single_image(folder_path: str, filename: str, request_id=None):
    """
    Process a single image using the enhanced Image class methods with modern error handling.
    Returns consistent status messages for progress tracking.
    """
    rid = request_id or get_request_id()
    start_time_image = time.time()
    max_retries = 3
    retry_count = 0

    debug_id(f"Starting to process image: {filename}", rid)

    while retry_count < max_retries:
        # Use the context manager for better session management
        try:
            with db_config.main_session() as session:
                source_file_path = os.path.join(folder_path, filename)

                # Load the current image model using ModelsConfig
                model_handler = ModelsConfig.load_image_model()
                debug_id(f"Loaded image model: {type(model_handler).__name__}", rid)

                if model_handler.allowed_file(filename):
                    # Extract the base name without the extension
                    file_base, ext = os.path.splitext(filename)

                    debug_id(f"Processing image: {filename}", rid)

                    # Use the enhanced Image.add_to_db method with request_id
                    new_image = Image.add_to_db(
                        session=session,
                        title=file_base,
                        file_path=source_file_path,
                        description="Auto-generated description",
                        clean_title=True,  # Enable title cleaning
                        request_id=rid
                    )

                    debug_id(f"Successfully processed and stored '{filename}' with ID: {new_image.id}", rid)
                    result_message = f"Processed: {new_image.title}"
                else:
                    debug_id(f"Skipping non-image file: '{filename}'", rid)
                    result_message = f"Skipped: {filename}"

                # Successfully processed, break retry loop
                break

        except Exception as e:
            # Check if it's a database lock error
            error_msg = str(e).lower()
            if "database is locked" in error_msg or "locked" in error_msg:
                retry_count += 1
                backoff_time = random.uniform(0.5, 2.0) * retry_count  # Exponential backoff with jitter
                warning_id(
                    f"Database locked while processing '{filename}'. Retry {retry_count}/{max_retries} after {backoff_time:.2f}s",
                    rid
                )

                # Wait before retrying
                time.sleep(backoff_time)

                # If it's not the last retry, continue to next iteration
                if retry_count < max_retries:
                    continue

            # For non-lock errors or final retry failure
            error_id(f"Failed to process '{filename}': {e}", rid, exc_info=True)
            result_message = f"Error: {filename}"
            break

    processing_time = time.time() - start_time_image
    debug_id(f"Completed processing {filename} in {processing_time:.2f}s", rid)
    return (result_message, processing_time)


@with_request_id
def process_and_store_images(folder_path: str, recursive=True, request_id=None):
    """
    Scans the given folder_path for images and processes them concurrently using multithreading.
    Enhanced with modern session management, progress tracking, and time estimation.

    Args:
        folder_path: Path to the folder to process
        recursive: If True, process all subfolders recursively
        request_id: Optional request ID for tracking
    """
    rid = request_id or get_request_id()

    with log_timed_operation(f"Processing folder: {folder_path} (recursive={recursive})", rid):
        info_id(f"Starting processing for folder: {folder_path} (recursive={recursive})", rid)

        # Ensure the destination folder exists (for database images)
        os.makedirs(DATABASE_PATH_IMAGES_FOLDER, exist_ok=True)

        # Collect all image files (with their full paths)
        info_id("Scanning for image files...", rid)
        image_files = []

        if recursive:
            # Walk through all subdirectories
            for root, dirs, files in os.walk(folder_path):
                for filename in files:
                    full_path = os.path.join(root, filename)
                    # Quick check if it might be an image file using ALLOWED_EXTENSIONS
                    if '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
                        image_files.append((root, filename, full_path))
        else:
            # Only process immediate folder contents
            for filename in os.listdir(folder_path):
                full_path = os.path.join(folder_path, filename)
                if os.path.isfile(full_path) and '.' in filename and filename.rsplit('.', 1)[
                    1].lower() in ALLOWED_EXTENSIONS:
                    image_files.append((folder_path, filename, full_path))

        num_files = len(image_files)
        info_id(f"Found {num_files} potential image files in '{folder_path}' (recursive={recursive})", rid)

        if num_files == 0:
            print("No image files found to process.")
            return

        # Ask user about time estimation
        do_estimation = input("Do you want a processing time estimate? (y/n, default: y): ").strip().lower()

        avg_time_per_file = 0
        if do_estimation != 'n':
            # Get time estimate by processing a small sample
            sample_size = min(20, max(5, num_files // 100))  # Sample 1% but at least 5, max 20
            avg_time_per_file, total_estimated_time = estimate_processing_time(image_files, sample_size, rid)

            if avg_time_per_file > 0:
                show_processing_estimate(num_files, avg_time_per_file, rid)

                # Ask if they still want to proceed after seeing estimate
                continue_choice = input("Do you want to continue with full processing? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    info_id("User chose not to proceed after seeing estimate", rid)
                    return
        else:
            # User declined estimation, ask if they want to proceed
            recursive_str = "and all subfolders " if recursive else ""
            proceed = input(
                f"Proceed to process {num_files} image files in folder '{folder_path}' {recursive_str}? (y/n): "
            ).strip().lower()

            if proceed != "y":
                info_id("User chose not to proceed without estimation", rid)
                return

        # Initialize progress tracker
        progress_tracker = ProgressTracker(num_files, rid)

        # More conservative worker settings - don't let CPU count dictate this
        # since SQLite is the bottleneck, not CPU
        max_workers = min(num_files, os.cpu_count() or 1, 8)  # No more than 8 workers
        info_id(f"Using {max_workers} worker threads for processing", rid)

        # Log connection stats for monitoring
        stats = db_config.get_connection_stats()
        info_id(f"Database connection stats: {stats}", rid)

        # Use smaller batches to reduce concurrent database load
        batch_size = max(1, min(20, num_files // 20))  # Smaller batches

        # Shuffle the image files to distribute large/small images more evenly
        random.shuffle(image_files)

        # Start main processing
        print(f"\n🚀 Starting to process {num_files:,} image files...")
        print("Progress will be shown below:\n")

        processing_start_time = time.time()

        for batch_start in range(0, num_files, batch_size):
            batch_end = min(batch_start + batch_size, num_files)
            batch_files = image_files[batch_start:batch_end]
            batch_num = batch_start // batch_size + 1
            total_batches = (num_files + batch_size - 1) // batch_size

            # Use a new ThreadPoolExecutor for each batch to ensure clean resources
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks for this batch with unique request IDs
                futures = {}
                for root, filename, full_path in batch_files:
                    # Create a unique request ID for each file processing
                    file_request_id = f"{rid}-{filename[:8]}"
                    future = executor.submit(process_single_image, root, filename, file_request_id)
                    futures[future] = (filename, root)

                # Process completed futures
                for future in as_completed(futures):
                    filename, root = futures[future]
                    try:
                        result_message, process_time = future.result()

                        # Determine status for progress tracking
                        if result_message.startswith("Processed:"):
                            status = "processed"
                        elif result_message.startswith("Skipped:"):
                            status = "skipped"
                        else:
                            status = "error"

                        # Update progress tracker
                        progress_tracker.update(process_time, status)

                    except Exception as e:
                        error_id(f"Error processing file '{filename}': {e}", rid, exc_info=True)
                        progress_tracker.update(0, "error")

            # Wait a short time between batches to let any lingering transactions complete
            time.sleep(0.5)

            # Log connection stats after each batch for monitoring (but not to console)
            stats = db_config.get_connection_stats()
            debug_id(f"Database connection stats after batch {batch_num}: {stats}", rid)

        # Final progress update and summary
        print("\n" + "=" * 80)

        # Get final summary
        summary = progress_tracker.get_summary()
        processing_time = time.time() - processing_start_time

        print(f"🎉 PROCESSING COMPLETE!")
        print(f"📊 Final Summary:")
        print(f"   Total files: {summary['total_files']:,}")
        print(f"   Successfully processed: {summary['successful_files']:,}")
        print(f"   Skipped files: {summary['skipped_files']:,}")
        print(f"   Error files: {summary['error_files']:,}")
        print(f"   Total time: {progress_tracker._format_time(processing_time)}")
        print(f"   Average time per file: {summary['average_time_per_file']:.2f}s")
        print(f"   Processing speed: {summary['files_per_second']:.1f} files/sec")

        if avg_time_per_file > 0:
            accuracy = abs(summary['average_time_per_file'] - avg_time_per_file) / avg_time_per_file * 100
            print(f"   Estimate accuracy: {100 - accuracy:.1f}% (predicted {avg_time_per_file:.2f}s/file)")

        print("=" * 80)

        # Log final summary
        info_id(
            f"Processing complete - {summary['successful_files']:,} processed, {summary['skipped_files']:,} skipped, {summary['error_files']:,} errors in {progress_tracker._format_time(processing_time)}",
            rid)


@with_request_id
def main(request_id=None):
    """
    Main function for image processing setup with enhanced error handling and logging.
    """
    rid = request_id or set_request_id()

    with log_timed_operation("EMTACDB Image Setup", rid):
        info_id("=== Starting EMTACDB Image Setup ===", rid)

        # Log database configuration
        stats = db_config.get_connection_stats()
        info_id(f"DatabaseConfig settings: {stats}", rid)

        # Ensure ModelsConfig table is initialized
        try:
            from plugins.ai_modules.ai_models import initialize_models_config
            if initialize_models_config():
                info_id("Models configuration initialized successfully", rid)
            else:
                warning_id("Models configuration initialization had issues", rid)
        except Exception as e:
            error_id(f"Failed to initialize models configuration: {e}", rid)

        # Set up models using the modern system
        set_models(request_id=rid)

        # Verify the selected model
        current_model = ModelsConfig.get_current_image_model_name()
        debug_id(f"Confirmed current image model: {current_model}", rid)

        # Handle folder input
        if len(sys.argv) > 1:
            folders = sys.argv[1:]
            debug_id(f"Received CLI arguments for folders: {folders}", rid)
        else:
            folders = []
            info_id("Enter folder paths containing images. Blank line finishes input", rid)
            while True:
                folder_path = input("> ").strip().strip('"').strip("'")
                if not folder_path:
                    debug_id("No more folders entered by the user", rid)
                    break
                if not os.path.isdir(folder_path):
                    warning_id(f"Invalid directory: {folder_path}", rid)
                else:
                    folders.append(folder_path)
                    debug_id(f"Added folder to processing list: {folder_path}", rid)

        if not folders:
            error_id("No valid folders provided. Exiting setup", rid)
            return

        # Ask user about recursive processing
        recursive_choice = input("Process subfolders recursively? (y/n, default: y): ").strip().lower()
        recursive = recursive_choice != 'n'  # Default to True unless explicitly 'n'
        info_id(f"Recursive processing: {'enabled' if recursive else 'disabled'}", rid)

        # Process each folder
        for folder in folders:
            info_id(f"\n--- Processing folder: {folder} ---", rid)
            try:
                process_and_store_images(folder, recursive=recursive, request_id=rid)
            except Exception as e:
                error_id(f"Error processing folder {folder}: {e}", rid, exc_info=True)

        info_id("=== EMTACDB Image Setup Complete! ===", rid)


if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure all sessions are closed
        try:
            db_config.get_main_session_registry().remove()
        except Exception as e:
            logger.error(f"Error closing session registry: {e}")

        try:
            close_initializer_logger()
        except Exception as e:
            logger.error(f"Error closing initializer logger: {e}")