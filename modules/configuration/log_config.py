import logging
import sys
import os
from logging.handlers import RotatingFileHandler
import gzip
import shutil
from datetime import datetime, timedelta

# Determine the root directory based on whether the code is frozen (e.g., PyInstaller .exe)
if getattr(sys, 'frozen', False):  # Check if running as an executable
    BASE_DIR = os.path.dirname(sys.executable)  # Use the directory of the executable
else:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # Use the AuMaintdb root directory

# Add the current directory to the Python module search path for flexibility
sys.path.append(BASE_DIR)

# Configure logging
logger = logging.getLogger('ematac_logger')
logger.setLevel(logging.DEBUG)

# Ensure the log directory exists
log_directory = os.path.join(BASE_DIR, 'logs')
log_backup_directory = os.path.join(BASE_DIR, 'log_backup')
os.makedirs(log_directory, exist_ok=True)
os.makedirs(log_backup_directory, exist_ok=True)

# Create a RotatingFileHandler
file_handler = RotatingFileHandler(
    os.path.join(log_directory, 'app.log'),
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=5
)
file_handler.setLevel(logging.DEBUG)

# Create a StreamHandler (console)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

# Create a formatter and set it for both handlers
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s'
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add both handlers to the logger if they aren't already added
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Optionally, prevent log messages from being propagated to the root logger
logger.propagate = False


# Function to compress and move logs to the backup folder
def compress_and_backup_logs(log_directory, backup_directory):
    """
    Compress and move logs older than 24 hours to the backup directory.
    """
    now = datetime.now()
    for file_name in os.listdir(log_directory):
        file_path = os.path.join(log_directory, file_name)

        # Skip directories
        if os.path.isdir(file_path):
            continue

        # Skip already compressed files
        if file_name.endswith('.gz'):
            continue

        # Compress logs if they are older than 24 hours
        file_modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        if now - file_modified_time > timedelta(hours=24):
            compressed_file_path = os.path.join(backup_directory, file_name + '.gz')
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_file_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(file_path)  # Remove the original file
            logger.info(f"Compressed and moved to backup: {file_path}")


# Function to delete old backups
def delete_old_backups(backup_directory, retention_period=3):
    """
    Delete backups older than the specified retention period.
    :param backup_directory: Path to the backup directory.
    :param retention_period: Retention period in days.
    """
    now = datetime.now()
    for file_name in os.listdir(backup_directory):
        file_path = os.path.join(backup_directory, file_name)

        # Skip directories
        if os.path.isdir(file_path):
            continue

        # Only process compressed log files
        if file_name.endswith('.gz'):
            file_modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if now - file_modified_time > timedelta(days=retention_period):
                os.remove(file_path)
                logger.info(f"Deleted old backup: {file_path}")


# Initial cleanup at startup
def initial_log_cleanup():
    """
    Perform initial cleanup by compressing logs and deleting old backups.
    """
    logger.info("Starting initial log cleanup...")
    compress_and_backup_logs(log_directory, log_backup_directory)
    delete_old_backups(log_backup_directory, retention_period=3)
    logger.info("Initial log cleanup completed.")



