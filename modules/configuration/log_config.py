import logging
import sys
import os
from logging.handlers import RotatingFileHandler
import gzip
import shutil
from datetime import datetime, timedelta

# Determine the root directory based on whether the code is frozen (e.g., PyInstaller .exe)
if getattr(sys, 'frozen', False):  # Running as an executable
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

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


def compress_and_backup_logs(log_directory, backup_directory):
    """
    Consolidate and compress log files older than 14 days into biweekly backup files.

    This function groups log files (that are not already compressed) by a biweekly period.
    For each biweekly group, the log files are concatenated into a single gzip file, and
    then the original files are removed.
    """
    now = datetime.now()
    biweekly_logs = {}

    # Group log files older than 14 days by their biweekly period.
    for file_name in os.listdir(log_directory):
        file_path = os.path.join(log_directory, file_name)
        # Skip directories and files that are already compressed.
        if os.path.isdir(file_path) or file_name.endswith('.gz'):
            continue

        file_modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        if now - file_modified_time > timedelta(days=14):
            # Calculate the biweekly period based on the day of the year.
            year = file_modified_time.year
            day_of_year = file_modified_time.timetuple().tm_yday
            biweek = (day_of_year - 1) // 14 + 1  # Determines the biweekly period number
            biweek_key = f"{year}-BW{biweek:02d}"
            biweekly_logs.setdefault(biweek_key, []).append(file_path)
            logger.debug(f"Grouping file {file_path} under biweekly period {biweek_key}")

    # For each biweekly group, consolidate logs into a single compressed backup file.
    for biweek_key, files in biweekly_logs.items():
        backup_file_path = os.path.join(backup_directory, f"backup_{biweek_key}.gz")
        logger.info(f"Creating biweekly backup: {backup_file_path} with {len(files)} file(s)")
        with gzip.open(backup_file_path, 'wb') as f_out:
            for file_path in files:
                with open(file_path, 'rb') as f_in:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(file_path)
                logger.info(f"Compressed and removed: {file_path}")


def delete_old_backups(backup_directory, retention_weeks=2):
    """
    Delete backup files older than the specified retention period (in weeks).

    :param backup_directory: Path to the backup directory.
    :param retention_weeks: Number of weeks to retain backups (default is 2).
    """
    now = datetime.now()
    for file_name in os.listdir(backup_directory):
        file_path = os.path.join(backup_directory, file_name)
        # Skip directories
        if os.path.isdir(file_path):
            continue
        # Process only compressed backup files.
        if file_name.endswith('.gz'):
            file_modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if now - file_modified_time > timedelta(weeks=retention_weeks):
                os.remove(file_path)
                logger.info(f"Deleted old backup: {file_path}")


def initial_log_cleanup():
    """
    Perform initial cleanup by consolidating log files into biweekly backups
    and deleting backups older than 2 weeks.
    """
    logger.info("Starting initial log cleanup...")
    compress_and_backup_logs(log_directory, log_backup_directory)
    delete_old_backups(log_backup_directory, retention_weeks=2)
    logger.info("Initial log cleanup completed.")
