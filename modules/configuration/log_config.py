# modules/configuration/log_config.py
import logging
import sys
import os
from logging.handlers import RotatingFileHandler
import gzip
import shutil
from datetime import datetime, timedelta
import uuid
import time
import threading
from flask import request, g, has_request_context
import functools

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

# NEW CODE FOR UUID-BASED REQUEST TRACKING
# =======================================

# Thread-local storage for request IDs outside of Flask context
_local = threading.local()

def with_request_id(func):
    """
    Decorator that adds request ID tracking to a function.
    Creates a new request ID if one doesn't exist.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        request_id = get_request_id()
        start_time = time.time()

        debug_id(f"Starting {func.__name__}", request_id)
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            debug_id(f"Completed {func.__name__} in {end_time - start_time:.3f}s", request_id)
            return result
        except Exception as e:
            end_time = time.time()
            error_id(f"Error in {func.__name__} after {end_time - start_time:.3f}s: {e}", request_id)
            raise

    return wrapper


def get_request_id():
    """
    Get the current request ID from Flask context if available,
    or from thread-local storage as a fallback.
    If neither exists, generate a new one.
    """
    # First try Flask request context
    if has_request_context() and hasattr(g, 'request_id'):
        return g.request_id

    # Then try thread-local storage
    if hasattr(_local, 'request_id'):
        return _local.request_id

    # If no request ID exists, generate a new one
    request_id = str(uuid.uuid4())[:8]
    _local.request_id = request_id
    return request_id


def set_request_id(request_id=None):
    """
    Set a request ID in thread-local storage.
    If no request_id is provided, generate a new one.
    Returns the request ID that was set.
    """
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]

    _local.request_id = request_id
    return request_id


def clear_request_id():
    """Clear the request ID from thread-local storage."""
    if hasattr(_local, 'request_id'):
        delattr(_local, 'request_id')

# Enhanced logging functions that automatically include request ID
# These don't replace the standard logger methods, they're additional
def log_with_id(level, message, *args, request_id=None, **kwargs):
    """Log a message with an included request ID."""
    # Determine the request ID
    rid = request_id or get_request_id()

    # First apply any %-format args to the message
    try:
        msg = message % args if args else message
    except Exception:
        # Fallback if formatting fails
        msg = message

    # Prefix with the request ID
    final = f"[REQ-{rid}] {msg}"

    # Dispatch to the logger at the proper level
    if level == logging.DEBUG:
        logger.debug(final, **kwargs)
    elif level == logging.INFO:
        logger.info(final, **kwargs)
    elif level == logging.WARNING:
        logger.warning(final, **kwargs)
    elif level == logging.ERROR:
        logger.error(final, **kwargs)
    elif level == logging.CRITICAL:
        logger.critical(final, **kwargs)

# Convenience functions that match the logger interface
def debug_id(message, request_id=None, *args, **kwargs):
    log_with_id(logging.DEBUG, message, request_id, *args, **kwargs)

def info_id(message, request_id=None, *args, **kwargs):
    log_with_id(logging.INFO, message, request_id, *args, **kwargs)


def warning_id(message, request_id=None, *args, **kwargs):
    log_with_id(logging.WARNING, message, request_id, *args, **kwargs)


def error_id(message, request_id=None, *args, **kwargs):
    log_with_id(logging.ERROR, message, request_id, *args, **kwargs)


def critical_id(message, request_id=None, *args, **kwargs):
    log_with_id(logging.CRITICAL, message, request_id, *args, **kwargs)

# Flask middleware for request ID tracking
def request_id_middleware(app):
    """
    Add request ID middleware to a Flask app.
    This sets a unique request ID for each HTTP request.
    """

    @app.before_request
    def before_request():
        # Generate a unique request ID and store it in Flask's g object
        g.request_id = str(uuid.uuid4())[:8]
        g.request_start_time = time.time()

        # Also store in thread-local for non-Flask code
        _local.request_id = g.request_id

        info_id(f"Processing request: {request.method} {request.path}")

    @app.after_request
    def after_request(response):
        if hasattr(g, 'request_start_time'):
            duration = time.time() - g.request_start_time
            info_id(f"Request completed in {duration:.3f}s with status {response.status_code}")

        # Clear the thread-local request ID
        clear_request_id()

        return response

    @app.teardown_request
    def teardown_request(exception=None):
        if exception:
            error_id(f"Request failed with exception: {str(exception)}")

        # Ensure thread-local storage is cleared even on exceptions
        clear_request_id()

    return app


# Helper for timing operations
def log_timed_operation(operation_name, request_id=None):
    """Context manager for timing and logging operations."""

    class TimedOperationContext:
        def __init__(self, operation_name, request_id):
            self.operation_name = operation_name
            self.request_id = request_id if request_id else get_request_id()

        def __enter__(self):
            self.start_time = time.time()
            debug_id(f"Starting operation: {self.operation_name}", self.request_id)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = time.time() - self.start_time
            if exc_type:
                error_id(f"Operation {self.operation_name} failed after {duration:.3f}s: {str(exc_val)}",
                         self.request_id)
            else:
                debug_id(f"Operation {self.operation_name} completed in {duration:.3f}s",
                         self.request_id)

    return TimedOperationContext(operation_name, request_id)

# EXISTING LOG ROTATION CODE - UNCHANGED
# =====================================

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