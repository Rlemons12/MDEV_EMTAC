#modules/configuration/log_config.py

import logging
import sys
import os
# Determine the root directory based on whether the code is frozen (e.g., PyInstaller .exe)
if getattr(sys, 'frozen', False):  # Check if running as an executable
    BASE_DIR = os.path.dirname(sys.executable)  # Use the directory of the executable
else:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # Use the AuMaintdb root directory

# Add the current directory to the Python module search path for flexibility
sys.path.append(BASE_DIR)

# Configure logging
logger = logging.getLogger('app_logger')
logger.setLevel(logging.DEBUG)

# Create a file handler
file_handler = logging.FileHandler('../../app.log')
file_handler.setLevel(logging.DEBUG)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

# Remove the default StreamHandler to disable logging to the terminal
logger.propagate = False
