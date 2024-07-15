import logging

# Configure logging
logger = logging.getLogger('app_logger')
logger.setLevel(logging.DEBUG)

# Create a file handler
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

# Remove the default StreamHandler to disable logging to the terminal
logger.propagate = False
