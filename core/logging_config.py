import logging
import sys

# Logger setup: console-only (cloud friendly)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Define a simple formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Console handler (stdout) for cloud logging
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

# Remove any pre-existing handlers to avoid duplicates
if logger.hasHandlers():
    logger.handlers.clear()

# Add only console handler
logger.addHandler(console_handler)