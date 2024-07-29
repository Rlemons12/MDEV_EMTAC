import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import DATABASE_URL, ALLOWED_EXTENSIONS

# Import the required classes
from .image_models import BaseImageModelHandler, NoImageModel, CLIPModelHandler
from .image_handler import ImageHandler

__all__ = ['BaseImageModelHandler', 'NoImageModel', 'CLIPModelHandler', 'ImageHandler']
