#plugins/image_modules/image_handler.py

import os
import imghdr
from PIL import Image, UnidentifiedImageError
from plugins.image_modules import CLIPModelHandler, NoImageModel
from modules.emtacdb.emtacdb_fts import Session, load_image_model_config_from_db
from modules.configuration.log_config import logger

class ImageHandler:
    def __init__(self):
        self.model_handlers = {
            "clip": CLIPModelHandler(),
            "no_model": NoImageModel()
        }
        self.Session = Session
        self.current_model = load_image_model_config_from_db()

    def allowed_file(self, filename, model_name=None):
        model_name = model_name or self.current_model
        return self.model_handlers[model_name].allowed_file(filename)

    def preprocess_image(self, image, model_name=None):
        model_name = model_name or self.current_model
        return self.model_handlers[model_name].preprocess_image(image)

    def get_image_embedding(self, image, model_name=None):
        model_name = model_name or self.current_model
        return self.model_handlers[model_name].get_image_embedding(image)

    def is_valid_image(self, image, model_name=None):
        model_name = model_name or self.current_model
        return self.model_handlers[model_name].is_valid_image(image)

    def store_image_metadata(self, session, title, description, file_path, embedding, model_name=None):
        model_name = model_name or self.current_model
        self.model_handlers[model_name].store_image_metadata(session, title, description, file_path, embedding, model_name)

    def load_image_safe(self, file_path):
        """Safely loads an image with error handling."""
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return None

        if os.path.getsize(file_path) == 0:
            logger.error(f"File is empty: {file_path}")
            return None

        # Verify file type using imghdr first
        file_type = imghdr.what(file_path)
        allowed_types = {"jpeg", "png", "jpg"}
        if file_type not in allowed_types:
            logger.warning(f"Invalid image type detected: {file_path}. Type: {file_type}")
            return None

        try:
            with Image.open(file_path) as img:
                img_format = img.format.lower()
                if img_format not in allowed_types:
                    logger.warning(f"Image format mismatch: {file_path}. Detected format: {img_format}")
                    return None
                return img.convert("RGB")
        except UnidentifiedImageError:
            logger.error(f"Cannot identify image file: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error while opening image {file_path}: {e}")
            return None
