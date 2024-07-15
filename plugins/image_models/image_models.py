import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import logging
from PIL import Image as PILImage, ImageFile
from abc import ABC, abstractmethod
import torch  # Add this import

# Import config variables from __init__.py
from plugins.image_models import DATABASE_URL, ALLOWED_EXTENSIONS

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

# SQLAlchemy setup
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

# Define the BaseImageModelHandler interface
class BaseImageModelHandler(ABC):
    @abstractmethod
    def allowed_file(self, filename):
        pass

    @abstractmethod
    def preprocess_image(self, image):
        pass

    @abstractmethod
    def get_image_embedding(self, image):
        pass

    @abstractmethod
    def is_valid_image(self, image):
        pass

    def store_image_metadata(self, session, title, description, file_path, embedding, model_name):
        from emtacdb_fts import Image, ImageEmbedding  # Import here to avoid circular import
        
        # Create Image entry
        image = Image(title=title, description=description, file_path=file_path)
        session.add(image)
        session.commit()

        # Create ImageEmbedding entry
        image_embedding = ImageEmbedding(image_id=image.id, model_name=model_name, model_embedding=embedding.tobytes())
        session.add(image_embedding)
        session.commit()

        logger.info(f"Stored image metadata and embedding for {file_path} using {model_name}.")

# Implement the NoImageModel handler
class NoImageModel(BaseImageModelHandler):
    def allowed_file(self, filename):
        return False

    def preprocess_image(self, image):
        return None

    def get_image_embedding(self, image):
        return None

    def is_valid_image(self, image):
        return False

    def store_image_metadata(self, session, title, description, file_path, embedding, model_name):
        logger.info("No image model selected, not storing image metadata.")

# Implement the CLIP model handler
class CLIPModelHandler(BaseImageModelHandler):
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def preprocess_image(self, image):
        image = image.resize((224, 224))
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        return inputs

    def get_image_embedding(self, image):
        try:
            inputs = self.preprocess_image(image)
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            return outputs.numpy().flatten()
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None

    def is_valid_image(self, image):
        width, height = image.size
        logger.info(f"Image dimensions: width={width}, height={height}")
        if width < 100 or height < 100:
            return False
        shortest_side = min(width, height)
        longest_side = max(width, height)
        return longest_side <= 5 * shortest_side
