# image_modules.py
import sys
from transformers import CLIPProcessor, CLIPModel
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import logging
from PIL import ImageFile
from abc import ABC, abstractmethod
import torch

# Import config variables from config.py
from modules.configuration.config import DATABASE_URL, ALLOWED_EXTENSIONS

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

# SQLAlchemy setup
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

# Function to dynamically import and instantiate the correct model handler
def get_image_model_handler(model_name):
    module = sys.modules[__name__]
    try:
        model_class = getattr(module, model_name)
        if issubclass(model_class, BaseImageModelHandler):
            return model_class()
        else:
            raise ValueError(f"{model_name} is not a subclass of BaseImageModelHandler")
    except AttributeError:
        logger.error(f"{model_name} not found in {__name__}")
        return NoImageModel()


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
        from modules.emtacdb.emtacdb_fts import Image, ImageEmbedding  # Import here to avoid circular import
        
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

        # Define minimum and maximum dimensions
        min_dimension = 100  # Minimum acceptable dimension
        max_dimension = 5000  # Maximum acceptable dimension

        if width < min_dimension or height < min_dimension:
            logger.info(f"Image is too small: width={width}, height={height}")
            return False
        if width > max_dimension or height > max_dimension:
            logger.info(f"Image is too large: width={width}, height={height}")
            return False

        # Define acceptable aspect ratio range
        min_aspect_ratio = 1 / 5  # Minimum aspect ratio (height/width)
        max_aspect_ratio = 5  # Maximum aspect ratio (width/height)

        aspect_ratio = width / height
        if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
            logger.info(f"Image aspect ratio {aspect_ratio} is outside the acceptable range. "
                        f"Min aspect ratio: {min_aspect_ratio}, Max aspect ratio: {max_aspect_ratio}")
            return False

        return True

