# plugins/image_modules/image_models.py
import os
import logging
import time  # ← This was missing!
from typing import Dict, Any, Optional
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import sys
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import logging
from PIL import ImageFile
from abc import ABC, abstractmethod
import torch
# Import config variables from config.py
from modules.configuration.config import DATABASE_URL, ALLOWED_EXTENSIONS
from typing import Dict, Any, Optional
from modules.configuration.config import BASE_DIR  # Import BASE_DIR

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
        from modules.emtacdb.emtacdb_fts import Image, ImageEmbedding
        # Ensure file_path is relative
        if os.path.isabs(file_path):
            relative_file_path = os.path.relpath(file_path, BASE_DIR)
            logger.debug(f"Converted absolute file path '{file_path}' to relative path '{relative_file_path}'.")
        else:
            relative_file_path = file_path
            logger.debug(f"Using existing relative file path '{relative_file_path}'.")

        # Create Image entry with relative path
        image = Image(title=title, description=description, file_path=relative_file_path)
        session.add(image)
        session.commit()

        # Create ImageEmbedding entry
        image_embedding = ImageEmbedding(image_id=image.id, model_name=model_name, model_embedding=embedding.tobytes())
        session.add(image_embedding)
        session.commit()

        logger.info(f"Stored image metadata and embedding for '{relative_file_path}' using '{model_name}'.")

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


class CLIPModelHandler:
    """Optimized CLIP model handler with offline capabilities and caching."""

    # Class-level cache to persist models across instances
    _model_cache = {}
    _processor_cache = {}

    def __init__(self):
        self.model_name = "CLIPModelHandler"
        self.clip_model_id = "openai/clip-vit-base-patch32"
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing CLIP model handler on device: {self.device}")

        # Load model immediately (with caching)
        self._load_model()

    def _load_model(self):
        """Load CLIP model with caching to avoid repeated loading."""
        cache_key = f"{self.clip_model_id}_{self.device}"

        # Check if model is already cached
        if cache_key in self._model_cache:
            logger.info("Using cached CLIP model - INSTANT LOAD!")
            self.model = self._model_cache[cache_key]
            self.processor = self._processor_cache[cache_key]
            return

        start_time = time.time()
        logger.info(f"Loading CLIP model: {self.clip_model_id}")

        try:
            # Load model and processor (offline mode prevents network calls)
            self.model = CLIPModel.from_pretrained(
                self.clip_model_id,
                local_files_only=True,  # Force local files only
                cache_dir=None  # Use default cache
            ).to(self.device)

            self.processor = CLIPProcessor.from_pretrained(
                self.clip_model_id,
                local_files_only=True,  # Force local files only
                cache_dir=None  # Use default cache
            )

            # Cache the loaded model and processor
            self._model_cache[cache_key] = self.model
            self._processor_cache[cache_key] = self.processor

            load_time = time.time() - start_time
            logger.info(f"✅ Successfully loaded and cached CLIP model in {load_time:.2f}s")

        except Exception as e:
            logger.error(f"Error loading CLIP model: {e}")
            # Fallback: try without local_files_only if offline loading fails
            try:
                logger.warning("Attempting fallback model loading with network access")
                self.model = CLIPModel.from_pretrained(self.clip_model_id).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(self.clip_model_id)

                # Cache the fallback model
                self._model_cache[cache_key] = self.model
                self._processor_cache[cache_key] = self.processor

                load_time = time.time() - start_time
                logger.info(f"Fallback model loaded in {load_time:.2f}s")
            except Exception as fallback_error:
                logger.error(f"Failed to load CLIP model even with fallback: {fallback_error}")
                raise

    def allowed_file(self, filename):
        """Check if file extension is allowed."""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def preprocess_image(self, image):
        """Preprocess image for CLIP model."""
        if not self.processor:
            raise RuntimeError("CLIP processor not loaded")

        # Resize image while maintaining aspect ratio
        image = image.resize((224, 224))
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        return inputs.to(self.device)

    def get_image_embedding(self, image):
        """Get CLIP embedding for an image."""
        try:
            if not self.model or not self.processor:
                logger.error("CLIP model or processor not loaded")
                return None

            inputs = self.preprocess_image(image)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize the features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Convert to numpy array for storage
            embedding = image_features.cpu().numpy().flatten()
            logger.info(f"Generated CLIP embedding (shape: {embedding.shape})")
            return embedding

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None

    def is_valid_image(self, image):
        """Validate if image meets requirements."""
        try:
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

        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            return False

    def compare_images(self, image1_path: str, image2_path: str) -> dict:
        """Compare two images using CLIP embeddings."""
        try:
            logger.info(f"Comparing images with CLIP: {image1_path} vs {image2_path}")

            if not self.model or not self.processor:
                return {
                    "similarity": 0.0,
                    "image1": image1_path,
                    "image2": image2_path,
                    "model": self.model_name,
                    "error": "Model not loaded",
                    "message": "Comparison failed"
                }

            # Load both images
            image1 = Image.open(image1_path).convert('RGB')
            image2 = Image.open(image2_path).convert('RGB')

            # Process both images
            inputs = self.processor(images=[image1, image2], return_tensors="pt").to(self.device)

            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Calculate cosine similarity
                similarity = torch.cosine_similarity(
                    image_features[0:1], image_features[1:2], dim=1
                ).item()

            logger.info(f"Image comparison similarity: {similarity:.4f}")

            return {
                "similarity": float(similarity),
                "image1": image1_path,
                "image2": image2_path,
                "model": self.model_name,
                "message": "Comparison completed successfully"
            }

        except Exception as e:
            logger.error(f"Error comparing images with CLIP: {e}")
            return {
                "similarity": 0.0,
                "image1": image1_path,
                "image2": image2_path,
                "model": self.model_name,
                "error": str(e),
                "message": "Comparison failed"
            }

    @classmethod
    def preload_model(cls) -> bool:
        """Class method to preload model during application startup."""
        try:
            logger.info("🚀 Preloading CLIP model for faster subsequent access...")
            start_time = time.time()

            # Create temporary instance to trigger model loading
            temp_handler = cls()

            preload_time = time.time() - start_time
            logger.info(f"✅ CLIP model preloaded successfully in {preload_time:.2f}s")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to preload CLIP model: {e}")
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get information about the model cache."""
        return {
            "models_cached": len(self._model_cache),
            "processors_cached": len(self._processor_cache),
            "cache_keys": list(self._model_cache.keys()),
            "model_loaded": self.model is not None,
            "processor_loaded": self.processor is not None,
            "device": str(self.device),
            "offline_mode": os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "model_id": self.clip_model_id,
            "device": str(self.device),
            "model_loaded": self.model is not None,
            "processor_loaded": self.processor is not None,
            "cache_size": len(self._model_cache),
            "offline_mode": os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
        }


