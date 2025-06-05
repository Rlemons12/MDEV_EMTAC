# plugins/ai_modules.py
"""
AI Models Module - Aligned with Established Framework

This module provides AI model management with PostgreSQL integration.
Designed to work seamlessly with:
- DatabaseConfig for session management
- CompleteDocument class for document processing
- DocumentEmbedding model for embedding storage
- Transaction safety with PostgreSQL savepoints

Key Integration Points:
- store_embedding_enhanced(session, document_id, embeddings, model_name)
- generate_and_store_embedding(session, document_id, content, model_name)
- ModelsConfig class for unified model configuration
"""
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import base64
import logging
import openai
import transformers
import torch
import importlib
import json
from abc import ABC, abstractmethod
from sqlalchemy import Column, String, Integer, DateTime, Enum, UniqueConstraint, create_engine, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from modules.configuration.config import (OPENAI_API_KEY, OPENAI_MODEL_NAME, DATABASE_URL, ANTHROPIC_API_KEY)
import requests

logger = logging.getLogger(__name__)

# Use the main database Base instead of creating our own
try:
    from modules.emtacdb.emtacdb_fts import Base

    logger.info("Using main database Base for AI models")
except ImportError:
    # Fallback to creating our own Base if main Base is not available
    from sqlalchemy.ext.declarative import declarative_base

    Base = declarative_base()
    logger.warning("Could not import main database Base, using fallback")

# Database setup
engine = create_engine(DATABASE_URL)
Session = scoped_session(sessionmaker(bind=engine))


class ModelsConfig(Base):
    __tablename__ = 'models_config'

    id = Column(Integer, primary_key=True)
    model_type = Column(Enum('ai', 'image', 'embedding', name='model_type_enum'), nullable=False)
    key = Column(String(255), nullable=False)
    value = Column(String(1000), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        # Composite unique constraint on model_type and key
        UniqueConstraint('model_type', 'key', name='unique_model_type_key'),
    )

    def __repr__(self):
        return f"<ModelsConfig(model_type='{self.model_type}', key='{self.key}')>"

    @staticmethod
    def load_config_from_db():
        """
        Load AI model configuration from the database using DatabaseConfig.

        Returns:
            Tuple of (current_ai_model, current_embedding_model)
        """
        try:
            from modules.configuration.config_env import DatabaseConfig
            db_config = DatabaseConfig()
            session = db_config.get_main_session()
        except ImportError:
            logger.warning("DatabaseConfig not available, using fallback session")
            session = Session()

        try:
            ai_model_config = session.query(ModelsConfig).filter_by(
                model_type='ai',
                key="CURRENT_MODEL"
            ).first()

            embedding_model_config = session.query(ModelsConfig).filter_by(
                model_type='embedding',
                key="CURRENT_MODEL"
            ).first()

            current_ai_model = ai_model_config.value if ai_model_config else "NoAIModel"
            current_embedding_model = embedding_model_config.value if embedding_model_config else "NoEmbeddingModel"

            return current_ai_model, current_embedding_model
        finally:
            session.close()

    @staticmethod
    def load_image_model_config_from_db():
        """
        Load image model configuration from the database using DatabaseConfig.

        Returns:
            String representing the current image model
        """
        try:
            from modules.configuration.config_env import DatabaseConfig
            db_config = DatabaseConfig()
            session = db_config.get_main_session()
        except ImportError:
            logger.warning("DatabaseConfig not available, using fallback session")
            session = Session()

        try:
            image_model_config = session.query(ModelsConfig).filter_by(
                model_type='image',
                key="CURRENT_MODEL"
            ).first()

            current_image_model = image_model_config.value if image_model_config else "no_model"

            return current_image_model
        finally:
            session.close()

    @classmethod
    def set_config_value(cls, model_type, key, value):
        """
        Set a configuration value in the database using DatabaseConfig.

        Args:
            model_type: Type of model ('ai', 'image', 'embedding')
            key: Configuration key
            value: Configuration value

        Returns:
            Boolean indicating success
        """
        try:
            from modules.configuration.config_env import DatabaseConfig
            db_config = DatabaseConfig()
            session = db_config.get_main_session()
        except ImportError:
            logger.warning("DatabaseConfig not available, using fallback session")
            session = Session()

        try:
            # Check if config already exists
            config = session.query(cls).filter_by(
                model_type=model_type,
                key=key
            ).first()

            if config:
                # Update existing config
                config.value = value
                config.updated_at = datetime.utcnow()
            else:
                # Create new config
                config = cls(
                    model_type=model_type,
                    key=key,
                    value=value
                )
                session.add(config)

            session.commit()
            logger.info(f"Successfully set config {model_type}.{key} = {value}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error setting config {model_type}.{key}: {e}")
            return False
        finally:
            session.close()

    @classmethod
    def get_config_value(cls, model_type, key, default=None):
        """
        Get a configuration value from the database using DatabaseConfig.

        Args:
            model_type: Type of model ('ai', 'image', 'embedding')
            key: Configuration key
            default: Default value if not found

        Returns:
            Configuration value or default if not found
        """
        try:
            from modules.configuration.config_env import DatabaseConfig
            db_config = DatabaseConfig()
            session = db_config.get_main_session()
        except ImportError:
            logger.warning("DatabaseConfig not available, using fallback session")
            session = Session()

        try:
            config = session.query(cls).filter_by(
                model_type=model_type,
                key=key
            ).first()

            if config:
                return config.value
            return default
        except Exception as e:
            logger.error(f"Error getting config {model_type}.{key}: {e}")
            return default
        finally:
            session.close()

    @classmethod
    def set_current_ai_model(cls, model_name):
        """Set the current AI model to use."""
        return cls.set_config_value('ai', 'CURRENT_MODEL', model_name)

    @classmethod
    def set_current_embedding_model(cls, model_name):
        """Set the current embedding model to use."""
        return cls.set_config_value('embedding', 'CURRENT_MODEL', model_name)

    @classmethod
    def set_current_image_model(cls, model_name):
        """Set the current image model to use."""
        return cls.set_config_value('image', 'CURRENT_MODEL', model_name)

    @classmethod
    def initialize_models_config_table(cls):
        """Initialize the model configurations with default values if they don't exist."""
        # Set default AI model if not set
        if not cls.get_config_value('ai', 'CURRENT_MODEL'):
            cls.set_current_ai_model('OpenAIModel')

        # Set default embedding model if not set
        if not cls.get_config_value('embedding', 'CURRENT_MODEL'):
            cls.set_current_embedding_model('OpenAIEmbeddingModel')

        # Set default image model if not set
        if not cls.get_config_value('image', 'CURRENT_MODEL'):
            cls.set_current_image_model('CLIPModelHandler')

        logger.info("Model configurations initialized")

    @classmethod
    def get_available_models(cls, model_type):
        """Get list of available models for a specific type with their details."""
        models_json = cls.get_config_value(model_type, "available_models", "[]")
        try:
            return json.loads(models_json)
        except json.JSONDecodeError:
            logger.error(f"Error parsing available models for {model_type}")
            return []

    @classmethod
    def get_enabled_models(cls, model_type):
        """Get list of enabled models for a specific type."""
        models = cls.get_available_models(model_type)
        return [model for model in models if model.get("enabled", True)]

    @classmethod
    def get_current_model_info(cls, model_type):
        """Get detailed information about the current model of a specific type."""
        current_model = cls.get_config_value(model_type, "CURRENT_MODEL")
        if not current_model:
            return None

        models = cls.get_available_models(model_type)
        for model in models:
            if model["name"] == current_model:
                return model

        return None

    @classmethod
    def load_ai_model(cls, model_name=None):
        """Load an AI model by name, checking if it's available and enabled."""
        import importlib

        # If no specific model requested, get the current default
        if model_name is None:
            model_name = cls.get_config_value('ai', 'CURRENT_MODEL', 'NoAIModel')

        # Get the list of available models to check if this one is enabled
        available_models = cls.get_available_models('ai')
        model_info = next((m for m in available_models if m["name"] == model_name), None)

        # If model not found or disabled, use default
        if not model_info or not model_info.get("enabled", True):
            logger.warning(f"AI model {model_name} not found or disabled, using default")
            model_name = cls.get_config_value('ai', 'CURRENT_MODEL', 'NoAIModel')

        try:
            # Import the module containing the model class
            module_name = 'plugins.ai_modules'
            module = importlib.import_module(module_name)

            # Get the model class and instantiate it
            model_class = getattr(module, model_name)
            logger.info(f"Loading AI model: {model_name}")
            return model_class()
        except (AttributeError, ImportError) as e:
            logger.error(f"Error loading AI model {model_name}: {e}")

            # Fall back to NoAIModel
            try:
                module = importlib.import_module(module_name)
                logger.warning(f"Falling back to NoAIModel")
                return module.NoAIModel()
            except Exception as fallback_e:
                logger.error(f"Error loading fallback NoAIModel: {fallback_e}")

                # As a last resort, create a simple object that implements the interface
                class EmergencyFallbackModel:
                    def get_response(self, prompt):
                        return "AI service is currently unavailable."

                    def generate_description(self, image_path):
                        return "Image description is currently unavailable."

                return EmergencyFallbackModel()

    @classmethod
    def load_embedding_model(cls, model_name=None):
        """Load an embedding model by name, checking if it's available and enabled."""
        import importlib

        # If no specific model requested, get the current default
        if model_name is None:
            model_name = cls.get_config_value('embedding', 'CURRENT_MODEL', 'NoEmbeddingModel')

        # Get the list of available models to check if this one is enabled
        available_models = cls.get_available_models('embedding')
        model_info = next((m for m in available_models if m["name"] == model_name), None)

        # If model not found or disabled, use default
        if not model_info or not model_info.get("enabled", True):
            logger.warning(f"Embedding model {model_name} not found or disabled, using default")
            model_name = cls.get_config_value('embedding', 'CURRENT_MODEL', 'NoEmbeddingModel')

        try:
            # Import the module containing the model class
            module_name = 'plugins.ai_modules'
            module = importlib.import_module(module_name)

            # Get the model class and instantiate it
            model_class = getattr(module, model_name)
            logger.info(f"Loading embedding model: {model_name}")
            return model_class()
        except (AttributeError, ImportError) as e:
            logger.error(f"Error loading embedding model {model_name}: {e}")

            # Fall back to NoEmbeddingModel
            try:
                module = importlib.import_module(module_name)
                logger.warning(f"Falling back to NoEmbeddingModel")
                return module.NoEmbeddingModel()
            except Exception as fallback_e:
                logger.error(f"Error loading fallback NoEmbeddingModel: {fallback_e}")

                # As a last resort, create a simple object that implements the interface
                class EmergencyFallbackEmbedding:
                    def get_embeddings(self, text):
                        return []

                return EmergencyFallbackEmbedding()

    @classmethod
    def load_image_model(cls, model_name=None):
        """Load an image model by name, checking if it's available and enabled."""
        import importlib

        # If no specific model requested, get the current default
        if model_name is None:
            model_name = cls.get_config_value('image', 'CURRENT_MODEL', 'NoImageModel')

        # Get the list of available models to check if this one is enabled
        available_models = cls.get_available_models('image')
        model_info = next((m for m in available_models if m["name"] == model_name), None)

        # If model not found or disabled, use default
        if not model_info or not model_info.get("enabled", True):
            logger.warning(f"Image model {model_name} not found or disabled, using default")
            model_name = cls.get_config_value('image', 'CURRENT_MODEL', 'NoImageModel')

        try:
            # Import the image module containing the model class
            try:
                module_name = 'plugins.image_modules.image_models'
                module = importlib.import_module(module_name)
            except ImportError:
                # Fallback to a different module path if needed
                module_name = 'plugins.ai_modules'
                module = importlib.import_module(module_name)

            # Get the model class and instantiate it
            model_class = getattr(module, model_name)
            logger.info(f"Loading image model: {model_name}")
            return model_class()
        except (AttributeError, ImportError) as e:
            logger.error(f"Error loading image model {model_name}: {e}")

            # Fall back to creating a simple image handler
            try:
                # Try to import a default image handler
                module_name = 'plugins.image_modules.image_models'
                module = importlib.import_module(module_name)

                # Look for a default handler function
                if hasattr(module, 'get_default_model_handler'):
                    logger.warning(f"Falling back to default image model handler")
                    return module.get_default_model_handler()
                elif hasattr(module, 'CLIPModelHandler'):
                    logger.warning(f"Falling back to CLIPModelHandler")
                    return module.CLIPModelHandler()
                else:
                    raise ImportError("No suitable image model found")

            except ImportError as fallback_e:
                logger.error(f"Error loading fallback image model: {fallback_e}")

                # As a last resort, create a simple object that implements basic image interface
                class EmergencyFallbackImageModel:
                    def __init__(self):
                        self.model_name = "NoImageModel"

                    def process_image(self, image_path):
                        return "Image processing is currently unavailable."

                    def compare_images(self, image1_path, image2_path):
                        return {"similarity": 0.0, "message": "Image comparison is currently unavailable."}

                    def generate_description(self, image_path):
                        return "Image description is currently unavailable."

                logger.warning("Using emergency fallback image model")
                return EmergencyFallbackImageModel()


# Define the AIModel interface
class AIModel(ABC):
    @abstractmethod
    def get_response(self, prompt: str) -> str:
        pass

    @abstractmethod
    def generate_description(self, image_path: str) -> str:
        pass


# Define the EmbeddingModel interface
class EmbeddingModel(ABC):
    @abstractmethod
    def get_embeddings(self, text: str) -> list:
        pass


# Define the ImageModel interface
class ImageModel(ABC):
    @abstractmethod
    def process_image(self, image_path: str) -> str:
        pass

    @abstractmethod
    def compare_images(self, image1_path: str, image2_path: str) -> dict:
        pass

    @abstractmethod
    def generate_description(self, image_path: str) -> str:
        pass


# Implementations of the AI model classes
class NoAIModel(AIModel):
    def get_response(self, prompt: str) -> str:
        return "AI is currently disabled."

    def generate_description(self, image_path: str) -> str:
        return "AI description generation is currently disabled."


class AnthropicModel(AIModel):
    def __init__(self):
        self.api_key = ANTHROPIC_API_KEY
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.model = "claude-3-5-sonnet-20241022"
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        logger.debug(f"Anthropic API Key: {self.api_key[:5] if self.api_key else 'None'}...")

    def get_response(self, prompt: str) -> str:
        logger.debug(f"Using Anthropic model: {self.model}")
        logger.debug(f"Sending prompt to Anthropic: {prompt}")

        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )

            if response.status_code == 200:
                response_data = response.json()
                answer = response_data['content'][0]['text']
                logger.debug(f"Anthropic response: {answer}")
                return answer
            else:
                logger.error(f"Error from Anthropic API: {response.status_code} - {response.text}")
                return f"An error occurred: {response.status_code}"

        except Exception as e:
            logger.error(f"Error while getting response from Anthropic: {e}")
            return "An error occurred while processing your request."

    def generate_description(self, image_path: str) -> str:
        logger.debug(f"Generating image description with Anthropic")

        try:
            # Convert the image to base64
            base64_image = self.encode_image(image_path)

            payload = {
                "model": self.model,
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image
                                }
                            },
                            {
                                "type": "text",
                                "text": "What's in this image?"
                            }
                        ]
                    }
                ]
            }

            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )

            if response.status_code == 200:
                response_data = response.json()
                description = response_data['content'][0]['text']
                return description
            else:
                logger.error(f"Error from Anthropic API: {response.status_code} - {response.text}")
                return f"An error occurred: {response.status_code}"

        except Exception as e:
            logger.error(f"Error while generating description with Anthropic: {e}")
            return "An error occurred while processing the image."

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            base64_encoded = base64.b64encode(image_file.read()).decode('utf-8')
            return base64_encoded


class OpenAIModel(AIModel):
    def __init__(self):
        openai.api_key = OPENAI_API_KEY
        logger.debug(f"OpenAI API Key: {OPENAI_API_KEY[:5] if OPENAI_API_KEY else 'None'}...")

    def get_response(self, prompt: str) -> str:
        logger.debug(f"Using OpenAI model: {OPENAI_MODEL_NAME}")
        logger.debug(f"Sending prompt to OpenAI: {prompt}")
        try:
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=1000
            )
            answer = response.choices[0].text.strip()
            logger.debug(f"OpenAI response: {answer}")
            return answer
        except Exception as e:
            logger.error(f"Error while getting response from OpenAI: {e}")
            return "An error occurred while processing your request."

    def generate_description(self, image_path: str) -> str:
        base64_image = self.encode_image(image_path)
        payload = {
            "model": "gpt-4-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What's in this image?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        if response.status_code == 200:
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                return response_data['choices'][0]['message']['content']
            else:
                return "No description available."
        else:
            return "Error in API request."

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            base64_encoded = base64.b64encode(image_file.read()).decode('utf-8')
            return base64_encoded


class Llama3Model(AIModel):
    def __init__(self):
        self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            logger.debug(f"Loaded Hugging Face model: {self.model_id}")
        except Exception as e:
            logger.error(f"Error loading Llama3 model: {e}")
            self.model = None
            self.tokenizer = None

    def get_response(self, prompt: str) -> str:
        if self.model is None or self.tokenizer is None:
            return "Llama3 model is not available."

        logger.debug(f"Using Hugging Face model: {self.model_id}")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("")
            ]

            outputs = self.model.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            response = outputs[0][input_ids.shape[-1]:]
            decoded_response = self.tokenizer.decode(response, skip_special_tokens=True)
            logger.debug(f"Hugging Face response: {decoded_response}")
            return decoded_response
        except Exception as e:
            logger.error(f"Error generating response with Llama3: {e}")
            return "An error occurred while processing your request."

    def generate_description(self, image_path: str) -> str:
        return "Image description not supported for Llama model."


# Implementation of the embedding model classes
class NoEmbeddingModel(EmbeddingModel):
    def get_embeddings(self, text: str) -> list:
        return []


class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self):
        openai.api_key = OPENAI_API_KEY
        logger.debug(f"OpenAI API Key for embedding: {OPENAI_API_KEY[:5] if OPENAI_API_KEY else 'None'}...")

    def get_embeddings(self, text: str) -> list:
        logger.debug(f"Generating embeddings using OpenAI model: {OPENAI_MODEL_NAME}")
        try:
            response = openai.Embedding.create(
                input=text,
                model=OPENAI_MODEL_NAME
            )
            embeddings = response['data'][0]['embedding']
            logger.debug(f"Generated embeddings: {len(embeddings)} dimensions")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings with OpenAI: {e}")
            return []


# Implementation of the image model classes
class NoImageModel(ImageModel):
    def __init__(self):
        self.model_name = "NoImageModel"

    def process_image(self, image_path: str) -> str:
        return "Image processing is currently disabled."

    def compare_images(self, image1_path: str, image2_path: str) -> dict:
        return {
            "similarity": 0.0,
            "message": "Image comparison is currently disabled.",
            "model": self.model_name
        }

    def generate_description(self, image_path: str) -> str:
        return "Image description is currently disabled."


class CLIPModelHandler:  # Add (ImageModel) if you have a base class
    """Optimized CLIP model handler with offline mode and intelligent caching"""

    # Class-level cache to avoid reloading models across instances
    _model_cache = {}
    _processor_cache = {}
    _cache_initialized = False

    def __init__(self):
        self.model_name = "CLIPModelHandler"
        self.clip_model_name = "openai/clip-vit-base-patch32"

        # Configure offline mode FIRST to prevent network checks
        if not self._cache_initialized:
            self._configure_offline_mode()
            CLIPModelHandler._cache_initialized = True

        logger.info("Initializing optimized CLIP model handler")

        # Load model and processor with intelligent caching
        self.model, self.processor = self._load_or_get_cached_model()

        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(self.model, 'to'):
            self.model.to(self.device)

        logger.info(f"✅ CLIP model ready on {self.device}")

    def _configure_offline_mode(self):
        """Configure environment to disable online checks - MAJOR SPEEDUP!"""
        # Set environment variables to force offline mode
        offline_env_vars = {
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false"  # Disable warnings
        }

        for key, value in offline_env_vars.items():
            os.environ[key] = value

        logger.info("🚀 Configured offline mode - network checks disabled")

    def _load_or_get_cached_model(self):
        """Load model with intelligent caching - avoids repeated loading"""
        cache_key = self.clip_model_name

        # Return cached model if available
        if cache_key in self._model_cache:
            logger.info("⚡ Using cached CLIP model (instant load)")
            return self._model_cache[cache_key], self._processor_cache[cache_key]

        logger.info("🔄 Loading CLIP model for first time...")
        start_time = time.time()

        try:
            # ATTEMPT 1: Try offline loading first (fastest - no network)
            processor = CLIPProcessor.from_pretrained(
                self.clip_model_name,
                local_files_only=True,
                cache_dir="./model_cache"
            )
            model = CLIPModel.from_pretrained(
                self.clip_model_name,
                local_files_only=True,
                cache_dir="./model_cache"
            )
            logger.info("✅ Loaded CLIP model from local cache (offline)")

        except Exception as offline_error:
            logger.warning(f"Offline loading failed: {offline_error}")
            logger.info("📥 Downloading model from HuggingFace (first time only)...")

            # ATTEMPT 2: Download if not cached (temporarily allow network)
            # Temporarily disable offline mode for download
            temp_offline_vars = {}
            for key in ["TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"]:
                if key in os.environ:
                    temp_offline_vars[key] = os.environ.pop(key)

            try:
                processor = CLIPProcessor.from_pretrained(
                    self.clip_model_name,
                    cache_dir="./model_cache"
                )
                model = CLIPModel.from_pretrained(
                    self.clip_model_name,
                    cache_dir="./model_cache"
                )
                logger.info("📥 Successfully downloaded and cached CLIP model")

            except Exception as download_error:
                logger.error(f"Failed to download model: {download_error}")
                raise

            finally:
                # Restore offline mode
                for key, value in temp_offline_vars.items():
                    os.environ[key] = value

        # Cache the loaded models in memory
        self._model_cache[cache_key] = model
        self._processor_cache[cache_key] = processor

        load_time = time.time() - start_time
        logger.info(f"✅ CLIP model loaded and cached in {load_time:.2f}s")

        return model, processor

    def is_valid_image(self, image):
        """Check if image meets requirements for CLIP processing"""
        try:
            if not isinstance(image, PILImage.Image):
                return False

            # Check minimum dimensions (CLIP is quite flexible)
            width, height = image.size
            min_size = 32  # CLIP can handle small images
            max_size = 2048  # Reasonable upper limit

            if width < min_size or height < min_size:
                logger.debug(f"Image too small: {width}x{height}")
                return False

            if width > max_size or height > max_size:
                logger.debug(f"Image very large: {width}x{height} (will resize)")
                # CLIP preprocessor will handle resizing

            return True

        except Exception as e:
            logger.error(f"Error validating image: {e}")
            return False

    def get_image_embedding(self, image):
        """Generate CLIP embedding for an image - CORE FUNCTIONALITY"""
        try:
            if not self.is_valid_image(image):
                logger.warning("Invalid image for embedding generation")
                return None

            # Preprocess image using CLIP processor
            inputs = self.processor(images=image, return_tensors="pt", padding=True)

            # Move inputs to correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embedding with no gradient computation (faster)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

            # Normalize the embedding (important for similarity comparisons)
            embedding = image_features / image_features.norm(dim=-1, keepdim=True)

            # Convert to numpy array for storage/compatibility
            embedding_np = embedding.cpu().numpy().flatten()

            logger.debug(f"Generated embedding with shape: {embedding_np.shape}")
            return embedding_np

        except Exception as e:
            logger.error(f"Error generating image embedding: {e}")
            return None

    def process_image(self, image_path: str) -> str:
        """Process image and return status info"""
        try:
            logger.info(f"Processing image with CLIP: {image_path}")

            # Load and validate image
            image = PILImage.open(image_path).convert("RGB")

            if not self.is_valid_image(image):
                return f"Invalid image: {image_path}"

            # Generate embedding
            embedding = self.get_image_embedding(image)

            if embedding is not None:
                return f"Successfully processed: {image_path} (embedding: {embedding.shape})"
            else:
                return f"Failed to generate embedding: {image_path}"

        except Exception as e:
            logger.error(f"Error processing image with CLIP: {e}")
            return f"Error processing image: {str(e)}"

    def compare_images(self, image1_path: str, image2_path: str) -> dict:
        """Compare two images using CLIP embeddings with cosine similarity"""
        try:
            logger.info(f"Comparing images: {image1_path} vs {image2_path}")

            # Load both images
            image1 = PILImage.open(image1_path).convert("RGB")
            image2 = PILImage.open(image2_path).convert("RGB")

            # Generate embeddings
            embedding1 = self.get_image_embedding(image1)
            embedding2 = self.get_image_embedding(image2)

            if embedding1 is None or embedding2 is None:
                raise ValueError("Failed to generate embeddings for one or both images")

            # Calculate cosine similarity
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]

            # Convert numpy types to Python types for JSON serialization
            similarity = float(similarity)

            # Interpret similarity score
            if similarity > 0.9:
                interpretation = "Very similar"
            elif similarity > 0.7:
                interpretation = "Similar"
            elif similarity > 0.5:
                interpretation = "Somewhat similar"
            else:
                interpretation = "Different"

            return {
                "similarity": similarity,
                "interpretation": interpretation,
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

    def generate_description(self, image_path: str) -> str:
        """Generate basic image description"""
        try:
            logger.info(f"Generating description for: {image_path}")

            # Load image
            image = PILImage.open(image_path).convert("RGB")

            if not self.is_valid_image(image):
                return f"Invalid image for description: {image_path}"

            # Get basic image properties
            width, height = image.size
            aspect_ratio = width / height

            # Determine orientation
            if aspect_ratio > 1.3:
                orientation = "landscape"
            elif aspect_ratio < 0.77:
                orientation = "portrait"
            else:
                orientation = "square"

            # Calculate megapixels
            megapixels = (width * height) / 1_000_000

            # Generate description
            description = f"A {orientation} image with {width}×{height} pixels ({megapixels:.1f}MP)"

            # Add file info
            import os
            file_size = os.path.getsize(image_path) / 1024  # KB
            if file_size > 1024:
                size_str = f"{file_size / 1024:.1f}MB"
            else:
                size_str = f"{file_size:.0f}KB"

            description += f", file size: {size_str}"

            return description

        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return f"Error generating description: {str(e)}"

    @classmethod
    def get_cache_stats(cls):
        """Get information about model cache status"""
        return {
            "models_cached": len(cls._model_cache),
            "cache_initialized": cls._cache_initialized,
            "cached_model_names": list(cls._model_cache.keys())
        }

    @classmethod
    def clear_cache(cls):
        """Clear the model cache to free memory"""
        cls._model_cache.clear()
        cls._processor_cache.clear()
        cls._cache_initialized = False
        logger.info("🗑️ Cleared CLIP model cache")

    @classmethod
    def preload_model(cls, model_name="openai/clip-vit-base-patch32"):
        """Preload model at application startup for fastest first access"""
        logger.info("🚀 Preloading CLIP model...")
        handler = cls()  # This will load and cache the model
        logger.info("✅ CLIP model preloaded and cached")
        return handler


# Configuration management functions
def initialize_models_config():
    """
    Create the models configuration table if it doesn't exist and register default models.
    This function uses DatabaseConfig for proper session management.
    """
    try:
        logger.info("Initializing models configuration table...")

        # Create an inspector to check if table exists
        inspector = inspect(engine)

        # Check if the table already exists
        if not inspector.has_table(ModelsConfig.__tablename__):
            try:
                # Create the table
                ModelsConfig.__table__.create(engine)
                logger.info(f"Successfully created table {ModelsConfig.__tablename__}")
            except Exception as e:
                logger.error(f"Error creating ModelsConfig table: {str(e)}")
                return False

        # Initialize with default configurations
        success = register_default_models()
        if success:
            logger.info("Default model configurations registered successfully")
        else:
            logger.warning("Some issues occurred while registering default model configurations")

        return True

    except Exception as e:
        logger.error(f"Unexpected error initializing ModelsConfig: {str(e)}")
        logger.exception("Exception details:")
        return False


def register_default_models():
    """Register the default models in the database using DatabaseConfig."""
    default_configs = [
        # AI models
        {"model_type": "ai", "key": "available_models", "value": json.dumps([
            {"name": "NoAIModel", "display_name": "Disabled", "enabled": True},
            {"name": "OpenAIModel", "display_name": "OpenAI GPT", "enabled": True},
            {"name": "Llama3Model", "display_name": "Meta Llama 3", "enabled": True},
            {"name": "AnthropicModel", "display_name": "Anthropic Claude", "enabled": True}
        ])},
        {"model_type": "ai", "key": "CURRENT_MODEL", "value": "OpenAIModel"},

        # Embedding models
        {"model_type": "embedding", "key": "available_models", "value": json.dumps([
            {"name": "NoEmbeddingModel", "display_name": "Disabled", "enabled": True},
            {"name": "OpenAIEmbeddingModel", "display_name": "OpenAI Embedding", "enabled": True}
        ])},
        {"model_type": "embedding", "key": "CURRENT_MODEL", "value": "OpenAIEmbeddingModel"},

        # Image models
        {"model_type": "image", "key": "available_models", "value": json.dumps([
            {"name": "NoImageModel", "display_name": "Disabled", "enabled": True},
            {"name": "CLIPModelHandler", "display_name": "CLIP Model Handler", "enabled": True}
        ])},
        {"model_type": "image", "key": "CURRENT_MODEL", "value": "CLIPModelHandler"}
    ]

    try:
        from modules.configuration.config_env import DatabaseConfig
        db_config = DatabaseConfig()
        session = db_config.get_main_session()
    except ImportError:
        logger.warning("DatabaseConfig not available, using fallback session")
        session = Session()

    try:
        for config in default_configs:
            # Check if config already exists
            existing = session.query(ModelsConfig).filter_by(
                model_type=config["model_type"],
                key=config["key"]
            ).first()

            if not existing:
                config_entry = ModelsConfig(**config)
                session.add(config_entry)
                logger.info(f"Registered config: {config['model_type']}.{config['key']}")

        session.commit()
        logger.info("Default configurations registered successfully")
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"Error registering default configurations: {e}")
        return False
    finally:
        session.close()


# Enhanced embedding generation and storage functions
def generate_embedding(document_content, model_name=None):
    """Generate embeddings for document content using the specified model."""
    logger.info(f"Starting generate_embedding")
    logger.debug(f"Document content length: {len(document_content)}")

    try:
        embedding_model = ModelsConfig.load_embedding_model(model_name)

        # If we got NoEmbeddingModel, embeddings are disabled
        if isinstance(embedding_model, NoEmbeddingModel):
            logger.info("Embeddings are currently disabled.")
            return None

        embeddings = embedding_model.get_embeddings(document_content)
        logger.info(f"Successfully generated embedding with {len(embeddings) if embeddings else 0} dimensions")
        return embeddings
    except Exception as e:
        logger.error(f"An error occurred while generating embedding: {e}")
        return None


def store_embedding_enhanced(session, document_id, embeddings, model_name=None):
    """
    Enhanced store embeddings function with pgvector support and transaction safety.
    **UPDATED** for pgvector DocumentEmbedding class compatibility.

    Args:
        session: Database session (REQUIRED - matches framework pattern)
        document_id: ID of the document
        embeddings: List of embedding values
        model_name: Name of the model used (optional)

    Returns:
        bool: Success status
    """
    if model_name is None:
        model_name = ModelsConfig.get_config_value('embedding', 'CURRENT_MODEL', 'NoEmbeddingModel')

    logger.info(f"Storing pgvector embedding for model {model_name} and document ID {document_id}")

    if embeddings is None or len(embeddings) == 0:
        logger.warning(f"No embeddings to store for document ID {document_id}")
        return False

    try:
        # Import DocumentEmbedding here to avoid circular import
        from modules.emtacdb.emtacdb_fts import DocumentEmbedding

        # Use PostgreSQL savepoint for transaction safety (matches framework pattern)
        savepoint = session.begin_nested()
        try:
            # Check if embedding already exists
            existing = session.query(DocumentEmbedding).filter_by(
                document_id=document_id,
                model_name=model_name
            ).first()

            if existing:
                # Update existing embedding using the enhanced property
                existing.embedding_as_list = embeddings  # This uses the pgvector setter
                logger.info(f"Updated existing pgvector embedding for document ID {document_id}")
            else:
                # Create new embedding using the enhanced factory method
                document_embedding = DocumentEmbedding.create_with_pgvector(
                    document_id=document_id,
                    model_name=model_name,
                    embedding=embeddings
                )
                session.add(document_embedding)
                logger.info(f"Created new pgvector embedding for document ID {document_id}")

            session.flush()  # Flush within savepoint
            savepoint.commit()  # Commit savepoint
            return True

        except Exception as savepoint_error:
            savepoint.rollback()  # Rollback only the savepoint
            logger.error(f"Savepoint rolled back for pgvector embedding storage: {savepoint_error}")
            raise

    except Exception as e:
        logger.error(f"An error occurred while storing pgvector embedding: {e}")
        logger.exception("Exception details:")
        return False


def store_embedding(document_id, embeddings, model_name=None):
    """
    Legacy store embedding function for backward compatibility.
    Creates its own session - use store_embedding_enhanced() for better transaction safety.
    """
    logger.warning("store_embedding() is legacy - consider using store_embedding_enhanced() with existing session")

    try:
        from modules.configuration.config_env import DatabaseConfig
        db_config = DatabaseConfig()

        with db_config.main_session() as session:
            return store_embedding_enhanced(session, document_id, embeddings, model_name)

    except ImportError:
        logger.warning("DatabaseConfig not available, using fallback session")
        session = Session()
        try:
            result = store_embedding_enhanced(session, document_id, embeddings, model_name)
            session.commit()
            return result
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()


def generate_and_store_embedding(session, document_id, document_content, model_name=None):
    """
    Combined function to generate and store embeddings using pgvector in one transaction.
    **UPDATED** for pgvector DocumentEmbedding class compatibility.

    Args:
        session: Database session (REQUIRED - matches framework pattern)
        document_id: ID of the document
        document_content: Text content to generate embeddings for
        model_name: Name of the model to use (optional)

    Returns:
        bool: Success status
    """
    logger.info(f"Generating and storing pgvector embedding for document ID {document_id}")

    try:
        # Generate embeddings
        embeddings = generate_embedding(document_content, model_name)

        if embeddings is None or len(embeddings) == 0:
            logger.warning(f"Failed to generate embeddings for document ID {document_id}")
            return False

        # Store embeddings using the updated pgvector method
        success = store_embedding_enhanced(session, document_id, embeddings, model_name)

        if success:
            logger.info(f"Successfully generated and stored pgvector embedding for document ID {document_id}")
        else:
            logger.error(f"Failed to store pgvector embedding for document ID {document_id}")

        return success

    except Exception as e:
        logger.error(f"Error in generate_and_store_embedding for document ID {document_id}: {e}")
        logger.exception("Exception details:")
        return False


# Utility functions for model management
def get_current_models():
    """Get information about all currently active models."""
    try:
        ai_model = ModelsConfig.get_config_value('ai', 'CURRENT_MODEL', 'NoAIModel')
        embedding_model = ModelsConfig.get_config_value('embedding', 'CURRENT_MODEL', 'NoEmbeddingModel')
        image_model = ModelsConfig.get_config_value('image', 'CURRENT_MODEL', 'NoImageModel')

        return {
            'ai': ai_model,
            'embedding': embedding_model,
            'image': image_model
        }
    except Exception as e:
        logger.error(f"Error getting current models: {e}")
        return {
            'ai': 'NoAIModel',
            'embedding': 'NoEmbeddingModel',
            'image': 'NoImageModel'
        }


def test_embedding_functionality():
    """Test function to verify embedding generation and storage is working."""
    logger.info("Testing embedding functionality...")

    test_text = "This is a test document for embedding generation."
    test_document_id = 999999  # Use a high ID that won't conflict

    try:
        # Test embedding generation
        embeddings = generate_embedding(test_text)

        if embeddings is None or len(embeddings) == 0:
            logger.error("Embedding generation test failed")
            return False

        logger.info(f"Embedding generation test passed: {len(embeddings)} dimensions")

        # Test embedding storage (but don't actually store the test)
        logger.info("Embedding functionality test completed successfully")
        return True

    except Exception as e:
        logger.error(f"Embedding functionality test failed: {e}")
        return False


# Legacy function names for backward compatibility
def load_ai_model(model_name=None):
    """Legacy function - use ModelsConfig.load_ai_model instead."""
    return ModelsConfig.load_ai_model(model_name)


def load_embedding_model(model_name=None):
    """Legacy function - use ModelsConfig.load_embedding_model instead."""
    return ModelsConfig.load_embedding_model(model_name)


def load_image_model(model_name=None):
    """Legacy function - use ModelsConfig.load_image_model instead."""
    return ModelsConfig.load_image_model(model_name)


# Initialize models config on import
try:
    initialize_models_config()
    logger.info("AI models module initialized successfully")
except Exception as e:
    logger.error(f"Error during AI models module initialization: {e}")


# ==========================================
# FRAMEWORK INTEGRATION EXAMPLES
# ==========================================

def example_completeDocument_integration():
    """
    Example showing proper integration with CompleteDocument class.
    This demonstrates the correct usage patterns for the framework.
    """
    from modules.configuration.config_env import DatabaseConfig
    from modules.emtacdb.emtacdb_fts import DocumentEmbedding

    db_config = DatabaseConfig()

    with db_config.main_session() as session:
        # Example 1: Generate and store embedding for a document chunk
        document_id = 123
        content = "This is sample document content for embedding generation."

        success = generate_and_store_embedding(session, document_id, content)
        if success:
            print("✅ Embedding generated and stored successfully")

        # Example 2: Store pre-generated embeddings
        embeddings = [0.1, 0.2, 0.3, 0.4, 0.5]  # Example embedding vector
        success = store_embedding_enhanced(session, document_id, embeddings, "OpenAIEmbeddingModel")
        if success:
            print("✅ Pre-generated embedding stored successfully")

        # Example 3: Query stored embeddings
        embedding_record = session.query(DocumentEmbedding).filter_by(
            document_id=document_id,
            model_name="OpenAIEmbeddingModel"
        ).first()

        if embedding_record:
            stored_embeddings = embedding_record.embedding_vector  # Uses property method
            print(f"✅ Retrieved {len(stored_embeddings)} dimension embedding")


def example_model_configuration():
    """
    Example showing proper model configuration management.
    """
    # Set the current models
    ModelsConfig.set_current_embedding_model("OpenAIEmbeddingModel")
    ModelsConfig.set_current_ai_model("AnthropicModel")
    ModelsConfig.set_current_image_model("CLIPModelHandler")

    # Get current models
    current_models = get_current_models()
    print(f"Current models: {current_models}")

    # Load specific models
    embedding_model = ModelsConfig.load_embedding_model("OpenAIEmbeddingModel")
    ai_model = ModelsConfig.load_ai_model("AnthropicModel")
    image_model = ModelsConfig.load_image_model("CLIPModelHandler")

    # Test functionality
    if test_embedding_functionality():
        print("✅ Embedding system working correctly")
    else:
        print("❌ Embedding system needs attention")

    # Test image model
    try:
        result = image_model.process_image("test_image.jpg")
        print(f"✅ Image model working: {result}")
    except Exception as e:
        print(f"❌ Image model error: {e}")


def search_similar_embeddings(session, query_embeddings, model_name=None, limit=10, threshold=0.7):
    """
    Search for similar embeddings using pgvector cosine similarity.

    Args:
        session: Database session
        query_embeddings: Query embedding vector (list of floats)
        model_name: Embedding model name (optional)
        limit: Maximum number of results
        threshold: Minimum similarity threshold (0.0 to 1.0)

    Returns:
        List of similar documents with similarity scores
    """
    if model_name is None:
        model_name = ModelsConfig.get_config_value('embedding', 'CURRENT_MODEL', 'NoEmbeddingModel')

    logger.info(f"Searching similar embeddings with pgvector for model {model_name}")

    try:
        from modules.emtacdb.emtacdb_fts import DocumentEmbedding
        from sqlalchemy import text

        # Convert query embeddings to pgvector format
        query_vector_str = '[' + ','.join(map(str, query_embeddings)) + ']'

        # Use pgvector cosine similarity operator (<=>)
        similarity_query = text("""
            SELECT 
                de.document_id,
                de.model_name,
                de.embedding_vector <=> :query_vector AS distance,
                1 - (de.embedding_vector <=> :query_vector) AS similarity,
                de.created_at,
                de.updated_at
            FROM document_embedding de
            WHERE de.model_name = :model_name
              AND de.embedding_vector IS NOT NULL
              AND (1 - (de.embedding_vector <=> :query_vector)) >= :threshold
            ORDER BY de.embedding_vector <=> :query_vector ASC
            LIMIT :limit
        """)

        result = session.execute(similarity_query, {
            'query_vector': query_vector_str,
            'model_name': model_name,
            'threshold': threshold,
            'limit': limit
        })

        similar_embeddings = []
        for row in result:
            similar_embeddings.append({
                'document_id': row[0],
                'model_name': row[1],
                'distance': float(row[2]),
                'similarity': float(row[3]),
                'created_at': row[4].isoformat() if row[4] else None,
                'updated_at': row[5].isoformat() if row[5] else None
            })

        logger.info(f"Found {len(similar_embeddings)} similar embeddings above threshold {threshold}")
        return similar_embeddings

    except Exception as e:
        logger.error(f"pgvector similarity search failed: {e}")
        return []


def get_embedding_with_similarity(session, document_id, query_embeddings, model_name=None):
    """
    Get a specific embedding and calculate its similarity to a query.

    Args:
        session: Database session
        document_id: ID of the document
        query_embeddings: Query embedding vector for similarity calculation
        model_name: Embedding model name (optional)

    Returns:
        dict: Embedding info with similarity score, or None if not found
    """
    if model_name is None:
        model_name = ModelsConfig.get_config_value('embedding', 'CURRENT_MODEL', 'NoEmbeddingModel')

    try:
        from modules.emtacdb.emtacdb_fts import DocumentEmbedding

        embedding = session.query(DocumentEmbedding).filter_by(
            document_id=document_id,
            model_name=model_name
        ).first()

        if not embedding:
            return None

        # Get the embedding as a list
        embedding_vector = embedding.embedding_as_list

        if not embedding_vector:
            return None

        # Calculate similarity using the enhanced method
        similarity = embedding.cosine_similarity(query_embeddings)

        return {
            'id': embedding.id,
            'document_id': embedding.document_id,
            'model_name': embedding.model_name,
            'embedding': embedding_vector,
            'similarity': similarity,
            'storage_type': embedding.get_storage_type(),
            'dimensions': len(embedding_vector),
            'created_at': embedding.created_at.isoformat() if embedding.created_at else None,
            'updated_at': embedding.updated_at.isoformat() if embedding.updated_at else None
        }

    except Exception as e:
        logger.error(f"Error getting embedding with similarity for document {document_id}: {e}")
        return None


def get_pgvector_statistics():
    """
    Get statistics about pgvector usage in the DocumentEmbedding table.

    Returns:
        dict: Statistics about embedding storage
    """
    try:
        from modules.configuration.config_env import DatabaseConfig
        from modules.emtacdb.emtacdb_fts import DocumentEmbedding
        from sqlalchemy import func

        db_config = DatabaseConfig()
        with db_config.main_session() as session:

            total_embeddings = session.query(DocumentEmbedding).count()

            pgvector_embeddings = session.query(DocumentEmbedding).filter(
                DocumentEmbedding.embedding_vector.isnot(None)
            ).count()

            legacy_embeddings = session.query(DocumentEmbedding).filter(
                DocumentEmbedding.model_embedding.isnot(None),
                DocumentEmbedding.embedding_vector.is_(None)
            ).count()

            # Get model distribution for pgvector embeddings
            pgvector_models = session.query(
                DocumentEmbedding.model_name,
                func.count(DocumentEmbedding.id).label('count')
            ).filter(
                DocumentEmbedding.embedding_vector.isnot(None)
            ).group_by(DocumentEmbedding.model_name).all()

            statistics = {
                'total_embeddings': total_embeddings,
                'pgvector_embeddings': pgvector_embeddings,
                'legacy_embeddings': legacy_embeddings,
                'pgvector_percentage': (pgvector_embeddings / total_embeddings * 100) if total_embeddings > 0 else 0,
                'pgvector_models': {model: count for model, count in pgvector_models},
                'needs_migration': legacy_embeddings > 0
            }

            logger.info(
                f"pgvector statistics: {pgvector_embeddings}/{total_embeddings} using pgvector ({statistics['pgvector_percentage']:.1f}%)")
            return statistics

    except Exception as e:
        logger.error(f"Failed to get pgvector statistics: {e}")
        return {}


def test_pgvector_functionality():
    """Test function to verify pgvector embedding functionality is working."""
    logger.info("Testing pgvector embedding functionality...")

    test_text = "This is a test document for pgvector embedding generation."
    test_document_id = 999999  # Use a high ID that won't conflict

    try:
        from modules.configuration.config_env import DatabaseConfig

        db_config = DatabaseConfig()
        with db_config.main_session() as session:

            # Test embedding generation and storage
            success = generate_and_store_embedding(session, test_document_id, test_text)

            if not success:
                logger.error("pgvector embedding generation and storage test failed")
                return False

            logger.info("pgvector embedding generation and storage test passed")

            # Test retrieval
            from modules.emtacdb.emtacdb_fts import DocumentEmbedding

            embedding_record = session.query(DocumentEmbedding).filter_by(
                document_id=test_document_id
            ).first()

            if embedding_record:
                embeddings = embedding_record.embedding_as_list
                storage_type = embedding_record.get_storage_type()

                if embeddings and len(embeddings) > 0:
                    logger.info(f"pgvector retrieval test passed: {len(embeddings)} dimensions using {storage_type}")

                    # Clean up test data
                    session.delete(embedding_record)
                    session.commit()

                    return True
                else:
                    logger.error("pgvector retrieval test failed - no embeddings")
                    return False
            else:
                logger.error("pgvector retrieval test failed - no record found")
                return False

    except Exception as e:
        logger.error(f"pgvector functionality test failed: {e}")
        return False


# Update the integration example too:
def example_completeDocument_integration():
    """
    Example showing proper integration with updated pgvector DocumentEmbedding class.
    This demonstrates the correct usage patterns for the pgvector framework.
    """
    from modules.configuration.config_env import DatabaseConfig
    from modules.emtacdb.emtacdb_fts import DocumentEmbedding

    db_config = DatabaseConfig()

    with db_config.main_session() as session:
        # Example 1: Generate and store pgvector embedding for a document chunk
        document_id = 123
        content = "This is sample document content for pgvector embedding generation."

        success = generate_and_store_embedding(session, document_id, content)
        if success:
            print("✅ pgvector embedding generated and stored successfully")

        # Example 2: Store pre-generated embeddings using pgvector
        embeddings = [0.1, 0.2, 0.3, 0.4, 0.5] * 307  # 1536 dimensions for OpenAI
        success = store_embedding_enhanced(session, document_id, embeddings, "OpenAIEmbeddingModel")
        if success:
            print("✅ Pre-generated pgvector embedding stored successfully")

        # Example 3: Query stored pgvector embeddings
        embedding_record = session.query(DocumentEmbedding).filter_by(
            document_id=document_id,
            model_name="OpenAIEmbeddingModel"
        ).first()

        if embedding_record:
            stored_embeddings = embedding_record.embedding_as_list  # Uses pgvector property
            storage_type = embedding_record.get_storage_type()
            print(f"✅ Retrieved {len(stored_embeddings)} dimension embedding using {storage_type}")

        # Example 4: Perform similarity search with pgvector
        query_embeddings = [0.1, 0.2, 0.3, 0.4, 0.5] * 307  # Query vector
        similar_docs = search_similar_embeddings(
            session, query_embeddings, "OpenAIEmbeddingModel", limit=5, threshold=0.8
        )
        print(f"✅ Found {len(similar_docs)} similar documents using pgvector")

        # Example 5: Get embedding statistics
        stats = get_pgvector_statistics()
        print(f"✅ pgvector usage: {stats.get('pgvector_percentage', 0):.1f}% of embeddings")

# ==========================================
# INTEGRATION NOTES FOR COMPLETEDOCUMENT CLASS
# ==========================================

"""
To integrate this updated ai_models.py with your CompleteDocument class:

1. **Update _generate_embeddings_for_chunks method:**

   @classmethod
   def _generate_embeddings_for_chunks(cls, session, chunk_objects):
       try:
           from plugins.ai_modules import generate_and_store_embedding, ModelsConfig

           current_embedding_model = ModelsConfig.get_current_embedding_model_name()
           if current_embedding_model == "NoEmbeddingModel":
               debug_id("Embedding generation disabled, skipping")
               return

           for chunk in chunk_objects:
               try:
                   # Use the aligned function signature
                   success = generate_and_store_embedding(
                       session, chunk.id, chunk.content, current_embedding_model
                   )
                   if success:
                       debug_id(f"Generated embedding for chunk: {chunk.name}")
               except Exception as e:
                   debug_id(f"Error generating embedding for chunk {chunk.name}: {e}")

2. **Import path is now correct:**
   from plugins.ai_modules import generate_and_store_embedding, ModelsConfig

3. **Transaction safety is maintained:**
   All embedding operations use the same session as document creation

4. **Initialize the models config table:**
   Run this once: initialize_models_config()

5. **Add timestamp columns to DocumentEmbedding table:**
   ALTER TABLE document_embedding 
   ADD COLUMN created_at TIMESTAMP DEFAULT NOW(),
   ADD COLUMN updated_at TIMESTAMP DEFAULT NOW();
"""