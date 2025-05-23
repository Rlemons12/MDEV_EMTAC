# plugins/ai_modules.py
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
from sqlalchemy import Column, String, Integer, DateTime, Enum, UniqueConstraint, create_engine
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

    # Replace the old functions with these new versions that use ModelsConfig

    @staticmethod
    def load_config_from_db():
        """
        Load AI model configuration from the database.

        Returns:
            Tuple of (current_ai_model, current_embedding_model)
        """
        from modules.configuration.config_env import DatabaseConfig

        db_config = DatabaseConfig()
        session = db_config.get_main_session()
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
        Load image model configuration from the database.

        Returns:
            String representing the current image model
        """
        from modules.configuration.config_env import DatabaseConfig

        db_config = DatabaseConfig()
        session = db_config.get_main_session()
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
        Set a configuration value in the database.

        Args:
            model_type: Type of model ('ai', 'image', 'embedding')
            key: Configuration key
            value: Configuration value

        Returns:
            Boolean indicating success
        """
        from modules.configuration.config_env import DatabaseConfig

        db_config = DatabaseConfig()
        session = db_config.get_main_session()
        try:
            # Check if config already exists
            config = session.query(cls).filter_by(
                model_type=model_type,
                key=key
            ).first()

            if config:
                # Update existing config
                config.value = value
            else:
                # Create new config
                config = cls(
                    model_type=model_type,
                    key=key,
                    value=value
                )
                session.add(config)

            session.commit()
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
        Get a configuration value from the database.

        Args:
            model_type: Type of model ('ai', 'image', 'embedding')
            key: Configuration key
            default: Default value if not found

        Returns:
            Configuration value or default if not found
        """
        from modules.configuration.config_env import DatabaseConfig

        db_config = DatabaseConfig()
        session = db_config.get_main_session()
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
        """
        Set the current AI model to use.

        Args:
            model_name: Name of the AI model class

        Returns:
            Boolean indicating success
        """
        return cls.set_config_value('ai', 'CURRENT_MODEL', model_name)

    @classmethod
    def set_current_embedding_model(cls, model_name):
        """
        Set the current embedding model to use.

        Args:
            model_name: Name of the embedding model class

        Returns:
            Boolean indicating success
        """
        return cls.set_config_value('embedding', 'CURRENT_MODEL', model_name)

    @classmethod
    def set_current_image_model(cls, model_name):
        """
        Set the current image model to use.

        Args:
            model_name: Name of the image model class

        Returns:
            Boolean indicating success
        """
        return cls.set_config_value('image', 'CURRENT_MODEL', model_name)

    @classmethod
    def initialize_model_configs(cls):
        """
        Initialize the model configurations with default values if they don't exist.
        """
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
    def migrate_legacy_configs(cls):
        """
        Migrate data from the old config tables to the new unified table.
        Should be called once during the migration process.
        """
        from modules.configuration.config_env import DatabaseConfig

        db_config = DatabaseConfig()
        session = db_config.get_main_session()
        try:
            # Check if legacy tables exist and have data
            try:
                # Check AIModelConfig
                ai_model_config = session.query(AIModelConfig).filter_by(key="CURRENT_AI_MODEL").first()
                if ai_model_config:
                    cls.set_config_value('ai', 'CURRENT_MODEL', ai_model_config.value)
                    logger.info(f"Migrated AI model config: {ai_model_config.value}")

                embedding_model_config = session.query(AIModelConfig).filter_by(key="CURRENT_EMBEDDING_MODEL").first()
                if embedding_model_config:
                    cls.set_config_value('embedding', 'CURRENT_MODEL', embedding_model_config.value)
                    logger.info(f"Migrated embedding model config: {embedding_model_config.value}")

                # Check ImageModelConfig
                image_model_config = session.query(ImageModelConfig).filter_by(key="CURRENT_IMAGE_MODEL").first()
                if image_model_config:
                    cls.set_config_value('image', 'CURRENT_MODEL', image_model_config.value)
                    logger.info(f"Migrated image model config: {image_model_config.value}")

                logger.info("Legacy config migration completed")
            except Exception as e:
                logger.warning(f"Legacy config migration skipped or failed: {e}")

        finally:
            session.close()

    # Example code to initialize the ModelsConfig table with default values
    # Add this to your database initialization code

    @classmethod
    def initialize_models_config_table(cls):
        """Initialize the ModelsConfig table with default values for all model types."""
        from sqlalchemy import inspect

        # Create table if it doesn't exist
        inspector = inspect(engine)
        if not inspector.has_table(cls.__tablename__):
            cls.__table__.create(engine)
            logger.info(f"Created table {cls.__tablename__}")

        # Call migrate_legacy_configs to migrate data from old tables if they exist
        cls.migrate_legacy_configs()

        # Initialize with default configurations
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
                {"name": "CLIPModelHandler", "display_name": "CLIP Model", "enabled": True},
                {"name": "NoImageModel", "display_name": "Disabled", "enabled": True}
            ])},
            {"model_type": "image", "key": "CURRENT_MODEL", "value": "CLIPModelHandler"}
        ]

        session = Session()
        try:
            for config in default_configs:
                # Check if config already exists
                existing = session.query(cls).filter_by(
                    model_type=config["model_type"],
                    key=config["key"]
                ).first()

                if not existing:
                    config_entry = cls(**config)
                    session.add(config_entry)
                    logger.info(f"Registered config: {config['model_type']}.{config['key']}")

            session.commit()
            logger.info("Default configurations registered successfully")
        except Exception as e:
            session.rollback()
            logger.error(f"Error registering default configurations: {e}")
        finally:
            session.close()

    # Helper functions to get model information
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
    def update_model_status(cls, model_type, model_name, enabled=True):
        """Enable or disable a specific model."""
        models = cls.get_available_models(model_type)

        for model in models:
            if model["name"] == model_name:
                model["enabled"] = enabled
                cls.set_config_value(model_type, "available_models", json.dumps(models))
                return True

        return False

    @classmethod
    def register_new_model(cls, model_type, model_name, display_name, enabled=True):
        """Register a new model in the available models list."""
        models = cls.get_available_models(model_type)

        # Check if model already exists
        for model in models:
            if model["name"] == model_name:
                return False

        # Add new model
        models.append({
            "name": model_name,
            "display_name": display_name,
            "enabled": enabled
        })

        return cls.set_config_value(model_type, "available_models", json.dumps(models))

    # Updated functions to load models using the unified ModelsConfig system

    @classmethod
    def load_ai_model(cls, model_name=None):
        """
        Load an AI model by name, checking if it's available and enabled.

        Args:
            model_name: Optional name of the model to load. If None, loads the current default model.

        Returns:
            An instance of the AI model
        """
        # Import module locally to avoid circular dependencies
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

            # If default model is also not valid, use NoAIModel
            model_info = next((m for m in available_models if m["name"] == model_name), None)
            if not model_info or not model_info.get("enabled", True):
                logger.warning(f"Default AI model {model_name} not found or disabled, using NoAIModel")
                model_name = 'NoAIModel'

        try:
            # Import the module containing the model class
            module_name = 'plugins.ai_modules.ai_models.ai_models'
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
        """
        Load an embedding model by name, checking if it's available and enabled.

        Args:
            model_name: Optional name of the model to load. If None, loads the current default model.

        Returns:
            An instance of the embedding model
        """
        # Import module locally to avoid circular dependencies
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

            # If default model is also not valid, use NoEmbeddingModel
            model_info = next((m for m in available_models if m["name"] == model_name), None)
            if not model_info or not model_info.get("enabled", True):
                logger.warning(f"Default embedding model {model_name} not found or disabled, using NoEmbeddingModel")
                model_name = 'NoEmbeddingModel'

        try:
            # Import the module containing the model class
            module_name = 'plugins.ai_modules.ai_models.ai_models'
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
        """
        Load an image model by name, checking if it's available and enabled.

        Args:
            model_name: Optional name of the model to load. If None, loads the current default model.

        Returns:
            An instance of the image model
        """
        # Import module locally to avoid circular dependencies
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

            # If default model is also not valid, use NoImageModel
            model_info = next((m for m in available_models if m["name"] == model_name), None)
            if not model_info or not model_info.get("enabled", True):
                logger.warning(f"Default image model {model_name} not found or disabled, using NoImageModel")
                model_name = 'NoImageModel'

        try:
            # Import the module containing the model class
            module_name = 'plugins.image_modules.image_models'
            module = importlib.import_module(module_name)

            # Get the model class and instantiate it
            model_class = getattr(module, model_name)
            logger.info(f"Loading image model: {model_name}")
            return model_class()
        except (AttributeError, ImportError) as e:
            logger.error(f"Error loading image model {model_name}: {e}")

            # Fall back to NoImageModel or similar
            try:
                module = importlib.import_module(module_name)
                logger.warning(f"Falling back to default image model handler")
                # Return a default handler if available, or None
                return getattr(module, 'get_default_model_handler')()
            except Exception as fallback_e:
                logger.error(f"Error loading fallback image model: {fallback_e}")
                return None

    @classmethod
    def get_active_model_names(cls):
        """Get the names of all currently active models."""
        return {
            'ai': cls.get_config_value('ai', 'CURRENT_MODEL', 'NoAIModel'),
            'embedding': cls.get_config_value('embedding', 'CURRENT_MODEL', 'NoEmbeddingModel'),
            'image': cls.get_config_value('image', 'CURRENT_MODEL', 'NoImageModel')
        }

    @classmethod
    def get_current_model_name(cls, model_type):
        """Get just the name of the current model for a specific type."""
        return cls.get_config_value(model_type, "CURRENT_MODEL", f"No{model_type.capitalize()}Model")

    @classmethod
    def get_current_ai_model_name(cls):
        """Get just the name of the current AI model."""
        return cls.get_current_model_name('ai')

    @classmethod
    def get_current_embedding_model_name(cls):
        """Get just the name of the current embedding model."""
        return cls.get_current_model_name('embedding')

    @classmethod
    def get_current_image_model_name(cls):
        """Get just the name of the current image model."""
        return cls.get_current_model_name('image')


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
        logger.debug(f"Anthropic API Key: {self.api_key[:5]}...")  # Mask most of the key for security

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

        # Read and encode the image
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
        logger.debug(f"OpenAI API Key: {OPENAI_API_KEY[:5]}...")  # Mask most of the key for security

    def get_response(self, prompt: str) -> str:
        logger.debug(f"Using OpenAI model: {OPENAI_MODEL_NAME}")
        logger.debug(f"Sending prompt to OpenAI: {prompt}")
        try:
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",  # or any other model name
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
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        logger.debug(f"Loaded Hugging Face model: {self.model_id}")

    def get_response(self, prompt: str) -> str:
        logger.debug(f"Using Hugging Face model: {self.model_id}")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

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

    def generate_description(self, image_path: str) -> str:
        return "Image description not supported for Llama model."


# Implementation of the embedding model classes
class NoEmbeddingModel(EmbeddingModel):
    def get_embeddings(self, text: str) -> list:
        return []


class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self):
        openai.api_key = OPENAI_API_KEY
        logger.debug(f"OpenAI API Key for embedding: {OPENAI_API_KEY[:5]}...")  # Mask most of the key for security

    def get_embeddings(self, text: str) -> list:
        logger.debug(f"Generating embeddings using OpenAI model: {OPENAI_MODEL_NAME}")
        response = openai.Embedding.create(
            input=text,
            model=OPENAI_MODEL_NAME
        )
        embeddings = response['data'][0]['embedding']
        logger.debug(f"Generated embeddings: {len(embeddings)} dimensions")
        return embeddings

# Configuration management functions
def initialize_models_config():
    """
    Create the models configuration table if it doesn't exist and register default models.

    This function initializes the ModelsConfig table that stores configuration for
    all AI, embedding, and image models in the system. It uses SQLAlchemy's inspect
    functionality to check if the table exists before creating it.

    Returns:
        bool: True if initialization was successful, False otherwise
    """
    try:
        # Import required modules
        from sqlalchemy import inspect
        from modules.emtacdb.emtacdb_fts import ModelsConfig

        logger.info("Initializing models configuration table...")

        # Create an inspector to check if table exists
        inspector = inspect(engine)

        # Check if the table already exists
        if not inspector.has_table(ModelsConfig.__tablename__):
            try:
                # Create the table
                ModelsConfig.__table__.create(engine)
                logger.info(f"Successfully created table {ModelsConfig.__tablename__}")

                # Register default models
                success = register_default_models()
                if success:
                    logger.info("Default model configurations registered successfully")
                else:
                    logger.warning("Some issues occurred while registering default model configurations")

                return True
            except Exception as e:
                logger.error(f"Error creating ModelsConfig table: {str(e)}")
                logger.exception("Table creation exception details:")
                return False
        else:
            logger.info(f"Table {ModelsConfig.__tablename__} already exists")
            return True

    except ImportError as e:
        logger.error(f"Import error during ModelsConfig initialization: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error initializing ModelsConfig: {str(e)}")
        logger.exception("Exception details:")
        return False


def register_default_models():
    """Register the default models in the database."""
    default_configs = [
        # AI models
        {"model_type": "ai", "key": "available_models", "value": json.dumps([
            {"name": "NoAIModel", "display_name": "Disabled", "enabled": True},
            {"name": "OpenAIModel", "display_name": "OpenAI GPT", "enabled": True},
            {"name": "Llama3Model", "display_name": "Meta Llama 3", "enabled": True},
            {"name": "AnthropicModel", "display_name": "Anthropic Claude", "enabled": True}
        ])},
        {"model_type": "ai", "key": "default_model", "value": "OpenAIModel"},

        # Embedding models
        {"model_type": "embedding", "key": "available_models", "value": json.dumps([
            {"name": "NoEmbeddingModel", "display_name": "Disabled", "enabled": True},
            {"name": "OpenAIEmbeddingModel", "display_name": "OpenAI Embedding", "enabled": True}
        ])},
        {"model_type": "embedding", "key": "default_model", "value": "OpenAIEmbeddingModel"},

        # Image models (for future expansion)
        {"model_type": "image", "key": "available_models", "value": json.dumps([
            {"name": "NoAIModel", "display_name": "Disabled", "enabled": True}
        ])},
        {"model_type": "image", "key": "default_model", "value": "NoAIModel"}
    ]

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
    except Exception as e:
        session.rollback()
        logger.error(f"Error registering default configurations: {e}")
    finally:
        session.close()


def get_config_value(model_type, key, default=None):
    """Get a configuration value from the database."""
    session = Session()
    try:
        config = session.query(ModelsConfig).filter_by(
            model_type=model_type,
            key=key
        ).first()

        if not config:
            logger.warning(f"No configuration found for {model_type}.{key}")
            return default

        return config.value
    except Exception as e:
        logger.error(f"Error retrieving configuration {model_type}.{key}: {e}")
        return default
    finally:
        session.close()


def set_config_value(model_type, key, value):
    """Set a configuration value in the database."""
    session = Session()
    try:
        config = session.query(ModelsConfig).filter_by(
            model_type=model_type,
            key=key
        ).first()

        if not config:
            config = ModelsConfig(
                model_type=model_type,
                key=key,
                value=value
            )
            session.add(config)
        else:
            config.value = value

        session.commit()
        logger.info(f"Updated configuration {model_type}.{key}")
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"Error updating configuration {model_type}.{key}: {e}")
        return False
    finally:
        session.close()


def get_available_models(model_type):
    """Get all available models for a specific model type."""
    models_json = get_config_value(model_type, "available_models", "[]")
    try:
        models = json.loads(models_json)
        # Filter to only enabled models
        enabled_models = [model for model in models if model.get("enabled", True)]
        logger.debug(f"Retrieved {len(enabled_models)} available {model_type} models")
        return enabled_models
    except json.JSONDecodeError:
        logger.error(f"Error parsing available models for {model_type}")
        return []


def get_default_model(model_type):
    """Get the default model for a specific model type."""
    default = get_config_value(model_type, "CURRENT_MODEL", f"No{model_type.capitalize()}Model")
    return default


def update_model_status(model_type, model_name, enabled=True):
    """Enable or disable a specific model."""
    models_json = get_config_value(model_type, "available_models", "[]")
    try:
        models = json.loads(models_json)
        found = False
        for model in models:
            if model["name"] == model_name:
                model["enabled"] = enabled
                found = True
                break

        if not found:
            logger.warning(f"Model {model_name} not found in {model_type} models")
            return False

        # Update the configuration
        success = set_config_value(model_type, "available_models", json.dumps(models))
        if success:
            status = "enabled" if enabled else "disabled"
            logger.info(f"Model {model_name} {status}")
        return success
    except json.JSONDecodeError:
        logger.error(f"Error parsing available models for {model_type}")
        return False


def add_model(model_type, model_name, display_name, enabled=True):
    """Add a new model to the available models."""
    models_json = get_config_value(model_type, "available_models", "[]")
    try:
        models = json.loads(models_json)

        # Check if model already exists
        for model in models:
            if model["name"] == model_name:
                logger.warning(f"Model {model_name} already exists in {model_type} models")
                return False

        # Add new model
        models.append({"name": model_name, "display_name": display_name, "enabled": enabled})
        success = set_config_value(model_type, "available_models", json.dumps(models))
        if success:
            logger.info(f"Added new model {model_name} to {model_type} models")
        return success
    except json.JSONDecodeError:
        logger.error(f"Error parsing available models for {model_type}")
        return False


def set_default_model(model_type, model_name):
    """Set the default model for a specific model type."""
    available_models = get_available_models(model_type)
    model_names = [model["name"] for model in available_models]

    if model_name not in model_names:
        logger.warning(f"Model {model_name} not found in available {model_type} models")
        return False

    success = set_config_value(model_type, "default_model", model_name)
    if success:
        logger.info(f"Set default {model_type} model to {model_name}")
    return success


# Model loading functions
def load_ai_model(model_name=None):
    """
    Load an AI model by name, checking if it's available and enabled.
    If model_name is None, load the default model.
    """
    if not model_name:
        model_name = get_default_model("ai")

    available_models = get_available_models("ai")
    model_info = next((m for m in available_models if m["name"] == model_name), None)

    if not model_info:
        logger.warning(f"Model {model_name} not found or not enabled, using default model")
        model_name = get_default_model("ai")

    try:
        # Load the model class
        module_name = 'plugins.ai_modules'
        module = importlib.import_module(module_name)
        model_class = getattr(module, model_name)
        logger.debug(f"Loading AI model class: {model_name}")
        return model_class()
    except (AttributeError, ImportError) as e:
        logger.error(f"Model class {model_name} not found: {e}")
        # Fall back to NoAIModel
        return NoAIModel()
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        # Fall back to NoAIModel
        return NoAIModel()


def load_embedding_model(model_name=None):
    """
    Load an embedding model by name, checking if it's available and enabled.
    If model_name is None, load the default model.
    """
    if not model_name:
        model_name = get_default_model("embedding")

    available_models = get_available_models("embedding")
    model_info = next((m for m in available_models if m["name"] == model_name), None)

    if not model_info:
        logger.warning(f"Embedding model {model_name} not found or not enabled, using default model")
        model_name = get_default_model("embedding")

    try:
        # Load the model class
        module_name = 'plugins.ai_modules'
        module = importlib.import_module(module_name)
        model_class = getattr(module, model_name)
        logger.debug(f"Loading embedding model class: {model_name}")
        return model_class()
    except (AttributeError, ImportError) as e:
        logger.error(f"Model class {model_name} not found: {e}")
        # Fall back to NoEmbeddingModel
        return NoEmbeddingModel()
    except Exception as e:
        logger.error(f"Error loading embedding model {model_name}: {e}")
        # Fall back to NoEmbeddingModel
        return NoEmbeddingModel()


# Utility functions
def get_model_choices(model_type):
    """Get a list of tuples (model_name, display_name) for UI dropdowns."""
    models = get_available_models(model_type)
    return [(model["name"], model["display_name"]) for model in models]


# Embedding generation and storage functions
def generate_embedding(document_content, model_name=None):
    """Generate embeddings for document content using the specified model."""
    logger.info(f"Starting generate_embedding")
    logger.debug(f"Document content length: {len(document_content)}")

    embedding_model = load_embedding_model(model_name)

    # If we got NoEmbeddingModel, embeddings are disabled
    if isinstance(embedding_model, NoEmbeddingModel):
        logger.info("Embeddings are currently disabled.")
        return None

    try:
        embeddings = embedding_model.get_embeddings(document_content)
        logger.info(f"Successfully generated embedding with {len(embeddings) if embeddings else 0} dimensions")
        return embeddings
    except Exception as e:
        logger.error(f"An error occurred while generating embedding: {e}")
        return None


def store_embedding(document_id, embeddings, model_name=None):
    """Store embeddings for a document in the database."""
    if model_name is None:
        model_name = get_default_model("embedding")

    logger.info(f"Storing embedding for model {model_name} and document ID {document_id}")

    if embeddings is None:
        logger.warning(f"No embeddings to store for document ID {document_id}")
        return False

    try:
        # Import DocumentEmbedding here to avoid circular import
        from modules.emtacdb.emtacdb_fts import DocumentEmbedding
        session = Session()

        # Check if embedding already exists
        existing = session.query(DocumentEmbedding).filter_by(
            document_id=document_id,
            model_name=model_name
        ).first()

        if existing:
            # Update existing embedding
            existing.model_embedding = json.dumps(embeddings).encode('utf-8')
            logger.info(f"Updated existing embedding for document ID {document_id}")
        else:
            # Create new embedding
            document_embedding = DocumentEmbedding(
                document_id=document_id,
                model_name=model_name,
                model_embedding=json.dumps(embeddings).encode('utf-8')
            )
            session.add(document_embedding)
            logger.info(f"Created new embedding for document ID {document_id}")

        session.commit()
        return True
    except Exception as e:
        logger.error(f"An error occurred while storing embedding: {e}")
        return False
    finally:
        session.close()


