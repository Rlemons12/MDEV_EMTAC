# plugins/ai_models.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import base64  # Add this import
import logging
import openai
import transformers
import torch
import importlib
import json
from abc import ABC, abstractmethod
from config import (OPENAI_API_KEY, HUGGINGFACE_API_KEY, OPENAI_MODEL_NAME, CURRENT_EMBEDDING_MODEL, DATABASE_URL)
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import create_engine
import requests
logger = logging.getLogger(__name__)

# Database setup
engine = create_engine(DATABASE_URL)
Session = scoped_session(sessionmaker(bind=engine))

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

class NoAIModel(AIModel):
    def get_response(self, prompt: str) -> str:
        return "AI is currently disabled."
    
    def generate_description(self, image_path: str) -> str:
        return "AI description generation is currently disabled."

class NoEmbeddingModel(EmbeddingModel):
    def get_embeddings(self, text: str) -> list:
        return []

# Implement the OpenAI model
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
                            "text": "What’s in this image?"
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

# Implement the Hugging Face Meta-Llama-3-8B-Instruct model
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

# Implement the OpenAI embedding model
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
        logger.debug(f"Generated embeddings: {embeddings}")
        return embeddings

def load_ai_model(model_name):
    module_name = 'plugins.ai_models'
    module = importlib.import_module(module_name)
    model_class = getattr(module, model_name)
    logger.debug(f"Loading AI model class: {model_name}")
    return model_class()

def load_embedding_model(model_class_name: str):
    module_name = 'plugins.ai_models'
    module = importlib.import_module(module_name)
    model_class = getattr(module, model_class_name)
    logger.debug(f"Loading embedding model class: {model_class_name}")
    return model_class()

def generate_embedding(document_content, model_name):
    logger.info(f"Starting generate_embedding for model {model_name}")
    logger.debug(f"Document content length: {len(document_content)}")

    if model_name == "NoEmbeddingModel":
        logger.info("Embeddings are currently disabled.")
        return None

    embedding_model = load_embedding_model(model_name)

    try:
        embeddings = embedding_model.get_embeddings(document_content)
        logger.info("Successfully generated embedding")
        return embeddings
    except Exception as e:
        logger.error(f"An error occurred while generating embedding: {e}")
        return None

def store_embedding(document_id, embeddings, model_name):
    logger.info(f"Storing embedding for model {model_name} and document ID {document_id}")
    try:
        # Import DocumentEmbedding here to avoid circular import
        from emtacdb_fts import DocumentEmbedding
        session = Session()
        document_embedding = DocumentEmbedding(
            document_id=document_id,
            model_name=model_name,
            model_embedding=json.dumps(embeddings).encode('utf-8')
        )
        session.add(document_embedding)
        session.commit()
        logger.info(f"Embedding for model {model_name} stored successfully for document ID {document_id}")
    except Exception as e:
        logger.error(f"An error occurred while storing embedding: {e}")
    finally:
        session.close()
