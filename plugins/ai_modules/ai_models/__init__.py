import requests
from .ai_models import (
    load_ai_model,
    load_embedding_model,
    generate_embedding,
    store_embedding,
    store_embedding_enhanced,  # Added this function
    generate_and_store_embedding,  # Added this function
    OpenAIModel,
    Llama3Model,
    AnthropicModel,
    OpenAIEmbeddingModel,
    NoAIModel,
    NoEmbeddingModel,
    ModelsConfig
)

__all__ = [
    'store_embedding',
    'store_embedding_enhanced',  # Added this function
    'generate_and_store_embedding',  # Added this function
    'load_ai_model',
    'load_embedding_model',
    'generate_embedding',
    'OpenAIModel',
    'Llama3Model',
    'AnthropicModel',
    'OpenAIEmbeddingModel',
    'NoAIModel',
    'NoEmbeddingModel',
    'ModelsConfig'
]