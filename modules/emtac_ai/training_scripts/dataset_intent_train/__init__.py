# modules/emtac_ai/training_scripts/dataset_intent_train/__init__.py

from .train_parts_ner import main as train_parts_ner_main
from .train_drawings_ner import main as train_drawings_ner_main

__all__ = [
    "train_parts_ner_main",
    "train_drawings_ner_main",
]
