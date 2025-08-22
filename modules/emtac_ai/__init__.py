# modules/emtac_ai/__init__.py
"""
emtac_ai package init.

This package exposes:
- Query expansion utilities and orchestrator
- DB search layer (repositories, repo manager, aggregate search)
- Intent/NER plugin (for intent classification & entity extraction)

After this, you can import from a single place, e.g.:
    from modules.emtac_ai import (
        EMTACQueryExpansionOrchestrator,
        REPOManager,
        AggregateSearch,
        PartSearchParams,
        IntentEntityPlugin,
    )
"""

__version__ = "0.2.0"

def get_version() -> str:
    return __version__

# --- Query expansion: utils + engine + orchestrator ---
from .query_expansion import (
    # utils
    dedup_preserve_order,
    replace_word_boundary,
    tokenize_words_lower,
    # engine + orchestrator
    SynonymLoader,
    QueryExpansionRAG,
    EMTACQueryExpansionOrchestrator,
    # DB search layer (re-exported by query_expansion.__init__)
    REPOManager,
    BaseRepository,
    PartRepository,
    ImageRepository,
    DocumentRepository,
    DrawingRepository,
    PositionRepository,
    CompleteDocumentRepository,
    AggregateSearch,
    PositionFilters,
    PartSearchParams,
)

# --- Intent / NER plugin ---
from .emtac_intent_entity import IntentEntityPlugin

__all__ = [
    "__version__",
    "get_version",
    # utils
    "dedup_preserve_order",
    "replace_word_boundary",
    "tokenize_words_lower",
    # engine + orchestrator
    "SynonymLoader",
    "QueryExpansionRAG",
    "EMTACQueryExpansionOrchestrator",
    # db search layer
    "REPOManager",
    "BaseRepository",
    "PartRepository",
    "ImageRepository",
    "DocumentRepository",
    "DrawingRepository",
    "PositionRepository",
    "CompleteDocumentRepository",
    "AggregateSearch",
    "PositionFilters",
    "PartSearchParams",
    # intent/NER
    "IntentEntityPlugin",
]
