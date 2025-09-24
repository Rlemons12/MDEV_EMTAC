"""
NLP search package for emtac_ai.
Provides tracking, ML classifiers, feedback handling, and spaCy-enhanced search.
"""

from .tracker import SearchQueryTracker, SearchSessionManager
from .models import (
    SearchSession, SearchQuery, SearchResultClick,
    MLModel, UserFeedback,
    SearchIntentHierarchy, IntentContext,
    PatternTemplate, PatternVariation,
    EntityType, EntitySynonym,
)
from .ml_models import IntentClassifierML, FeedbackLearner
from .feedback import (
    record_feedback,
    get_feedback_for_query,
    average_rating_for_query,
)
from .spacy_search import SpaCyEnhancedAggregateSearch
from .factories import (
    create_search_session,
    create_search_query,
)

__all__ = [
    # Tracker
    "SearchQueryTracker", "SearchSessionManager",

    # Models
    "SearchSession", "SearchQuery", "SearchResultClick",
    "MLModel", "UserFeedback",
    "SearchIntentHierarchy", "IntentContext", "PatternTemplate", "PatternVariation",
    "EntityType", "EntitySynonym",

    # ML
    "IntentClassifierML", "FeedbackLearner",

    # Feedback helpers
    "record_feedback", "get_feedback_for_query", "average_rating_for_query",

    # SpaCy pipeline
    "SpaCyEnhancedAggregateSearch",

    # Factories
    "create_search_session", "create_search_query",
]
