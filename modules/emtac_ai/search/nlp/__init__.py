"""
NLP search package for emtac_ai.
Provides tracking, ML classifiers, feedback learning, and spaCy-enhanced search.
"""

from .tracker import SearchQueryTracker
from .models import (
    SearchSession, SearchQuery, SearchResultClick,
    MLModel, UserFeedback,
    SearchIntentHierarchy, IntentContext,
    PatternTemplate, PatternVariation,
    EntityType, EntitySynonym,
)
from .ml_classifier import IntentClassifierML
from .feedback import FeedbackLearner
from .spacy_search import SpaCyEnhancedAggregateSearch, EnhancedSpaCyAggregateSearch
from .factories import (
    create_enhanced_search_system,
    create_ml_enhanced_search_system,
)

__all__ = [
    "SearchQueryTracker", "SearchSession", "SearchQuery", "SearchResultClick",
    "MLModel", "UserFeedback",
    "SearchIntentHierarchy", "IntentContext", "PatternTemplate", "PatternVariation",
    "EntityType", "EntitySynonym",
    "IntentClassifierML", "FeedbackLearner",
    "SpaCyEnhancedAggregateSearch", "EnhancedSpaCyAggregateSearch",
    "create_enhanced_search_system", "create_ml_enhanced_search_system",
]
