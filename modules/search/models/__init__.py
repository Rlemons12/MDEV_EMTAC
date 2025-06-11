# modules/search/models/__init__.py
"""
Search models package initialization.
Exports all search-related database models including tracking capabilities.
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Try to import all search models including new tracking models
try:
    from .search_models import (
        # Original search models
        SearchIntent,
        IntentPattern,
        IntentKeyword,
        EntityExtractionRule,
        SearchAnalytics,

        # New tracking models
        SearchResultClick,
        UnifiedSearchWithTracking
    )

    MODELS_AVAILABLE = True
    logger.debug("Search models with tracking imported successfully")

except ImportError as e:
    logger.warning(f"Failed to import search models: {e}")
    MODELS_AVAILABLE = False


    # Create placeholder classes to prevent import errors
    class SearchIntent:
        pass


    class IntentPattern:
        pass


    class IntentKeyword:
        pass


    class EntityExtractionRule:
        pass


    class SearchAnalytics:
        pass


    class SearchResultClick:
        pass


    class UnifiedSearchWithTracking:
        def __init__(self, *args, **kwargs):
            logger.warning("UnifiedSearchWithTracking placeholder - tracking disabled")

        def start_user_session(self, *args, **kwargs):
            return None

        def execute_unified_search_with_tracking(self, question, **kwargs):
            return {
                'status': 'error',
                'message': 'Search tracking not available - models not imported',
                'tracking_enabled': False
            }

        def record_satisfaction(self, *args, **kwargs):
            return False

        def track_result_click(self, *args, **kwargs):
            return False

        def get_performance_report(self, *args, **kwargs):
            return {"error": "Query tracker not available"}

        def end_session(self):
            return False

# Export all models including tracking capabilities
__all__ = [
    # Original search models
    'SearchIntent',
    'IntentPattern',
    'IntentKeyword',
    'EntityExtractionRule',
    'SearchAnalytics',

    # New tracking models
    'SearchResultClick',
    'UnifiedSearchWithTracking',

    # Status flag
    'MODELS_AVAILABLE'
]


# Convenience imports for backward compatibility
def get_tracking_models():
    """Get all tracking-related models."""
    if MODELS_AVAILABLE:
        return {
            'SearchResultClick': SearchResultClick,
            'UnifiedSearchWithTracking': UnifiedSearchWithTracking
        }
    else:
        return {}


def get_search_models():
    """Get all core search models."""
    return {
        'SearchIntent': SearchIntent,
        'IntentPattern': IntentPattern,
        'IntentKeyword': IntentKeyword,
        'EntityExtractionRule': EntityExtractionRule,
        'SearchAnalytics': SearchAnalytics
    }


def is_tracking_available():
    """Check if search tracking is available."""
    return MODELS_AVAILABLE


# Enhanced imports that include tracking capabilities
try:
    # Try to import additional tracking models from nlp_search if available
    from ..nlp_search import SearchSession, SearchQuery, SearchQueryTracker

    # Add to exports
    __all__.extend(['SearchSession', 'SearchQuery', 'SearchQueryTracker'])

    logger.debug("Enhanced tracking models from nlp_search imported successfully")

except ImportError as e:
    logger.debug(f"Enhanced tracking models not available: {e}")


    # Create placeholders for these as well
    class SearchSession:
        pass


    class SearchQuery:
        pass


    class SearchQueryTracker:
        def __init__(self, *args, **kwargs):
            logger.warning("SearchQueryTracker placeholder - tracking disabled")

        def start_search_session(self, *args, **kwargs):
            return None

        def track_search_query(self, *args, **kwargs):
            return None

        def record_user_satisfaction(self, *args, **kwargs):
            return False

        def track_result_click(self, *args, **kwargs):
            return False

        def get_performance_report(self, *args, **kwargs):
            return {"error": "Query tracker not available"}

        def end_search_session(self, *args, **kwargs):
            return False


def create_tracking_wrapper(unified_search_mixin):
    """
    Factory function to create a tracking wrapper around a UnifiedSearchMixin instance.

    Args:
        unified_search_mixin: Instance of UnifiedSearchMixin or AistManager

    Returns:
        UnifiedSearchWithTracking instance or None if not available
    """
    if MODELS_AVAILABLE:
        try:
            return UnifiedSearchWithTracking(unified_search_mixin)
        except Exception as e:
            logger.error(f"Failed to create tracking wrapper: {e}")
            return None
    else:
        logger.warning("Cannot create tracking wrapper - models not available")
        return None


def get_module_status():
    """Get comprehensive module status information."""
    status = {
        "models_available": MODELS_AVAILABLE,
        "tracking_available": is_tracking_available(),
        "core_models": list(get_search_models().keys()),
        "tracking_models": list(get_tracking_models().keys()) if MODELS_AVAILABLE else [],
        "enhanced_tracking": 'SearchQueryTracker' in __all__
    }

    # Test model instantiation
    try:
        # Test core models
        intent = SearchIntent()
        status["core_models_functional"] = True
    except Exception as e:
        status["core_models_functional"] = False
        status["core_models_error"] = str(e)

    try:
        # Test tracking models if available
        if MODELS_AVAILABLE:
            click = SearchResultClick()
            status["tracking_models_functional"] = True
        else:
            status["tracking_models_functional"] = False
    except Exception as e:
        status["tracking_models_functional"] = False
        status["tracking_models_error"] = str(e)

    return status


# Module initialization logging
if MODELS_AVAILABLE:
    logger.info("✅ Search models package initialized with tracking capabilities")
else:
    logger.warning("⚠️ Search models package initialized in limited mode - tracking disabled")

# Debug information
logger.debug(f"Available models: {__all__}")
logger.debug(f"Module status: {get_module_status()}")