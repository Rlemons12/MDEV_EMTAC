# modules/search/__init__.py
"""
Search module for comprehensive manufacturing and maintenance search capabilities.

This module provides:
- Unified search interface (UnifiedSearchMixin)
- NLP-enhanced search with spaCy integration
- Database pattern integration and management
- Aggregated search across multiple entity types
- Search analytics and performance tracking
- Pattern management and synonym resolution

Key Components:
- UnifiedSearchMixin: Main search interface for AistManager
- SpaCyEnhancedAggregateSearch: NLP-powered search engine
- AggregateSearch: Core aggregated search functionality
- SearchPatternManager: Database pattern management
- Search tracking and analytics models
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Module availability flags
_COMPONENTS_AVAILABLE = {
    'unified_search': False,
    'nlp_search': False,
    'aggregate_search': False,
    'pattern_manager': False,
    'search_models': False,
    'tracking_models': False,
    'utils': False
}

_IMPORT_ERRORS = {}

# ===== CORE SEARCH COMPONENTS =====

# 1. Unified Search Interface (Primary interface for AistManager)
try:
    from .UnifiedSearchMixin import UnifiedSearchMixin

    _COMPONENTS_AVAILABLE['unified_search'] = True
    logger.debug("UnifiedSearchMixin imported successfully")
except ImportError as e:
    _IMPORT_ERRORS['unified_search'] = str(e)
    logger.warning(f"⚠UnifiedSearchMixin not available: {e}")


    # Create placeholder class
    class UnifiedSearchMixin:
        """Placeholder for UnifiedSearchMixin when not available."""

        def __init__(self):
            logger.warning("UnifiedSearchMixin placeholder - search functionality disabled")

        def is_unified_search_query(self, question: str) -> bool:
            return False

        def execute_unified_search(self, question: str, user_id: str = None, request_id: str = None) -> Dict[str, Any]:
            return {
                'status': 'error',
                'message': 'UnifiedSearchMixin not available',
                'search_type': 'unavailable'
            }

# 2. NLP-Enhanced Search Engine
try:
    from .nlp_search import (
        SpaCyEnhancedAggregateSearch,
        EnhancedSpaCyAggregateSearch,
        SearchQueryTracker,
        create_enhanced_search_system,
        create_ml_enhanced_search_system
    )

    _COMPONENTS_AVAILABLE['nlp_search'] = True
    logger.debug("NLP search components imported successfully")
except ImportError as e:
    _IMPORT_ERRORS['nlp_search'] = str(e)
    logger.warning(f"⚠NLP search components not available: {e}")


    # Create placeholder classes
    class SpaCyEnhancedAggregateSearch:
        def __init__(self, *args, **kwargs):
            logger.warning("SpaCyEnhancedAggregateSearch placeholder - NLP search disabled")

        def execute_nlp_aggregated_search(self, query: str) -> Dict[str, Any]:
            return {'status': 'error', 'message': 'NLP search not available'}


    class EnhancedSpaCyAggregateSearch(SpaCyEnhancedAggregateSearch):
        pass


    class SearchQueryTracker:
        def __init__(self, *args, **kwargs):
            logger.warning("SearchQueryTracker placeholder - tracking disabled")


    def create_enhanced_search_system(*args, **kwargs):
        logger.warning("Enhanced search system not available")
        return None


    def create_ml_enhanced_search_system(*args, **kwargs):
        logger.warning("ML enhanced search system not available")
        return None

# 3. Core Aggregate Search
try:
    from .aggregate_search import AggregateSearch

    _COMPONENTS_AVAILABLE['aggregate_search'] = True
    logger.debug("AggregateSearch imported successfully")
except ImportError as e:
    _IMPORT_ERRORS['aggregate_search'] = str(e)
    logger.warning(f"⚠AggregateSearch not available: {e}")


    class AggregateSearch:
        def __init__(self, *args, **kwargs):
            logger.warning("AggregateSearch placeholder - core search disabled")

        def execute_aggregated_search(self, query: str) -> Dict[str, Any]:
            return {'status': 'error', 'message': 'AggregateSearch not available'}

# 4. Pattern Management
try:
    from .pattern_manager import SearchPatternManager

    _COMPONENTS_AVAILABLE['pattern_manager'] = True
    logger.debug("SearchPatternManager imported successfully")
except ImportError as e:
    _IMPORT_ERRORS['pattern_manager'] = str(e)
    logger.warning(f"⚠SearchPatternManager not available: {e}")


    class SearchPatternManager:
        def __init__(self, *args, **kwargs):
            logger.warning("SearchPatternManager placeholder - pattern management disabled")

        def initialize_default_patterns(self) -> Dict[str, Any]:
            return {'status': 'error', 'message': 'SearchPatternManager not available'}

# 5. Search Models (Core models from models/__init__.py)
try:
    from .models import (
        # Core search models
        SearchIntent,
        IntentPattern,
        IntentKeyword,
        EntityExtractionRule,
        SearchAnalytics,

        # Tracking models
        SearchResultClick,
        UnifiedSearchWithTracking,

        # Status flags
        MODELS_AVAILABLE,

        # Utility functions
        get_tracking_models,
        get_search_models,
        is_tracking_available,
        create_tracking_wrapper,
        get_module_status
    )

    _COMPONENTS_AVAILABLE['search_models'] = True
    logger.debug("Search models imported successfully")
except ImportError as e:
    _IMPORT_ERRORS['search_models'] = str(e)
    logger.warning(f"⚠Search models not available: {e}")

    # Create placeholder classes and functions
    MODELS_AVAILABLE = False


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


    def get_tracking_models():
        return {}


    def get_search_models():
        return {}


    def is_tracking_available():
        return False


    def create_tracking_wrapper(*args, **kwargs):
        return None


    def get_module_status():
        return {"models_available": False, "tracking_available": False}

# 6. Search Utilities
try:
    from .utils import (
        normalize_part_number,
        extract_numeric_ids,
        extract_area_identifiers,
        extract_part_numbers,
        extract_search_terms,
        log_search_performance,
        validate_search_parameters,
        format_search_results,
        merge_search_results,
        clean_text_for_search,
        extract_equipment_keywords,
        calculate_text_similarity,
        format_error_response,
        get_search_suggestions
    )

    _COMPONENTS_AVAILABLE['utils'] = True
    logger.debug("Search utilities imported successfully")
except ImportError as e:
    _IMPORT_ERRORS['utils'] = str(e)
    logger.warning(f"⚠Search utilities not available: {e}")


    # Create placeholder functions
    def normalize_part_number(part_number: str) -> str:
        return part_number.strip().upper() if part_number else ""


    def extract_numeric_ids(text: str) -> List[int]:
        import re
        return [int(m) for m in re.findall(r'\b(\d{3,})\b', text)]


    def extract_area_identifiers(text: str) -> List[str]:
        import re
        return re.findall(r'\b(?:area|zone)\s+([A-Z0-9]+)\b', text, re.IGNORECASE)


    def extract_part_numbers(text: str) -> List[str]:
        import re
        return re.findall(r'\b([A-Z0-9]{2,}[-\.][A-Z0-9]+)\b', text.upper())


    def extract_search_terms(text: str) -> List[str]:
        import re
        return re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())


    def log_search_performance(*args, **kwargs):
        pass


    def validate_search_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
        return params


    def format_search_results(results: List[Dict], search_type: str = "generic") -> Dict[str, Any]:
        return {"status": "success", "count": len(results), "results": results, "search_type": search_type}


    def merge_search_results(*result_sets) -> Dict[str, Any]:
        return {"status": "error", "message": "Merge function not available"}


    def clean_text_for_search(text: str) -> str:
        return text.strip() if text else ""


    def extract_equipment_keywords(text: str) -> List[str]:
        return []


    def calculate_text_similarity(text1: str, text2: str) -> float:
        return 0.0


    def format_error_response(error_message: str, error_type: str = "general_error",
                              additional_data: Dict[str, Any] = None) -> Dict[str, Any]:
        return {"status": "error", "message": error_message, "error_type": error_type}


    def get_search_suggestions(failed_query: str) -> List[str]:
        return ["Try a different search term", "Check spelling", "Use simpler terms"]

# 7. Enhanced NLP Models (if available)
try:
    from .nlp_search import (
        SearchSession,
        SearchQuery,
        SearchIntentHierarchy,
        IntentContext,
        PatternTemplate,
        PatternVariation,
        EntityType,
        EntitySynonym,
        MLModel,
        UserFeedback
    )

    _COMPONENTS_AVAILABLE['tracking_models'] = True
    logger.debug("Enhanced NLP models imported successfully")
except ImportError as e:
    _IMPORT_ERRORS['tracking_models'] = str(e)
    logger.debug(f"Enhanced NLP models not available: {e}")

    # These are optional, so we don't create placeholders

# ===== MODULE CONFIGURATION =====

# Primary exports for external use
__all__ = [
    # Primary interfaces
    'UnifiedSearchMixin',
    'SpaCyEnhancedAggregateSearch',
    'AggregateSearch',
    'SearchPatternManager',

    # Enhanced interfaces
    'EnhancedSpaCyAggregateSearch',
    'SearchQueryTracker',

    # Factory functions
    'create_enhanced_search_system',
    'create_ml_enhanced_search_system',
    'create_tracking_wrapper',

    # Core models
    'SearchIntent',
    'IntentPattern',
    'IntentKeyword',
    'EntityExtractionRule',
    'SearchAnalytics',
    'SearchResultClick',
    'UnifiedSearchWithTracking',

    # Utilities
    'normalize_part_number',
    'extract_numeric_ids',
    'extract_area_identifiers',
    'extract_part_numbers',
    'extract_search_terms',
    'log_search_performance',
    'validate_search_parameters',
    'format_search_results',
    'merge_search_results',
    'clean_text_for_search',
    'extract_equipment_keywords',
    'calculate_text_similarity',
    'format_error_response',
    'get_search_suggestions',

    # Status functions
    'get_module_status',
    'get_tracking_models',
    'get_search_models',
    'is_tracking_available',
    'get_search_system_status',
    'get_search_capabilities',

    # Constants
    'MODELS_AVAILABLE'
]


# ===== STATUS AND HEALTH FUNCTIONS =====

def get_search_system_status() -> Dict[str, Any]:
    """
    Get comprehensive status of the search system.

    Returns:
        Dictionary with detailed status information
    """
    from datetime import datetime

    status = {
        'timestamp': datetime.utcnow().isoformat(),
        'overall_status': 'operational' if _COMPONENTS_AVAILABLE['unified_search'] else 'limited',
        'component_availability': _COMPONENTS_AVAILABLE.copy(),
        'import_errors': _IMPORT_ERRORS.copy(),
        'available_components': [name for name, available in _COMPONENTS_AVAILABLE.items() if available],
        'unavailable_components': [name for name, available in _COMPONENTS_AVAILABLE.items() if not available],
        'total_components': len(_COMPONENTS_AVAILABLE),
        'available_count': sum(_COMPONENTS_AVAILABLE.values()),
        'models_available': MODELS_AVAILABLE,
        'tracking_available': is_tracking_available()
    }

    # Calculate health score
    available_count = status['available_count']
    total_count = status['total_components']
    status['health_score'] = (available_count / total_count) * 100 if total_count > 0 else 0

    # Determine operational level
    if status['health_score'] >= 80:
        status['operational_level'] = 'full'
    elif status['health_score'] >= 60:
        status['operational_level'] = 'good'
    elif status['health_score'] >= 40:
        status['operational_level'] = 'limited'
    else:
        status['operational_level'] = 'minimal'

    # Add recommendations
    recommendations = []
    if not _COMPONENTS_AVAILABLE['unified_search']:
        recommendations.append("Install UnifiedSearchMixin for full search capabilities")
    if not _COMPONENTS_AVAILABLE['nlp_search']:
        recommendations.append("Install spaCy and NLP dependencies for enhanced search")
    if not _COMPONENTS_AVAILABLE['search_models']:
        recommendations.append("Configure database models for advanced search features")

    status['recommendations'] = recommendations

    return status


def get_search_capabilities() -> Dict[str, Any]:
    """
    Get information about available search capabilities.

    Returns:
        Dictionary describing available search features
    """
    capabilities = {
        'unified_search': {
            'available': _COMPONENTS_AVAILABLE['unified_search'],
            'description': 'Main search interface with automatic query type detection',
            'features': [
                'Natural language query processing',
                'Automatic intent detection',
                'Organized results by entity type',
                'Quick actions and suggestions'
            ] if _COMPONENTS_AVAILABLE['unified_search'] else ['Not available']
        },
        'nlp_search': {
            'available': _COMPONENTS_AVAILABLE['nlp_search'],
            'description': 'Advanced NLP-powered search with spaCy integration',
            'features': [
                'Entity extraction and recognition',
                'Intent classification',
                'Semantic similarity search',
                'Pattern matching and synonyms'
            ] if _COMPONENTS_AVAILABLE['nlp_search'] else ['Not available']
        },
        'aggregate_search': {
            'available': _COMPONENTS_AVAILABLE['aggregate_search'],
            'description': 'Core aggregated search across multiple entity types',
            'features': [
                'Multi-entity search (parts, images, positions)',
                'Comprehensive result aggregation',
                'Legacy keyword compatibility',
                'Performance caching'
            ] if _COMPONENTS_AVAILABLE['aggregate_search'] else ['Not available']
        },
        'pattern_management': {
            'available': _COMPONENTS_AVAILABLE['pattern_manager'],
            'description': 'Database-driven pattern and intent management',
            'features': [
                'Dynamic pattern loading',
                'Intent hierarchy support',
                'Pattern performance tracking',
                'Synonym resolution'
            ] if _COMPONENTS_AVAILABLE['pattern_manager'] else ['Not available']
        },
        'analytics_tracking': {
            'available': _COMPONENTS_AVAILABLE['search_models'] and is_tracking_available(),
            'description': 'Search analytics and performance tracking',
            'features': [
                'Query tracking and analytics',
                'User satisfaction monitoring',
                'Performance metrics',
                'Result click tracking'
            ] if _COMPONENTS_AVAILABLE['search_models'] and is_tracking_available() else ['Not available']
        }
    }

    return capabilities


def diagnose_search_system() -> Dict[str, Any]:
    """
    Diagnose search system issues and provide troubleshooting information.

    Returns:
        Dictionary with diagnostic information and suggestions
    """
    diagnosis = {
        'system_status': get_search_system_status(),
        'capabilities': get_search_capabilities(),
        'health_check': {},
        'troubleshooting': {}
    }

    # Health checks
    health_checks = {}

    # Check primary search interface
    if _COMPONENTS_AVAILABLE['unified_search']:
        try:
            # Test instantiation
            test_search = UnifiedSearchMixin()
            health_checks['unified_search_instantiation'] = 'passed'
        except Exception as e:
            health_checks['unified_search_instantiation'] = f'failed: {e}'
    else:
        health_checks['unified_search_instantiation'] = 'skipped - component not available'

    # Check NLP capabilities
    if _COMPONENTS_AVAILABLE['nlp_search']:
        try:
            # Test NLP search instantiation
            nlp_search = SpaCyEnhancedAggregateSearch()
            health_checks['nlp_search_instantiation'] = 'passed'
        except Exception as e:
            health_checks['nlp_search_instantiation'] = f'failed: {e}'
    else:
        health_checks['nlp_search_instantiation'] = 'skipped - component not available'

    # Check database models
    if _COMPONENTS_AVAILABLE['search_models']:
        try:
            status = get_module_status()
            health_checks['search_models_status'] = 'passed' if status.get('models_available') else 'failed'
        except Exception as e:
            health_checks['search_models_status'] = f'failed: {e}'
    else:
        health_checks['search_models_status'] = 'skipped - models not available'

    diagnosis['health_check'] = health_checks

    # Troubleshooting suggestions
    troubleshooting = {}

    for component, available in _COMPONENTS_AVAILABLE.items():
        if not available and component in _IMPORT_ERRORS:
            error = _IMPORT_ERRORS[component]
            suggestions = []

            if 'spacy' in error.lower():
                suggestions.extend([
                    "Install spaCy: pip install spacy",
                    "Download spaCy model: python -m spacy download en_core_web_sm"
                ])
            elif 'sqlalchemy' in error.lower():
                suggestions.append("Install SQLAlchemy: pip install sqlalchemy")
            elif 'module' in error.lower() and 'not found' in error.lower():
                suggestions.append(f"Install missing dependencies or check module path")
            else:
                suggestions.append("Check dependencies and module configuration")

            troubleshooting[component] = {
                'error': error,
                'suggestions': suggestions
            }

    diagnosis['troubleshooting'] = troubleshooting

    return diagnosis


# ===== FACTORY FUNCTIONS FOR EASY SETUP =====

def create_unified_search_system(session=None, user_context=None, nlp_instance=None):
    """
    Factory function to create a complete unified search system.

    Args:
        session: Database session
        user_context: Optional user context dictionary
        nlp_instance: Optional pre-loaded spaCy instance

    Returns:
        Configured search system or None if not available
    """
    if not _COMPONENTS_AVAILABLE['unified_search']:
        logger.error("Cannot create unified search system - UnifiedSearchMixin not available")
        return None

    try:
        # Create the primary search interface
        search_system = UnifiedSearchMixin()

        # Initialize with enhanced capabilities if available
        if _COMPONENTS_AVAILABLE['nlp_search'] and session:
            try:
                enhanced_search = create_enhanced_search_system(
                    session=session,
                    user_context=user_context,
                    nlp_instance=nlp_instance
                )
                if enhanced_search:
                    search_system.unified_search_system = enhanced_search
                    logger.info("Unified search system created with NLP enhancement")
            except Exception as e:
                logger.warning(f"NLP enhancement failed, using basic search: {e}")

        return search_system

    except Exception as e:
        logger.error(f"Failed to create unified search system: {e}")
        return None


def get_best_available_search_system(session=None, user_context=None):
    """
    Get the best available search system based on installed components.

    Args:
        session: Database session
        user_context: Optional user context

    Returns:
        Best available search system instance
    """
    # Try unified search first (preferred)
    if _COMPONENTS_AVAILABLE['unified_search']:
        return create_unified_search_system(session, user_context)

    # Fall back to NLP search
    elif _COMPONENTS_AVAILABLE['nlp_search']:
        try:
            return create_enhanced_search_system(session, user_context)
        except Exception as e:
            logger.warning(f"NLP search creation failed: {e}")

    # Fall back to basic aggregate search
    elif _COMPONENTS_AVAILABLE['aggregate_search']:
        try:
            return AggregateSearch(session)
        except Exception as e:
            logger.warning(f"Aggregate search creation failed: {e}")

    # No search system available
    logger.error("No search system components available")
    return None


# ===== MODULE INITIALIZATION =====

def initialize_search_module():
    """Initialize the search module and log status."""
    status = get_search_system_status()

    logger.info(f"Search module initialized - Status: {status['operational_level']}")
    logger.info(f"Components: {status['available_count']}/{status['total_components']} available "
                f"({status['health_score']:.1f}% health)")

    if status['available_components']:
        logger.info(f"Available: {', '.join(status['available_components'])}")

    if status['unavailable_components']:
        logger.warning(f"⚠Unavailable: {', '.join(status['unavailable_components'])}")

    if status['recommendations']:
        logger.info("💡 Recommendations:")
        for rec in status['recommendations']:
            logger.info(f"   - {rec}")

    return status


# Initialize module on import
_MODULE_STATUS = initialize_search_module()

# Make status available at module level
MODULE_STATUS = _MODULE_STATUS
SEARCH_HEALTH_SCORE = _MODULE_STATUS['health_score']
OPERATIONAL_LEVEL = _MODULE_STATUS['operational_level']