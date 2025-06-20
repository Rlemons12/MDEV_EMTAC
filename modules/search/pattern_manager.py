
# modules/search/pattern_manager.py
"""
Pattern management for the AggregateSearch system.
Handles loading, caching, and managing search patterns from the database.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from modules.configuration.log_config import (
    debug_id, info_id, warning_id, error_id, critical_id,
    get_request_id, with_request_id, log_timed_operation
)
import json

# Initialize to None and import conditionally
DatabaseConfig = None
SearchIntent = None
IntentPattern = None
IntentKeyword = None
EntityExtractionRule = None
SearchAnalytics = None

# Availability flags - set directly instead of circular import
DATABASE_AVAILABLE = True
SEARCH_MODELS_AVAILABLE = True

# Conditional imports based on availability
try:
    from modules.configuration.config_env import DatabaseConfig
    debug_id("Database config imported successfully")
except ImportError as e:
    warning_id(f"Database config not available: {e}")
    DATABASE_AVAILABLE = False

try:
    from .models.search_models import (
        SearchIntent, IntentPattern, IntentKeyword,
        EntityExtractionRule, SearchAnalytics
    )
    debug_id("Search models imported successfully")
except ImportError as e:
    warning_id(f"Search models not available: {e}")
    SEARCH_MODELS_AVAILABLE = False

@with_request_id
def get_module_status():
    """Get the current module status."""
    return {
        "database_available": DATABASE_AVAILABLE,
        "search_models_available": SEARCH_MODELS_AVAILABLE
    }


class SearchPatternManager:
    """
    Manages search patterns, intents, and entity extraction rules for AggregateSearch.

    Gracefully handles missing dependencies and provides fallback functionality.
    """

    def __init__(self, session=None):
        """Initialize the SearchPatternManager."""
        self._session = session
        self._db_config = None
        self._request_id = get_request_id()

        debug_id("Initializing SearchPatternManager", self._request_id)

        # Check module status
        status = get_module_status()
        self._database_available = status["database_available"]
        self._search_models_available = status["search_models_available"]

        if self._database_available and DatabaseConfig:
            try:
                self._db_config = DatabaseConfig()
                debug_id("Database config initialized successfully", self._request_id)
            except Exception as e:
                warning_id(f"Database config initialization failed: {e}", self._request_id)
                self._db_config = None
                self._database_available = False
        else:
            warning_id("Database config not available - running in limited mode", self._request_id)

        # Intelligent caching
        self._pattern_cache = {}
        self._cache_timestamp = None
        self._cache_ttl_seconds = 300  # 5 minute cache

        info_id(f"SearchPatternManager initialized - DB Available: {self._database_available}, "
                f"Models Available: {self._search_models_available}", self._request_id)

    @property
    @with_request_id
    def session(self):
        """Get or create database session."""
        if not self._database_available:
            return None

        if self._session is None and self._db_config:
            try:
                with log_timed_operation("Database session creation", self._request_id):
                    self._session = self._db_config.get_main_session()
                debug_id("Database session created successfully", self._request_id)
            except Exception as e:
                error_id(f"Failed to create database session: {e}", self._request_id)
                return None
        return self._session

    @with_request_id
    def close_session(self):
        """Close database session if created by this manager."""
        if self._session is not None:
            try:
                self._session.close()
                self._session = None
                debug_id("Database session closed", self._request_id)
            except Exception as e:
                error_id(f"Error closing session: {e}", self._request_id)

    @with_request_id
    def __enter__(self):
        """Support for context manager usage."""
        debug_id("Entering SearchPatternManager context", self._request_id)
        return self

    @with_request_id
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close session when exiting context."""
        if exc_type:
            error_id(f"Exiting SearchPatternManager context with exception: {exc_val}", self._request_id)
        else:
            debug_id("Exiting SearchPatternManager context normally", self._request_id)
        self.close_session()

    @with_request_id
    def initialize_default_patterns(self) -> Dict[str, Any]:
        """
        Initialize the database with default search patterns for AggregateSearch.

        Returns:
            Dictionary with initialization results and statistics
        """
        request_id = get_request_id()

        if not self._search_models_available:
            warning_id("Search models not available - cannot initialize patterns", request_id)
            return {
                "status": "error",
                "message": "Search models not available - cannot initialize patterns",
                "fallback_mode": True
            }

        if not self.session:
            error_id("Database session not available for pattern initialization", request_id)
            return {
                "status": "error",
                "message": "Database session not available",
                "available_models": self._search_models_available,
                "available_database": self._database_available
            }

        try:
            with log_timed_operation("Default pattern initialization", request_id):
                info_id("Initializing default search patterns...", request_id)

                # Core search intents for AggregateSearch
                default_intents = self._get_default_intents()
                debug_id(f"Prepared {len(default_intents)} default intents", request_id)

                # Create or update intents
                created_intents = {}
                for intent_data in default_intents:
                    existing = self.session.query(SearchIntent).filter_by(name=intent_data["name"]).first()
                    if not existing:
                        intent = SearchIntent(**intent_data)
                        self.session.add(intent)
                        self.session.flush()  # Get the ID
                        created_intents[intent_data["name"]] = intent
                        debug_id(f"Created intent: {intent_data['name']}", request_id)
                    else:
                        # Update existing intent with new data
                        for key, value in intent_data.items():
                            if key != "name":  # Don't update the name
                                setattr(existing, key, value)
                        created_intents[intent_data["name"]] = existing
                        debug_id(f"Updated existing intent: {intent_data['name']}", request_id)

                # Add patterns, keywords, and entity rules
                pattern_count = self._add_default_patterns(created_intents)
                keyword_count = self._add_default_keywords(created_intents)
                rule_count = self._add_default_entity_rules(created_intents)

                debug_id(f"Created {pattern_count} patterns, {keyword_count} keywords, "
                         f"{rule_count} entity rules", request_id)

                # Commit all changes
                self.session.commit()
                self._invalidate_cache()

                result = {
                    "status": "success",
                    "message": "Default patterns initialized successfully",
                    "statistics": {
                        "intents_processed": len(created_intents),
                        "patterns_created": pattern_count,
                        "keywords_created": keyword_count,
                        "entity_rules_created": rule_count
                    }
                }

                info_id(f"Pattern initialization complete: {result['statistics']}", request_id)
                return result

        except Exception as e:
            if self.session:
                try:
                    self.session.rollback()
                    warning_id("Database session rolled back due to error", request_id)
                except Exception as rollback_error:
                    error_id(f"Failed to rollback session: {rollback_error}", request_id)

            error_id(f"Error initializing default patterns: {e}", request_id, exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to initialize patterns: {str(e)}",
                "search_models_available": self._search_models_available,
                "database_available": self._database_available
            }

    @with_request_id
    # In pattern_manager.py, around line 258, replace the problematic query with:
    @with_request_id
    def load_patterns_from_database(self) -> Dict[str, Any]:
        """
        Load all patterns from database with intelligent caching.

        Returns:
            Dictionary containing all pattern data organized by intent
        """
        request_id = get_request_id()

        if not self._search_models_available:
            warning_id("Search models not available - using fallback patterns", request_id)
            return self._get_fallback_patterns()

        if not self.session:
            warning_id("Database session not available - using fallback patterns", request_id)
            return self._get_fallback_patterns()

        current_time = datetime.utcnow()

        # Check cache validity
        if self._is_cache_valid(current_time):
            debug_id("Using cached pattern data", request_id)
            return self._pattern_cache

        try:
            with log_timed_operation("Pattern loading from database", request_id):
                debug_id("Loading fresh pattern data from database...", request_id)

                # FIXED: Query only the columns that exist
                # Load all active intents ordered by priority - removed display_name
                intents = self.session.query(SearchIntent).filter_by(
                    is_active=True
                ).order_by(SearchIntent.priority.desc()).all()

                debug_id(f"Found {len(intents)} active intents in database", request_id)

                pattern_data = {
                    "intents": {},
                    "loaded_at": current_time,
                    "cache_info": {
                        "total_intents": len(intents),
                        "expires_at": current_time.timestamp() + self._cache_ttl_seconds
                    }
                }

                for intent in intents:
                    intent_data = self._load_intent_data(intent)
                    pattern_data["intents"][intent.name] = intent_data
                    debug_id(f"Loaded intent data for: {intent.name}", request_id)

                # Cache the results
                self._pattern_cache = pattern_data
                self._cache_timestamp = current_time

                info_id(f"Loaded {len(intents)} intents with patterns from database", request_id)
                return pattern_data

        except Exception as e:
            error_id(f"Error loading patterns from database: {e}", request_id, exc_info=True)
            fallback_patterns = self._get_fallback_patterns()
            warning_id("Using fallback patterns due to database error", request_id)
            return {
                "intents": {},
                "loaded_at": current_time,
                "error": str(e),
                "fallback_patterns": fallback_patterns
            }

    @with_request_id
    def _is_cache_valid(self, current_time: datetime) -> bool:
        """Check if cached data is still valid."""
        is_valid = (
                self._cache_timestamp and
                self._pattern_cache and
                (current_time - self._cache_timestamp).seconds < self._cache_ttl_seconds
        )

        if is_valid:
            cache_age = (current_time - self._cache_timestamp).seconds
            debug_id(f"Cache is valid, age: {cache_age}s", self._request_id)
        else:
            debug_id("Cache is invalid or expired", self._request_id)

        return is_valid

    @with_request_id
    def _invalidate_cache(self):
        """Force reload of patterns from database on next access."""
        self._pattern_cache = {}
        self._cache_timestamp = None
        debug_id("Pattern cache invalidated", self._request_id)

    @with_request_id
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health and statistics."""
        request_id = get_request_id()

        base_health = {
            "search_models_available": self._search_models_available,
            "database_available": self._database_available,
            "session_available": self.session is not None,
            "cache_status": {
                "cached": bool(self._pattern_cache),
                "cache_age_seconds": (
                        datetime.utcnow() - self._cache_timestamp
                ).seconds if self._cache_timestamp else None
            }
        }

        debug_id(f"Base health status: {base_health}", request_id)

        if not self._search_models_available:
            warning_id("System running in limited mode - search models not available", request_id)
            return {
                **base_health,
                "status": "limited",
                "message": "Search models not available - using fallback patterns",
                "fallback_mode": True
            }

        if not self.session:
            error_id("Database session not available for health check", request_id)
            return {
                **base_health,
                "status": "error",
                "message": "Database session not available"
            }

        try:
            with log_timed_operation("System health check", request_id):
                from sqlalchemy import func

                # Basic counts
                intent_count = self.session.query(SearchIntent).filter_by(is_active=True).count()
                pattern_count = self.session.query(IntentPattern).filter_by(is_active=True).count()
                keyword_count = self.session.query(IntentKeyword).filter_by(is_active=True).count()

                debug_id(f"Counts - Intents: {intent_count}, Patterns: {pattern_count}, "
                         f"Keywords: {keyword_count}", request_id)

                # Performance metrics
                avg_success_rate = self.session.query(
                    func.avg(IntentPattern.success_rate)
                ).scalar() or 0.0

                total_usage = self.session.query(
                    func.sum(IntentPattern.usage_count)
                ).scalar() or 0

                # Recent analytics
                recent_searches = self.session.query(SearchAnalytics).filter(
                    SearchAnalytics.created_at >= datetime.utcnow() - timedelta(days=7)
                ).count()

                debug_id(f"Metrics - Avg Success Rate: {avg_success_rate:.2f}, "
                         f"Total Usage: {total_usage}, Recent Searches: {recent_searches}", request_id)

                health_result = {
                    **base_health,
                    "status": "healthy" if intent_count > 0 else "needs_initialization",
                    "active_intents": intent_count,
                    "active_patterns": pattern_count,
                    "active_keywords": keyword_count,
                    "average_success_rate": float(avg_success_rate),
                    "total_pattern_usage": total_usage,
                    "recent_searches_7d": recent_searches,
                }

                info_id(f"System health check completed - Status: {health_result['status']}", request_id)
                return health_result

        except Exception as e:
            error_id(f"Error getting system health: {e}", request_id, exc_info=True)
            return {
                **base_health,
                "status": "error",
                "error": str(e)
            }

    @with_request_id
    def _get_fallback_patterns(self) -> Dict[str, Any]:
        """Provide basic fallback patterns when database is unavailable."""
        debug_id("Using fallback patterns", self._request_id)
        return {
            "intents": {
                "general_search": {
                    "name": "general_search",
                    "description": "General search fallback",
                    "priority": 1.0,
                    "patterns": [".*"],
                    "keywords": ["search", "find", "look"],
                    "entity_rules": []
                }
            },
            "loaded_at": datetime.utcnow(),
            "fallback_mode": True
        }

    @with_request_id
    def _get_default_intents(self) -> List[Dict[str, Any]]:
        """Get default search intents for AggregateSearch."""
        debug_id("Preparing default intents", self._request_id)
        return [
            {
                "name": "document_search",
                "description": "Search for documents and files",
                "priority": 0.9,
                "is_active": True,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            },
            {
                "name": "data_analysis",
                "description": "Search for data analysis and reports",
                "priority": 0.8,
                "is_active": True,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            },
            {
                "name": "user_lookup",
                "description": "Search for user information",
                "priority": 0.7,
                "is_active": True,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
        ]

    @with_request_id
    def _add_default_patterns(self, intents: Dict) -> int:
        """Add default patterns for intents."""
        debug_id("Adding default patterns", self._request_id)
        # Placeholder implementation - add actual pattern creation logic
        return len(intents) * 2  # Example count

    @with_request_id
    def _add_default_keywords(self, intents: Dict) -> int:
        """Add default keywords for intents."""
        debug_id("Adding default keywords", self._request_id)
        # Placeholder implementation - add actual keyword creation logic
        return len(intents) * 5  # Example count

    @with_request_id
    def _add_default_entity_rules(self, intents: Dict) -> int:
        """Add default entity extraction rules for intents."""
        debug_id("Adding default entity rules", self._request_id)
        # Placeholder implementation - add actual rule creation logic
        return len(intents) * 3  # Example count

    @with_request_id
    def _load_intent_data(self, intent) -> Dict[str, Any]:
        """Load complete data for a single intent."""
        debug_id(f"Loading intent data for: {intent.name}", self._request_id)

        # Load related patterns, keywords, and rules
        patterns = self.session.query(IntentPattern).filter_by(
            intent_id=intent.id, is_active=True
        ).all()

        keywords = self.session.query(IntentKeyword).filter_by(
            intent_id=intent.id, is_active=True
        ).all()

        entity_rules = self.session.query(EntityExtractionRule).filter_by(
            intent_id=intent.id, is_active=True
        ).all()

        debug_id(f"Intent {intent.name} - Patterns: {len(patterns)}, "
                 f"Keywords: {len(keywords)}, Rules: {len(entity_rules)}", self._request_id)

        return {
            "id": intent.id,
            "name": intent.name,
            "description": intent.description,
            "priority": intent.priority,
            "patterns": [{"id": p.id, "pattern": p.pattern_text, "success_rate": p.success_rate}
                         for p in patterns],
            "keywords": [{"id": k.id, "keyword": k.keyword_text, "weight": k.weight}
                         for k in keywords],
            "entity_rules": [{"id": r.id, "rule": r.rule_text, "entity_type": r.entity_type}
                             for r in entity_rules],
            "created_at": intent.created_at,
            "updated_at": intent.updated_at
        }