# modules/search/UnifiedSearchMixin.py
"""
Unified search interface that provides comprehensive, organized results
for manufacturing and maintenance queries.

Handles natural language queries like:
- "What's in room 2312?" → Returns organized parts, images, equipment
- "What does part 123131 look like?" → Returns images and usage locations
- "Show me pump maintenance procedures" → Returns organized maintenance content
- "Find motor repair documentation" → Returns relevant manuals and guides

Automatically detects query intent and organizes results by entity type
(images, documents, parts, equipment, procedures) with quick actions.
"""

import time
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import search system components
from .nlp_search import SpaCyEnhancedAggregateSearch
from modules.search.aggregate_search import AggregateSearch
from modules.search.pattern_manager import SearchPatternManager

# Import utilities
try:
    from modules.configuration.log_config import log_timed_operation
except ImportError:
    # Fallback if log_timed_operation is not available
    from contextlib import contextmanager


    @contextmanager
    def log_timed_operation(operation_name):
        yield

logger = logging.getLogger(__name__)


class UnifiedSearchMixin:
    """Enhanced UnifiedSearchMixin with working query tracking"""

    def __init__(self):
        """Initialize the unified search capabilities."""
        logger.info("🔧 Initializing UnifiedSearchMixin...")

        # Initialize the comprehensive unified search system
        self.unified_search_system = None
        self.search_pattern_manager = None

        # Query patterns that indicate unified search intent
        self.unified_search_patterns = [
            # Location-based queries
            r'what[\'s\s]+in\s+(room|area|zone|section|location)\s*([A-Z0-9]+)',
            r'show\s+me\s+(everything|all|what[\'s]*)\s+in\s+([A-Z0-9\s]+)',
            r'list\s+(contents|items|equipment)\s+(in|at|from)\s+([A-Z0-9\s]+)',

            # Part-based queries (improved to handle descriptions)
            r'(?:i\s+)?need\s+(?:the\s+)?part\s+number\s+for\s+(.+)',
            r'what\s+(?:is\s+)?(?:the\s+)?part\s+number\s+for\s+(.+)',
            r'part\s+number\s+for\s+(.+)',
            r'what\s+does\s+part\s+([A-Za-z0-9\-\.]+)\s+look\s+like',
            r'show\s+(me\s+)?part\s+([A-Za-z0-9\-\.]+)',
            r'find\s+(all\s+)?(.+)',
            r'show\s+(all\s+)?(.+)'
        ]

        # Initialize the search system
        self._init_unified_search()

        logger.info(" UnifiedSearchMixin initialized")

    def _init_query_tracking(self):
        """Initialize query tracking for the unified search system - FIXED IMPORTS."""
        try:
            session = getattr(self, 'db_session', None)
            if not session:
                logger.warning(" No database session - query tracking disabled")
                self.query_tracker = None
                self.tracked_search = None
                return

            logger.info("🔧 Starting query tracking initialization...")

            # FIXED: Correct import paths
            try:
                from modules.search.models.search_models import SearchQueryTracker, UnifiedSearchWithTracking
                logger.info(" Tracking modules imported successfully")
            except ImportError as import_error:
                logger.error(f" Failed to import tracking modules: {import_error}")
                self.query_tracker = None
                self.tracked_search = None
                return

            # Create the tracker
            try:
                self.query_tracker = SearchQueryTracker(session)
                logger.info(" SearchQueryTracker initialized")
            except Exception as tracker_error:
                logger.error(f" Failed to create SearchQueryTracker: {tracker_error}")
                self.query_tracker = None
                self.tracked_search = None
                return

            # Create the tracking wrapper around your unified search
            try:
                self.tracked_search = UnifiedSearchWithTracking(self)  # Pass self as unified_search
                self.tracked_search.query_tracker = self.query_tracker
                logger.info(" UnifiedSearchWithTracking initialized")
            except Exception as wrapper_error:
                logger.error(f" Failed to create tracking wrapper: {wrapper_error}")
                self.query_tracker = None
                self.tracked_search = None
                return

            # Test that the tracking method exists
            if hasattr(self.tracked_search, 'execute_unified_search_with_tracking'):
                logger.info(" Tracking method available")
                logger.info(" QUERY TRACKING FULLY INITIALIZED!")
            else:
                logger.error(" Tracking method not found")
                self.query_tracker = None
                self.tracked_search = None

        except Exception as e:
            logger.error(f" Query tracking initialization failed: {e}")
            self.query_tracker = None
            self.tracked_search = None

    def _init_unified_search(self):
        """Initialize the unified search system - ROBUST VERSION"""
        try:
            # Use the session from this AistManager instance
            session = getattr(self, 'db_session', None)
            logger.info(f"🔧 Initializing unified search with session: {session is not None}")

            # Try multiple search system options in order of preference
            search_system_initialized = False

            # Option 1: Try SpaCyEnhancedAggregateSearch
            if not search_system_initialized:
                try:
                    from modules.search.nlp_search import SpaCyEnhancedAggregateSearch
                    self.unified_search_system = SpaCyEnhancedAggregateSearch(session=session)
                    logger.info(" SpaCyEnhancedAggregateSearch initialized successfully")
                    search_system_initialized = True
                except Exception as e:
                    logger.warning(f" SpaCyEnhancedAggregateSearch failed: {e}")

            # Option 2: Try basic AggregateSearch as fallback
            if not search_system_initialized:
                try:
                    from modules.search.aggregate_search import AggregateSearch
                    self.unified_search_system = AggregateSearch(session=session)
                    logger.warning(" Using basic AggregateSearch as fallback")
                    search_system_initialized = True
                except Exception as e:
                    logger.error(f" Basic AggregateSearch also failed: {e}")

            # Option 3: Create a simple fallback search system
            if not search_system_initialized:
                logger.error(" All search systems failed - creating minimal fallback")
                self.unified_search_system = self._create_minimal_search_system(session)
                search_system_initialized = True

            # Initialize pattern manager if possible
            try:
                from modules.search.pattern_manager import SearchPatternManager
                self.search_pattern_manager = SearchPatternManager(session=session)
                logger.debug(" Search pattern manager initialized")
            except Exception as e:
                logger.warning(f" Search pattern manager initialization failed: {e}")
                self.search_pattern_manager = None

            logger.info(
                f" Search system final status: {'AVAILABLE' if self.unified_search_system else 'NOT AVAILABLE'}")

        except Exception as e:
            logger.error(f" Failed to initialize unified search system: {e}")
            # Last resort - create minimal search system
            try:
                self.unified_search_system = self._create_minimal_search_system(getattr(self, 'db_session', None))
                logger.warning(" Using minimal search system as last resort")
            except Exception as final_error:
                logger.error(f" Even minimal search system failed: {final_error}")
                self.unified_search_system = None

    def _create_minimal_search_system(self, session):
        """Create a minimal search system that always works"""

        class MinimalSearchSystem:
            def __init__(self, session):
                self.session = session
                logger.info(" Minimal search system created")

            def execute_aggregated_search(self, query):
                logger.info(f" Minimal search executing: {query}")
                return {
                    'status': 'success',
                    'results': [],
                    'total_results': 0,
                    'message': f"Minimal search completed for: {query}",
                    'search_method': 'minimal_fallback'
                }

            def execute_nlp_aggregated_search(self, query):
                return self.execute_aggregated_search(query)

        return MinimalSearchSystem(session)

    def execute_unified_search(self, question: str, user_id: str = None, request_id: str = None) -> Dict[str, Any]:
        """
        FIXED: Execute unified search with correct method names
        """
        import time

        search_start = time.time()
        logger.info(f"🔍 Executing unified search for: {question}")

        # Check if we have a search system
        if not self.unified_search_system:
            logger.error("❌ No search system available!")
            return {
                'status': 'error',
                'message': 'Search system not available',
                'search_type': 'unified',
                'query': question,
                'total_results': 0,
                'results_by_type': {},
                'summary': f"I'm sorry, but the search system is currently not available. Please try again later.",
                'quick_actions': [],
                'related_searches': [],
                'timestamp': datetime.utcnow().isoformat(),
                'search_method': 'system_unavailable'
            }

        # Execute search with the available system
        try:
            system_type = type(self.unified_search_system).__name__
            logger.info(f"🔍 Using search system: {system_type}")

            # FIXED: Try correct method names based on system type
            search_result = None

            if hasattr(self.unified_search_system, 'execute_nlp_aggregated_search'):
                # SpaCyEnhancedAggregateSearch uses this method
                logger.info("🔍 Using execute_nlp_aggregated_search method")
                search_result = self.unified_search_system.execute_nlp_aggregated_search(question)
            elif hasattr(self.unified_search_system, 'execute_aggregated_search'):
                # Basic AggregateSearch uses this method
                logger.info("🔍 Using execute_aggregated_search method")
                search_result = self.unified_search_system.execute_aggregated_search(question)
            else:
                # MinimalSearchSystem fallback
                logger.info("🔍 Using fallback search method")
                search_result = {
                    'status': 'success',
                    'results': [],
                    'total_results': 0,
                    'message': f"Search completed for: {question}",
                    'search_method': 'fallback'
                }

            # Organize and enhance results
            if search_result and search_result.get('status') == 'success':
                enhanced_result = self._organize_unified_results(search_result, question)
                search_time = time.time() - search_start
                enhanced_result['search_time_ms'] = int(search_time * 1000)
                logger.info(
                    f"✅ Search completed successfully in {search_time:.2f}s with {enhanced_result.get('total_results', 0)} results")
                return enhanced_result
            else:
                logger.warning(f"⚠️ Search returned no results for: {question}")
                return self._no_unified_results_response(question, search_result)

        except Exception as e:
            search_time = time.time() - search_start
            logger.error(f"❌ Search failed after {search_time:.3f}s: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': f"Search error: {str(e)}",
                'search_type': 'unified',
                'query': question,
                'total_results': 0,
                'results_by_type': {},
                'summary': f"I encountered an error while searching for '{question}'. Please try again.",
                'timestamp': datetime.utcnow().isoformat(),
                'search_method': 'error_fallback'
            }
    # FIXED VERSION - Database-Safe Intent Detection
    # This version handles missing columns and transaction failures gracefully

    def is_unified_search_query(self, question: str) -> bool:
        """
        FIXED: Database-driven intent detection with proper error handling.
        Handles missing columns and transaction failures gracefully.
        """
        question_lower = question.lower().strip()

        # Strategy 1: Try database patterns with error handling
        try:
            database_result = self._query_intent_patterns_safe(question_lower)
            if database_result['is_search_query']:
                logger.info(
                    f"Database intent detected: {database_result['intent_name']} (confidence: {database_result['confidence']:.2f})")
                return True
        except Exception as e:
            logger.warning(f" Database intent detection failed: {e}")
            # Continue to fallback

        # Strategy 2: Enhanced fallback (includes manufacturer detection)
        logger.debug(" Using enhanced fallback patterns")
        return self._enhanced_fallback_detection(question_lower)

    def _query_intent_patterns_safe(self, question_lower: str) -> Dict[str, Any]:
        """
        FIXED: Safe version that handles database errors and missing columns.
        """
        try:
            # Check database access
            if not hasattr(self, 'db_session') or not self.db_session:
                return {'is_search_query': False, 'database_available': False}

            # Rollback any existing transaction to ensure clean state
            try:
                self.db_session.rollback()
            except:
                pass

            from modules.search.models.search_models import SearchIntent

            # Get active intents - simple query first
            active_intents = self.db_session.query(SearchIntent).filter(
                SearchIntent.is_active == True
            ).order_by(SearchIntent.priority.desc()).all()

            logger.debug(f" Testing against {len(active_intents)} active search intents")

            best_match = None
            best_confidence = 0.0

            for intent in active_intents:
                try:
                    # Test patterns for this intent (with error handling)
                    pattern_confidence = self._test_regex_patterns_safe(question_lower, intent)

                    # Test keywords for this intent (with error handling)
                    keyword_confidence = self._test_keyword_matches_safe(question_lower, intent)

                    # Combined confidence
                    combined_confidence = max(pattern_confidence, keyword_confidence * 0.8)

                    if combined_confidence > best_confidence:
                        best_confidence = combined_confidence
                        best_match = {
                            'intent': intent,
                            'confidence': combined_confidence,
                            'pattern_confidence': pattern_confidence,
                            'keyword_confidence': keyword_confidence
                        }

                    logger.debug(
                        f"Intent '{intent.name}': pattern={pattern_confidence:.2f}, keyword={keyword_confidence:.2f}, combined={combined_confidence:.2f}")

                except Exception as intent_error:
                    logger.warning(f" Error testing intent {intent.name}: {intent_error}")
                    # Continue with next intent instead of failing completely
                    continue

            # Determine result
            threshold = 0.6

            if best_match and best_confidence >= threshold:
                intent = best_match['intent']
                return {
                    'is_search_query': True,
                    'intent_name': intent.name,
                    'intent_id': intent.id,
                    'confidence': best_confidence,
                    'search_method': intent.search_method,
                    'database_available': True,
                    'method': 'database_intent_detection_safe'
                }
            else:
                return {
                    'is_search_query': False,
                    'best_confidence': best_confidence,
                    'threshold': threshold,
                    'database_available': True,
                    'tested_intents': len(active_intents)
                }

        except Exception as e:
            logger.error(f" Database intent detection failed: {e}")
            # Always rollback on error
            try:
                if self.db_session:
                    self.db_session.rollback()
            except:
                pass

            return {
                'is_search_query': False,
                'database_available': False,
                'error': str(e)
            }

    def _test_regex_patterns_safe(self, question_lower: str, intent) -> float:
        """
        FIXED: Safe version that handles transaction failures.
        """
        max_confidence = 0.0

        try:
            # Ensure fresh transaction state
            try:
                self.db_session.rollback()
            except:
                pass

            # Access patterns with error handling
            patterns = []
            try:
                patterns = list(intent.patterns)  # Convert to list to avoid lazy loading issues
            except Exception as pattern_load_error:
                logger.warning(f" Could not load patterns for intent {intent.name}: {pattern_load_error}")
                return 0.0

            for pattern in patterns:
                try:
                    if not getattr(pattern, 'is_active', True):
                        continue

                    pattern_text = getattr(pattern, 'pattern_text', '')
                    if not pattern_text:
                        continue

                    # Test the regex pattern
                    match = re.search(pattern_text, question_lower, re.IGNORECASE)

                    if match:
                        # Calculate confidence
                        success_rate = getattr(pattern, 'success_rate', 0.8)
                        base_confidence = success_rate if success_rate > 0 else 0.8
                        usage_count = getattr(pattern, 'usage_count', 0)
                        usage_boost = min(0.1, usage_count / 1000)

                        pattern_confidence = min(1.0, base_confidence + usage_boost)

                        if pattern_confidence > max_confidence:
                            max_confidence = pattern_confidence

                        logger.debug(
                            f" Pattern '{pattern_text[:50]}...' matched with confidence {pattern_confidence:.2f}")

                except re.error as regex_error:
                    logger.warning(f" Invalid regex pattern: {regex_error}")
                    continue
                except Exception as pattern_error:
                    logger.warning(f" Error testing pattern: {pattern_error}")
                    continue

        except Exception as e:
            logger.warning(f" Error testing regex patterns for intent {intent.name}: {e}")

        return max_confidence

    def _test_keyword_matches_safe(self, question_lower: str, intent) -> float:
        """
        FIXED: Safe version that handles missing columns.
        """
        try:
            # Ensure fresh transaction state
            try:
                self.db_session.rollback()
            except:
                pass

            total_weight = 0.0
            matched_weight = 0.0

            # Access keywords with error handling
            keywords = []
            try:
                keywords = list(intent.keywords)  # Convert to list to avoid lazy loading
            except Exception as keyword_load_error:
                logger.warning(f" Could not load keywords for intent {intent.name}: {keyword_load_error}")
                return 0.0

            for keyword in keywords:
                try:
                    if not getattr(keyword, 'is_active', True):
                        continue

                    weight = getattr(keyword, 'weight', 1.0)
                    keyword_text = getattr(keyword, 'keyword_text', '')

                    if not keyword_text:
                        continue

                    total_weight += weight
                    keyword_text_lower = keyword_text.lower()

                    # FIXED: Handle missing is_exact_match column
                    is_exact_match = getattr(keyword, 'is_exact_match', False)  # Default to False

                    if is_exact_match:
                        # Word boundary match
                        if re.search(r'\b' + re.escape(keyword_text_lower) + r'\b', question_lower):
                            matched_weight += weight
                            logger.debug(f" Exact keyword match: '{keyword_text}' (weight: {weight})")
                    else:
                        # Partial match
                        if keyword_text_lower in question_lower:
                            matched_weight += weight
                            logger.debug(f" Partial keyword match: '{keyword_text}' (weight: {weight})")

                except Exception as keyword_error:
                    logger.warning(f" Error testing keyword: {keyword_error}")
                    continue

            # Calculate confidence
            confidence = matched_weight / total_weight if total_weight > 0 else 0.0

            if confidence > 0:
                logger.debug(
                    f" Keyword confidence for intent '{intent.name}': {matched_weight}/{total_weight} = {confidence:.2f}")

            return confidence

        except Exception as e:
            logger.warning(f" Error testing keywords for intent {intent.name}: {e}")
            return 0.0

    def _enhanced_fallback_detection(self, question_lower: str) -> bool:
        """
        ENHANCED: Better fallback that handles manufacturer + product queries.
        This runs when database is unavailable or fails.
        """
        logger.debug(" Using enhanced fallback patterns")

        # Essential search patterns including manufacturer detection
        essential_patterns = [
            # Part number patterns
            r'part\s+number\s+for',  # "part number for"
            r'find\s+part',  # "find part"
            r'what\s+does\s+part',  # "what does part"
            r'show\s+me.*part',  # "show me part"

            # Manufacturer + product patterns (NEW!)
            r'([A-Za-z]+)\s+(sensors?|motors?|valves?|pumps?|bearings?|switches?|seals?|filters?)',  # "Banner sensors"
            r'show\s+me\s+([A-Za-z]+)\s+(sensors?|motors?|valves?)',  # "show me Banner sensors"
            r'([A-Za-z]+)\s+(parts?|components?)',  # "Banner parts"

            # Direct part number patterns
            r'[A-Z]\d{5,}',  # A115957 style part numbers
            r'\d{6,}',  # 6+ digit numbers

            # Location patterns
            r'what.*in\s+room',  # "what's in room"
            r'show.*in\s+(room|area|zone)',  # "show in room"

            # Image patterns
            r'show.*images',  # "show images"
            r'pictures\s+of',  # "pictures of"

            # Equipment category patterns (NEW!)
            r'(?:i\'?m\s+)?(?:looking\s+for|need|want|find)\s+(?:a\s+|an\s+|some\s+)?(motors?|valves?|pumps?|bearings?|switches?|seals?)',
            r'(?:show\s+me\s+|get\s+me\s+)?(motors?|valves?|pumps?|bearings?|switches?|seals?)',
        ]

        # Test each pattern
        for pattern in essential_patterns:
            try:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    logger.debug(f" Enhanced fallback pattern matched: {pattern}")
                    return True
            except re.error as regex_error:
                logger.warning(f" Invalid fallback pattern '{pattern}': {regex_error}")
                continue

        # Additional heuristics for manufacturer detection
        if self._detect_manufacturer_query_fallback(question_lower):
            logger.debug(" Manufacturer query detected via fallback")
            return True

        logger.debug(" No enhanced fallback patterns matched")
        return False

    def _detect_manufacturer_query_fallback(self, question_lower: str) -> bool:
        """
        ENHANCED: Detect manufacturer queries when database is unavailable.
        """
        # Known manufacturers (you can expand this list)
        manufacturers = [
            'banner', 'siemens', 'allen bradley', 'ab', 'schneider', 'omron', 'keyence',
            'sick', 'pepperl', 'fuchs', 'turck', 'ifm', 'balluff', 'festo', 'smc',
            'parker', 'rexroth', 'eaton', 'murphy', 'woodward', 'honeywell'
        ]

        # Equipment types
        equipment_types = [
            'sensor', 'sensors', 'motor', 'motors', 'valve', 'valves', 'pump', 'pumps',
            'bearing', 'bearings', 'switch', 'switches', 'relay', 'relays', 'starter',
            'starters', 'contactor', 'contactors', 'drive', 'drives'
        ]

        # Check for manufacturer + equipment combination
        words = question_lower.split()
        has_manufacturer = any(mfg in words for mfg in manufacturers)
        has_equipment = any(equip in words for equip in equipment_types)

        # Search indicators
        search_indicators = ['show', 'find', 'get', 'list', 'display', 'we have', 'available']
        has_search_context = any(indicator in question_lower for indicator in search_indicators)

        # Return true if we have manufacturer + equipment + search context
        result = has_manufacturer and has_equipment and has_search_context

        if result:
            logger.debug(
                f" Manufacturer+equipment detected: mfg={has_manufacturer}, equip={has_equipment}, search={has_search_context}")

        return result

    # DEBUGGING METHOD - Add this too
    def debug_intent_detection_safe(self, question: str) -> Dict[str, Any]:
        """
        SAFE: Debug intent detection without causing database errors.
        """
        print(f" DEBUGGING INTENT DETECTION FOR: '{question}'")
        print("=" * 60)

        question_lower = question.lower().strip()

        # Test database patterns safely
        try:
            database_result = self._query_intent_patterns_safe(question_lower)
            print(f" Database Available: {database_result.get('database_available', False)}")
            print(f" Search Query Detected: {database_result.get('is_search_query', False)}")

            if database_result.get('is_search_query'):
                print(f" Intent: {database_result.get('intent_name', 'Unknown')}")
                print(f" Confidence: {database_result.get('confidence', 0):.3f}")
            else:
                print(f" No intent detected (best: {database_result.get('best_confidence', 0):.3f})")

        except Exception as db_error:
            print(f" Database error: {db_error}")
            database_result = {'is_search_query': False, 'database_available': False}

        # Test fallback
        fallback_result = self._enhanced_fallback_detection(question_lower)
        print(f" Enhanced Fallback Result: {fallback_result}")

        # Final decision
        final_decision = database_result.get('is_search_query', False) or fallback_result
        print(f" FINAL DECISION: {'SEARCH QUERY' if final_decision else 'CHAT QUERY'}")
        print()

        return {
            'original_question': question,
            'database_result': database_result,
            'fallback_result': fallback_result,
            'final_decision': final_decision
        }

    def _query_intent_patterns(self, question_lower: str) -> Dict[str, Any]:
        """
        Query the SearchIntent and IntentPattern tables to detect search intent.

        Returns:
            Dict with is_search_query, intent_name, confidence, method
        """
        try:
            # Check if we have database access
            if not hasattr(self, 'db_session') or not self.db_session:
                logger.warning("No database session available for intent detection")
                return {'is_search_query': False, 'database_available': False}

            # Import your models
            from modules.search.models.search_models import SearchIntent, IntentPattern, IntentKeyword

            # Get all active search intents (ordered by priority)
            active_intents = self.db_session.query(SearchIntent).filter(
                SearchIntent.is_active == True
            ).order_by(SearchIntent.priority.desc()).all()

            logger.debug(f" Testing against {len(active_intents)} active search intents")

            best_match = None
            best_confidence = 0.0

            for intent in active_intents:
                # Test regex patterns for this intent
                pattern_confidence = self._test_regex_patterns(question_lower, intent)

                # Test keywords for this intent
                keyword_confidence = self._test_keyword_matches(question_lower, intent)

                # Combined confidence (patterns have higher weight)
                combined_confidence = max(pattern_confidence, keyword_confidence * 0.8)

                if combined_confidence > best_confidence:
                    best_confidence = combined_confidence
                    best_match = {
                        'intent': intent,
                        'confidence': combined_confidence,
                        'pattern_confidence': pattern_confidence,
                        'keyword_confidence': keyword_confidence
                    }

                    logger.debug(
                        f"Intent '{intent.name}': pattern={pattern_confidence:.2f}, keyword={keyword_confidence:.2f}, combined={combined_confidence:.2f}")

            # Determine if we have a strong enough match
            threshold = 0.6  # Minimum confidence for search intent

            if best_match and best_confidence >= threshold:
                intent = best_match['intent']

                # Update pattern usage statistics
                self._update_pattern_usage_stats(intent, question_lower, best_match)

                return {
                    'is_search_query': True,
                    'intent_name': intent.name,
                    'intent_id': intent.id,
                    'confidence': best_confidence,
                    'search_method': intent.search_method,
                    'pattern_confidence': best_match['pattern_confidence'],
                    'keyword_confidence': best_match['keyword_confidence'],
                    'database_available': True,
                    'method': 'database_intent_detection'
                }
            else:
                logger.debug(f" No intent above threshold: best={best_confidence:.2f}, threshold={threshold}")
                return {
                    'is_search_query': False,
                    'best_confidence': best_confidence,
                    'threshold': threshold,
                    'database_available': True,
                    'tested_intents': len(active_intents)
                }

        except Exception as e:
            logger.error(f" Database intent detection failed: {e}")
            # Rollback transaction if needed
            try:
                if self.db_session:
                    self.db_session.rollback()
            except:
                pass

            return {
                'is_search_query': False,
                'database_available': False,
                'error': str(e)
            }

    def _test_regex_patterns(self, question_lower: str, intent) -> float:
        """
        Test question against all regex patterns for an intent.

        Returns the highest confidence from any matching pattern.
        """
        max_confidence = 0.0

        try:
            for pattern in intent.patterns:
                if not pattern.is_active:
                    continue

                try:
                    # Test the regex pattern
                    match = re.search(pattern.pattern_text, question_lower, re.IGNORECASE)

                    if match:
                        # Calculate confidence based on pattern's success rate
                        base_confidence = pattern.success_rate if pattern.success_rate > 0 else 0.8

                        # Boost confidence for patterns with higher usage
                        usage_boost = min(0.1, pattern.usage_count / 1000)  # Small boost for frequently used patterns

                        pattern_confidence = min(1.0, base_confidence + usage_boost)

                        if pattern_confidence > max_confidence:
                            max_confidence = pattern_confidence

                        logger.debug(
                            f" Pattern '{pattern.pattern_text[:50]}...' matched with confidence {pattern_confidence:.2f}")

                        # Update pattern usage (async to avoid blocking)
                        try:
                            pattern.usage_count = (pattern.usage_count or 0) + 1
                            # Don't commit here - let the main transaction handle it
                        except Exception:
                            pass  # Don't fail on usage tracking

                except re.error as regex_error:
                    logger.warning(f" Invalid regex pattern '{pattern.pattern_text}': {regex_error}")
                    continue
                except Exception as pattern_error:
                    logger.warning(f" Error testing pattern {pattern.id}: {pattern_error}")
                    continue

        except Exception as e:
            logger.warning(f" Error testing regex patterns for intent {intent.name}: {e}")

        return max_confidence

    def _test_keyword_matches(self, question_lower: str, intent) -> float:
        """
        Test question against all keywords for an intent.

        Returns confidence based on weighted keyword matches.
        """
        try:
            total_weight = 0.0
            matched_weight = 0.0

            for keyword in intent.keywords:
                if not keyword.is_active:
                    continue

                total_weight += keyword.weight
                keyword_text = keyword.keyword_text.lower()

                # Test for match based on exact_match setting
                if keyword.is_exact_match:
                    # Word boundary match
                    if re.search(r'\b' + re.escape(keyword_text) + r'\b', question_lower):
                        matched_weight += keyword.weight
                        logger.debug(f" Exact keyword match: '{keyword_text}' (weight: {keyword.weight})")
                else:
                    # Partial match
                    if keyword_text in question_lower:
                        matched_weight += keyword.weight
                        logger.debug(f" Partial keyword match: '{keyword_text}' (weight: {keyword.weight})")

            # Calculate confidence as percentage of matched weight
            confidence = matched_weight / total_weight if total_weight > 0 else 0.0

            if confidence > 0:
                logger.debug(
                    f" Keyword confidence for intent '{intent.name}': {matched_weight}/{total_weight} = {confidence:.2f}")

            return confidence

        except Exception as e:
            logger.warning(f" Error testing keywords for intent {intent.name}: {e}")
            return 0.0

    def _update_pattern_usage_stats(self, intent, question: str, match_info: Dict):
        """
        Update usage statistics for the matched intent and patterns.
        This helps improve the system over time.
        """
        try:
            # Update intent usage (if you want to track this)
            # intent.usage_count = (intent.usage_count or 0) + 1

            # Could also log to SearchAnalytics table here
            from modules.search.models.search_models import SearchAnalytics

            analytics = SearchAnalytics(
                query_text=question,
                detected_intent=intent.name,
                intent_confidence=match_info['confidence'],
                search_method=intent.search_method,
                success=True  # We'll update this later based on search results
            )

            self.db_session.add(analytics)
            # Don't commit here - let the main transaction handle it

            logger.debug(f" Updated usage stats for intent '{intent.name}'")

        except Exception as e:
            logger.warning(f" Failed to update usage stats: {e}")

    def _minimal_fallback_detection(self, question_lower: str) -> bool:
        """
        Minimal fallback when database is unavailable.
        Only the most essential patterns to keep the system working.
        """
        logger.warning(" Using minimal fallback patterns (database unavailable)")

        essential_patterns = [
            r'part\s+number\s+for',  # "part number for"
            r'find\s+part',  # "find part"
            r'what\s+does\s+part',  # "what does part"
            r'show\s+me.*part',  # "show me part"
            r'[A-Z]\d{5,}',  # A115957 style part numbers
            r'what.*in\s+room',  # "what's in room"
            r'show.*images',  # "show images"
            r'([A-Za-z]+)\s+(sensors?|motors?|valves?|pumps?)',  # "Banner sensors"
        ]

        for pattern in essential_patterns:
            if re.search(pattern, question_lower, re.IGNORECASE):
                logger.debug(f" Fallback pattern matched: {pattern}")
                return True

        logger.debug(" No fallback patterns matched")
        return False

    def test_intent_detection(self, test_queries: List[str]) -> Dict[str, Any]:
        """
        Test the intent detection system with a list of queries.
        Useful for debugging and validating your database patterns.

        Args:
            test_queries: List of test questions to analyze

        Returns:
            Dict with results for each query
        """
        results = {}

        print(" Testing Intent Detection with Database Patterns:")
        print("=" * 60)

        for query in test_queries:
            result = self._query_intent_patterns(query.lower())
            results[query] = {
                'is_search_query': result.get('is_search_query', False),
                'intent_name': result.get('intent_name', 'None'),
                'confidence': result.get('confidence', 0.0),
                'search_method': result.get('search_method', 'None'),
                'method': result.get('method', 'unknown')
            }

            # Format output
            status = " SEARCH" if results[query]['is_search_query'] else " CHAT"
            intent = results[query]['intent_name']
            confidence = results[query]['confidence']

            print(f"{status} | '{query}'")
            print(f"      Intent: {intent} (confidence: {confidence:.2f})")
            print()

        return results

    def run_pattern_tests(self):
        """
        Test some example queries to see how the database patterns work.
        Call this method to validate your pattern setup.
        """
        test_queries = [
            # Parts searches that should be detected
            "Banner sensors we have",
            "What's the part number for BEARING ASSEMBLY",
            "Siemens motors available",
            "find part A115957",
            "I need the part number for VALVE BYPASS",
            "show me Omron switches",

            # Location searches that should be detected
            "what's in room 2312",
            "show me everything in area B",

            # Image searches that should be detected
            "show me images of pumps",
            "pictures of motor assembly",

            # Conversational queries that should NOT be detected
            "How are you today?",
            "What's the weather like?",
            "Tell me a joke",
            "How do I reset my password?"
        ]

        results = self.test_intent_detection(test_queries)

        # Summary
        search_queries = sum(1 for r in results.values() if r['is_search_query'])
        chat_queries = len(test_queries) - search_queries

        print(" SUMMARY:")
        print(f"   Search queries detected: {search_queries}/{len(test_queries)}")
        print(f"   Chat queries detected: {chat_queries}/{len(test_queries)}")
        print()

        # Show any unexpected results
        unexpected = []
        expected_search = [
            "Banner sensors we have", "What's the part number for BEARING ASSEMBLY",
            "Siemens motors available", "find part A115957",
            "I need the part number for VALVE BYPASS", "show me Omron switches",
            "what's in room 2312", "show me everything in area B",
            "show me images of pumps", "pictures of motor assembly"
        ]

        expected_chat = [
            "How are you today?", "What's the weather like?",
            "Tell me a joke", "How do I reset my password?"
        ]

        for query, result in results.items():
            if query in expected_search and not result['is_search_query']:
                unexpected.append(f" Expected SEARCH but got CHAT: '{query}'")
            elif query in expected_chat and result['is_search_query']:
                unexpected.append(f" Expected CHAT but got SEARCH: '{query}'")

        if unexpected:
            print("  UNEXPECTED RESULTS:")
            for issue in unexpected:
                print(f"   {issue}")
            print()
            print("💡 Consider updating your database patterns to handle these cases.")
        else:
            print(" All test queries behaved as expected!")

        return results

    def get_intent_detection_status(self) -> Dict[str, Any]:
        """
        Get the current status of the intent detection system.
        Useful for debugging and monitoring.
        """
        try:
            if not hasattr(self, 'db_session') or not self.db_session:
                return {
                    'status': 'no_database_session',
                    'database_available': False,
                    'active_intents': 0,
                    'total_patterns': 0,
                    'total_keywords': 0
                }

            from modules.search.models.search_models import SearchIntent, IntentPattern, IntentKeyword

            # Count active intents
            active_intents = self.db_session.query(SearchIntent).filter(
                SearchIntent.is_active == True
            ).count()

            # Count total patterns
            total_patterns = self.db_session.query(IntentPattern).filter(
                IntentPattern.is_active == True
            ).count()

            # Count total keywords
            total_keywords = self.db_session.query(IntentKeyword).filter(
                IntentKeyword.is_active == True
            ).count()

            # Get intent details
            intent_details = []
            intents = self.db_session.query(SearchIntent).filter(
                SearchIntent.is_active == True
            ).all()

            for intent in intents:
                pattern_count = len([p for p in intent.patterns if p.is_active])
                keyword_count = len([k for k in intent.keywords if k.is_active])

                intent_details.append({
                    'name': intent.name,
                    'search_method': intent.search_method,
                    'priority': intent.priority,
                    'patterns': pattern_count,
                    'keywords': keyword_count
                })

            return {
                'status': 'operational',
                'database_available': True,
                'active_intents': active_intents,
                'total_patterns': total_patterns,
                'total_keywords': total_keywords,
                'intent_details': intent_details,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            return {
                'status': 'error',
                'database_available': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    def debug_intent_detection(self, question: str) -> Dict[str, Any]:
        """
        Debug a specific question to see exactly how intent detection works.
        Shows detailed breakdown of pattern matching and scoring.

        Args:
            question: The question to debug

        Returns:
            Detailed breakdown of the intent detection process
        """
        print(f" DEBUGGING INTENT DETECTION FOR: '{question}'")
        print("=" * 60)

        question_lower = question.lower().strip()
        debug_info = {
            'original_question': question,
            'normalized_question': question_lower,
            'database_result': None,
            'fallback_used': False,
            'final_decision': None,
            'intent_breakdown': []
        }

        # Test database patterns
        database_result = self._query_intent_patterns(question_lower)
        debug_info['database_result'] = database_result

        if database_result.get('database_available', True):
            print(f" Database Available: {database_result['database_available']}")
            print(f" Search Query Detected: {database_result['is_search_query']}")

            if database_result['is_search_query']:
                print(f" Intent: {database_result['intent_name']}")
                print(f" Confidence: {database_result['confidence']:.3f}")
                print(f"🔧 Search Method: {database_result['search_method']}")
            else:
                print(f" No intent detected (best confidence: {database_result.get('best_confidence', 0):.3f})")
                print(f"🚪 Threshold: {database_result.get('threshold', 0.6)}")

            debug_info['final_decision'] = database_result['is_search_query']

        else:
            print("  Database unavailable, testing fallback...")
            fallback_result = self._minimal_fallback_detection(question_lower)
            debug_info['fallback_used'] = True
            debug_info['final_decision'] = fallback_result
            print(f" Fallback Result: {fallback_result}")

        print()
        return debug_info

    def execute_unified_search(self, question: str, user_id: str = None, request_id: str = None) -> Dict[str, Any]:
        """
        FIXED: Execute unified search with query tracking.
        This is the main method that should be called.
        """
        logger.info(f" Executing unified search: '{question}'")

        # CHECK 1: Do we have tracking available?
        has_tracking = (
                hasattr(self, 'tracked_search') and
                self.tracked_search is not None and
                hasattr(self.tracked_search, 'execute_unified_search_with_tracking')
        )

        if has_tracking:
            logger.info(" Using TRACKED search")

            # Start a session if we don't have one
            if not hasattr(self.tracked_search, 'current_session_id') or not self.tracked_search.current_session_id:
                if self.query_tracker and user_id:
                    session_id = self.query_tracker.start_search_session(user_id)
                    self.tracked_search.current_session_id = session_id
                    logger.info(f" Started tracking session: {session_id}")

            # Use the tracked search method
            try:
                result = self.tracked_search.execute_unified_search_with_tracking(
                    question=question,
                    user_id=user_id or "anonymous",
                    request_id=request_id
                )

                # Log tracking success
                tracking_info = result.get('tracking_info', {})
                query_id = tracking_info.get('query_id')
                if query_id:
                    logger.info(f" Query tracked with ID: {query_id}")
                    print(f" TRACKING SUCCESS: Query {query_id} tracked!")
                else:
                    logger.warning("  Search executed but query not tracked")

                return result

            except Exception as e:
                logger.error(f" Tracked search failed, falling back: {e}")
                # Fall through to untracked search
        else:
            logger.warning("  Query tracking not available - using untracked search")
            print("  NO TRACKING: Using fallback search")

        # FALLBACK: Use your existing untracked search logic
        return self._execute_untracked_search(question, user_id, request_id)

    # ADD THIS METHOD TO YOUR UnifiedSearchMixin.py FILE
    # Put it right after your execute_unified_search method

    def _execute_untracked_search(self, question: str, user_id: str = None, request_id: str = None) -> Dict[str, Any]:
        """
        MISSING METHOD: Execute untracked search using your existing logic.
        This is the fallback when tracking is not available.
        """
        import time

        search_start = time.time()
        logger.info(f" Executing UNTRACKED unified search for: {question}")

        if not self.unified_search_system:
            return self._search_system_unavailable_response()

        # STEP 1: Detect intent using your existing system
        detected_intent = self._detect_search_intent(question)  # This should exist in your mixin

        if detected_intent:
            intent_name = detected_intent.get('intent', '')
            confidence = detected_intent.get('confidence', 0.0)
            search_method = detected_intent.get('search_method', '')

            logger.info(f" Detected intent: {intent_name} (confidence: {confidence:.2f})")
            logger.info(f" Search method: {search_method}")

            # STEP 2: Route based on search_method from database
            if search_method == 'comprehensive_part_search':
                logger.info(f"🔧 Routing to comprehensive_part_search based on intent: {intent_name}")
                return self._execute_part_search(question, intent_name, detected_intent)

            elif search_method == 'comprehensive_image_search':
                logger.info(f"🖼 Routing to comprehensive_image_search based on intent: {intent_name}")
                return self._execute_image_search(question, intent_name, detected_intent)

            elif search_method == 'comprehensive_position_search':
                logger.info(f"📍 Routing to comprehensive_position_search based on intent: {intent_name}")
                return self._execute_position_search(question, intent_name, detected_intent)

            else:
                logger.warning(f"❓ Unknown search_method: {search_method}, falling back to default")

        # STEP 3: Fallback to existing logic only if no intent detected
        logger.info(f" No specific intent detected, using default search logic")

        try:
            # Use your existing unified search system
            if hasattr(self.unified_search_system, 'execute_nlp_aggregated_search'):
                logger.debug(" Using execute_nlp_aggregated_search")
                search_result = self.unified_search_system.execute_nlp_aggregated_search(question)
            elif hasattr(self.unified_search_system, 'execute_aggregated_search'):
                logger.debug(" Using execute_aggregated_search")
                search_result = self.unified_search_system.execute_aggregated_search(question)
            else:
                logger.error(" No search method available on unified_search_system")
                return self._unified_search_error_response(question, "No search method available")

            # Calculate execution time
            execution_time = int((time.time() - search_start) * 1000)

            # Process and enhance the results
            if search_result and search_result.get('status') == 'success':
                enhanced_result = self._enhance_unified_search_results(search_result, question)
                enhanced_result.update({
                    'tracking_info': {
                        'query_id': None,
                        'session_id': None,
                        'execution_time_ms': execution_time,
                        'search_method': 'untracked_fallback'
                    },
                    'status': 'success'
                })

                logger.info(
                    f" Untracked search completed: {enhanced_result.get('total_results', 0)} results in {execution_time}ms")
                return enhanced_result
            else:
                logger.warning(f" Search returned no results or failed")
                return self._no_unified_results_response(question, search_result)

        except Exception as e:
            execution_time = int((time.time() - search_start) * 1000)
            logger.error(f" Untracked search failed after {execution_time}ms: {e}")
            return self._unified_search_error_response(question, str(e))

    def _execute_part_search(self, question: str, intent_name: str, detected_intent: dict) -> Dict[str, Any]:
        """
        Execute part search using comprehensive_part_search method
        """
        try:
            # Build search parameters based on intent
            search_params = {
                'search_text': question,
                'entity_type': 'part',
                'fields': ['name', 'part_number', 'oem_mfg', 'model', 'notes'],
                'limit': 20,
                'raw_input': question,
                'intent': intent_name,
                'extraction_method': f'intent_based_{intent_name.lower()}'
            }

            # Add intent-specific parameters
            if intent_name == 'FIND_SENSOR':
                search_params['equipment_type'] = 'sensor'
            elif intent_name == 'FIND_MOTOR':
                search_params['equipment_type'] = 'motor'
            elif intent_name == 'FIND_VALVE':
                search_params['equipment_type'] = 'valve'
            # Add more equipment types as needed

            # Extract any regex groups from pattern matching
            extracted_data = detected_intent.get('extracted_data', {})
            if extracted_data:
                search_params.update(extracted_data)

            # Get aggregate search instance
            aggregate_search = None
            if hasattr(self.unified_search_system, '_aggregate_search'):
                aggregate_search = self.unified_search_system._aggregate_search
            elif hasattr(self.unified_search_system, 'comprehensive_part_search'):
                aggregate_search = self.unified_search_system

            if aggregate_search and hasattr(aggregate_search, 'comprehensive_part_search'):
                logger.info(f" Executing comprehensive_part_search for intent: {intent_name}")

                search_result = aggregate_search.comprehensive_part_search(search_params)

                if search_result and search_result.get('status') == 'success':
                    enhanced_result = self._enhance_unified_search_results(search_result, question)
                    enhanced_result['routing_method'] = f'intent_based_{intent_name}'
                    enhanced_result['search_method'] = 'comprehensive_part_search'
                    logger.info(f" Part search successful: {enhanced_result.get('total_results', 0)} results")
                    return enhanced_result
                else:
                    logger.warning(f" comprehensive_part_search returned no results: {search_result}")
                    return self._no_unified_results_response(question, search_result)
            else:
                logger.error(f" comprehensive_part_search method not available")
                return self._unified_search_error_response(question, "Part search method not available")

        except Exception as e:
            logger.error(f" Part search execution failed: {e}", exc_info=True)
            return self._unified_search_error_response(question, f"Part search failed: {str(e)}")

    def _execute_position_search(self, question: str, intent_name: str, detected_intent: dict) -> Dict[str, Any]:
        """
        Execute position search using comprehensive_position_search method
        """
        # Similar structure to _execute_part_search but for positions
        # This is where your current position-based logic should go
        pass

    def _execute_image_search(self, question: str, intent_name: str, detected_intent: dict) -> Dict[str, Any]:
        """
        Execute image search using comprehensive_image_search method
        """
        # Similar structure for image searches
        pass

    def test_your_fixed_method_now(self):
        """
        Add this method to test your comprehensive_part_search directly
        """
        logger.error(f" TESTING YOUR FIXED METHOD DIRECTLY")

        # Test params that match what the logs show
        test_params = {
            'search_text': 'valve bypass 1-1/2" 110-120v',
            'entity_type': 'part',
            'fields': ['name', 'part_number', 'oem_mfg', 'model', 'notes'],
            'limit': 10,
            'raw_input': 'What is the part number for VALVE BYPASS 1-1/2" 110-120V'
        }

        try:
            # Try to find and call your method
            if hasattr(self, 'comprehensive_part_search'):
                logger.error(f" Found comprehensive_part_search on self")
                result = self.comprehensive_part_search(test_params)
            elif hasattr(self, 'unified_search_system') and hasattr(self.unified_search_system, '_aggregate_search'):
                logger.error(f" Found comprehensive_part_search on aggregate_search")
                result = self.unified_search_system._aggregate_search.comprehensive_part_search(test_params)
            else:
                logger.error(f" Cannot find comprehensive_part_search method")
                return {"status": "error", "message": "Method not found"}

            logger.error(f" TEST RESULT: {result.get('status')} with {result.get('count', 0)} results")

            if result.get('results'):
                for i, part in enumerate(result['results'][:3]):
                    pn = part.get('part_number', 'No PN')
                    name = part.get('name', 'No name')
                    logger.error(f"  Part {i + 1}: {pn} - {name}")

            return result

        except Exception as e:
            logger.error(f" TEST FAILED: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _enhance_unified_search_results(self, search_result: Dict[str, Any],
                                        question: str) -> Dict[str, Any]:
        """
        Enhance search results for unified presentation with organized, actionable data.
        """
        enhanced = {
            'search_type': 'unified',
            'query': question,
            'status': 'success',
            'timestamp': datetime.utcnow().isoformat()
        }

        # Extract key information from the search result
        results = search_result.get('results', [])
        count = search_result.get('count', 0)
        entity_type = search_result.get('entity_type', 'mixed')
        method = search_result.get('method', 'unknown')

        # Get NLP analysis if available
        nlp_analysis = search_result.get('nlp_analysis', {})
        detected_intent = nlp_analysis.get('detected_intent', 'UNKNOWN')
        confidence = nlp_analysis.get('overall_confidence', 0)

        enhanced.update({
            'total_results': count,
            'primary_entity_type': entity_type,
            'search_method': method,
            'detected_intent': detected_intent,
            'confidence_score': confidence
        })

        # Organize results by type for better presentation
        organized_results = self._organize_results_by_type(results)
        enhanced['results_by_type'] = organized_results

        # Create summary based on the type of query and results
        enhanced['summary'] = self._create_unified_search_summary(question, organized_results, detected_intent)

        # Add quick actions based on the results
        enhanced['quick_actions'] = self._generate_quick_actions(organized_results, question)

        # Add related searches
        enhanced['related_searches'] = self._generate_related_searches(question, detected_intent, nlp_analysis)

        return enhanced

    def _organize_results_by_type(self, results: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """
        FINAL FIX: Extract actual parts from position objects and organize properly
        """
        organized = {
            'parts': [],
            'images': [],
            'positions': [],
            'other': []
        }

        logger.debug(f" Organizing {len(results)} results...")

        for i, result in enumerate(results):
            logger.debug(
                f"Result {i + 1}: {type(result)} - Keys: {list(result.keys()) if isinstance(result, dict) else 'Not dict'}")

            # Check if this is a POSITION object with embedded parts
            if isinstance(result, dict) and 'parts' in result and 'area' in result:
                logger.debug(f"  🏗  POSITION object detected with {result.get('part_count', 0)} parts")

                # Extract the actual parts from the 'parts' array
                embedded_parts = result.get('parts', [])
                if embedded_parts and isinstance(embedded_parts, list):
                    for j, part in enumerate(embedded_parts):
                        try:
                            if isinstance(part, dict):
                                # This is an actual part object
                                formatted_part = {
                                    'id': part.get('id'),
                                    'part_number': part.get('part_number', f'Unknown-{i + 1}-{j + 1}'),
                                    'name': part.get('name', 'Unknown Part'),
                                    'manufacturer': part.get('oem_mfg', 'Unknown'),
                                    'model': part.get('model', 'Unknown'),
                                    'description': part.get('notes', ''),
                                    'type': 'part',
                                    'location_context': {
                                        'area': result.get('area'),
                                        'equipment_group': result.get('equipment_group'),
                                        'location': result.get('location')
                                    }
                                }
                                organized['parts'].append(formatted_part)
                                logger.debug(f"     Extracted PART: {formatted_part['part_number']}")
                            else:
                                # Handle non-dict part (maybe an object)
                                formatted_part = {
                                    'id': getattr(part, 'id', None),
                                    'part_number': getattr(part, 'part_number', f'Unknown-{i + 1}-{j + 1}'),
                                    'name': getattr(part, 'name', 'Unknown Part'),
                                    'manufacturer': getattr(part, 'oem_mfg', 'Unknown'),
                                    'type': 'part',
                                    'location_context': {
                                        'area': result.get('area'),
                                        'equipment_group': result.get('equipment_group')
                                    }
                                }
                                organized['parts'].append(formatted_part)
                                logger.debug(f"     Extracted PART (object): {formatted_part['part_number']}")
                        except Exception as e:
                            logger.warning(f"      Failed to extract part {j + 1}: {e}")
                            continue

                # Also keep the position info if no parts were extracted
                if not embedded_parts:
                    position_info = {
                        'id': result.get('id'),
                        'area': result.get('area'),
                        'equipment_group': result.get('equipment_group'),
                        'location': result.get('location'),
                        'type': 'position'
                    }
                    organized['positions'].append(position_info)
                    logger.debug(f"  📍 Added POSITION (no parts): {position_info.get('area', 'Unknown')}")

            # Check if this is a direct PART object
            elif isinstance(result, dict) and any(
                    field in result for field in ['part_number', 'oem_mfg']) and 'area' not in result:
                try:
                    formatted_part = {
                        'id': result.get('id'),
                        'part_number': result.get('part_number', f'Direct-{i + 1}'),
                        'name': result.get('name', 'Unknown Part'),
                        'manufacturer': result.get('oem_mfg', 'Unknown'),
                        'model': result.get('model', 'Unknown'),
                        'description': result.get('notes', ''),
                        'type': 'part'
                    }
                    organized['parts'].append(formatted_part)
                    logger.debug(f"   Direct PART: {formatted_part['part_number']}")
                except Exception as e:
                    logger.warning(f"    Failed to format direct part: {e}")

            # Check for images
            elif isinstance(result, dict) and any(field in result for field in ['image_id', 'file_path']) or str(
                    result.get('type', '')).lower() in ['image', 'picture']:
                try:
                    image_info = self._format_image_result(result)
                    organized['images'].append(image_info)
                    logger.debug(f"  🖼  Added IMAGE")
                except Exception as e:
                    logger.warning(f"    Failed to format image: {e}")

            # Everything else goes to other
            else:
                organized['other'].append(result)
                logger.debug(f"  ❓ Added to OTHER")

        # Log final counts
        for category, items in organized.items():
            if items:
                logger.info(f" {category.upper()}: {len(items)} items")
                if category == 'parts' and items:
                    # Log first few part numbers for verification
                    part_numbers = [p.get('part_number', 'Unknown') for p in items[:3]]
                    logger.info(f" Sample part numbers: {part_numbers}")

        # Remove empty categories
        return {k: v for k, v in organized.items() if v}

    def _format_image_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format image result for unified presentation"""
        return {
            'id': result.get('id'),
            'title': result.get('title', 'Untitled Image'),
            'description': result.get('description', ''),
            'thumbnail_url': f"/serve_image/{result.get('id')}/thumbnail" if result.get('id') else None,
            'full_url': f"/serve_image/{result.get('id')}" if result.get('id') else None,
            'metadata': {
                'file_path': result.get('file_path'),
                'created_date': result.get('created_date'),
                'tags': result.get('tags', [])
            },
            'context': result.get('associations', {}),  # Where it's used, related parts, etc.
            'type': 'image'
        }

    def _format_document_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format document result for unified presentation"""
        return {
            'id': result.get('id'),
            'title': result.get('title', 'Untitled Document'),
            'type': 'document',
            'preview': result.get('content', '')[:200] + '...' if result.get('content') else '',
            'url': f"/view_document/{result.get('id')}" if result.get('id') else None,
            'metadata': {
                'revision': result.get('rev'),
                'file_type': result.get('file_type'),
                'size': result.get('file_size'),
                'last_modified': result.get('last_modified')
            },
            'relevance_score': result.get('similarity_score', 0)
        }

    def _format_part_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format part result for unified presentation"""
        return {
            'id': result.get('id'),
            'part_number': result.get('part_number'),
            'name': result.get('name', 'Unknown Part'),
            'type': 'part',
            'manufacturer': result.get('oem_mfg'),
            'model': result.get('model'),
            'description': result.get('notes', ''),
            'usage_locations': result.get('usage_locations', []),
            'related_images': result.get('images', []),
            'equipment_types': result.get('equipment_types', []),
            'availability': result.get('availability', 'unknown')
        }

    def _format_position_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format position result for unified presentation"""
        return {
            'id': result.get('id'),
            'type': 'position',
            'location': {
                'area': result.get('area'),
                'equipment_group': result.get('equipment_group'),
                'model': result.get('model'),
                'location': result.get('location')
            },
            'contents': {
                'parts': result.get('parts', []),
                'images': result.get('images', []),
                'part_count': result.get('part_count', 0),
                'image_count': result.get('image_count', 0)
            }
        }

    def _format_drawing_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format drawing result for unified presentation"""
        return {
            'id': result.get('id'),
            'number': result.get('number'),
            'name': result.get('name', 'Untitled Drawing'),
            'type': 'drawing',
            'revision': result.get('revision'),
            'equipment_name': result.get('equipment_name'),
            'view_url': f"/view_drawing/{result.get('id')}" if result.get('id') else None,
            'download_url': f"/download_drawing/{result.get('id')}" if result.get('id') else None,
            'related_parts': result.get('related_parts', [])
        }

    def _format_equipment_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format equipment result for unified presentation"""
        return {
            'id': result.get('id'),
            'name': result.get('name'),
            'type': 'equipment',
            'equipment_type': result.get('type'),
            'location': result.get('location'),
            'status': result.get('status', 'unknown'),
            'maintenance_info': result.get('maintenance_info', {}),
            'related_documents': result.get('related_documents', []),
            'related_images': result.get('related_images', [])
        }

    def _format_procedure_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format procedure/task result for unified presentation"""
        return {
            'id': result.get('id'),
            'name': result.get('name'),
            'type': 'procedure',
            'description': result.get('description', ''),
            'steps': result.get('steps', []),
            'equipment_required': result.get('equipment_required', []),
            'estimated_time': result.get('estimated_time'),
            'difficulty': result.get('difficulty', 'unknown'),
            'related_documents': result.get('related_documents', [])
        }

    def _create_unified_search_summary(self, question: str, results_by_type: Dict[str, List],
                                       detected_intent: str) -> str:
        """
        ENHANCED: Create better summaries for extracted parts
        """
        total_results = sum(len(results) for results in results_by_type.values())

        if total_results == 0:
            return f"I couldn't find any parts matching '{question}'. The part description might not exist in our database, or it might be catalogued under different terms."

        parts = results_by_type.get('parts', [])

        if len(parts) == 1:
            part = parts[0]
            part_number = part.get('part_number', 'Unknown')
            name = part.get('name', '')
            manufacturer = part.get('manufacturer', '')
            location_context = part.get('location_context', {})

            summary = f"**Found the part:** {part_number}"
            if name and name != 'Unknown' and name != part_number:
                summary += f" - {name}"
            if manufacturer and manufacturer != 'Unknown':
                summary += f" (Manufacturer: {manufacturer})"

            # Add location context if available
            if location_context.get('area') or location_context.get('equipment_group'):
                summary += f"\n**Location:** "
                if location_context.get('area'):
                    summary += f"Area {location_context['area']}"
                if location_context.get('equipment_group'):
                    summary += f", Equipment: {location_context['equipment_group']}"

            return summary

        elif len(parts) > 1:
            summary = f"**Found {len(parts)} matching parts:**\n"
            for i, part in enumerate(parts[:5], 1):
                pn = part.get('part_number', 'Unknown')
                name = part.get('name', '')
                location_context = part.get('location_context', {})

                summary += f"{i}. **{pn}**"
                if name and name != pn and name != 'Unknown':
                    summary += f" - {name}"
                if location_context.get('area'):
                    summary += f" (Area {location_context['area']})"
                summary += "\n"

            if len(parts) > 5:
                summary += f"... and {len(parts) - 5} more parts"

            return summary

        # Fallback
        return f"Found {total_results} results but couldn't extract specific part information."

    def _search_parts_directly(self, search_text: str) -> List[Dict[str, Any]]:
        """
        FALLBACK: Search parts table directly if position search isn't working
        """
        if not hasattr(self, 'db_session') or not self.db_session:
            logger.warning("No database session for direct part search")
            return []

        try:
            # Import Part model
            from modules.emtacdb.emtacdb_fts import Part

            # Search parts directly
            parts = Part.search(
                session=self.db_session,
                search_text=search_text,
                fields=['name', 'part_number', 'oem_mfg', 'model', 'notes'],
                limit=20
            )

            if parts:
                logger.info(f" Direct part search found {len(parts)} parts")
                formatted_parts = []
                for part in parts:
                    formatted_part = {
                        'id': getattr(part, 'id', None),
                        'part_number': getattr(part, 'part_number', 'Unknown'),
                        'name': getattr(part, 'name', 'Unknown Part'),
                        'manufacturer': getattr(part, 'oem_mfg', 'Unknown'),
                        'model': getattr(part, 'model', 'Unknown'),
                        'description': getattr(part, 'notes', ''),
                        'type': 'part',
                        'search_method': 'direct_part_search'
                    }
                    formatted_parts.append(formatted_part)

                return formatted_parts

            return []

        except Exception as e:
            logger.error(f"Direct part search failed: {e}")
            return []

    def _no_unified_results_response(self, question: str, search_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle cases where unified search returns no results."""
        return {
            'search_type': 'unified',
            'query': question,
            'status': 'no_results',
            'total_results': 0,
            'message': f"No results found for: {question}",
            'results_by_type': {},
            'summary': f"I couldn't find any results for '{question}'. This might be because the part number doesn't exist in our database, or it might be spelled differently.",
            'quick_actions': [],
            'related_searches': [
                f"Search for parts containing '{question.split()[-1][:4]}'",
                "Show all available parts",
                "List parts by category",
                "Search equipment by type"
            ],
            'search_suggestions': [
                "Try searching with just the number (e.g., '115982')",
                "Check if the part number format is correct",
                "Search for similar part numbers",
                "Browse parts by equipment type"
            ],
            'timestamp': datetime.utcnow().isoformat(),
            'original_search_result': search_result
        }

    def _unified_search_error_response(self, question: str, error_message: str) -> Dict[str, Any]:
        """Handle unified search errors with helpful information."""
        return {
            'search_type': 'unified',
            'query': question,
            'status': 'error',
            'total_results': 0,
            'message': f"Search error: {error_message}",
            'error_details': {
                'error_message': error_message,
                'query': question,
                'search_system_status': 'error'
            },
            'results_by_type': {},
            'summary': f"I encountered an error while searching for '{question}'. The search system may be temporarily unavailable.",
            'quick_actions': [
                {
                    'action': 'retry_search',
                    'label': 'Try Again',
                    'description': 'Retry the same search'
                },
                {
                    'action': 'browse_parts',
                    'label': 'Browse Parts',
                    'description': 'Browse available parts manually'
                }
            ],
            'related_searches': [
                "Browse all parts",
                "Search by equipment type",
                "View recent searches",
                "Get help with search"
            ],
            'timestamp': datetime.utcnow().isoformat(),
            'technical_details': {
                'error_type': 'search_system_error',
                'component': 'unified_search',
                'user_friendly': True
            }
        }

    def _search_system_unavailable_response(self) -> Dict[str, Any]:
        """Handle cases where the search system is completely unavailable."""
        return {
            'search_type': 'unified',
            'status': 'system_unavailable',
            'total_results': 0,
            'message': "Search system is currently unavailable",
            'results_by_type': {},
            'summary': "The search system is temporarily unavailable. Please try again later or contact support.",
            'quick_actions': [
                {
                    'action': 'manual_browse',
                    'label': 'Browse Manually',
                    'description': 'Browse parts and equipment manually'
                },
                {
                    'action': 'contact_support',
                    'label': 'Contact Support',
                    'description': 'Get help from technical support'
                }
            ],
            'related_searches': [],
            'timestamp': datetime.utcnow().isoformat(),
            'technical_details': {
                'error_type': 'system_unavailable',
                'component': 'search_backend',
                'requires_attention': True
            }
        }

    def _generate_quick_actions(self, results_by_type: Dict[str, List], question: str) -> List[Dict[str, str]]:
        """Generate quick actions based on search results."""
        actions = []

        if results_by_type.get('parts'):
            actions.append({
                'action': 'view_part_details',
                'label': 'View Part Details',
                'description': 'See detailed information about found parts'
            })

        if results_by_type.get('images'):
            actions.append({
                'action': 'view_image_gallery',
                'label': 'View Images',
                'description': 'Browse all related images'
            })

        if results_by_type.get('positions'):
            actions.append({
                'action': 'view_locations',
                'label': 'View Locations',
                'description': 'See where items are located'
            })

        if results_by_type.get('documents'):
            actions.append({
                'action': 'view_documentation',
                'label': 'View Documents',
                'description': 'Access related documentation'
            })

        # Always add a general search action
        actions.append({
            'action': 'refine_search',
            'label': 'Refine Search',
            'description': 'Modify search criteria'
        })

        return actions

    def _generate_related_searches(self, question: str, detected_intent: str, nlp_analysis: Dict) -> List[str]:
        """Generate related search suggestions."""
        related = []

        # Extract entities for related searches
        entities = nlp_analysis.get('extracted_entities', {})

        if detected_intent == 'FIND_PART':
            if entities.get('part_numbers'):
                part = entities['part_numbers'][0]['text']
                related.extend([
                    f"Where is part {part} used?",
                    f"Images of part {part}",
                    f"Parts similar to {part}",
                    f"Documentation for part {part}"
                ])

        elif detected_intent == 'SHOW_IMAGES':
            related.extend([
                "Show maintenance procedures",
                "Find related parts",
                "View equipment documentation",
                "Browse by location"
            ])

        elif detected_intent == 'LOCATION_SEARCH':
            if entities.get('areas'):
                area = entities['areas'][0]['text']
                related.extend([
                    f"Equipment in area {area}",
                    f"Maintenance history for area {area}",
                    f"Parts inventory in area {area}",
                    f"Images from area {area}"
                ])

        # Add general suggestions if none found
        if not related:
            related = [
                "Browse all parts",
                "Search by equipment type",
                "View maintenance procedures",
                "Find documentation"
            ]

        return related[:5]  # Limit to 5 suggestions

    def _record_unified_search_analytics(self, question, result, user_id, request_id):
        """Record analytics for unified search."""
        try:
            logging.info(f" Unified search: '{question}' -> {result.get('status')} ({result.get('count', 0)} results)")
        except Exception as e:
            logging.warning(f"Analytics recording failed: {e}")

    class SynonymResolver:
        """Resolves synonyms using the database with proper error handling"""

        def __init__(self, session):
            self.session = session
            self._synonym_cache = {}

        def expand_query_with_synonyms(self, query_text: str) -> Dict[str, List[str]]:
            """
            Expand query with synonyms from database

            Args:
                query_text: Original search query like "search roller bearing assembly"

            Returns:
                Dict with original terms and their synonyms
            """
            import re

            # Remove common search words
            stop_words = {'search', 'find', 'show', 'get', 'locate', 'display', 'for', 'the', 'a', 'an', 'part',
                          'number'}
            words = re.findall(r'\b\w+\b', query_text.lower())
            key_terms = [word for word in words if word not in stop_words and len(word) > 2]

            expanded_terms = {}

            for term in key_terms:
                # Check cache first
                cache_key = term.lower()
                if cache_key in self._synonym_cache:
                    expanded_terms[term] = self._synonym_cache[cache_key]
                    continue

                # Get synonyms with fallback strategy
                synonyms = self._get_synonyms_with_fallback(term)
                self._synonym_cache[cache_key] = synonyms
                expanded_terms[term] = synonyms

            return expanded_terms

        def _get_synonyms_with_fallback(self, term: str) -> List[str]:
            """Get synonyms using database with fallback to hardcoded synonyms."""

            # Try database lookup first
            try:
                return self._get_synonyms_from_database(term)
            except Exception as e:
                logger.debug(f"Database synonym lookup failed for '{term}': {e}")
                # Fallback to hardcoded synonyms
                return self._get_synonyms_fallback(term)

        def _get_synonyms_from_database(self, term: str) -> List[str]:
            """Get synonyms from database with proper transaction handling."""
            try:
                # Ensure fresh transaction
                if self.session.in_transaction():
                    self.session.rollback()

                # Fixed SQL query - includes confidence_score in SELECT for ORDER BY
                sql_query = text("""
                    SELECT es.canonical_value, es.synonym_value, es.confidence_score
                    FROM entity_synonym es
                    JOIN entity_type et ON es.entity_type_id = et.id
                    WHERE (
                        LOWER(es.canonical_value) LIKE :term OR 
                        LOWER(es.synonym_value) LIKE :term
                    )
                    AND et.name = 'EQUIPMENT_TYPE'
                    AND es.confidence_score > 0.5
                    ORDER BY es.confidence_score DESC
                    LIMIT 10
                """)

                result = self.session.execute(sql_query, {'term': f'%{term.lower()}%'})

                # Collect unique related terms
                related_terms = set()
                for row in result:
                    canonical = row[0].lower() if row[0] else ''
                    synonym = row[1].lower() if row[1] else ''

                    if canonical and canonical != term.lower():
                        related_terms.add(canonical)
                    if synonym and synonym != term.lower():
                        related_terms.add(synonym)

                synonyms_list = list(related_terms)[:5]

                if synonyms_list:
                    logger.debug(f"Database synonyms for '{term}': {synonyms_list}")
                else:
                    logger.debug(f"No database synonyms found for '{term}'")

                return synonyms_list

            except Exception as e:
                logger.warning(f"Database synonym lookup failed for {term}: {e}")
                # Clean rollback
                try:
                    self.session.rollback()
                except:
                    pass
                raise

        def _get_synonyms_fallback(self, term: str) -> List[str]:
            """Fallback method with hardcoded synonyms."""

            logger.debug(f"Using fallback synonyms for '{term}'")

            # Comprehensive fallback synonym dictionary
            fallback_synonyms = {
                # Core component types
                'valve': ['valves', 'control valve', 'ball valve', 'gate valve', 'check valve'],
                'bearing': ['bearings', 'ball bearing', 'roller bearing', 'thrust bearing'],
                'switch': ['switches', 'limit switch', 'pressure switch', 'temperature switch'],
                'motor': ['motors', 'electric motor', 'ac motor', 'servo motor'],
                'belt': ['belts', 'drive belt', 'v-belt', 'timing belt'],
                'cable': ['cables', 'power cable', 'control cable', 'data cable'],
                'sensor': ['sensors', 'temperature sensor', 'pressure sensor', 'level sensor'],
                'seal': ['seals', 'oil seal', 'shaft seal', 'hydraulic seal'],
                'relay': ['relays', 'control relay', 'time relay', 'power relay'],
                'pump': ['pumps', 'centrifugal pump', 'hydraulic pump', 'water pump'],
                'spring': ['springs', 'compression spring', 'extension spring'],
                'filter': ['filters', 'air filter', 'oil filter', 'hydraulic filter'],
                'gear': ['gears', 'spur gear', 'bevel gear', 'worm gear'],
                'tube': ['tubes', 'hydraulic tube', 'pneumatic tube'],
                'hose': ['hoses', 'hydraulic hose', 'air hose', 'vacuum hose'],
                'wire': ['wires', 'electrical wire', 'control wire'],
                'fan': ['fans', 'cooling fan', 'exhaust fan', 'ventilation fan'],

                # Assembly terms
                'assembly': ['assemblies', 'unit', 'component', 'module'],
                'component': ['components', 'part', 'piece', 'element'],
                'unit': ['units', 'assembly', 'module', 'system'],

                # Banner-specific (from the log context)
                'banner': ['banner engineering', 'banner corp', 'banner sensors'],
            }

            term_lower = term.lower()

            # Exact match
            if term_lower in fallback_synonyms:
                synonyms = fallback_synonyms[term_lower][:4]
                logger.debug(f"Fallback exact match for '{term}': {synonyms}")
                return synonyms

            # Partial match for compound terms
            for key, values in fallback_synonyms.items():
                if key in term_lower or term_lower in key:
                    synonyms = values[:3]
                    logger.debug(f"Fallback partial match for '{term}' (matched '{key}'): {synonyms}")
                    return synonyms

            # No synonyms found
            logger.debug(f"No fallback synonyms found for '{term}'")
            return []

        def clear_cache(self):
            """Clear the synonym cache."""
            cache_size = len(self._synonym_cache)
            self._synonym_cache.clear()
            logger.debug(f"Cleared synonym cache ({cache_size} entries)")
            return cache_size

        def get_cache_stats(self):
            """Get cache statistics."""
            return {
                'size': len(self._synonym_cache),
                'entries': list(self._synonym_cache.keys())
            }

    class PartNumberExtractor:
        """Extract and normalize part numbers from search queries"""

        @staticmethod
        def extract_part_number_from_query(query: str) -> str:
            """Extract the actual part number from a search query"""
            import re

            # Pattern for "find part number A115957" - extract A115957
            patterns = [
                r'part\s+number\s+([A-Za-z0-9\-\.]+)',  # "part number A115957"
                r'find\s+part\s+([A-Za-z0-9\-\.]+)',  # "find part A115957"
                r'search\s+for\s+part\s+([A-Za-z0-9\-\.]+)',  # "search for part A115957"
                r'part\s+([A-Za-z0-9\-\.]+)',  # "part A115957"
                r'([A-Za-z]\d{5,})',  # "A115957" pattern directly
                r'(\d{5,})',  # "115957" (5+ digits)
            ]

            for pattern in patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    part_number = match.group(1).upper().strip()
                    logger.debug(
                        f"Extracted part number '{part_number}' from query '{query}' using pattern '{pattern}'")
                    return part_number

            logger.debug(f"No part number extracted from query: '{query}'")
            return ""

        @staticmethod
        def normalize_part_number(part_number: str) -> str:
            """Normalize part number format"""
            if not part_number:
                return ""

            # Remove extra whitespace and convert to uppercase
            normalized = part_number.strip().upper()

            # Common normalizations
            normalizations = [
                (r'\s+', ''),  # Remove all whitespace
                (r'[_]+', '-'),  # Convert underscores to dashes
                (r'[-]{2,}', '-'),  # Multiple dashes to single dash
            ]

            for pattern, replacement in normalizations.items():
                normalized = re.sub(pattern, replacement, normalized)

            logger.debug(f"Normalized '{part_number}' to '{normalized}'")
            return normalized

    # Integration function to fix the existing search functionality
    def fix_part_search_extraction(query: str) -> Dict[str, Any]:
        """
        Fix the part number extraction that's failing in your current system

        This function properly extracts "A115957" from "find part number A115957"
        """
        extractor = PartNumberExtractor()

        # Extract the actual part number
        part_number = extractor.extract_part_number_from_query(query)

        if part_number:
            # Normalize it
            normalized_part = extractor.normalize_part_number(part_number)

            return {
                'part_number': normalized_part,
                'raw_input': query,
                'entity_type': 'part',
                'search_method': 'part_number_extraction'
            }
        else:
            # Fallback to original behavior
            return {
                'raw_input': query,
                'entity_type': 'unknown'
            }

    def execute_unified_search_with_direct_part_search(self, question: str, user_id: str = None,
                                                       request_id: str = None):
        """
        ALTERNATIVE: Execute search with direct part table fallback
        """
        # Try normal search first
        result = self.execute_unified_search(question, user_id, request_id)

        # If we got position results but no actual part numbers, try direct search
        parts = result.get('results_by_type', {}).get('parts', [])
        if parts and all(p.get('part_number', '').startswith(('Part-', 'Unknown-', 'Direct-')) for p in parts):
            logger.info(" Position search didn't return actual part numbers, trying direct part search...")

            # Extract search text from the original analysis
            nlp_analysis = result.get('nlp_analysis', {})
            search_params = nlp_analysis.get('search_parameters', {}) if 'nlp_analysis' in result else {}
            search_text = search_params.get('search_text', '')

            if search_text:
                direct_parts = self._search_parts_directly(search_text)
                if direct_parts:
                    # Replace the parts in the result
                    result['results_by_type']['parts'] = direct_parts
                    result['total_results'] = len(direct_parts)

                    # Regenerate summary with actual parts
                    result['summary'] = self._create_unified_search_summary(
                        question,
                        result['results_by_type'],
                        result.get('detected_intent', 'FIND_PART')
                    )

                    result['search_method'] = 'direct_part_search_fallback'
                    logger.info(f" Direct search found {len(direct_parts)} actual parts")

        return result

    # 2. Enhanced search method for your UnifiedSearchMixin

    def execute_unified_search_with_synonyms(self, question: str, user_id: str = None, request_id: str = None) -> Dict[
        str, Any]:
        """
        Enhanced unified search that uses your synonym database

        Add this method to your UnifiedSearchMixin class
        """
        import time
        from modules.search.models.search_models import EntitySynonym

        search_start = time.time()
        logger.info(f"Executing synonym-enhanced unified search for: {question}")

        if not self.unified_search_system:
            return self._search_system_unavailable_response()

        try:
            # 1. Initialize synonym resolver
            session = getattr(self, 'db_session', None)
            if not session:
                logger.warning("No database session available for synonym resolution")
                synonym_resolver = None
            else:
                synonym_resolver = SynonymResolver(session)

            # 2. Expand query with synonyms
            original_query = question
            expanded_terms = {}

            if synonym_resolver:
                try:
                    expanded_terms = synonym_resolver.expand_query_with_synonyms(question)
                    logger.debug(f"Expanded terms: {expanded_terms}")

                    # Create enhanced query variations
                    enhanced_queries = self._create_enhanced_queries(question, expanded_terms)
                    logger.debug(f"Enhanced queries: {enhanced_queries}")

                except Exception as e:
                    logger.warning(f"Synonym expansion failed: {e}")
                    enhanced_queries = [question]
            else:
                enhanced_queries = [question]

            # 3. Try search with each enhanced query
            best_result = None
            best_score = 0

            for query_variant in enhanced_queries:
                try:
                    # Use your existing NLP search system
                    if hasattr(self.unified_search_system, 'execute_nlp_aggregated_search'):
                        result = self.unified_search_system.execute_nlp_aggregated_search(query_variant)
                    else:
                        result = self.unified_search_system.execute_aggregated_search(query_variant)

                    # Score the result
                    if result and result.get('status') == 'success':
                        result_score = result.get('count', 0)
                        if result_score > best_score:
                            best_score = result_score
                            best_result = result
                            best_result['matched_query'] = query_variant
                            best_result['synonym_enhanced'] = query_variant != original_query

                except Exception as e:
                    logger.warning(f"Search failed for query variant '{query_variant}': {e}")
                    continue

            # 4. Process the best result
            if best_result and best_score > 0:
                enhanced_result = self._enhance_unified_search_results(best_result, original_query)

                # Add synonym information
                enhanced_result.update({
                    'synonym_expansion': {
                        'original_query': original_query,
                        'matched_query': best_result.get('matched_query'),
                        'expanded_terms': expanded_terms,
                        'synonym_enhanced': best_result.get('synonym_enhanced', False)
                    }
                })

                # Record analytics
                self._record_unified_search_analytics(original_query, enhanced_result, user_id, request_id)

                search_time = time.time() - search_start
                enhanced_result['search_time_ms'] = int(search_time * 1000)

                logger.info(
                    f"Synonym-enhanced search completed in {search_time:.3f}s with {enhanced_result.get('total_results', 0)} results")
                return enhanced_result

            else:
                # No results found even with synonyms
                return self._no_unified_results_response_with_synonyms(original_query, expanded_terms)

        except Exception as e:
            search_time = time.time() - search_start
            logger.error(f"Synonym-enhanced unified search failed after {search_time:.3f}s: {e}", exc_info=True)
            return self._unified_search_error_response(original_query, str(e))

    def _create_enhanced_queries(self, original_query: str, expanded_terms: Dict[str, List[str]]) -> List[str]:
        """Create query variations using synonyms"""
        queries = [original_query]

        # For each term with synonyms, create variations
        for original_term, synonyms in expanded_terms.items():
            if not synonyms:
                continue

            # Try replacing with the most common synonyms
            for synonym in synonyms[:3]:  # Limit to top 3 synonyms
                # Simple replacement (you could make this more sophisticated)
                enhanced_query = original_query.replace(original_term, synonym)
                if enhanced_query != original_query and enhanced_query not in queries:
                    queries.append(enhanced_query)

        return queries

    def _no_unified_results_response_with_synonyms(self, question: str, expanded_terms: Dict[str, List[str]]) -> Dict[
        str, Any]:
        """Enhanced no results response with synonym suggestions"""

        # Create search suggestions based on synonyms
        suggestions = []
        for term, synonyms in expanded_terms.items():
            for synonym in synonyms[:2]:  # Top 2 synonyms per term
                suggestion = question.replace(term, synonym)
                if suggestion != question:
                    suggestions.append(f"Try: '{suggestion}'")

        if not suggestions:
            suggestions = [
                "Try using different terms (e.g., 'bearing' instead of 'bearings')",
                "Search for specific part numbers",
                "Use broader categories (e.g., 'motor parts')"
            ]

        return {
            'search_type': 'unified',
            'query': question,
            'status': 'no_results',
            'total_results': 0,
            'message': f"No results found for: {question}",
            'results_by_type': {},
            'summary': f"I couldn't find any results for '{question}' even with synonym expansion.",
            'quick_actions': [],
            'related_searches': suggestions[:5],
            'synonym_info': {
                'expanded_terms': expanded_terms,
                'suggestions_based_on_synonyms': True
            },
            'timestamp': datetime.utcnow().isoformat()
        }

    def check_tracking_status(self) -> Dict[str, Any]:
        """Check if query tracking is working."""
        status = {
            'has_db_session': hasattr(self, 'db_session') and self.db_session is not None,
            'has_query_tracker': hasattr(self, 'query_tracker') and self.query_tracker is not None,
            'has_tracked_search': hasattr(self, 'tracked_search') and self.tracked_search is not None,
            'has_tracking_method': False,
            'current_session_id': None,
            'tracking_enabled': False
        }

        # Check if tracking method exists
        if status['has_tracked_search']:
            status['has_tracking_method'] = hasattr(self.tracked_search, 'execute_unified_search_with_tracking')
            status['current_session_id'] = getattr(self.tracked_search, 'current_session_id', None)

        # Overall tracking status
        status['tracking_enabled'] = all([
            status['has_db_session'],
            status['has_query_tracker'],
            status['has_tracked_search'],
            status['has_tracking_method']
        ])

        print("=== QUERY TRACKING STATUS ===")
        for key, value in status.items():
            icon = "" if value else ""
            print(f"{icon} {key}: {value}")
        print("=============================")

        return status

    def test_tracking(self, test_query: str = "test tracking query") -> Dict[str, Any]:
        """Test that query tracking is working end-to-end."""
        print(f" Testing query tracking with: '{test_query}'")

        # First check status
        status = self.check_tracking_status()
        if not status['tracking_enabled']:
            return {
                'success': False,
                'error': 'Tracking not enabled',
                'status': status
            }

        # Try to execute a tracked search
        try:
            result = self.execute_unified_search(test_query, user_id="test_user")

            tracking_info = result.get('tracking_info', {})
            query_id = tracking_info.get('query_id')

            if query_id:
                print(f" SUCCESS: Test query tracked with ID {query_id}")

                # Try to record satisfaction
                if self.query_tracker:
                    satisfaction_recorded = self.query_tracker.record_user_satisfaction(query_id, 5)
                    print(f" Satisfaction recording: {'' if satisfaction_recorded else ''}")

                return {
                    'success': True,
                    'query_id': query_id,
                    'tracking_info': tracking_info,
                    'satisfaction_recorded': satisfaction_recorded
                }
            else:
                print(" FAILED: No query_id returned")
                return {
                    'success': False,
                    'error': 'No query_id in result',
                    'result': result
                }

        except Exception as e:
            print(f" FAILED: Exception during test: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """Get search performance report."""
        if hasattr(self, 'query_tracker') and self.query_tracker:
            return self.query_tracker.get_search_performance_report(days)
        return {"error": "Query tracker not available"}

    def _organize_unified_results(self, search_result, question):
        """Organize search results into a unified format"""
        try:
            # Extract results from the search_result
            results = search_result.get('results', [])

            # Basic organization
            organized = {
                'search_type': 'unified',
                'query': question,
                'status': 'success',
                'total_results': len(results),
                'results_by_type': {
                    'parts': [],
                    'images': [],
                    'documents': [],
                    'positions': []
                },
                'summary': f"Found {len(results)} results for: {question}",
                'search_method': search_result.get('search_method', 'unified_search'),
                'timestamp': datetime.utcnow().isoformat()
            }

            # If we have results, organize them
            if results:
                for result in results:
                    if isinstance(result, dict):
                        # Determine result type and add to appropriate category
                        if 'part_number' in result:
                            organized['results_by_type']['parts'].append(result)
                        elif 'image' in str(result).lower():
                            organized['results_by_type']['images'].append(result)
                        elif 'document' in str(result).lower():
                            organized['results_by_type']['documents'].append(result)
                        else:
                            organized['results_by_type']['positions'].append(result)

            return organized

        except Exception as e:
            logger.error(f"Error organizing search results: {e}")
            return {
                'search_type': 'unified',
                'query': question,
                'status': 'success',
                'total_results': 0,
                'results_by_type': {},
                'summary': f"Search completed but results could not be organized properly.",
                'search_method': 'organization_error',
                'timestamp': datetime.utcnow().isoformat()
            }

    def _no_unified_results_response(self, question: str, search_result: Dict = None) -> Dict[str, Any]:
        """Handle cases where no results are found"""
        return {
            'search_type': 'unified',
            'query': question,
            'status': 'success',
            'total_results': 0,
            'message': f"No results found for: {question}",
            'results_by_type': {},
            'summary': f"I couldn't find any results for '{question}'. This might be because the part number doesn't exist in our database, or it might be spelled differently.",
            'quick_actions': [],
            'related_searches': [
                f"Search for parts containing '{question.split()[-1][:4]}'",
                "Show all available parts",
                "List parts by category",
                "Search equipment by type"
            ],
            'search_suggestions': [
                "Try searching with just the number (e.g., '115982')",
                "Check if the part number format is correct",
                "Search for similar part numbers",
                "Browse parts by equipment type"
            ],
            'timestamp': datetime.utcnow().isoformat(),
            'original_search_result': search_result
        }

