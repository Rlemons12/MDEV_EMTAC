from sqlalchemy import Column, Integer, String, Text, Boolean, Float, DateTime, ForeignKey, UniqueConstraint, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from modules.configuration.base import Base  # Adjust path as needed
from modules.configuration.log_config import logger
from typing import Dict, Any, Optional

class SearchIntent(Base):
    """
    Search intents (FIND_PART, SHOW_IMAGES, etc.)
    """
    __tablename__ = 'search_intent'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)  # e.g., "FIND_PART"
    display_name = Column(String(255))  # e.g., "Find Parts"
    description = Column(Text)
    search_method = Column(String(100))  # e.g., "comprehensive_part_search"
    priority = Column(Float, default=1.0)  # FIXED: Changed from Integer to Float
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    patterns = relationship("IntentPattern", back_populates="intent",
                            cascade="all, delete-orphan", passive_deletes=True)
    keywords = relationship("IntentKeyword", back_populates="intent",
                            cascade="all, delete-orphan", passive_deletes=True)
    entity_rules = relationship("EntityExtractionRule", back_populates="intent",
                                cascade="all, delete-orphan", passive_deletes=True)


class IntentPattern(Base):
    """
    Regex patterns for intent detection - FIXED to match actual database schema
    """
    __tablename__ = 'intent_pattern'

    id = Column(Integer, primary_key=True)
    intent_id = Column(Integer, ForeignKey('search_intent.id'), nullable=False)
    pattern_text = Column(Text, nullable=False)  # FIXED: was 'spacy_pattern'
    pattern_type = Column(String(50), default='regex')  # FIXED: added this column
    success_rate = Column(Float, default=0.0)  # FIXED: added this column
    usage_count = Column(Integer, default=0)  # FIXED: added this column
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    intent = relationship("SearchIntent", back_populates="patterns")


class IntentKeyword(Base):
    """
    Keywords and synonyms for intent classification - FIXED to match actual database schema
    """
    __tablename__ = 'intent_keyword'

    id = Column(Integer, primary_key=True)
    intent_id = Column(Integer, ForeignKey('search_intent.id'), nullable=False)
    keyword_text = Column(String(200), nullable=False)  # FIXED: was 'keyword'
    weight = Column(Float, default=1.0)
    is_exact_match = Column(Boolean, default=False)  # FIXED: added this column
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    intent = relationship("SearchIntent", back_populates="keywords")


class EntityExtractionRule(Base):
    """
    Rules for extracting entities from user input - FIXED to match actual database schema
    """
    __tablename__ = 'entity_extraction_rule'

    id = Column(Integer, primary_key=True)
    intent_id = Column(Integer, ForeignKey('search_intent.id'))  # FIXED: nullable=True
    entity_type = Column(String(100), nullable=False)
    rule_text = Column(Text, nullable=False)  # FIXED: was 'pattern'
    rule_type = Column(String(50), default='regex')  # FIXED: added this column
    extraction_pattern = Column(Text)  # FIXED: added this column
    validation_pattern = Column(Text)  # FIXED: added this column
    confidence_threshold = Column(Float, default=0.7)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    intent = relationship("SearchIntent", back_populates="entity_rules")


class SearchAnalytics(Base):
    """
    Analytics and performance tracking for search operations - FIXED to match actual database schema
    """
    __tablename__ = 'search_analytics'

    id = Column(Integer, primary_key=True)
    user_id = Column(String(100))  # FIXED: added this column
    session_id = Column(String(100))
    query_text = Column(Text)  # FIXED: was 'user_input'
    detected_intent = Column(String(100))  # FIXED: was ForeignKey
    intent_confidence = Column(Float)  # FIXED: was 'confidence_score'
    search_method = Column(String(100))
    execution_time_ms = Column(Integer)
    result_count = Column(Integer)
    success = Column(Boolean)  # FIXED: added this column
    error_message = Column(Text)  # FIXED: added this column
    user_agent = Column(Text)  # FIXED: added this column
    ip_address = Column(String(45))  # FIXED: added this column (inet type maps to string)
    created_at = Column(DateTime, default=datetime.utcnow)


class UnifiedSearchWithTracking:
    """
    Enhanced version of UnifiedSearchMixin that tracks all searches using SearchQuery.
    Drop-in replacement that adds comprehensive search tracking.
    """

    def __init__(self, unified_search_mixin):
        """Initialize with reference to your existing UnifiedSearchMixin instance."""
        self.unified_search = unified_search_mixin
        self.query_tracker = None
        self.current_session_id = None

        # Initialize tracker if database session is available
        if hasattr(unified_search_mixin, 'db_session') and unified_search_mixin.db_session:
            from modules.search.nlp_search import SearchQueryTracker
            self.query_tracker = SearchQueryTracker(unified_search_mixin.db_session)
            logger.info("✅ Search tracking initialized")
        else:
            logger.warning("⚠️ No database session available - tracking disabled")

    def start_user_session(self, user_id: str, context_data: Dict = None) -> Optional[int]:
        """Start a search session for a user."""
        if self.query_tracker:
            self.current_session_id = self.query_tracker.start_search_session(user_id, context_data)
            return self.current_session_id
        return None

    def execute_unified_search_with_tracking(self, question: str, user_id: str = None,
                                             request_id: str = None) -> Dict[str, Any]:
        """
        Enhanced search with comprehensive SearchQuery tracking.
        This wraps your existing execute_unified_search method.
        """
        import time

        search_start = time.time()
        user_id = user_id or "anonymous"

        # Start session if we don't have one
        if not self.current_session_id and self.query_tracker:
            self.current_session_id = self.query_tracker.start_search_session(user_id)

        # Initialize tracking variables
        detected_intent_id = None
        intent_confidence = 0.0
        search_method = "unknown"
        extracted_entities = {}
        normalized_query = question.lower().strip()

        logger.info(f"🔍 Executing tracked unified search for: {question}")

        try:
            # STEP 1: Intent Detection with Tracking
            if (hasattr(self.unified_search, 'unified_search_system') and
                    self.unified_search.unified_search_system and
                    hasattr(self.unified_search.unified_search_system, 'analyze_user_input')):

                # Get NLP analysis with intent and keyword detection
                analysis = self.unified_search.unified_search_system.analyze_user_input(question)

                detected_intent = analysis.get('intent', {}).get('intent', 'UNKNOWN')
                intent_confidence = analysis.get('intent', {}).get('confidence', 0.0)
                extracted_entities = {
                    'search_parameters': analysis.get('search_parameters', {}),
                    'entities': analysis.get('entities', {}),
                    'semantic_info': analysis.get('semantic_info', {})
                }
                search_method = analysis.get('processing_method', 'nlp_analysis')

                # Get intent ID
                if self.query_tracker:
                    detected_intent_id = self.query_tracker.get_intent_id(detected_intent)

                logger.debug(
                    f"Intent detected: {detected_intent} (ID: {detected_intent_id}, confidence: {intent_confidence:.2f})")

            # STEP 2: Execute Search (using your existing logic)
            if self._looks_like_part_query(question):
                result = self._execute_part_search_with_tracking(question,
                                                                 extracted_entities.get('search_parameters', {}))
                search_method = result.get('search_method', 'part_search_bypass')
            else:
                # Use your existing unified search
                result = self.unified_search.execute_unified_search(question, user_id, request_id)
                search_method = result.get('search_method', 'unified_search')

            # STEP 3: Calculate metrics
            execution_time = int((time.time() - search_start) * 1000)
            result_count = result.get('total_results', 0)

            # STEP 4: Track in SearchQuery
            query_id = None
            if self.query_tracker:
                query_id = self.query_tracker.track_search_query(
                    session_id=self.current_session_id,
                    query_text=question,
                    detected_intent_id=detected_intent_id,
                    intent_confidence=intent_confidence,
                    search_method=search_method,
                    result_count=result_count,
                    execution_time_ms=execution_time,
                    extracted_entities=extracted_entities,
                    normalized_query=normalized_query
                )

            # STEP 5: Enhance result with tracking info
            result.update({
                'tracking_info': {
                    'query_id': query_id,
                    'session_id': self.current_session_id,
                    'detected_intent_id': detected_intent_id,
                    'intent_confidence': intent_confidence,
                    'execution_time_ms': execution_time,
                    'search_method': search_method
                },
                'satisfaction_request': {
                    'query_id': query_id,
                    'message': 'How satisfied are you with these results?',
                    'scale': '1 (Very Poor) to 5 (Excellent)'
                } if query_id else None
            })

            logger.info(f"✅ Search tracked: Query {query_id}, {result_count} results, {execution_time}ms")
            return result

        except Exception as e:
            search_time = time.time() - search_start
            execution_time = int(search_time * 1000)

            # Track failed searches too
            if self.query_tracker:
                query_id = self.query_tracker.track_search_query(
                    session_id=self.current_session_id,
                    query_text=question,
                    detected_intent_id=detected_intent_id,
                    intent_confidence=intent_confidence,
                    search_method=f"{search_method}_error",
                    result_count=0,
                    execution_time_ms=execution_time,
                    extracted_entities={'error': str(e)}
                )

            logger.error(f"❌ Search failed and tracked: Query {query_id}, {execution_time}ms: {e}")
            return self.unified_search._unified_search_error_response(question, str(e))

    def record_satisfaction(self, query_id: int, satisfaction_score: int) -> bool:
        """Record user satisfaction for a query."""
        if self.query_tracker:
            return self.query_tracker.record_user_satisfaction(query_id, satisfaction_score)
        return False

    def track_result_click(self, query_id: int, result_type: str, result_id: int,
                           click_position: int, action_taken: str = "view") -> bool:
        """Track when user clicks on a result."""
        if self.query_tracker:
            return self.query_tracker.track_result_click(
                query_id, result_type, result_id, click_position, action_taken
            )
        return False

    def get_performance_report(self, days: int = 7) -> Dict[str, Any]:
        """Get search performance report."""
        if self.query_tracker:
            return self.query_tracker.get_search_performance_report(days)
        return {"error": "Query tracker not available"}

    def end_session(self) -> bool:
        """End the current search session."""
        if self.current_session_id and self.query_tracker:
            success = self.query_tracker.end_search_session(self.current_session_id)
            if success:
                self.current_session_id = None
            return success
        return False

    def _looks_like_part_query(self, question: str) -> bool:
        """Quick check if query looks like a part search."""
        question_lower = question.lower()
        part_indicators = [
            'part number for', 'find part', 'looking for', 'what is the part number',
            'gear', 'sensor', 'motor', 'pump', 'valve', 'bearing'
        ]
        return any(indicator in question_lower for indicator in part_indicators)

    def _execute_part_search_with_tracking(self, question: str, search_params: Dict) -> Dict[str, Any]:
        """Execute part search with enhanced tracking."""
        try:
            # Use your existing bypass logic
            result = self.unified_search.execute_unified_search(question)

            # Add tracking-specific metadata
            result.update({
                'search_method': 'enhanced_part_search_bypass',
                'parameters_used': search_params,
                'bypass_method': 'direct_comprehensive_part_search'
            })

            return result

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'search_method': 'enhanced_part_search_bypass_error'
            }


class SearchResultClick(Base):
    """Track which results users actually click on."""
    __tablename__ = 'search_result_click'

    id = Column(Integer, primary_key=True)
    query_id = Column(Integer, ForeignKey('search_query.id'))
    result_type = Column(String(50))  # 'part', 'image', 'document'
    result_id = Column(Integer)
    click_position = Column(Integer)  # Position in result list
    dwell_time_seconds = Column(Integer)
    action_taken = Column(String(100))  # 'view', 'download', 'share'
    created_at = Column(DateTime, default=datetime.utcnow)