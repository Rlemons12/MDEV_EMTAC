from sqlalchemy import Column, Integer, String, Text, Boolean, Float, DateTime, ForeignKey, UniqueConstraint, JSON, text
from sqlalchemy.orm import relationship
from datetime import datetime
from modules.configuration.base import Base  # Adjust path as needed
from modules.configuration.log_config import logger, with_request_id
from typing import Dict, Any, Optional

class SearchIntent(Base):
    """
    Search intents (FIND_PART, SHOW_IMAGES, etc.)j
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

class PatternInterpreter:
    """
    Dynamic pattern interpreter that analyzes regex patterns and automatically
    determines how to extract data and build search parameters.
    """

    def __init__(self):
        # Define pattern analysis rules
        self.pattern_analyzers = [
            self._analyze_part_number_pattern,
            self._analyze_manufacturer_equipment_pattern,
            self._analyze_safety_procedure_pattern,
            self._analyze_availability_query_pattern,
            self._analyze_general_search_pattern
        ]

    def interpret_pattern_and_extract(self, pattern_text: str, user_query: str, extracted_groups: list) -> Dict[
        str, Any]:
        """
        Analyze a regex pattern and determine what it extracts and how to use it.

        Args:
            pattern_text: The regex pattern from the database
            user_query: The actual user query that matched
            extracted_groups: The captured groups from the regex match

        Returns:
            Dict with interpretation and search parameters
        """
        # Analyze the pattern to understand its purpose
        pattern_analysis = self._analyze_pattern_structure(pattern_text)

        # Extract semantic meaning from the groups
        extracted_entities = self._extract_entities_from_groups(
            pattern_analysis, extracted_groups, user_query
        )

        # Build search parameters based on what we found
        search_params = self._build_search_params_from_entities(extracted_entities)

        return {
            'pattern_analysis': pattern_analysis,
            'extracted_entities': extracted_entities,
            'search_params': search_params,
            'interpretation': self._generate_human_readable_interpretation(pattern_analysis, extracted_entities)
        }

    def _analyze_pattern_structure(self, pattern_text: str) -> Dict[str, Any]:
        """Analyze the regex pattern to understand what it's looking for."""

        analysis = {
            'pattern_type': 'unknown',
            'capture_groups': [],
            'keywords': [],
            'context_clues': []
        }

        # Run all pattern analyzers
        for analyzer in self.pattern_analyzers:
            result = analyzer(pattern_text)
            if result['confidence'] > 0:
                analysis.update(result)
                break

        return analysis

    def _analyze_part_number_pattern(self, pattern: str) -> Dict[str, Any]:
        """Analyze patterns that look for part numbers."""
        import re

        # Look for part number indicators
        part_indicators = [
            r'search.*part',
            r'find.*part',
            r'part.*number',
            r'\[A-Za-z0-9\\\-\\\.\]',  # Character classes for part numbers
            r'\{3,\}',  # Length requirements typical of part numbers
        ]

        confidence = 0
        for indicator in part_indicators:
            if re.search(indicator, pattern, re.IGNORECASE):
                confidence += 20

        if confidence >= 40:
            return {
                'pattern_type': 'part_number_search',
                'confidence': min(confidence, 100),
                'expected_entities': ['part_number'],
                'search_strategy': 'direct_part_lookup'
            }

        return {'confidence': 0}

    def _analyze_manufacturer_equipment_pattern(self, pattern: str) -> Dict[str, Any]:
        """Analyze patterns that look for manufacturer + equipment combinations."""
        import re

        # Look for manufacturer/equipment indicators
        mfg_indicators = [
            r'from',
            r'by',
            r'made\s+by',
            r'parts.*from',
            r'components.*from',
            r'sensors?.*valves?.*motors?'  # Equipment type lists
        ]

        confidence = 0
        capture_count = len(re.findall(r'\([^)]+\)', pattern))

        for indicator in mfg_indicators:
            if re.search(indicator, pattern, re.IGNORECASE):
                confidence += 25

        # Two capture groups often means manufacturer + equipment
        if capture_count == 2:
            confidence += 30

        if confidence >= 50:
            return {
                'pattern_type': 'manufacturer_equipment_search',
                'confidence': min(confidence, 100),
                'expected_entities': ['equipment_type', 'manufacturer'],
                'search_strategy': 'manufacturer_filtered_search'
            }

        return {'confidence': 0}

    def _analyze_safety_procedure_pattern(self, pattern: str) -> Dict[str, Any]:
        """Analyze patterns that ask about safety procedures."""
        import re

        safety_indicators = [
            r'how.*safely',
            r'safety.*procedure',
            r'safe.*to',
            r'safely.*handle'
        ]

        confidence = 0
        for indicator in safety_indicators:
            if re.search(indicator, pattern, re.IGNORECASE):
                confidence += 30

        if confidence >= 30:
            return {
                'pattern_type': 'safety_procedure_query',
                'confidence': min(confidence, 100),
                'expected_entities': ['procedure_or_equipment'],
                'search_strategy': 'safety_documentation_search'
            }

        return {'confidence': 0}

    def _analyze_availability_query_pattern(self, pattern: str) -> Dict[str, Any]:
        """Analyze patterns that ask about parts availability."""
        import re

        availability_indicators = [
            r'do\s+we\s+have',
            r'any.*parts.*from',
            r'available.*from',
            r'in\s+stock',
            r'inventory'
        ]

        confidence = 0
        capture_count = len(re.findall(r'\([^)]+\)', pattern))

        for indicator in availability_indicators:
            if re.search(indicator, pattern, re.IGNORECASE):
                confidence += 25

        # Two captures often means equipment + manufacturer for availability
        if capture_count == 2 and re.search(r'from', pattern):
            confidence += 40

        if confidence >= 50:
            return {
                'pattern_type': 'availability_query',
                'confidence': min(confidence, 100),
                'expected_entities': ['equipment_type', 'manufacturer'],
                'search_strategy': 'inventory_availability_search'
            }

        return {'confidence': 0}

    def _analyze_general_search_pattern(self, pattern: str) -> Dict[str, Any]:
        """Fallback analyzer for general search patterns."""
        import re

        search_indicators = [
            r'find',
            r'search',
            r'show',
            r'get',
            r'looking\s+for'
        ]

        confidence = 0
        for indicator in search_indicators:
            if re.search(indicator, pattern, re.IGNORECASE):
                confidence += 15

        if confidence >= 15:
            return {
                'pattern_type': 'general_search',
                'confidence': min(confidence, 100),
                'expected_entities': ['search_term'],
                'search_strategy': 'general_text_search'
            }

        return {'confidence': 10}  # Always provide fallback

    def _extract_entities_from_groups(self, pattern_analysis: Dict, captured_groups: list, user_query: str) -> Dict[
        str, Any]:
        """Extract semantic entities from regex capture groups."""

        entities = {}
        expected_entities = pattern_analysis.get('expected_entities', [])
        pattern_type = pattern_analysis.get('pattern_type', 'unknown')

        # Map captured groups to expected entities based on pattern type
        if pattern_type == 'part_number_search' and captured_groups:
            entities['part_number'] = captured_groups[0]

        elif pattern_type == 'manufacturer_equipment_search' and len(captured_groups) >= 2:
            entities['equipment_type'] = captured_groups[0]
            entities['manufacturer'] = captured_groups[1]

        elif pattern_type == 'availability_query' and len(captured_groups) >= 2:
            entities['equipment_type'] = captured_groups[0]
            entities['manufacturer'] = captured_groups[1]

        elif pattern_type == 'safety_procedure_query' and captured_groups:
            entities['procedure_or_equipment'] = captured_groups[0]

        elif captured_groups:
            # Fallback: use first capture as main entity
            entities['main_entity'] = captured_groups[0]
            if len(captured_groups) > 1:
                entities['secondary_entity'] = captured_groups[1]

        # Clean up extracted entities
        for key, value in entities.items():
            if isinstance(value, str):
                entities[key] = value.strip()

        return entities

    def _build_search_params_from_entities(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Build Part.search() parameters from extracted entities."""

        search_params = {
            'limit': 20,
            'use_fts': True,
            'exact_match': False
        }

        # Part number search
        if 'part_number' in entities:
            search_params.update({
                'part_number': entities['part_number']
            })

        # Manufacturer + equipment search
        elif 'manufacturer' in entities and 'equipment_type' in entities:
            search_params.update({
                'oem_mfg': entities['manufacturer'].upper(),
                'search_text': entities['equipment_type'],
                'fields': ['name', 'notes', 'documentation', 'class_flag', 'type']
            })

        # Manufacturer only
        elif 'manufacturer' in entities:
            search_params.update({
                'oem_mfg': entities['manufacturer'].upper()
            })

        # Equipment type only
        elif 'equipment_type' in entities:
            search_params.update({
                'search_text': entities['equipment_type'],
                'fields': ['name', 'notes', 'documentation', 'class_flag', 'type']
            })

        # General search
        elif 'main_entity' in entities:
            search_params.update({
                'search_text': entities['main_entity'],
                'fields': ['part_number', 'name', 'oem_mfg', 'model', 'notes', 'documentation']
            })

        return search_params

    def _generate_human_readable_interpretation(self, pattern_analysis: Dict, entities: Dict) -> str:
        """Generate human-readable explanation of what the pattern does."""

        pattern_type = pattern_analysis.get('pattern_type', 'unknown')

        if pattern_type == 'part_number_search':
            return f"Direct part number search for: {entities.get('part_number', 'N/A')}"

        elif pattern_type == 'manufacturer_equipment_search':
            mfg = entities.get('manufacturer', 'N/A')
            equip = entities.get('equipment_type', 'N/A')
            return f"Search for {equip} parts from manufacturer {mfg}"

        elif pattern_type == 'availability_query':
            mfg = entities.get('manufacturer', 'N/A')
            equip = entities.get('equipment_type', 'N/A')
            return f"Checking availability of {equip} parts from {mfg}"

        elif pattern_type == 'safety_procedure_query':
            procedure = entities.get('procedure_or_equipment', 'N/A')
            return f"Safety procedure query for: {procedure}"

        else:
            main = entities.get('main_entity', 'N/A')
            return f"General search for: {main}"

class UnifiedSearchWithTracking:
    """
    Enhanced version of UnifiedSearchMixin that tracks all searches using SearchQuery.
    NEW: Added smart intent classification using database patterns.
    """

    def __init__(self, unified_search_mixin):
        """Initialize with reference to your existing UnifiedSearchMixin instance."""
        self.unified_search = unified_search_mixin
        self.execute_chunk_based_search = unified_search_mixin
        self.query_tracker = None
        self.current_session_id = None

        # detected_intent_id = intent_classification['intent_id'] Store db_session so other components (like analytics) can access it
        self.db_session = getattr(unified_search_mixin, 'db_session', None)

        # detected_intent_id = intent_classification['intent_id'] Initialize PatternInterpreter
        self.pattern_interpreter = PatternInterpreter()
        logger.info("PatternInterpreter initialized")

        # detected_intent_id = intent_classification['intent_id'] If we have a DB session, init tracker and AAA search with session
        if self.db_session:
            try:
                from modules.search.nlp_search import SearchQueryTracker
                self.query_tracker = SearchQueryTracker(self.db_session)

                from modules.search.aggregate_search import AggregateSearch
                self.aggregate_search = AggregateSearch(session=self.db_session)

                logger.info("QueryTracker and AAA comprehensive part search initialized with DB session")
            except Exception as e:
                logger.error(f"Failed initializing tracker or aggregate search: {e}")
                self.query_tracker = None
                self.aggregate_search = None
        else:
            logger.warning("No database session available - tracking disabled")
            try:
                from modules.search.aggregate_search import AggregateSearch
                self.aggregate_search = AggregateSearch()
                logger.info("AAA comprehensive part search initialized (no session)")
            except Exception as e:
                logger.error(f"Failed to initialize AggregateSearch: {e}")
                self.aggregate_search = None

    def _init_aggregate_search(self):
        """Initialize the AggregateSearch class for chunk finding methods"""
        try:
            from modules.search.aggregate_search import AggregateSearch

            # Pass the session if available
            session = getattr(self, 'db_session', None)

            # Initialize AggregateSearch instance
            self.aggregate_search = AggregateSearch(session=session)
            logger.info("AggregateSearch initialized successfully")

        except ImportError as e:
            logger.error(f"Failed to import AggregateSearch: {e}")
            self.aggregate_search = None
        except Exception as e:
            logger.error(f"Failed to initialize AggregateSearch: {e}")
            self.aggregate_search = None

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
        NEW: Smart intent classification using database patterns.
        """
        import time

        if not hasattr(self, '_in_tracking_call'):
            logger.warning("Recursion flag missing, initializing...")
            self._in_tracking_call = False

        if self._in_tracking_call:
            logger.warning("Recursion detected! Falling back to direct search.")
            try:
                if hasattr(self.unified_search, 'execute_unified_search'):
                    return self.unified_search.execute_unified_search(question, user_id, request_id)
                else:
                    return {'status': 'error', 'message': 'Search method not available'}
            except Exception as e:
                return {'status': 'error', 'message': f'Fallback search failed: {str(e)}'}

        self._in_tracking_call = True

        try:
            search_start = time.time()
            user_id = user_id or "anonymous"

            logger.info(f"Executing tracked unified search for: {question}")

            if not self.current_session_id and self.query_tracker:
                try:
                    self.current_session_id = self.query_tracker.start_search_session(user_id)
                except Exception as e:
                    logger.warning(f"Failed to start search session: {e}")

            detected_intent_id = None
            intent_confidence = 0.0
            search_method = "direct_search"
            extracted_entities = {}
            normalized_query = question.lower().strip()

            logger.info(f"Starting smart intent classification for: '{question}'")
            intent_classification = self._classify_intent_from_database(question)

            if intent_classification:
                logger.info(f"Intent classified: {intent_classification['intent_name']} "
                            f"(priority: {intent_classification['priority']}, "
                            f"success_rate: {intent_classification['success_rate']})")

                detected_intent_id = intent_classification['intent_id']
                intent_confidence = intent_classification['success_rate']
                extracted_entities = intent_classification.get('extracted_data', {})

                result = self._route_by_intent_classification(question, intent_classification)
                search_method = result.get('method', intent_classification['intent_name'].lower())

            else:
                logger.info("No database pattern match found, using fallback logic")
                result = None

                if self._looks_like_part_query(question):
                    result = self._execute_part_search_with_tracking(question, {})
                    search_method = "part_search_bypass"
                elif hasattr(self.unified_search,
                             'unified_search_system') and self.unified_search.unified_search_system:
                    try:
                        if hasattr(self.unified_search.unified_search_system, 'execute_nlp_aggregated_search'):
                            search_result = self.unified_search.unified_search_system.execute_nlp_aggregated_search(
                                question)
                        else:
                            search_result = self.unified_search.unified_search_system.execute_aggregated_search(
                                question)

                        if hasattr(self.unified_search, '_organize_unified_results'):
                            result = self.unified_search._organize_unified_results(search_result, question)
                        else:
                            result = search_result

                        search_method = "nlp_aggregated_search"
                    except Exception as e:
                        logger.error(f"Search system error: {e}")
                        result = {'status': 'error', 'message': f'Search system error: {str(e)}'}
                        search_method = "search_error"
                else:
                    result = {'status': 'error', 'message': 'No search system available'}
                    search_method = "no_search_system"

            execution_time = int((time.time() - search_start) * 1000)
            result_count = result.get('total_results', 0) if isinstance(result, dict) else 0

            query_id = None
            if self.query_tracker and self.current_session_id:
                try:
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
                    logger.info(f"TRACKING SUCCESS: Query {query_id} tracked!")

                    if intent_classification:
                        self._update_pattern_usage_stats(intent_classification, result)

                    # detected_intent_id = intent_classification['intent_id'] INSERT SEARCH ANALYTICS
                    try:
                        analytics_entry = SearchAnalytics(
                            user_id=user_id,
                            session_id=self.current_session_id,
                            query_text=question,
                            detected_intent=intent_classification['intent_name'] if intent_classification else None,
                            intent_confidence=intent_confidence,
                            search_method=search_method,
                            execution_time_ms=execution_time,
                            result_count=result_count,
                            success=result.get("status") != "error" if isinstance(result, dict) else False,
                            error_message=result.get("message") if isinstance(result, dict) and result.get(
                                "status") == "error" else None,
                            user_agent=self.request_meta.get("user_agent") if hasattr(self, "request_meta") else None,
                            ip_address=self.request_meta.get("ip_address") if hasattr(self, "request_meta") else None,
                        )

                        if hasattr(self, 'db_session') and self.db_session:
                            self.db_session.add(analytics_entry)
                            self.db_session.commit()
                            logger.debug(f"SearchAnalytics entry committed for session {self.current_session_id}")
                        else:
                            logger.warning("No db_session available to commit SearchAnalytics")

                    except Exception as e:
                        logger.warning(f"Failed to insert SearchAnalytics: {e}")

                except Exception as e:
                    logger.warning(f"Failed to track query: {e}")

            if isinstance(result, dict):
                result.update({
                    'tracking_info': {
                        'query_id': query_id,
                        'session_id': self.current_session_id,
                        'detected_intent_id': detected_intent_id,
                        'intent_confidence': intent_confidence,
                        'execution_time_ms': execution_time,
                        'search_method': search_method,
                        'intent_classification_used': intent_classification is not None
                    }
                })

                if intent_classification:
                    result.update({
                        'intent_classification': {
                            'intent_name': intent_classification['intent_name'],
                            'matched_pattern': intent_classification['pattern_text'],
                            'priority': intent_classification['priority'],
                            'success_rate': intent_classification['success_rate'],
                            'extracted_data': intent_classification.get('extracted_data', {})
                        }
                    })

            return result

        except Exception as e:
            logger.error(f"Tracking search failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': f"Tracked search failed: {str(e)}",
                'search_method': 'tracking_error'
            }

        finally:
            self._in_tracking_call = False

    @with_request_id
    def _classify_intent_from_database(self, question: str) -> Dict[str, Any]:
        """
        Classify user intent using database intent_pattern table.
        Returns the highest priority matching pattern.
        """
        try:
            # Get database session from the unified_search object
            db_session = None

            if hasattr(self.unified_search, 'db_session') and self.unified_search.db_session:
                db_session = self.unified_search.db_session
            elif hasattr(self, 'query_tracker') and self.query_tracker:
                # Try to get session from query tracker
                if hasattr(self.query_tracker, 'session'):
                    db_session = self.query_tracker.session

            if not db_session:
                logger.warning("No database session available for intent classification")
                return None

            # FIXED: Use text() for raw SQL query
            from sqlalchemy import text

            # Query for matching patterns ordered by priority
            query = text("""
            SELECT 
                si.id as intent_id,  
                si.name as intent_name,
                si.priority as intent_priority,
                si.search_method,
                si.description,
                ip.pattern_text,
                ip.priority,
                ip.success_rate,
                ip.pattern_type,
                ip.usage_count,
                ip.id as pattern_id
            FROM intent_pattern ip
            JOIN search_intent si ON ip.intent_id = si.id
            WHERE ip.is_active = true 
              AND si.is_active = true
              AND :question ~ ip.pattern_text
            ORDER BY ip.priority DESC, ip.success_rate DESC, si.priority DESC
            LIMIT 1
            """)

            result = db_session.execute(query, {"question": question})
            row = result.fetchone()

            if row:
                # Extract data from the matched pattern
                extracted_data = self._extract_data_from_pattern(question, row.pattern_text, row.pattern_type)

                classification = {
                    'intent_id': row.intent_id,
                    'intent_name': row.intent_name,
                    'intent_priority': float(row.intent_priority),
                    'search_method': row.search_method,
                    'description': row.description,
                    'pattern_text': row.pattern_text,
                    'priority': float(row.priority),
                    'success_rate': float(row.success_rate),
                    'pattern_type': row.pattern_type,
                    'usage_count': row.usage_count,
                    'pattern_id': row.pattern_id,
                    'extracted_data': extracted_data
                }

                logger.debug(f"Intent classified: {row.intent_name} (priority: {row.priority})")
                return classification
            else:
                logger.debug("No matching intent patterns found in database")
                return None

        except Exception as e:
            logger.error(f"Error in database intent classification: {e}", exc_info=True)
            return None

    @with_request_id
    def _route_by_intent_classification(self, question: str, classification: Dict) -> Dict[str, Any]:
        """
        Route the search based on intent classification.
        This is the smart routing that distinguishes between AI queries and part searches.
        """
        intent_name = classification['intent_name']
        extracted_data = classification.get('extracted_data', {})

        logger.info(f"from sqlalchemy import Column, Integer, String, Text, Boolean, Float, DateTime, ForeignKey, UniqueConstraint, JSON, text Routing to intent: {intent_name}")

        try:
            # KNOWLEDGE QUERIES - AI Document Synthesis
            if intent_name == 'KNOWLEDGE_QUERY':
                return self._handle_knowledge_query(question, extracted_data, classification)

            # DIAGNOSTIC QUERIES - Problem Analysis
            elif intent_name == 'PROBLEM_DIAGNOSIS':
                return self._handle_diagnostic_query(question, extracted_data, classification)

            # PROCEDURAL QUERIES - How-to and Safety
            elif intent_name == 'HOW_TO_PROCEDURE':
                return self._handle_procedure_query(question, extracted_data, classification)

            # PART SEARCHES - Direct Database Lookups
            elif intent_name in ['FIND_PART', 'FIND_BY_MANUFACTURER', 'FIND_MOTOR', 'FIND_BEARING',
                                 'FIND_VALVE', 'FIND_SENSOR', 'FIND_BY_MODEL', 'FIND_BY_SPECIFICATION',
                                 'FIND_SWITCH', 'FIND_BELT', 'FIND_CABLE', 'FIND_SEAL']:
                return self._handle_part_search_classified(question, extracted_data, classification)

            # IMAGE SEARCHES
            elif intent_name == 'SHOW_IMAGES':
                return self._handle_image_search_classified(question, extracted_data, classification)

            # LOCATION SEARCHES
            elif intent_name == 'LOCATION_SEARCH':
                return self._handle_location_search_classified(question, extracted_data, classification)

            # OTHER INTENTS - Use existing part search method as fallback
            else:
                logger.info(f"from sqlalchemy import Column, Integer, String, Text, Boolean, Float, DateTime, ForeignKey, UniqueConstraint, JSON, text Using existing part search method for intent: {intent_name}")
                return self._execute_part_search_with_tracking(question, extracted_data)

        except Exception as e:
            logger.error(f"Error in intent routing: {e}", exc_info=True)
            # Fallback to existing search method
            return self._execute_part_search_with_tracking(question, {})

    @with_request_id
    def _extract_captured_groups_from_data(self, extracted_data: Dict) -> list:
        """Extract the captured groups from the extraction data."""
        # This depends on how your _extract_data_from_pattern stores the groups
        # Assuming it stores them in a predictable way
        groups = []

        main_entity = extracted_data.get('main_entity')
        if main_entity:
            groups.append(main_entity)

        # Look for additional captured groups
        for key, value in extracted_data.items():
            if key.startswith('group_') or key.startswith('capture_'):
                groups.append(value)

        return groups

    # Integration with the existing handler
    @with_request_id
    def _handle_part_search_classified(self, question: str, extracted_data: Dict, classification: Dict) -> Dict[
        str, Any]:
        """
        Updated handler that uses PatternInterpreter to dynamically understand patterns.
        """
        pattern_text = classification.get('pattern_text', '')
        captured_groups = self._extract_captured_groups_from_data(extracted_data)

        # Use the pattern interpreter
        interpreter = PatternInterpreter()
        interpretation = interpreter.interpret_pattern_and_extract(
            pattern_text=pattern_text,
            user_query=question,
            extracted_groups=captured_groups
        )

        # Get the search parameters from interpretation
        search_params = interpretation['search_params']

        logger.info(f"Pattern interpretation: {interpretation['interpretation']}")
        logger.info(f"Search parameters: {search_params}")

        # Execute the search
        try:
            from modules.emtacdb.emtacdb_fts import Part
            parts = Part.search(**search_params)

            return {
                'status': 'success',
                'results': parts,
                'total_results': len(parts),
                'interpretation': interpretation,
                'search_params': search_params
            }

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'interpretation': interpretation,
                'search_params': search_params
            }

    @with_request_id
    def _looks_like_part_number(self, text: str) -> bool:
        """Simple check if text looks like a part number."""
        import re
        return bool(re.match(r'^[A-Z]\d{5,}$', text.upper())) if text else False

    def _handle_knowledge_query(self, question: str, extracted_data: Dict, classification: Dict) -> Dict[str, Any]:
        """
        Handle knowledge queries - the "smart person in the room" functionality.
        Now uses chunk-based search + AI synthesis for intelligent responses.
        """
        logger.info(f"Handling knowledge query: {question}")

        topic = extracted_data.get('topic', extracted_data.get('main_entity', question))

        # Step 1: Try to find relevant document chunks
        try:
            # Get the AistManager instance (assuming self.unified_search is the AistManager)
            aist_manager = getattr(self, 'unified_search', None)

            if not aist_manager:
                logger.error("AistManager instance not available")
                return {
                    'status': 'error',
                    'answer': f"**Configuration Error**\n\nThe knowledge query system is not properly configured.\n\nTopic: **{topic}**",
                    'method': 'knowledge_query_config_error',
                    'search_type': 'knowledge_query',
                    'topic': topic,
                    'total_results': 0,
                    'intent_matched': classification['intent_name'],
                    'pattern_used': classification['pattern_text'],
                    'error': 'AistManager not available'
                }

            # Call the chunk search method on the AistManager
            chunk_result = aist_manager.execute_chunk_based_search(
                question=question,
                user_id=extracted_data.get('user_id'),
                request_id=extracted_data.get('request_id')
            )

            # Step 2: If we found relevant chunks, send to AI for synthesis
            if chunk_result.get('chunk_found') and chunk_result.get('content'):
                logger.info(f"Found relevant chunk with {chunk_result.get('similarity', 0):.1%} similarity")

                # Use the AI query method to synthesize the answer
                ai_result = aist_manager.query_ai_with_context(
                    user_query=question,
                    context_text=chunk_result['content'],
                    user_id=extracted_data.get('user_id'),
                    request_id=extracted_data.get('request_id')
                )

                if ai_result.get('status') == 'success':
                    return {
                        'status': 'success',
                        'answer': ai_result['answer'],
                        'method': 'ai_knowledge_synthesis_with_chunks',
                        'search_type': 'knowledge_query_enhanced',
                        'topic': topic,
                        'source_info': {
                            'chunk_similarity': chunk_result.get('similarity', 0.0),
                            'chunk_metadata': chunk_result.get('metadata', {}),
                            'document_source': chunk_result.get('metadata', {}).get('document_title', 'Unknown')
                        },
                        'total_results': 1,
                        'intent_matched': classification['intent_name'],
                        'pattern_used': classification['pattern_text'],
                        'ai_model_used': ai_result.get('model_name', 'unknown')
                    }

            # Step 3: If no chunks found, try direct AI query
            logger.info("No relevant chunks found, using direct AI query")

            ai_result = aist_manager.query_ai_simple(
                question=question,
                user_id=extracted_data.get('user_id'),
                request_id=extracted_data.get('request_id')
            )

            if ai_result.get('status') == 'success':
                return {
                    'status': 'success',
                    'answer': ai_result['answer'],
                    'method': 'ai_knowledge_synthesis_direct',
                    'search_type': 'knowledge_query_direct',
                    'topic': topic,
                    'source_info': {
                        'source_type': 'ai_general_knowledge',
                        'note': 'No specific documentation found, using AI general knowledge'
                    },
                    'total_results': 1,
                    'intent_matched': classification['intent_name'],
                    'pattern_used': classification['pattern_text'],
                    'ai_model_used': ai_result.get('model_name', 'unknown')
                }

            # Step 4: Fallback if AI also fails
            return {
                'status': 'partial_success',
                'answer': f"**Knowledge Query Processing**\n\nYou asked about: **{topic}**\n\nI attempted to find relevant documentation and provide an AI-synthesized answer, but encountered some issues. The search and AI systems are configured, but may need attention.\n\nWould you like to try rephrasing your question?",
                'method': 'knowledge_query_fallback',
                'search_type': 'knowledge_query',
                'topic': topic,
                'total_results': 0,
                'intent_matched': classification['intent_name'],
                'pattern_used': classification['pattern_text'],
                'error_info': {
                    'chunk_search_status': chunk_result.get('status', 'unknown'),
                    'ai_query_status': ai_result.get('status',
                                                     'unknown') if 'ai_result' in locals() else 'not_attempted'
                }
            }

        except Exception as e:
            logger.error(f"Error in knowledge query processing: {e}")

            return {
                'status': 'error',
                'answer': f"**Processing Error**\n\nI encountered an error while trying to answer your question about **{topic}**.\n\nError: {str(e)}\n\nPlease try again or contact support.",
                'method': 'knowledge_query_error',
                'search_type': 'knowledge_query',
                'topic': topic,
                'total_results': 0,
                'intent_matched': classification['intent_name'],
                'pattern_used': classification['pattern_text'],
                'error': str(e)
            }

    def _handle_diagnostic_query(self, question: str, extracted_data: Dict, classification: Dict) -> Dict[str, Any]:
        """Handle diagnostic/troubleshooting queries."""
        logger.info(f"🔧 Handling diagnostic query: {question}")

        problem = extracted_data.get('main_entity', question)

        return {
            'status': 'success',
            'answer': f"🔧 **Diagnostic Query Detected!**\n\nYou asked about: **{problem}**\n\nThis would search diagnostic documentation and provide troubleshooting analysis.\n\n*[AI Diagnostic Analysis would happen here]*",
            'method': 'ai_diagnostic_analysis',
            'search_type': 'diagnostic_query',
            'problem': problem,
            'total_results': 1,
            'intent_matched': classification['intent_name']
        }

    def _handle_procedure_query(self, question: str, extracted_data: Dict, classification: Dict) -> Dict[str, Any]:
        """Handle procedure/how-to queries."""
        logger.info(f"from sqlalchemy import Column, Integer, String, Text, Boolean, Float, DateTime, ForeignKey, UniqueConstraint, JSON, text Handling procedure query: {question}")

        procedure = extracted_data.get('main_entity', question)

        return {
            'status': 'success',
            'answer': f"from sqlalchemy import Column, Integer, String, Text, Boolean, Float, DateTime, ForeignKey, UniqueConstraint, JSON, text **Procedure Query Detected!**\n\nYou asked about: **{procedure}**\n\nThis would search procedure documentation and provide step-by-step guidance.\n\n*[AI Procedure Explanation would happen here]*",
            'method': 'ai_procedure_explanation',
            'search_type': 'procedure_query',
            'procedure': procedure,
            'total_results': 1,
            'intent_matched': classification['intent_name']
        }

    def _handle_image_search_classified(self, question: str, extracted_data: Dict, classification: Dict) -> Dict[
        str, Any]:
        """Handle image search with classification."""
        logger.info(f"📸 Handling image search: {question}")

        # For now, use existing part search as fallback
        return self._execute_part_search_with_tracking(question, extracted_data)

    def _handle_location_search_classified(self, question: str, extracted_data: Dict, classification: Dict) -> Dict[
        str, Any]:
        """Handle location search with classification."""
        logger.info(f"📍 Handling location search: {question}")

        # For now, use existing part search as fallback
        return self._execute_part_search_with_tracking(question, extracted_data)

    def _extract_data_from_pattern(self, question: str, pattern_text: str, pattern_type: str) -> Dict[str, Any]:
        """
        Simplified: Extract basic data from regex groups and trust the database intent classification.
        The database patterns have already done the hard work of intent classification.
        """
        import re

        try:
            extracted = {}

            # Use regex to extract capture groups from the pattern
            match = re.search(pattern_text, question, re.IGNORECASE)
            if not match:
                logger.warning(f"Pattern matched in SQL but not in Python: {pattern_text}")
                return extracted

            groups = match.groups()
            logger.debug(f"Pattern '{pattern_text[:50]}...' captured groups: {groups}")

            # Simple extraction - just capture what the regex found
            if len(groups) >= 1 and groups[0]:
                main_entity = groups[0].strip()
                extracted['main_entity'] = main_entity
                extracted['topic'] = main_entity
                logger.debug(f" Extracted main entity: {main_entity}")

            # Capture secondary entity if present
            if len(groups) >= 2 and groups[1]:
                secondary_entity = groups[1].strip()
                extracted['secondary_entity'] = secondary_entity
                logger.debug(f" Extracted secondary entity: {secondary_entity}")

            # Add pattern metadata
            extracted.update({
                'pattern_matched': pattern_text,
                'pattern_type': pattern_type,
                'extraction_confidence': 75.0,  # Default confidence - let database classification handle specifics
                'groups_captured': len(groups),
                'full_match': match.group(0) if match else ''
            })

            logger.debug(f" Simple extraction complete: {extracted}")
            return extracted

        except Exception as e:
            logger.error(f"Error extracting data from pattern '{pattern_text}': {e}")
            return {'error': str(e), 'pattern_text': pattern_text}

    def _update_pattern_usage_stats(self, classification: Dict, result: Dict):
        """Update usage statistics for the matched pattern."""
        try:
            if not hasattr(self.unified_search, 'db_session') or not self.unified_search.db_session:
                return

            pattern_id = classification.get('pattern_id')
            if pattern_id:
                # Increment usage count
                update_query = text("""
                UPDATE intent_pattern 
                SET usage_count = usage_count + 1,
                    updated_at = NOW()
                WHERE id = :pattern_id
                """)
                self.unified_search.db_session.execute(update_query,
                                                       {"pattern_id": pattern_id})  #  Dictionary for named parameters
                self.unified_search.db_session.commit()

                logger.debug(f"Updated usage count for pattern {pattern_id}")

        except Exception as e:
            logger.error(f"Error updating pattern usage stats: {e}")
            # Don't let this error break the search flow
            pass

    # Keep all existing methods unchanged
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
            'gear', 'sensor', 'motor', 'pump', 'valve', 'bearing', 'banner'
        ]
        return any(indicator in question_lower for indicator in part_indicators)

    def _execute_part_search_with_tracking(self, question: str, search_params: Dict) -> Dict[str, Any]:
        """Execute part search with enhanced tracking - FIXED METHOD CALLS."""
        try:
            logger.info(f"from sqlalchemy import Column, Integer, String, Text, Boolean, Float, DateTime, ForeignKey, UniqueConstraint, JSON, text Executing part search with tracking for: {question}")

            result = None

            if hasattr(self.unified_search, 'unified_search_system') and self.unified_search.unified_search_system:
                # Call the search system directly with correct method name
                if hasattr(self.unified_search.unified_search_system, 'execute_nlp_aggregated_search'):
                    logger.info(" Using AAA comprehensive part search")
                    aaa_params = self._convert_to_aaa_search_params(question, search_params)
                    search_result = self.aggregate_search.aaa_comprehensive_part_search(aaa_params)
                elif hasattr(self.unified_search.unified_search_system, 'execute_aggregated_search'):
                    logger.info(" Using execute_aggregated_search for part search")
                    search_result = self.unified_search.unified_search_system.execute_aggregated_search(question)
                else:
                    logger.warning("from sqlalchemy import Column, Integer, String, Text, Boolean, Float, DateTime, ForeignKey, UniqueConstraint, JSON, textfrom sqlalchemy import Column, Integer, String, Text, Boolean, Float, DateTime, ForeignKey, UniqueConstraint, JSON, text No search method available, using fallback")
                    search_result = {
                        'status': 'success',
                        'results': [],
                        'total_results': 0,
                        'message': f"Part search completed for: {question}",
                        'search_method': 'part_search_fallback'
                    }

                # Organize results if we have an organizer method
                if hasattr(self.unified_search, '_organize_unified_results'):
                    result = self.unified_search._organize_unified_results(search_result, question)
                else:
                    # Basic organization
                    result = {
                        'status': search_result.get('status', 'success'),
                        'results_by_type': {'parts': search_result.get('results', [])},
                        'total_results': search_result.get('total_results', 0),
                        'message': search_result.get('message', f"Search completed for: {question}"),
                        'search_method': search_result.get('search_method', 'part_search'),
                        'timestamp': datetime.utcnow().isoformat()
                    }
            else:
                # Fallback if no search system available
                logger.warning("from sqlalchemy import Column, Integer, String, Text, Boolean, Float, DateTime, ForeignKey, UniqueConstraint, JSON, textfrom sqlalchemy import Column, Integer, String, Text, Boolean, Float, DateTime, ForeignKey, UniqueConstraint, JSON, text No unified search system available")
                result = {
                    'status': 'error',
                    'message': 'Search system not available',
                    'search_method': 'part_search_no_system',
                    'total_results': 0,
                    'results_by_type': {}
                }

            # Add tracking-specific metadata
            if isinstance(result, dict):
                result.update({
                    'search_method': 'enhanced_part_search_bypass',
                    'parameters_used': search_params,
                    'bypass_method': 'direct_comprehensive_part_search',
                    'tracking_enabled': True
                })

            logger.info(f"from sqlalchemy import Column, Integer, String, Text, Boolean, Float, DateTime, ForeignKey, UniqueConstraint, JSON, text Part search completed: {result.get('total_results', 0)} results")
            return result

        except Exception as e:
            logger.error(f"detected_intent_id = intent_classification['intent_id'] Part search with tracking failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e),
                'search_method': 'enhanced_part_search_bypass_error',
                'total_results': 0,
                'results_by_type': {}
            }

    def _convert_to_aaa_search_params(self, question: str, enhanced_params: Dict) -> Dict[str, Any]:
        """
        Convert enhanced classification params to AAA comprehensive part search parameters.
        Add this method to the same class that contains _execute_part_search_with_tracking.
        """
        logger.debug(f"Converting params for AAA search: {enhanced_params}")

        # Base AAA parameters
        aaa_params = {
            'search_text': question,
            'user_id': enhanced_params.get('user_id', 'classified_user'),
            'request_id': enhanced_params.get('request_id'),
            'limit': enhanced_params.get('limit', 20),
            'extraction_method': 'database_pattern_classification'
        }

        # PRIORITY 1: Direct part number (highest confidence)
        if 'part_number' in enhanced_params:
            aaa_params['part_number'] = enhanced_params['part_number']
            aaa_params['search_text'] = enhanced_params['part_number']  # Override search text
            logger.info(f"AAA: Direct part number search for {enhanced_params['part_number']}")

        # PRIORITY 2: Manufacturer + equipment combination
        elif 'manufacturer' in enhanced_params and 'equipment_type' in enhanced_params:
            # This will trigger manufacturer_plus_equipment strategy in AAA search
            aaa_params['search_text'] = f"{enhanced_params['manufacturer']} {enhanced_params['equipment_type']}"
            logger.info(
                f"AAA: Manufacturer+equipment search for {enhanced_params['manufacturer']} {enhanced_params['equipment_type']}")

        # PRIORITY 3: Manufacturer only
        elif 'manufacturer' in enhanced_params:
            aaa_params['search_text'] = enhanced_params['manufacturer']
            logger.info(f"AAA: Manufacturer search for {enhanced_params['manufacturer']}")

        # PRIORITY 4: Equipment type only
        elif 'equipment_type' in enhanced_params:
            aaa_params['search_text'] = enhanced_params['equipment_type']
            logger.info(f"AAA: Equipment search for {enhanced_params['equipment_type']}")

        # PRIORITY 5: Location search
        elif 'location' in enhanced_params or 'location_id' in enhanced_params:
            location = enhanced_params.get('location') or enhanced_params.get('location_id')
            aaa_params['search_text'] = f"location {location}"
            aaa_params['entity_type'] = 'location'
            logger.info(f"AAA: Location search for {location}")

        # PRIORITY 6: Description/general search
        elif 'description' in enhanced_params:
            aaa_params['search_text'] = enhanced_params['description']
            logger.info(f"AAA: Description search for {enhanced_params['description']}")

        # Add classification metadata for analytics
        aaa_params['classification_data'] = {
            'intent': enhanced_params.get('classification'),
            'entities_extracted': enhanced_params,
            'extraction_confidence': enhanced_params.get('extraction_confidence', 0),
            'pattern_matched': enhanced_params.get('pattern_matched')
        }

        logger.debug(f"Final AAA params: {aaa_params}")
        return aaa_params

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