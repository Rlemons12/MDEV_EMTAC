# modules/search/aggregate_search.py
"""
Comprehensive aggregated search system for the chatbot.
Aggregates information from multiple sources and search paths to provide
comprehensive contextual search results across all database entities.
"""

import json
import re
from modules.configuration.log_config import logger,with_request_id,get_request_id,debug_id,error_id
from fuzzywuzzy import fuzz, process
from typing import List, Tuple, Dict, Any, Optional
from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import (
        Image, Part, Position, Area, EquipmentGroup, Model, Location,
        PartsPositionImageAssociation, ToolImageAssociation,
        ImageTaskAssociation, ImageProblemAssociation,
        KeywordAction  # Legacy compatibility
    )
from modules.emtacdb.emtacdb_fts import Document, DocumentEmbedding, CompleteDocument
from plugins.ai_modules import generate_embedding, ModelsConfig
from modules.configuration.config_env import DatabaseConfig
from sqlalchemy import text, func

MODELS_AVAILABLE = True

class AggregateSearch:
    """
    Comprehensive aggregated search system for the chatbot.
    Aggregates information from multiple sources and search paths to provide
    comprehensive contextual search results across all database entities.
    """

    # Cache for storing recent search results (limit size in a production environment)
    _search_cache = {}
    _cache_limit = 100  # Maximum number of cached searches

    def __init__(self, session=None):
        """
        Initialize the AggregateSearch class.

        Args:
            session: SQLAlchemy session (optional)
        """
        self._session = session
        self._known_manufacturers = None  # Cache for manufacturers
        if MODELS_AVAILABLE:
            self._db_config = DatabaseConfig()
        else:
            self._db_config = None
            logger.warning("Database config not available - running in limited mode")

    @property
    def session(self):
        """Get or create a database session if needed."""
        if not MODELS_AVAILABLE:
            return None

        if self._session is None and self._db_config:
            self._session = self._db_config.get_main_session()
        return self._session

    def close_session(self):
        """Close the session if it was created by this class."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self):
        """Support for context manager usage."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close session when exiting context."""
        self.close_session()

    #initilize know manufactuers
    @with_request_id
    def _get_known_manufacturers(self):
        """Get all unique manufacturers from the database (cached)."""
        if self._known_manufacturers is None:
            try:
                if not self.session:
                    logger.warning("No database session available for manufacturer loading")
                    self._known_manufacturers = set()
                    return self._known_manufacturers

                from modules.emtacdb.emtacdb_fts import Part

                # Get all unique manufacturers, excluding nulls and empty strings
                manufacturers = self.session.query(Part.oem_mfg).distinct().filter(
                    Part.oem_mfg.isnot(None),
                    Part.oem_mfg != ''
                ).all()

                # Create a set of uppercase manufacturer names for fast lookup
                self._known_manufacturers = {
                    mfg[0].upper().strip()
                    for mfg in manufacturers
                    if mfg[0] and mfg[0].strip()
                }

                logger.info(f" Loaded {len(self._known_manufacturers)} manufacturers from database")

            except Exception as e:
                logger.error(f"Failed to load manufacturers: {e}")
                self._known_manufacturers = set()

        return self._known_manufacturers

    def _is_known_manufacturer(self, word: str) -> bool:
        """Check if a word is a known manufacturer."""
        if not word or len(word) < 2:
            return False

        known_mfgs = self._get_known_manufacturers()
        return word.upper().strip() in known_mfgs

    def _extract_manufacturer_name(self, text: str, entities: dict) -> str:
        """
        Extract manufacturer name by comparing question words against database manufacturers.
        Intent classification already determined this is manufacturer-related.
        Now find and validate the actual manufacturer name.
        """

        # 1. Trust the regex extraction first
        if 'main_entity' in entities:
            entity = entities['main_entity'].strip()

            # Clean up common patterns
            if entity.startswith('of '):
                entity = entity[3:].strip()  # "of banner" -> "banner"

            # Validate: "Yep, this is a real manufacturer in our database"
            if self._is_known_manufacturer(entity):
                logger.debug(f" Confirmed manufacturer from entities: {entity}")
                return entity.upper()
            else:
                logger.debug(f" Entity '{entity}' not found in manufacturer database")

        # 2. Scan question words as backup
        words = text.upper().split()
        for word in words:
            # Skip common words
            if word.lower() in ['list', 'of', 'show', 'get', 'find', 'the', 'a', 'an']:
                continue

            if self._is_known_manufacturer(word):
                logger.debug(f" Found manufacturer in question: {word}")
                return word

        # 3. Intent was confident but we can't find the manufacturer
        logger.warning(f" Intent said manufacturer, but couldn't extract from: '{text}'")
        logger.warning(f"Available entities: {entities}")

        # Return the main entity anyway, or fallback to first meaningful word
        if 'main_entity' in entities:
            entity = entities['main_entity'].strip()
            if entity.startswith('of '):
                entity = entity[3:].strip()
            return entity.upper()

        # Last resort: return first non-common word
        words = text.split()
        for word in words:
            if word.lower() not in ['list', 'of', 'show', 'get', 'find', 'the', 'a', 'an']:
                return word.upper()

        return "UNKNOWN_MANUFACTURER"

    def _extract_equipment_type(self, text: str) -> str:
        """Extract equipment type from the question."""
        equipment_types = [
            'sensors', 'sensor', 'motors', 'motor', 'valves', 'valve',
            'pumps', 'pump', 'switches', 'switch', 'bearings', 'bearing',
            'filters', 'filter', 'relays', 'relay', 'belts', 'belt',
            'seals', 'seal', 'gaskets', 'gasket', 'actuators', 'actuator'
        ]

        words = text.lower().split()
        for word in words:
            if word in equipment_types:
                logger.debug(f" Found equipment type: {word}")
                return word

        logger.debug(f" No specific equipment type found in: {text}")
        return ""
    #===================updated=========================
    # ============================================================================
    # AAA_PART SEARCH FUNCTIONS - Consolidated and Prefixed
    # ============================================================================
    # All part search related functions grouped together with aaa_ prefix
    # for easy identification and organization in AggregateSearch class
    @with_request_id
    def aaa_comprehensive_part_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        ULTIMATE: Complete part search combining all best practices
        UPDATED: Trust database intent classification first, uses manufacturer extraction

        Features:
        - Database intent pattern classification (primary)
        - Manufacturer extraction from database validation
        - Enhanced dual part number detection (fallback)
        - Fuzzy search fallback for typos/variations
        - Robust SQL fallback for database errors
        - Comprehensive result formatting with relationships
        - Parameter enhancement and validation
        - Logs successful queries for analytics
        - Leverages PostgreSQL TSVECTOR for performance
        """
        logger.debug(f"Starting ultimate part search with params: {params}")

        if not MODELS_AVAILABLE:
            return {
                "status": "error",
                "message": "Database models not available for part search"
            }

        try:
            if self.session:
                try:
                    self.session.rollback()
                    logger.debug("Session rolled back for clean state")
                except Exception:
                    pass

            from modules.emtacdb.emtacdb_fts import Part
            from modules.search.utils import extract_search_terms

            search_kwargs = {}
            search_method = "unknown"
            detection_results = {}
            fuzzy_results = []
            confidence = 0

            if 'search_text' in params and params['search_text']:
                original_search_text = params['search_text']
                logger.info(f"ENHANCED DUAL SEARCH for: '{original_search_text}'")

                # NEW: Check if we already have database intent classification
                classification_data = params.get('classification_data', {})
                if classification_data and classification_data.get('intent'):
                    # Trust the database classification
                    intent = classification_data['intent']
                    logger.info(f"Using database intent classification: {intent}")

                    entities_extracted = classification_data.get('entities_extracted', {})

                    if intent == 'FIND_BY_MANUFACTURER':
                        # UPDATED: Use new manufacturer extraction method
                        manufacturer = self._extract_manufacturer_name(original_search_text, entities_extracted)
                        equipment = self._extract_equipment_type(original_search_text)

                        logger.info(f"Extracted manufacturer: '{manufacturer}', equipment: '{equipment}'")

                        if equipment:
                            # Manufacturer + Equipment search
                            search_strategy = f"manufacturer:{manufacturer} equipment:{equipment}"
                            search_kwargs = {
                                'oem_mfg': manufacturer,
                                'search_text': equipment,
                                'fields': ['name', 'notes', 'documentation', 'class_flag', 'type'],
                                'exact_match': False,
                                'use_fts': True,  # Enable TSVECTOR for equipment search
                                'limit': int(params.get('limit', 20))
                            }
                            search_method = "manufacturer_plus_equipment"
                            logger.info(f"Using manufacturer+equipment search: {manufacturer} + {equipment}")
                        else:
                            # Manufacturer only search
                            search_strategy = f"manufacturer:{manufacturer}"
                            search_kwargs = {
                                'oem_mfg': manufacturer,
                                'exact_match': False,
                                'use_fts': False,  # Direct field match for manufacturer
                                'limit': int(params.get('limit', 20))
                            }
                            search_method = "manufacturer_only"
                            logger.info(f"Using manufacturer-only search: {manufacturer}")

                        confidence = 90
                        detection_type = "database_intent_classification"
                        detection_results = classification_data

                    elif intent in ['FIND_PART', 'FIND_BY_MODEL', 'FIND_BY_SPECIFICATION']:
                        # For part searches, use general search
                        search_strategy = f"equipment:{original_search_text}"
                        search_kwargs = {
                            'search_text': original_search_text,
                            'fields': ['name', 'notes', 'documentation', 'class_flag', 'type'],
                            'exact_match': False,
                            'use_fts': True,  # Enable TSVECTOR for equipment search
                            'limit': int(params.get('limit', 20))
                        }
                        search_method = "database_classified_part"
                        confidence = 85
                        detection_type = "database_intent_classification"
                        detection_results = classification_data

                    else:
                        # Default to equipment search for other intents
                        search_strategy = f"equipment:{original_search_text}"
                        search_kwargs = {
                            'search_text': original_search_text,
                            'fields': ['name', 'notes', 'documentation', 'class_flag', 'type'],
                            'exact_match': False,
                            'use_fts': True,  # Enable TSVECTOR for equipment search
                            'limit': int(params.get('limit', 20))
                        }
                        search_method = "database_classified_general"
                        confidence = 80
                        detection_type = "database_intent_classification"
                        detection_results = classification_data

                else:
                    # Fallback to complex detection only if no database classification
                    logger.info("No database classification found, using complex detection")
                    search_result = self.aaa_enhanced_search_with_fuzzy_fallback(
                        original_search_text, self.session
                    )

                    search_strategy = search_result['search_strategy']
                    detection_type = search_result['detection_type']
                    confidence = search_result['confidence']
                    detection_results = search_result.get('detected_categories', {})
                    fuzzy_results = search_result.get('fuzzy_results', [])

                logger.info(f"Strategy: {search_strategy} ({detection_type}, confidence: {confidence})")

                # Process the search strategy - REMOVED old manufacturer: parsing since we now set search_kwargs directly
                if search_strategy.startswith('company_part:'):
                    part_number = search_strategy.replace('company_part:', '').strip()
                    search_kwargs = {
                        'part_number': part_number,
                        'exact_match': True,
                        'use_fts': True,  # Enable TSVECTOR
                        'limit': int(params.get('limit', 20))
                    }
                    search_method = "company_part_number"

                elif search_strategy.startswith('mfg_part:'):
                    model = search_strategy.replace('mfg_part:', '').strip()
                    search_kwargs = {
                        'model': model,
                        'exact_match': False,
                        'use_fts': True,  # Enable TSVECTOR
                        'limit': int(params.get('limit', 20))
                    }
                    search_method = "manufacturer_part_number"

                elif search_strategy.startswith('manufacturer:') and 'equipment:' in search_strategy:
                    # This is handled above in the FIND_BY_MANUFACTURER section
                    pass

                elif search_strategy.startswith('manufacturer:'):
                    # This is handled above in the FIND_BY_MANUFACTURER section
                    pass

                elif search_strategy.startswith('equipment:'):
                    equipment = search_strategy.replace('equipment:', '').strip()
                    search_kwargs = {
                        'search_text': equipment,
                        'fields': ['name', 'notes', 'documentation', 'class_flag', 'type'],
                        'exact_match': False,
                        'use_fts': True,  # Enable TSVECTOR for equipment search
                        'limit': int(params.get('limit', 20))
                    }
                    search_method = "equipment_only"

                elif not search_kwargs:  # Only use fallback if search_kwargs wasn't already set above
                    enhanced_params = self.aaa_enhance_comprehensive_part_search_parameters(params, params)
                    search_kwargs = enhanced_params
                    search_method = "enhanced_general_search"

            elif 'part_number' in params:
                logger.info(f"DIRECT PART SEARCH for part number: '{params['part_number']}'")
                search_kwargs = {
                    'part_number': params['part_number'],
                    'exact_match': True,
                    'use_fts': False,  # Direct part number match doesn't need FTS
                    'limit': int(params.get('limit', 20))
                }
                search_method = "direct_part_number_search"
                confidence = 95

            elif 'extracted_id' in params:
                logger.info(f"DIRECT PART SEARCH for ID: {params['extracted_id']}")
                search_kwargs = {
                    'part_id': params['extracted_id'],
                    'limit': 1
                }
                search_method = "id_search"
                confidence = 100

            else:
                logger.warning("No valid search parameters for part search")
                return {
                    'status': 'error',
                    'message': 'No valid search parameters (need search_text, part_number, or extracted_id)',
                    'params_received': params
                }

            logger.debug(f"Executing Part.search with kwargs: {search_kwargs}")

            parts = []
            try:
                search_kwargs['session'] = self.session
                if params.get('request_id'):
                    search_kwargs['request_id'] = params['request_id']

                # This leverages Part.search() which uses TSVECTOR when use_fts=True
                parts = Part.search(**search_kwargs)
                logger.info(f"Part.search found {len(parts)} parts using {search_method}")

            except Exception as search_error:
                logger.warning(f"Part.search failed: {search_error}")
                parts = self.aaa_fallback_sql_part_search(search_kwargs)
                search_method += "_sql_fallback"

            comprehensive_results = []

            for i, part in enumerate(parts):
                try:
                    if hasattr(part, 'part_number'):
                        part_result = self.aaa_enhance_part_result(part)
                        part_result.update({
                            'search_method': search_method,
                            'confidence': confidence,
                            'detection_type': detection_results.get('detection_type', search_method),
                            'entity_type': 'part'
                        })
                    elif isinstance(part, dict):
                        part_result = self.aaa_create_part_result_from_dict(part, i, search_method, confidence)
                    else:
                        part_result = self.aaa_create_part_result_from_object(part, i, search_method)

                    comprehensive_results.append(part_result)
                    logger.debug(f"Enhanced part {i + 1}: {part_result['part_number']}")

                except Exception as format_error:
                    logger.warning(f"Failed to format part {i + 1}: {format_error}")
                    comprehensive_results.append(self.aaa_create_error_part_result(i, search_method))

            result = {
                'status': 'success',
                'count': len(comprehensive_results),
                'results': comprehensive_results,
                'entity_type': 'part',
                'search_type': 'ultimate_comprehensive_part_search',
                'search_method': search_method,
                'confidence': confidence,
                'search_parameters': search_kwargs,
                'detection_results': detection_results,
                'fuzzy_results': fuzzy_results,
                'enhancement_stats': self.aaa_calculate_enhancement_stats(comprehensive_results),
                'note': f'Ultimate comprehensive search using {search_method} with TSVECTOR and relationship enhancement'
            }

            # Track analytics if user_id provided
            if params.get("user_id"):
                try:
                    result["query_id"] = self.aaa_track_search_analytics(params, comprehensive_results, confidence,
                                                                         search_method)
                except Exception as e:
                    logger.warning(f"[tracking] Failed to log SearchQuery: {e}")

            logger.info(f"Ultimate part search SUCCESS: {len(comprehensive_results)} enhanced parts found")
            return result

        except Exception as e:
            logger.error(f"Ultimate part search ERROR: {e}", exc_info=True)
            if self.session:
                try:
                    self.session.rollback()
                except Exception:
                    pass
            return {
                'status': 'error',
                'message': f"Ultimate part search failed: {str(e)}",
                'entity_type': 'part',
                'search_parameters': params,
                'error_type': 'ultimate_part_search_error'
            }

    @with_request_id
    def aaa_enhanced_search_with_fuzzy_fallback(self, query: str, session) -> Dict[str, Any]:
        """
        Enhanced search with intelligent dual part detection + fuzzy fallback

        Flow:
        1. Try exact enhanced detection
        2. If no results, try fuzzy matching
        3. Return best results with confidence scores
        """
        from modules.search.utils import extract_search_terms

        # Extract terms from query
        terms = [term.strip() for term in query.upper().split() if term.strip()]
        logger.debug(f"Analyzing terms: {terms}")

        # STEP 1: Try exact enhanced detection first
        detected = self.aaa_detect_part_numbers_and_manufacturers(terms, session)

        # Check if we found exact matches
        exact_matches_found = any([
            detected['company_part_numbers'],
            detected['mfg_part_numbers'],
            detected['manufacturers']
        ])

        if exact_matches_found:
            # Use exact detection
            enhanced_query = self.aaa_enhanced_search_with_dual_part_detection(query, session)
            return {
                'search_strategy': enhanced_query,
                'detection_type': 'exact_database_match',
                'confidence': 95,
                'detected_categories': detected
            }

        # STEP 2: No exact matches - try fuzzy search
        logger.info(f"No exact matches found, trying fuzzy search for: {query}")

        # Extract potential part candidates from the query
        part_candidates = self.aaa_extract_part_candidates(query)

        # Add all terms as potential part numbers if no specific candidates found
        if not part_candidates:
            part_candidates.extend(terms)

            # Add combinations for multi-word part numbers
            if len(terms) >= 2:
                part_candidates.append(''.join(terms))  # "ABC 123" → "ABC123"
                part_candidates.append('-'.join(terms))  # "ABC 123" → "ABC-123"
                part_candidates.append('_'.join(terms))  # "ABC 123" → "ABC_123"

        logger.debug(f"Fuzzy search candidates: {part_candidates}")

        # Perform fuzzy search
        fuzzy_results = self.aaa_fuzzy_part_search(part_candidates, threshold=70)

        if fuzzy_results:
            # Get best fuzzy match
            best_part, best_score = fuzzy_results[0]

            logger.info(f"Best fuzzy match: {best_part.part_number} (score: {best_score})")

            # Determine what type of match this was
            if best_score >= 90:
                search_strategy = f"company_part:{best_part.part_number}"  # High confidence
                detection_type = 'fuzzy_exact'
            elif best_part.model and any(term in best_part.model.upper() for term in terms):
                search_strategy = f"mfg_part:{best_part.model}"
                detection_type = 'fuzzy_manufacturer_part'
            elif best_part.oem_mfg and any(term in best_part.oem_mfg.upper() for term in terms):
                search_strategy = f"manufacturer:{best_part.oem_mfg}"
                detection_type = 'fuzzy_manufacturer'
            else:
                search_strategy = f"company_part:{best_part.part_number}"
                detection_type = 'fuzzy_general'

            return {
                'search_strategy': search_strategy,
                'detection_type': detection_type,
                'confidence': best_score,
                'fuzzy_results': fuzzy_results[:3],  # Top 3 matches
                'detected_categories': {
                    'fuzzy_matches': [f"{p.part_number} ({s})" for p, s in fuzzy_results[:3]]
                }
            }

        # STEP 3: No fuzzy matches either - fall back to equipment search
        logger.info(f"No fuzzy matches, falling back to equipment search")

        return {
            'search_strategy': f"equipment:{' '.join(terms)}",
            'detection_type': 'equipment_fallback',
            'confidence': 30,
            'detected_categories': {'equipment': terms}
        }

    @with_request_id
    def aaa_fallback_sql_part_search(self, search_kwargs: Dict) -> List:
        """
        SQL fallback when Part.search fails
        Uses raw SQL with TSVECTOR if available
        """
        try:
            from sqlalchemy import text

            search_text = search_kwargs.get('search_text', '')
            part_number = search_kwargs.get('part_number', '')
            oem_mfg = search_kwargs.get('oem_mfg', '')
            model = search_kwargs.get('model', '')
            use_fts = search_kwargs.get('use_fts', False)

            if not any([search_text, part_number, oem_mfg, model]):
                logger.warning("No search criteria for SQL fallback")
                return []

            logger.info(
                f"Trying SQL fallback with: text='{search_text}', part='{part_number}', mfg='{oem_mfg}', model='{model}', use_fts={use_fts}")

            # Build dynamic SQL based on available parameters
            conditions = []
            params = {}

            # Use TSVECTOR if available and requested
            if search_text and use_fts:
                try:
                    conditions.append("search_vector @@ plainto_tsquery('english', :search_term)")
                    params['search_term'] = search_text
                    logger.debug("Using TSVECTOR in SQL fallback")
                except Exception as fts_error:
                    logger.warning(f"TSVECTOR fallback failed: {fts_error}")
                    # Fall back to ILIKE
                    conditions.append("""(
                        LOWER(name) LIKE LOWER(:search_term) OR
                        LOWER(part_number) LIKE LOWER(:search_term) OR  
                        LOWER(oem_mfg) LIKE LOWER(:search_term) OR
                        LOWER(model) LIKE LOWER(:search_term) OR
                        LOWER(notes) LIKE LOWER(:search_term)
                    )""")
                    params['search_term'] = f'%{search_text}%'
            elif search_text:
                conditions.append("""(
                    LOWER(name) LIKE LOWER(:search_term) OR
                    LOWER(part_number) LIKE LOWER(:search_term) OR  
                    LOWER(oem_mfg) LIKE LOWER(:search_term) OR
                    LOWER(model) LIKE LOWER(:search_term) OR
                    LOWER(notes) LIKE LOWER(:search_term)
                )""")
                params['search_term'] = f'%{search_text}%'

            if part_number:
                conditions.append("LOWER(part_number) LIKE LOWER(:part_number)")
                params['part_number'] = f'%{part_number}%'

            if oem_mfg:
                conditions.append("LOWER(oem_mfg) LIKE LOWER(:oem_mfg)")
                params['oem_mfg'] = f'%{oem_mfg}%'

            if model:
                conditions.append("LOWER(model) LIKE LOWER(:model)")
                params['model'] = f'%{model}%'

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            # Use TSVECTOR ranking if available
            if search_text and use_fts and 'search_vector' in conditions[0] if conditions else False:
                sql_query = text(f"""
                    SELECT id, part_number, name, oem_mfg, model, class_flag, ud6, type, notes, documentation,
                           ts_rank(search_vector, plainto_tsquery('english', :search_term)) as rank
                    FROM part 
                    WHERE {where_clause}
                    ORDER BY rank DESC
                    LIMIT :limit
                """)
            else:
                sql_query = text(f"""
                    SELECT id, part_number, name, oem_mfg, model, class_flag, ud6, type, notes, documentation
                    FROM part 
                    WHERE {where_clause}
                    ORDER BY part_number
                    LIMIT :limit
                """)

            params['limit'] = search_kwargs.get('limit', 20)

            result = self.session.execute(sql_query, params)

            parts = []
            for row in result:
                parts.append({
                    'id': row[0],
                    'part_number': row[1] or 'Unknown',
                    'name': row[2] or 'Unknown Part',
                    'oem_mfg': row[3] or 'Unknown',
                    'model': row[4] or 'Unknown',
                    'class_flag': row[5] or '',
                    'ud6': row[6] or '',
                    'type': row[7] or '',
                    'notes': row[8] or '',
                    'documentation': row[9] or '',
                    'relevance_score': row[10] if len(row) > 10 else 0  # TSVECTOR rank if available
                })

            logger.info(f"SQL fallback found {len(parts)} parts")
            return parts

        except Exception as e:
            logger.error(f"SQL fallback also failed: {e}")
            return []

    @with_request_id
    def aaa_enhance_part_result(self, part) -> Dict[str, Any]:
        """Enhance part result with related positions and images using TSVECTOR context."""
        part_data = {
            'id': part.id,
            'part_number': getattr(part, 'part_number', 'Unknown'),
            'name': getattr(part, 'name', 'Unknown'),
            'oem_mfg': getattr(part, 'oem_mfg', 'Unknown'),
            'model': getattr(part, 'model', 'Unknown'),
            'class_flag': getattr(part, 'class_flag', None),
            'notes': getattr(part, 'notes', None),
            'type': 'part',
            'search_vector_available': hasattr(part, 'search_vector')
        }

        try:
            # Try to find associations, but don't fail if there's an error
            positions_data = []
            images_data = []

            # Use a separate try block for association queries
            try:
                from modules.emtacdb.emtacdb_fts import PartsPositionImageAssociation

                part_positions = PartsPositionImageAssociation.search(
                    session=self.session,
                    part_id=part.id
                )

                for assoc in part_positions:
                    # Add position information
                    if assoc.position:
                        pos_data = {
                            'id': assoc.position.id,
                            'area': assoc.position.area.name if assoc.position.area else None,
                            'equipment_group': assoc.position.equipment_group.name if assoc.position.equipment_group else None,
                            'model': assoc.position.model.name if assoc.position.model else None,
                            'location': assoc.position.location.name if assoc.position.location else None,
                        }
                        if pos_data not in positions_data:
                            positions_data.append(pos_data)

                    # Add image information
                    if assoc.image:
                        img_data = {
                            'id': assoc.image.id,
                            'title': getattr(assoc.image, 'title', 'Untitled'),
                            'description': getattr(assoc.image, 'description', ''),
                            'file_path': getattr(assoc.image, 'file_path', ''),
                            'url': f"/serve_image/{assoc.image.id}"
                        }
                        if img_data not in images_data:
                            images_data.append(img_data)
            except Exception as assoc_error:
                logger.debug(f"Could not retrieve associations for part {part.id}: {assoc_error}")

            # Add related information to part data
            part_data.update({
                'positions': positions_data,
                'position_count': len(positions_data),
                'images': images_data,
                'image_count': len(images_data),
                'usage_locations': list(set([pos['area'] for pos in positions_data if pos['area']])),
                'equipment_types': list(
                    set([pos['equipment_group'] for pos in positions_data if pos['equipment_group']]))
            })

        except Exception as e:
            logger.warning(f"Could not enhance part result {part.id}: {e}")
            # Return basic part data even if enhancement fails
            part_data.update({
                'positions': [],
                'position_count': 0,
                'images': [],
                'image_count': 0,
                'usage_locations': [],
                'equipment_types': []
            })

        return part_data

    @with_request_id
    def aaa_detect_part_numbers_and_manufacturers(self, terms: List[str], session, request_id: Optional[str] = None) -> \
    Dict[str, List[str]]:
        """
        FIXED: Detect which terms are company part numbers, manufacturer part numbers,
        manufacturers, or equipment types with proper filtering and TSVECTOR awareness.
        """
        from modules.emtacdb.emtacdb_fts import Part

        company_part_numbers = []
        mfg_part_numbers = []
        manufacturers = []
        equipment = []

        # CRITICAL: Filter out common English words that shouldn't be part numbers
        common_words = {
            'what', 'is', 'the', 'part', 'number', 'for', 'a', 'an', 'and', 'or', 'but',
            'in', 'on', 'at', 'to', 'from', 'with', 'by', 'of', 'as', 'this', 'that',
            'these', 'those', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'can', 'may', 'might', 'must', 'have', 'has', 'had', 'be', 'am', 'are',
            'was', 'were', 'being', 'been', 'get', 'got', 'give', 'gave', 'take',
            'took', 'make', 'made', 'come', 'came', 'go', 'went', 'see', 'saw',
            'know', 'knew', 'think', 'thought', 'say', 'said', 'tell', 'told',
            'find', 'found', 'show', 'need', 'want', 'like', 'use', 'work'
        }

        debug_id(f"Starting detection for {len(terms)} terms: {terms}", request_id)

        for term in terms:
            if len(term) < 2:
                continue

            term_lower = term.lower()
            term_upper = term.upper()

            # SKIP common English words entirely
            if term_lower in common_words:
                debug_id(f"Skipping common word: '{term}'", request_id)
                continue

            # SKIP very short terms (likely noise)
            if len(term) < 3:
                debug_id(f"Skipping short term: '{term}'", request_id)
                continue

            found_category = False

            try:
                # Check if term is a company part number (part.part_number)
                # Use more specific matching to avoid false positives
                company_part_result = session.query(Part.part_number).filter(
                    Part.part_number.ilike(f'{term}%')  # Starts with term (more specific)
                ).first()

                if not company_part_result:
                    # Try exact match
                    company_part_result = session.query(Part.part_number).filter(
                        Part.part_number.ilike(term)
                    ).first()

                if company_part_result:
                    company_part_numbers.append(term)
                    found_category = True
                    debug_id(f"'{term}' detected as company part number", request_id)
                    continue

                # Check if term is a manufacturer part number (part.model)
                # More specific matching for manufacturer parts
                mfg_part_result = session.query(Part.model).filter(
                    Part.model.ilike(f'{term}%')  # Starts with term
                ).first()

                if not mfg_part_result:
                    # Try exact match
                    mfg_part_result = session.query(Part.model).filter(
                        Part.model.ilike(term)
                    ).first()

                if mfg_part_result:
                    # Additional validation: must look like a part number
                    if self.aaa_looks_like_part_number(term):
                        mfg_part_numbers.append(term)
                        found_category = True
                        debug_id(f"'{term}' detected as manufacturer part number", request_id)
                        continue
                    else:
                        debug_id(f"🚫 '{term}' found in model column but doesn't look like part number", request_id)

                # Check if term is a manufacturer name (part.oem_mfg)
                # More specific matching for manufacturers
                manufacturer_result = session.query(Part.oem_mfg).filter(
                    Part.oem_mfg.ilike(f'{term}%')  # Starts with term
                ).first()

                if not manufacturer_result:
                    # Try exact match
                    manufacturer_result = session.query(Part.oem_mfg).filter(
                        Part.oem_mfg.ilike(term)
                    ).first()

                if manufacturer_result:
                    # Additional validation: must look like a manufacturer name
                    if self.aaa_looks_like_manufacturer(term):
                        manufacturers.append(term)
                        found_category = True
                        debug_id(f"'{term}' detected as manufacturer", request_id)
                        continue
                    else:
                        debug_id(f"🚫 '{term}' found in oem_mfg column but doesn't look like manufacturer", request_id)

            except Exception as e:
                error_id(f"Error checking term '{term}': {e}", request_id)
                continue

            # If not found in any part-related column, classify as equipment
            if not found_category:
                # Check against equipment keywords
                equipment_keywords = {
                    'sensor', 'sensors', 'motor', 'motors', 'pump', 'pumps',
                    'valve', 'valves', 'bearing', 'bearings', 'filter', 'filters',
                    'switch', 'switches', 'relay', 'relays', 'belt', 'belts',
                    'seal', 'seals', 'gasket', 'gaskets', 'coupling', 'gear', 'gears',
                    'hydraulic', 'pneumatic', 'electrical', 'mechanical', 'bypass'
                }

                if term_lower in equipment_keywords:
                    equipment.append(term)
                    debug_id(f"'{term}' detected as equipment", request_id)
                else:
                    # Only add if it looks like it could be equipment/part related
                    if self.aaa_could_be_equipment(term):
                        equipment.append(term)
                        debug_id(f" '{term}' classified as potential equipment", request_id)
                    else:
                        debug_id(f"Skipping unlikely term: '{term}'", request_id)

        result = {
            'company_part_numbers': company_part_numbers,
            'mfg_part_numbers': mfg_part_numbers,
            'manufacturers': manufacturers,
            'equipment': equipment
        }

        debug_id(f"Detection complete: {result}", request_id)
        return result

    @with_request_id
    def aaa_enhanced_search_with_dual_part_detection(self, query: str, session) -> str:
        """
        Enhanced search that automatically detects:
        1. Company part numbers (part.part_number column) - leverages TSVECTOR
        2. Manufacturer part numbers (part.model column) - leverages TSVECTOR
        3. Manufacturer names (part.oem_mfg column)
        4. Equipment types (filters, sensors, etc.) - leverages TSVECTOR

        Prioritizes searches based on specificity:
        Company part number > Manufacturer part number > Manufacturer + Equipment > Manufacturer > Equipment
        """
        from modules.search.utils import extract_search_terms

        # Extract terms from query
        terms = [term.strip() for term in query.upper().split() if term.strip()]
        logger.debug(f"Analyzing terms: {terms}")

        # Detect what each term represents
        detected = self.aaa_detect_part_numbers_and_manufacturers(terms, session)

        # Build search strategy based on what was detected
        search_components = []

        # Priority 1: Company part numbers (highest priority - your internal system)
        if detected['company_part_numbers']:
            company_parts = ' '.join(detected['company_part_numbers'])
            search_components.append(f"company_part:{company_parts}")
            logger.info(f"Priority 1: Company part number detected: {company_parts}")

        # Priority 2: Manufacturer part numbers (specific manufacturer models)
        if detected['mfg_part_numbers']:
            mfg_parts = ' '.join(detected['mfg_part_numbers'])
            search_components.append(f"mfg_part:{mfg_parts}")
            logger.info(f"Priority 2: Manufacturer part number detected: {mfg_parts}")

        # Priority 3: Manufacturer + Equipment combination
        if detected['manufacturers'] and detected['equipment']:
            manufacturer = detected['manufacturers'][0]  # Use first manufacturer
            equipment = ' '.join(detected['equipment'])
            search_components.append(f"manufacturer:{manufacturer} equipment:{equipment}")
            logger.info(f"Priority 3: Manufacturer+Equipment: {manufacturer} + {equipment}")

        # Priority 4: Manufacturer only
        elif detected['manufacturers']:
            manufacturer = ' '.join(detected['manufacturers'])
            search_components.append(f"manufacturer:{manufacturer}")
            logger.info(f"Priority 4: Manufacturer-only: {manufacturer}")

        # Priority 5: Equipment only
        elif detected['equipment']:
            equipment = ' '.join(detected['equipment'])
            search_components.append(f"equipment:{equipment}")
            logger.info(f"Priority 5: Equipment-only: {equipment}")

        # If nothing specific detected, use original query
        if not search_components:
            logger.info(f"No specific detection, using original query: {query}")
            return query

        # Return the highest priority search component
        selected_strategy = search_components[0]
        logger.info(f"Selected search strategy: {selected_strategy}")
        return selected_strategy

    @with_request_id
    def aaa_extract_part_candidates(self, text: str) -> List[str]:
        """Extract potential part numbers from text using multiple patterns."""
        import re
        candidates = []

        if not text:
            return candidates

        # Pattern 1: Look for explicit part number requests first
        part_description_patterns = [
            r'(?:i\s+)?need\s+(?:the\s+)?part\s+number\s+for\s+(.+?)(?:\s*$|\s*\?)',
            # "I need part number for BEARING ASSEMBLY"
            r'what\s+(?:is\s+)?(?:the\s+)?part\s+number\s+for\s+(.+?)(?:\s*$|\s*\?)',  # "what is part number for..."
            r'part\s+number\s+for\s+(.+?)(?:\s*$|\s*\?)',  # "part number for BEARING ASSEMBLY"
            r'(?:find|get|show)\s+(?:me\s+)?(?:the\s+)?part\s+number\s+for\s+(.+?)(?:\s*$|\s*\?)',
            # "find part number for..."
        ]

        # Check for part description requests (these should be treated as search text, not part numbers)
        for pattern in part_description_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                description = match.strip()
                # If it's a multi-word description, mark it as search text
                if len(description.split()) > 1:
                    # Return the description as a special marker that indicates it's search text
                    return [f"DESCRIPTION:{description}"]

        # Pattern 2: Look for actual part numbers (alphanumeric codes)
        part_number_patterns = [
            r'\b([A-Z]\d{5,})\b',  # A120404 (letter followed by 5+ digits)
            r'\b(\d{5,})\b',  # 120404 (5+ digits standalone)
            r'\b([A-Z0-9]{2,}[-][A-Z0-9]{2,})\b',  # ABC-123 (alphanumeric with dash)
            r'\b([A-Z0-9]{2,}[\.][A-Z0-9]{2,})\b',  # ABC.123 (alphanumeric with dot)
            r'\b([A-Z]{2,}\d{2,})\b',  # AB123 (letters followed by digits)
            r'\b(\d{2,}[A-Z]{2,})\b',  # 123AB (digits followed by letters)
        ]

        for pattern in part_number_patterns:
            matches = re.findall(pattern, text.upper())
            candidates.extend(matches)

        # Pattern 3: Extract from specific "part X" contexts (only if X looks like a part number)
        specific_part_patterns = [
            r'(?:show|find|get)\s+part\s+([A-Z0-9\-\.]{3,})\b',  # "show part ABC123"
            r'part\s+([A-Z0-9\-\.]{3,})\b',  # "part ABC123"
            r'part\s+number\s+([A-Z0-9\-\.]{3,})\b',  # "part number ABC123"
        ]

        for pattern in specific_part_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # FIXED: Correct the malformed regex patterns
                # Only include if it looks like a part number (not just words)
                if re.match(r'^[A-Z0-9\-\.]{3,}$', match.upper()) and not re.match(r'^[A-Z]+$', match.upper()):
                    candidates.append(match.upper())

        # Pattern 4: Standalone alphanumeric codes (be more selective)
        standalone_patterns = [
            r'\b([A-Z0-9]{6,})\b',  # 6+ character alphanumeric codes
            r'\b([A-Z]\d{4,})\b',  # Letter + 4+ digits
            r'\b(\d{4,}[A-Z]+)\b',  # 4+ digits + letters
        ]

        for pattern in standalone_patterns:
            matches = re.findall(pattern, text.upper())
            for match in matches:
                # Exclude common words that might match these patterns
                excluded_words = {'BEARING', 'ASSEMBLY', 'STAGE', 'FIRST', 'SECOND', 'MOTOR', 'VALVE', 'PUMP'}
                if match not in excluded_words:
                    candidates.append(match)

        # Clean and deduplicate
        cleaned_candidates = []
        for candidate in candidates:
            if candidate.startswith('DESCRIPTION:'):
                # Return description immediately if found
                return [candidate]

            cleaned = candidate.strip().upper()
            # More selective filtering
            if (len(cleaned) >= 3 and
                    cleaned not in cleaned_candidates and
                    not cleaned.isalpha() and  # Exclude pure alphabetic strings
                    cleaned not in {'FOR', 'THE', 'AND', 'WITH', 'FROM'}):  # Exclude common words
                cleaned_candidates.append(cleaned)

        return cleaned_candidates

    @with_request_id
    def aaa_fuzzy_part_search(self, part_candidates: List[str], threshold: int = 70) -> List[Tuple[Any, int]]:
        """
        ENHANCED: Perform fuzzy search for parts using multiple strategies with dual part number support.
        Leverages TSVECTOR when available for better matching.

        Now searches both:
        - Company part numbers (part.part_number)
        - Manufacturer part numbers (part.model)
        - Manufacturer names (part.oem_mfg)

        Returns:
            List of (part_object, confidence_score) tuples
        """
        try:
            from fuzzywuzzy import fuzz, process
        except ImportError:
            logger.warning("fuzzywuzzy not available for fuzzy search")
            return []

        if not part_candidates:
            return []

        # Get all parts from database
        from modules.emtacdb.emtacdb_fts import Part
        all_parts = self.session.query(Part).all()
        if not all_parts:
            return []

        results = []

        logger.debug(f"Starting fuzzy search with {len(part_candidates)} candidates against {len(all_parts)} parts")

        for candidate in part_candidates:
            if not candidate or len(candidate) < 2:
                continue

            candidate_upper = candidate.upper()

            # STRATEGY 1: Company part numbers (part.part_number) - HIGHEST PRIORITY
            company_part_numbers = [p.part_number for p in all_parts if p.part_number]
            if company_part_numbers:
                fuzzy_matches = process.extract(candidate, company_part_numbers, limit=5, scorer=fuzz.ratio)

                for match_text, score in fuzzy_matches:
                    if score >= threshold:
                        part = next((p for p in all_parts if p.part_number == match_text), None)
                        if part:
                            # Boost score for company parts (highest priority)
                            boosted_score = min(100, score + 5)
                            results.append((part, boosted_score))
                            logger.debug(f"Company part fuzzy match: {candidate} → {match_text} ({boosted_score})")

            # STRATEGY 2: Manufacturer part numbers (part.model)
            mfg_part_numbers = [p.model for p in all_parts if p.model]
            if mfg_part_numbers:
                fuzzy_matches = process.extract(candidate, mfg_part_numbers, limit=5, scorer=fuzz.ratio)

                for match_text, score in fuzzy_matches:
                    if score >= threshold:
                        part = next((p for p in all_parts if p.model == match_text), None)
                        if part:
                            results.append((part, score))
                            logger.debug(f"Manufacturer part fuzzy match: {candidate} → {match_text} ({score})")

            # STRATEGY 3: Manufacturer names (part.oem_mfg)
            manufacturers = [p.oem_mfg for p in all_parts if p.oem_mfg]
            if manufacturers:
                # Remove duplicates
                unique_manufacturers = list(set(manufacturers))
                fuzzy_matches = process.extract(candidate, unique_manufacturers, limit=3, scorer=fuzz.ratio)

                for match_text, score in fuzzy_matches:
                    if score >= threshold:
                        # Find all parts from this manufacturer
                        manufacturer_parts = [p for p in all_parts if p.oem_mfg == match_text]
                        for part in manufacturer_parts[:3]:  # Limit to top 3 parts per manufacturer
                            results.append((part, score - 10))  # Slightly lower priority
                            logger.debug(f"Manufacturer fuzzy match: {candidate} → {match_text} ({score - 10})")

            # STRATEGY 4: Partial ratio for embedded matches in company parts
            for part in all_parts:
                if part.part_number:
                    partial_score = fuzz.partial_ratio(candidate_upper, part.part_number.upper())
                    if partial_score >= threshold + 10:  # Higher threshold for partial
                        results.append((part, partial_score - 5))
                        logger.debug(
                            f"Company part partial match: {candidate} → {part.part_number} ({partial_score - 5})")

            # STRATEGY 5: Partial ratio for embedded matches in manufacturer parts
            for part in all_parts:
                if part.model:
                    partial_score = fuzz.partial_ratio(candidate_upper, part.model.upper())
                    if partial_score >= threshold + 10:  # Higher threshold for partial
                        results.append((part, partial_score - 10))
                        logger.debug(
                            f"Manufacturer part partial match: {candidate} → {part.model} ({partial_score - 10})")

            # STRATEGY 6: Token set ratio for flexible matching (company part + name)
            for part in all_parts:
                if part.part_number and part.name:
                    combined_text = f"{part.part_number} {part.name}"
                    token_score = fuzz.token_set_ratio(candidate, combined_text)
                    if token_score >= threshold:
                        results.append((part, token_score - 15))
                        logger.debug(f"Token set match: {candidate} → {combined_text} ({token_score - 15})")

        # Remove duplicates and sort by confidence
        unique_results = {}
        for part, score in results:
            part_id = part.id
            if part_id not in unique_results or score > unique_results[part_id][1]:
                unique_results[part_id] = (part, score)

        # Sort by score (highest first)
        sorted_results = sorted(unique_results.values(), key=lambda x: x[1], reverse=True)

        # Log top results
        logger.info(f"Fuzzy search found {len(sorted_results)} unique matches")
        for i, (part, score) in enumerate(sorted_results[:3]):
            logger.info(f"  {i + 1}. {part.part_number} | {part.oem_mfg} {part.model} ({score})")

        return sorted_results

    @with_request_id
    def aaa_enhance_comprehensive_part_search_parameters(self, analysis_params, original_params):
        """
        Convert analysis parameters to proper Part.search parameters with TSVECTOR support.
        Add this method to your AggregateSearch class.
        """
        # Start with the analysis parameters
        part_search_params = {}

        # Direct field mappings
        field_mappings = {
            'name': 'name',
            'part_number': 'part_number',
            'oem_mfg': 'oem_mfg',
            'model': 'model',
            'search_text': 'search_text',
            'fields': 'fields'
        }

        for analysis_key, part_key in field_mappings.items():
            if analysis_key in analysis_params:
                part_search_params[part_key] = analysis_params[analysis_key]
                logger.debug(f"Mapped {analysis_key} -> {part_key}: {analysis_params[analysis_key]}")

        # Handle extracted_id as part_id
        if 'extracted_id' in analysis_params:
            part_search_params['part_id'] = analysis_params['extracted_id']
            logger.debug(f"Mapped extracted_id -> part_id: {analysis_params['extracted_id']}")

        # Set defaults with TSVECTOR support
        part_search_params['limit'] = original_params.get('limit', 20)
        part_search_params['exact_match'] = False
        part_search_params['use_fts'] = True  # Enable TSVECTOR by default

        return part_search_params

    @with_request_id
    def aaa_create_part_result_from_dict(self, part: dict, index: int, search_method: str, confidence: int) -> Dict[
        str, Any]:
        """Create standardized part result from dictionary data."""
        return {
            'id': part.get('id'),
            'part_number': part.get('part_number', f'Dict-{index + 1}'),
            'name': part.get('name', 'Unknown Part'),
            'oem_mfg': part.get('oem_mfg', 'Unknown'),
            'model': part.get('model', 'Unknown'),
            'class_flag': part.get('class_flag', ''),
            'ud6': part.get('ud6', ''),
            'type': part.get('type', ''),
            'notes': part.get('notes', ''),
            'documentation': part.get('documentation', ''),
            'entity_type': 'part',
            'search_method': search_method,
            'confidence': confidence,
            'positions': [],
            'position_count': 0,
            'images': [],
            'image_count': 0,
            'usage_locations': [],
            'equipment_types': [],
            'relevance_score': part.get('relevance_score', 0)  # TSVECTOR ranking if available
        }

    @with_request_id
    def aaa_create_part_result_from_object(self, part: Any, index: int, search_method: str) -> Dict[str, Any]:
        """Create standardized part result from object data."""
        return {
            'id': getattr(part, 'id', None) or index + 1,
            'part_number': str(part) if part else f'Object-{index + 1}',
            'name': getattr(part, 'name', 'Unknown Part'),
            'oem_mfg': getattr(part, 'oem_mfg', 'Unknown'),
            'model': getattr(part, 'model', 'Unknown'),
            'entity_type': 'part',
            'search_method': search_method + "_object",
            'confidence': 30,
            'positions': [],
            'position_count': 0,
            'images': [],
            'image_count': 0,
            'usage_locations': [],
            'equipment_types': []
        }

    @with_request_id
    def aaa_create_error_part_result(self, index: int, search_method: str) -> Dict[str, Any]:
        """Create error part result for handling formatting failures."""
        return {
            'id': index + 1,
            'part_number': f'Error-{index + 1}',
            'name': 'Formatting Error',
            'entity_type': 'part',
            'search_method': search_method + "_error",
            'confidence': 0,
            'positions': [],
            'position_count': 0,
            'images': [],
            'image_count': 0,
            'usage_locations': [],
            'equipment_types': []
        }

    @with_request_id
    def aaa_calculate_enhancement_stats(self, comprehensive_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate enhancement statistics for comprehensive results."""
        return {
            'total_positions': sum(r.get('position_count', 0) for r in comprehensive_results),
            'total_images': sum(r.get('image_count', 0) for r in comprehensive_results),
            'unique_locations': len(
                set(loc for r in comprehensive_results for loc in r.get('usage_locations', []))),
            'unique_equipment': len(
                set(eq for r in comprehensive_results for eq in r.get('equipment_types', []))),
            'average_relevance': sum(r.get('relevance_score', 0) for r in comprehensive_results) / len(
                comprehensive_results) if comprehensive_results else 0,
            'tsvector_enhanced': sum(1 for r in comprehensive_results if r.get('relevance_score', 0) > 0),
            'total_enhanced_results': len(comprehensive_results)
        }

    @with_request_id
    def aaa_track_search_analytics(self, params: Dict[str, Any], comprehensive_results: List[Dict[str, Any]],
                                   confidence: int, search_method: str, intent_name: str = "FIND_PART") -> Optional[
        str]:
        """Track search analytics with TSVECTOR performance metrics in both tables."""
        try:
            from modules.search.query_tracker import SearchQueryTracker

            # Get query details
            query_text = params.get("search_text", params.get("raw_input", "unknown"))
            result_count = len(comprehensive_results)
            session_id = params.get("session_id")
            user_id = params.get("user_id")

            # Calculate enhanced metrics
            tsvector_used = any(r.get('relevance_score', 0) > 0 for r in comprehensive_results)
            average_relevance = sum(r.get('relevance_score', 0) for r in comprehensive_results) / len(
                comprehensive_results) if comprehensive_results else 0
            was_successful = result_count > 0

            # 1. Insert into search_query table (via tracker)
            tracker = SearchQueryTracker(session=self.session)
            query_id = tracker.track_search_query(
                session_id=session_id,
                query_text=query_text,
                detected_intent_id=self.get_intent_id(intent_name),  # Convert to integer ID
                intent_confidence=confidence / 100.0 if confidence else 0.8,
                extracted_entities={
                    "part_numbers": [r.get("part_number", "") for r in comprehensive_results if r.get("part_number")],
                    "tsvector_used": tsvector_used,
                    "average_relevance": average_relevance
                },
                search_method=search_method,
                result_count=result_count,
                execution_time_ms=0  # Add timing if available
            )

            # 2. Insert into search_analytics table (detailed metrics)
            analytics = SearchAnalytics(
                user_id=user_id,
                session_id=str(session_id) if session_id else None,
                query_text=query_text,
                detected_intent=intent_name,  # String name for analytics
                intent_confidence=confidence / 100.0 if confidence else 0.8,
                search_method=search_method,
                execution_time_ms=0,  # Add timing if available
                result_count=result_count,
                success=was_successful,
                created_at=datetime.utcnow()
            )

            self.session.add(analytics)
            self.session.commit()

            logger.debug(
                f"[tracking] SearchQuery logged with ID: {query_id}, Analytics ID: {analytics.id}, Intent: {intent_name}")
            return query_id

        except Exception as e:
            logger.warning(f"[tracking] Failed to log SearchQuery: {e}")
            try:
                self.session.rollback()
            except:
                pass
            return None

    def get_intent_id(self, intent_name: str) -> Optional[int]:
        """Get intent ID for any intent name from database."""
        try:
            intent = self.session.query(SearchIntent).filter(
                SearchIntent.name == intent_name,
                SearchIntent.is_active == True
            ).first()
            return intent.id if intent else None
        except Exception as e:
            logger.error(f"Error getting intent ID for {intent_name}: {e}")
            return None

    def aaa_could_be_equipment(cls, term: str) -> bool:
        """Check if term could reasonably be equipment-related."""
        # Equipment terms usually:
        # - Are at least 4 characters
        # - Don't contain weird punctuation
        # - Might have numbers/dashes for specifications

        if len(term) < 4:
            return False

        # Contains voltage patterns (110V, 120V, etc.)
        if re.search(r'\d+[-]?\d*V', term):
            return True

        # Contains size patterns (1-1/2", 3/4", etc.)
        if re.search(r'\d+[-/]\d+', term):
            return True

        # Reasonable length without too much punctuation
        punctuation_count = sum(1 for c in term if c in '!"#$%&\'()*+,.:;<=>?@[\\]^`{|}~')
        if punctuation_count <= 2:  # Allow some punctuation for specs
            return True

        return False
    # ============================================================================
    # AAA_PART SEARCH INTEGRATION METHODS
    # ============================================================================
    @with_request_id
    def aaa_execute_part_search_with_tsvector(self, question: str, user_id: str = None, request_id: str = None) -> Dict[
        str, Any]:
        """
        Main integration method for AistManager to call part search with full TSVECTOR support.

        This method provides the interface between AistManager and the comprehensive part search system.
        """
        logger.info(f"Executing TSVECTOR-enhanced part search for: '{question}'")

        try:
            # Build comprehensive search parameters
            search_params = {
                'search_text': question,
                'user_id': user_id,
                'request_id': request_id,
                'limit': 20,
                'raw_input': question,
                'extraction_method': 'tsvector_enhanced',
                'use_tsvector': True
            }

            # Execute the comprehensive part search (which uses TSVECTOR)
            result = self.aaa_comprehensive_part_search(search_params)

            # Add TSVECTOR-specific metadata
            if result.get('status') == 'success':
                result['tsvector_features'] = {
                    'fts_enabled': True,
                    'ranking_applied': True,
                    'fallback_available': True,
                    'performance_optimized': True
                }

                logger.info(f"TSVECTOR part search completed: {result.get('count', 0)} results found")

            return result

        except Exception as e:
            logger.error(f"TSVECTOR part search failed: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': f"TSVECTOR part search failed: {str(e)}",
                'entity_type': 'part',
                'tsvector_features': {
                    'fts_enabled': False,
                    'error_occurred': True
                }
            }

    @with_request_id
    def aaa_get_part_search_capabilities(self) -> Dict[str, Any]:
        """
        Get comprehensive information about part search capabilities including TSVECTOR support.
        """
        return {
            "part_search_features": {
                "tsvector_support": True,
                "postgresql_fts": True,
                "gin_indexing": True,
                "relevance_ranking": True,
                "fuzzy_fallback": True,
                "dual_part_detection": True,
                "manufacturer_detection": True,
                "equipment_categorization": True,
                "sql_fallback": True,
                "relationship_enhancement": True,
                "analytics_tracking": True
            },
            "search_strategies": [
                "company_part_number",
                "manufacturer_part_number",
                "manufacturer_plus_equipment",
                "manufacturer_only",
                "equipment_only",
                "fuzzy_matching",
                "sql_fallback"
            ],
            "tsvector_capabilities": {
                "indexed_fields": ["part_number", "name", "oem_mfg", "model", "notes", "documentation"],
                "language_processing": "english",
                "stemming_enabled": True,
                "stop_word_removal": True,
                "phrase_matching": True,
                "ranking_algorithm": "ts_rank"
            },
            "performance_features": {
                "gin_index_usage": True,
                "query_optimization": True,
                "result_caching": True,
                "fallback_strategies": ["fuzzy_search", "sql_ilike", "basic_matching"]
            }
        }
#=======================================================================================================
    # ======== Legacy Keyword Management Methods (for backward compatibility) ========

    def register_keyword(self, keyword: str, action_type: str, search_pattern: str = None,
                         entity_type: str = None, description: str = None) -> Dict[str, Any]:
        """
        Legacy method for registering keywords.

        Note: This is kept for backward compatibility but new implementations
        should use the pattern management system via SearchPatternManager.
        """
        if not MODELS_AVAILABLE:
            return {"status": "error", "message": "Models not available"}

        try:
            existing = self.session.query(KeywordAction).filter_by(keyword=keyword).first()

            action_data = {
                "type": action_type,
                "search_pattern": search_pattern,
                "entity_type": entity_type,
                "description": description,
                "created_at": datetime.utcnow().isoformat()
            }

            if existing:
                existing.action = json.dumps(action_data)
                self.session.commit()
                logger.info(f"Updated existing keyword: {keyword}")
                return {"status": "updated", "keyword": keyword}

            new_keyword = KeywordAction(
                keyword=keyword,
                action=json.dumps(action_data)
            )

            self.session.add(new_keyword)
            self.session.commit()
            logger.info(f"Registered new keyword: {keyword}")
            return {"status": "created", "keyword": keyword}

        except Exception as e:
            if self.session:
                self.session.rollback()
            logger.error(f"Error registering keyword '{keyword}': {e}")
            return {"status": "error", "message": str(e)}

    def get_all_keywords(self) -> List[Dict[str, Any]]:
        """
        Legacy method for getting all keywords.
        """
        if not MODELS_AVAILABLE:
            return []

        try:
            keywords = self.session.query(KeywordAction).all()
            result = []

            for kw in keywords:
                try:
                    action_data = json.loads(kw.action)
                    result.append({
                        "id": kw.id,
                        "keyword": kw.keyword,
                        "action_type": action_data.get("type"),
                        "search_pattern": action_data.get("search_pattern"),
                        "entity_type": action_data.get("entity_type"),
                        "description": action_data.get("description")
                    })
                except json.JSONDecodeError:
                    # Handle legacy action format
                    result.append({
                        "id": kw.id,
                        "keyword": kw.keyword,
                        "action": kw.action
                    })

            return result

        except Exception as e:
            logger.error(f"Error retrieving keywords: {e}")
            return []

    # ======== Pattern Matching Methods ========

    def match_pattern(self, pattern: str, text: str) -> Optional[Dict[str, str]]:
        """
        Match a pattern against text and extract parameters.

        Args:
            pattern: The pattern with {param} placeholders
            text: The text to match against

        Returns:
            Dictionary of extracted parameters or None if no match
        """
        if not pattern:
            return None

        # Convert pattern like "show {equipment} in {area}" to regex
        pattern_regex = pattern.replace("{", "(?P<").replace("}", ">.*?)")
        match = re.match(pattern_regex, text, re.IGNORECASE)

        if match:
            params = match.groupdict()
            # Clean up parameters - remove extra spaces and make lowercase for matching
            return {k: v.strip().lower() if v else v for k, v in params.items()}

        return None

    def extract_search_parameters(self, user_input: str, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract search parameters from user input based on action data.

        Args:
            user_input: User's input text
            action_data: Action data containing search pattern

        Returns:
            Dictionary of extracted parameters and search metadata
        """
        search_pattern = action_data.get("search_pattern")
        entity_type = action_data.get("entity_type")

        params = {}

        # Try to match pattern if available
        if search_pattern:
            matched_params = self.match_pattern(search_pattern, user_input)
            if matched_params:
                params.update(matched_params)

        # Add basic search information
        params.update({
            "entity_type": entity_type,
            "action_type": action_data.get("type"),
            "raw_input": user_input,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Extract any ID numbers that might be in the input
        id_matches = re.findall(r'\b(id|number|#)[\s:]*(\d+)\b', user_input, re.IGNORECASE)
        if id_matches:
            for match_type, match_id in id_matches:
                params["extracted_id"] = int(match_id)

        return params

    # ======== Legacy Search Execution Methods ========

    def execute_aggregated_search(self, user_input: str) -> Dict[str, Any]:
        """
        Legacy entry point for executing a comprehensive aggregated search.

        Note: This method is kept for backward compatibility. New implementations
        should use the NLP-enhanced search via SpaCyEnhancedAggregateSearch.

        Args:
            user_input: User's input text

        Returns:
            Dictionary with aggregated search results and metadata
        """
        if not MODELS_AVAILABLE:
            return {
                "status": "error",
                "message": "Database models not available",
                "input": user_input
            }

        # Check cache first
        cache_key = user_input.lower().strip()
        if cache_key in self._search_cache:
            cached_result = self._search_cache[cache_key]
            cached_result["from_cache"] = True
            return cached_result

        try:
            # Find matching keyword using legacy system
            keyword, action, _ = KeywordAction.find_best_match(user_input, self.session)

            if not keyword:
                return {
                    "status": "error",
                    "message": "No matching keyword found",
                    "input": user_input,
                    "suggestion": "Try using the NLP-enhanced search for better understanding"
                }

            # Parse action data
            try:
                action_data = json.loads(action)
            except (json.JSONDecodeError, TypeError):
                # Fallback for legacy format
                action_data = {"type": action}

            action_type = action_data.get("type")

            # Extract search parameters
            params = self.extract_search_parameters(user_input, action_data)

            # Dispatch to appropriate comprehensive search handler
            result = self._dispatch_search_method(action_type, params)

            if result:
                # Add metadata to result
                result.update({
                    "keyword": keyword,
                    "action_type": action_type,
                    "parameters": params,
                    "timestamp": datetime.utcnow().isoformat(),
                    "method": "legacy_keyword_search"
                })

                # Cache result
                self._add_to_cache(cache_key, result)

            return result or {
                "status": "error",
                "message": f"Unknown action type: {action_type}",
                "input": user_input
            }

        except Exception as e:
            logger.error(f"Error executing legacy search: {e}")
            return {
                "status": "error",
                "message": f"Error executing search: {str(e)}",
                "input": user_input
            }

    def _dispatch_search_method(self, action_type: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Dispatch search to appropriate method based on action type."""
        method_map = {
            "image_search": self.comprehensive_image_search,
            "image_similarity": self.comprehensive_search_by_similarity,
            "part_search": self.comprehensive_part_search,
            "drawing_search": self.comprehensive_drawing_search,
            "tool_search": self.comprehensive_tool_search,
            "position_search": self.comprehensive_position_search,
            "problem_search": self.comprehensive_problem_search,
            "task_search": self.comprehensive_task_search
        }

        search_method = method_map.get(action_type)
        if search_method:
            return search_method(params)
        return None

    def _add_to_cache(self, key: str, result: Dict[str, Any]) -> None:
        """Add a result to the search cache with LRU behavior."""
        # Implement simple LRU cache
        if len(self._search_cache) >= self._cache_limit:
            # Remove oldest entry
            oldest_key = next(iter(self._search_cache))
            del self._search_cache[oldest_key]

        # Add to cache
        self._search_cache[key] = result

    # ======== Comprehensive Aggregated Search Methods ========

    def comprehensive_image_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive aggregated image search using the powerful Image.search_images method.
        Aggregates images with comprehensive context from multiple association paths.

        Args:
            params: Dictionary of search parameters

        Returns:
            Dictionary with comprehensive aggregated search results
        """
        if not MODELS_AVAILABLE:
            return {
                "status": "error",
                "message": "Database models not available for image search"
            }

        try:
            # Map our parameters to Image.search_images parameters
            search_kwargs = {}

            # Direct image parameters
            if 'title' in params:
                search_kwargs['title'] = params['title']
            if 'description' in params:
                search_kwargs['description'] = params['description']

            # Hierarchy parameters from pattern matching
            if 'area' in params:
                area = self.session.query(Area).filter(Area.name.ilike(f"%{params['area']}%")).first()
                if area:
                    search_kwargs['area_id'] = area.id

            if 'equipment' in params or 'equipment_group' in params:
                equipment_name = params.get('equipment') or params.get('equipment_group')
                equipment = self.session.query(EquipmentGroup).filter(
                    EquipmentGroup.name.ilike(f"%{equipment_name}%")).first()
                if equipment:
                    search_kwargs['equipment_group_id'] = equipment.id

            if 'model' in params:
                model = self.session.query(Model).filter(Model.name.ilike(f"%{params['model']}%")).first()
                if model:
                    search_kwargs['model_id'] = model.id

            if 'location' in params:
                location = self.session.query(Location).filter(Location.name.ilike(f"%{params['location']}%")).first()
                if location:
                    search_kwargs['location_id'] = location.id

            # Association parameters
            for param in ['tool_id', 'task_id', 'problem_id', 'completed_document_id', 'position_id']:
                if param in params:
                    search_kwargs[param] = params[param]

            # Set limit
            search_kwargs['limit'] = int(params.get('limit', 10))

            # Handle specific ID search if provided
            if 'extracted_id' in params:
                specific_image = self.session.query(Image).filter(Image.id == params['extracted_id']).first()
                if specific_image:
                    search_kwargs['image_id'] = params['extracted_id']

            # Execute the comprehensive search using Image.search_images
            comprehensive_results = Image.search_images(
                session=self.session,
                **search_kwargs
            )

            # Enhance results with additional context
            enhanced_results = []
            for result in comprehensive_results:
                enhanced_result = self._enhance_image_result(result)
                enhanced_results.append(enhanced_result)

            return {
                'status': 'success',
                'count': len(enhanced_results),
                'results': enhanced_results,
                'entity_type': 'image',
                'search_type': 'comprehensive',
                'search_parameters': search_kwargs,
                'comprehensive_features': {
                    'position_hierarchy_search': any(
                        key in search_kwargs for key in ['area_id', 'equipment_group_id', 'model_id', 'location_id']),
                    'association_search': any(
                        key in search_kwargs for key in ['tool_id', 'task_id', 'problem_id', 'completed_document_id']),
                    'multi_path_search': True
                }
            }

        except Exception as e:
            logger.error(f"Error in comprehensive image search: {e}")
            return {
                'status': 'error',
                'message': f"Error in comprehensive image search: {str(e)}",
                'entity_type': 'image',
                'search_type': 'comprehensive'
            }

        #====== Part Search======

    #====================Part Search===========================

    def comprehensive_part_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        ULTIMATE: Complete part search combining all best practices

        Features:
        - Enhanced dual part number detection (company vs manufacturer)
        - Fuzzy search fallback for typos/variations
        - Robust SQL fallback for database errors
        - Comprehensive result formatting with relationships
        - Parameter enhancement and validation
        - Logs successful queries for analytics
        """
        logger.debug(f"Starting ultimate part search with params: {params}")

        if not MODELS_AVAILABLE:
            return {
                "status": "error",
                "message": "Database models not available for part search"
            }

        try:
            if self.session:
                try:
                    self.session.rollback()
                    logger.debug("Session rolled back for clean state")
                except Exception:
                    pass

            from modules.emtacdb.emtacdb_fts import Part
            from modules.search.utils import extract_search_terms

            search_kwargs = {}
            search_method = "unknown"
            detection_results = {}
            fuzzy_results = []
            confidence = 0

            if 'search_text' in params and params['search_text']:
                original_search_text = params['search_text']
                logger.info(f"ENHANCED DUAL SEARCH for: '{original_search_text}'")

                search_result = self.enhanced_search_with_fuzzy_fallback(
                    original_search_text, self.session
                )

                search_strategy = search_result['search_strategy']
                detection_type = search_result['detection_type']
                confidence = search_result['confidence']
                detection_results = search_result.get('detected_categories', {})
                fuzzy_results = search_result.get('fuzzy_results', [])

                logger.info(f"Strategy: {search_strategy} ({detection_type}, confidence: {confidence})")

                if search_strategy.startswith('company_part:'):
                    part_number = search_strategy.replace('company_part:', '').strip()
                    search_kwargs = {
                        'part_number': part_number,
                        'exact_match': True,
                        'limit': int(params.get('limit', 20))
                    }
                    search_method = "company_part_number"

                elif search_strategy.startswith('mfg_part:'):
                    model = search_strategy.replace('mfg_part:', '').strip()
                    search_kwargs = {
                        'model': model,
                        'exact_match': False,
                        'limit': int(params.get('limit', 20))
                    }
                    search_method = "manufacturer_part_number"

                elif search_strategy.startswith('manufacturer:') and 'equipment:' in search_strategy:
                    parts = search_strategy.split(' equipment:')
                    manufacturer = parts[0].replace('manufacturer:', '').strip()
                    equipment = parts[1].strip()

                    search_kwargs = {
                        'oem_mfg': manufacturer,
                        'search_text': equipment,
                        'fields': ['name', 'notes', 'documentation', 'class_flag', 'type'],
                        'exact_match': False,
                        'limit': int(params.get('limit', 20))
                    }
                    search_method = "manufacturer_plus_equipment"

                elif search_strategy.startswith('manufacturer:'):
                    manufacturer = search_strategy.replace('manufacturer:', '').strip()
                    search_kwargs = {
                        'oem_mfg': manufacturer,
                        'exact_match': False,
                        'limit': int(params.get('limit', 20))
                    }
                    search_method = "manufacturer_only"

                elif search_strategy.startswith('equipment:'):
                    equipment = search_strategy.replace('equipment:', '').strip()
                    search_kwargs = {
                        'search_text': equipment,
                        'fields': ['name', 'notes', 'documentation', 'class_flag', 'type'],
                        'exact_match': False,
                        'limit': int(params.get('limit', 20))
                    }
                    search_method = "equipment_only"

                else:
                    enhanced_params = self.enhance_comprehensive_part_search_parameters(params, params)
                    search_kwargs = enhanced_params
                    search_method = "enhanced_general_search"

            elif 'part_number' in params:
                logger.info(f" DIRECT PART SEARCH for part number: '{params['part_number']}'")
                search_kwargs = {
                    'part_number': params['part_number'],
                    'exact_match': True,
                    'limit': int(params.get('limit', 20))
                }
                search_method = "direct_part_number_search"
                confidence = 95

            elif 'extracted_id' in params:
                logger.info(f" DIRECT PART SEARCH for ID: {params['extracted_id']}")
                search_kwargs = {
                    'part_id': params['extracted_id'],
                    'limit': 1
                }
                search_method = "id_search"
                confidence = 100

            else:
                logger.warning("No valid search parameters for part search")
                return {
                    'status': 'error',
                    'message': 'No valid search parameters (need search_text, part_number, or extracted_id)',
                    'params_received': params
                }

            logger.debug(f"Executing Part.search with kwargs: {search_kwargs}")

            parts = []
            try:
                search_kwargs['session'] = self.session
                if params.get('request_id'):
                    search_kwargs['request_id'] = params['request_id']

                parts = Part.search(**search_kwargs)
                logger.info(f"Part.search found {len(parts)} parts using {search_method}")

            except Exception as search_error:
                logger.warning(f"Part.search failed: {search_error}")
                parts = self._fallback_sql_part_search(search_kwargs)
                search_method += "_sql_fallback"

            comprehensive_results = []

            for i, part in enumerate(parts):
                try:
                    if hasattr(part, 'part_number'):
                        part_result = self._enhance_part_result(part)
                        part_result.update({
                            'search_method': search_method,
                            'confidence': confidence,
                            'detection_type': detection_results.get('detection_type', search_method),
                            'entity_type': 'part'
                        })
                    elif isinstance(part, dict):
                        part_result = {
                            'id': part.get('id'),
                            'part_number': part.get('part_number', f'Dict-{i + 1}'),
                            'name': part.get('name', 'Unknown Part'),
                            'oem_mfg': part.get('oem_mfg', 'Unknown'),
                            'model': part.get('model', 'Unknown'),
                            'class_flag': part.get('class_flag', ''),
                            'ud6': part.get('ud6', ''),
                            'type': part.get('type', ''),
                            'notes': part.get('notes', ''),
                            'documentation': part.get('documentation', ''),
                            'entity_type': 'part',
                            'search_method': search_method,
                            'confidence': confidence,
                            'positions': [],
                            'position_count': 0,
                            'images': [],
                            'image_count': 0,
                            'usage_locations': [],
                            'equipment_types': []
                        }
                    else:
                        part_result = {
                            'id': getattr(part, 'id', None) or i + 1,
                            'part_number': str(part) if part else f'Object-{i + 1}',
                            'name': 'Unknown Part',
                            'entity_type': 'part',
                            'search_method': search_method + "_object",
                            'confidence': 30,
                            'positions': [],
                            'position_count': 0,
                            'images': [],
                            'image_count': 0,
                            'usage_locations': [],
                            'equipment_types': []
                        }

                    comprehensive_results.append(part_result)
                    logger.debug(f" Enhanced part {i + 1}: {part_result['part_number']}")

                except Exception as format_error:
                    logger.warning(f"Failed to format part {i + 1}: {format_error}")
                    comprehensive_results.append({
                        'id': i + 1,
                        'part_number': f'Error-{i + 1}',
                        'name': 'Formatting Error',
                        'entity_type': 'part',
                        'search_method': search_method + "_error",
                        'confidence': 0,
                        'positions': [],
                        'position_count': 0,
                        'images': [],
                        'image_count': 0,
                        'usage_locations': [],
                        'equipment_types': []
                    })

            result = {
                'status': 'success',
                'count': len(comprehensive_results),
                'results': comprehensive_results,
                'entity_type': 'part',
                'search_type': 'ultimate_comprehensive_part_search',
                'search_method': search_method,
                'confidence': confidence,
                'search_parameters': search_kwargs,
                'detection_results': detection_results,
                'fuzzy_results': fuzzy_results,
                'enhancement_stats': {
                    'total_positions': sum(r.get('position_count', 0) for r in comprehensive_results),
                    'total_images': sum(r.get('image_count', 0) for r in comprehensive_results),
                    'unique_locations': len(
                        set(loc for r in comprehensive_results for loc in r.get('usage_locations', []))),
                    'unique_equipment': len(
                        set(eq for r in comprehensive_results for eq in r.get('equipment_types', [])))
                },
                'note': f'Ultimate comprehensive search using {search_method} with relationship enhancement'
            }

            if params.get("user_id"):
                try:
                    from modules.search.query_tracker import SearchQueryTracker
                    tracker = SearchQueryTracker(session=self.session)
                    query_id = tracker.track_search_query(
                        user_id=params["user_id"],
                        query_text=params.get("search_text", params.get("raw_input", "unknown")),
                        intent_name="FIND_PART",
                        detected_entities={"part_numbers": [r["part_number"] for r in comprehensive_results]},
                        result_count=len(comprehensive_results),
                        metadata={
                            "confidence": confidence,
                            "search_method": search_method,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                    result["query_id"] = query_id
                    logger.debug(f"[tracking] SearchQuery logged with ID: {query_id}")
                except Exception as e:
                    logger.warning(f"[tracking] Failed to log SearchQuery: {e}")

            logger.info(f" Ultimate part search SUCCESS: {len(comprehensive_results)} enhanced parts found")
            return result

        except Exception as e:
            logger.error(f"Ultimate part search ERROR: {e}", exc_info=True)
            if self.session:
                try:
                    self.session.rollback()
                except Exception:
                    pass
            return {
                'status': 'error',
                'message': f"Ultimate part search failed: {str(e)}",
                'entity_type': 'part',
                'search_parameters': params,
                'error_type': 'ultimate_part_search_error'
            }

    def enhanced_search_with_fuzzy_fallback(self, query: str, session) -> Dict[str, Any]:
        """
        Enhanced search with intelligent dual part detection + fuzzy fallback

        Flow:
        1. Try exact enhanced detection
        2. If no results, try fuzzy matching
        3. Return best results with confidence scores
        """
        from modules.search.utils import extract_search_terms

        # Extract terms from query
        terms = [term.strip() for term in query.upper().split() if term.strip()]
        logger.debug(f"Analyzing terms: {terms}")

        # STEP 1: Try exact enhanced detection first
        detected = self.detect_part_numbers_and_manufacturers(terms, session)

        # Check if we found exact matches
        exact_matches_found = any([
            detected['company_part_numbers'],
            detected['mfg_part_numbers'],
            detected['manufacturers']
        ])

        if exact_matches_found:
            # Use exact detection
            enhanced_query = self.enhanced_search_with_dual_part_detection(query, session)
            return {
                'search_strategy': enhanced_query,
                'detection_type': 'exact_database_match',
                'confidence': 95,
                'detected_categories': detected
            }

        # STEP 2: No exact matches - try fuzzy search
        logger.info(f"No exact matches found, trying fuzzy search for: {query}")

        # Extract potential part candidates from the query
        part_candidates = self._extract_part_candidates(query)

        # Add all terms as potential part numbers if no specific candidates found
        if not part_candidates:
            part_candidates.extend(terms)

            # Add combinations for multi-word part numbers
            if len(terms) >= 2:
                part_candidates.append(''.join(terms))  # "ABC 123" → "ABC123"
                part_candidates.append('-'.join(terms))  # "ABC 123" → "ABC-123"
                part_candidates.append('_'.join(terms))  # "ABC 123" → "ABC_123"

        logger.debug(f"Fuzzy search candidates: {part_candidates}")

        # Perform fuzzy search
        fuzzy_results = self._fuzzy_part_search(part_candidates, threshold=70)

        if fuzzy_results:
            # Get best fuzzy match
            best_part, best_score = fuzzy_results[0]

            logger.info(f"Best fuzzy match: {best_part.part_number} (score: {best_score})")

            # Determine what type of match this was
            if best_score >= 90:
                search_strategy = f"company_part:{best_part.part_number}"  # High confidence
                detection_type = 'fuzzy_exact'
            elif best_part.model and any(term in best_part.model.upper() for term in terms):
                search_strategy = f"mfg_part:{best_part.model}"
                detection_type = 'fuzzy_manufacturer_part'
            elif best_part.oem_mfg and any(term in best_part.oem_mfg.upper() for term in terms):
                search_strategy = f"manufacturer:{best_part.oem_mfg}"
                detection_type = 'fuzzy_manufacturer'
            else:
                search_strategy = f"company_part:{best_part.part_number}"
                detection_type = 'fuzzy_general'

            return {
                'search_strategy': search_strategy,
                'detection_type': detection_type,
                'confidence': best_score,
                'fuzzy_results': fuzzy_results[:3],  # Top 3 matches
                'detected_categories': {
                    'fuzzy_matches': [f"{p.part_number} ({s})" for p, s in fuzzy_results[:3]]
                }
            }

        # STEP 3: No fuzzy matches either - fall back to equipment search
        logger.info(f"No fuzzy matches, falling back to equipment search")

        return {
            'search_strategy': f"equipment:{' '.join(terms)}",
            'detection_type': 'equipment_fallback',
            'confidence': 30,
            'detected_categories': {'equipment': terms}
        }

    def _fallback_sql_part_search(self, search_kwargs: Dict) -> List:
        """
        SQL fallback when Part.search fails
        """
        try:
            from sqlalchemy import text

            search_text = search_kwargs.get('search_text', '')
            part_number = search_kwargs.get('part_number', '')
            oem_mfg = search_kwargs.get('oem_mfg', '')
            model = search_kwargs.get('model', '')

            if not any([search_text, part_number, oem_mfg, model]):
                logger.warning("No search criteria for SQL fallback")
                return []

            logger.info(
                f"Trying SQL fallback with: text='{search_text}', part='{part_number}', mfg='{oem_mfg}', model='{model}'")

            # Build dynamic SQL based on available parameters
            conditions = []
            params = {}

            if search_text:
                conditions.append("""(
                    LOWER(name) LIKE LOWER(:search_term) OR
                    LOWER(part_number) LIKE LOWER(:search_term) OR  
                    LOWER(oem_mfg) LIKE LOWER(:search_term) OR
                    LOWER(model) LIKE LOWER(:search_term) OR
                    LOWER(notes) LIKE LOWER(:search_term)
                )""")
                params['search_term'] = f'%{search_text}%'

            if part_number:
                conditions.append("LOWER(part_number) LIKE LOWER(:part_number)")
                params['part_number'] = f'%{part_number}%'

            if oem_mfg:
                conditions.append("LOWER(oem_mfg) LIKE LOWER(:oem_mfg)")
                params['oem_mfg'] = f'%{oem_mfg}%'

            if model:
                conditions.append("LOWER(model) LIKE LOWER(:model)")
                params['model'] = f'%{model}%'

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            sql_query = text(f"""
                SELECT id, part_number, name, oem_mfg, model, class_flag, ud6, type, notes, documentation
                FROM part 
                WHERE {where_clause}
                LIMIT :limit
            """)

            params['limit'] = search_kwargs.get('limit', 20)

            result = self.session.execute(sql_query, params)

            parts = []
            for row in result:
                parts.append({
                    'id': row[0],
                    'part_number': row[1] or 'Unknown',
                    'name': row[2] or 'Unknown Part',
                    'oem_mfg': row[3] or 'Unknown',
                    'model': row[4] or 'Unknown',
                    'class_flag': row[5] or '',
                    'ud6': row[6] or '',
                    'type': row[7] or '',
                    'notes': row[8] or '',
                    'documentation': row[9] or ''
                })

            logger.info(f"SQL fallback found {len(parts)} parts")
            return parts

        except Exception as e:
            logger.error(f"SQL fallback also failed: {e}")
            return []

    def _fallback_part_search(self, search_kwargs: Dict) -> List:
        """
        Fallback part search using raw SQL if Part.search fails
        """
        try:
            from sqlalchemy import text, or_

            search_text = search_kwargs.get('search_text', '')
            if not search_text:
                return []

            # Use raw SQL search as fallback
            sql_query = text("""
                SELECT id, part_number, name, oem_mfg, model, notes
                FROM part 
                WHERE 
                    LOWER(name) LIKE LOWER(:search_term) OR
                    LOWER(part_number) LIKE LOWER(:search_term) OR  
                    LOWER(oem_mfg) LIKE LOWER(:search_term) OR
                    LOWER(model) LIKE LOWER(:search_term) OR
                    LOWER(notes) LIKE LOWER(:search_term)
                LIMIT :limit
            """)

            result = self.session.execute(sql_query, {
                'search_term': f'%{search_text}%',
                'limit': search_kwargs.get('limit', 20)
            })

            parts = []
            for row in result:
                parts.append({
                    'id': row[0],
                    'part_number': row[1],
                    'name': row[2],
                    'oem_mfg': row[3],
                    'model': row[4],
                    'notes': row[5]
                })

            logger.info(f" Fallback SQL search found {len(parts)} parts")
            return parts

        except Exception as e:
            logger.error(f" Fallback part search also failed: {e}")
            return []

    @classmethod
    @with_request_id
    def _extract_part_candidates(self, text: str) -> List[str]:
        """Extract potential part numbers from text using multiple patterns."""
        import re
        candidates = []

        if not text:
            return candidates

        # Pattern 1: Look for explicit part number requests first
        part_description_patterns = [
            r'(?:i\s+)?need\s+(?:the\s+)?part\s+number\s+for\s+(.+?)(?:\s*$|\s*\?)',
            # "I need part number for BEARING ASSEMBLY"
            r'what\s+(?:is\s+)?(?:the\s+)?part\s+number\s+for\s+(.+?)(?:\s*$|\s*\?)',  # "what is part number for..."
            r'part\s+number\s+for\s+(.+?)(?:\s*$|\s*\?)',  # "part number for BEARING ASSEMBLY"
            r'(?:find|get|show)\s+(?:me\s+)?(?:the\s+)?part\s+number\s+for\s+(.+?)(?:\s*$|\s*\?)',
            # "find part number for..."
        ]

        # Check for part description requests (these should be treated as search text, not part numbers)
        for pattern in part_description_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                description = match.strip()
                # If it's a multi-word description, mark it as search text
                if len(description.split()) > 1:
                    # Return the description as a special marker that indicates it's search text
                    return [f"DESCRIPTION:{description}"]

        # Pattern 2: Look for actual part numbers (alphanumeric codes)
        part_number_patterns = [
            r'\b([A-Z]\d{5,})\b',  # A120404 (letter followed by 5+ digits)
            r'\b(\d{5,})\b',  # 120404 (5+ digits standalone)
            r'\b([A-Z0-9]{2,}[-][A-Z0-9]{2,})\b',  # ABC-123 (alphanumeric with dash)
            r'\b([A-Z0-9]{2,}[\.][A-Z0-9]{2,})\b',  # ABC.123 (alphanumeric with dot)
            r'\b([A-Z]{2,}\d{2,})\b',  # AB123 (letters followed by digits)
            r'\b(\d{2,}[A-Z]{2,})\b',  # 123AB (digits followed by letters)
        ]

        for pattern in part_number_patterns:
            matches = re.findall(pattern, text.upper())
            candidates.extend(matches)

        # Pattern 3: Extract from specific "part X" contexts (only if X looks like a part number)
        specific_part_patterns = [
            r'(?:show|find|get)\s+part\s+([A-Z0-9\-\.]{3,})\b',  # "show part ABC123"
            r'part\s+([A-Z0-9\-\.]{3,})\b',  # "part ABC123"
            r'part\s+number\s+([A-Z0-9\-\.]{3,})\b',  # "part number ABC123"
        ]

        for pattern in specific_part_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Only include if it looks like a part number (not just words)
                if re.match(r'^[A-Z0-9\-\.]{3,}$', match.upper()) and not re.match(r'^[A-Z]+$', match.upper()):
                    candidates.append(match.upper())

        # Pattern 4: Standalone alphanumeric codes (be more selective)
        standalone_patterns = [
            r'\b([A-Z0-9]{6,})\b',  # 6+ character alphanumeric codes
            r'\b([A-Z]\d{4,})\b',  # Letter + 4+ digits
            r'\b(\d{4,}[A-Z]+)\b',  # 4+ digits + letters
        ]

        for pattern in standalone_patterns:
            matches = re.findall(pattern, text.upper())
            for match in matches:
                # Exclude common words that might match these patterns
                excluded_words = {'BEARING', 'ASSEMBLY', 'STAGE', 'FIRST', 'SECOND', 'MOTOR', 'VALVE', 'PUMP'}
                if match not in excluded_words:
                    candidates.append(match)

        # Clean and deduplicate
        cleaned_candidates = []
        for candidate in candidates:
            if candidate.startswith('DESCRIPTION:'):
                # Return description immediately if found
                return [candidate]

            cleaned = candidate.strip().upper()
            # More selective filtering
            if (len(cleaned) >= 3 and
                    cleaned not in cleaned_candidates and
                    not cleaned.isalpha() and  # Exclude pure alphabetic strings
                    cleaned not in {'FOR', 'THE', 'AND', 'WITH', 'FROM'}):  # Exclude common words
                cleaned_candidates.append(cleaned)

        return cleaned_candidates

    def _fuzzy_part_search(self, part_candidates: List[str], threshold: int = 70) -> List[Tuple[Any, int]]:
        """
        ENHANCED: Perform fuzzy search for parts using multiple strategies with dual part number support.

        Now searches both:
        - Company part numbers (part.part_number)
        - Manufacturer part numbers (part.model)
        - Manufacturer names (part.oem_mfg)

        Returns:
            List of (part_object, confidence_score) tuples
        """
        try:
            from fuzzywuzzy import fuzz, process
        except ImportError:
            logger.warning("fuzzywuzzy not available for fuzzy search")
            return []

        if not part_candidates:
            return []

        # Get all parts from database
        from modules.emtacdb.emtacdb_fts import Part
        all_parts = self.session.query(Part).all()
        if not all_parts:
            return []

        results = []

        logger.debug(f"Starting fuzzy search with {len(part_candidates)} candidates against {len(all_parts)} parts")

        for candidate in part_candidates:
            if not candidate or len(candidate) < 2:
                continue

            candidate_upper = candidate.upper()

            # STRATEGY 1: Company part numbers (part.part_number) - HIGHEST PRIORITY
            company_part_numbers = [p.part_number for p in all_parts if p.part_number]
            if company_part_numbers:
                fuzzy_matches = process.extract(candidate, company_part_numbers, limit=5, scorer=fuzz.ratio)

                for match_text, score in fuzzy_matches:
                    if score >= threshold:
                        part = next((p for p in all_parts if p.part_number == match_text), None)
                        if part:
                            # Boost score for company parts (highest priority)
                            boosted_score = min(100, score + 5)
                            results.append((part, boosted_score))
                            logger.debug(f" Company part fuzzy match: {candidate} → {match_text} ({boosted_score})")

            # STRATEGY 2: Manufacturer part numbers (part.model)
            mfg_part_numbers = [p.model for p in all_parts if p.model]
            if mfg_part_numbers:
                fuzzy_matches = process.extract(candidate, mfg_part_numbers, limit=5, scorer=fuzz.ratio)

                for match_text, score in fuzzy_matches:
                    if score >= threshold:
                        part = next((p for p in all_parts if p.model == match_text), None)
                        if part:
                            results.append((part, score))
                            logger.debug(f"Manufacturer part fuzzy match: {candidate} → {match_text} ({score})")

            # STRATEGY 3: Manufacturer names (part.oem_mfg)
            manufacturers = [p.oem_mfg for p in all_parts if p.oem_mfg]
            if manufacturers:
                # Remove duplicates
                unique_manufacturers = list(set(manufacturers))
                fuzzy_matches = process.extract(candidate, unique_manufacturers, limit=3, scorer=fuzz.ratio)

                for match_text, score in fuzzy_matches:
                    if score >= threshold:
                        # Find all parts from this manufacturer
                        manufacturer_parts = [p for p in all_parts if p.oem_mfg == match_text]
                        for part in manufacturer_parts[:3]:  # Limit to top 3 parts per manufacturer
                            results.append((part, score - 10))  # Slightly lower priority
                            logger.debug(f"Manufacturer fuzzy match: {candidate} → {match_text} ({score - 10})")

            # STRATEGY 4: Partial ratio for embedded matches in company parts
            for part in all_parts:
                if part.part_number:
                    partial_score = fuzz.partial_ratio(candidate_upper, part.part_number.upper())
                    if partial_score >= threshold + 10:  # Higher threshold for partial
                        results.append((part, partial_score - 5))
                        logger.debug(
                            f" Company part partial match: {candidate} → {part.part_number} ({partial_score - 5})")

            # STRATEGY 5: Partial ratio for embedded matches in manufacturer parts
            for part in all_parts:
                if part.model:
                    partial_score = fuzz.partial_ratio(candidate_upper, part.model.upper())
                    if partial_score >= threshold + 10:  # Higher threshold for partial
                        results.append((part, partial_score - 10))
                        logger.debug(
                            f"Manufacturer part partial match: {candidate} → {part.model} ({partial_score - 10})")

            # STRATEGY 6: Token set ratio for flexible matching (company part + name)
            for part in all_parts:
                if part.part_number and part.name:
                    combined_text = f"{part.part_number} {part.name}"
                    token_score = fuzz.token_set_ratio(candidate, combined_text)
                    if token_score >= threshold:
                        results.append((part, token_score - 15))
                        logger.debug(f" Token set match: {candidate} → {combined_text} ({token_score - 15})")

        # Remove duplicates and sort by confidence
        unique_results = {}
        for part, score in results:
            part_id = part.id
            if part_id not in unique_results or score > unique_results[part_id][1]:
                unique_results[part_id] = (part, score)

        # Sort by score (highest first)
        sorted_results = sorted(unique_results.values(), key=lambda x: x[1], reverse=True)

        # Log top results
        logger.info(f"Fuzzy search found {len(sorted_results)} unique matches")
        for i, (part, score) in enumerate(sorted_results[:3]):
            logger.info(f"  {i + 1}. {part.part_number} | {part.oem_mfg} {part.model} ({score})")

        return sorted_results

    def comprehensive_position_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive aggregated position search that finds positions AND all related content.
        Aggregates parts, images, and hierarchical context for each position.

        Args:
            params: Dictionary of search parameters

        Returns:
            Dictionary with comprehensive aggregated search results
        """
        if not MODELS_AVAILABLE:
            return {
                "status": "error",
                "message": "Database models not available for position search"
            }

        try:
            # Use PartsPositionImageAssociation to get positions based on hierarchy
            position_filters = {}

            # Map hierarchy parameters
            if 'area' in params:
                area = self.session.query(Area).filter(Area.name.ilike(f"%{params['area']}%")).first()
                if area:
                    position_filters['area_id'] = area.id

            if 'equipment_group' in params or 'equipment' in params:
                equipment_name = params.get('equipment_group') or params.get('equipment')
                equipment = self.session.query(EquipmentGroup).filter(
                    EquipmentGroup.name.ilike(f"%{equipment_name}%")).first()
                if equipment:
                    position_filters['equipment_group_id'] = equipment.id

            if 'model' in params:
                model = self.session.query(Model).filter(Model.name.ilike(f"%{params['model']}%")).first()
                if model:
                    position_filters['model_id'] = model.id

            if 'location' in params:
                location = self.session.query(Location).filter(Location.name.ilike(f"%{params['location']}%")).first()
                if location:
                    position_filters['location_id'] = location.id

            if 'extracted_id' in params:
                position_filters['position_id'] = params['extracted_id']

            # Get position IDs using the association method
            if position_filters:
                position_ids = PartsPositionImageAssociation.get_corresponding_position_ids(
                    session=self.session,
                    **position_filters
                )
            else:
                # Fallback to all positions if no specific filters
                all_positions = self.session.query(Position).limit(int(params.get('limit', 10))).all()
                position_ids = [pos.id for pos in all_positions]

            comprehensive_results = []
            for position_id in position_ids[:int(params.get('limit', 10))]:
                position = self.session.query(Position).get(position_id)
                if position:
                    position_result = self._enhance_position_result(position)
                    comprehensive_results.append(position_result)

            return {
                'status': 'success',
                'count': len(comprehensive_results),
                'results': comprehensive_results,
                'entity_type': 'position',
                'search_type': 'comprehensive',
                'total_parts': sum(result.get('part_count', 0) for result in comprehensive_results),
                'total_images': sum(result.get('image_count', 0) for result in comprehensive_results)
            }

        except Exception as e:
            logger.error(f"Error in comprehensive position search: {e}")
            return {
                'status': 'error',
                'message': f"Error in comprehensive position search: {str(e)}",
                'entity_type': 'position'
            }

    def comprehensive_search_by_similarity(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Advanced similarity-based aggregated search using image embeddings.
        Aggregates visually similar content from across the database.

        Args:
            params: Dictionary containing either 'reference_image_id' or 'search_text'

        Returns:
            Dictionary with aggregated similarity search results
        """
        if not MODELS_AVAILABLE:
            return {
                "status": "error",
                "message": "Database models not available for similarity search"
            }

        try:
            similarity_results = []

            if 'reference_image_id' in params:
                # Find images similar to a reference image
                similar_images = Image.find_similar_images(
                    session=self.session,
                    reference_image_id=params['reference_image_id'],
                    limit=int(params.get('limit', 10)),
                    similarity_threshold=float(params.get('similarity_threshold', 0.7))
                )
                similarity_results = similar_images

            elif 'search_text' in params or 'raw_input' in params:
                # Try text-to-image similarity if available
                search_text = params.get('search_text') or params.get('raw_input')
                try:
                    # This would require the ModelsConfig to be available
                    # For now, we'll return a placeholder
                    return {
                        'status': 'info',
                        'message': 'Text-to-image similarity search requires additional configuration',
                        'entity_type': 'image',
                        'search_type': 'similarity'
                    }
                except Exception as e:
                    logger.warning(f"Text similarity search failed: {e}")
                    return {
                        'status': 'error',
                        'message': f'Text similarity search failed: {str(e)}',
                        'entity_type': 'image',
                        'search_type': 'similarity'
                    }

            # Enhance similarity results with comprehensive context
            enhanced_results = []
            for result in similarity_results:
                enhanced_result = self._enhance_similarity_result(result)
                enhanced_results.append(enhanced_result)

            return {
                'status': 'success',
                'count': len(enhanced_results),
                'results': enhanced_results,
                'entity_type': 'image',
                'search_type': 'similarity',
                'average_similarity': sum(r.get('similarity_score', 0) for r in enhanced_results) / len(
                    enhanced_results) if enhanced_results else 0,
                'similarity_features': {
                    'using_reference_image': 'reference_image_id' in params,
                    'using_text_query': 'search_text' in params or 'raw_input' in params,
                    'threshold': params.get('similarity_threshold', 0.7),
                    'pgvector_enabled': True
                }
            }

        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return {
                'status': 'error',
                'message': f"Error in similarity search: {str(e)}",
                'entity_type': 'image',
                'search_type': 'similarity'
            }

    def _enhance_part_result(self, part) -> Dict[str, Any]:
        """Enhance part result with related positions and images."""
        part_data = {
            'id': part.id,
            'part_number': getattr(part, 'part_number', 'Unknown'),
            'name': getattr(part, 'name', 'Unknown'),
            'oem_mfg': getattr(part, 'oem_mfg', 'Unknown'),
            'model': getattr(part, 'model', 'Unknown'),
            'class_flag': getattr(part, 'class_flag', None),
            'notes': getattr(part, 'notes', None),
            'type': 'part'
        }

        try:
            # Try to find associations, but don't fail if there's an error
            positions_data = []
            images_data = []

            # Use a separate try block for association queries
            try:
                part_positions = PartsPositionImageAssociation.search(
                    session=self.session,
                    part_id=part.id
                )

                for assoc in part_positions:
                    # Add position information
                    if assoc.position:
                        pos_data = {
                            'id': assoc.position.id,
                            'area': assoc.position.area.name if assoc.position.area else None,
                            'equipment_group': assoc.position.equipment_group.name if assoc.position.equipment_group else None,
                            'model': assoc.position.model.name if assoc.position.model else None,
                            'location': assoc.position.location.name if assoc.position.location else None,
                        }
                        if pos_data not in positions_data:
                            positions_data.append(pos_data)

                    # Add image information
                    if assoc.image:
                        img_data = {
                            'id': assoc.image.id,
                            'title': getattr(assoc.image, 'title', 'Untitled'),
                            'description': getattr(assoc.image, 'description', ''),
                            'file_path': getattr(assoc.image, 'file_path', ''),
                            'url': f"/serve_image/{assoc.image.id}"
                        }
                        if img_data not in images_data:
                            images_data.append(img_data)
            except Exception as assoc_error:
                logger.debug(f"Could not retrieve associations for part {part.id}: {assoc_error}")

            # Add related information to part data
            part_data.update({
                'positions': positions_data,
                'position_count': len(positions_data),
                'images': images_data,
                'image_count': len(images_data),
                'usage_locations': list(set([pos['area'] for pos in positions_data if pos['area']])),
                'equipment_types': list(
                    set([pos['equipment_group'] for pos in positions_data if pos['equipment_group']]))
            })

        except Exception as e:
            logger.warning(f"Could not enhance part result {part.id}: {e}")
            # Return basic part data even if enhancement fails
            part_data.update({
                'positions': [],
                'position_count': 0,
                'images': [],
                'image_count': 0,
                'usage_locations': [],
                'equipment_types': []
            })

        return part_data

    def _enhance_position_result(self, position) -> Dict[str, Any]:
        """Enhanced position result with proper null checking."""
        # FIXED: Proper null checking for position object
        if not position:
            logger.warning("Position object is None")
            return {
                'id': None,
                'area': None,
                'equipment_group': None,
                'model': None,
                'location': None,
                'parts': [],
                'images': [],
                'part_count': 0,
                'image_count': 0
            }

        position_data = {
            'id': getattr(position, 'id', None),
            'area': getattr(position.area, 'name', None) if hasattr(position, 'area') and position.area else None,
            'equipment_group': getattr(position.equipment_group, 'name', None) if hasattr(position,
                                                                                          'equipment_group') and position.equipment_group else None,
            'model': getattr(position.model, 'name', None) if hasattr(position, 'model') and position.model else None,
            'location': getattr(position.location, 'name', None) if hasattr(position,
                                                                            'location') and position.location else None,
            'area_id': getattr(position, 'area_id', None),
            'equipment_group_id': getattr(position, 'equipment_group_id', None),
            'model_id': getattr(position, 'model_id', None),
            'asset_number_id': getattr(position, 'asset_number_id', None),
            'location_id': getattr(position, 'location_id', None)
        }

        try:
            # Find all parts and images at this position
            position_associations = PartsPositionImageAssociation.search(
                session=self.session,
                position_id=position.id
            )

            parts_data = []
            images_data = []

            for assoc in position_associations:
                # Add part information
                if assoc.part:
                    part_data = {
                        'id': getattr(assoc.part, 'id', None),
                        'part_number': getattr(assoc.part, 'part_number', None),
                        'name': getattr(assoc.part, 'name', None),
                        'oem_mfg': getattr(assoc.part, 'oem_mfg', None),
                        'model': getattr(assoc.part, 'model', None)
                    }
                    if part_data not in parts_data:
                        parts_data.append(part_data)

                # Add image information
                if assoc.image:
                    img_data = {
                        'id': getattr(assoc.image, 'id', None),
                        'title': getattr(assoc.image, 'title', None),
                        'description': getattr(assoc.image, 'description', None),
                        'file_path': getattr(assoc.image, 'file_path', None),
                        'url': f"/serve_image/{assoc.image.id}" if assoc.image.id else None
                    }
                    if img_data not in images_data:
                        images_data.append(img_data)

            # Add related information
            position_data.update({
                'parts': parts_data,
                'part_count': len(parts_data),
                'images': images_data,
                'image_count': len(images_data)
            })

        except Exception as e:
            logger.warning(f"Could not enhance position result {getattr(position, 'id', 'unknown')}: {e}")
            # Ensure we still return a valid structure
            position_data.update({
                'parts': [],
                'part_count': 0,
                'images': [],
                'image_count': 0
            })

        return position_data

    def enhance_comprehensive_part_search_parameters(self, analysis_params, original_params):
        """
        Convert analysis parameters to proper Part.search parameters.
        Add this method to your AggregateSearch class.
        """
        # Start with the analysis parameters
        part_search_params = {}

        # Direct field mappings
        field_mappings = {
            'name': 'name',
            'part_number': 'part_number',
            'oem_mfg': 'oem_mfg',
            'model': 'model',
            'search_text': 'search_text',
            'fields': 'fields'
        }

        for analysis_key, part_key in field_mappings.items():
            if analysis_key in analysis_params:
                part_search_params[part_key] = analysis_params[analysis_key]
                print(f" Mapped {analysis_key} -> {part_key}: {analysis_params[analysis_key]}")

        # Handle extracted_id as part_id
        if 'extracted_id' in analysis_params:
            part_search_params['part_id'] = analysis_params['extracted_id']
            print(f" Mapped extracted_id -> part_id: {analysis_params['extracted_id']}")

        # Set defaults
        part_search_params['limit'] = original_params.get('limit', 20)
        part_search_params['exact_match'] = False

        return part_search_params

    @classmethod
    @with_request_id
    def detect_part_numbers_and_manufacturers(cls, terms: List[str], session, request_id: Optional[str] = None) -> Dict[
        str, List[str]]:
        """
        FIXED: Detect which terms are company part numbers, manufacturer part numbers,
        manufacturers, or equipment types with proper filtering.
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        from modules.emtacdb.emtacdb_fts import Part

        company_part_numbers = []
        mfg_part_numbers = []
        manufacturers = []
        equipment = []

        # CRITICAL: Filter out common English words that shouldn't be part numbers
        common_words = {
            'what', 'is', 'the', 'part', 'number', 'for', 'a', 'an', 'and', 'or', 'but',
            'in', 'on', 'at', 'to', 'from', 'with', 'by', 'of', 'as', 'this', 'that',
            'these', 'those', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'can', 'may', 'might', 'must', 'have', 'has', 'had', 'be', 'am', 'are',
            'was', 'were', 'being', 'been', 'get', 'got', 'give', 'gave', 'take',
            'took', 'make', 'made', 'come', 'came', 'go', 'went', 'see', 'saw',
            'know', 'knew', 'think', 'thought', 'say', 'said', 'tell', 'told',
            'find', 'found', 'show', 'need', 'want', 'like', 'use', 'work'
        }

        debug_id(f"Starting detection for {len(terms)} terms: {terms}", rid)

        for term in terms:
            if len(term) < 2:
                continue

            term_lower = term.lower()
            term_upper = term.upper()

            # SKIP common English words entirely
            if term_lower in common_words:
                debug_id(f"Skipping common word: '{term}'", rid)
                continue

            # SKIP very short terms (likely noise)
            if len(term) < 3:
                debug_id(f"Skipping short term: '{term}'", rid)
                continue

            found_category = False

            try:
                # Check if term is a company part number (part.part_number)
                # Use more specific matching to avoid false positives
                company_part_result = session.query(Part.part_number).filter(
                    Part.part_number.ilike(f'{term}%')  # Starts with term (more specific)
                ).first()

                if not company_part_result:
                    # Try exact match
                    company_part_result = session.query(Part.part_number).filter(
                        Part.part_number.ilike(term)
                    ).first()

                if company_part_result:
                    company_part_numbers.append(term)
                    found_category = True
                    debug_id(f" '{term}' detected as company part number", rid)
                    continue

                # Check if term is a manufacturer part number (part.model)
                # More specific matching for manufacturer parts
                mfg_part_result = session.query(Part.model).filter(
                    Part.model.ilike(f'{term}%')  # Starts with term
                ).first()

                if not mfg_part_result:
                    # Try exact match
                    mfg_part_result = session.query(Part.model).filter(
                        Part.model.ilike(term)
                    ).first()

                if mfg_part_result:
                    # Additional validation: must look like a part number
                    if cls._looks_like_part_number(term):
                        mfg_part_numbers.append(term)
                        found_category = True
                        debug_id(f"'{term}' detected as manufacturer part number", rid)
                        continue
                    else:
                        debug_id(f"🚫 '{term}' found in model column but doesn't look like part number", rid)

                # Check if term is a manufacturer name (part.oem_mfg)
                # More specific matching for manufacturers
                manufacturer_result = session.query(Part.oem_mfg).filter(
                    Part.oem_mfg.ilike(f'{term}%')  # Starts with term
                ).first()

                if not manufacturer_result:
                    # Try exact match
                    manufacturer_result = session.query(Part.oem_mfg).filter(
                        Part.oem_mfg.ilike(term)
                    ).first()

                if manufacturer_result:
                    # Additional validation: must look like a manufacturer name
                    if cls._looks_like_manufacturer(term):
                        manufacturers.append(term)
                        found_category = True
                        debug_id(f"'{term}' detected as manufacturer", rid)
                        continue
                    else:
                        debug_id(f"🚫 '{term}' found in oem_mfg column but doesn't look like manufacturer", rid)

            except Exception as e:
                error_id(f"Error checking term '{term}': {e}", rid)
                continue

            # If not found in any part-related column, classify as equipment
            if not found_category:
                # Check against equipment keywords
                equipment_keywords = {
                    'sensor', 'sensors', 'motor', 'motors', 'pump', 'pumps',
                    'valve', 'valves', 'bearing', 'bearings', 'filter', 'filters',
                    'switch', 'switches', 'relay', 'relays', 'belt', 'belts',
                    'seal', 'seals', 'gasket', 'gaskets', 'coupling', 'gear', 'gears',
                    'hydraulic', 'pneumatic', 'electrical', 'mechanical', 'bypass'
                }

                if term_lower in equipment_keywords:
                    equipment.append(term)
                    debug_id(f"'{term}' detected as equipment", rid)
                else:
                    # Only add if it looks like it could be equipment/part related
                    if cls._could_be_equipment(term):
                        equipment.append(term)
                        debug_id(f" '{term}' classified as potential equipment", rid)
                    else:
                        debug_id(f"Skipping unlikely term: '{term}'", rid)

        result = {
            'company_part_numbers': company_part_numbers,
            'mfg_part_numbers': mfg_part_numbers,
            'manufacturers': manufacturers,
            'equipment': equipment
        }

        debug_id(f"Detection complete: {result}", rid)
        return result

    @classmethod
    @with_request_id
    def _looks_like_part_number(cls, term: str) -> bool:
        """Check if term looks like a part number (not just common words)."""
        import re

        # Must have some alphanumeric pattern
        if not re.search(r'[A-Z0-9]', term.upper()):
            return False

        # Must be longer than 2 characters
        if len(term) < 3:
            return False

        # Should contain numbers or be mostly uppercase
        has_numbers = re.search(r'\d', term)
        mostly_upper = term.isupper() and len(term) > 3
        has_dash_or_underscore = '-' in term or '_' in term

        return has_numbers or mostly_upper or has_dash_or_underscore

    @classmethod
    def _looks_like_manufacturer(cls, term: str) -> bool:
        """Check if term looks like a manufacturer name."""
        # Manufacturer names are usually:
        # - All caps (BANNER, DOLLINGER)
        # - Or proper case (Banner, Dollinger)
        # - At least 3 characters
        # - Not common English words

        if len(term) < 3:
            return False

        # All caps and not a common word
        if term.isupper() and len(term) >= 4:
            return True

        # Proper case (first letter capital)
        if term[0].isupper() and term[1:].islower() and len(term) >= 4:
            return True

        return False

    @classmethod
    def _could_be_equipment(cls, term: str) -> bool:
        """Check if term could reasonably be equipment-related."""
        # Equipment terms usually:
        # - Are at least 4 characters
        # - Don't contain weird punctuation
        # - Might have numbers/dashes for specifications

        if len(term) < 4:
            return False

        # Contains voltage patterns (110V, 120V, etc.)
        if re.search(r'\d+[-]?\d*V', term):
            return True

        # Contains size patterns (1-1/2", 3/4", etc.)
        if re.search(r'\d+[-/]\d+', term):
            return True

        # Reasonable length without too much punctuation
        punctuation_count = sum(1 for c in term if c in '!"#$%&\'()*+,.:;<=>?@[\\]^`{|}~')
        if punctuation_count <= 2:  # Allow some punctuation for specs
            return True

        return False

    @classmethod
    @with_request_id
    def enhanced_search_with_dual_part_detection(self, query: str, session) -> str:
        """
        Enhanced search that automatically detects:
        1. Company part numbers (part.part_number column)
        2. Manufacturer part numbers (part.model column)
        3. Manufacturer names (part.oem_mfg column)
        4. Equipment types (filters, sensors, etc.)

        Prioritizes searches based on specificity:
        Company part number > Manufacturer part number > Manufacturer + Equipment > Manufacturer > Equipment
        """
        from modules.search.utils import extract_search_terms

        # Extract terms from query
        terms = [term.strip() for term in query.upper().split() if term.strip()]
        logger.debug(f"Analyzing terms: {terms}")

        # Detect what each term represents
        detected = self.detect_part_numbers_and_manufacturers(terms, session)

        # Build search strategy based on what was detected
        search_components = []

        # Priority 1: Company part numbers (highest priority - your internal system)
        if detected['company_part_numbers']:
            company_parts = ' '.join(detected['company_part_numbers'])
            search_components.append(f"company_part:{company_parts}")
            logger.info(f" Priority 1: Company part number detected: {company_parts}")

        # Priority 2: Manufacturer part numbers (specific manufacturer models)
        if detected['mfg_part_numbers']:
            mfg_parts = ' '.join(detected['mfg_part_numbers'])
            search_components.append(f"mfg_part:{mfg_parts}")
            logger.info(f"Priority 2: Manufacturer part number detected: {mfg_parts}")

        # Priority 3: Manufacturer + Equipment combination
        if detected['manufacturers'] and detected['equipment']:
            manufacturer = detected['manufacturers'][0]  # Use first manufacturer
            equipment = ' '.join(detected['equipment'])
            search_components.append(f"manufacturer:{manufacturer} equipment:{equipment}")
            logger.info(f"Priority 3: Manufacturer+Equipment: {manufacturer} + {equipment}")

        # Priority 4: Manufacturer only
        elif detected['manufacturers']:
            manufacturer = ' '.join(detected['manufacturers'])
            search_components.append(f"manufacturer:{manufacturer}")
            logger.info(f"Priority 4: Manufacturer-only: {manufacturer}")

        # Priority 5: Equipment only
        elif detected['equipment']:
            equipment = ' '.join(detected['equipment'])
            search_components.append(f"equipment:{equipment}")
            logger.info(f"Priority 5: Equipment-only: {equipment}")

        # If nothing specific detected, use original query
        if not search_components:
            logger.info(f"No specific detection, using original query: {query}")
            return query

        # Return the highest priority search component
        selected_strategy = search_components[0]
        logger.info(f"Selected search strategy: {selected_strategy}")
        return selected_strategy

    #=======End of Part Search==========

    # ======== Enhancement Methods for Results ========

    def _enhance_image_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance image result with additional context."""
        image_id = result.get('id')
        if not image_id:
            return result

        try:
            # Get tool associations
            tool_associations = self.session.query(ToolImageAssociation).filter(
                ToolImageAssociation.image_id == image_id
            ).all()

            # Get task associations
            task_associations = self.session.query(ImageTaskAssociation).filter(
                ImageTaskAssociation.image_id == image_id
            ).all()

            # Get problem associations
            problem_associations = self.session.query(ImageProblemAssociation).filter(
                ImageProblemAssociation.image_id == image_id
            ).all()

            # Add comprehensive context
            result['comprehensive_context'] = {
                'tools': [{'id': assoc.tool.id, 'name': assoc.tool.name} for assoc in tool_associations if assoc.tool],
                'tasks': [{'id': assoc.task.id, 'name': assoc.task.name} for assoc in task_associations if assoc.task],
                'problems': [{'id': assoc.problem.id, 'name': assoc.problem.name} for assoc in problem_associations if
                             assoc.problem],
                'search_method': 'comprehensive_image_search'
            }

        except Exception as e:
            logger.warning(f"Could not enhance position result {position.id}: {e}")

        return position_data

    def _enhance_similarity_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance similarity result with comprehensive context."""
        image_data = result.get('image', {})
        image_id = image_data.get('id')

        if image_id:
            try:
                # Get comprehensive associations for similar image
                comprehensive_context = Image._get_enhanced_image_associations(
                    self.session,
                    image_id
                )

                enhanced_result = {
                    'similarity_score': result.get('similarity'),
                    'image': image_data,
                    'associations': comprehensive_context,
                    'search_metadata': {
                        'search_type': 'similarity',
                        'similarity_method': 'pgvector_embedding',
                        'confidence': result.get('similarity', 0)
                    }
                }

                return enhanced_result

            except Exception as e:
                logger.warning(f"Could not enhance similarity result for image {image_id}: {e}")

        return result

    # ======== Setup and Configuration Methods ========

    def setup_default_comprehensive_keywords(self) -> Dict[str, Any]:
        """
        Set up default keywords optimized for comprehensive aggregated searches.

        Note: This is a legacy method. New implementations should use
        SearchPatternManager.initialize_default_patterns() instead.
        """
        if not MODELS_AVAILABLE:
            return {
                "status": "error",
                "message": "Models not available for keyword setup"
            }

        try:
            default_keywords = [
                # Comprehensive aggregated image searches
                {
                    "keyword": "show images",
                    "action_type": "image_search",
                    "search_pattern": "show images {description}",
                    "description": "Find images with comprehensive aggregated context"
                },
                {
                    "keyword": "images in",
                    "action_type": "image_search",
                    "search_pattern": "images in {area}",
                    "description": "Find all images in a specific area"
                },
                {
                    "keyword": "pictures of",
                    "action_type": "image_search",
                    "search_pattern": "pictures of {equipment}",
                    "description": "Find images of specific equipment"
                },
                {
                    "keyword": "find similar images",
                    "action_type": "image_similarity",
                    "search_pattern": "find similar images to {reference_image_id}",
                    "description": "Find visually similar images"
                },
                {
                    "keyword": "visual search",
                    "action_type": "image_similarity",
                    "search_pattern": "visual search {search_text}",
                    "description": "Semantic visual search"
                },

                # Comprehensive aggregated part searches
                {
                    "keyword": "find part",
                    "action_type": "part_search",
                    "search_pattern": "find part {part_number}",
                    "description": "Find parts with aggregated context"
                },
                {
                    "keyword": "parts for",
                    "action_type": "part_search",
                    "search_pattern": "parts for {equipment}",
                    "description": "Find all parts used in specific equipment"
                },
                {
                    "keyword": "show part",
                    "action_type": "part_search",
                    "search_pattern": "show part {name}",
                    "description": "Comprehensive part search"
                },

                # Position-based aggregated searches
                {
                    "keyword": "what's in",
                    "action_type": "position_search",
                    "search_pattern": "what's in {area}",
                    "description": "Find all content in an area"
                },
                {
                    "keyword": "equipment at",
                    "action_type": "position_search",
                    "search_pattern": "equipment at {location}",
                    "description": "Find equipment at specific locations"
                }
            ]

            results = {
                "added": 0,
                "updated": 0,
                "failed": 0,
                "errors": []
            }

            for keyword_data in default_keywords:
                try:
                    result = self.register_keyword(**keyword_data)
                    if result['status'] == 'created':
                        results["added"] += 1
                    elif result['status'] == 'updated':
                        results["updated"] += 1
                    else:
                        results["failed"] += 1
                        results["errors"].append(
                            f"Failed to register '{keyword_data['keyword']}': {result.get('message')}")

                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(f"Error with '{keyword_data['keyword']}': {str(e)}")

            results["status"] = "success"
            results[
                "message"] = f"Setup complete: {results['added']} added, {results['updated']} updated, {results['failed']} failed"

            return results

        except Exception as e:
            logger.error(f"Error setting up default keywords: {e}")
            return {
                "status": "error",
                "message": f"Error setting up default keywords: {str(e)}"
            }

    def get_search_capabilities(self) -> Dict[str, Any]:
        """
        Get information about the comprehensive aggregated search capabilities available.

        Returns:
            Dictionary describing aggregated search features and capabilities
        """
        return {
            "search_types": {
                "image_search": {
                    "description": "Comprehensive aggregated image search with multi-path association discovery",
                    "features": [
                        "Position hierarchy aggregation (area/equipment/model/location)",
                        "Parts association discovery and aggregation",
                        "Tool/task/problem association aggregation",
                        "Document associations with structure-guided context aggregation",
                        "Text similarity matching with semantic aggregation",
                        "Hybrid ranking with relevance scoring across multiple sources"
                    ],
                    "implemented": True
                },
                "image_similarity": {
                    "description": "AI-powered visual similarity search with result aggregation",
                    "features": [
                        "Find visually similar images using aggregated embeddings",
                        "Reference image similarity search across all sources",
                        "Configurable similarity thresholds with aggregated filtering",
                        "pgvector similarity engine with comprehensive result aggregation"
                    ],
                    "implemented": True,
                    "ai_powered": True
                },
                "part_search": {
                    "description": "Comprehensive aggregated part search with usage context",
                    "features": [
                        "Full Part.search() method integration with result aggregation",
                        "Usage location discovery via aggregated PartsPositionImageAssociation",
                        "Related images and documentation aggregation",
                        "Equipment and position context aggregation",
                        "Multi-field search capabilities with comprehensive aggregation"
                    ],
                    "implemented": True
                },
                "position_search": {
                    "description": "Hierarchical position search with comprehensive context aggregation",
                    "features": [
                        "Hierarchy-based filtering with aggregated results (area/equipment/model)",
                        "Parts and images aggregation at each position",
                        "Cross-reference discovery and aggregation",
                        "Location-based context with comprehensive aggregation"
                    ],
                    "implemented": True
                },
                "drawing_search": {
                    "description": "Drawing search capabilities",
                    "implemented": False,
                    "status": "placeholder"
                },
                "tool_search": {
                    "description": "Tool search capabilities",
                    "implemented": False,
                    "status": "placeholder"
                },
                "problem_search": {
                    "description": "Problem diagnosis search capabilities",
                    "implemented": False,
                    "status": "placeholder"
                },
                "task_search": {
                    "description": "Task procedure search capabilities",
                    "implemented": False,
                    "status": "placeholder"
                }
            },
            "aggregated_features": {
                "comprehensive_associations": "Aggregates related content across all entity types",
                "multi_path_discovery": "Uses multiple search paths and aggregates complete results",
                "semantic_search": "AI-powered similarity and text understanding with result aggregation",
                "intelligent_caching": "LRU cache for performance optimization of aggregated results",
                "pattern_matching": "Flexible pattern extraction with comprehensive parameter aggregation",
                "legacy_compatibility": "Supports both legacy keyword system and modern NLP patterns"
            },
            "supported_patterns": [
                "show {equipment} in {area}",
                "find part {part_number}",
                "images of {description}",
                "what's in {location}",
                "similar to image {id}"
            ],
            "database_requirements": {
                "models_available": MODELS_AVAILABLE,
                "required_models": [
                    "Image", "Part", "Position", "Area", "EquipmentGroup",
                    "Model", "Location", "PartsPositionImageAssociation"
                ],
                "optional_models": [
                    "ToolImageAssociation", "ImageTaskAssociation",
                    "ImageProblemAssociation", "KeywordAction"
                ]
            },
            "integration_notes": {
                "nlp_enhancement": "Use SpaCyEnhancedAggregateSearch for advanced NLP capabilities",
                "pattern_management": "Use SearchPatternManager for modern pattern management",
                "legacy_support": "Legacy keyword system still supported for backward compatibility"
            }
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and health information."""
        status = {
            "system_status": "operational" if MODELS_AVAILABLE else "limited",
            "models_available": MODELS_AVAILABLE,
            "database_connected": self.session is not None,
            "cache_size": len(self._search_cache),
            "cache_limit": self._cache_limit,
            "timestamp": datetime.utcnow().isoformat()
        }

        if MODELS_AVAILABLE and self.session:
            try:
                # Test database connectivity
                test_query = self.session.query(Image).limit(1).first()
                status["database_test"] = "passed"
                status["sample_image_available"] = test_query is not None
            except Exception as e:
                status["database_test"] = "failed"
                status["database_error"] = str(e)

        return status(f"Could not enhance image result {image_id}: {e}")

        return result

#==============ai question and answer

    def find_most_relevant_document_chunk(self, question, model_name=None, session=None, request_id=None):
        """
        Find the most relevant document chunk - FAST VERSION.
        Skip slow batch processing, use pgvector only.
        """
        from plugins.ai_modules import generate_embedding, ModelsConfig
        from modules.configuration.config_env import DatabaseConfig

        # Get model name
        if model_name is None:
            model_name = ModelsConfig.get_config_value('embedding', 'CURRENT_MODEL', 'NoEmbeddingModel')

        if model_name == "NoEmbeddingModel":
            logger.info("Embeddings are disabled. Returning None for chunk search.")
            return None

        # Session management
        session_created = False
        if session is None:
            db_config = DatabaseConfig()
            session = db_config.get_main_session()
            session_created = True

        try:
            # Check cache first
            cache_key = f"chunk:{question}:{model_name}"
            cached_result = self._get_cached_chunk_result(cache_key, session)
            if cached_result is not None:
                return cached_result

            # Generate embedding for the question
            question_embedding = generate_embedding(question, model_name)
            if not question_embedding:
                logger.info("No embeddings generated for question. Returning None.")
                self._cache_chunk_result(cache_key, None)
                return None

            # ONLY try pgvector - skip slow batch processing
            try:
                chunk = self._search_chunks_with_pgvector(session, question_embedding, model_name)
                if chunk:
                    logger.info(f"pgvector found relevant chunk: Document ID {chunk.id}")
                    self._cache_chunk_result(cache_key, chunk.id)
                    return chunk
                else:
                    logger.info("pgvector found no results above threshold")
                    self._cache_chunk_result(cache_key, None)
                    return None

            except Exception as e:
                logger.error(f"pgvector chunk search failed: {e}")
                self._cache_chunk_result(cache_key, None)
                return None

        except Exception as e:
            logger.error(f"Error in chunk search: {e}")
            return None
        finally:
            if session_created and session:
                session.close()

    def _search_chunks_with_pgvector(self, session, question_embedding, model_name):
        """
        Fast pgvector search - optimized to avoid problematic embeddings.
        """
        from sqlalchemy import text

        query_vector_str = '[' + ','.join(map(str, question_embedding)) + ']'

        # Optimized query that filters out truncated embeddings
        search_query = text("""
            SELECT 
                de.document_id,
                d.content,
                d.name as chunk_name,
                cd.title as document_title,
                1 - (de.embedding_vector <=> :query_vector) AS similarity
            FROM document_embedding de
            JOIN document d ON de.document_id = d.id
            LEFT JOIN complete_document cd ON d.complete_document_id = cd.id
            WHERE de.model_name = :model_name
              AND de.embedding_vector IS NOT NULL
              AND d.content IS NOT NULL
              AND LENGTH(d.content) > 50
              AND de.actual_dimensions > 1000  -- Only use embeddings with proper dimensions
              AND (1 - (de.embedding_vector <=> :query_vector)) >= 0.3  -- Higher threshold for speed
            ORDER BY de.embedding_vector <=> :query_vector ASC
            LIMIT 1
        """)

        try:
            result = session.execute(search_query, {
                'query_vector': query_vector_str,
                'model_name': model_name
            }).fetchone()

            if result:
                doc_id, content, chunk_name, doc_title, similarity = result
                logger.info(f"pgvector found chunk {doc_id} with similarity {similarity:.4f} from '{doc_title}'")

                chunk = session.query(Document).get(doc_id)
                if chunk:
                    # Attach similarity metadata
                    chunk._similarity_score = float(similarity)
                    chunk._search_metadata = {
                        'method': 'pgvector_fast',
                        'similarity': float(similarity),
                        'document_title': doc_title,
                        'chunk_name': chunk_name
                    }
                return chunk

            return None

        except Exception as e:
            logger.error(f"Fast pgvector search failed: {e}")
            return None

    @with_request_id
    def _get_cached_chunk_result(self, cache_key, session):
        """Get cached chunk result if available"""
        if hasattr(self.find_most_relevant_document_chunk,
                   'cache') and cache_key in self.find_most_relevant_document_chunk.cache:
            logger.info("Using cached chunk result")
            cached_doc_id = self.find_most_relevant_document_chunk.cache[cache_key]
            if cached_doc_id is None:
                return None
            chunk = session.query(Document).get(cached_doc_id)
            if chunk:
                logger.info(f"Retrieved cached chunk: Document ID {cached_doc_id}")
            return chunk
        return None

    @with_request_id
    def _cache_chunk_result(self, cache_key, doc_id):
        """Cache chunk search results"""
        if not hasattr(self.find_most_relevant_document_chunk, 'cache'):
            self.find_most_relevant_document_chunk.cache = {}

        cache = self.find_most_relevant_document_chunk.cache
        if len(cache) > 100:  # Keep cache size reasonable
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(cache))
            del cache[oldest_key]

        cache[cache_key] = doc_id
        logger.debug(f"Cached chunk result: {cache_key} -> {doc_id}")





