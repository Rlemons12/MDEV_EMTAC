# modules/search/aggregate_search.py
"""
Comprehensive aggregated search system for the chatbot.
Aggregates information from multiple sources and search paths to provide
comprehensive contextual search results across all database entities.
"""

import json
import re
import logging
from fuzzywuzzy import fuzz, process
from typing import List, Tuple, Dict, Any, Optional
from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import (
        Image, Part, Position, Area, EquipmentGroup, Model, Location,
        PartsPositionImageAssociation, ToolImageAssociation,
        ImageTaskAssociation, ImageProblemAssociation,
        KeywordAction  # Legacy compatibility
    )

MODELS_AVAILABLE = True
logger = logging.getLogger(__name__)


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

    def comprehensive_part_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        FIXED: Direct part table search instead of position-based search

        This replaces your existing comprehensive_part_search method to force
        direct Part.search() calls with description text.
        """
        logger.debug(f"🔍 Starting direct part search with params: {params}")

        if not MODELS_AVAILABLE:
            return {
                "status": "error",
                "message": "Database models not available for part search"
            }

        try:
            # Clean session state
            if self.session:
                try:
                    self.session.rollback()
                    logger.debug("Session rolled back for clean state")
                except Exception:
                    pass

            # Import Part model
            from modules.emtacdb.emtacdb_fts import Part

            search_kwargs = {}
            search_method = "unknown"

            # STRATEGY 1: Description-based search (your main use case)
            if 'search_text' in params and params['search_text']:
                search_text = params['search_text']
                logger.info(f"🎯 DIRECT PART SEARCH for description: '{search_text}'")

                search_kwargs = {
                    'search_text': search_text,
                    'fields': ['name', 'part_number', 'oem_mfg', 'model', 'notes'],
                    'limit': int(params.get('limit', 20))
                }
                search_method = "description_search"

            # STRATEGY 2: Direct part number search
            elif 'part_number' in params:
                logger.info(f"🎯 DIRECT PART SEARCH for part number: '{params['part_number']}'")
                search_kwargs = {
                    'part_number': params['part_number'],
                    'limit': int(params.get('limit', 20))
                }
                search_method = "part_number_search"

            # STRATEGY 3: ID-based search
            elif 'extracted_id' in params:
                logger.info(f"🎯 DIRECT PART SEARCH for ID: {params['extracted_id']}")
                search_kwargs = {
                    'part_id': params['extracted_id'],
                    'limit': 1
                }
                search_method = "id_search"

            else:
                logger.warning("❌ No valid search parameters for direct part search")
                return {
                    'status': 'error',
                    'message': 'No valid search parameters (need search_text, part_number, or extracted_id)',
                    'params_received': params
                }

            # EXECUTE DIRECT PART SEARCH
            logger.debug(f"🔍 Executing Part.search with kwargs: {search_kwargs}")

            try:
                parts = Part.search(session=self.session, **search_kwargs)
                logger.info(f"✅ Part.search found {len(parts)} parts using {search_method}")

            except Exception as search_error:
                logger.warning(f"⚠️ Part.search failed: {search_error}")
                # Fallback to SQL search
                parts = self._fallback_sql_part_search(search_kwargs)
                search_method += "_sql_fallback"

            # FORMAT RESULTS AS PROPER PART OBJECTS
            comprehensive_results = []

            for i, part in enumerate(parts):
                try:
                    # Handle SQLAlchemy Part objects
                    if hasattr(part, 'part_number'):
                        part_result = {
                            'id': getattr(part, 'id', None),
                            'part_number': getattr(part, 'part_number', f'Unknown-{i + 1}'),
                            'name': getattr(part, 'name', 'Unknown Part'),
                            'oem_mfg': getattr(part, 'oem_mfg', 'Unknown'),
                            'model': getattr(part, 'model', 'Unknown'),
                            'notes': getattr(part, 'notes', ''),
                            'type': 'part',
                            'entity_type': 'part',  # Critical for classification
                            'search_method': search_method
                        }

                    # Handle dictionary results
                    elif isinstance(part, dict):
                        part_result = {
                            'id': part.get('id'),
                            'part_number': part.get('part_number', f'Dict-{i + 1}'),
                            'name': part.get('name', 'Unknown Part'),
                            'oem_mfg': part.get('oem_mfg', 'Unknown'),
                            'model': part.get('model', 'Unknown'),
                            'notes': part.get('notes', ''),
                            'type': 'part',
                            'entity_type': 'part',
                            'search_method': search_method
                        }

                    else:
                        # Unknown object type fallback
                        part_result = {
                            'id': getattr(part, 'id', None) if hasattr(part, 'id') else i + 1,
                            'part_number': str(part) if part else f'Object-{i + 1}',
                            'name': 'Unknown Part',
                            'type': 'part',
                            'entity_type': 'part',
                            'search_method': search_method + "_object"
                        }

                    comprehensive_results.append(part_result)
                    logger.debug(f"✅ Formatted part {i + 1}: {part_result['part_number']}")

                except Exception as format_error:
                    logger.warning(f"⚠️ Failed to format part {i + 1}: {format_error}")
                    # Add minimal part info so search doesn't fail completely
                    comprehensive_results.append({
                        'id': i + 1,
                        'part_number': f'Error-{i + 1}',
                        'name': 'Formatting Error',
                        'type': 'part',
                        'entity_type': 'part',
                        'search_method': search_method + "_error"
                    })

            # RETURN SUCCESS RESULT
            result = {
                'status': 'success',
                'count': len(comprehensive_results),
                'results': comprehensive_results,
                'entity_type': 'part',
                'search_type': 'direct_part_search',
                'search_method': search_method,
                'search_parameters': search_kwargs,
                'note': f'Searched parts table directly with {search_method}'
            }

            logger.info(f"🎉 Direct part search SUCCESS: {len(comprehensive_results)} parts found")
            return result

        except Exception as e:
            logger.error(f"❌ Direct part search ERROR: {e}", exc_info=True)

            # Rollback session on error
            if self.session:
                try:
                    self.session.rollback()
                except Exception:
                    pass

            return {
                'status': 'error',
                'message': f"Direct part search failed: {str(e)}",
                'entity_type': 'part',
                'search_parameters': params,
                'error_type': 'direct_part_search_error'
            }

    def _fallback_sql_part_search(self, search_kwargs: Dict) -> List:
        """
        SQL fallback when Part.search fails
        """
        try:
            from sqlalchemy import text

            search_text = search_kwargs.get('search_text', '')
            if not search_text:
                logger.warning("No search_text for SQL fallback")
                return []

            logger.info(f"🔄 Trying SQL fallback for: '{search_text}'")

            # Use raw SQL search
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
                    'part_number': row[1] or 'Unknown',
                    'name': row[2] or 'Unknown Part',
                    'oem_mfg': row[3] or 'Unknown',
                    'model': row[4] or 'Unknown',
                    'notes': row[5] or ''
                })

            logger.info(f"✅ SQL fallback found {len(parts)} parts")
            return parts

        except Exception as e:
            logger.error(f"❌ SQL fallback also failed: {e}")
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

            logger.info(f"🔄 Fallback SQL search found {len(parts)} parts")
            return parts

        except Exception as e:
            logger.error(f"❌ Fallback part search also failed: {e}")
            return []

    def _extract_part_candidates(self, text: str) -> List[str]:
        """Extract potential part numbers from text using multiple patterns."""
        import re
        candidates = []

        if not text:
            return candidates

        # FIXED: Better patterns that handle part descriptions vs part numbers

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
        Perform fuzzy search for parts using multiple strategies.

        Returns:
            List of (part_object, confidence_score) tuples
        """
        try:
            from fuzzywuzzy import fuzz, process
        except ImportError:
            return []

        if not part_candidates:
            return []

        # Get all parts from database
        all_parts = self.session.query(Part).all()
        if not all_parts:
            return []

        results = []

        for candidate in part_candidates:
            # Strategy 1: Exact fuzzy match on part_number
            part_numbers = [p.part_number for p in all_parts if p.part_number]
            if part_numbers:
                fuzzy_matches = process.extract(candidate, part_numbers, limit=5, scorer=fuzz.ratio)

                for match_text, score in fuzzy_matches:
                    if score >= threshold:
                        part = next((p for p in all_parts if p.part_number == match_text), None)
                        if part:
                            results.append((part, score))

            # Strategy 2: Partial ratio for embedded matches
            for part in all_parts:
                if part.part_number:
                    partial_score = fuzz.partial_ratio(candidate.upper(), part.part_number.upper())
                    if partial_score >= threshold + 10:  # Higher threshold for partial
                        results.append((part, partial_score))

            # Strategy 3: Token set ratio for flexible matching
            for part in all_parts:
                if part.part_number and part.name:
                    combined_text = f"{part.part_number} {part.name}"
                    token_score = fuzz.token_set_ratio(candidate, combined_text)
                    if token_score >= threshold:
                        results.append((part, token_score))

        # Remove duplicates and sort by confidence
        unique_results = {}
        for part, score in results:
            part_id = part.id
            if part_id not in unique_results or score > unique_results[part_id][1]:
                unique_results[part_id] = (part, score)

        return sorted(unique_results.values(), key=lambda x: x[1], reverse=True)

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

    # ======== Placeholder Methods for Legacy Compatibility ========

    def comprehensive_drawing_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive drawing search placeholder.

        Note: This would need to be implemented based on your drawing model structure.
        """
        return {
            'status': 'info',
            'message': 'Drawing search not yet implemented in this version',
            'entity_type': 'drawing',
            'search_type': 'comprehensive'
        }

    def comprehensive_tool_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive tool search placeholder.

        Note: This would need to be implemented based on your tool model structure.
        """
        return {
            'status': 'info',
            'message': 'Tool search not yet implemented in this version',
            'entity_type': 'tool',
            'search_type': 'comprehensive'
        }

    def comprehensive_problem_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive problem search placeholder.

        Note: This would need to be implemented based on your problem model structure.
        """
        return {
            'status': 'info',
            'message': 'Problem search not yet implemented in this version',
            'entity_type': 'problem',
            'search_type': 'comprehensive'
        }

    def comprehensive_task_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive task search placeholder.

        Note: This would need to be implemented based on your task model structure.
        """
        return {
            'status': 'info',
            'message': 'Task search not yet implemented in this version',
            'entity_type': 'task',
            'search_type': 'comprehensive'
        }

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
                print(f"✅ Mapped {analysis_key} -> {part_key}: {analysis_params[analysis_key]}")

        # Handle extracted_id as part_id
        if 'extracted_id' in analysis_params:
            part_search_params['part_id'] = analysis_params['extracted_id']
            print(f"✅ Mapped extracted_id -> part_id: {analysis_params['extracted_id']}")

        # Set defaults
        part_search_params['limit'] = original_params.get('limit', 20)
        part_search_params['exact_match'] = False

        return part_search_params
