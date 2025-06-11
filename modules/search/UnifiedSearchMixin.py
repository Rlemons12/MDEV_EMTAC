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
    """
    Unified search interface that provides comprehensive, organized results
    for manufacturing and maintenance queries.

    This mixin adds unified search capabilities to AistManager, providing:
    - Automatic query type detection (search vs conversational)
    - Organized results by entity type (images, documents, parts, etc.)
    - Quick actions and related search suggestions
    - Natural language result summaries

    Designed to work seamlessly with existing AistManager functionality.
    """

    def __init__(self):
        # Initialize the comprehensive unified search system
        self.unified_search_system = None
        self.search_pattern_manager = None
        self._init_unified_search()

        # Query patterns that indicate unified search intent
        self.unified_search_patterns = [
            # Location-based queries
            r'what[\'s\s]+in\s+(room|area|zone|section|location)\s*([A-Z0-9]+)',
            r'show\s+me\s+(everything|all|what[\'s]*)\s+in\s+([A-Z0-9\s]+)',
            r'list\s+(contents|items|equipment)\s+(in|at|from)\s+([A-Z0-9\s]+)',

            # Part-based queries (improved to handle descriptions)
            r'(?:i\s+)?need\s+(?:the\s+)?part\s+number\s+for\s+(.+)',  # "I need the part number for BEARING ASSEMBLY"
            r'what\s+(?:is\s+)?(?:the\s+)?part\s+number\s+for\s+(.+)',  # "what is the part number for..."
            r'part\s+number\s+for\s+(.+)',  # "part number for BEARING ASSEMBLY"
            r'what\s+does\s+part\s+([A-Za-z0-9\-\.]+)\s+look\s+like',
            r'show\s+(me\s+)?part\s+([A-Za-z0-9\-\.]+)',
            r'find\s+part\s+([A-Za-z0-9\-\.]+)',
            r'search\s+for\s+part\s+([A-Za-z0-9\-\.]+)',
            r'where\s+is\s+part\s+([A-Za-z0-9\-\.]+)\s+used',
            r'part\s+([A-Za-z0-9\-\.]+)',  # Simple "part A115957"
            r'([A-Za-z]\d{5,})',  # Pattern like "A115957" directly
            r'(\d{6,})',  # 6+ digit numbers that might be part numbers

            # Image-based queries
            r'show\s+(me\s+)?(images|pictures|photos)\s+of\s+(.+)',
            r'(images|pictures|photos)\s+(of|from|in)\s+(.+)',
            r'what\s+does\s+(.+)\s+look\s+like',

            # Equipment/maintenance queries
            r'(maintenance|repair|procedure)\s+(for|on)\s+(.+)',
            r'how\s+to\s+(fix|repair|maintain)\s+(.+)',
            r'(problems|issues)\s+with\s+(.+)',

            # Document queries
            r'(documents|manuals|guides)\s+(for|about|on)\s+(.+)',
            r'find\s+(documentation|manual)\s+(for|about)\s+(.+)',

            # General search queries
            r'search\s+(for\s+)?(.+)',
            r'find\s+(all\s+)?(.+)',
            r'show\s+(all\s+)?(.+)'
        ]

    # FIX: Move this method OUT of __init__ and to the class level
    def is_unified_search_query(self, question: str) -> bool:
        """
        Determine if the user's question is a unified search query
        rather than a conversational AI query.
        """
        question_lower = question.lower().strip()

        # Check against patterns with case-insensitive matching
        for pattern in self.unified_search_patterns:
            if re.search(pattern, question_lower, re.IGNORECASE):
                logger.debug(f"Matched unified search pattern: {pattern}")
                return True

        # Additional heuristics for unified search queries
        unified_indicators = [
            # Direct search commands
            'show me', 'find', 'search for', 'list', 'display',

            # Question words that typically want structured data
            'what\'s in', 'where is', 'what does', 'how many',

            # Specific entity references
            'part ', 'room ', 'area ', 'zone ', 'equipment ',
            'image', 'picture', 'photo', 'document', 'manual',

            # Location patterns
            'in room', 'in area', 'at location'
        ]

        # Count indicators
        indicator_count = sum(1 for indicator in unified_indicators if indicator in question_lower)

        # If multiple indicators or specific patterns, likely a search query
        if indicator_count >= 2:
            logger.debug(f"Multiple indicators ({indicator_count}) suggest unified search")
            return True

        # Check for specific ID patterns (room numbers, part numbers)
        id_patterns = [
            r'\b(room|area|zone)\s+[A-Z0-9]+\b',
            r'\bpart\s+[A-Za-z0-9\-\.]+\b',
            r'\b[A-Za-z0-9]{2,}\-[A-Za-z0-9]+\b',  # Part number patterns
            r'\b[A-Za-z]\d{5,}\b',  # A115957 style
            r'\b\d{6,}\b'  # 6+ digit numbers
        ]

        for pattern in id_patterns:
            if re.search(pattern, question_lower, re.IGNORECASE):
                logger.debug(f"Matched ID pattern: {pattern}")
                return True

        return False

    def _init_unified_search(self):
        """Initialize the unified search system"""
        try:
            # Use the session from the parent AistManager
            session = getattr(self, 'db_session', None)

            # Initialize the NLP-enhanced search system
            self.unified_search_system = SpaCyEnhancedAggregateSearch(session=session)
            logger.info("Unified search system initialized successfully")

            # Initialize pattern manager
            try:
                self.search_pattern_manager = SearchPatternManager(session=session)
                logger.debug("Search pattern manager initialized")
            except Exception as e:
                logger.warning(f"Search pattern manager initialization failed: {e}")
                self.search_pattern_manager = None

        except Exception as e:
            logger.error(f"Failed to initialize unified search system: {e}")
            # Fallback to basic aggregate search
            try:
                session = getattr(self, 'db_session', None)
                self.unified_search_system = AggregateSearch(session=session)
                logger.warning("Using basic aggregate search as fallback")
            except Exception as fallback_error:
                logger.error(f"Fallback search system also failed: {fallback_error}")
                self.unified_search_system = None

    def is_unified_search_query(self, question: str) -> bool:
        """
        Determine if the user's question is a unified search query
        rather than a conversational AI query.
        """
        question_lower = question.lower().strip()

        # Check against patterns
        for pattern in self.unified_search_patterns:
            if re.search(pattern, question_lower):
                logger.debug(f"Matched unified search pattern: {pattern}")
                return True

        # Additional heuristics for unified search queries
        unified_indicators = [
            # Direct search commands
            'show me', 'find', 'search for', 'list', 'display',

            # Question words that typically want structured data
            'what\'s in', 'where is', 'what does', 'how many',

            # Specific entity references
            'part ', 'room ', 'area ', 'zone ', 'equipment ',
            'image', 'picture', 'photo', 'document', 'manual',

            # Location patterns
            'in room', 'in area', 'at location'
        ]

        # Count indicators
        indicator_count = sum(1 for indicator in unified_indicators if indicator in question_lower)

        # If multiple indicators or specific patterns, likely a search query
        if indicator_count >= 2:
            return True

        # Check for specific ID patterns (room numbers, part numbers)
        id_patterns = [
            r'\b(room|area|zone)\s+[A-Z0-9]+\b',
            r'\bpart\s+[A-Z0-9\-\.]+\b',
            r'\b[A-Z0-9]{2,}\-[A-Z0-9]+\b'  # Part number patterns
        ]

        for pattern in id_patterns:
            if re.search(pattern, question_lower):
                return True

        return False

    def execute_unified_search(self, question: str, user_id: str = None, request_id: str = None) -> Dict[str, Any]:
        """
        REPLACE this method in UnifiedSearchMixin to force your fixed method
        """
        search_start = time.time()
        logger.info(f"🔍 Executing unified search for: {question}")

        if not self.unified_search_system:
            return self._search_system_unavailable_response()

        # FORCE DIRECT PART SEARCH for part number queries
        question_lower = question.lower()
        is_part_number_query = any(phrase in question_lower for phrase in [
            'part number for', 'what is the part number', 'part number'
        ])

        if is_part_number_query:
            logger.error(f"🚨 INTERCEPTED PART NUMBER QUERY - BYPASSING BROKEN ROUTING")

            # Extract description using the SAME logic that already works
            desc_patterns = [
                r'part\s+number\s+for\s+(.+?)(?:\s*$|\s*\?)',
                r'what\s+(?:is\s+)?(?:the\s+)?part\s+number\s+for\s+(.+?)(?:\s*$|\s*\?)',
            ]

            description = None
            for pattern in desc_patterns:
                match = re.search(pattern, question_lower, re.IGNORECASE)
                if match:
                    description = match.group(1).strip()
                    break

            if description:
                logger.error(f"🎯 BYPASSING TO DIRECT PART SEARCH: '{description}'")

                # Build the exact params your fixed method expects
                direct_params = {
                    'search_text': description,
                    'entity_type': 'part',
                    'fields': ['name', 'part_number', 'oem_mfg', 'model', 'notes'],
                    'limit': 20,
                    'raw_input': question,
                    'extraction_method': 'bypass_direct_call'
                }

                # CALL YOUR FIXED METHOD DIRECTLY
                try:
                    # Get the aggregate search instance
                    if hasattr(self.unified_search_system, '_aggregate_search'):
                        aggregate_search = self.unified_search_system._aggregate_search
                    elif hasattr(self, 'unified_search_system') and hasattr(self.unified_search_system,
                                                                            'comprehensive_part_search'):
                        aggregate_search = self.unified_search_system
                    else:
                        logger.error(f"❌ Cannot find aggregate search instance")
                        return self._unified_search_error_response(question, "Cannot find search instance")

                    if hasattr(aggregate_search, 'comprehensive_part_search'):
                        logger.error(f"✅ CALLING YOUR FIXED comprehensive_part_search DIRECTLY")

                        # This should trigger your "🎯 DIRECT PART SEARCH for description" log
                        search_result = aggregate_search.comprehensive_part_search(direct_params)

                        logger.error(
                            f"🎉 BYPASS RESULT: {search_result.get('status')} with {search_result.get('count', 0)} results")

                        if search_result and search_result.get('status') == 'success':
                            # Use your existing result enhancement
                            enhanced_result = self._enhance_unified_search_results(search_result, question)

                            # Add timing
                            search_time = time.time() - search_start
                            enhanced_result['search_time_ms'] = int(search_time * 1000)
                            enhanced_result['bypass_method'] = 'direct_comprehensive_part_search'

                            logger.error(f"🚀 BYPASS SUCCESS: {enhanced_result.get('total_results', 0)} total results")
                            return enhanced_result
                        else:
                            logger.error(f"❌ Fixed method returned error: {search_result}")
                            return self._no_unified_results_response(question, search_result)

                    else:
                        logger.error(f"❌ comprehensive_part_search method not found")
                        return self._unified_search_error_response(question, "Fixed method not found")

                except Exception as e:
                    logger.error(f"❌ Direct method call failed: {e}", exc_info=True)
                    return self._unified_search_error_response(question, f"Direct call failed: {str(e)}")

        # For non-part queries, use normal search
        try:
            logger.info(f"🔄 Using normal unified search for non-part query")

            if hasattr(self.unified_search_system, 'execute_nlp_aggregated_search'):
                search_result = self.unified_search_system.execute_nlp_aggregated_search(question)
            else:
                search_result = self.unified_search_system.execute_aggregated_search(question)

            if search_result and search_result.get('status') == 'success':
                enhanced_result = self._enhance_unified_search_results(search_result, question)
                search_time = time.time() - search_start
                enhanced_result['search_time_ms'] = int(search_time * 1000)
                return enhanced_result
            else:
                return self._no_unified_results_response(question, search_result)

        except Exception as e:
            search_time = time.time() - search_start
            logger.error(f"❌ Normal search failed after {search_time:.3f}s: {e}", exc_info=True)
            return self._unified_search_error_response(question, str(e))

    ## ALTERNATIVE: Quick test to verify your method exists and works

    def test_your_fixed_method_now(self):
        """
        Add this method to test your comprehensive_part_search directly
        """
        logger.error(f"🧪 TESTING YOUR FIXED METHOD DIRECTLY")

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
                logger.error(f"✅ Found comprehensive_part_search on self")
                result = self.comprehensive_part_search(test_params)
            elif hasattr(self, 'unified_search_system') and hasattr(self.unified_search_system, '_aggregate_search'):
                logger.error(f"✅ Found comprehensive_part_search on aggregate_search")
                result = self.unified_search_system._aggregate_search.comprehensive_part_search(test_params)
            else:
                logger.error(f"❌ Cannot find comprehensive_part_search method")
                return {"status": "error", "message": "Method not found"}

            logger.error(f"🧪 TEST RESULT: {result.get('status')} with {result.get('count', 0)} results")

            if result.get('results'):
                for i, part in enumerate(result['results'][:3]):
                    pn = part.get('part_number', 'No PN')
                    name = part.get('name', 'No name')
                    logger.error(f"  Part {i + 1}: {pn} - {name}")

            return result

        except Exception as e:
            logger.error(f"🧪 TEST FAILED: {e}", exc_info=True)
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

        logger.debug(f"📊 Organizing {len(results)} results...")

        for i, result in enumerate(results):
            logger.debug(
                f"Result {i + 1}: {type(result)} - Keys: {list(result.keys()) if isinstance(result, dict) else 'Not dict'}")

            # Check if this is a POSITION object with embedded parts
            if isinstance(result, dict) and 'parts' in result and 'area' in result:
                logger.debug(f"  🏗️  POSITION object detected with {result.get('part_count', 0)} parts")

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
                                logger.debug(f"    ✅ Extracted PART: {formatted_part['part_number']}")
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
                                logger.debug(f"    ✅ Extracted PART (object): {formatted_part['part_number']}")
                        except Exception as e:
                            logger.warning(f"    ⚠️  Failed to extract part {j + 1}: {e}")
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
                    logger.debug(f"  ✅ Direct PART: {formatted_part['part_number']}")
                except Exception as e:
                    logger.warning(f"  ⚠️  Failed to format direct part: {e}")

            # Check for images
            elif isinstance(result, dict) and any(field in result for field in ['image_id', 'file_path']) or str(
                    result.get('type', '')).lower() in ['image', 'picture']:
                try:
                    image_info = self._format_image_result(result)
                    organized['images'].append(image_info)
                    logger.debug(f"  🖼️  Added IMAGE")
                except Exception as e:
                    logger.warning(f"  ⚠️  Failed to format image: {e}")

            # Everything else goes to other
            else:
                organized['other'].append(result)
                logger.debug(f"  ❓ Added to OTHER")

        # Log final counts
        for category, items in organized.items():
            if items:
                logger.info(f"📊 {category.upper()}: {len(items)} items")
                if category == 'parts' and items:
                    # Log first few part numbers for verification
                    part_numbers = [p.get('part_number', 'Unknown') for p in items[:3]]
                    logger.info(f"📊 Sample part numbers: {part_numbers}")

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
                logger.info(f"🔍 Direct part search found {len(parts)} parts")
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
            logging.info(f"🔍 Unified search: '{question}' -> {result.get('status')} ({result.get('count', 0)} results)")
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
                # Check if we've cached this term
                cache_key = term.lower()
                if cache_key in self._synonym_cache:
                    expanded_terms[term] = self._synonym_cache[cache_key]
                    continue

                # Query the synonym database with proper error handling
                synonyms = self._get_synonyms_for_term_safe(term)
                self._synonym_cache[cache_key] = synonyms
                expanded_terms[term] = synonyms

            return expanded_terms

        def _get_synonyms_for_term_safe(self, term: str) -> List[str]:
            """FIXED: Get synonyms with proper SQL syntax."""

            # Try Method 1: Fixed PostgreSQL query without ORDER BY issues
            try:
                return self._get_synonyms_postgresql_fixed(term)
            except Exception as e:
                logger.debug(f"PostgreSQL method failed for '{term}': {e}")

            # Try Method 2: Simple query without ORDER BY
            try:
                return self._get_synonyms_simple_query(term)
            except Exception as e:
                logger.debug(f"Simple query method failed for '{term}': {e}")

            # Fallback: Use hardcoded synonyms
            return self._get_synonyms_fallback(term)

        def _get_synonyms_postgresql_fixed(self, term: str) -> List[str]:
            """FIXED: PostgreSQL query without ORDER BY issues."""
            try:
                # Ensure we have a fresh transaction
                if self.session.in_transaction():
                    self.session.rollback()

                # FIXED: Include ORDER BY columns in SELECT to avoid PostgreSQL error
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

                # Collect all related terms
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
                    logger.debug(f"PostgreSQL synonyms for '{term}': {synonyms_list}")
                    return synonyms_list
                else:
                    logger.debug(f"No PostgreSQL synonyms found for '{term}'")
                    return []

            except Exception as e:
                logger.warning(f"PostgreSQL synonym lookup failed for {term}: {e}")
                # Rollback the transaction
                try:
                    self.session.rollback()
                except:
                    pass
                raise

        def _get_synonyms_sqlalchemy_core(self, term: str) -> List[str]:
            """Alternative method using SQLAlchemy Core with proper error handling"""
            try:
                # Ensure fresh transaction
                if self.session.in_transaction():
                    self.session.rollback()

                # Reflect the tables
                metadata = MetaData()
                entity_synonym = Table('entity_synonym', metadata, autoload_with=self.session.bind)
                entity_type = Table('entity_type', metadata, autoload_with=self.session.bind)

                # Build the query without ORDER BY to avoid issues
                query = select(
                    entity_synonym.c.canonical_value,
                    entity_synonym.c.synonym_value
                ).select_from(
                    entity_synonym.join(entity_type, entity_synonym.c.entity_type_id == entity_type.c.id)
                ).where(
                    and_(
                        or_(
                            entity_synonym.c.canonical_value.ilike(f'%{term}%'),
                            entity_synonym.c.synonym_value.ilike(f'%{term}%')
                        ),
                        entity_type.c.name == 'EQUIPMENT_TYPE',
                        entity_synonym.c.confidence_score > 0.5
                    )
                ).limit(10)

                result = self.session.execute(query)

                # Collect related terms
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
                    logger.debug(f"SQLAlchemy Core synonyms for '{term}': {synonyms_list}")
                else:
                    logger.debug(f"No SQLAlchemy Core synonyms found for '{term}'")

                return synonyms_list

            except Exception as e:
                logger.warning(f"SQLAlchemy Core synonym lookup failed for {term}: {e}")
                # Rollback the transaction
                try:
                    self.session.rollback()
                except:
                    pass
                raise

        def _get_synonyms_simple_query(self, term: str) -> List[str]:
            """FIXED: Simple query without ORDER BY."""
            try:
                # Ensure fresh transaction
                if self.session.in_transaction():
                    self.session.rollback()

                sql_query = text("""
                    SELECT es.canonical_value, es.synonym_value
                    FROM entity_synonym es
                    JOIN entity_type et ON es.entity_type_id = et.id
                    WHERE (
                        LOWER(es.canonical_value) LIKE :term OR 
                        LOWER(es.synonym_value) LIKE :term
                    )
                    AND et.name = 'EQUIPMENT_TYPE'
                    LIMIT 10
                """)

                result = self.session.execute(sql_query, {'term': f'%{term.lower()}%'})

                # Collect all related terms
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
                    logger.debug(f"Simple query synonyms for '{term}': {synonyms_list}")
                else:
                    logger.debug(f"No simple query synonyms found for '{term}'")

                return synonyms_list

            except Exception as e:
                logger.warning(f"Simple query synonym lookup failed for {term}: {e}")
                # Rollback the transaction
                try:
                    self.session.rollback()
                except:
                    pass
                raise

        def _get_synonyms_fallback(self, term: str) -> List[str]:
            """Fallback method with hardcoded synonyms based on your data"""

            logger.debug(f"Using fallback synonyms for '{term}'")

            # Based on your synonym database structure
            fallback_synonyms = {
                'valve': ['valves', 'control valve', 'ball valve', 'gate valve', 'check valve', 'relief valve'],
                'bearing': ['bearings', 'ball bearing', 'roller bearing', 'thrust bearing', 'bearing assembly'],
                'switch': ['switches', 'limit switch', 'pressure switch', 'temperature switch', 'safety switch'],
                'motor': ['motors', 'electric motor', 'ac motor', 'dc motor', 'servo motor'],
                'belt': ['belts', 'drive belt', 'v-belt', 'timing belt', 'serpentine belt'],
                'cable': ['cables', 'power cable', 'control cable', 'data cable'],
                'sensor': ['sensors', 'temperature sensor', 'pressure sensor', 'level sensor'],
                'seal': ['seals', 'oil seal', 'shaft seal', 'hydraulic seal'],
                'relay': ['relays', 'control relay', 'time relay', 'power relay'],
                'pump': ['pumps', 'centrifugal pump', 'hydraulic pump', 'water pump'],
                'spring': ['springs', 'compression spring', 'extension spring'],
                'filter': ['filters', 'air filter', 'oil filter', 'hydraulic filter'],
                'gear': ['gears', 'spur gear', 'bevel gear'],
                'tube': ['tubes', 'hydraulic tube'],
                'hose': ['hoses', 'hydraulic hose', 'air hose'],
                'wire': ['wires', 'electrical wire'],
                'fan': ['fans', 'cooling fan', 'exhaust fan'],

                # Equipment types that might appear in part searches
                'assembly': ['assemblies', 'unit', 'component'],
                'component': ['components', 'part', 'piece'],
                'unit': ['units', 'assembly', 'module']
            }

            term_lower = term.lower()

            # Exact match
            if term_lower in fallback_synonyms:
                synonyms = fallback_synonyms[term_lower][:4]
                logger.debug(f"Fallback exact match synonyms for '{term}': {synonyms}")
                return synonyms

            # Partial match (for compound terms like "ball valve")
            for key, values in fallback_synonyms.items():
                if key in term_lower or term_lower in key:
                    synonyms = values[:3]
                    logger.debug(f"Fallback partial match synonyms for '{term}' (matched '{key}'): {synonyms}")
                    return synonyms

            # No synonyms found
            logger.debug(f"No fallback synonyms found for '{term}'")
            return []

        def clear_cache(self):
            """Clear the synonym cache"""
            cache_size = len(self._synonym_cache)
            self._synonym_cache.clear()
            logger.debug(f"Cleared synonym cache ({cache_size} entries)")
            return cache_size

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
            logger.info("🔄 Position search didn't return actual part numbers, trying direct part search...")

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
                    logger.info(f"✅ Direct search found {len(direct_parts)} actual parts")

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

