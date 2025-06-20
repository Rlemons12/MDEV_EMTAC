# modules/ai/ai_steward.py
import time
from sqlalchemy import text
import re
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from sqlalchemy.ext.declarative import declarative_base
from modules.configuration.log_config import logger, with_request_id, log_timed_operation
from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import (
    Position, CompleteDocument, Image, KeywordSearch, ChatSession, User,
    QandA, Drawing, CompletedDocumentPositionAssociation,
    KeywordAction, Document, DocumentEmbedding, ImageEmbedding,
    PartsPositionImageAssociation, ImageCompletedDocumentAssociation,
    ImagePositionAssociation, ToolImageAssociation, ImageTaskAssociation,
    ImageProblemAssociation, Part
)
from plugins.ai_modules.ai_models import ModelsConfig
import threading
from collections import defaultdict
from modules.search.UnifiedSearchMixin import UnifiedSearchMixin


Base = declarative_base()


def get_request_id():
    """Helper function to get request ID from context or generate one"""
    try:
        from modules.configuration.log_config import get_current_request_id
        return get_current_request_id()
    except:
        import uuid
        return str(uuid.uuid4())[:8]


import time
import threading
import json
from modules.emtacdb.emtacdb_fts import DocumentEmbedding, Document
from modules.configuration.log_config import logger, get_request_id


class VectorSearchClient:
    """Optimized VectorSearchClient using pgvector for high-performance similarity search with detailed timing."""

    def __init__(self):
        self.np = __import__('numpy')
        self.embedding_cache = {}
        self.cache_expiry = {}
        self.cache_max_size = 1000
        self.cache_ttl = 3600
        logger.debug("pgvector-compatible vector search client initialized")

    def search(self, query, limit=5, threshold=0.2):
        """Vector search using pgvector for optimal performance with detailed timing."""
        search_start = time.time()
        request_id = get_request_id()

        logger.info(
            f"[TIMING] Starting vector search for query: '{query[:50]}...' (limit={limit}, threshold={threshold})")

        try:
            result = [None]
            exception = [None]

            def search_worker():
                try:
                    result[0] = self._perform_search(query, limit, threshold)
                except Exception as e:
                    exception[0] = e

            worker_start = time.time()
            search_thread = threading.Thread(target=search_worker)
            search_thread.daemon = True
            search_thread.start()

            logger.info(f"[TIMING] Search thread started in {time.time() - worker_start:.3f}s")

            join_start = time.time()
            search_thread.join(timeout=5.0)  # Reduced from 10s to 5s
            join_time = time.time() - join_start

            logger.info(f"[TIMING] Thread join completed in {join_time:.3f}s")

            if search_thread.is_alive():
                logger.warning(f"[TIMING] Vector search timed out after 5 seconds for query: {query[:50]}...")
                return []

            if exception[0]:
                raise exception[0]

            results = result[0] if result[0] else []
            total_time = time.time() - search_start

            logger.info(f"[TIMING] Vector search COMPLETED: {total_time:.3f}s total, found {len(results)} results")
            return results

        except Exception as e:
            error_time = time.time() - search_start
            logger.error(f"[TIMING] Error in vector search after {error_time:.3f}s: {e}", exc_info=True)
            return []

    def _perform_search(self, query, limit, threshold=0.3):
        """High-performance search using pgvector similarity operators with detailed timing."""
        method_start = time.time()
        request_id = get_request_id()

        logger.info(f"[TIMING] _perform_search started")

        from modules.configuration.config_env import DatabaseConfig
        from plugins.ai_modules import generate_embedding, ModelsConfig
        from sqlalchemy import text

        # 1. Database setup timing
        db_setup_start = time.time()
        db_config = DatabaseConfig()
        session = db_config.get_main_session()
        db_setup_time = time.time() - db_setup_start
        logger.info(f"[TIMING] Database setup: {db_setup_time:.3f}s")

        try:
            # 2. Model config timing
            config_start = time.time()
            current_model = ModelsConfig.get_config_value('embedding', 'CURRENT_MODEL', 'OpenAIEmbeddingModel')
            config_time = time.time() - config_start
            logger.info(f"[TIMING] Model config retrieval: {config_time:.3f}s")

            # 3. Embedding generation timing
            embedding_start = time.time()
            query_embedding = generate_embedding(query, current_model)
            embedding_time = time.time() - embedding_start
            logger.info(f"[TIMING] Embedding generation: {embedding_time:.3f}s")

            if not query_embedding:
                logger.warning("[TIMING] Failed to generate embedding for query")
                return []

            # 4. Vector string preparation timing
            vector_prep_start = time.time()
            query_vector_str = '[' + ','.join(map(str, query_embedding)) + ']'
            vector_prep_time = time.time() - vector_prep_start
            logger.info(f"[TIMING] Vector string preparation: {vector_prep_time:.3f}s")

            # 5. SQL query preparation timing
            query_prep_start = time.time()
            similarity_query = text("""
                SELECT 
                    de.document_id,
                    1 - (de.embedding_vector <=> :query_vector) AS similarity,
                    d.content,
                    cd.title as doc_title,
                    cd.file_path,
                    cd.rev
                FROM document_embedding de
                LEFT JOIN document d ON de.document_id = d.id
                LEFT JOIN complete_document cd ON d.complete_document_id = cd.id
                WHERE de.model_name = :model_name
                  AND de.embedding_vector IS NOT NULL
                  AND (1 - (de.embedding_vector <=> :query_vector)) >= :threshold
                ORDER BY de.embedding_vector <=> :query_vector ASC
                LIMIT :limit
            """)
            query_prep_time = time.time() - query_prep_start
            logger.info(f"[TIMING] SQL query preparation: {query_prep_time:.3f}s")

            # 6. Database execution timing
            db_exec_start = time.time()
            result = session.execute(similarity_query, {
                'query_vector': query_vector_str,
                'model_name': current_model,
                'threshold': threshold,
                'limit': limit
            })
            db_exec_time = time.time() - db_exec_start
            logger.info(f"[TIMING] DATABASE QUERY EXECUTION: {db_exec_time:.3f}s")

            # 7. Results processing timing
            processing_start = time.time()
            top_docs = []
            row_count = 0

            for row in result:
                row_start = time.time()
                doc_id, similarity, content, doc_title, file_path, rev = row
                row_count += 1

                display_content = content or doc_title or f"Document {doc_id}"

                if display_content and len(display_content) > 1000:
                    truncated = display_content[:1000]
                    last_period = truncated.rfind('.')
                    if last_period > 800:
                        display_content = truncated[:last_period + 1]
                    else:
                        display_content = truncated + "..."

                top_docs.append({
                    "id": doc_id,
                    "content": display_content,
                    "similarity": float(similarity),
                    "title": doc_title,
                    "file_path": file_path,
                    "revision": rev,
                    "source": f"Document {doc_id}",
                    "model_name": current_model
                })

                row_time = time.time() - row_start
                if row_count <= 5:  # Log first 5 rows
                    logger.debug(f"[TIMING] Row {row_count} processing: {row_time:.3f}s")

            processing_time = time.time() - processing_start
            logger.info(f"[TIMING] Results processing: {processing_time:.3f}s ({row_count} rows)")

            # 8. Total method timing
            total_method_time = time.time() - method_start
            logger.info(f"[TIMING] _perform_search TOTAL: {total_method_time:.3f}s")

            # Breakdown summary
            logger.info(f"[TIMING] BREAKDOWN - DB Setup: {db_setup_time:.3f}s, "
                        f"Config: {config_time:.3f}s, "
                        f"Embedding: {embedding_time:.3f}s, "
                        f"Vector Prep: {vector_prep_time:.3f}s, "
                        f"Query Prep: {query_prep_time:.3f}s, "
                        f"DB Execution: {db_exec_time:.3f}s, "
                        f"Processing: {processing_time:.3f}s")

            logger.info(f"Vector search found {len(top_docs)} results (threshold: {threshold})")
            return top_docs

        except Exception as e:
            error_time = time.time() - method_start
            logger.error(f"[TIMING] pgvector search failed after {error_time:.3f}s: {e}", exc_info=True)
            return self._legacy_search_fallback(query, limit, session)
        finally:
            session_close_start = time.time()
            session.close()
            session_close_time = time.time() - session_close_start
            logger.info(f"[TIMING] Session close: {session_close_time:.3f}s")

    def _legacy_search_fallback(self, query, limit, session):
        """Fallback to legacy search if pgvector fails with timing."""
        fallback_start = time.time()
        logger.info(f"[TIMING] Starting legacy search fallback")

        try:
            logger.info("Using legacy vector search fallback")

            # Model loading timing
            model_start = time.time()
            embedding_model = ModelsConfig.load_embedding_model()
            model_time = time.time() - model_start
            logger.info(f"[TIMING] Legacy model loading: {model_time:.3f}s")

            # Embedding generation timing
            embed_start = time.time()
            query_embedding = embedding_model.get_embeddings(query)
            embed_time = time.time() - embed_start
            logger.info(f"[TIMING] Legacy embedding generation: {embed_time:.3f}s")

            if not query_embedding:
                return []

            # Vector preparation timing
            prep_start = time.time()
            query_embedding_np = self.np.array(query_embedding)
            query_norm = self.np.linalg.norm(query_embedding_np)
            prep_time = time.time() - prep_start
            logger.info(f"[TIMING] Legacy vector preparation: {prep_time:.3f}s")

            if query_norm == 0:
                return []

            # Database query timing
            db_query_start = time.time()
            doc_embeddings = session.query(
                DocumentEmbedding.document_id,
                DocumentEmbedding.model_embedding
            ).filter(
                DocumentEmbedding.model_embedding.isnot(None)
            ).limit(1000).all()
            db_query_time = time.time() - db_query_start
            logger.info(f"[TIMING] Legacy database query: {db_query_time:.3f}s")

            # Similarity calculation timing
            similarity_start = time.time()
            similarities = []
            processed_count = 0

            for doc_id, doc_embedding_raw in doc_embeddings:
                try:
                    if isinstance(doc_embedding_raw, bytes):
                        try:
                            doc_embedding = self.np.array(json.loads(doc_embedding_raw.decode('utf-8')))
                        except:
                            doc_embedding = self.np.frombuffer(doc_embedding_raw, dtype=self.np.float32)
                    else:
                        doc_embedding = self.np.array(doc_embedding_raw)

                    doc_norm = self.np.linalg.norm(doc_embedding)
                    if doc_norm > 0:
                        similarity = self.np.dot(query_embedding_np, doc_embedding) / (query_norm * doc_norm)
                        if similarity > 0.3:
                            similarities.append((doc_id, float(similarity)))

                    processed_count += 1
                    if processed_count % 100 == 0:
                        logger.debug(f"[TIMING] Legacy processed {processed_count} documents")

                except Exception as e:
                    logger.debug(f"Error processing legacy embedding for document {doc_id}: {e}")

            similarity_time = time.time() - similarity_start
            logger.info(f"[TIMING] Legacy similarity calculation: {similarity_time:.3f}s ({processed_count} docs)")

            # Sorting and final processing timing
            final_start = time.time()
            similarities.sort(key=lambda x: x[1], reverse=True)

            top_docs = []
            for doc_id, similarity in similarities[:limit]:
                doc = session.query(Document).get(doc_id)
                if doc:
                    content = doc.content or ""
                    snippet = content[:1000] + "..." if len(content) > 1000 else content

                    top_docs.append({
                        "id": doc_id,
                        "content": snippet,
                        "similarity": similarity,
                        "source": f"Document {doc_id} (legacy)",
                        "method": "legacy_fallback"
                    })

            final_time = time.time() - final_start
            total_fallback_time = time.time() - fallback_start

            logger.info(f"[TIMING] Legacy final processing: {final_time:.3f}s")
            logger.info(f"[TIMING] Legacy fallback TOTAL: {total_fallback_time:.3f}s")

            return top_docs

        except Exception as e:
            error_time = time.time() - fallback_start
            logger.error(f"[TIMING] Legacy search fallback failed after {error_time:.3f}s: {e}")
            return []

    def search_with_adaptive_threshold(self, query, limit=5, min_results=1):
        """Adaptive search that automatically adjusts threshold to find results with timing."""
        adaptive_start = time.time()
        logger.info(f"[TIMING] Starting adaptive threshold search")

        thresholds = [0.7, 0.5, 0.3, 0.1, 0.0]

        for i, threshold in enumerate(thresholds):
            threshold_start = time.time()
            results = self.search(query, limit, threshold)
            threshold_time = time.time() - threshold_start

            logger.info(
                f"[TIMING] Threshold {threshold} attempt {i + 1}: {threshold_time:.3f}s, {len(results)} results")

            if len(results) >= min_results:
                total_adaptive_time = time.time() - adaptive_start
                logger.info(f"[TIMING] Adaptive search completed in {total_adaptive_time:.3f}s after {i + 1} attempts")
                return results

        total_adaptive_time = time.time() - adaptive_start
        logger.info(f"[TIMING] Adaptive search completed with no results in {total_adaptive_time:.3f}s")
        return []

class ResponseFormatter:
    """Utility class for formatting search responses."""

    @staticmethod
    def format_search_results(result):
        """Format search results into a user-friendly response."""
        try:
            if not result or not isinstance(result, dict):
                return "I couldn't find relevant information for your query."

            if result.get('status') != 'success':
                error_msg = result.get('message', 'Search failed')
                return f"Search error: {error_msg}"

            # NEW: Handle AI-generated knowledge query responses
            if 'answer' in result and result.get('method') in ['ai_knowledge_synthesis_with_chunks',
                                                               'ai_knowledge_synthesis_direct']:
                ai_answer = result['answer']

                # Add source information if available
                if 'source_info' in result:
                    source_info = result['source_info']
                    if source_info.get('document_source') and source_info.get('chunk_similarity'):
                        similarity = source_info['chunk_similarity']
                        doc_source = source_info['document_source']
                        ai_answer += f"\n\n*Source: {doc_source} (Similarity: {similarity:.1%})*"
                    elif source_info.get('source_type') == 'ai_general_knowledge':
                        ai_answer += f"\n\n*Source: AI General Knowledge*"

                return ai_answer

            # Existing search result formatting logic
            total_results = result.get('total_results', 0)
            if total_results == 0:
                return "No results found for your query."

            # Check for organized results structure
            if 'organized_results' in result:
                return ResponseFormatter._format_organized_results(result['organized_results'], total_results)
            elif 'results_by_type' in result:
                return ResponseFormatter._format_results_by_type(result['results_by_type'], total_results)
            elif 'results' in result and isinstance(result['results'], list):
                return ResponseFormatter._format_direct_results(result['results'], total_results)
            elif total_results > 0:
                return ResponseFormatter._format_main_result_structure(result, total_results)

            return f"Found {total_results} results for your query."

        except Exception as e:
            logger.error(f"Error formatting search results: {e}", exc_info=True)
            return "Found some results, but had trouble formatting them."

    @staticmethod
    def _format_organized_results(organized_results, total_results):
        """Format organized results structure."""
        parts = []

        if 'parts' in organized_results and organized_results['parts']:
            parts_list = organized_results['parts'][:10]
            parts.append(f"Found {len(parts_list)} Banner sensor{'s' if len(parts_list) != 1 else ''}:")

            for i, part in enumerate(parts_list, 1):
                part_info = f"{i}. {part.get('part_number', 'Unknown')}"
                if part.get('name'):
                    part_info += f" - {part.get('name')}"
                if part.get('oem_mfg'):
                    part_info += f" (Manufacturer: {part['oem_mfg']})"
                parts.append(part_info)

        if 'images' in organized_results and organized_results['images']:
            image_count = len(organized_results['images'])
            parts.append(f"\nFound {image_count} related image{'s' if image_count != 1 else ''}.")

        if 'positions' in organized_results and organized_results['positions']:
            position_count = len(organized_results['positions'])
            parts.append(f"\nFound {position_count} installation location{'s' if position_count != 1 else ''}.")

        return "\n".join(parts) if parts else f"Found {total_results} results for your query."

    @staticmethod
    def _format_results_by_type(results_by_type, total_results):
        """Format results_by_type structure."""
        response_parts = []

        # Handle parts
        if 'parts' in results_by_type and results_by_type['parts']:
            parts_list = results_by_type['parts'][:10]
            response_parts.append(f"Found {len(parts_list)} part{'s' if len(parts_list) != 1 else ''}:")

            for i, part in enumerate(parts_list, 1):
                part_info = f"{i}. {part.get('part_number', 'Unknown')}"
                if part.get('name'):
                    part_info += f" - {part.get('name')}"
                if part.get('oem_mfg'):
                    part_info += f" (Manufacturer: {part['oem_mfg']})"
                response_parts.append(part_info)

        # Handle other types
        for result_type, results in results_by_type.items():
            if result_type == 'parts' or not results:
                continue

            if response_parts:
                response_parts.append("")

            type_name = result_type.replace('_', ' ').title()
            response_parts.append(f"Found {len(results)} {type_name}:")

            for i, item in enumerate(results[:5], 1):
                item_info = f"{i}. {item.get('title', item.get('name', 'Unknown'))}"
                response_parts.append(item_info)

        return "\n".join(response_parts) if response_parts else f"Found {total_results} results."

    @staticmethod
    def _format_direct_results(results, total_results):
        """Format direct results list structure - handles both dicts and SQLAlchemy objects."""

        if not results or not isinstance(results, list):
            return f"Found {total_results} results for your query."

        if len(results) == 0:
            return "No results found for your query."

        response_parts = []
        results_list = results[:10]  # Limit to first 10 results

        response_parts.append(f"Found {len(results_list)} result{'s' if len(results_list) != 1 else ''}:")

        for i, item in enumerate(results_list, 1):
            # Handle SQLAlchemy Part objects
            if hasattr(item, 'part_number'):  # This is a Part object
                part_info = f"{i}. {item.part_number or 'Unknown Part'}"

                # Add name/description if available
                if item.name:
                    part_info += f" - {item.name}"

                # Add manufacturer if available
                if item.oem_mfg:
                    part_info += f" (Manufacturer: {item.oem_mfg})"

                # Add model if available and different from name
                if item.model and item.model != item.name:
                    part_info += f" [Model: {item.model}]"

                response_parts.append(part_info)

            # Handle dictionary objects
            elif isinstance(item, dict):
                if 'part_number' in item:
                    part_info = f"{i}. {item.get('part_number', 'Unknown')}"
                    if item.get('name'):
                        part_info += f" - {item.get('name')}"
                    if item.get('oem_mfg'):
                        part_info += f" (Manufacturer: {item['oem_mfg']})"
                    response_parts.append(part_info)
                else:
                    # General dictionary handling
                    name = (item.get('name') or
                            item.get('title') or
                            item.get('id') or
                            'Unknown')

                    description = (item.get('description') or
                                   item.get('notes') or
                                   item.get('model') or
                                   item.get('oem_mfg') or
                                   '')

                    if description:
                        response_parts.append(f"{i}. {name} - {description}")
                    else:
                        response_parts.append(f"{i}. {name}")
            else:
                # Handle other object types
                response_parts.append(f"{i}. {str(item)}")

        return "\n".join(response_parts)

    @staticmethod
    def _format_main_result_structure(result, total_results):
        """Handle main result structure."""
        response_parts = []

        if 'summary' in result:
            response_parts.append(result['summary'])

        found_results = False
        for key in ['parts', 'results', 'data', 'items']:
            if key in result and isinstance(result[key], list) and result[key]:
                found_results = True
                items_list = result[key][:10]

                if not response_parts:
                    response_parts.append(f"Found {len(items_list)} result{'s' if len(items_list) != 1 else ''}:")

                for i, item in enumerate(items_list, 1):
                    if isinstance(item, dict):
                        if 'part_number' in item:
                            part_info = f"{i}. {item.get('part_number', 'Unknown')}"
                            if item.get('name'):
                                part_info += f" - {item.get('name')}"
                            response_parts.append(part_info)
                        else:
                            name = item.get('name', item.get('title', item.get('id', 'Unknown')))
                            response_parts.append(f"{i}. {name}")
                    else:
                        response_parts.append(f"{i}. {str(item)}")
                break

        if not found_results:
            response_parts.append(f"Found {total_results} results for your query.")

        return "\n".join(response_parts) if response_parts else f"Found {total_results} results."

class SearchUtils:
    """Utility class for search-related operations."""

    @staticmethod
    def execute_with_timeout(func, timeout_seconds=10, *args, **kwargs):
        """Windows-compatible timeout wrapper using threading."""
        result = [None]
        exception = [None]
        completed = [False]

        def worker():
            try:
                result[0] = func(*args, **kwargs)
                completed[0] = True
            except Exception as e:
                exception[0] = e
                completed[0] = True

        worker_thread = threading.Thread(target=worker)
        worker_thread.daemon = True
        worker_thread.start()
        worker_thread.join(timeout=timeout_seconds)

        if not completed[0]:
            logger.warning(f"Operation timed out after {timeout_seconds} seconds")
            return None, "Operation timed out"

        if exception[0]:
            raise exception[0]

        return result[0], None

    @staticmethod
    def extract_search_params(query):
        """Extract search parameters from user query."""
        params = {
            'title': None,
            'area': None,
            'equipment_group': None,
            'model': None,
            'asset_number': None,
            'location': None,
            'description': None,
            'query': query
        }

        # Extract title
        title_match = re.search(r'of\s+(["\'])(.*?)\1', query)
        if title_match:
            params['title'] = title_match.group(2)
        else:
            title_match = re.search(r'of\s+([\w\s]+)(?:$|\s+(?:in|at|for|from))', query)
            if title_match:
                params['title'] = title_match.group(1).strip()

        # Extract area
        area_match = re.search(r'(?:in|of)\s+area\s+(["\'])(.*?)\1', query)
        if area_match:
            params['area'] = area_match.group(2)
        else:
            area_match = re.search(r'(?:in|of)\s+area\s+([\w\s]+)(?:$|\s+(?:and|with|that|which))', query)
            if area_match:
                params['area'] = area_match.group(1).strip()

        # Extract other parameters
        equipment_match = re.search(r'(?:for|of)\s+equipment\s+group\s+([\w\s]+)(?:$|\s+(?:and|with))', query)
        if equipment_match:
            params['equipment_group'] = equipment_match.group(1).strip()

        model_match = re.search(r'(?:for|of)\s+model\s+([\w\s]+)(?:$|\s+(?:and|with))', query)
        if model_match:
            params['model'] = model_match.group(1).strip()

        asset_match = re.search(r'(?:for|of)\s+asset\s+(?:number\s+)?([\w\-\d]+)', query)
        if asset_match:
            params['asset_number'] = asset_match.group(1).strip()

        drawing_match = re.search(r'drawing\s+(?:number\s+)?([A-Z0-9\-]+)', query, re.IGNORECASE)
        if drawing_match:
            params['drawing_number'] = drawing_match.group(1).strip()

        return params

class AistManager(UnifiedSearchMixin):
    """
    AI Steward manages the search strategies and response generation process.
    Cleaned and optimized version with consolidated functionality.
    """

    def __init__(self, ai_model=None, db_session=None):
        """Initialize with optional AI model and database session."""
        self.ai_model = ai_model
        self.db_session = db_session
        self.start_time = None
        self.db_config = DatabaseConfig()
        self.performance_history = []
        self.current_request_id = None

        # Initialize tracking attributes
        self.tracked_search = None
        self.current_user_id = None
        self.current_session_id = None
        self.query_tracker = None

        logger.info("=== AIST MANAGER INITIALIZATION STARTING ===")

        # Initialize vector search client
        try:
            self.vector_search_client = VectorSearchClient()
            logger.debug("Vector search client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize vector search client: {e}")
            self.vector_search_client = None

        # Initialize unified search capabilities
        try:
            UnifiedSearchMixin.__init__(self)
            logger.info("UnifiedSearchMixin initialized")
        except Exception as e:
            logger.error(f"UnifiedSearchMixin initialization failed: {e}")
            self.unified_search_system = None

        # Initialize tracking
        self._init_tracking()

        logger.info("=== AIST MANAGER INITIALIZATION COMPLETE ===")

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

    @with_request_id
    def _init_tracking(self):
        """Initialize search tracking components."""
        if not self.db_session:
            logger.warning("No database session - tracking disabled")
            return False

        try:
            from modules.search.nlp_search import SearchQueryTracker
            from modules.search.models.search_models import UnifiedSearchWithTracking

            self.query_tracker = SearchQueryTracker(self.db_session)
            self.tracked_search = UnifiedSearchWithTracking(self)
            self.tracked_search.query_tracker = self.query_tracker

            logger.info("Search tracking initialized successfully")
            return True

        except ImportError as e:
            logger.warning(f"Tracking modules not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Tracking initialization failed: {e}")
            return False

    # ===== USER SESSION MANAGEMENT =====
    @with_request_id
    def set_current_user(self, user_id: str, context_data: Dict = None) -> bool:
        """Set the current user and start a tracking session."""
        try:
            self.current_user_id = user_id

            if self.tracked_search:
                self.current_session_id = self.tracked_search.start_user_session(
                    user_id=user_id,
                    context_data=context_data or {
                        'component': 'aist_manager',
                        'session_started_at': datetime.utcnow().isoformat()
                    }
                )
                logger.info(f"Started tracking session {self.current_session_id} for user {user_id}")
                return True
            else:
                logger.debug(f"Set current user {user_id} (tracking disabled)")
                return False

        except Exception as e:
            logger.error(f"Failed to set current user {user_id}: {e}")
            return False

    @with_request_id
    def end_user_session(self) -> bool:
        """End the current user session."""
        try:
            if self.tracked_search and self.current_session_id:
                success = self.tracked_search.end_session()
                if success:
                    logger.info(f"Ended tracking session {self.current_session_id}")
                    self.current_session_id = None
                    self.current_user_id = None
                return success
            return False

        except Exception as e:
            logger.error(f"Failed to end user session: {e}")
            return False

    # ===== MAIN SEARCH METHODS =====

    def answer_question(self, user_id, question, client_type='web', request_id=None):
        """Main question answering method with improved structure."""
        self.start_time = time.time()
        self.current_request_id = request_id or get_request_id()

        logger.info(f"Processing question: {question}")

        try:
            # Prepare search context
            search_context = self._prepare_search_context(user_id, client_type)

            # Delegate to UnifiedSearchMixin for primary search
            result = result = self._execute_primary_search(question, search_context)

            # Format final response
            return self._format_final_response(result, question, user_id)

        except Exception as e:
            logger.error(f"Error in answer_question: {e}", exc_info=True)
            return self._create_error_response(e, question, user_id)

    @with_request_id
    def _prepare_search_context(self, user_id, client_type):
        """Prepare search context and check capabilities."""
        has_tracking = (self.query_tracker is not None and
                        self.tracked_search is not None and
                        hasattr(self.tracked_search, 'execute_unified_search_with_tracking'))

        has_vector_search = (hasattr(self, 'vector_search_client') and
                             self.vector_search_client is not None)

        # Initialize user session if needed
        if has_tracking and not self.current_session_id:
            try:
                self.set_current_user(user_id or "anonymous", {'client_type': client_type})
            except Exception as e:
                logger.warning(f"Failed to set user session: {e}")

        return {
            'has_tracking': has_tracking,
            'has_vector_search': has_vector_search,
            'user_id': user_id or "anonymous"
        }


    @with_request_id
    def _handle_search_fallbacks(self, question, context):
        """Handle search fallbacks when primary search fails."""
        if not context['has_vector_search'] or len(question.strip()) <= 2:
            return self._create_helpful_no_results_response(question)

        logger.info(f"Attempting vector search fallback for: '{question}'")

        try:
            # Use shorter timeout for vector search to avoid slow responses
            vector_results = self.vector_search_client.search(question, limit=5, threshold=0.2)

            if vector_results and len(vector_results) > 0:
                logger.info(f"Vector search found {len(vector_results)} results!")
                return self._format_vector_results(vector_results, question)
            else:
                logger.info("Vector search found no results")
                return self._create_helpful_no_results_response(question)

        except Exception as e:
            logger.error(f"Vector search failed: {e}", exc_info=True)
            return self._create_helpful_no_results_response(question)

    @with_request_id
    def _create_helpful_no_results_response(self, question):
        """Create a helpful response when no results are found."""
        # Analyze the question to provide specific guidance
        question_lower = question.lower()

        if any(term in question_lower for term in ['enzyme', 'protein', 'biochemical']):
            suggestion = "You might want to search for specific part numbers, equipment names, or documentation related to your facility's systems."
        elif any(term in question_lower for term in ['part', 'component']):
            suggestion = "Try searching with a specific part number, manufacturer name, or equipment type."
        elif any(term in question_lower for term in ['image', 'picture', 'photo']):
            suggestion = "Try searching for images by equipment area, asset number, or specific equipment names."
        else:
            suggestion = "Try using more specific terms like equipment names, part numbers, or area designations."

        return {
            'status': 'success',  # Not an error, just no results
            'answer': f"I couldn't find specific information about '{question}' in our facility database. {suggestion}",
            'method': 'helpful_no_results',
            'total_results': 0,
            'request_id': self.current_request_id,
            'suggestions': [
                "Try searching for specific part numbers",
                "Search by equipment area or location",
                "Use manufacturer names or model numbers",
                "Look for related documentation or manuals"
            ]
        }

    @with_request_id
    def _format_vector_results(self, vector_results, question):
        """Format vector search results into response structure."""
        answer_parts = ["Based on semantic similarity, I found relevant information:\n"]

        for i, vr in enumerate(vector_results[:5], 1):
            content = vr.get('content', '')
            similarity = vr.get('similarity', 0)
            doc_id = vr.get('id', 'unknown')

            if len(content) > 300:
                truncated = content[:300]
                last_period = truncated.rfind('.')
                if last_period > 200:
                    content = truncated[:last_period + 1]
                else:
                    content = truncated + "..."

            answer_parts.append(
                f"{i}. {content}\n   (Document ID: {doc_id}, Similarity: {similarity:.3f})\n"
            )

        return {
            'status': 'success',
            'answer': "\n".join(answer_parts),
            'method': 'vector_search_fallback',
            'total_results': len(vector_results),
            'request_id': self.current_request_id
        }

    @with_request_id
    def _format_final_response(self, result, question, user_id):
        """Format the final response."""
        # Determine the answer based on result
        if result and result.get('status') == 'success':
            answer = ResponseFormatter.format_search_results(result)
        elif result and result.get('message'):
            answer = result.get('message', 'Search failed - no specific error message')
        elif result and result.get('answer'):
            answer = result.get('answer')
        else:
            # Provide a helpful fallback message
            answer = f"I couldn't find specific information about '{question}' in the system. This might be because the search term doesn't match our current database content, or it may require a different search approach."

        # Record interaction
        try:
            interaction = self.record_interaction(user_id or "anonymous", question, answer)
            if interaction:
                logger.info(f"Recorded interaction: {interaction.id}")
        except Exception as e:
            logger.warning(f"Failed to record interaction: {e}")

        # Determine status - only mark as error if there was an actual error, not just no results
        status = 'success'
        if result:
            if result.get('status') in ['error', 'failed']:
                status = 'error'
            elif result.get('status') in ['success', 'no_results']:
                status = 'success'
            else:
                status = result.get('status', 'success')

        return {
            'status': status,
            'answer': answer,
            'method': result.get('search_method', 'unified_search') if result else 'unified_search_fallback',
            'total_results': result.get('total_results', 0) if result else 0,
            'request_id': self.current_request_id
        }

    @with_request_id
    def _create_error_response(self, error, question, user_id):
        """Create error response."""
        error_msg = f"I encountered an error while processing your question: {str(error)}"

        try:
            self.record_interaction(user_id or "anonymous", question, error_msg)
        except:
            pass

        return {
            'status': 'error',
            'answer': error_msg,
            'message': str(error),
            'method': 'error_fallback',
            'total_results': 0,
            'request_id': self.current_request_id
        }

    # ===== SEARCH ANALYTICS =====
    @with_request_id
    def execute_search_with_analytics(self, question: str, request_id: str = None) -> Dict[str, Any]:
        """Main search method with comprehensive tracking and analytics."""
        search_start = time.time()
        self.current_request_id = request_id or get_request_id()

        logger.info(f"Execute search with analytics: '{question}' (Request: {self.current_request_id})")

        try:
            if self.tracked_search:
                result = self.tracked_search.execute_unified_search_with_tracking(
                    question=question,
                    user_id=self.current_user_id or "anonymous",
                    request_id=self.current_request_id
                )

                result.update({
                    'aist_manager_info': {
                        'request_id': self.current_request_id,
                        'user_id': self.current_user_id,
                        'session_id': self.current_session_id,
                        'vector_search_available': self.vector_search_client is not None,
                        'tracking_enabled': True
                    }
                })

                return result
            else:
                result = self.execute_unified_search(
                    question=question,
                    user_id=self.current_user_id,
                    request_id=self.current_request_id
                )

                result.update({
                    'aist_manager_info': {
                        'request_id': self.current_request_id,
                        'user_id': self.current_user_id,
                        'vector_search_available': self.vector_search_client is not None,
                        'tracking_enabled': False,
                        'fallback_reason': 'tracking_unavailable'
                    }
                })

                return result

        except Exception as e:
            search_time = time.time() - search_start
            logger.error(f"Search failed after {search_time:.3f}s: {e}")

            return {
                'status': 'error',
                'message': f"Search failed: {str(e)}",
                'search_type': 'aist_manager_error',
                'aist_manager_info': {
                    'request_id': self.current_request_id,
                    'error_type': type(e).__name__,
                    'execution_time_ms': int(search_time * 1000)
                }
            }

    @with_request_id
    def get_search_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive search analytics and performance metrics."""
        try:
            if self.tracked_search:
                analytics = self.tracked_search.get_performance_report(days)
                analytics.update({
                    'aist_manager_metrics': self._get_aist_manager_metrics(),
                    'system_health': self._get_system_health_status(),
                    'generated_at': datetime.utcnow().isoformat(),
                    'report_type': 'comprehensive_search_analytics'
                })
                return analytics
            else:
                return {
                    'status': 'limited',
                    'message': 'Search tracking not available',
                    'aist_manager_metrics': self._get_aist_manager_metrics(),
                    'system_health': self._get_system_health_status()
                }

        except Exception as e:
            logger.error(f"Failed to generate analytics: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'error_type': 'analytics_generation_failed'
            }

    @with_request_id
    def _get_aist_manager_metrics(self) -> Dict[str, Any]:
        """Get AistManager-specific performance metrics."""
        return {
            'vector_search_available': self.vector_search_client is not None,
            'database_available': self.db_session is not None,
            'performance_history_count': len(self.performance_history),
            'current_session_active': self.current_session_id is not None,
            'current_user': self.current_user_id,
            'tracking_enabled': self.tracked_search is not None
        }

    @with_request_id
    def _get_system_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        health = {
            'overall_status': 'healthy',
            'components': {
                'unified_search': hasattr(self, 'unified_search_system') and self.unified_search_system is not None,
                'vector_search': self.vector_search_client is not None,
                'database': self.db_session is not None,
                'search_tracking': self.tracked_search is not None,
                'ai_model': self.ai_model is not None
            },
            'warnings': []
        }

        if not health['components']['search_tracking']:
            health['warnings'].append('Search tracking disabled - analytics limited')

        if not health['components']['vector_search']:
            health['warnings'].append('Vector search unavailable - similarity search limited')

        if not health['components']['database']:
            health['warnings'].append('Database unavailable - persistent search disabled')

        critical_components = ['unified_search', 'database']
        if not all(health['components'][comp] for comp in critical_components):
            health['overall_status'] = 'degraded'
        elif len(health['warnings']) > 0:
            health['overall_status'] = 'healthy_with_warnings'

        return health

    # ===== SEARCH EXECUTION METHODS =====

    @with_request_id
    def try_keyword_search(self, question):
        """Try to answer using keyword search with improved error handling."""
        logger.debug('Attempting keyword search')
        local_session = None

        try:
            session = self.db_session or self.db_config.get_main_session()
            if not self.db_session:
                local_session = session

            try:
                keyword, action, details = KeywordAction.find_best_match(question, session)
            except Exception as keyword_error:
                logger.error(f"KeywordAction.find_best_match failed: {keyword_error}")
                keyword, action, details = None, None, None

            if keyword and action:
                logger.info(f"Found matching keyword: {keyword} with action: {action}")
                try:
                    result = self._execute_keyword_action(keyword, action, details, question)
                except Exception as action_error:
                    logger.error(f"execute_keyword_action failed: {action_error}")
                    result = self._perform_advanced_keyword_search(question)

                if result:
                    entity_type = result.get('entity_type', 'generic')
                    results = result.get('results', [])

                    if results:
                        answer = self._format_entity_results(results, entity_type)
                        logger.info(f"Found {len(results)} {entity_type} results via keyword search")
                    else:
                        answer = action if isinstance(action, str) else "Action executed successfully."
                        logger.info(f"Executed keyword action for '{keyword}'")

                    return {'success': True, 'answer': answer, 'results': results}

            logger.debug("Keyword search found no matching keywords or actions")
            return {'success': False}

        except Exception as e:
            logger.error(f"Error in keyword search: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
        finally:
            if local_session:
                local_session.close()

    @with_request_id
    def try_fulltext_search(self, question):
        """Try to find answers using full-text search strategies."""
        logger.debug(f"Trying fulltext search for: {question}")

        try:
            local_session = None
            session = self.db_session or self.db_config.get_main_session()
            if not self.db_session:
                local_session = session

            try:
                search_results = CompleteDocument.search_by_text(
                    query=question,
                    session=session,
                    similarity_threshold=70,
                    with_links=False
                )

                if search_results and len(search_results) > 0:
                    doc_titles = [doc.title for doc in search_results[:5]]
                    response = "I found the following documents that might contain relevant information:\n"
                    response += "\n".join([f"- {title}" for title in doc_titles])
                    response += "\n\nWould you like me to retrieve one of these documents for you?"

                    return {
                        "success": True,
                        "answer": response,
                        "results": [{"title": doc.title, "id": doc.id} for doc in search_results[:5]]
                    }

                return {"success": False}

            finally:
                if local_session:
                    local_session.close()

        except Exception as e:
            logger.error(f"Error in fulltext search: {e}", exc_info=True)
            return {"success": False}

    @with_request_id
    def try_vector_search(self, user_id, question):
        """Try to find answers using vector search strategies."""
        logger.debug(f"Trying vector search for user {user_id}: {question}")

        try:
            if not hasattr(self, "vector_search_client") or not self.vector_search_client:
                logger.warning("Vector search client not available")
                return {"success": False}

            vector_results = self.vector_search_client.search(question, limit=5)

            if vector_results and len(vector_results) > 0:
                response = "Based on semantic search, I found these potentially relevant passages:\n\n"

                for i, result in enumerate(vector_results[:3], 1):
                    content = result.get("content", "").strip()

                    if len(content) > 300:
                        content = content[:297] + "..."

                    source = result.get("source", "document")
                    response += f"{i}. {content}\n"
                    response += f"   Source: {source}\n\n"

                response += "Does any of this information address your question?"

                return {
                    "success": True,
                    "answer": response,
                    "results": vector_results[:5]
                }

            return {"success": False}

        except Exception as e:
            logger.error(f"Error in vector search: {e}", exc_info=True)
            return {"success": False}

    # ===== UTILITY AND SUPPORT METHODS =====
    @with_request_id
    def _execute_keyword_action(self, keyword, action, details, original_question):
        """Execute the appropriate action based on the keyword match."""
        try:
            search_params = SearchUtils.extract_search_params(original_question)

            if action.startswith("search_images_bp"):
                return self._handle_image_search(action, search_params)
            elif action.startswith("search_documents_bp"):
                return self._handle_document_search(action, search_params)
            elif action.startswith("search_drawing_by_number_bp"):
                return self._handle_drawing_search(search_params)
            elif action.startswith("RESPOND:"):
                response_text = action[8:].strip()
                return {"entity_type": "response", "results": [{"text": response_text}]}
            else:
                return {"entity_type": "response", "results": [{"text": f"I'll help you with: {keyword}"}]}

        except Exception as e:
            logger.error(f"Error executing keyword action: {e}", exc_info=True)
            return {"entity_type": "error", "results": [{"text": "I encountered an error processing your request."}]}

    @with_request_id
    def _handle_image_search(self, action, search_params):
        """Handle image search based on action type."""
        if "_title" in action:
            search_params['title'] = search_params.get('title', '')
        elif "_area" in action:
            search_params['area'] = search_params.get('area', '')
        elif "_equipment_group" in action:
            search_params['equipment_group'] = search_params.get('equipment_group', '')
        elif "_model" in action:
            search_params['model'] = search_params.get('model', '')
        elif "_asset_number" in action:
            search_params['asset_number'] = search_params.get('asset_number', '')

        return self._search_images(search_params)

    @with_request_id
    def _handle_document_search(self, action, search_params):
        """Handle document search based on action type."""
        if "_title" in action:
            search_params['title'] = search_params.get('title', '')
        elif "_area" in action:
            search_params['area'] = search_params.get('area', '')

        return self._search_documents(search_params)

    @with_request_id
    def _handle_drawing_search(self, search_params):
        """Handle drawing search."""
        return self._search_drawings(search_params)

    @with_request_id
    def _search_images(self, params):
        """Search for images based on parameters."""
        local_session = None
        try:
            session = self.db_session or self.db_config.get_main_session()
            if not self.db_session:
                local_session = session

            results = Image.search(
                search_text=params.get('query'),
                title=params.get('title'),
                limit=20,
                session=session
            )

            if results:
                image_results = []
                for img in results:
                    img_data = Image.serve_file(image_id=img.id, session=session)
                    if img_data and img_data['exists']:
                        image_results.append({
                            'id': img.id,
                            'title': img.title,
                            'description': img.description,
                            'url': f"/search_images_bp/serve_image/{img.id}"
                        })

                return {"entity_type": "image", "results": image_results}
            else:
                return {"entity_type": "response", "results": [{"text": "No images found matching your criteria."}]}

        except Exception as e:
            logger.error(f"Error in image search: {e}", exc_info=True)
            return {"entity_type": "error", "results": [{"text": f"Error searching for images: {str(e)}"}]}
        finally:
            if local_session:
                local_session.close()

    @with_request_id
    def _search_documents(self, params):
        """Search for documents based on parameters."""
        local_session = None
        try:
            session = self.db_session or self.db_config.get_main_session()
            if not self.db_session:
                local_session = session

            search_text = params.get('query')
            title = params.get('title')

            if title:
                results = CompleteDocument.dynamic_search(session, title=title)
            elif search_text:
                results = CompleteDocument.search_by_text(
                    search_text,
                    session=session,
                    similarity_threshold=80,
                    with_links=False
                )
            else:
                results = []

            if results:
                document_results = []
                for doc in results:
                    document_results.append({
                        'id': doc.id,
                        'title': doc.title,
                        'rev': doc.rev,
                        'url': f"/search_documents_bp/view_document/{doc.id}"
                    })

                return {"entity_type": "document", "results": document_results}
            else:
                return {"entity_type": "response", "results": [{"text": "No documents found matching your criteria."}]}

        except Exception as e:
            logger.error(f"Error in document search: {e}", exc_info=True)
            return {"entity_type": "error", "results": [{"text": f"Error searching for documents: {str(e)}"}]}
        finally:
            if local_session:
                local_session.close()

    @with_request_id
    def _search_drawings(self, params):
        """Search for drawings based on parameters."""
        local_session = None
        try:
            session = self.db_session or self.db_config.get_main_session()
            if not self.db_session:
                local_session = session

            search_text = params.get('query')
            drawing_number = params.get('drawing_number')

            return Drawing.search_and_format(
                search_text=search_text,
                drw_number=drawing_number,
                limit=20,
                session=session
            )

        except Exception as e:
            logger.error(f"Error in drawing search: {e}", exc_info=True)
            return {"entity_type": "error", "results": [{"text": f"Error searching for drawings: {str(e)}"}]}
        finally:
            if local_session:
                local_session.close()

    @with_request_id
    def _perform_advanced_keyword_search(self, query):
        """Perform an advanced keyword search across multiple entity types."""
        try:
            with KeywordSearch(session=self.db_session) as keyword_search:
                keyword_result = keyword_search.execute_search(query)

                if keyword_result.get("status") == "success" and keyword_result.get("count", 0) > 0:
                    return keyword_result

            # Fallback to multi-entity search
            search_params = SearchUtils.extract_search_params(query)
            search_params["query"] = query

            all_results = []

            # Try different search types
            for search_method in [self._search_documents, self._search_images, self._search_drawings]:
                try:
                    results = search_method(search_params)
                    if results.get('results'):
                        for item in results['results'][:3]:
                            item['type'] = results.get('entity_type', 'unknown')
                            all_results.append(item)
                except Exception as e:
                    logger.warning(f"Search method failed: {e}")

            return {
                "status": "success" if all_results else "not_found",
                "entity_type": "mixed",
                "count": len(all_results),
                "results": all_results
            }

        except Exception as e:
            logger.error(f"Error in advanced keyword search: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error in keyword search: {str(e)}",
                "query": query
            }

    @with_request_id
    def _format_entity_results(self, results, entity_type):
        """Format search results based on entity type."""
        if not results:
            return "No results found."

        entity_type_readable = entity_type.replace('_', ' ').title()
        response = f"I found {len(results)} {entity_type_readable}{'s' if len(results) > 1 else ''} that might be relevant:\n\n"

        for idx, item in enumerate(results[:5], 1):
            if entity_type == "document":
                title = item.get('title', f"Document #{item.get('id', 'Unknown')}")
                response += f"{idx}. {title}\n"
            elif entity_type == "image":
                title = item.get('title', f"Image #{item.get('id', 'Unknown')}")
                description = item.get('description', '')
                if description:
                    description = f" - {description}"
                response += f"{idx}. {title}{description}\n"
            elif entity_type == "drawing":
                number = item.get('number', '')
                name = item.get('name', '')
                response += f"{idx}. Drawing {number}: {name}\n"
            elif entity_type == "part":
                part_number = item.get('part_number', '')
                name = item.get('name', f"Part #{item.get('id', 'Unknown')}")
                response += f"{idx}. {part_number}: {name}\n"
            elif entity_type == "response":
                text = item.get('text', '')
                if text:
                    response = text
                    break
            else:
                item_name = item.get('title') or item.get('name') or f"Item #{item.get('id', 'Unknown')}"
                response += f"{idx}. {item_name}\n"

        return response

    @with_request_id
    def record_interaction(self, user_id, question, answer):
        """Record interaction with enhanced debugging."""
        try:
            local_session = None
            session = self.db_session or self.db_config.get_main_session()
            if not self.db_session:
                local_session = session

            try:
                processing_time = None
                if hasattr(self, 'start_time') and self.start_time:
                    processing_time = int((time.time() - self.start_time) * 1000)

                interaction = QandA.record_interaction(
                    user_id=user_id,
                    question=question,
                    answer=answer,
                    session=session,
                    processing_time_ms=processing_time
                )

                if interaction:
                    logger.info(f"Successfully recorded interaction {interaction.id} for user {user_id}")
                    return interaction
                else:
                    logger.error("QandA.record_interaction returned None")
                    return None

            finally:
                if local_session:
                    local_session.close()

        except Exception as e:
            logger.error(f"Error in record_interaction: {e}", exc_info=True)
            return None

    @with_request_id
    def begin_request(self, request_id=None):
        """Start timing a new request."""
        self.start_time = time.time()
        if request_id:
            self.current_request_id = request_id
            logger.debug(f"Request {request_id} started with performance tracking")
        else:
            self.current_request_id = None

    @with_request_id
    def get_response_time(self):
        """Calculate the response time so far."""
        if self.start_time:
            return time.time() - self.start_time
        return 0

    @with_request_id
    def format_response(self, answer, client_type=None, results=None):
        """Enhanced format_response that includes performance information."""
        formatted_answer = answer.strip()

        if hasattr(self, 'start_time') and self.start_time:
            response_time = time.time() - self.start_time

            if response_time > 2.0 or client_type == 'debug':
                if '<div class="performance-note">' not in formatted_answer:
                    formatted_answer += f"<div class='performance-note'><small>Response time: {response_time:.2f}s</small></div>"

        if '<a href=' not in formatted_answer and ('http://' in formatted_answer or 'https://' in formatted_answer):
            formatted_answer = re.sub(
                r'(https?://[^\s]+)',
                r'<a href="\1" target="_blank">\1</a>',
                formatted_answer
            )

        return formatted_answer

    @with_request_id
    def find_most_relevant_document(self, question, session=None, request_id=None):
        """Find the most relevant document with timeout protection."""
        search_start = time.time()
        logger.debug(f"Finding most relevant document for question: {question[:50]}...")

        local_session = None
        if not session:
            if self.db_session:
                session = self.db_session
            else:
                local_session = self.db_config.get_main_session()
                session = local_session

        try:
            embedding_model_name = ModelsConfig.get_config_value('embedding', 'CURRENT_MODEL', 'NoEmbeddingModel')

            if embedding_model_name == "NoEmbeddingModel":
                logger.info("Embeddings are disabled. Returning None for document search.")
                return None

            def search_operation():
                if hasattr(self, 'vector_search_client') and self.vector_search_client:
                    vector_results = self.vector_search_client.search(question, limit=1)

                    if vector_results:
                        best_result = vector_results[0]
                        doc_id = best_result['id']
                        similarity = best_result['similarity']

                        document = session.query(Document).get(doc_id)
                        if document:
                            search_time = time.time() - search_start
                            logger.info(
                                f"Found most relevant document (ID: {doc_id}, similarity: {similarity:.4f}) in {search_time:.3f}s")
                            return document

                logger.debug("Using fallback document search")
                recent_docs = session.query(Document).limit(5).all()
                if recent_docs:
                    search_time = time.time() - search_start
                    logger.info(f"Fallback: returning recent document in {search_time:.3f}s")
                    return recent_docs[0]

                return None

            result, error = SearchUtils.execute_with_timeout(search_operation, 3.0)

            if error:
                search_time = time.time() - search_start
                logger.warning(f"Document search timed out after {search_time:.3f}s for request {request_id}")
                return None

            return result

        except Exception as e:
            search_time = time.time() - search_start
            logger.error(f"Error in document search after {search_time:.3f}s: {e}", exc_info=True)
            return None
        finally:
            if local_session:
                local_session.close()

    @with_request_id
    def cleanup_session(self):
        """Clean up the current session and resources."""
        try:
            if self.current_session_id:
                self.end_user_session()

            self.current_request_id = None
            self.start_time = None

            logger.debug("Session cleanup completed")

        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_session()

        if hasattr(self, 'unified_search_system') and self.unified_search_system:
            if hasattr(self.unified_search_system, 'close_session'):
                try:
                    self.unified_search_system.close_session()
                except Exception as e:
                    logger.error(f"Failed to close unified search session: {e}")

    @with_request_id
    def query_ai_model(self, prompt, user_id=None, request_id=None):
        """Send a prompt directly to the currently selected AI model for an answer."""
        self.current_request_id = request_id or get_request_id()

        logger.info(f"Sending query to AI model (Request: {self.current_request_id})")
        logger.debug(f"Prompt length: {len(prompt)} characters")

        try:
            # Load the currently selected AI model
            ai_model = ModelsConfig.load_ai_model()

            if not ai_model:
                error_msg = "No AI model is currently available"
                logger.error(error_msg)
                return {
                    'status': 'error',
                    'answer': error_msg,
                    'method': 'ai_model_direct',
                    'model_name': 'none',
                    'request_id': self.current_request_id
                }

            # Check if AI is disabled
            if hasattr(ai_model, '__class__') and 'NoAIModel' in ai_model.__class__.__name__:
                logger.info("AI model is disabled (NoAIModel)")
                return {
                    'status': 'success',
                    'answer': "AI is currently disabled.",
                    'method': 'ai_model_disabled',
                    'model_name': 'NoAIModel',
                    'request_id': self.current_request_id
                }

            # Get the model name for logging
            model_name = getattr(ai_model, 'model_name', ai_model.__class__.__name__)
            logger.info(f"Using AI model: {model_name}")

            # Send prompt to AI model
            start_time = time.time()
            response = ai_model.get_response(prompt)
            response_time = time.time() - start_time

            logger.info(f"AI model responded in {response_time:.2f}s")
            logger.debug(f"Response length: {len(response)} characters")

            # Record interaction if user_id provided
            if user_id:
                try:
                    self.record_interaction(user_id, prompt, response)
                    logger.debug(f"Interaction recorded for user: {user_id}")
                except Exception as e:
                    logger.warning(f"Failed to record interaction: {e}")

            return {
                'status': 'success',
                'answer': response,
                'method': 'ai_model_direct',
                'model_name': model_name,
                'response_time_seconds': round(response_time, 2),
                'request_id': self.current_request_id
            }

        except Exception as e:
            error_msg = f"Error querying AI model: {str(e)}"
            logger.error(error_msg)
            logger.exception("AI model query exception:")

            return {
                'status': 'error',
                'answer': error_msg,
                'method': 'ai_model_direct',
                'model_name': 'unknown',
                'error': str(e),
                'request_id': self.current_request_id
            }

    @with_request_id
    def query_ai_with_context(self, user_query, context_text, user_id=None, request_id=None):
        """Send a user query with context text to the AI model."""
        self.current_request_id = request_id or get_request_id()

        logger.info(f"Sending query with context to AI model (Request: {self.current_request_id})")

        # Build the contextual prompt
        prompt = f"""Based on the following context information, please answer the user's question accurately and helpfully.

    CONTEXT:
    {context_text}

    USER QUESTION:
    {user_query}

    Please provide a clear, helpful answer based on the context provided. If the context doesn't contain enough information to fully answer the question, please say so and provide what information you can."""

        # Use the direct query function
        return self.query_ai_model(prompt, user_id, request_id)

    @with_request_id
    def query_ai_simple(self, question, user_id=None, request_id=None):
        """Send a simple question directly to the AI model without additional context."""
        return self.query_ai_model(question, user_id, request_id)



# ===== GLOBAL INSTANCE MANAGEMENT =====

global_aist_manager = None

@with_request_id
def get_or_create_aist_manager():
    """Get or create a global AistManager instance with database session for tracking."""
    global global_aist_manager

    if global_aist_manager is None:
        try:
            logger.info("Creating AistManager with tracking support...")

            db_config = DatabaseConfig()
            db_session = db_config.get_session()

            if not db_session:
                logger.error("Could not get database session")
                global_aist_manager = AistManager()
            else:
                logger.info("Database session obtained")
                ai_model = ModelsConfig.load_ai_model()
                global_aist_manager = AistManager(
                    ai_model=ai_model,
                    db_session=db_session
                )

            logger.info("Global AistManager created successfully")

        except Exception as e:
            logger.error(f"Failed to create AistManager with tracking: {e}")
            try:
                ai_model = ModelsConfig.load_ai_model()
                global_aist_manager = AistManager(ai_model=ai_model)
                logger.info("Created fallback AistManager without tracking")
            except Exception as fallback_error:
                logger.error(f"Fallback AistManager creation failed: {fallback_error}")
                global_aist_manager = AistManager()

    return global_aist_manager
