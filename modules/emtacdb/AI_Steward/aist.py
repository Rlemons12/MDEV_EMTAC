# modules/ai/ai_steward.py
import time
import re
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from sqlalchemy.ext.declarative import declarative_base
from modules.configuration.log_config import logger, with_request_id, log_timed_operation
from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import (
    Position, CompleteDocument, Image, KeywordSearch, ChatSession, User,
    QandA, Drawing, CompletedDocumentPositionAssociation,
    KeywordAction, Document, DocumentEmbedding, ImageEmbedding,
    PartsPositionImageAssociation, ImageCompletedDocumentAssociation,
    ImagePositionAssociation, ToolImageAssociation, ImageTaskAssociation,
    ImageProblemAssociation
)
from plugins.ai_modules.ai_models import ModelsConfig
import threading
import signal
from collections import defaultdict
from datetime import datetime, timedelta
from modules.search.UnifiedSearchMixin import UnifiedSearchMixin
# Base class for SQLAlchemy models
Base = declarative_base()


# REPLACE your existing VectorSearchClient class with this optimized version
# Keep the same class name for compatibility

def get_request_id():
    """Helper function to get request ID from context or generate one"""
    try:
        from modules.configuration.log_config import get_current_request_id
        return get_current_request_id()
    except:
        import uuid
        return str(uuid.uuid4())[:8]

# WINDOWS-COMPATIBLE VectorSearchClient (complete replacement)
class VectorSearchClient:
    """Windows-compatible VectorSearchClient with threading-based timeouts."""

    def __init__(self):
        self.np = __import__('numpy')

        # Performance optimization features
        self.embedding_cache = {}
        self.cache_expiry = {}
        self.cache_max_size = 1000
        self.cache_ttl = 3600

        logger.debug("Windows-compatible vector search client initialized")

    def search(self, query, limit=5):
        """Vector search with Windows-compatible timeout."""
        from modules.emtacdb.emtacdb_fts import DocumentEmbedding, Document
        from modules.configuration.config_env import DatabaseConfig

        search_start = time.time()

        try:
            # Windows-compatible timeout using threading
            result = [None]
            exception = [None]

            def search_worker():
                try:
                    result[0] = self._perform_search(query, limit)
                except Exception as e:
                    exception[0] = e

            search_thread = threading.Thread(target=search_worker)
            search_thread.daemon = True
            search_thread.start()
            search_thread.join(timeout=400.0)  # 3-second timeout

            if search_thread.is_alive():
                logger.warning(f"Vector search timed out after 400 seconds for query: {query[:50]}...")
                return []

            if exception[0]:
                raise exception[0]

            return result[0] if result[0] else []

        except Exception as e:
            logger.error(f"Error in vector search: {e}", exc_info=True)
            return []
        finally:
            search_time = time.time() - search_start
            if search_time > 1.0:
                logger.warning(f"Vector search completed in {search_time:.3f}s")
            else:
                logger.debug(f"Vector search completed in {search_time:.3f}s")

    def _perform_search(self, query, limit):
        """Actual search implementation without timeout."""
        from modules.emtacdb.emtacdb_fts import DocumentEmbedding, Document
        from modules.configuration.config_env import DatabaseConfig

        # Get database session
        db_config = DatabaseConfig()
        session = db_config.get_main_session()

        try:
            # Get embedding model
            embedding_model = ModelsConfig.load_embedding_model()

            # Generate query embedding
            query_embedding = embedding_model.get_embeddings(query)
            if not query_embedding:
                logger.warning("Failed to generate embedding for query")
                return []

            # Convert query embedding to numpy array
            query_embedding_np = self.np.array(query_embedding)
            query_norm = self.np.linalg.norm(query_embedding_np)

            if query_norm == 0:
                logger.warning("Query embedding has zero norm")
                return []

            # Simple search implementation - limit to first 100 for speed
            doc_embeddings = session.query(
                DocumentEmbedding.document_id,
                DocumentEmbedding.model_embedding
            ).limit(100).all()

            similarities = []
            for doc_id, doc_embedding_raw in doc_embeddings:
                try:
                    # Parse embedding
                    if isinstance(doc_embedding_raw, bytes):
                        try:
                            doc_embedding = self.np.array(json.loads(doc_embedding_raw.decode('utf-8')))
                        except:
                            doc_embedding = self.np.frombuffer(doc_embedding_raw, dtype=self.np.float32)
                    else:
                        doc_embedding = self.np.array(doc_embedding_raw)

                    # Calculate similarity
                    doc_norm = self.np.linalg.norm(doc_embedding)
                    if doc_norm > 0:
                        similarity = self.np.dot(query_embedding_np, doc_embedding) / (query_norm * doc_norm)
                        if similarity > 0.6:  # Minimum threshold
                            similarities.append((doc_id, float(similarity)))

                except Exception as e:
                    logger.debug(f"Error processing embedding for document {doc_id}: {e}")

            # Sort and get top results
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Fetch documents
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
                        "source": f"Document {doc_id}"
                    })

            logger.info(f"Vector search found {len(top_docs)} results")
            return top_docs

        finally:
            session.close()

    def cosine_similarity(self, vector1, vector2):
        """Calculate cosine similarity between two vectors."""
        dot_product = self.np.dot(vector1, vector2)
        norm_product = self.np.linalg.norm(vector1) * self.np.linalg.norm(vector2)

        if norm_product == 0:
            return 0

        return dot_product / norm_product

class AistManager(UnifiedSearchMixin):  # ← Inherit from UnifiedSearchMixin
    """
    AI Steward manages the search strategies and response generation process.
    It orchestrates keyword search, full-text search, vector similarity search,
    AI model fallback, and unified search capabilities with comprehensive tracking.

    Enhanced with search analytics, user behavior tracking, and performance monitoring.
    """

    def __init__(self, ai_model=None, db_session=None):
        """Initialize with optional AI model and database session."""
        # Initialize existing AistManager functionality
        self.ai_model = ai_model
        self.db_session = db_session
        self.start_time = None
        self.db_config = DatabaseConfig()
        self.performance_history = []
        self.step_timings = {}
        self.current_request_id = None
        self.performance_thresholds = {
            'keyword_search': 0.5,
            'fulltext_search': 1.0,
            'vector_search': 2.0,
            'ai_response': 3.0,
            'total_request': 5.0,
        }

        # Initialize vector search client
        try:
            self.vector_search_client = VectorSearchClient()
            logger.debug("Vector search client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize vector search client: {e}")
            self.vector_search_client = None

        # Initialize unified search capabilities FIRST
        UnifiedSearchMixin.__init__(self)

        # ===== NEW: Initialize Search Tracking System =====
        self.tracked_search = None
        self.current_user_id = None
        self.current_session_id = None
        self._init_search_tracking()

        logger.debug("AistManager with unified search and tracking capabilities initialized")

    def _init_search_tracking(self):
        """Initialize the search tracking system."""
        try:
            # Import the tracking system
            from modules.search.models.search_models import UnifiedSearchWithTracking

            # Initialize tracking wrapper around this instance
            self.tracked_search = UnifiedSearchWithTracking(self)
            logger.info("✅ Search tracking system initialized")

            # Try to create database tables if they don't exist
            self._ensure_tracking_tables()

        except ImportError as e:
            logger.warning(f"⚠️ Search tracking not available - missing models: {e}")
            self.tracked_search = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize search tracking: {e}")
            self.tracked_search = None

    def _ensure_tracking_tables(self):
        """Ensure tracking tables exist in the database."""
        try:
            if self.db_session:
                from modules.search.models.search_models import (
                    SearchSession, SearchQuery, SearchResultClick
                )
                from modules.configuration.base import Base

                # Create tables if they don't exist
                engine = self.db_session.bind
                Base.metadata.create_all(engine, tables=[
                    SearchSession.__table__,
                    SearchQuery.__table__,
                    SearchResultClick.__table__
                ])
                logger.debug("✅ Search tracking tables verified/created")

        except Exception as e:
            logger.warning(f"⚠️ Could not ensure tracking tables: {e}")

    # ===== USER SESSION MANAGEMENT =====

    def set_current_user(self, user_id: str, context_data: Dict = None) -> bool:
        """
        Set the current user and start a tracking session.

        Args:
            user_id: User identifier
            context_data: Additional context (role, department, etc.)

        Returns:
            bool: True if session started successfully
        """
        try:
            self.current_user_id = user_id

            if self.tracked_search:
                # Start a new tracking session
                self.current_session_id = self.tracked_search.start_user_session(
                    user_id=user_id,
                    context_data=context_data or {
                        'component': 'aist_manager',
                        'session_started_at': datetime.utcnow().isoformat()
                    }
                )
                logger.info(f"👤 Started tracking session {self.current_session_id} for user {user_id}")
                return True
            else:
                logger.debug(f"👤 Set current user {user_id} (tracking disabled)")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to set current user {user_id}: {e}")
            return False

    def end_user_session(self) -> bool:
        """End the current user session."""
        try:
            if self.tracked_search and self.current_session_id:
                success = self.tracked_search.end_session()
                if success:
                    logger.info(f"👋 Ended tracking session {self.current_session_id}")
                    self.current_session_id = None
                    self.current_user_id = None
                return success
            return False

        except Exception as e:
            logger.error(f"❌ Failed to end user session: {e}")
            return False

    # ===== ENHANCED SEARCH METHODS =====

    def execute_search_with_analytics(self, question: str, request_id: str = None) -> Dict[str, Any]:
        """
        Main search method with comprehensive tracking and analytics.
        This should be your primary search method going forward.

        Args:
            question: User's search query
            request_id: Optional request identifier

        Returns:
            Enhanced search results with tracking information
        """
        search_start = time.time()
        self.current_request_id = request_id or str(uuid.uuid4())

        logger.info(f"🔍 Execute search with analytics: '{question}' (Request: {self.current_request_id})")

        try:
            # Use tracking system if available
            if self.tracked_search:
                result = self.tracked_search.execute_unified_search_with_tracking(
                    question=question,
                    user_id=self.current_user_id or "anonymous",
                    request_id=self.current_request_id
                )

                # Add AistManager-specific metadata
                result.update({
                    'aist_manager_info': {
                        'request_id': self.current_request_id,
                        'user_id': self.current_user_id,
                        'session_id': self.current_session_id,
                        'vector_search_available': self.vector_search_client is not None,
                        'tracking_enabled': True
                    }
                })

                logger.info(
                    f"✅ Tracked search completed: {result.get('status')} with {result.get('total_results', 0)} results")
                return result

            else:
                # Fallback to regular unified search
                result = self.execute_unified_search(
                    question=question,
                    user_id=self.current_user_id,
                    request_id=self.current_request_id
                )

                # Add basic metadata
                result.update({
                    'aist_manager_info': {
                        'request_id': self.current_request_id,
                        'user_id': self.current_user_id,
                        'vector_search_available': self.vector_search_client is not None,
                        'tracking_enabled': False,
                        'fallback_reason': 'tracking_unavailable'
                    }
                })

                logger.info(
                    f"✅ Untracked search completed: {result.get('status')} with {result.get('total_results', 0)} results")
                return result

        except Exception as e:
            search_time = time.time() - search_start
            logger.error(f"❌ Search failed after {search_time:.3f}s: {e}")

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

    # ===== ANALYTICS AND FEEDBACK METHODS =====

    def record_search_satisfaction(self, query_id: int, rating: int, feedback: str = None) -> bool:
        """
        Record user satisfaction for a search query.

        Args:
            query_id: Query ID from tracking system
            rating: Satisfaction rating (1-5 scale)
            feedback: Optional text feedback

        Returns:
            bool: True if recorded successfully
        """
        try:
            if self.tracked_search:
                success = self.tracked_search.record_satisfaction(query_id, rating)
                if success:
                    logger.info(f"📊 Recorded satisfaction {rating}/5 for query {query_id}")

                    # TODO: Store additional feedback if provided
                    if feedback:
                        logger.debug(f"💬 Feedback for query {query_id}: {feedback}")

                return success
            else:
                logger.warning(f"⚠️ Cannot record satisfaction - tracking disabled")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to record satisfaction for query {query_id}: {e}")
            return False

    def track_result_interaction(self, query_id: int, result_type: str, result_id: int,
                                 position: int, action: str = "view") -> bool:
        """
        Track user interaction with search results.

        Args:
            query_id: Query ID from tracking system
            result_type: Type of result ('part', 'image', 'document')
            result_id: ID of the result item
            position: Position in search results (1-based)
            action: Action taken ('view', 'download', 'share')

        Returns:
            bool: True if tracked successfully
        """
        try:
            if self.tracked_search:
                success = self.tracked_search.track_result_click(
                    query_id=query_id,
                    result_type=result_type,
                    result_id=result_id,
                    click_position=position,
                    action_taken=action
                )

                if success:
                    logger.debug(f"👆 Tracked {action} on {result_type} {result_id} at position {position}")

                return success
            else:
                logger.debug(f"⚠️ Cannot track interaction - tracking disabled")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to track interaction: {e}")
            return False

    def get_search_analytics(self, days: int = 7) -> Dict[str, Any]:
        """
        Get comprehensive search analytics and performance metrics.

        Args:
            days: Number of days to include in report

        Returns:
            Analytics report with performance metrics
        """
        try:
            if self.tracked_search:
                # Get tracking analytics
                analytics = self.tracked_search.get_performance_report(days)

                # Add AistManager-specific metrics
                aist_metrics = self._get_aist_manager_metrics()
                analytics.update({
                    'aist_manager_metrics': aist_metrics,
                    'system_health': self._get_system_health_status(),
                    'generated_at': datetime.utcnow().isoformat(),
                    'report_type': 'comprehensive_search_analytics'
                })

                logger.info(f"📊 Generated {days}-day analytics report")
                return analytics

            else:
                return {
                    'status': 'limited',
                    'message': 'Search tracking not available',
                    'aist_manager_metrics': self._get_aist_manager_metrics(),
                    'system_health': self._get_system_health_status()
                }

        except Exception as e:
            logger.error(f"❌ Failed to generate analytics: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'error_type': 'analytics_generation_failed'
            }

    def _get_aist_manager_metrics(self) -> Dict[str, Any]:
        """Get AistManager-specific performance metrics."""
        return {
            'vector_search_available': self.vector_search_client is not None,
            'database_available': self.db_session is not None,
            'performance_thresholds': self.performance_thresholds,
            'performance_history_count': len(self.performance_history),
            'current_session_active': self.current_session_id is not None,
            'current_user': self.current_user_id,
            'tracking_enabled': self.tracked_search is not None
        }

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

        # Check for issues
        if not health['components']['search_tracking']:
            health['warnings'].append('Search tracking disabled - analytics limited')

        if not health['components']['vector_search']:
            health['warnings'].append('Vector search unavailable - similarity search limited')

        if not health['components']['database']:
            health['warnings'].append('Database unavailable - persistent search disabled')

        # Determine overall status
        critical_components = ['unified_search', 'database']
        if not all(health['components'][comp] for comp in critical_components):
            health['overall_status'] = 'degraded'
        elif len(health['warnings']) > 0:
            health['overall_status'] = 'healthy_with_warnings'

        return health

    # ===== ENHANCED QUERY ANALYSIS =====

    def analyze_query_intent(self, question: str) -> Dict[str, Any]:
        """
        Analyze query intent using the enhanced NLP system.

        Args:
            question: User's query

        Returns:
            Detailed intent analysis
        """
        try:
            if hasattr(self, 'unified_search_system') and self.unified_search_system:
                if hasattr(self.unified_search_system, 'analyze_user_input'):
                    analysis = self.unified_search_system.analyze_user_input(question)
                    logger.debug(f"🧠 Intent analysis completed for: '{question}'")
                    return analysis

            # Fallback intent analysis
            return {
                'status': 'limited',
                'message': 'Enhanced NLP analysis not available',
                'basic_analysis': {
                    'query': question,
                    'is_search_query': self.is_unified_search_query(question),
                    'analyzed_at': datetime.utcnow().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"❌ Intent analysis failed: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'query': question
            }

    # ===== BACKWARD COMPATIBILITY METHODS =====

    def execute_search(self, question: str, user_id: str = None, request_id: str = None) -> Dict[str, Any]:
        """
        Backward-compatible search method.
        Redirects to the new analytics-enabled search.
        """
        logger.debug("🔄 Redirecting legacy search to analytics-enabled search")

        # Set user if provided
        if user_id and user_id != self.current_user_id:
            self.set_current_user(user_id)

        return self.execute_search_with_analytics(question, request_id)

    # ===== CLEANUP AND MAINTENANCE =====

    def cleanup_session(self):
        """Clean up the current session and resources."""
        try:
            # End tracking session
            if self.current_session_id:
                self.end_user_session()

            # Clear request-specific data
            self.current_request_id = None
            self.start_time = None
            self.step_timings.clear()

            logger.debug("🧹 Session cleanup completed")

        except Exception as e:
            logger.error(f"❌ Session cleanup failed: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_session()

        # Close unified search session if it exists
        if hasattr(self, 'unified_search_system') and self.unified_search_system:
            if hasattr(self.unified_search_system, 'close_session'):
                try:
                    self.unified_search_system.close_session()
                except Exception as e:
                    logger.error(f"❌ Failed to close unified search session: {e}")

    def get_tracking_info(self) -> Dict[str, Any]:
        """Get current tracking system information."""
        return {
            'tracking_enabled': self.tracked_search is not None,
            'current_user': self.current_user_id,
            'current_session': self.current_session_id,
            'current_request': self.current_request_id,
            'system_health': self._get_system_health_status(),
            'capabilities': {
                'unified_search': hasattr(self, 'unified_search_system'),
                'vector_search': self.vector_search_client is not None,
                'search_analytics': self.tracked_search is not None,
                'intent_analysis': hasattr(self, 'unified_search_system'),
                'result_tracking': self.tracked_search is not None
            }
        }

    def begin_request(self, request_id=None):
        """
        Start timing a new request.

        Args:
            request_id: Optional request ID for tracking (backwards compatible)
        """
        self.start_time = time.time()

        # Store request_id if provided (for enhanced performance tracking)
        if request_id:
            self.current_request_id = request_id
            logger.debug(f"Request {request_id} started with performance tracking")
        else:
            self.current_request_id = None

    def track_performance_step(self, step_name, duration, metadata=None):
        """
        Track performance for individual steps in the question answering process.

        Args:
            step_name: Name of the step (e.g., 'keyword_search', 'ai_response')
            duration: Time taken for the step in seconds
            metadata: Additional metadata about the step
        """
        if not hasattr(self, 'step_timings'):
            self.step_timings = {}

        self.step_timings[step_name] = {
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        # Check against thresholds and log warnings
        threshold = self.performance_thresholds.get(step_name, 1.0)
        if duration > threshold:
            logger.warning(f"Step '{step_name}' exceeded threshold ({duration:.3f}s > {threshold}s) "
                           f"for request {self.current_request_id}")

        logger.debug(f"Step '{step_name}' completed in {duration:.3f}s for request {self.current_request_id}")

    def get_performance_summary(self):
        """Get a summary of current request performance."""
        if not hasattr(self, 'start_time') or not self.start_time:
            return {}

        total_time = time.time() - self.start_time

        return {
            'total_time': total_time,
            'start_time': self.start_time,
            'performance_score': self._calculate_performance_score(total_time),
            'request_id': getattr(self, 'current_request_id', None)
        }

    def _calculate_performance_score(self, total_time):
        """Calculate a performance score (1-10, higher is better)."""
        if total_time < 1.0:
            return 10
        elif total_time < 2.0:
            return 8
        elif total_time < 3.0:
            return 6
        elif total_time < 5.0:
            return 4
        elif total_time < 10.0:
            return 2
        else:
            return 1

    def record_request_performance(self, user_id, question, result_status, method_used):
        """Record performance data for analysis and optimization."""
        if not hasattr(self, 'performance_history'):
            self.performance_history = []

        performance_record = {
            'timestamp': datetime.now().isoformat(),
            'request_id': self.current_request_id,
            'user_id': user_id,
            'question_length': len(question),
            'result_status': result_status,
            'method_used': method_used,
            'performance_summary': self.get_performance_summary()
        }

        # Keep only last 1000 records to prevent memory issues
        self.performance_history.append(performance_record)
        if len(self.performance_history) > 1000:
            self.performance_history.pop(0)

    def get_performance_analytics(self, hours=24):
        """Get performance analytics for the specified time period."""
        if not hasattr(self, 'performance_history'):
            return {}

        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Filter recent records
        recent_records = [
            record for record in self.performance_history
            if datetime.fromisoformat(record['timestamp']) > cutoff_time
        ]

        if not recent_records:
            return {'message': 'No performance data available for the specified period'}

        # Calculate analytics
        total_requests = len(recent_records)
        method_stats = defaultdict(lambda: {'count': 0, 'total_time': 0, 'avg_time': 0})
        step_stats = defaultdict(lambda: {'count': 0, 'total_time': 0, 'avg_time': 0})

        for record in recent_records:
            method = record.get('method_used', 'unknown')
            method_stats[method]['count'] += 1

            perf_summary = record.get('performance_summary', {})
            total_time = perf_summary.get('total_time', 0)
            method_stats[method]['total_time'] += total_time

            # Analyze step performance
            steps = perf_summary.get('steps', {})
            for step_name, step_data in steps.items():
                step_duration = step_data.get('duration', 0)
                step_stats[step_name]['count'] += 1
                step_stats[step_name]['total_time'] += step_duration

        # Calculate averages
        for method_data in method_stats.values():
            if method_data['count'] > 0:
                method_data['avg_time'] = method_data['total_time'] / method_data['count']

        for step_data in step_stats.values():
            if step_data['count'] > 0:
                step_data['avg_time'] = step_data['total_time'] / step_data['count']

        # Calculate overall metrics
        total_time_all = sum(record.get('performance_summary', {}).get('total_time', 0) for record in recent_records)
        avg_response_time = total_time_all / total_requests if total_requests > 0 else 0

        # Performance distribution
        performance_scores = [
            record.get('performance_summary', {}).get('performance_score', 0)
            for record in recent_records
        ]

        analytics = {
            'period_hours': hours,
            'total_requests': total_requests,
            'avg_response_time': round(avg_response_time, 3),
            'avg_performance_score': round(sum(performance_scores) / len(performance_scores),
                                           2) if performance_scores else 0,
            'method_performance': dict(method_stats),
            'step_performance': dict(step_stats),
            'performance_distribution': {
                'excellent': len([s for s in performance_scores if s >= 8]),
                'good': len([s for s in performance_scores if 6 <= s < 8]),
                'average': len([s for s in performance_scores if 4 <= s < 6]),
                'poor': len([s for s in performance_scores if 2 <= s < 4]),
                'very_poor': len([s for s in performance_scores if s < 2])
            },
            'slowest_requests': sorted(
                [
                    {
                        'request_id': r.get('request_id'),
                        'total_time': r.get('performance_summary', {}).get('total_time', 0),
                        'method': r.get('method_used'),
                        'question_length': r.get('question_length', 0)
                    }
                    for r in recent_records
                ],
                key=lambda x: x['total_time'],
                reverse=True
            )[:10]
        }

        return analytics

    def get_response_time(self):
        """Calculate the response time so far."""
        if self.start_time:
            return time.time() - self.start_time
        return 0

    @with_request_id
    def find_most_relevant_document(self, question, session=None, request_id=None):
        """
        Find the most relevant document with Windows-compatible timeout.
        FIXED: Removed Unix-only signal.SIGALRM usage.
        """
        search_start = time.time()
        logger.debug(f"Finding most relevant document for question: {question[:50]}...")

        # Use the provided session or the class's session
        local_session = None
        if not session:
            if self.db_session:
                session = self.db_session
            else:
                local_session = self.db_config.get_main_session()
                session = local_session

        try:
            # Get embedding model information
            embedding_model_name = ModelsConfig.get_config_value('embedding', 'CURRENT_MODEL', 'NoEmbeddingModel')

            # Check if embeddings are disabled
            if embedding_model_name == "NoEmbeddingModel":
                logger.info("Embeddings are disabled. Returning None for document search.")
                return None

            # FIXED: Use Windows-compatible timeout instead of signal.SIGALRM
            def search_operation():
                # Use the vector search client with timeout protection
                if hasattr(self, 'vector_search_client') and self.vector_search_client:
                    # Use optimized vector search
                    vector_results = self.vector_search_client.search(
                        question,
                        limit=1
                    )

                    if vector_results:
                        best_result = vector_results[0]
                        doc_id = best_result['id']
                        similarity = best_result['similarity']

                        # Fetch the full document
                        document = session.query(Document).get(doc_id)
                        if document:
                            search_time = time.time() - search_start
                            logger.info(
                                f"Found most relevant document (ID: {doc_id}, similarity: {similarity:.4f}) in {search_time:.3f}s")
                            return document

                # Fallback: try simple recent document search
                logger.debug("Using fallback document search")
                recent_docs = session.query(Document).limit(5).all()
                if recent_docs:
                    search_time = time.time() - search_start
                    logger.info(f"Fallback: returning recent document in {search_time:.3f}s")
                    return recent_docs[0]

                return None

            # Execute with 3-second timeout
            result, error = self._execute_with_timeout(search_operation, 3.0)

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
    def answer_question(self, user_id, question, client_type=None, request_id=None):
        """
        Enhanced answer_question with unified search and Windows-compatible timeouts.
        NOW WITH SYNONYM INTELLIGENCE!

        Now automatically routes between:
        - Unified Search: For specific queries like "What's in room 2312?" (with synonym expansion)
        - Conversational AI: For general questions and explanations
        """
        # Call the updated begin_request method
        try:
            self.begin_request(request_id)
        except TypeError:
            # Fallback for compatibility
            self.begin_request()

        logger.debug(f"Processing question for user {user_id}: {question}")
        local_session = None
        session_id = None
        combined_context = ""
        method_used = "unknown"

        # Set maximum response time
        max_response_time = 400.0  # 5 seconds max
        start_time = time.time()

        # Check for session end request
        if question.lower() == "end session please":
            logger.info("User requested to end the session")
            return {"status": "end_session"}

        # Define reset phrases that will clear the conversation context
        reset_phrases = [
            "clear context", "reset context", "clear conversation", "reset conversation",
            "forget previous context", "start fresh", "start over", "clear chat history",
            "reset chat", "forget what i said"
        ]

        # Check if the question contains any reset phrases
        is_reset_request = any(phrase in question.lower() for phrase in reset_phrases)

        try:
            # Set up session management
            if not self.db_session:
                db_config = DatabaseConfig()
                local_session = db_config.get_main_session()
                session = local_session
            else:
                session = self.db_session

            # ADD SESSION ROLLBACK: Ensure clean session state
            try:
                if session:
                    session.rollback()  # Clear any pending state
                    logger.debug("Session state cleared for new request")
            except Exception as e:
                logger.debug(f"Session rollback note: {e}")

            # Get or create user session for conversation history
            latest_session = self.get_session(user_id, session)
            if latest_session:
                session_id = latest_session.session_id
                session_data = latest_session.session_data if latest_session.session_data else []
                session_data.append(question)
                logger.debug(f"Using existing session {session_id} with {len(session_data)} messages")
            else:
                session_id = self.create_session(user_id, question, session)
                session_data = [question]
                logger.debug(f"Created new session {session_id} for user {user_id}")

            # Handle reset request if detected
            if is_reset_request and session_id:
                logger.info(f"Reset request detected: '{question}'")
                success = ChatSession.clear_conversation_summary(session_id, session)
                reset_response = "Context has been reset. I've cleared our conversation history."
                self.update_session(session_id, session_data, reset_response, session)
                formatted_answer = self.format_response(reset_response, client_type)
                method_used = "reset_context"
                return {"status": "success", "answer": formatted_answer, "method": method_used}

            # ================================================================
            # NEW: ENHANCED UNIFIED SEARCH WITH SYNONYM INTELLIGENCE
            # ================================================================
            if hasattr(self, 'is_unified_search_query') and self.is_unified_search_query(question):
                logger.info(f"Detected unified search query: {question}")

                # STEP 1: Try original unified search first
                search_result = self.execute_unified_search(question, user_id, request_id)

                # STEP 2: If no results, try with synonym expansion
                if not search_result or search_result.get('status') != 'success' or search_result.get('total_results',
                                                                                                      0) == 0:
                    logger.info("Original search found no results, trying with synonym expansion...")
                    search_result = self._execute_unified_search_with_synonyms(question, user_id, request_id, session)

                # STEP 3: Process successful results
                if search_result and search_result.get('status') == 'success' and search_result.get('total_results',
                                                                                                    0) > 0:
                    # Format unified search results
                    formatted_response = self._format_unified_search_response(search_result, client_type)

                    # Record the interaction
                    self.record_interaction(user_id, question, formatted_response)
                    self.update_session(session_id, session_data, formatted_response, session)

                    # Update conversation summary
                    summarized_data = session_data[-3:] if len(session_data) > 3 else session_data
                    ChatSession.update_conversation_summary(session_id, summarized_data, session)

                    total_time = time.time() - start_time
                    search_method = "unified_search_with_synonyms" if search_result.get(
                        'synonym_enhanced') else "unified_search"
                    logger.info(
                        f"Question answered using {search_method} in {total_time:.3f}s with {search_result.get('total_results', 0)} results")

                    return {
                        "status": "success",
                        "answer": formatted_response,
                        "method": search_method,
                        "search_results": search_result,
                        "is_structured_data": True,
                        "synonym_enhanced": search_result.get('synonym_enhanced', False)
                    }
                else:
                    # Unified search failed completely, log and fall back to conversational AI
                    logger.warning(
                        f"Unified search with synonyms failed, falling back to AI: {search_result.get('message', 'Unknown error') if search_result else 'No search result'}")

            # ================================================================
            # EXISTING CONVERSATIONAL AI LOGIC (when unified search doesn't apply or fails)
            # ================================================================
            logger.debug("Processing as conversational AI query")

            # OPTIMIZATION: Check remaining time before each expensive operation
            elapsed_time = time.time() - start_time

            # Strategy 1: Try keyword search first (fastest)
            if elapsed_time < max_response_time - 4:  # Leave 4 seconds for other operations
                with log_timed_operation("keyword_search"):
                    keyword_result = self.try_keyword_search(question)
                    if keyword_result.get('success'):
                        answer = self.format_response(keyword_result['answer'], client_type,
                                                      keyword_result.get('results', []))
                        self.record_interaction(user_id, question, keyword_result['answer'])
                        self.update_session(session_id, session_data, answer, session)

                        # Update conversation summary
                        summarized_data = session_data[-3:] if len(session_data) > 3 else session_data
                        ChatSession.update_conversation_summary(session_id, summarized_data, session)

                        method_used = "keyword"
                        return {"status": "success", "answer": answer, "method": method_used}

            # Strategy 2: Try full-text search (medium speed)
            elapsed_time = time.time() - start_time
            if elapsed_time < max_response_time - 2:  # Leave 2 seconds for AI response
                fulltext_content = ""
                with log_timed_operation("fulltext_search"):
                    try:
                        if session:
                            # ADD SESSION ROLLBACK: Clear state before fulltext search
                            try:
                                session.rollback()
                            except Exception:
                                pass

                            fts_documents = CompleteDocument.search_by_text(
                                question,
                                session=session,
                                similarity_threshold=50,
                                with_links=False
                            )

                            if fts_documents and isinstance(fts_documents, list) and len(fts_documents) > 0:
                                fulltext_content += "\n\nFULL-TEXT SEARCH RESULTS:\n"
                                for i, doc in enumerate(fts_documents[:3], 1):
                                    title = getattr(doc, 'title', f"Document #{getattr(doc, 'id', i)}")
                                    content = getattr(doc, 'content', '')
                                    snippet = content[:300] + "..." if content and len(content) > 300 else content
                                    fulltext_content += f"{i}. {title}: {snippet}\n\n"

                                logger.info(
                                    f"Added {len(fts_documents[:3])} documents from full-text search to context")
                                combined_context += fulltext_content
                    except Exception as e:
                        logger.error(f"Error collecting full-text search results: {e}", exc_info=True)
                        # ADD SESSION ROLLBACK: Clear state if fulltext search failed
                        try:
                            if session:
                                session.rollback()
                        except Exception:
                            pass

            # Strategy 3: Try vector search (slowest, with timeout)
            most_relevant_doc = None
            elapsed_time = time.time() - start_time
            if elapsed_time < max_response_time - 1.5:  # Leave 1.5 seconds for AI response
                with log_timed_operation("find_relevant_document"):
                    # ADD SESSION ROLLBACK: Clear state before vector search
                    try:
                        if session:
                            session.rollback()
                    except Exception:
                        pass

                    most_relevant_doc = self.find_most_relevant_document(question, session)
                    if most_relevant_doc:
                        logger.info(f"Found most relevant document with ID {most_relevant_doc.id}")
                        doc_content = most_relevant_doc.content
                        doc_snippet = doc_content[:1000] + "..." if len(doc_content) > 1000 else doc_content
                        combined_context += f"\n\nMOST RELEVANT DOCUMENT:\n{doc_snippet}\n\n"

            # Strategy 4: Generate AI response (always try this)
            ai_model = self._ensure_ai_model()
            if not ai_model:
                logger.error("Failed to load AI model")
                return {"status": "error", "error": "Failed to load AI model"}

            # Get conversation history for context
            try:
                # ADD SESSION ROLLBACK: Clear state before getting conversation summary
                if session:
                    session.rollback()
            except Exception:
                pass

            conversation_summary = ChatSession.get_conversation_summary(session_id, session)
            conversation_context = ""
            if conversation_summary:
                recent_msgs = conversation_summary[-5:] if len(conversation_summary) > 5 else conversation_summary
                conversation_context = "\n".join([f"User: {msg}" if i % 2 == 0 else f"Assistant: {msg}"
                                                  for i, msg in enumerate(recent_msgs)])

            # Create optimized prompt (shorter for faster processing)
            if combined_context:
                # Use context if available
                comprehensive_prompt = (
                    "Answer based on this information:\n"
                    f"{combined_context[:1500]}...\n\n"  # Limit context size
                    f"Question: {question}\n"
                    "Provide a concise answer:"
                )
                method_used = "ai_with_context"
            else:
                # Fallback to direct AI response
                comprehensive_prompt = f"Please answer this question: {question}"
                method_used = "ai_direct"

            # Get AI response with Windows-compatible timeout
            with log_timed_operation("ai_comprehensive_response"):
                def ai_operation():
                    return ai_model.get_response(comprehensive_prompt)

                remaining_time = max_response_time - (time.time() - start_time)
                ai_answer, error = self._execute_with_timeout(
                    ai_operation,
                    max(300.0, remaining_time)  # At least 1 second for AI
                )

                if error:
                    logger.warning("AI response timed out, using fallback")
                    ai_answer = "I'm processing your question but it's taking longer than expected. Could you please rephrase or try again?"
                    method_used = "ai_timeout_fallback"

            # Format and record response
            formatted_answer = self.format_response(ai_answer, client_type)
            self.record_interaction(user_id, question, ai_answer)

            # Update session with the interaction
            self.update_session(session_id, session_data, ai_answer, session)

            # Update conversation summary
            summarized_session_data = session_data[-3:] if len(session_data) > 3 else session_data
            summarized_session_data.append(question)
            summarized_session_data.append(ai_answer)
            ChatSession.update_conversation_summary(session_id, summarized_session_data, session)

            total_time = time.time() - start_time
            logger.info(f"Question answered using '{method_used}' in {total_time:.3f}s")

            return {"status": "success", "answer": formatted_answer, "method": method_used}

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Error in answer_question after {total_time:.3f}s: {e}", exc_info=True)

            # ADD SESSION ROLLBACK: Clear state if there was an error
            try:
                if session:
                    session.rollback()
                    logger.debug("Session rolled back due to error in answer_question")
            except Exception:
                pass

            return {"status": "error", "error": str(e)}
        finally:
            if local_session:
                local_session.close()
                logger.debug("Local database session closed")

    # ================================================================
    # NEW SYNONYM-ENHANCED SEARCH METHOD
    # Add this method to your AistManager class
    # ================================================================

    def _execute_unified_search_with_synonyms(self, question: str, user_id: str, request_id: str, session) -> Dict[
        str, Any]:
        """
        Execute unified search with synonym expansion using the database we created.
        This method tries multiple query variations using your synonym database.
        """
        import re

        logger.info(f"Executing synonym-enhanced search for: {question}")

        try:
            # Step 1: Extract key terms from the query (remove search words)
            stop_words = {'search', 'find', 'show', 'get', 'locate', 'display', 'for', 'the', 'a', 'an', 'with'}
            words = re.findall(r'\b\w+\b', question.lower())
            key_terms = [word for word in words if word not in stop_words and len(word) > 2]

            logger.debug(f"Key terms extracted: {key_terms}")

            # Step 2: Get synonyms for each key term
            expanded_terms = {}
            for term in key_terms:
                synonyms = self._get_synonyms_for_term(term, session)
                if synonyms:
                    expanded_terms[term] = synonyms
                    logger.debug(f"Found synonyms for '{term}': {synonyms}")

            # Step 3: Create enhanced query variations
            enhanced_queries = self._create_enhanced_queries(question, expanded_terms)
            logger.info(f"Created {len(enhanced_queries)} query variations: {enhanced_queries}")

            # Step 4: Try each query variation
            best_result = None
            best_score = 0

            for i, query_variant in enumerate(enhanced_queries):
                logger.debug(f"Trying query variation {i + 1}/{len(enhanced_queries)}: {query_variant}")

                try:
                    # Use your existing unified search system
                    result = self.execute_unified_search(query_variant, user_id, request_id)

                    if result and result.get('status') == 'success':
                        result_score = result.get('total_results', 0)
                        logger.debug(f"Query '{query_variant}' found {result_score} results")

                        if result_score > best_score:
                            best_score = result_score
                            best_result = result
                            best_result['matched_query'] = query_variant
                            best_result['synonym_enhanced'] = query_variant != question
                            best_result['expanded_terms'] = expanded_terms

                            # If we found good results, we can stop
                            if result_score >= 5:  # Good enough threshold
                                logger.info(f"Found sufficient results ({result_score}) with query: {query_variant}")
                                break

                except Exception as e:
                    logger.warning(f"Search failed for query variant '{query_variant}': {e}")
                    continue

            # Step 5: Return the best result
            if best_result and best_score > 0:
                logger.info(
                    f"Synonym search successful: {best_score} results with query '{best_result.get('matched_query')}'")
                return best_result
            else:
                logger.warning("No results found even with synonym expansion")
                return {
                    'status': 'no_results',
                    'message': f"No results found for '{question}' even with synonym variations",
                    'total_results': 0,
                    'expanded_terms': expanded_terms,
                    'tried_queries': enhanced_queries
                }

        except Exception as e:
            logger.error(f"Error in synonym-enhanced search: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': f"Synonym search failed: {str(e)}",
                'total_results': 0
            }

    # Fixed SQL query for _get_synonyms_for_term method
    # Replace the SQL query section in your method with this:

    def _get_synonyms_for_term(self, term: str, session) -> List[str]:
        """Get synonyms for a specific term with comprehensive error handling and transaction management"""

        # CRITICAL FIX: Always ensure clean transaction state before any database operations
        try:
            if session.in_transaction():
                session.rollback()
                logger.debug(f"🔄 Rolled back existing transaction before synonym lookup for '{term}'")
        except Exception as rollback_error:
            logger.warning(f"⚠️ Could not rollback transaction: {rollback_error}")
            # If we can't rollback, skip database lookup entirely
            return self._get_fallback_synonyms(term)

        # Try Method 1: Fixed PostgreSQL query (removed ORDER BY to avoid DISTINCT issues)
        try:
            from sqlalchemy import text

            # FIXED: Removed ORDER BY to avoid PostgreSQL DISTINCT error
            sql_query = text("""
                SELECT es.canonical_value, es.synonym_value
                FROM entity_synonym es
                JOIN entity_type et ON es.entity_type_id = et.id
                WHERE (
                    LOWER(es.canonical_value) LIKE :term OR 
                    LOWER(es.synonym_value) LIKE :term
                )
                AND et.name = 'EQUIPMENT_TYPE'
                AND es.confidence_score > 0.5
                LIMIT 10
            """)

            result = session.execute(sql_query, {'term': f'%{term.lower()}%'})

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
                logger.debug(f"✅ Database synonyms for '{term}': {synonyms_list}")
                return synonyms_list
            else:
                logger.debug(f"ℹ️ No database synonyms found for '{term}'")

        except Exception as e:
            logger.debug(f"⚠️ Database synonym lookup failed for '{term}': {e}")

            # CRITICAL: Always rollback on database error to prevent failed transaction state
            try:
                session.rollback()
                logger.debug(f"🔄 Rolled back failed transaction for '{term}'")
            except Exception as rollback_error:
                logger.warning(f"⚠️ Failed to rollback after error: {rollback_error}")

        # Try Method 2: Alternative SQLAlchemy Core approach (if Method 1 fails)
        try:
            if session.in_transaction():
                session.rollback()

            from sqlalchemy import MetaData, Table, select, and_, or_

            # Reflect the tables
            metadata = MetaData()
            entity_synonym = Table('entity_synonym', metadata, autoload_with=session.bind)
            entity_type = Table('entity_type', metadata, autoload_with=session.bind)

            # Build query without ORDER BY
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
                    entity_type.c.name == 'EQUIPMENT_TYPE'
                )
            ).limit(5)

            result = session.execute(query)

            # Collect related terms
            related_terms = set()
            for row in result:
                canonical = row[0].lower() if row[0] else ''
                synonym = row[1].lower() if row[1] else ''

                if canonical and canonical != term.lower():
                    related_terms.add(canonical)
                if synonym and synonym != term.lower():
                    related_terms.add(synonym)

            synonyms_list = list(related_terms)[:3]

            if synonyms_list:
                logger.debug(f"✅ SQLAlchemy Core synonyms for '{term}': {synonyms_list}")
                return synonyms_list

        except Exception as e:
            logger.debug(f"⚠️ SQLAlchemy Core synonym lookup failed for '{term}': {e}")

            # Rollback on error
            try:
                session.rollback()
            except:
                pass

        # Method 3: Fallback to hardcoded synonyms
        return self._get_fallback_synonyms(term)

    # Alternative Method 2: Using SQLAlchemy Core (if Method 1 doesn't work)
    def _get_synonyms_for_term_alternative(self, term: str, session) -> List[str]:
        """Alternative method using SQLAlchemy Core"""
        try:
            from sqlalchemy import MetaData, Table, select, and_, or_

            # Reflect the tables
            metadata = MetaData()
            entity_synonym = Table('entity_synonym', metadata, autoload_with=session.bind)
            entity_type = Table('entity_type', metadata, autoload_with=session.bind)

            # Build the query
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
                    entity_type.c.name == 'EQUIPMENT_TYPE'
                )
            ).limit(10)

            result = session.execute(query)

            # Collect related terms
            related_terms = set()
            for row in result:
                canonical = row[0].lower() if row[0] else ''
                synonym = row[1].lower() if row[1] else ''

                if canonical and canonical != term.lower():
                    related_terms.add(canonical)
                if synonym and synonym != term.lower():
                    related_terms.add(synonym)

            return list(related_terms)[:5]

        except Exception as e:
            logger.warning(f"Alternative synonym lookup failed for {term}: {e}")
            return []

    # Method 3: Simple fallback using basic synonyms (if database fails)
    def _get_synonyms_for_term_fallback(self, term: str, session) -> List[str]:
        """Fallback method with hardcoded synonyms based on your data"""

        # Based on your synonym database we created
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

            # Additional search action synonyms
            'find': ['search', 'locate', 'get', 'show', 'display'],
            'part': ['component', 'spare', 'item', 'piece'],
            'image': ['picture', 'photo', 'pic', 'diagram']
        }

        # Look for exact matches or partial matches
        synonyms = []
        term_lower = term.lower()

        # Exact match
        if term_lower in fallback_synonyms:
            synonyms = fallback_synonyms[term_lower][:4]  # Top 4 synonyms
            logger.debug(f"Fallback synonyms for '{term}': {synonyms}")

        # Partial match (for compound terms like "ball valve")
        if not synonyms:
            for key, values in fallback_synonyms.items():
                if key in term_lower or term_lower in key:
                    synonyms = values[:3]  # Top 3 for partial matches
                    logger.debug(f"Partial fallback synonyms for '{term}' (matched '{key}'): {synonyms}")
                    break

        return synonyms

    # UPDATED: Complete _get_synonyms_for_term method with all fallbacks
    def _get_synonyms_for_term(self, term: str, session) -> List[str]:
        """Get synonyms for a specific term with multiple fallback methods"""

        # Try Method 1: Raw SQL query (most reliable)
        try:
            from sqlalchemy import text

            sql_query = text("""
                SELECT DISTINCT 
                    es.canonical_value,
                    es.synonym_value
                FROM entity_synonym es
                JOIN entity_type et ON es.entity_type_id = et.id
                WHERE (
                    LOWER(es.canonical_value) LIKE :term OR 
                    LOWER(es.synonym_value) LIKE :term
                )
                AND et.name = 'EQUIPMENT_TYPE'
                ORDER BY es.confidence_score DESC
                LIMIT 10
            """)

            result = session.execute(sql_query, {'term': f'%{term.lower()}%'})

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
                logger.debug(f"Database synonyms for '{term}': {synonyms_list}")
                return synonyms_list

        except Exception as e:
            logger.debug(f"Database synonym lookup failed for {term}: {e}, trying fallback...")

        # Fallback: Use hardcoded synonyms based on your database
        fallback_synonyms = {
            'valve': ['valves', 'control valve', 'ball valve', 'gate valve', 'check valve'],
            'bearing': ['bearings', 'ball bearing', 'roller bearing', 'thrust bearing'],
            'switch': ['switches', 'limit switch', 'pressure switch', 'safety switch'],
            'motor': ['motors', 'electric motor', 'servo motor', 'ac motor'],
            'belt': ['belts', 'drive belt', 'v-belt', 'timing belt'],
            'cable': ['cables', 'power cable', 'control cable'],
            'sensor': ['sensors', 'temperature sensor', 'pressure sensor'],
            'seal': ['seals', 'oil seal', 'hydraulic seal'],
            'relay': ['relays', 'control relay', 'power relay'],
            'pump': ['pumps', 'centrifugal pump', 'hydraulic pump'],
            'spring': ['springs'], 'filter': ['filters'], 'gear': ['gears'],
            'tube': ['tubes'], 'hose': ['hoses'], 'wire': ['wires'], 'fan': ['fans']
        }

        term_lower = term.lower()

        # Exact match
        if term_lower in fallback_synonyms:
            synonyms = fallback_synonyms[term_lower][:4]
            logger.debug(f"Fallback synonyms for '{term}': {synonyms}")
            return synonyms

        # Partial match for compound terms
        for key, values in fallback_synonyms.items():
            if key in term_lower or term_lower in key:
                synonyms = values[:3]
                logger.debug(f"Partial fallback synonyms for '{term}' (matched '{key}'): {synonyms}")
                return synonyms

        # No synonyms found
        logger.debug(f"No synonyms found for '{term}'")
        return []

    def _create_enhanced_queries(self, original_query: str, expanded_terms: Dict[str, List[str]]) -> List[str]:
        """Create query variations using synonyms"""
        queries = [original_query]

        # For each term with synonyms, create variations
        for original_term, synonyms in expanded_terms.items():
            if not synonyms:
                continue

            # Try replacing with the best synonyms
            for synonym in synonyms[:3]:  # Top 3 synonyms per term
                enhanced_query = original_query.replace(original_term, synonym)
                if enhanced_query != original_query and enhanced_query not in queries:
                    queries.append(enhanced_query)

        # Create combination queries (be careful not to create too many)
        if len(expanded_terms) >= 2:
            # Try one combination with the first synonym of each term
            combined_query = original_query
            for original_term, synonyms in list(expanded_terms.items())[:2]:
                if synonyms:
                    combined_query = combined_query.replace(original_term, synonyms[0])

            if combined_query != original_query and combined_query not in queries:
                queries.append(combined_query)

        return queries[:6]  # Limit to 6 total queries to avoid performance issues

    def _format_unified_search_response(self, search_result: Dict[str, Any], client_type: str = None) -> str:
        """
        Format unified search results into a response string.
        Different formatting for different client types (web, mobile, etc.)
        """
        if client_type == 'json':
            # Return raw JSON for API clients
            import json
            return json.dumps(search_result, indent=2)

        # Create a natural language response with structured data
        summary = search_result.get('summary', 'Found search results')
        results_by_type = search_result.get('results_by_type', {})

        response_parts = [summary, "\n"]

        # Format each result type
        for result_type, results in results_by_type.items():
            if not results:
                continue

            type_title = result_type.title().replace('_', ' ')
            response_parts.append(f"\n**{type_title}:**")

            for i, result in enumerate(results[:5], 1):  # Limit to 5 per type
                if result_type == 'images':
                    title = result.get('title', f'Image {i}')
                    desc = result.get('description', '')
                    desc_part = f" - {desc}" if desc else ""
                    response_parts.append(f"{i}. {title}{desc_part}")
                    if result.get('full_url'):
                        response_parts.append(f"   🖼️ [View Image]({result['full_url']})")

                elif result_type == 'documents':
                    title = result.get('title', f'Document {i}')
                    preview = result.get('preview', '')
                    response_parts.append(f"{i}. {title}")
                    if preview:
                        response_parts.append(f"   📄 {preview}")
                    if result.get('url'):
                        response_parts.append(f"   [Open Document]({result['url']})")

                elif result_type == 'parts':
                    part_number = result.get('part_number', 'Unknown')
                    name = result.get('name', 'Unknown Part')
                    manufacturer = result.get('manufacturer', '')
                    mfg_part = f" ({manufacturer})" if manufacturer else ""
                    response_parts.append(f"{i}. Part {part_number}: {name}{mfg_part}")

                    # Add usage locations
                    locations = result.get('usage_locations', [])
                    if locations:
                        loc_str = ', '.join(locations[:3])
                        response_parts.append(f"   📍 Used in: {loc_str}")

                    # Add related images count
                    img_count = len(result.get('related_images', []))
                    if img_count > 0:
                        response_parts.append(f"   🖼️ {img_count} related image{'s' if img_count > 1 else ''}")


                elif result_type == 'positions':

                    # FIX: Handle None location properly

                    location = result.get('location', {})

                    if not location or not isinstance(location, dict):
                        location = {}

                    area = location.get('area', 'Unknown Area')

                    equipment = location.get('equipment_group', '')

                    eq_part = f" - {equipment}" if equipment else ""

                    response_parts.append(f"{i}. {area}{eq_part}")

                    # FIX: Handle None contents properly

                    contents = result.get('contents', {})

                    if not contents or not isinstance(contents, dict):
                        contents = {}

                    part_count = contents.get('part_count', 0)

                    img_count = contents.get('image_count', 0)

                    if part_count > 0 or img_count > 0:

                        content_info = []

                        if part_count > 0:
                            content_info.append(f"{part_count} part{'s' if part_count > 1 else ''}")

                        if img_count > 0:
                            content_info.append(f"{img_count} image{'s' if img_count > 1 else ''}")

                        response_parts.append(f"   📦 Contains: {', '.join(content_info)}")

                elif result_type == 'drawings':
                    number = result.get('number', 'Unknown')
                    name = result.get('name', 'Unknown Drawing')
                    revision = result.get('revision', '')
                    rev_part = f" (Rev. {revision})" if revision else ""
                    response_parts.append(f"{i}. Drawing {number}: {name}{rev_part}")
                    if result.get('view_url'):
                        response_parts.append(f"   📐 [View Drawing]({result['view_url']})")

                elif result_type == 'equipment':
                    name = result.get('name', f'Equipment {i}')
                    eq_type = result.get('equipment_type', '')
                    location = result.get('location', '')
                    status = result.get('status', '')

                    type_part = f" ({eq_type})" if eq_type else ""
                    response_parts.append(f"{i}. {name}{type_part}")

                    details = []
                    if location:
                        details.append(f"Location: {location}")
                    if status:
                        details.append(f"Status: {status}")

                    if details:
                        response_parts.append(f"   ⚙️ {', '.join(details)}")

                elif result_type == 'procedures':
                    name = result.get('name', f'Procedure {i}')
                    difficulty = result.get('difficulty', '')
                    est_time = result.get('estimated_time', '')

                    response_parts.append(f"{i}. {name}")

                    proc_details = []
                    if difficulty:
                        proc_details.append(f"Difficulty: {difficulty}")
                    if est_time:
                        proc_details.append(f"Time: {est_time}")

                    if proc_details:
                        response_parts.append(f"   🔧 {', '.join(proc_details)}")

                else:
                    # Generic formatting for other types
                    title = result.get('title') or result.get('name') or f'Item {i}'
                    response_parts.append(f"{i}. {title}")

        # Add quick actions if available
        quick_actions = search_result.get('quick_actions', [])
        if quick_actions:
            response_parts.append("\n**Quick Actions:**")
            for action in quick_actions:
                label = action.get('label', 'Action')
                url = action.get('url', '#')
                response_parts.append(f"• [🔗 {label}]({url})")

        # Add related searches
        related_searches = search_result.get('related_searches', [])
        if related_searches:
            response_parts.append("\n**Related Searches:**")
            for suggestion in related_searches[:3]:  # Limit to 3
                response_parts.append(f"• {suggestion}")

        # Add search performance info for debug clients
        if client_type == 'debug':
            search_time = search_result.get('search_time_ms', 0)
            confidence = search_result.get('confidence_score', 0)
            method = search_result.get('search_method', 'unknown')
            response_parts.append(
                f"\n*Search completed in {search_time}ms using {method} (confidence: {confidence:.2f})*")

        # Filter out None values before joining
        response_parts = [part for part in response_parts if part is not None]
        if response_parts:
            return "\n".join(response_parts)
        else:
            return "Search completed successfully but no formatted response was generated."

    @with_request_id
    def try_keyword_search(self, question):
        """Try to answer using keyword search with improved error handling."""
        logger.debug('Attempting keyword search')
        local_session = None
        try:
            # Use the provided session or get a new one from DatabaseConfig
            if not self.db_session:
                local_session = self.db_config.get_main_session()
                session = local_session
            else:
                session = self.db_session

            # Use KeywordAction to find the best match with error handling
            try:
                keyword, action, details = KeywordAction.find_best_match(question, session)
            except Exception as keyword_error:
                logger.error(f"KeywordAction.find_best_match failed: {keyword_error}")
                # Continue with search instead of failing completely
                keyword, action, details = None, None, None

            if keyword and action:
                logger.info(f"Found matching keyword: {keyword} with action: {action}")

                # Execute the appropriate action based on the keyword match
                try:
                    result = self.execute_keyword_action(keyword, action, details, question)
                except Exception as action_error:
                    logger.error(f"execute_keyword_action failed: {action_error}")
                    # Fallback to simple search
                    result = self.perform_advanced_keyword_search(question)

                if result:
                    # Format the response
                    if isinstance(result, dict):
                        entity_type = result.get('entity_type', 'generic')
                        results = result.get('results', [])
                    else:
                        entity_type = 'generic'
                        results = []

                    # Format based on entity type and results
                    if results:
                        answer = self.format_entity_results(results, entity_type)
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
            # Only close the session if we created it locally
            if local_session:
                local_session.close()
                logger.debug("Closed local database session")

    def execute_keyword_action(self, keyword, action, details, original_question):
        """Execute the appropriate action based on the keyword match."""
        logger.debug(f"Executing action for keyword '{keyword}': {action}")

        try:
            # Simple case - action is a direct response
            if action.startswith("RESPOND:"):
                response_text = action[8:].strip()  # Remove "RESPOND:" prefix
                return {"entity_type": "response", "results": [{"text": response_text}]}

            # Search case - action triggers a search
            elif action.startswith("SEARCH:"):
                search_type = action[7:].strip()  # Remove "SEARCH:" prefix
                if search_type == "KEYWORD":
                    return {"entity_type": "search", "results": self.perform_advanced_keyword_search(original_question)}
                elif search_type == "FTS":
                    # Use the CompleteDocument class method directly
                    local_session = None
                    try:
                        if not self.db_session:
                            local_session = self.db_config.get_main_session()
                            session = local_session
                        else:
                            session = self.db_session

                        # Call the search_by_text method from CompleteDocument
                        fts_results = CompleteDocument.search_by_text(
                            original_question,
                            session=session,
                            similarity_threshold=70,  # Lower threshold for better recall
                            with_links=False  # Get document objects instead of HTML
                        )

                        # Format the results to match our expected structure
                        return {"entity_type": "document", "results": self.format_fts_results(fts_results)}
                    finally:
                        if local_session:
                            local_session.close()

            # Database lookup case
            elif action.startswith("DB_LOOKUP:"):
                entity_type = action[10:].strip()  # Remove "DB_LOOKUP:" prefix
                return self.perform_db_lookup(entity_type, details, original_question)

            # Function call case
            elif action.startswith("FUNCTION:"):
                function_name = action[9:].strip()  # Remove "FUNCTION:" prefix
                return self.execute_function(function_name, details, original_question)

            # Default case - return the action as a direct response
            else:
                return {"entity_type": "response", "results": [{"text": action}]}

        except Exception as e:
            logger.error(f"Error executing keyword action: {e}", exc_info=True)
            return None

    def _detect_entity_type(self, query):
        """
        Detect the entity type from a query when no keyword match is found.

        Args:
            query: User's search query text

        Returns:
            String with the detected entity type or None
        """
        # Simple rule-based entity detection
        query_lower = query.lower()

        entity_patterns = {
            "image": ["image", "picture", "photo", "photograph"],
            "document": ["document", "doc", "manual", "guide", "instruction"],
            "part": ["part", "component", "spare", "replacement"],
            "tool": ["tool", "equipment", "device", "instrument"],
            "position": ["position", "location", "area", "equipment group", "model"],
            "problem": ["problem", "issue", "fault", "error", "trouble"],
            "task": ["task", "job", "procedure", "operation"]
        }

        # Check each pattern
        for entity_type, patterns in entity_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    return entity_type

        # Default to image if we can't determine
        return None

    def _extract_search_params(self, query, details):
        """
        Extract search parameters from the query and details.

        Args:
            query: User's search query text
            details: Additional search parameters

        Returns:
            Dictionary of search parameters
        """
        # Start with the provided details
        params = details.copy() if details else {}

        # Extract common search parameters from query using regex
        # Look for area mentions
        area_match = re.search(r'(in|at|from)\s+(?:the\s+)?(?:area|zone)\s+(?:of\s+)?["\']?([^"\']+)["\']?', query,
                               re.IGNORECASE)
        if area_match:
            params["area"] = area_match.group(2).strip()

        # Look for equipment group mentions
        equipment_match = re.search(r'(?:equipment|machine|system)\s+(?:group\s+)?["\']?([^"\']+)["\']?', query,
                                    re.IGNORECASE)
        if equipment_match:
            params["equipment_group"] = equipment_match.group(1).strip()

        # Look for model mentions
        model_match = re.search(r'model\s+["\']?([^"\']+)["\']?', query, re.IGNORECASE)
        if model_match:
            params["model"] = model_match.group(1).strip()

        # Look for title mentions
        title_match = re.search(r'(?:called|named|titled)\s+["\']?([^"\']+)["\']?', query, re.IGNORECASE)
        if title_match:
            params["title"] = title_match.group(1).strip()

        return params

    def format_entity_results(self, results, entity_type):
        """
        Format search results based on entity type.

        Args:
            results: List of search result items
            entity_type: Type of entity (e.g., 'image', 'document', 'drawing')

        Returns:
            Formatted response string
        """
        if not results:
            return "No results found."

        # Start with a header based on entity type
        entity_type_readable = entity_type.replace('_', ' ').title()
        response = f"I found {len(results)} {entity_type_readable}{'s' if len(results) > 1 else ''} that might be relevant:\n\n"

        # Format the results based on entity type
        if entity_type == "document" or entity_type == "powerpoint":
            for idx, item in enumerate(results[:5], 1):
                title = item.get('title', f"Document #{item.get('id', 'Unknown')}")
                response += f"{idx}. {title}\n"

        elif entity_type == "image":
            for idx, item in enumerate(results[:5], 1):
                title = item.get('title', f"Image #{item.get('id', 'Unknown')}")
                description = item.get('description', '')
                if description:
                    description = f" - {description}"
                response += f"{idx}. {title}{description}\n"

        elif entity_type == "drawing":
            for idx, item in enumerate(results[:5], 1):
                number = item.get('number', '')
                name = item.get('name', '')
                response += f"{idx}. Drawing {number}: {name}\n"

        elif entity_type == "user":
            for idx, item in enumerate(results[:5], 1):
                name = item.get('name', f"User #{item.get('id', 'Unknown')}")
                emp_id = item.get('employee_id', '')
                if emp_id:
                    emp_id = f" (ID: {emp_id})"
                response += f"{idx}. {name}{emp_id}\n"

        elif entity_type == "part":
            for idx, item in enumerate(results[:5], 1):
                part_number = item.get('part_number', '')
                name = item.get('name', f"Part #{item.get('id', 'Unknown')}")
                response += f"{idx}. {part_number}: {name}\n"

        elif entity_type == "tool":
            for idx, item in enumerate(results[:5], 1):
                name = item.get('name', f"Tool #{item.get('id', 'Unknown')}")
                type_info = item.get('type', '')
                if type_info:
                    type_info = f" ({type_info})"
                response += f"{idx}. {name}{type_info}\n"

        elif entity_type == "response":
            # For direct response items
            for item in results:
                text = item.get('text', '')
                if text:
                    response = text  # Just use the text directly
                    break
        else:
            # Generic format for other entity types
            for idx, item in enumerate(results[:5], 1):
                item_name = item.get('title') or item.get('name') or f"Item #{item.get('id', 'Unknown')}"
                response += f"{idx}. {item_name}\n"

        # Add a note about viewing if appropriate
        if entity_type not in ["response", "error"] and len(results) > 0:
            if any('url' in item for item in results):
                response += "\nClick on the links to view the full items."
            else:
                response += "\nWould you like more information about any of these items?"

        return response

    # Helper handlers for execute_function

    def _handle_register_keyword(self, details, query, request_id=None):
        """Handle keyword registration."""
        try:
            with KeywordSearch(session=self.db_session) as keyword_search:
                result = keyword_search.register_keyword(
                    keyword=details.get("keyword"),
                    action_type=details.get("action_type"),
                    search_pattern=details.get("search_pattern"),
                    entity_type=details.get("entity_type"),
                    description=details.get("description")
                )
                return result
        except Exception as e:
            logger.error(f"Error registering keyword: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _handle_delete_keyword(self, details, query, request_id=None):
        """Handle keyword deletion."""
        try:
            with KeywordSearch(session=self.db_session) as keyword_search:
                result = keyword_search.delete_keyword(details.get("keyword"))
                return result
        except Exception as e:
            logger.error(f"Error deleting keyword: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _handle_list_keywords(self, details, query, request_id=None):
        """Handle keyword listing."""
        try:
            with KeywordSearch(session=self.db_session) as keyword_search:
                keywords = keyword_search.get_all_keywords()
                return {"status": "success", "keywords": keywords, "count": len(keywords)}
        except Exception as e:
            logger.error(f"Error listing keywords: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _handle_import_keywords(self, details, query, request_id=None):
        """Handle keyword import from Excel."""
        try:
            file_path = details.get("file_path")
            if not file_path:
                return {"status": "error", "message": "No file path provided"}

            with KeywordSearch(session=self.db_session) as keyword_search:
                result = keyword_search.load_keywords_from_excel(file_path)
                return result
        except Exception as e:
            logger.error(f"Error importing keywords: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _handle_search_parts(self, details, query, request_id=None):
        """Handle part search."""
        try:
            with KeywordSearch(session=self.db_session) as keyword_search:
                return keyword_search.search_parts(details)
        except Exception as e:
            logger.error(f"Error searching parts: {e}", exc_info=True)
            return {"status": "error", "message": str(e), "entity_type": "part"}

    def _handle_search_tools(self, details, query, request_id=None):
        """Handle tool search."""
        try:
            with KeywordSearch(session=self.db_session) as keyword_search:
                return keyword_search.search_tools(details)
        except Exception as e:
            logger.error(f"Error searching tools: {e}", exc_info=True)
            return {"status": "error", "message": str(e), "entity_type": "tool"}

    def _handle_search_positions(self, details, query, request_id=None):
        """Handle position search."""
        try:
            with KeywordSearch(session=self.db_session) as keyword_search:
                return keyword_search.search_positions(details)
        except Exception as e:
            logger.error(f"Error searching positions: {e}", exc_info=True)
            return {"status": "error", "message": str(e), "entity_type": "position"}

    def _handle_search_problems(self, details, query, request_id=None):
        """Handle problem search."""
        try:
            with KeywordSearch(session=self.db_session) as keyword_search:
                return keyword_search.search_problems(details)
        except Exception as e:
            logger.error(f"Error searching problems: {e}", exc_info=True)
            return {"status": "error", "message": str(e), "entity_type": "problem"}

    def _handle_search_tasks(self, details, query, request_id=None):
        """Handle task search."""
        try:
            with KeywordSearch(session=self.db_session) as keyword_search:
                return keyword_search.search_tasks(details)
        except Exception as e:
            logger.error(f"Error searching tasks: {e}", exc_info=True)
            return {"status": "error", "message": str(e), "entity_type": "task"}

    @with_request_id
    def try_fulltext_search(self, question):
        """
        Try to find answers using full-text search strategies.

        This method searches through document content using full-text search
        to find relevant information for the user's question.

        Args:
            question: User's question text

        Returns:
            Dictionary with search results and success status
        """
        logger.debug(f"Trying fulltext search for: {question}")

        try:
            # Try to use CompleteDocument.search_by_text to find relevant documents
            local_session = None

            if not self.db_session:
                local_session = self.db_config.get_main_session()
                session = local_session
            else:
                session = self.db_session

            try:
                # Search for documents matching the question text
                search_results = CompleteDocument.search_by_text(
                    query=question,
                    session=session,
                    similarity_threshold=70,  # Use a lower threshold for better recall
                    with_links=False
                )

                if search_results and len(search_results) > 0:
                    # Format the document results
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
        """
        Try to find answers using vector search strategies.

        This method uses vector-based semantic search to find information
        semantically similar to the user's question.

        Args:
            user_id: User ID for tracking
            question: User's question text

        Returns:
            Dictionary with search results and success status
        """
        logger.debug(f"Trying vector search for user {user_id}: {question}")

        try:
            # Check if the vector search service is available
            if not hasattr(self, "vector_search_client") or not self.vector_search_client:
                logger.warning("Vector search client not available")
                return {"success": False}

            # Perform vector search
            vector_results = self.vector_search_client.search(question, limit=5)

            if vector_results and len(vector_results) > 0:
                # Format results
                response = "Based on semantic search, I found these potentially relevant passages:\n\n"

                for i, result in enumerate(vector_results[:3], 1):
                    content = result.get("content", "").strip()

                    # Truncate long passages
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

    @with_request_id
    def get_session(self, user_id, session):
        """
        Get the most recent chat session for a user.

        Args:
            user_id (str): The ID of the user
            session: SQLAlchemy session

        Returns:
            ChatSession or None: The most recent chat session for the user
        """
        try:
            logger.debug(f"Getting latest session for user {user_id}")
            # Get the most recent session for the user
            latest_session = session.query(ChatSession).filter_by(
                user_id=user_id
            ).order_by(
                ChatSession.session_id.desc()
            ).first()

            if latest_session:
                logger.debug(f"Found session {latest_session.session_id} for user {user_id}")
            else:
                logger.debug(f"No session found for user {user_id}")

            return latest_session
        except Exception as e:
            logger.error(f"Error getting session for user {user_id}: {e}", exc_info=True)
            return None

    @with_request_id
    def create_session(self, user_id, initial_message, session):
        """
        Create a new chat session for a user.

        Args:
            user_id (str): The ID of the user
            initial_message (str): The initial message for the session
            session: SQLAlchemy session

        Returns:
            int: The ID of the new session
        """
        try:
            logger.debug(f"Creating new session for user {user_id}")
            current_time = datetime.now().isoformat()

            # Create a new session
            new_session = ChatSession(
                user_id=user_id,
                start_time=current_time,
                last_interaction=current_time,
                session_data=[initial_message],
                conversation_summary=[]
            )

            session.add(new_session)
            session.commit()

            logger.info(f"Created new session {new_session.session_id} for user {user_id}")
            return new_session.session_id
        except Exception as e:
            logger.error(f"Error creating session for user {user_id}: {e}", exc_info=True)
            session.rollback()
            return None

    @with_request_id
    def update_session(self, session_id, session_data, answer, db_session):
        """
        Update an existing chat session with new data.

        Args:
            session_id (int): The ID of the session to update
            session_data (list): The updated session data
            answer (str): The answer to add to the session data
            db_session: SQLAlchemy session

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.debug(f"Updating session {session_id}")
            # Get the session
            chat_session = db_session.query(ChatSession).filter_by(
                session_id=session_id
            ).first()

            if not chat_session:
                logger.warning(f"Session {session_id} not found")
                return False

            # Update session data and last interaction time
            chat_session.session_data = session_data + [answer]
            chat_session.last_interaction = datetime.now().isoformat()

            db_session.commit()
            logger.debug(f"Session {session_id} updated successfully")
            return True
        except Exception as e:
            logger.error(f"Error updating session {session_id}: {e}", exc_info=True)
            db_session.rollback()
            return False

    @with_request_id
    def execute_keyword_action(self, keyword, action, details, original_question):
        """Execute the appropriate action based on the keyword match."""
        logger.debug(f"Executing action for keyword '{keyword}': {action}")

        try:
            # Extract search parameters from the query
            search_params = self.extract_search_params(original_question)

            # Map route names to method calls
            if action.startswith("search_images_bp"):
                # Parse the type of image search from the action
                if "_title" in action:
                    return self.search_images_by_title(search_params.get('title', ''), search_params)
                elif "_area" in action:
                    return self.search_images_by_area(search_params.get('area', ''), search_params)
                elif "_equipment_group" in action:
                    return self.search_images_by_equipment_group(search_params.get('equipment_group', ''), search_params)
                elif "_model" in action:
                    return self.search_images_by_model(search_params.get('model', ''), search_params)
                elif "_asset_number" in action:
                    return self.search_images_by_asset_number(search_params.get('asset_number', ''), search_params)
                else:
                    # Default image search
                    return self.search_images(search_params)

            elif action.startswith("search_documents_bp"):
                # Parse the type of document search from the action
                if "_title" in action:
                    return self.search_documents_by_title(search_params.get('title', ''), search_params)
                elif "_area" in action:
                    return self.search_documents_by_area(search_params.get('area', ''), search_params)
                elif "_equipment_group" in action:
                    return self.search_documents_by_equipment_group(search_params.get('equipment_group', ''), search_params)
                elif "_model" in action:
                    return self.search_documents_by_model(search_params.get('model', ''), search_params)
                elif "_asset_number" in action:
                    return self.search_documents_by_asset_number(search_params.get('asset_number', ''), search_params)
                else:
                    # Default document search
                    return self.search_documents(search_params)

            elif action.startswith("search_powerpoints_bp"):
                # Parse the type of powerpoint search from the action
                if "_title" in action:
                    return self.search_powerpoints_by_title(search_params.get('title', ''), search_params)
                elif "_area" in action:
                    return self.search_powerpoints_by_area(search_params.get('area', ''), search_params)
                elif "_equipment_group" in action:
                    return self.search_powerpoints_by_equipment_group(search_params.get('equipment_group', ''), search_params)
                elif "_model" in action:
                    return self.search_powerpoints_by_model(search_params.get('model', ''), search_params)
                elif "_asset_number" in action:
                    return self.search_powerpoints_by_asset_number(search_params.get('asset_number', ''), search_params)
                else:
                    # Default powerpoint search
                    return self.search_powerpoints(search_params)

            elif action.startswith("search_drawing_by_number_bp"):
                return self.search_drawings(search_params)

            # Simple case - action is a direct response
            elif action.startswith("RESPOND:"):
                response_text = action[8:].strip()  # Remove "RESPOND:" prefix
                return {"entity_type": "response", "results": [{"text": response_text}]}

            # Default case - return the action as a direct response
            else:
                return {"entity_type": "response", "results": [{"text": f"I'll help you with: {keyword}"}]}

        except Exception as e:
            logger.error(f"Error executing keyword action: {e}", exc_info=True)
            return {"entity_type": "error", "results": [{"text": "I encountered an error processing your request."}]}

    @with_request_id
    def extract_search_params(self, query):
        """Extract search parameters from the user query."""
        params = {
            'title': None,
            'area': None,
            'equipment_group': None,
            'model': None,
            'asset_number': None,
            'location': None,
            'description': None,
            'query': query  # Store the original query
        }

        # Extract title if "of Title" or similar pattern exists
        title_match = re.search(r'of\s+(["\'])(.*?)\1', query)
        if title_match:
            params['title'] = title_match.group(2)
        else:
            # Try another pattern for title
            title_match = re.search(r'of\s+([\w\s]+)(?:$|\s+(?:in|at|for|from))', query)
            if title_match:
                params['title'] = title_match.group(1).strip()

        # Extract area if "in area" or "of area" pattern exists
        area_match = re.search(r'(?:in|of)\s+area\s+(["\'])(.*?)\1', query)
        if area_match:
            params['area'] = area_match.group(2)
        else:
            area_match = re.search(r'(?:in|of)\s+area\s+([\w\s]+)(?:$|\s+(?:and|with|that|which))', query)
            if area_match:
                params['area'] = area_match.group(1).strip()

        # Similar extraction for other parameters
        equipment_match = re.search(r'(?:for|of)\s+equipment\s+group\s+([\w\s]+)(?:$|\s+(?:and|with))', query)
        if equipment_match:
            params['equipment_group'] = equipment_match.group(1).strip()

        model_match = re.search(r'(?:for|of)\s+model\s+([\w\s]+)(?:$|\s+(?:and|with))', query)
        if model_match:
            params['model'] = model_match.group(1).strip()

        asset_match = re.search(r'(?:for|of)\s+asset\s+(?:number\s+)?([\w\-\d]+)', query)
        if asset_match:
            params['asset_number'] = asset_match.group(1).strip()

        # Extract any number that might be a drawing number
        drawing_match = re.search(r'drawing\s+(?:number\s+)?([A-Z0-9\-]+)', query, re.IGNORECASE)
        if drawing_match:
            params['drawing_number'] = drawing_match.group(1).strip()

        logger.debug(f"Extracted search parameters: {params}")
        return params

    # IMAGE SEARCH METHODS
    @with_request_id
    def search_images(self, params, request_id=None):
        """
        Search for images based on comprehensive parameters using the Image.search class method.

        Args:
            params: Dictionary containing search parameters
            request_id: Optional request ID for tracking this operation in logs

        Returns:
            Dictionary with search results or error information
        """
        logger.debug(f"Searching for images with params: {params}")
        local_session = None
        try:
            if not self.db_session:
                local_session = self.db_config.get_main_session()
                session = local_session
            else:
                session = self.db_session

            # Extract search parameters
            search_text = params.get('query')
            title = params.get('title')
            description = params.get('description')
            file_path = params.get('file_path')
            image_id = params.get('image_id')
            tool_id = params.get('tool_id')
            complete_document_id = params.get('complete_document_id')
            exact_match = params.get('exact_match', False)
            limit = params.get('limit', 20)
            offset = params.get('offset', 0)
            sort_by = params.get('sort_by', 'id')
            sort_order = params.get('sort_order', 'asc')
            fields = params.get('fields')

            # Location-based parameters that require position lookup
            area_id = params.get('area')
            equipment_group_id = params.get('equipment_group')
            model_id = params.get('model')
            asset_number_id = params.get('asset_number')

            # First, check if we need to look up via position
            position_id = None
            if any([area_id, equipment_group_id, model_id, asset_number_id]):
                position_query = session.query(Position)
                if area_id:
                    position_query = position_query.filter(Position.area_id == int(area_id))
                if equipment_group_id:
                    position_query = position_query.filter(Position.equipment_group_id == int(equipment_group_id))
                if model_id:
                    position_query = position_query.filter(Position.model_id == int(model_id))
                if asset_number_id:
                    position_query = position_query.filter(Position.asset_number_id == int(asset_number_id))

                position = position_query.first()
                if position:
                    position_id = position.id
                    logger.debug(f"Found position ID {position_id} from criteria")

            # Use the Image.search method for comprehensive search
            results = Image.search(
                search_text=search_text,
                fields=fields,
                image_id=image_id,
                title=title,
                description=description,
                file_path=file_path,
                position_id=position_id,
                tool_id=tool_id,
                complete_document_id=complete_document_id,
                exact_match=exact_match,
                limit=limit,
                offset=offset,
                sort_by=sort_by,
                sort_order=sort_order,
                request_id=request_id,
                session=session
            )

            if results:
                # Format the results for the client
                image_results = []
                for img in results:
                    # Get full image data for serving
                    img_data = Image.serve_file(image_id=img.id, session=session)
                    if img_data and img_data['exists']:
                        image_results.append({
                            'id': img.id,
                            'title': img.title,
                            'description': img.description,
                            'url': f"/search_images_bp/serve_image/{img.id}"
                        })

                logger.info(f"Found {len(image_results)} images matching the criteria")
                return {
                    "entity_type": "image",
                    "results": image_results
                }
            else:
                logger.info("No images found matching criteria")
                return {
                    "entity_type": "response",
                    "results": [{"text": "No images found matching your criteria."}]
                }

        except Exception as e:
            logger.error(f"Error in image search: {e}", exc_info=True)
            return {
                "entity_type": "error",
                "results": [{"text": f"Error searching for images: {str(e)}"}]
            }
        finally:
            if local_session:
                local_session.close()
                logger.debug("Closed local database session")

    @with_request_id
    def search_images_by_title(self, title, params, request_id=None):
        """Search for images by title."""
        logger.debug(f"Searching for images by title: {title}")
        params['title'] = title
        return self.search_images(params, request_id=request_id)

    @with_request_id
    def search_images_by_area(self, area, params, request_id=None):
        """Search for images by area."""
        logger.debug(f"Searching for images by area: {area}")
        params['area'] = area
        return self.search_images(params, request_id=request_id)

    @with_request_id
    def search_images_by_equipment_group(self, equipment_group, params, request_id=None):
        """Search for images by equipment group."""
        logger.debug(f"Searching for images by equipment group: {equipment_group}")
        params['equipment_group'] = equipment_group
        return self.search_images(params, request_id=request_id)

    @with_request_id
    def search_images_by_model(self, model, params, request_id=None):
        """Search for images by model."""
        logger.debug(f"Searching for images by model: {model}")
        params['model'] = model
        return self.search_images(params, request_id=request_id)

    @with_request_id
    def search_images_by_asset_number(self, asset_number, params, request_id=None):
        """Search for images by asset number."""
        logger.debug(f"Searching for images by asset number: {asset_number}")
        params['asset_number'] = asset_number
        return self.search_images(params, request_id=request_id)

    @with_request_id
    def search_documents(self, params, request_id=None):
        """
        Search for documents based on comprehensive parameters using CompleteDocument search methods.

        This method intelligently selects between CompleteDocument.dynamic_search and
        CompleteDocument.search_by_text based on the provided parameters.

        Args:
            params: Dictionary of search parameters
            request_id: Optional request ID for tracking this operation in logs

        Returns:
            Dictionary with entity_type and results
        """
        logger.debug(f"Searching for documents with params: {params}")
        local_session = None
        try:
            # Get session
            if not self.db_session:
                local_session = self.db_config.get_main_session()
                session = local_session
            else:
                session = self.db_session

            # Extract search parameters
            search_text = params.get('query')
            title = params.get('title')
            rev = params.get('rev')
            file_path = params.get('file_path')
            content = params.get('content')
            similarity_threshold = params.get('similarity_threshold', 80)

            # Location-based parameters that require position lookup
            area_id = params.get('area')
            equipment_group_id = params.get('equipment_group')
            model_id = params.get('model')
            asset_number_id = params.get('asset_number')
            location_id = params.get('location')

            # Prepare search filters for dynamic_search
            search_filters = {}

            # Map direct attributes
            if title:
                search_filters['title'] = title
            if rev:
                search_filters['rev'] = rev
            if file_path:
                search_filters['file_path'] = file_path
            if content:
                search_filters['content'] = content

            # Check if we need to filter by position attributes
            position_filters = {}
            if area_id:
                position_filters['area_id'] = int(area_id)
            if equipment_group_id:
                position_filters['equipment_group_id'] = int(equipment_group_id)
            if model_id:
                position_filters['model_id'] = int(model_id)
            if asset_number_id:
                position_filters['asset_number_id'] = int(asset_number_id)
            if location_id:
                position_filters['location_id'] = int(location_id)

            # If we have position filters, we need to get positions first
            position_ids = []
            if position_filters:
                position_query = session.query(Position)
                for attr, value in position_filters.items():
                    position_query = position_query.filter(getattr(Position, attr) == value)

                positions = position_query.all()
                if positions:
                    position_ids = [p.id for p in positions]
                    logger.debug(f"Found {len(position_ids)} positions matching criteria")
                else:
                    logger.debug("No positions found matching criteria")
                    return {
                        "entity_type": "response",
                        "results": [{"text": "No documents found matching position criteria."}]
                    }

            # If we have position IDs, we need to get document IDs
            document_ids = []
            if position_ids:
                assoc_query = session.query(CompletedDocumentPositionAssociation.complete_document_id).filter(
                    CompletedDocumentPositionAssociation.position_id.in_(position_ids)
                )
                document_ids = [row[0] for row in assoc_query.all()]
                logger.debug(f"Found {len(document_ids)} document IDs associated with positions")

                if not document_ids:
                    return {
                        "entity_type": "response",
                        "results": [{"text": "No documents found associated with the specified positions."}]
                    }

            # Determine search strategy based on parameters
            results = []

            # Strategy 1: If we have document IDs from position filtering
            if document_ids:
                query = session.query(CompleteDocument).filter(
                    CompleteDocument.id.in_(document_ids)
                )

                # Apply any other direct filters
                for attr, value in search_filters.items():
                    query = query.filter(getattr(CompleteDocument, attr).ilike(f"%{value}%"))

                results = query.all()

            # Strategy 2: If we have specific filters but no position filtering
            elif search_filters:
                results = CompleteDocument.dynamic_search(session, **search_filters)

            # Strategy 3: If we just have search text, use fuzzy text search
            elif search_text:
                results = CompleteDocument.search_by_text(
                    search_text,
                    session=session,
                    similarity_threshold=similarity_threshold,
                    with_links=False,
                )

            # Format the results for the client
            if results and not isinstance(results, str):
                document_results = []
                for doc in results:
                    document_results.append({
                        'id': doc.id,
                        'title': doc.title,
                        'rev': doc.rev,
                        'url': f"/search_documents_bp/view_document/{doc.id}"
                    })

                logger.info(f"Found {len(document_results)} documents matching the criteria")
                return {
                    "entity_type": "document",
                    "results": document_results
                }
            else:
                # If no results or error message returned, try one more time with text search
                # if we haven't tried it already
                if not search_text and (title or rev):
                    fallback_text = title or rev
                    fts_results = CompleteDocument.search_by_text(
                        fallback_text,
                        session=session,
                        with_links=False,
                        request_id=request_id
                    )

                    if fts_results and isinstance(fts_results, list) and len(fts_results) > 0:
                        document_results = []
                        for doc in fts_results:
                            document_results.append({
                                'id': doc.id,
                                'title': doc.title,
                                'rev': doc.rev,
                                'url': f"/search_documents_bp/view_document/{doc.id}"
                            })

                        logger.info(f"Found {len(document_results)} documents using fallback text search")
                        return {
                            "entity_type": "document",
                            "results": document_results
                        }

                logger.info("No documents found matching criteria")
                return {
                    "entity_type": "response",
                    "results": [{"text": "No documents found matching your criteria."}]
                }

        except Exception as e:
            logger.error(f"Error in document search: {e}", exc_info=True)
            return {
                "entity_type": "error",
                "results": [{"text": f"Error searching for documents: {str(e)}"}]
            }
        finally:
            if local_session:
                local_session.close()
                logger.debug("Closed local database session")

    @with_request_id
    def search_documents_by_title(self, title, params, request_id=None):
        """Search for documents by title."""
        logger.debug(f"Searching for documents by title: {title}")
        params['title'] = title
        return self.search_documents(params, request_id=request_id)

    @with_request_id
    def search_documents_by_area(self, area, params, request_id=None):
        """Search for documents by area."""
        logger.debug(f"Searching for documents by area: {area}")
        params['area'] = area
        return self.search_documents(params, request_id=request_id)

    @with_request_id
    def search_documents_by_equipment_group(self, equipment_group, params, request_id=None):
        """Search for documents by equipment group."""
        logger.debug(f"Searching for documents by equipment group: {equipment_group}")
        params['equipment_group'] = equipment_group
        return self.search_documents(params, request_id=request_id)

    @with_request_id
    def search_documents_by_model(self, model, params, request_id=None):
        """Search for documents by model."""
        logger.debug(f"Searching for documents by model: {model}")
        params['model'] = model
        return self.search_documents(params, request_id=request_id)

    @with_request_id
    def search_documents_by_asset_number(self, asset_number, params, request_id=None):
        """Search for documents by asset number."""
        logger.debug(f"Searching for documents by asset number: {asset_number}")
        params['asset_number'] = asset_number
        return self.search_documents(params, request_id=request_id)

    @with_request_id
    def search_documents_by_location(self, location, params, request_id=None):
        """Search for documents by location."""
        logger.debug(f"Searching for documents by location: {location}")
        params['location'] = location
        return self.search_documents(params, request_id=request_id)

    @with_request_id
    def search_documents_by_revision(self, rev, params, request_id=None):
        """Search for documents by revision number."""
        logger.debug(f"Searching for documents by revision: {rev}")
        params['rev'] = rev
        return self.search_documents(params, request_id=request_id)

#region Review: Powerpoint
    # # POWERPOINT SEARCH METHODS
    #
    #     def search_powerpoints(self, params):
    #         """Search for PowerPoint presentations based on general parameters."""
    #         logger.debug(f"Searching for PowerPoints with params: {params}")
    #         local_session = None
    #         try:
    #             if not self.db_session:
    #                 local_session = self.db_config.get_main_session()
    #                 session = local_session
    #             else:
    #                 session = self.db_session
    #
    #             # Prepare search criteria based on available parameters
    #             search_criteria = {}
    #
    #             if params.get('title'):
    #                 search_criteria['title'] = params['title'] + " PowerPoint"
    #             if params.get('area'):
    #                 search_criteria['area'] = params['area']
    #             if params.get('equipment_group'):
    #                 search_criteria['equipment_group'] = params['equipment_group']
    #             if params.get('model'):
    #                 search_criteria['model'] = params['model']
    #             if params.get('asset_number'):
    #                 search_criteria['asset_number'] = params['asset_number']
    #
    #             # Note: Since I don't see a dedicated PowerPoints table, we'll search in documents
    #             # with PowerPoint-related keywords
    #             try:
    #                 from modules.emtacdb.search_powerpoints_bp import search_powerpoints_db
    #                 result = search_powerpoints_db(session, **search_criteria)
    #             except ImportError:
    #                 # Fallback to searching for PowerPoint files through documents
    #                 from modules.emtacdb.search_documents_bp import search_documents_db
    #                 if 'title' not in search_criteria:
    #                     search_criteria['title'] = "PowerPoint"
    #                 result = search_documents_db(session, **search_criteria)
    #
    #             if ('powerpoints' in result and result['powerpoints']) or ('documents' in result and result['documents']):
    #                 # Format the results
    #                 pp_results = []
    #                 if 'powerpoints' in result:
    #                     items = result['powerpoints']
    #                     url_prefix = "/search_powerpoints_bp/view_powerpoint/"
    #                 else:
    #                     items = result['documents']
    #                     url_prefix = "/search_documents_bp/view_document/"
    #
    #                 for pp in items:
    #                     pp_results.append({
    #                         'id': pp['id'],
    #                         'title': pp['title'],
    #                         'url': f"{url_prefix}{pp['id']}"
    #                     })
    #
    #                 return {
    #                     "entity_type": "powerpoint",
    #                     "results": pp_results
    #                 }
    #             else:
    #                 return {
    #                     "entity_type": "response",
    #                     "results": [{"text": "No presentations found matching your criteria."}]
    #                 }
    #
    #         except Exception as e:
    #             logger.error(f"Error in PowerPoint search: {e}", exc_info=True)
    #             return {
    #                 "entity_type": "error",
    #                 "results": [{"text": f"Error searching for presentations: {str(e)}"}]
    #             }
    #         finally:
    #             if local_session:
    #                 local_session.close()
    #
    #     def search_powerpoints_by_title(self, title, params):
    #         """Search for PowerPoints by title."""
    #         logger.debug(f"Searching for PowerPoints by title: {title}")
    #         params['title'] = title
    #         return self.search_powerpoints(params)
    #
    #     def search_powerpoints_by_area(self, area, params):
    #         """Search for PowerPoints by area."""
    #         logger.debug(f"Searching for PowerPoints by area: {area}")
    #         params['area'] = area
    #         return self.search_powerpoints(params)
    #
    #     def search_powerpoints_by_equipment_group(self, equipment_group, params):
    #         """Search for PowerPoints by equipment group."""
    #         logger.debug(f"Searching for PowerPoints by equipment group: {equipment_group}")
    #         params['equipment_group'] = equipment_group
    #         return self.search_powerpoints(params)
    #
    #     def search_powerpoints_by_model(self, model, params):
    #         """Search for PowerPoints by model."""
    #         logger.debug(f"Searching for PowerPoints by model: {model}")
    #         params['model'] = model
    #         return self.search_powerpoints(params)
    #
    #     def search_powerpoints_by_asset_number(self, asset_number, params):
    #         """Search for PowerPoints by asset number."""
    #         logger.debug(f"Searching for PowerPoints by asset number: {asset_number}")
    #         params['asset_number'] = asset_number
    #         return self.search_powerpoints(params)

    # DRAWING SEARCH METHODSn  Po

    @with_request_id
    def search_drawings(self, params, request_id=None):
        """
        Search for drawings based on parameters using the Drawing.search_and_format method.

        Args:
            params: Dictionary containing search parameters
            request_id: Optional request ID for tracking this operation in logs

        Returns:
            Dictionary with entity_type and results
        """
        logger.debug(f"Searching for drawings with params: {params}")
        local_session = None

        try:
            # Get session
            if not self.db_session:
                local_session = self.db_config.get_main_session()
                session = local_session
            else:
                session = self.db_session

            # Extract search parameters
            search_text = params.get('query')
            exact_match = params.get('exact_match', False)
            fields = params.get('fields')

            # Extract drawing-specific parameters
            drawing_id = params.get('drawing_id')
            drw_equipment_name = params.get('equipment_name')
            drw_number = params.get('drawing_number') or params.get('number')
            drw_name = params.get('name')
            drw_revision = params.get('revision')
            drw_spare_part_number = params.get('spare_part_number')
            file_path = params.get('file_path')
            limit = params.get('limit', 20)

            # If no specific drawing number was extracted but we have a query,
            # try to find a drawing number pattern in the query
            if not drw_number and search_text:
                # Look for common drawing number patterns (alphanumeric with possible hyphens)
                number_match = re.search(r'\b([A-Z0-9][\w\-]*\d)\b', search_text, re.IGNORECASE)
                if number_match:
                    drw_number = number_match.group(1)
                    logger.debug(f"Extracted drawing number from query: {drw_number}")

            # Use the new search_and_format method that handles everything
            return Drawing.search_and_format(
                search_text=search_text,
                fields=fields,
                exact_match=exact_match,
                drawing_id=drawing_id,
                drw_equipment_name=drw_equipment_name,
                drw_number=drw_number,
                drw_name=drw_name,
                drw_revision=drw_revision,
                drw_spare_part_number=drw_spare_part_number,
                file_path=file_path,
                limit=limit,
                request_id=request_id,
                session=session
            )

        except Exception as e:
            logger.error(f"Error in drawing search: {e}", exc_info=True)
            return {
                "entity_type": "error",
                "results": [{"text": f"Error searching for drawings: {str(e)}"}]
            }
        finally:
            if local_session:
                local_session.close()
                logger.debug("Closed local database session")

    @with_request_id
    def search_drawings_by_number(self, number, params, request_id=None):
        """Search for drawings by drawing number."""
        logger.debug(f"Searching for drawings by number: {number}")
        params['drawing_number'] = number
        return self.search_drawings(params, request_id=request_id)

    @with_request_id
    def search_drawings_by_asset_number(self, asset_number, params, request_id=None):
        """Search for drawings by asset number."""
        logger.debug(f"Searching for drawings by asset number: {asset_number}")
        local_session = None

        try:
            # Get session
            if not self.db_session:
                local_session = self.db_config.get_main_session()
                session = local_session
            else:
                session = self.db_session

            # Get drawings
            drawings = Drawing.search_by_asset_number(
                asset_number_value=asset_number,
                request_id=request_id,
                session=session
            )

            # Format using Drawing.search_and_format
            if drawings:
                drawing_ids = [drawing.id for drawing in drawings]
                return Drawing.search_and_format(
                    drawing_id=drawing_ids,
                    request_id=request_id,
                    session=session
                )
            else:
                return {
                    "entity_type": "response",
                    "results": [{"text": f"No drawings found for asset number '{asset_number}'."}]
                }
        except Exception as e:
            logger.error(f"Error searching drawings by asset number: {e}", exc_info=True)
            return {
                "entity_type": "error",
                "results": [{"text": f"Error searching for drawings: {str(e)}"}]
            }
        finally:
            if local_session:
                local_session.close()

    @with_request_id
    def search_drawings_by_name(self, name, params, request_id=None):
        """Search for drawings by name."""
        logger.debug(f"Searching for drawings by name: {name}")
        params['name'] = name
        return self.search_drawings(params, request_id=request_id)

    @with_request_id
    def search_drawings_by_revision(self, revision, params, request_id=None):
        """Search for drawings by revision."""
        logger.debug(f"Searching for drawings by revision: {revision}")
        params['revision'] = revision
        return self.search_drawings(params, request_id=request_id)

    @with_request_id
    def search_drawings_by_spare_part(self, part_number, params, request_id=None):
        """Search for drawings by spare part number."""
        logger.debug(f"Searching for drawings by spare part number: {part_number}")
        params['spare_part_number'] = part_number
        return self.search_drawings(params, request_id=request_id)

    def perform_advanced_keyword_search(self, query, request_id=None):
        """
        Perform an advanced keyword search across multiple entity types.

        This method searches across documents, images, drawings, and PowerPoints
        to find the most relevant results for the query.

        Args:
            query: User's search query text
            request_id: Optional request ID for tracking this operation in logs

        Returns:
            List of search results from various entity types
        """
        logger.debug(f"Performing advanced keyword search for: {query}")

        try:
            # First try using the KeywordSearch class
            with KeywordSearch(session=self.db_session) as keyword_search:
                # Execute the search
                keyword_result = keyword_search.execute_search(query)

                # If we got successful results from the keyword search, return them
                if keyword_result.get("status") == "success" and keyword_result.get("count", 0) > 0:
                    # Add request_id to the result metadata if provided
                    if request_id:
                        keyword_result["request_id"] = request_id
                    return keyword_result

            # If keyword search didn't find anything, try our own multi-entity search

            # Extract search parameters
            search_params = self._extract_search_params(query, {})
            search_params["query"] = query

            # Store all results
            all_results = []

            # Try document search
            doc_results = self.search_documents(search_params, request_id=request_id)
            if doc_results.get('entity_type') == 'document' and 'results' in doc_results:
                # Add the top 3 document results
                for doc in doc_results['results'][:3]:
                    doc['type'] = 'document'
                    all_results.append(doc)

            # Try image search
            img_results = self.search_images(search_params, request_id=request_id)
            if img_results.get('entity_type') == 'image' and 'results' in img_results:
                # Add the top 3 image results
                for img in img_results['results'][:3]:
                    img['type'] = 'image'
                    all_results.append(img)

            # Try drawing search
            drw_results = self.search_drawings(search_params, request_id=request_id)
            if drw_results.get('entity_type') == 'drawing' and 'results' in drw_results:
                # Add the top 3 drawing results
                for drw in drw_results['results'][:3]:
                    drw['type'] = 'drawing'
                    all_results.append(drw)

            # Try PowerPoint search
#            pp_results = self.search_powerpoints(search_params, request_id=request_id)
#            if pp_results.get('entity_type') == 'powerpoint' and 'results' in pp_results:
                # Add the top 3 PowerPoint results
#                for pp in pp_results['results'][:3]:
#                    pp['type'] = 'powerpoint'
#                    all_results.append(pp)

            # Format the combined results
            return {
                "status": "success",
                "entity_type": "mixed",
                "count": len(all_results),
                "results": all_results,
                "request_id": request_id
            } if all_results else {
                "status": "not_found",
                "message": "No results found for your query",
                "query": query,
                "request_id": request_id
            }

        except Exception as e:
            logger.error(f"Error in advanced keyword search: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error in keyword search: {str(e)}",
                "query": query,
                "request_id": request_id
            }

    def perform_db_lookup(self, entity_type, details, query, request_id=None):
        """
        Perform a database lookup for a specific entity type.

        This method routes the search to the appropriate entity-specific search method
        based on the provided entity_type.

        Args:
            entity_type: Type of entity to search (e.g., 'image', 'document', 'drawing')
            details: Dictionary of additional search parameters
            query: Original search query text
            request_id: Optional request ID for tracking this operation in logs

        Returns:
            Dictionary with search results
        """
        logger.debug(f"Performing DB lookup for entity type: {entity_type}")

        try:
            # Convert query into search parameters
            params = self._extract_search_params(query, details)

            # Add the original query as a search parameter
            params["query"] = query

            # Route to appropriate search method based on entity type
            if entity_type == "image":
                return self.search_images(params, request_id=request_id)
            elif entity_type == "document":
                return self.search_documents(params, request_id=request_id)
            elif entity_type == "drawing":
                return self.search_drawings(params, request_id=request_id)
            elif entity_type == "part":
                # Create KeywordSearch instance to use its part search capability
                with KeywordSearch(session=self.db_session) as keyword_search:
                    return keyword_search.search_parts(params)
            elif entity_type == "tool":
                with KeywordSearch(session=self.db_session) as keyword_search:
                    return keyword_search.search_tools(params)
            elif entity_type == "position":
                with KeywordSearch(session=self.db_session) as keyword_search:
                    return keyword_search.search_positions(params)
            elif entity_type == "problem":
                with KeywordSearch(session=self.db_session) as keyword_search:
                    return keyword_search.search_problems(params)
            elif entity_type == "task":
                with KeywordSearch(session=self.db_session) as keyword_search:
                    return keyword_search.search_tasks(params)
            elif entity_type == "powerpoint":
                return self.search_powerpoints(params, request_id=request_id)
            elif entity_type == "user":
                return self.search_users(params, request_id=request_id)
            else:
                logger.warning(f"Unknown entity type: {entity_type}")
                return {
                    "entity_type": "response",
                    "results": [{"text": f"I'm not sure how to look up {entity_type} entities."}]
                }

        except Exception as e:
            logger.error(f"Error in DB lookup for {entity_type}: {e}", exc_info=True)
            return {
                "entity_type": "error",
                "message": f"Error in database lookup: {str(e)}",
                "entity_type": entity_type
            }

    def search_users(self, params):
        """Search for users based on parameters."""
        logger.debug(f"Searching for users with params: {params}")
        local_session = None
        try:
            if not self.db_session:
                local_session = self.db_config.get_main_session()
                session = local_session
            else:
                session = self.db_session

            # Extract name from the query if present
            name = None
            employee_id = None

            # Try to extract name from query
            name_match = re.search(r'(?:named|called|for)\s+([\w\s]+)(?:$|\?|\.)', params['query'], re.IGNORECASE)
            if name_match:
                name = name_match.group(1).strip()

            # Try to extract employee ID from query
            id_match = re.search(r'(?:employee|user|id)\s+(?:id\s+)?(\w+)', params['query'], re.IGNORECASE)
            if id_match:
                employee_id = id_match.group(1)

            # Lookup user in database
            user = None
            if employee_id:
                user = session.query(User).filter_by(employee_id=employee_id).first()
            elif name:
                # Try to match by first or last name
                name_parts = name.split()
                if len(name_parts) == 1:
                    # Single name provided, could be first or last
                    users = session.query(User).filter(
                        (User.first_name.ilike(f"%{name}%")) |
                        (User.last_name.ilike(f"%{name}%"))
                    ).all()
                    if users:
                        user = users[0]  # Take the first matching user
                elif len(name_parts) >= 2:
                    # First and last name provided
                    first_name = name_parts[0]
                    last_name = ' '.join(name_parts[1:])
                    user = session.query(User).filter(
                        User.first_name.ilike(f"%{first_name}%"),
                        User.last_name.ilike(f"%{last_name}%")
                    ).first()

            if user:
                return {
                    "entity_type": "user",
                    "results": [{
                        "id": user.id,
                        "employee_id": user.employee_id,
                        "name": f"{user.first_name} {user.last_name}",
                        "current_shift": user.current_shift,
                        "primary_area": user.primary_area
                    }]
                }
            else:
                return {
                    "entity_type": "response",
                    "results": [{"text": "No user found matching your criteria."}]
                }

        except Exception as e:
            logger.error(f"Error in user search: {e}", exc_info=True)
            return {
                "entity_type": "error",
                "results": [{"text": f"Error searching for users: {str(e)}"}]
            }
        finally:
            if local_session:
                local_session.close()

    def get_search_suggestions(self, question):
        """
        Generate search suggestions based on the user's question.

        This function analyzes the question and suggests alternative
        search approaches or related queries.

        Args:
            question: User's question text

        Returns:
            String with search suggestions
        """
        # Extract keywords from the question
        keywords = [word for word in re.findall(r'\b\w{4,}\b', question.lower())
                    if word not in {"what", "where", "when", "which", "about", "would", "could", "should"}]

        if not keywords:
            return ""

        # Generate alternative search suggestions
        suggestions = []

        # Suggest a drawing search if the question might be about a drawing
        if any(word in question.lower() for word in ["drawing", "schematic", "diagram", "sketch"]):
            suggestions.append("Try asking for a specific drawing number if you know it")

        # Suggest an image search if the question might be about images
        if any(word in question.lower() for word in ["image", "picture", "photo", "see"]):
            suggestions.append("Try asking to see images from a specific area or of specific equipment")

        # Suggest document search
        if any(word in question.lower() for word in ["document", "manual", "instruction", "procedure"]):
            suggestions.append("Try asking for documents with a specific title or from a specific area")

        # Create alternative search suggestion using main keywords
        if len(keywords) >= 2:
            alt_query = " ".join(keywords[:3])
            suggestions.append(f"Try searching for '{alt_query}'")

        if not suggestions:
            return ""

        # Format suggestions
        return "\n\nHere are some search suggestions:\n- " + "\n- ".join(suggestions)

    def format_response(self, answer, client_type=None, results=None):
        """
        Enhanced format_response that includes performance information.
        """
        # Basic formatting for all client types
        formatted_answer = answer.strip()

        # Add performance metrics if available and response was slow
        if hasattr(self, 'start_time') and self.start_time:
            response_time = time.time() - self.start_time

            # Add performance info for slow responses or debug mode
            if response_time > 2.0 or client_type == 'debug':
                if '<div class="performance-note">' not in formatted_answer:
                    formatted_answer += f"<div class='performance-note'><small>Response time: {response_time:.2f}s</small></div>"

        # Format URLs if not already done
        if '<a href=' not in formatted_answer and ('http://' in formatted_answer or 'https://' in formatted_answer):
            formatted_answer = re.sub(
                r'(https?://[^\s]+)',
                r'<a href="\1" target="_blank">\1</a>',
                formatted_answer
            )

        return formatted_answer

    def record_interaction(self, user_id, question, answer):
        """
        Simple wrapper that calls QandA.record_interaction.
        Handles session management and calculates processing time.
        """
        try:
            local_session = None

            # Get an appropriate session
            if not self.db_session:
                local_session = self.db_config.get_main_session()
                session = local_session
            else:
                session = self.db_session

            try:
                # Calculate processing time if available
                processing_time = None
                if hasattr(self, 'start_time') and self.start_time:
                    processing_time = int((time.time() - self.start_time) * 1000)

                # Call the QandA class method to record the interaction
                interaction = QandA.record_interaction(
                    user_id=user_id,
                    question=question,
                    answer=answer,
                    session=session,
                    processing_time_ms=processing_time
                )

                if interaction:
                    logger.debug(f"Successfully recorded interaction for user {user_id}")
                    return interaction
                else:
                    logger.warning("QandA.record_interaction returned None")
                    return None

            finally:
                # Only close the session if we created it locally
                if local_session:
                    local_session.close()

        except Exception as e:
            logger.error(f"Error in AistManager.record_interaction: {e}", exc_info=True)
            # Don't let recording errors break the main application flow
            # Just log the error and continue
            return None

    def _ensure_ai_model(self):
        """Ensure AI model is loaded if needed"""
        if self.ai_model is None:
            from plugins import load_ai_model
            self.ai_model = load_ai_model()
            logger.debug("AI model lazy-loaded")
        return self.ai_model

    def enhance_ai_response(answer, question, relevant_doc):
        """
        Enhance the raw AI response with document context and improved formatting.

        Args:
            answer: Raw AI model response text
            question: Original user question
            relevant_doc: Document object used to generate the response

        Returns:
            Enhanced and formatted response string
        """
        # Initialize enhanced answer
        enhanced = answer.strip()

        # Add document citation if not already present
        if relevant_doc and hasattr(relevant_doc, 'id'):
            # Create citation with document info
            doc_title = getattr(relevant_doc, 'title', f'Document #{relevant_doc.id}')
            doc_info = f"(Source: {doc_title})"

            # Add citation at the end if not already present
            if doc_info not in enhanced:
                enhanced += f"\n\n{doc_info}"

        # Format paragraph breaks for better readability
        import re
        enhanced = re.sub(r'(\n{3,})', '\n\n', enhanced)

        logger.debug(f"Enhanced AI response from {len(answer)} to {len(enhanced)} characters")
        return enhanced

    def get_performance_recommendations(aist_manager):
        """Analyze performance data and provide optimization recommendations."""
        analytics = aist_manager.get_performance_analytics(hours=24)

        if not analytics or analytics.get('total_requests', 0) == 0:
            return []

        recommendations = []

        # Check overall response time
        avg_time = analytics.get('avg_response_time', 0)
        if avg_time > 3.0:
            recommendations.append({
                'priority': 'high',
                'category': 'response_time',
                'message': f'Average response time is {avg_time:.2f}s, consider optimizing search strategies'
            })

        # Check step performance
        step_perf = analytics.get('step_performance', {})

        if 'ai_response' in step_perf and step_perf['ai_response']['avg_time'] > 2.0:
            recommendations.append({
                'priority': 'medium',
                'category': 'ai_model',
                'message': f'AI model responses averaging {step_perf["ai_response"]["avg_time"]:.2f}s, consider model optimization'
            })

        if 'vector_search' in step_perf and step_perf['vector_search']['avg_time'] > 1.5:
            recommendations.append({
                'priority': 'medium',
                'category': 'vector_search',
                'message': f'Vector search averaging {step_perf["vector_search"]["avg_time"]:.2f}s, consider index optimization'
            })

        if 'fulltext_search' in step_perf and step_perf['fulltext_search']['avg_time'] > 1.0:
            recommendations.append({
                'priority': 'low',
                'category': 'fulltext_search',
                'message': f'Full-text search averaging {step_perf["fulltext_search"]["avg_time"]:.2f}s, consider database indexing'
            })

        # Check performance distribution
        perf_dist = analytics.get('performance_distribution', {})
        poor_performance_ratio = (perf_dist.get('poor', 0) + perf_dist.get('very_poor', 0)) / analytics[
            'total_requests']

        if poor_performance_ratio > 0.2:  # More than 20% poor performance
            recommendations.append({
                'priority': 'high',
                'category': 'overall',
                'message': f'{poor_performance_ratio * 100:.1f}% of requests have poor performance, investigate bottlenecks'
            })

        return recommendations

    def _execute_with_timeout(self, func, timeout_seconds=400, *args, **kwargs):
        """
        Windows-compatible timeout wrapper using threading.
        Replaces the Unix-only signal.SIGALRM approach.
        """
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

        # Start the worker thread
        worker_thread = threading.Thread(target=worker)
        worker_thread.daemon = True
        worker_thread.start()

        # Wait for completion or timeout
        worker_thread.join(timeout=timeout_seconds)

        if not completed[0]:
            # Thread is still running - timeout occurred
            logger.warning(f"Operation timed out after {timeout_seconds} seconds")
            return None, "Operation timed out"

        if exception[0]:
            # Thread completed with exception
            raise exception[0]

        # Thread completed successfully
        return result[0], None

    def _direct_part_search_with_synonyms(self, query: str, session) -> Dict[str, Any]:
        """
        Direct part search that actually returns part numbers.
        This is the missing piece - we need to search for actual parts!
        """
        try:
            # Import Part model
            from modules.emtacdb.emtacdb_fts import Part

            # Strategy 1: Search for parts using the Part.search method
            search_params = {
                'search_text': query,
                'fields': ['part_number', 'name', 'oem_mfg', 'model', 'notes'],
                'limit': 20,
                'session': session
            }

            logger.debug(f"Searching parts with: {search_params}")

            # Execute part search
            parts = Part.search(**search_params)

            if parts:
                # Format parts for unified search response
                formatted_parts = []
                for part in parts:
                    part_data = {
                        'id': part.id,
                        'part_number': part.part_number,
                        'name': part.name,
                        'oem_mfg': part.oem_mfg,
                        'model': part.model,
                        'type': 'part',
                        'description': part.notes or '',
                        'match_type': 'synonym_search'
                    }
                    formatted_parts.append(part_data)

                return {
                    'status': 'success',
                    'total_results': len(formatted_parts),
                    'results_by_type': {
                        'parts': formatted_parts
                    },
                    'search_method': 'direct_part_search',
                    'query': query
                }
            else:
                # Strategy 2: Try broader search with individual terms
                words = query.split()
                for word in words:
                    if len(word) > 3:  # Skip short words
                        broader_params = {
                            'search_text': word,
                            'fields': ['part_number', 'name', 'oem_mfg'],
                            'limit': 10,
                            'session': session
                        }

                        broader_parts = Part.search(**broader_params)
                        if broader_parts:
                            formatted_parts = []
                            for part in broader_parts:
                                part_data = {
                                    'id': part.id,
                                    'part_number': part.part_number,
                                    'name': part.name,
                                    'oem_mfg': part.oem_mfg,
                                    'model': part.model,
                                    'type': 'part',
                                    'description': part.notes or '',
                                    'match_type': 'broader_search',
                                    'matched_term': word
                                }
                                formatted_parts.append(part_data)

                            return {
                                'status': 'success',
                                'total_results': len(formatted_parts),
                                'results_by_type': {
                                    'parts': formatted_parts
                                },
                                'search_method': 'broader_term_search',
                                'query': query,
                                'matched_term': word
                            }

                return {
                    'status': 'no_results',
                    'total_results': 0,
                    'message': f"No parts found for query: {query}"
                }

        except Exception as e:
            logger.error(f"Error in direct part search: {e}", exc_info=True)
            return {
                'status': 'error',
                'total_results': 0,
                'message': f"Part search failed: {str(e)}"
            }

    def _get_fallback_synonyms(self, term: str) -> List[str]:
        """Enhanced fallback synonyms with better coverage"""

        logger.debug(f"🔄 Using fallback synonyms for '{term}'")

        # Enhanced fallback synonyms based on your database structure
        fallback_synonyms = {
            # Equipment types
            'valve': ['valves', 'control valve', 'ball valve', 'gate valve', 'check valve', 'relief valve'],
            'bearing': ['bearings', 'ball bearing', 'roller bearing', 'thrust bearing', 'bearing assembly'],
            'switch': ['switches', 'limit switch', 'pressure switch', 'safety switch', 'temperature switch'],
            'motor': ['motors', 'electric motor', 'servo motor', 'ac motor', 'dc motor'],
            'belt': ['belts', 'drive belt', 'v-belt', 'timing belt', 'serpentine belt'],
            'cable': ['cables', 'power cable', 'control cable', 'data cable'],
            'sensor': ['sensors', 'temperature sensor', 'pressure sensor', 'level sensor', 'flow sensor'],
            'seal': ['seals', 'oil seal', 'hydraulic seal', 'shaft seal', 'mechanical seal'],
            'relay': ['relays', 'control relay', 'power relay', 'time relay'],
            'pump': ['pumps', 'centrifugal pump', 'hydraulic pump', 'water pump', 'circulation pump'],
            'spring': ['springs', 'compression spring', 'extension spring', 'torsion spring'],
            'filter': ['filters', 'air filter', 'oil filter', 'hydraulic filter', 'fuel filter'],
            'gear': ['gears', 'spur gear', 'bevel gear', 'worm gear'],
            'tube': ['tubes', 'hydraulic tube', 'pneumatic tube'],
            'hose': ['hoses', 'hydraulic hose', 'air hose', 'water hose'],
            'wire': ['wires', 'electrical wire', 'control wire'],
            'fan': ['fans', 'cooling fan', 'exhaust fan', 'ventilation fan'],
            'coupling': ['couplings', 'flexible coupling', 'rigid coupling'],
            'gasket': ['gaskets', 'rubber gasket', 'metal gasket'],
            'bushing': ['bushings', 'bronze bushing', 'rubber bushing'],
            'bracket': ['brackets', 'mounting bracket', 'support bracket'],

            # Component descriptors
            'assembly': ['assemblies', 'unit', 'component', 'module'],
            'component': ['components', 'part', 'piece', 'element'],
            'unit': ['units', 'assembly', 'module', 'system'],
            'part': ['parts', 'component', 'piece'],
            'piece': ['pieces', 'part', 'component'],

            # Actions/search terms
            'find': ['search', 'locate', 'get', 'show', 'display'],
            'search': ['find', 'locate', 'look for'],
            'show': ['display', 'present', 'exhibit'],
            'get': ['obtain', 'retrieve', 'fetch'],

            # Size/type modifiers
            'small': ['compact', 'mini', 'tiny'],
            'large': ['big', 'oversized', 'heavy duty'],
            'heavy': ['robust', 'industrial', 'heavy duty'],
            'light': ['lightweight', 'compact'],

            # Material types
            'steel': ['stainless steel', 'carbon steel', 'alloy steel'],
            'aluminum': ['aluminium', 'alloy', 'light metal'],
            'plastic': ['polymer', 'synthetic', 'composite'],
            'rubber': ['elastomer', 'flexible material'],
        }

        term_lower = term.lower()

        # Exact match
        if term_lower in fallback_synonyms:
            synonyms = fallback_synonyms[term_lower][:4]
            logger.debug(f"✅ Fallback exact match synonyms for '{term}': {synonyms}")
            return synonyms

        # Partial match for compound terms
        for key, values in fallback_synonyms.items():
            if key in term_lower or term_lower in key:
                synonyms = values[:3]
                logger.debug(f"✅ Fallback partial match synonyms for '{term}' (matched '{key}'): {synonyms}")
                return synonyms

        # Stem-based matching (basic)
        stems = {
            'bear': 'bearing',
            'valv': 'valve',
            'mot': 'motor',
            'pump': 'pump',
            'seal': 'seal',
            'switch': 'switch',
            'sens': 'sensor',
            'belt': 'belt',
            'cabl': 'cable',
            'wire': 'wire',
            'hose': 'hose',
            'tube': 'tube',
            'gear': 'gear',
            'spring': 'spring',
            'filter': 'filter',
            'fan': 'fan'
        }

        for stem, full_term in stems.items():
            if stem in term_lower and full_term in fallback_synonyms:
                synonyms = fallback_synonyms[full_term][:2]
                logger.debug(
                    f"✅ Fallback stem match synonyms for '{term}' (stem '{stem}' -> '{full_term}'): {synonyms}")
                return synonyms

        # No synonyms found
        logger.debug(f"ℹ️ No fallback synonyms found for '{term}'")
        return []

    # Helper function to test database connectivity before synonym lookup
    def test_synonym_database_connectivity(session) -> bool:
        """Test if we can safely query the synonym database"""
        try:
            if session.in_transaction():
                session.rollback()

            from sqlalchemy import text

            # Simple test query
            test_query = text("SELECT COUNT(*) FROM entity_synonym LIMIT 1")
            result = session.execute(test_query)
            count = result.scalar()

            logger.debug(f"✅ Synonym database connectivity test passed (found {count} synonyms)")
            return True

        except Exception as e:
            logger.warning(f"⚠️ Synonym database connectivity test failed: {e}")
            try:
                session.rollback()
            except:
                pass
            return False

    # Updated synonym expansion that uses the fixed method
    def expand_query_with_synonyms_safe(self, query_text: str) -> Dict[str, List[str]]:
        """Safely expand query with synonyms, handling all database errors"""

        # Test database connectivity first
        if not test_synonym_database_connectivity(self.session):
            logger.warning("⚠️ Synonym database unavailable, using fallback only")
            return self._expand_query_with_fallback_only(query_text)

        import re

        # Remove common search words and extract key terms
        stop_words = {'search', 'find', 'show', 'get', 'locate', 'display', 'for', 'the', 'a', 'an', 'part', 'number'}
        words = re.findall(r'\b\w+\b', query_text.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]

        expanded_terms = {}

        for term in key_terms:
            # Use the fixed synonym lookup method
            synonyms = self._get_synonyms_for_term(term, self.session)
            if synonyms:
                expanded_terms[term] = synonyms

        logger.debug(f"🔍 Expanded terms for '{query_text}': {expanded_terms}")
        return expanded_terms

    def _expand_query_with_fallback_only(self, query_text: str) -> Dict[str, List[str]]:
        """Expand query using only fallback synonyms when database is unavailable"""

        import re

        stop_words = {'search', 'find', 'show', 'get', 'locate', 'display', 'for', 'the', 'a', 'an', 'part', 'number'}
        words = re.findall(r'\b\w+\b', query_text.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]

        expanded_terms = {}

        for term in key_terms:
            synonyms = self._get_fallback_synonyms(term)
            if synonyms:
                expanded_terms[term] = synonyms

        logger.debug(f"🔄 Fallback-only expanded terms for '{query_text}': {expanded_terms}")
        return expanded_terms