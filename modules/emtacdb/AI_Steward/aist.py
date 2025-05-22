# modules/ai/ai_steward.py
import time
import re
import json
from datetime import datetime
from sqlalchemy.ext.declarative import declarative_base
from modules.configuration.log_config import logger, with_request_id, log_timed_operation
from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import (Position, CompleteDocument, Image, KeywordSearch, ChatSession, User,
                                         QandA, Drawing, CompletedDocumentPositionAssociation,
                                         KeywordAction, Document,DocumentEmbedding)
from plugins.ai_modules.ai_models import ModelsConfig
# Base class for SQLAlchemy models
Base = declarative_base()


class VectorSearchClient:
    def __init__(self):
        self.np = __import__('numpy')
        logger.debug("Vector search client initialized")

    def cosine_similarity(self, vector1, vector2):
        """Calculate cosine similarity between two vectors."""
        dot_product = self.np.dot(vector1, vector2)
        norm_vector1 = self.np.linalg.norm(vector1)
        norm_vector2 = self.np.linalg.norm(vector2)

        # Handle zero division
        if norm_vector1 == 0 or norm_vector2 == 0:
            return 0

        return dot_product / (norm_vector1 * norm_vector2)

    def search(self, query, limit=5):
        """
        Perform vector search for a query.

        Args:
            query: Text query to search for
            limit: Maximum number of results to return

        Returns:
            List of results ordered by similarity
        """
        from modules.emtacdb.emtacdb_fts import DocumentEmbedding, Document
        from modules.configuration.config_env import DatabaseConfig

        try:
            # Get database session
            db_config = DatabaseConfig()
            session = db_config.get_main_session()

            # Get embedding model
            embedding_model = ModelsConfig.load_embedding_model()

            # Generate query embedding
            query_embedding = embedding_model.get_embeddings(query)
            if not query_embedding:
                logger.warning("Failed to generate embedding for query")
                return []

            # Convert query embedding to numpy array
            query_embedding_np = self.np.array(query_embedding)

            # Get all document embeddings
            doc_embeddings = session.query(
                DocumentEmbedding.document_id,
                DocumentEmbedding.model_embedding
            ).all()

            # Calculate similarities
            similarities = []
            for doc_id, doc_embedding_raw in doc_embeddings:
                try:
                    # Parse embedding from database
                    if isinstance(doc_embedding_raw, bytes):
                        try:
                            doc_embedding = self.np.array(json.loads(doc_embedding_raw.decode('utf-8')))
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            doc_embedding = self.np.frombuffer(doc_embedding_raw, dtype=self.np.float32)
                    else:
                        doc_embedding = self.np.array(doc_embedding_raw)

                    # Calculate similarity
                    similarity = self.cosine_similarity(query_embedding_np, doc_embedding)
                    similarities.append((doc_id, similarity))
                except Exception as e:
                    logger.error(f"Error calculating similarity for document {doc_id}: {e}")

            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Get top results
            top_docs = []
            for doc_id, similarity in similarities[:limit]:
                if similarity > 0.6:  # Minimum similarity threshold
                    doc = session.query(Document).get(doc_id)
                    if doc:
                        top_docs.append({
                            "id": doc_id,
                            "content": doc.content[:1000] + "..." if len(doc.content) > 1000 else doc.content,
                            "similarity": float(similarity),
                            "source": f"Document {doc_id}"
                        })

            logger.info(f"Vector search found {len(top_docs)} results for query: {query[:50]}...")
            return top_docs

        except Exception as e:
            logger.error(f"Error in vector search: {e}", exc_info=True)
            return []
        finally:
            if 'session' in locals():
                session.close()

class AistManager:
    """
    AI Steward manages the search strategies and response generation process.
    It orchestrates keyword search, full-text search, vector similarity search,
    and AI model fallback in a clean, manageable way.
    """

    def __init__(self, ai_model=None, db_session=None):
        """Initialize with optional AI model and database session."""
        self.ai_model = ai_model
        self.db_session = db_session
        self.start_time = None
        self.db_config = DatabaseConfig()

        # Initialize vector search client
        try:
            self.vector_search_client = VectorSearchClient()
            logger.debug("Vector search client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize vector search client: {e}")
            self.vector_search_client = None

        logger.debug("AistManager initialized")

    def begin_request(self):
        """Start timing a new request."""
        self.start_time = time.time()

    def get_response_time(self):
        """Calculate the response time so far."""
        if self.start_time:
            return time.time() - self.start_time
        return 0

    @with_request_id
    def answer_question(self, user_id, question, client_type=None):
        """
        Main entry point to answer a user question by combining search strategies.
        Supports resetting conversation context with specific phrases.
        """
        self.begin_request()
        logger.debug(f"Processing question for user {user_id}: {question}")
        local_session = None
        session_id = None
        combined_context = ""

        # Check for session end request
        if question.lower() == "end session please":
            logger.info("User requested to end the session")
            return {"status": "end_session"}

        # Define reset phrases that will clear the conversation context
        reset_phrases = [
            "clear context",
            "reset context",
            "clear conversation",
            "reset conversation",
            "forget previous context",
            "start fresh",
            "start over",
            "clear chat history",
            "reset chat",
            "forget what i said"
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

                # Update session with this interaction
                self.update_session(session_id, session_data, reset_response, session)
                formatted_answer = self.format_response(reset_response, client_type)

                return {"status": "success", "answer": formatted_answer, "method": "reset_context"}

            # First try keyword search for command-like queries
            with log_timed_operation("keyword_search"):
                keyword_result = self.try_keyword_search(question)
                if keyword_result.get('success'):
                    answer = self.format_response(keyword_result['answer'], client_type,
                                                  keyword_result.get('results', []))
                    self.record_interaction(user_id, question, keyword_result['answer'])
                    self.update_session(session_id, session_data, answer, session)

                    # Update conversation summary using the class method
                    summarized_data = session_data[-3:] if len(session_data) > 3 else session_data
                    ChatSession.update_conversation_summary(session_id, summarized_data, session)

                    return {"status": "success", "answer": answer, "method": "keyword"}

            # Run full-text search to collect context
            fulltext_content = ""
            with log_timed_operation("fulltext_search"):
                try:
                    if session:
                        fts_documents = CompleteDocument.search_by_text(
                            question,
                            session=session,
                            similarity_threshold=50,  # Lower threshold to get more results
                            with_links=False  # Get document objects instead of HTML links
                        )

                        if fts_documents and isinstance(fts_documents, list) and len(fts_documents) > 0:
                            fulltext_content += "\n\nFULL-TEXT SEARCH RESULTS:\n"
                            for i, doc in enumerate(fts_documents[:3], 1):
                                title = getattr(doc, 'title', f"Document #{getattr(doc, 'id', i)}")
                                content = getattr(doc, 'content', '')
                                snippet = content[:300] + "..." if content and len(content) > 300 else content
                                fulltext_content += f"{i}. {title}: {snippet}\n\n"

                            logger.info(f"Added {len(fts_documents[:3])} documents from full-text search to context")
                            combined_context += fulltext_content
                except Exception as e:
                    logger.error(f"Error collecting full-text search results: {e}", exc_info=True)

            # Get most relevant document if available
            most_relevant_doc = None
            with log_timed_operation("find_relevant_document"):
                most_relevant_doc = self.find_most_relevant_document(question, session)
                if most_relevant_doc:
                    logger.info(f"Found most relevant document with ID {most_relevant_doc.id}")
                    doc_content = most_relevant_doc.content
                    doc_snippet = doc_content[:1000] + "..." if len(doc_content) > 1000 else doc_content
                    combined_context += f"\n\nMOST RELEVANT DOCUMENT:\n{doc_snippet}\n\n"

            # Prepare to send everything to the AI model
            ai_model = self._ensure_ai_model()
            if not ai_model:
                logger.error("Failed to load AI model")
                return {"status": "error", "error": "Failed to load AI model"}

            # Get conversation history for context using the class method
            conversation_summary = ChatSession.get_conversation_summary(session_id, session)

            # Format conversation context
            conversation_context = ""
            if conversation_summary:
                # Limit to last 5 messages
                recent_msgs = conversation_summary[-5:] if len(conversation_summary) > 5 else conversation_summary
                conversation_context = "\n".join([f"User: {msg}" if i % 2 == 0 else f"Assistant: {msg}"
                                                  for i, msg in enumerate(recent_msgs)])

            # Create comprehensive prompt
            comprehensive_prompt = (
                "You are a helpful AI assistant for a technical knowledge base system. "
                "Answer the user's question based on the search results provided below. "
                "If the search results don't contain relevant information, provide a helpful response "
                "based on your general knowledge, but prioritize information from the search results "
                "if available.\n\n"
            )

            if conversation_context:
                comprehensive_prompt += f"PREVIOUS CONVERSATION:\n{conversation_context}\n\n"

            if combined_context:
                comprehensive_prompt += f"SEARCH RESULTS:{combined_context}\n\n"

            comprehensive_prompt += (
                f"USER QUESTION: {question}\n\n"
                "Please provide a concise, helpful answer based on the above information."
            )

            # Get AI response
            with log_timed_operation("ai_comprehensive_response"):
                ai_answer = ai_model.get_response(comprehensive_prompt)

            # Format and record response
            formatted_answer = self.format_response(ai_answer, client_type)
            self.record_interaction(user_id, question, ai_answer)

            # Update session with the interaction
            self.update_session(session_id, session_data, ai_answer, session)

            # Create summarized session data for the summary
            summarized_session_data = session_data[-3:] if len(session_data) > 3 else session_data
            summarized_session_data.append(question)
            summarized_session_data.append(ai_answer)

            # Update the conversation summary using the class method
            ChatSession.update_conversation_summary(session_id, summarized_session_data, session)

            return {"status": "success", "answer": formatted_answer, "method": "ai_comprehensive"}

        except Exception as e:
            logger.error(f"Error in answer_question: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
        finally:
            if local_session:
                local_session.close()
                logger.debug("Local database session closed")

    @with_request_id
    def find_most_relevant_document(self, question, session=None, request_id=None):
        """
        Find the most relevant document for a given question using vector similarity.

        Optimized for performance with:
        1. Batched processing
        2. Query-side filtering
        3. In-memory similarity calculation
        4. Simple LRU caching

        Args:
            question: The user's question to find a relevant document for
            session: Optional SQLAlchemy session (will use self.db_session if not provided)
            request_id: Optional request ID for tracking

        Returns:
            Document object that is most relevant or None if no relevant document found
        """
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

            # Check cache first (initialize if needed)
            if not hasattr(self, '_document_cache'):
                self._document_cache = {}  # Could be replaced with an LRU cache

            cache_key = f"{question}:{embedding_model_name}"
            if cache_key in self._document_cache:
                logger.info("Using cached document result")
                cached_doc_id = self._document_cache[cache_key]
                if cached_doc_id is None:
                    return None
                return session.query(Document).get(cached_doc_id)

            # Load embedding model and generate embedding for the question
            embedding_model = ModelsConfig.load_embedding_model(embedding_model_name)
            question_embedding = embedding_model.get_embeddings(question)

            if not question_embedding:
                logger.info("No embeddings generated. Returning None.")
                self._document_cache[cache_key] = None
                return None

            # Convert to numpy array for faster calculations
            import numpy as np
            question_embedding_np = np.array(question_embedding)

            # Process in batches to reduce memory usage
            BATCH_SIZE = 100
            most_relevant_document = None
            most_relevant_document_id = None
            highest_similarity = -1
            threshold = 0.01  # Minimum similarity threshold

            # Get total count for progress logging
            total_docs = session.query(Document).join(DocumentEmbedding).filter(
                DocumentEmbedding.model_name == embedding_model_name
            ).count()

            logger.info(f"Searching through {total_docs} documents")

            # Process in batches
            for offset in range(0, total_docs, BATCH_SIZE):
                # Get a batch of document IDs with the current embedding model
                doc_batch = session.query(Document.id).join(DocumentEmbedding).filter(
                    DocumentEmbedding.model_name == embedding_model_name
                ).offset(offset).limit(BATCH_SIZE).all()

                doc_ids = [d[0] for d in doc_batch]

                if not doc_ids:
                    continue

                # Get embeddings for this batch
                embeddings_batch = session.query(DocumentEmbedding).filter(
                    DocumentEmbedding.document_id.in_(doc_ids),
                    DocumentEmbedding.model_name == embedding_model_name
                ).all()

                # Calculate similarities
                for embedding_record in embeddings_batch:
                    try:
                        # Extract binary embedding and convert to list
                        if isinstance(embedding_record.model_embedding, bytes):
                            try:
                                # Try to decode as JSON first
                                doc_embedding = np.array(json.loads(embedding_record.model_embedding.decode('utf-8')))
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                # If not JSON, try as binary float array
                                doc_embedding = np.frombuffer(embedding_record.model_embedding, dtype=np.float32)
                        else:
                            # If it's already a list or similar
                            doc_embedding = np.array(embedding_record.model_embedding)

                        # Normalize vectors for cosine similarity
                        question_norm = np.linalg.norm(question_embedding_np)
                        doc_norm = np.linalg.norm(doc_embedding)

                        if question_norm > 0 and doc_norm > 0:
                            # Calculate cosine similarity
                            similarity = np.dot(question_embedding_np, doc_embedding) / (question_norm * doc_norm)

                            if similarity > highest_similarity:
                                highest_similarity = similarity
                                most_relevant_document_id = embedding_record.document_id
                                logger.debug(
                                    f"New highest similarity: {similarity:.4f} for document ID {most_relevant_document_id}")
                    except Exception as e:
                        logger.warning(f"Error processing embedding for document {embedding_record.document_id}: {e}")

                logger.debug(f"Processed {offset + len(doc_ids)} of {total_docs} documents")

            # Return the most relevant document if above threshold
            if highest_similarity >= threshold and most_relevant_document_id is not None:
                # Fetch the full document only if we found a good match
                most_relevant_document = session.query(Document).get(most_relevant_document_id)
                logger.info(
                    f"Found most relevant document with ID {most_relevant_document_id} and similarity {highest_similarity:.4f}")
                self._document_cache[cache_key] = most_relevant_document_id
                return most_relevant_document
            else:
                logger.info("No relevant document found with sufficient similarity")
                self._document_cache[cache_key] = None
                return None

        except Exception as e:
            logger.error(f"An error occurred while finding the most relevant document: {e}", exc_info=True)
            return None
        finally:
            if local_session:
                local_session.close()

    @with_request_id
    def try_keyword_search(self, question):
        """Try to answer using keyword search."""
        logger.debug('Attempting keyword search')
        local_session = None
        try:
            # Use the provided session or get a new one from DatabaseConfig
            if not self.db_session:
                local_session = self.db_config.get_main_session()
                session = local_session
            else:
                session = self.db_session

            # Use KeywordAction to find the best match
            keyword, action, details = KeywordAction.find_best_match(question, session)

            if keyword and action:
                logger.info(f"Found matching keyword: {keyword} with action: {action}")

                # Execute the appropriate action based on the keyword match
                result = self.execute_keyword_action(keyword, action, details, question)

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

                    # Add performance note if slow
                    response_time = self.get_response_time()
                    if response_time > 1.5:
                        answer += f"<div class='performance-note'><small>Response time: {response_time:.2f}s</small></div>"
                        logger.warning(f"Slow keyword search response: {response_time:.2f}s")

                    return {'success': True, 'answer': answer, 'results': results}

            logger.debug("Keyword search found no matching keywords")
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
    def try_fulltext_search(self, question, request_id=None):
        """Try to answer using full-text search."""
        logger.debug('Attempting full-text search')
        local_session = None

        try:
            # Get session
            if not self.db_session:
                local_session = self.db_config.get_main_session()
                session = local_session
            else:
                session = self.db_session

            # Use CompleteDocument's search_by_text method directly
            fts_results = CompleteDocument.search_by_text(
                question,
                session=session,
                similarity_threshold=60,  # Lower threshold for better recall
                with_links=True,  # Get formatted HTML links since we want to display them
                request_id=request_id
            )

            if fts_results and isinstance(fts_results,
                                          str) and fts_results != "No documents found" and "error" not in fts_results.lower():
                logger.info('Full-text search found matching documents')
                answer = "Here are some documents that might help answer your question:\n" + fts_results

                # Add performance note if slow
                response_time = self.get_response_time()
                if response_time > 1.5:
                    answer += f"<div class='performance-note'><small>Response time: {response_time:.2f}s</small></div>"
                    logger.warning(f"Slow full-text search response: {response_time:.2f}s")

                return {'success': True, 'answer': answer}

            logger.debug("Full-text search found no matching documents")
            return {'success': False}

        except Exception as e:
            logger.error(f"Error in full-text search: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
        finally:
            # Close the session if we created it
            if local_session:
                local_session.close()
                logger.debug("Closed local database session for full-text search")

    @with_request_id
    def try_vector_search(self, user_id, question, request_id=None):
        """
        Try to answer using vector similarity search and AI model.

        Args:
            user_id: User ID for tracking
            question: User's question text
            request_id: Optional request ID for tracking

        Returns:
            Dictionary with search results and success status
        """
        logger.debug('Attempting vector similarity search')
        local_session = None
        try:
            # Use the provided session or get a new one from DatabaseConfig
            if not self.db_session:
                local_session = self.db_config.get_main_session()
                session = local_session
            else:
                session = self.db_session

            # Session management using ChatSession
            latest_session = self.get_session(user_id, session)
            if latest_session:
                session_id = latest_session.session_id
                session_data = latest_session.session_data
                session_data.append(question)
                logger.debug(f"Using existing session {session_id} for user {user_id}")
            else:
                session_id = self.create_session(user_id, question, session)
                session_data = [question]
                logger.debug(f"Created new session {session_id} for user {user_id}")

            # Find relevant document using our class method
            relevant_doc = self.find_most_relevant_document(question, session, request_id)

            if relevant_doc:
                doc_content = relevant_doc.content
                logger.info(f"Found relevant document (id: {relevant_doc.id}) via vector search")

                # Get conversation summary for context
                conversation_summary = self.get_conversation_summary(session_id, session)

                if conversation_summary:
                    conversation_summary = conversation_summary[-10:]  # Keep last 10 entries
                    logger.debug(f"Retrieved conversation summary with {len(conversation_summary)} entries")
                else:
                    conversation_summary = []
                    logger.debug("No conversation summary found")

                # Generate AI response
                prompt = f"Conversation Summary: {conversation_summary}\nDocument content (Row {relevant_doc.id}): {doc_content}\n\nQuestion: {question}\nAnswer:"

                # Use lazy loading to ensure AI model is available
                ai_model = self._ensure_ai_model()
                if ai_model:
                    with log_timed_operation("ai_response_generation"):
                        answer = ai_model.get_response(prompt)
                        logger.debug(f"Raw AI response: {answer[:100]}...")  # Log first 100 chars

                    # Enhance the AI response with our formatter
                    try:
                        enhanced_answer = self.enhance_ai_response(answer, question, relevant_doc)
                        logger.info("Successfully enhanced AI response")
                        answer = enhanced_answer
                    except Exception as enhance_err:
                        logger.error(f"Error enhancing AI response: {enhance_err}. Using raw response.")
                        # Continue with the raw answer if enhancement fails

                    logger.info("Generated AI response using vector search results")
                else:
                    answer = "No model could be loaded for generating a response"
                    logger.warning("Failed to load AI model for generating response")

                # Update session and conversation summary
                update_success = self.update_session(session_id, session_data, answer, session)
                if not update_success:
                    logger.warning(f"Failed to update session {session_id}")

                # Create summarized session data for the summary
                summarized_session_data = session_data[-3:]  # Keep last 3 entries
                summarized_session_data.append(question)

                # Update the conversation summary
                summary_success = self.update_conversation_summary(session_id, summarized_session_data, session)
                if not summary_success:
                    logger.warning(f"Failed to update conversation summary for session {session_id}")

                logger.debug(f"Updated session {session_id} and conversation summary")

                return {'success': True, 'answer': answer}

            logger.debug("Vector search found no relevant documents")
            return {'success': False}
        except Exception as e:
            logger.error(f"Error in vector search: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
        finally:
            # Only close the session if we created it locally
            if local_session:
                local_session.close()
                logger.debug("Closed local database session")

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
                    img_data = Image.serve_image(image_id=img.id, session=session)
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

    def try_keyword_search(self, question):
        """
        Try to find answers using keyword-based search strategies.

        This method uses the advanced keyword search to find the most relevant
        resources based on keywords in the user's question.

        Args:
            question: User's question text

        Returns:
            Dictionary with search results and success status
        """
        logger.debug(f"Trying keyword search for: {question}")

        try:
            # Use our advanced keyword search method
            results = self.perform_advanced_keyword_search(question)

            if isinstance(results, dict) and results.get("status") == "success" and results.get("count", 0) > 0:
                # Format the results into a user-friendly answer
                result_items = results.get("results", [])

                if len(result_items) > 0:
                    # Create response based on entity types found
                    entity_counts = {}
                    for item in result_items:
                        item_type = item.get('type', results.get('entity_type', 'item'))
                        entity_counts[item_type] = entity_counts.get(item_type, 0) + 1

                    # Generate response text
                    response_parts = ["I found the following resources that might help:"]

                    for entity_type, count in entity_counts.items():
                        response_parts.append(f"- {count} {entity_type}{'s' if count > 1 else ''}")

                    response = "\n".join(response_parts)

                    return {
                        "success": True,
                        "answer": response,
                        "results": result_items
                    }

            return {"success": False}

        except Exception as e:
            logger.error(f"Error in keyword search: {e}", exc_info=True)
            return {"success": False}

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
        Format the response based on client type and results.

        Args:
            answer: Main answer text
            client_type: Type of client (e.g., 'web', 'api', 'chat')
            results: Optional list of search results

        Returns:
            Formatted response string or dictionary
        """
        # Basic formatting for all client types
        formatted_answer = answer.strip()

        # Add performance metrics if slow and not already included
        if self.start_time and time.time() - self.start_time > 2.0 and '<div class="performance-note">' not in formatted_answer:
            response_time = time.time() - self.start_time
            formatted_answer += f"<div class='performance-note'><small>Response time: {response_time:.2f}s</small></div>"
            logger.debug(f"Added performance metrics: {response_time:.2f}s")

        # Format URLs directly in the text if not already done
        if '<a href=' not in formatted_answer and ('http://' in formatted_answer or 'https://' in formatted_answer):
            formatted_answer = re.sub(
                r'(https?://[^\s]+)',
                r'<a href="\1" target="_blank">\1</a>',
                formatted_answer
            )

        if client_type == "web":
            # For web clients, add clickable links to results
            if results:
                result_links = []
                for result in results:
                    if 'url' in result and 'title' in result:
                        result_links.append(f"<a href='{result['url']}'>{result['title']}</a>")
                    elif 'id' in result and 'type' in result:
                        entity_type = result['type']
                        entity_id = result['id']
                        title = result.get('title') or result.get('name') or f"{entity_type} #{entity_id}"
                        url = f"/{entity_type}s/view/{entity_id}"
                        result_links.append(f"<a href='{url}'>{title}</a>")

                if result_links:
                    formatted_answer += "\n\n<div class='search-results'><h4>Search Results:</h4><ul>"
                    for link in result_links:
                        formatted_answer += f"<li>{link}</li>"
                    formatted_answer += "</ul></div>"

            return formatted_answer

        elif client_type == "api":
            # For API clients, return a structured response
            response_data = {
                "answer": formatted_answer,
                "results": results if results else []
            }

            # Include response time in API responses
            if self.start_time:
                response_data["response_time"] = time.time() - self.start_time

            return response_data

        # Default formatting (for chat or other clients)
        if results:
            # Add plain text result information
            result_items = []
            for result in results:
                if 'title' in result:
                    result_items.append(f"- {result['title']}")
                elif 'name' in result:
                    result_items.append(f"- {result['name']}")
                elif 'type' in result and 'id' in result:
                    result_items.append(f"- {result['type']} #{result['id']}")

            if result_items:
                formatted_answer += "\n\nSearch Results:\n" + "\n".join(result_items)

        return formatted_answer

    def begin_request(self):
        """Initialize request state for a new request."""
        # This can be expanded later with any initialization needed for each request
        pass

    def record_interaction(self, user_id, question, answer):
        """
        Record the interaction between the user and the system using the QandA class method.

        Args:
            user_id: ID of the user
            question: User's question
            answer: System's answer
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
                # Use the QandA class method to record the interaction
                QandA.record_interaction(user_id, question, answer, session)
            finally:
                # Only close the session if we created it
                if local_session:
                    local_session.close()

        except Exception as e:
            logger.error(f"Error in record_interaction: {e}", exc_info=True)

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




