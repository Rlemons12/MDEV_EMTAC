import logging
import os
import sys
import time
import re
import uuid
from modules.emtacdb.AI_Steward.aist import AistManager
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Blueprint, request, jsonify, current_app, url_for, redirect
from sqlalchemy.exc import SQLAlchemyError
from modules.emtacdb.emtacdb_fts import (QandA, ChatSession, KeywordSearch, CompleteDocument)
from datetime import datetime
from plugins.ai_modules.ai_models import ModelsConfig
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (logger, with_request_id, log_timed_operation, debug_id,
                                              info_id, error_id, warning_id)

# Initialize AI model at module level
ai_model = None

# Create blueprint
chatbot_bp = Blueprint('chatbot_bp', __name__)


@chatbot_bp.route('/update_qanda', methods=['POST'])
def update_qanda():
    """Update Q&A entries with ratings and comments."""
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]  # Generate shorter request ID

    info_id(f"Received request to update Q&A", request_id)

    try:
        # Parse request data
        user_id = request.json.get('user_id', 'anonymous')
        question = request.json.get('question', '')
        answer = request.json.get('answer', '')
        rating = request.json.get('rating')
        comment = request.json.get('comment')

        debug_id(
            f"Update data - user_id: {user_id}, rating: {rating}, comment length: {len(str(comment)) if comment else 0}",
            request_id)

        # Create database session using DatabaseConfig
        local_session = None
        db_config = None

        try:
            with log_timed_operation("create_db_session", request_id):
                db_config = DatabaseConfig()
                local_session = db_config.get_main_session()
                debug_id(f"Database session created for Q&A update", request_id)

            with log_timed_operation("update_qa_entry", request_id):
                last_qanda_entry = local_session.query(QandA).order_by(QandA.id.desc()).first()
                debug_id(f"Retrieved last Q&A entry (ID: {last_qanda_entry.id if last_qanda_entry else 'None'})",
                         request_id)

                if last_qanda_entry and last_qanda_entry.rating is None and last_qanda_entry.comment is None:
                    debug_id(f"Updating existing Q&A entry (ID: {last_qanda_entry.id})", request_id)
                    last_qanda_entry.rating = rating
                    last_qanda_entry.comment = comment
                else:
                    debug_id(f"Creating new Q&A entry", request_id)
                    new_qanda = QandA(
                        user_id=user_id,
                        question=question,
                        answer=answer,
                        rating=rating,
                        comment=comment,
                        timestamp=datetime.now().isoformat()
                    )
                    local_session.add(new_qanda)

                local_session.commit()
                info_id(f"Q&A updated successfully in {time.time() - start_time:.3f}s", request_id)
                return jsonify({'message': 'Q&A updated successfully'})

        except SQLAlchemyError as e:
            if local_session:
                local_session.rollback()
            error_id(f"Database error while updating Q&A: {e}", request_id, exc_info=True)
            return jsonify({'error': str(e)}), 500
        finally:
            if local_session:
                local_session.close()
                debug_id(f"Database session closed", request_id)

    except Exception as e:
        error_id(f"Unexpected error in update_qanda: {e}", request_id, exc_info=True)
        return jsonify({'error': 'An unexpected error occurred'}), 500


@chatbot_bp.route('/ask', methods=['POST'])
def ask():
    """Process questions and return answers using AistManager."""
    start_time = time.time()
    local_session = None
    request_id = str(uuid.uuid4())[:8]  # Generate shorter request ID for logs
    db_config = None

    info_id(f"New chat request received", request_id)

    try:
        # Parse request data
        data = request.json
        debug_id(f"Request data: {data}", request_id)

        user_id = data.get('userId', 'anonymous')
        question = data.get('question', '').strip()
        client_type = data.get('clientType', 'web')
        rating = data.get('rating')
        comment = data.get('comment')

        info_id(f"Processing question from user {user_id}: '{question[:100]}{'...' if len(question) > 100 else ''}'",
                request_id)
        debug_id(f"Client type: {client_type}, Rating: {rating}, Comment length: {len(str(comment)) if comment else 0}",
                 request_id)

        # Input validation
        if not question or len(question) < 3:
            warning_id(f"Question too short: '{question}'", request_id)
            return jsonify({
                'answer': "Please provide a more detailed question so I can help you better."
            })

        # Check for direct session end request
        if question.lower() == "end session please":
            info_id(f"User {user_id} requested to end the session", request_id)
            return redirect(url_for('logout_bp.logout'))

        # Ensure AI model is loaded with better error handling
        global ai_model
        model_name = None
        with log_timed_operation("load_ai_model", request_id):
            try:
                if ai_model is None:
                    # Get current model name from ModelsConfig
                    model_name = ModelsConfig.get_current_ai_model_name()
                    info_id(f"Loading AI model: {model_name}...", request_id)

                    # Load the model using ModelsConfig
                    ai_model = ModelsConfig.load_ai_model()
                    info_id(f"AI model loaded successfully: {type(ai_model).__name__}", request_id)
                else:
                    debug_id(f"Using existing AI model: {type(ai_model).__name__}", request_id)
                    # Store model name for existing model
                    model_name = type(ai_model).__name__
            except Exception as model_err:
                error_id(f"Error loading AI model: {model_err}", request_id, exc_info=True)
                try:
                    # Try explicit fallback to NoAIModel
                    warning_id(f"Attempting fallback to NoAIModel", request_id)
                    ai_model = ModelsConfig.load_ai_model('NoAIModel')
                    model_name = 'NoAIModel'
                    info_id(f"Successfully loaded fallback model: NoAIModel", request_id)
                except Exception as fallback_err:
                    error_id(f"Error loading fallback model: {fallback_err}", request_id, exc_info=True)
                    # Continue with None model - AistManager should handle this gracefully

        # Create database session using DatabaseConfig
        with log_timed_operation("create_db_session", request_id):
            db_config = DatabaseConfig()
            local_session = db_config.get_main_session()
            debug_id(f"Database session created", request_id)

        try:
            # Initialize AistManager with all necessary components
            with log_timed_operation("initialize_ai_steward", request_id):
                ai_steward = AistManager(ai_model=ai_model, db_session=local_session)
                debug_id(f"AistManager initialized", request_id)

            # Process question through AistManager
            with log_timed_operation("process_question", request_id):
                info_id(f"Sending question to AistManager for processing", request_id)
                result = ai_steward.answer_question(user_id, question, client_type)
                debug_id(f"AistManager result status: {result.get('status')}", request_id)

            # Check for special response types
            if result.get('status') == 'end_session':
                info_id(f"AistManager requested session end", request_id)
                return redirect(url_for('logout_bp.logout'))

            # Process successful responses
            if result.get('status') == 'success':
                answer = result.get('answer', '')
                method = result.get('method', 'unknown')
                answer_length = len(answer)

                info_id(f"Question answered using '{method}' strategy, answer length: {answer_length}", request_id)

                if answer_length > 200:
                    debug_id(f"Answer preview: {answer[:200]}...", request_id)

                # Record the Q&A interaction with model tracking
                with log_timed_operation("record_qa_interaction", request_id):
                    try:
                        # Determine if QandA has model_name field
                        has_model_field = hasattr(QandA, 'model_name')

                        if hasattr(QandA, 'record_interaction'):
                            try:
                                # Check if record_interaction accepts model_name parameter
                                param_count = QandA.record_interaction.__code__.co_argcount

                                if param_count >= 5:  # class method + 4 regular params
                                    debug_id(f"Recording Q&A using class method with model tracking", request_id)
                                    QandA.record_interaction(user_id, question, answer, local_session, model_name)
                                else:
                                    debug_id(f"Recording Q&A using class method without model tracking", request_id)
                                    QandA.record_interaction(user_id, question, answer, local_session)

                                debug_id(f"Successfully recorded Q&A interaction for user {user_id}", request_id)
                            except Exception as qa_err:
                                error_id(f"Error in record_interaction: {qa_err}", request_id, exc_info=True)
                                # Continue processing - Q&A recording is not critical for the user response
                        else:
                            # Fallback to direct record creation if no class method exists
                            try:
                                debug_id(f"Recording Q&A using direct record creation", request_id)
                                qa_data = {
                                    'user_id': user_id,
                                    'question': question,
                                    'answer': answer,
                                    'timestamp': datetime.now().isoformat()
                                }

                                # Add model_name if the field exists
                                if hasattr(QandA, 'model_name'):
                                    qa_data['model_name'] = model_name

                                new_qanda = QandA(**qa_data)
                                local_session.add(new_qanda)
                                local_session.commit()
                                debug_id(f"Successfully created Q&A record for user {user_id}", request_id)
                            except Exception as direct_qa_err:
                                error_id(f"Error creating Q&A record: {direct_qa_err}", request_id, exc_info=True)
                                local_session.rollback()

                        local_session.commit()
                        info_id(f"Recorded Q&A interaction for user {user_id}", request_id)
                    except Exception as qa_err:
                        error_id(f"Error recording Q&A: {qa_err}", request_id, exc_info=True)
                        local_session.rollback()
                        # Continue processing - this is not critical to the response

                # Add performance metrics if needed and not already included
                response_time = time.time() - start_time
                if response_time > 2.0 and '<div class="performance-note">' not in answer:
                    debug_id(f"Adding performance note for slow response ({response_time:.2f}s)", request_id)
                    answer += f"<div class='performance-note'><small>Response time: {response_time:.2f}s</small></div>"

                info_id(f"Request completed successfully in {response_time:.2f}s", request_id)
                return jsonify({'answer': answer})
            else:
                # Handle error case from AistManager
                error_message = result.get('error',
                                           "I encountered an issue while processing your question. Please try again.")
                error_id(f"AistManager failed to provide a successful response: {result}", request_id)

                # Add a small delay to ensure a consistent user experience
                if time.time() - start_time < 1.0:
                    delay_time = 1.0 - (time.time() - start_time)
                    debug_id(f"Adding delay of {delay_time:.2f}s for consistent UX", request_id)
                    time.sleep(delay_time)

                total_time = time.time() - start_time
                info_id(f"Request completed with error in {total_time:.2f}s", request_id)
                return jsonify({'answer': error_message}), 500

        except Exception as e:
            if local_session:
                local_session.rollback()
            error_id(f"Error in AI steward processing: {e}", request_id, exc_info=True)
            error_message = "I encountered an unexpected issue. Please try rephrasing your question."

            # Add a small delay to ensure a consistent user experience
            if time.time() - start_time < 1.0:
                delay_time = 1.0 - (time.time() - start_time)
                debug_id(f"Adding delay of {delay_time:.2f}s for consistent UX", request_id)
                time.sleep(delay_time)

            total_time = time.time() - start_time
            info_id(f"Request completed with exception in {total_time:.2f}s", request_id)
            return jsonify({'answer': error_message}), 500
        finally:
            if local_session:
                local_session.close()
                debug_id(f"Database session closed", request_id)

    except SQLAlchemyError as e:
        error_id(f"Database error: {e}", request_id, exc_info=True)
        error_message = "I'm having trouble accessing the information you need right now. Please try again in a moment."
        total_time = time.time() - start_time
        info_id(f"Request completed with database error in {total_time:.2f}s", request_id)
        return jsonify({'answer': error_message}), 500
    except Exception as e:
        error_id(f"Unexpected error: {e}", request_id, exc_info=True)
        error_message = "I encountered an unexpected issue. Please try rephrasing your question."
        total_time = time.time() - start_time
        info_id(f"Request completed with unexpected error in {total_time:.2f}s", request_id)
        return jsonify({'answer': error_message}), 500