#chatbot_bp.py
import logging
from flask import Blueprint, request, jsonify, current_app, url_for, redirect
import openai
from sqlalchemy.exc import SQLAlchemyError
from emtacdb_fts import (
    search_documents_fts, search_images_by_keyword, find_keyword_and_extract_detail,
    load_keywords_to_db, perform_action_based_on_keyword, load_keywords_and_patterns,
    find_most_relevant_document, create_session, update_session, get_session, QandA,
    ChatSession, Area, EquipmentGroup, Model, AssetNumber, Location, SiteLocation, Position,
    Document, Image, Drawing, Problem, Solution, CompleteDocument, PowerPoint, 
    PartsPositionImageAssociation, ImagePositionAssociation, DrawingPositionAssociation,
    CompletedDocumentPositionAssociation, ImageCompletedDocumentAssociation,
    ProblemPositionAssociation, ImageProblemAssociation, CompleteDocumentProblemAssociation,
    CompleteDocumentProblemAssociation, ImageSolutionAssociation
)
from datetime import datetime
import logging
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as LocalSession  # Import LocalSession
from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import declarative_base
from config import DATABASE_URL
import re
from utilities.auth_utils import logout
from blueprints.logout_bp import logout_bp
from flask import Flask, render_template, send_file, Blueprint, current_app, session, jsonify

app = Flask(__name__)
app.secret_key = '1234'

app.register_blueprint(logout_bp)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Enable SQLAlchemy SQL statement logging
sqlalchemy_logger = logging.getLogger('sqlalchemy.engine')
sqlalchemy_logger.setLevel(logging.INFO)  # Adjust the log level as needed
engine = create_engine(DATABASE_URL)
LocalSession = scoped_session(sessionmaker(bind=engine))
session = Session()

chatbot_bp = Blueprint('chatbot_bp', __name__)

@chatbot_bp.route('/update_qanda', methods=['POST'])
def update_qanda():
    logger.debug("Received request to update Q&A")
    user_id = request.json.get('user_id')
    question = request.json.get('question')
    answer = request.json.get('answer')
    rating = request.json.get('rating')
    comment = request.json.get('comment')
    
    logger.debug(f"Extracted data from request - user_id: {user_id}, question: {question}, answer: {answer}, rating: {rating}, comment: {comment}")

    try:
        session = LocalSession()
        logger.debug("Session created")
        
        last_qanda_entry = session.query(QandA).order_by(QandA.id.desc()).first()
        logger.debug(f"Last Q&A entry: {last_qanda_entry}")

        if last_qanda_entry and last_qanda_entry.rating is None and last_qanda_entry.comment is None:
            logger.debug("Updating the last Q&A entry with new rating and comment")
            last_qanda_entry.rating = rating
            last_qanda_entry.comment = comment
        else:
            logger.debug("Creating a new Q&A entry")
            new_qanda = QandA(
                user_id=user_id, 
                question=question, 
                answer=answer, 
                rating=rating, 
                comment=comment, 
                timestamp=datetime.now().isoformat()
            )
            session.add(new_qanda)
            logger.debug(f"New Q&A entry added: {new_qanda}")

        session.commit()
        logger.debug("Session committed successfully")
        return jsonify({'message': 'Q&A updated successfully'})
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error while updating Q&A: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()
        logger.debug("Session closed")

@chatbot_bp.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        logger.debug(f"Request data: {data}")
        
        user_id = data.get('userId')
        question = data.get('question').strip()
        rating = None
        comment = None
        
        logger.debug(f"Extracted user_id: {user_id}, question: {question}")
        
        timestamp = datetime.now().isoformat()
        logger.debug(f"Timestamp: {timestamp}")
        
        with current_app.app_context():
            session = LocalSession()
            keywords_and_patterns, keyword_to_action_mapping = load_keywords_and_patterns(session)
            logger.debug(f"Loaded keywords and patterns: {keywords_and_patterns}")
            logger.debug(f"Loaded keyword to action mapping: {keyword_to_action_mapping}")

        logger.debug(f"Received userId: {user_id}, question: {question}")

        if question.lower() == "end session please":
            logger.info("User requested to end the session")
            return redirect(url_for('logout_bp.logout'))

        logger.debug('Detecting keywords and extracting details')
        keyword, details = find_keyword_and_extract_detail(question, keywords_and_patterns, keyword_to_action_mapping, session)
        logger.debug(f"Detected keyword: {keyword}, details: {details}")
    
        if keyword:
            logger.debug('A keyword was detected, perform the action based on the keyword and details')
            action = keyword_to_action_mapping.get(keyword)
            logger.debug(f"Action for keyword '{keyword}': {action}")
            
            if action:
                action_result = perform_action_based_on_keyword(action, details)
                logger.debug(f"Action result: {action_result}")
                return jsonify({'answer': action_result})
            elif keyword == "search_documents":
                search_results = search_documents_fts(details)
                logger.debug(f"Search results for documents: {search_results}")
                return jsonify({'answer': format_search_results(search_results)})
            elif keyword == "search_images_keyword_bp":
                image_info = search_images_by_keyword(details, session=session)
                logger.debug(f"Image info: {image_info}")
                answer_message = "".join([f"<img src='{url_for('static', filename=image.file_path)}' alt='{image.title}' />" for image in image_info])
                return jsonify({'answer': answer_message})
        else:
            logger.debug('No keyword detected, proceed with AI logic')
            latest_session = get_session(user_id, session)
            logger.debug(f"Latest session: {latest_session}")
            
            if latest_session:
                session_id = latest_session.session_id
                session_data = latest_session.session_data if isinstance(latest_session.session_data, list) else [latest_session.session_data]
                session_data.append(question)
                logger.debug(f"Updated session data: {session_data}")
            else:
                session_id = create_session(user_id, "User: " + question, session)
                session_data = [question]
                logger.debug(f"Created new session with ID {session_id} and session data: {session_data}")

            relevant_doc = find_most_relevant_document(question, session)
            logger.debug(f"Relevant document: {relevant_doc}")
            
            if relevant_doc:
                doc_content = relevant_doc.content
                conversation_summary = get_conversation_summary(session_id)
                logger.debug(f"Conversation summary: {conversation_summary}")
                
                if conversation_summary:
                    conversation_summary = conversation_summary[-10:]
                else:
                    conversation_summary = []

                prompt = f"Conversation Summary: {conversation_summary}\nDocument content (Row {relevant_doc.id}): {doc_content}\n\nQuestion: {question}\nAnswer:"
                logger.debug(f"Generated prompt for OpenAI: {prompt}")
                
                response = openai.Completion.create(
                    engine="gpt-3.5-turbo-instruct",
                    prompt=prompt,
                    max_tokens=600
                )
                logger.debug(f"OpenAI response: {response}")
                
                answer = response.choices[0].text.strip()
                session_data.append(question)
                logger.debug(f"Appended question to session data: {session_data}")
                
                update_session(session_id, session_data, answer, session)
                now = datetime.now().isoformat()
                new_qanda = QandA(user_id=user_id, question=question, answer=answer, rating=rating, comment=comment, timestamp=timestamp)
                session.add(new_qanda)
                session.commit()
                logger.debug("Updated Q&A and session successfully.")
            else:
                answer = "No relevant document found."
                logger.debug("No relevant document found.")

            summarized_session_data = session_data[-3:]
            summarized_session_data.append(question)
            logger.debug(f"Summarized session data: {summarized_session_data}")
            
            update_conversation_summary(session_id, summarized_session_data)

            answer_with_links = answer.replace('http://', '<a href="http://">').replace('https://', '<a href="https://">') + "</a>"
            logger.debug(f"Final answer with links: {answer_with_links}")
            return jsonify({'answer': answer_with_links})
    except SQLAlchemyError as e:
        session.rollback()
        logger.error("Database error: %s", e)
        return jsonify({'error': 'Database error occurred.'}), 500
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        return jsonify({'error': 'An unexpected error occurred.'}), 500

def update_conversation_summary(session_id, summarized_session_data):
    logger.debug(f"Updating conversation summary for session ID: {session_id}")
    try:
        with LocalSession() as session:
            chat_session = session.query(ChatSession).filter(ChatSession.session_id == session_id).first()
            if chat_session:
                chat_session.conversation_summary = summarized_session_data
                session.commit()
                logger.debug(f"Conversation summary updated for session ID {session_id}")
            else:
                logger.error(f"No chat session found for session ID {session_id}")
    except SQLAlchemyError as e:
        logger.error(f"Database error while updating conversation summary for session ID {session_id}: {e}")
        session.rollback()

def get_conversation_history(session_id):
    logger.debug(f"Attempting to retrieve conversation history for session ID: {session_id}")
    try:
        with LocalSession() as session:
            chat_session = session.query(ChatSession).filter(ChatSession.session_id == session_id).first()
            if chat_session:
                logger.debug(f"Conversation history found for session ID {session_id}: {chat_session.session_data}")
                return chat_session.session_data
            else:
                logger.debug(f"No conversation history found for session ID {session_id}")
                return ""
    except SQLAlchemyError as e:
        LocalSession.rollback()
        logger.error(f"Database error while getting conversation history for session ID {session_id}: {e}")
        return ""

def generate_html_links(documents):
    logger.debug("Generating HTML links for documents")
    html_links = []
    for document in documents:
        html_link = f"<a href='{document.link}'>{document.title}</a>"
        html_links.append(html_link)
        logger.debug(f"Generated HTML link for document ID {document.id}: {html_link}")
    return html_links

def add_qa_to_session(session_id, question, answer, db_session):
    logger.debug(f"Adding Q&A to session. Session ID: {session_id}, Question: {question}, Answer: {answer}")
    chat_session = db_session.query(ChatSession).filter_by(session_id=session_id).first()
    if not chat_session:
        logger.error(f"No chat session found for session ID {session_id}. Unable to add Q&A.")
        return  # Handle error or create a new session
    
    qa_list = chat_session.session_data or []
    qa_list.append({"question": question, "answer": answer})
    logger.debug(f"Updated Q&A list for session ID {session_id}: {qa_list}")
    
    chat_session.session_data = qa_list[-10:]  # Keep only the last 10 Q&A pairs
    db_session.commit()
    logger.debug(f"Committed Q&A to the database for session ID {session_id}")

def prepare_prompt(session_id, db_session):
    logger.debug(f"Preparing prompt for session ID: {session_id}")
    chat_session = db_session.query(ChatSession).filter_by(session_id=session_id).first()
    if not chat_session:
        logger.debug(f"No chat session found for session ID {session_id}. Returning fallback prompt.")
        return "User:"  # Fallback prompt if session not found
    
    prompt = "\n".join([f"User: {qa['question']}\nBot: {qa['answer']}" for qa in chat_session.session_data]) + "\nUser:"
    logger.debug(f"Constructed prompt for session ID {session_id}: {prompt}")
    return prompt

def get_conversation_summary(session_id):
    logger.debug(f"Attempting to retrieve conversation summary for session ID: {session_id}")
    try:
        with LocalSession() as session:
            chat_session = session.query(ChatSession).filter(ChatSession.session_id == session_id).first()
            if chat_session:
                conversation_summary = chat_session.conversation_summary or ""
                logger.debug(f"Retrieved conversation summary for session ID {session_id}: {conversation_summary}")
                return conversation_summary
            else:
                logger.debug(f"No conversation summary found for session ID {session_id}")
                return ""
    except SQLAlchemyError as e:
        LocalSession.rollback()
        logger.error(f"Database error while getting conversation summary for session ID {session_id}: {e}")
        return ""
