import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from emtacdb_fts import  (QandA,
    ChatSession, Area, EquipmentGroup, Model, AssetNumber, Location, SiteLocation, Position,
    Document, Image, Drawing, Problem, Solution, CompleteDocument, Part, ImageEmbedding, PowerPoint, 
    PartsPositionImageAssociation, ImagePositionAssociation, DrawingPositionAssociation,
    CompletedDocumentPositionAssociation, ImageCompletedDocumentAssociation,
    ProblemPositionAssociation, ImageProblemAssociation, CompleteDocumentProblemAssociation,
    ImageSolutionAssociation)
from emtac_revision_control_db import AreaSnapshot
from flask import Flask, render_template, redirect, url_for, session, flash, request
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from config import UPLOAD_FOLDER, DATABASE_URL
from emtacdb_fts import serve_image, Image, UserLevel
import logging
import webbrowser
from threading import Timer
from blueprints import register_blueprints
from emtacdb_fts import load_config_from_db
from config import DATABASE_DIR,DATABASE_URL,REVISION_CONTROL_DB_PATH
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.orm import declarative_base, configure_mappers, relationship, scoped_session, sessionmaker
from event_listeners import register_event_listeners

# Set the Python path to include the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize database engines
engine = create_engine(DATABASE_URL)
revision_control_engine = create_engine(f'sqlite:///{REVISION_CONTROL_DB_PATH}')

# Log SQL queries executed by SQLAlchemy
@event.listens_for(engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    logger.debug(f"Executing query: {statement}")
    logger.debug(f"With parameters: {parameters}")

# Create sessions
SessionFactory = sessionmaker(bind=engine)
RevisionControlSessionFactory = sessionmaker(bind=revision_control_engine)

def create_app():
    app = Flask(__name__)
    app.secret_key = '1234'
    
    # Set the upload folder in the app's configuration
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
     
    # Register blueprints
    register_blueprints(app)
    
    # Register event listeners
    register_event_listeners()

    # Define routes
    @app.route('/')
    def index():
        session.permanent = False  # Make the session non-permanent
        user_id = session.get('user_id', '')  
        if not user_id:
            return redirect(url_for('login_bp.login'))  # Redirect to login page if user is not logged in
        
        # Load the current AI and embedding models from the database
        current_ai_model, current_embedding_model = load_config_from_db()

        return render_template('index.html', current_ai_model=current_ai_model, current_embedding_model=current_embedding_model)


    @app.route('/upload_image')
    def upload_image_page():
        session.permanent = False  # Make the session non-permanent
        total_pages = 1
        page = 1
        return render_template('upload_image.html', total_pages=total_pages, page=page)


    @app.route('/success')
    def upload_success():
        session.permanent = False  # Make the session non-permanent
        return render_template('success.html')

    @app.route('/view_pdf_by_title/<string:title>')
    def view_pdf_by_title_route(title):
        session.permanent = False  # Make the session non-permanent
        return view_pdf_by_title(title)

    @app.route('/serve_image/<int:image_id>')
    def serve_image_route(image_id):
        logger.debug(f"Request to serve image with ID: {image_id}")
        with SessionFactory() as session:
            try:
                return serve_image(session, image_id)
            except Exception as e:
                logger.error(f"Error serving image {image_id}: {e}")
                flash(f"Error serving image {image_id}", "error")
                return "Image not found", 404

    @app.route('/document_success')
    def document_upload_success():
        session.permanent = False  # Make the session non-permanent
        return render_template('success.html')

    @app.route('/troubleshooting_guide')
    def troubleshooting_guide():
        session.permanent = False  # Make the session non-permanent
        return render_template('troubleshooting_guide.html')
    
    @app.route('/tsg_search_problems')
    def tsg_search_problems():
        session.permanent = False  # Make the session non-permanent
        return render_template('tsg_search_problems.html')

    @app.route('/search_bill_of_material', methods=['GET'])
    def search_bill_of_material():
        return render_template('search_bill_of_material.html')

    return app


def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    Timer(1, open_browser).start()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
    app = create_app()
    app.run(debug=False)
