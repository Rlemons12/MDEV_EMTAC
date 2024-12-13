import sys
import os
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import create_engine, event, inspect
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from flask import Blueprint, request, jsonify, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from modules.emtacdb.emtacdb_fts import (Session, load_config_from_db)
from modules.emtacdb.utlity.main_database.database import create_position, add_image_to_db
import logging
from plugins.ai_models import load_ai_model
from modules.emtacdb.utlity.revision_database.auditlog import commit_audit_logs

from modules.configuration.config import DATABASE_DIR, DATABASE_URL,DATABASE_PATH_IMAGES_FOLDER

# Configure logging to write to a file with timestamps
log_file = f'logs/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'))

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Also add a stream handler for console output
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

# Database setup
engine = create_engine(
    DATABASE_URL, 
    pool_size=20, 
    max_overflow=30, 
    connect_args={"check_same_thread": False}
)

Session = scoped_session(sessionmaker(bind=engine))

# Revision control database configuration
REVISION_CONTROL_DB_PATH = os.path.join(DATABASE_DIR, 'emtac_revision_control_db.db')
revision_control_engine = create_engine(
    f'sqlite:///{REVISION_CONTROL_DB_PATH}',
    pool_size=20,            # Set a small pool size
    max_overflow=30,         # Allow up to 10 additional connections
    connect_args={"check_same_thread": False}  # Needed for SQLite when using threading
)

RevisionControlBase = declarative_base()
RevisionControlSession = scoped_session(sessionmaker(bind=revision_control_engine))  # Use distinct name
revision_control_session = RevisionControlSession()

# Apply PRAGMA settings to SQLite database
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA synchronous = OFF")
    cursor.execute("PRAGMA journal_mode = MEMORY")
    cursor.execute("PRAGMA cache_size = -64000")
    cursor.execute("PRAGMA temp_store = MEMORY")
    cursor.close()
# Create a blueprint for the image routes
image_bp = Blueprint('image_bp', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MINIMUM_SIZE = (100, 100)  # Define the minimum width and height for the image

# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to serve the upload_search_database.html page
@image_bp.route('/upload_search_database', methods=['GET'])
def upload_image_page():
    filename = request.args.get('filename', '')
    return render_template('upload_search_database.html', filename=filename)

# Route for uploading images
@image_bp.route('/upload_search_database', methods=['GET', 'POST'])
def upload_image():
    session = Session()  # Create a session at the beginning of the route
    try:
        # Inspect and log the available tables in the database
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        logger.info(f"Available tables in the database: {tables}")
        
        if request.method == 'POST':
            title = request.form.get('title')
            if not title:  # If no title is provided, use the filename as the title
                filename = secure_filename(request.files['image'].filename)
                title = os.path.splitext(filename)[0]

            # Collect general metadata from the request
            area = request.form.get('area', None)
            equipment_group = request.form.get('equipment_group', None)
            model = request.form.get('model', None)
            asset_number = request.form.get('asset_number', None)
            location = request.form.get('location', None)
            site_location = request.form.get('site_location', None)
            description = request.form.get('description', None)

            # Log the values
            logging.info(f"Area: {area}")
            logging.info(f"Equipment Group: {equipment_group}")
            logging.info(f"Model: {model}")
            logging.info(f"Asset Number: {asset_number}")
            logging.info(f"Location: {location}")
            logging.info(f"Site Location: {site_location}")
            logging.info(f"Description: {description}") 

            area_id = int(area) if area else None
            equipment_group_id = int(equipment_group) if equipment_group else None
            model_id = int(model) if model else None
            asset_number_id = int(asset_number) if asset_number else None
            location_id = int(location) if location else None
            site_location_id = int(site_location) if site_location else None

            logger.debug(f"Form data: area_id={area_id}, equipment_group_id={equipment_group_id}, model_id={model_id}, asset_number_id={asset_number_id}, location_id={location_id}, site_location_id={site_location_id}")
            
            # Retrieve or create the position and get the ID
            position_id = create_position(area_id, equipment_group_id, model_id, asset_number_id, location_id, site_location_id, session)

            if position_id:
                logger.info(f'Position ID: {position_id}')
            else:
                logger.error('Failed to create or retrieve a position')
                session.rollback()
                return jsonify({'message': 'Failed to create or retrieve a position'}), 500

            if 'image' not in request.files:
                logger.error("No image part in the request")
                session.rollback()
                return redirect(request.url)

            file = request.files['image']
            if file.filename == '':
                logger.error("No file selected for uploading")
                session.rollback()
                return redirect(request.url)

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                relative_path = os.path.join("DB_IMAGES", filename)
                file_path = os.path.join(DATABASE_PATH_IMAGES_FOLDER, filename)
                file.save(file_path)

                # Load the current AI model setting (if needed)
                current_ai_model, _ = load_config_from_db()
                ai_model = load_ai_model(current_ai_model)
                logger.info(f"Using AI model: {current_ai_model}")

                # Generate description if not provided
                if not description:
                    description = ai_model.generate_description(file_path)

                # Add the image metadata to the database
                new_image_id = add_image_to_db(title, file_path, position_id, None, None, description)

                session.commit()  # Commit the session after all operations

                # Commit audit logs after the main operation
                commit_audit_logs()

                if new_image_id:
                    return redirect(url_for('image_bp.upload_image_page', filename=filename))
                else:
                    logger.error("Failed to add the image to the database.")
                    session.rollback()
                    return jsonify({'message': 'Failed to add the image to the database'}), 500

        return render_template('upload.html')

    except Exception as e:
        logger.error(f"An error occurred in upload_image: {e}", exc_info=True)
        session.rollback()  # Rollback the session in case of an error
        commit_audit_logs()  # Ensure audit logs are committed even in case of an error
        return jsonify({'message': 'An internal error occurred', 'error': str(e)}), 500
    finally:
        session.close()  # Ensure the session is closed after the request is handled



# Route for adding images
@image_bp.route('/add_image', methods=['POST'])
def add_image():
    session = Session()  # Create a session at the beginning of the route
    try:
        logger.info("Received a POST request to /images/add_image")
        logger.info(f"Content-Type: {request.content_type}")

        # Collect and validate inputs from the form
        title = request.form.get('title')
        complete_document_id = request.form.get('complete_document_id')
        completed_document_position_association_id = request.form.get('completed_document_position_association_id')
        position_id = request.form.get('position_id')

        # Log the collected inputs
        logger.info(f"Title: {title}")
        logger.info(f"Complete Document ID: {complete_document_id}")
        logger.info(f"Completed Document Position Association ID: {completed_document_position_association_id}")
        logger.info(f"Position ID: {position_id}")

        # Check if an image file was provided in the request
        if 'image' not in request.files:
            logger.error("No image file part found in the request")
            return jsonify({'message': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            logger.error("No file selected for uploading")
            return jsonify({'message': 'No selected file'}), 400

        # Validate and save the image file
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(DATABASE_PATH_IMAGES_FOLDER, filename)
            file.save(file_path)
            logger.info(f"File saved successfully: {file_path}")

            # Generate title from filename if not provided
            if not title:
                title = os.path.splitext(filename)[0].replace('_', '')
                logger.info(f"No title provided, using filename as title: {title}")

            description = request.form.get('description', '')
            if not description:
                # Load AI model and generate description
                logger.info("No description provided, generating using AI model")
                current_ai_model, _ = load_config_from_db()
                ai_model = load_ai_model(current_ai_model)
                description = ai_model.generate_description(file_path)
                logger.info(f"Generated description: {description}")

            logger.info(f'title: {title}')
            logger.info(f'description: {description}')
            logger.info(f'position_id: {position_id}')
            logger.info(f'complete_document_id: {complete_document_id}')
            logger.info(f'complete_document_position_association_id: {completed_document_position_association_id}')

            # Pass the collected data to the add_image_to_db function
            logger.info("Passing data to add_image_to_db function")
            new_image_id = add_image_to_db(
                title=title,
                file_path=file_path,
                position_id=position_id,
                completed_document_position_association_id=completed_document_position_association_id,
                complete_document_id=complete_document_id,
                description=description
            )

            if new_image_id:
                logger.info(f"Successfully added image with ID {new_image_id}")
                return redirect(url_for('image_bp.upload_image_page', filename=filename))
            else:
                logger.error("Failed to add image to the database")
                return jsonify({'message': 'Failed to add image to the database'}), 500

        else:
            logger.error("Unsupported file type")
            return jsonify({'message': 'Unsupported file type'}), 400

    except Exception as e:
        logger.error(f"An internal error occurred in add_image: {e}", exc_info=True)
        return jsonify({'message': 'An internal error occurred', 'error': str(e)}), 500
    finally:
        session.close()


