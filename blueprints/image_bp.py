import sys
import os
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import create_engine, event, inspect
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from flask import Blueprint, request, jsonify, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from modules.emtacdb.emtacdb_fts import (Session,ImagePositionAssociation)
from modules.emtacdb.utlity.main_database.database import (create_position, add_image_to_db,
                                                           create_image_position_association)
import logging
from plugins.ai_modules import  load_ai_model
from modules.emtacdb.utlity.revision_database.auditlog import commit_audit_logs
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.config import DATABASE_DIR, DATABASE_URL,DATABASE_PATH_IMAGES_FOLDER
from plugins.ai_modules import ModelsConfig
db_config = DatabaseConfig()

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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

current_ai_model = ModelsConfig.load_config_from_db()

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MINIMUM_SIZE = (100, 100)  # Define the minimum width and height for the image

# Create a blueprint for the image routes
image_bp = Blueprint('image_bp', __name__)

# Helper Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to serve the upload_search_database.html page
@image_bp.route('/upload_search_database', methods=['GET'])
def upload_image_page():
    filename = request.args.get('filename', '')
    return render_template('upload_search_database/upload_search_database.html', filename=filename)

# Route for uploading images
@image_bp.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    try:
        # Obtain a session using the getter method
        with db_config.get_main_session() as session:
            # Inspect and log the available tables in the database
            inspector = inspect(session.bind)
            tables = inspector.get_table_names()
            logger.info(f"Available tables in the database: {tables}")

            if request.method == 'POST':
                # Retrieve form data
                title = request.form.get('title')
                image_file = request.files.get('image')

                if not image_file:
                    logger.error("No image file part in the request")
                    return jsonify({'message': 'No image file part in the request'}), 400

                if image_file.filename == '':
                    logger.error("No file selected for uploading")
                    return jsonify({'message': 'No file selected for uploading'}), 400

                # If no title is provided, use the filename (without extension) as the title
                filename = secure_filename(image_file.filename)
                if not title:
                    title = os.path.splitext(filename)[0]

                # Collect general metadata from the request
                area = request.form.get('area', None)
                equipment_group = request.form.get('equipment_group', None)
                model = request.form.get('model', None)
                asset_number = request.form.get('asset_number', None)
                location = request.form.get('location', None)
                site_location = request.form.get('site_location', None)
                description = request.form.get('description', None)

                # Log the received metadata
                logger.info(f"Received metadata - Area: {area}, Equipment Group: {equipment_group}, Model: {model}, "
                            f"Asset Number: {asset_number}, Location: {location}, Site Location: {site_location}, "
                            f"Description: {description}")

                # Helper function to safely convert form data to integers
                def convert_to_int(value, field_name):
                    try:
                        return int(value) if value and value.isdigit() else None
                    except ValueError:
                        logger.warning(f"Invalid value for {field_name}: {value}")
                        return None

                # Convert metadata to integer IDs if possible
                area_id = convert_to_int(area, 'area')
                equipment_group_id = convert_to_int(equipment_group, 'equipment_group')
                model_id = convert_to_int(model, 'model')
                asset_number_id = convert_to_int(asset_number, 'asset_number')
                location_id = convert_to_int(location, 'location')
                site_location_id = convert_to_int(site_location, 'site_location')

                logger.debug(f"Processed form data - area_id={area_id}, equipment_group_id={equipment_group_id}, "
                             f"model_id={model_id}, asset_number_id={asset_number_id}, location_id={location_id}, "
                             f"site_location_id={site_location_id}")

                # Retrieve or create the position and get the ID
                position_id = create_position(area_id, equipment_group_id, model_id, asset_number_id, location_id,
                                              site_location_id, session)

                if position_id:
                    logger.info(f'Position ID: {position_id}')
                else:
                    logger.error('Failed to create or retrieve a position')
                    # No need for session.rollback() because the context manager handles it
                    return jsonify({'message': 'Failed to create or retrieve a position'}), 500

                # Check if the uploaded file is allowed
                if not allowed_file(image_file.filename):
                    logger.error("Unsupported file format")
                    return jsonify({'message': 'Unsupported file format'}), 400

                # Save the uploaded file
                file_path = os.path.join(DATABASE_PATH_IMAGES_FOLDER, filename)
                image_file.save(file_path)
                logger.info(f"Saved file to {file_path}")

                # Load the current AI model setting (if needed)
                current_ai_model = ModelsConfig.load_config_from_db()
                ai_model = load_ai_model(current_ai_model)
                logger.info(f"Using AI model: {current_ai_model}")

                # Generate description if not provided
                if not description:
                    description = ai_model.generate_description(file_path)
                    logger.info(f"Generated description: {description}")

                # Add the image metadata to the database
                new_image_id = add_image_to_db(title, file_path, position_id, None, None, description)

                if new_image_id:
                    logger.info(f"Added new image with ID: {new_image_id}")
                else:
                    logger.error("Failed to add the image to the database.")
                    return jsonify({'message': 'Failed to add the image to the database'}), 500

                # Create the ImagePositionAssociation entry
                try:
                    association = create_image_position_association(new_image_id, position_id, session)
                    logger.info(f"Created ImagePositionAssociation with ID: {association.id}")
                except Exception as assoc_e:
                    logger.error(f"Failed to create ImagePositionAssociation: {assoc_e}")
                    return jsonify({'message': 'Failed to associate image with position'}), 500

                # Commit the session after all operations
                session.commit()
                logger.info("Committed all changes to the database.")

                # Commit audit logs after the main operation
                commit_audit_logs()
                logger.info("Committed audit logs.")

                # Redirect to the upload success page
                return redirect(url_for('image_bp.upload_image_page', filename=filename))

        # If GET request, render the upload page
        return render_template('upload.html')

    except Exception as e:
        logger.error(f"An error occurred in upload_image: {e}", exc_info=True)
        # Attempt to commit audit logs even if an error occurs
        try:
            commit_audit_logs()
            logger.info("Committed audit logs despite the error.")
        except Exception as audit_e:
            logger.error(f"Failed to commit audit logs after error: {audit_e}")
        return jsonify({'message': 'An internal error occurred', 'error': str(e)}), 500

# Route for adding images
@image_bp.route('/add_image', methods=['POST'])
def add_image():

    try:
        with db_config.get_main_session() as session:
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
                current_ai_model = ModelsConfig.load_config_from_db()
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


