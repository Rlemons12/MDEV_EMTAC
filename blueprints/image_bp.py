import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from flask import Blueprint, request, jsonify, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from emtacdb_fts import (add_image_to_db, Session, Image, FileLog, Location, Area, create_position, CompletedDocumentPositionAssociation, 
                         load_config_from_db, ImagePositionAssociation, load_image_model_config_from_db)
import logging
from PIL import Image as PILImage
from plugins.ai_models import load_ai_model 
from plugins.image_models import get_image_model_handler
from blueprints import TEMPORARY_FILES, OPENAI_API_KEY, DATABASE_PATH_IMAGES_FOLDER

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

# Route to serve the upload_image.html page
@image_bp.route('/upload_image', methods=['GET'])
def upload_image_page():
    filename = request.args.get('filename', '')
    return render_template('upload_image.html', filename=filename)

# Route for uploading images
@image_bp.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    try:
        if request.method == 'POST':
            title = request.form.get('title')
            if not title:  # If no title is provided, use the filename as the title
                filename = secure_filename(request.files['image'].filename)
                title = os.path.splitext(filename)[0]
            area = request.form.get('area', None)
            equipment_group = request.form.get('equipment_group', None)
            model = request.form.get('model', None)
            asset_number = request.form.get('asset_number', None)
            location = request.form.get('location', None)
            site_location = request.form.get('site_location', None)  # Get site_location from the form data, default to None
            description = request.form.get('description', None)
            
            area_id = int(area) if area else None
            equipment_group_id = int(equipment_group) if equipment_group else None
            model_id = int(model) if model else None
            asset_number_id = int(asset_number) if asset_number else None
            location_id = int(location) if location else None
            site_location_id = int(site_location) if site_location else None
            
            logger.debug(f"Form data: area_id={area_id}, equipment_group_id={equipment_group_id}, model_id={model_id}, asset_number_id={asset_number_id}, location_id={location_id}, site_location_id={site_location_id}")
            
            position_id = create_position(area_id, equipment_group_id, model_id, asset_number_id, location_id, site_location_id)
            
            if 'image' not in request.files:
                logger.error("No image part in the request")
                return redirect(request.url)
            file = request.files['image']
            if file.filename == '':
                logger.error("No file selected for uploading")
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # Define relative and absolute paths
                relative_path = os.path.join("DB_IMAGES", filename)
                file_path = os.path.join(DATABASE_PATH_IMAGES_FOLDER, filename)
                file.save(file_path)

                # Load the current AI model setting
                current_ai_model, _ = load_config_from_db()
                ai_model = load_ai_model(current_ai_model)
                logger.info(f"Using AI model: {current_ai_model}")

                # Generate description if not provided
                if not description:
                    description = ai_model.generate_description(file_path)

                # Add the image metadata to the database
                add_image_to_db(title, file_path, position_id, None, None, description)

                return redirect(url_for('image_bp.upload_image_page', filename=filename))
        return render_template('upload.html')
    except Exception as e:
        logger.error(f"An error occurred in upload_image: {e}", exc_info=True)
        return jsonify({'message': 'An internal error occurred', 'error': str(e)}), 500

# Route for adding images
@image_bp.route('/add_image', methods=['POST'])
def add_image():
    try:
        logger.info("Received a POST request to /images/add_image")
        logger.info(f"Content-Type: {request.content_type}")

        # Step 1: Collect and validate inputs from the form
        title = request.form.get('title')
        complete_document_id = request.form.get('complete_document_id')
        completed_document_position_association_id = request.form.get('completed_document_position_association_id')

        # Check if an image file was provided in the request
        if 'image' not in request.files:
            logger.error("No image file part found in the request")
            return jsonify({'message': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            logger.error("No file selected for uploading")
            return jsonify({'message': 'No selected file'}), 400

        # Step 2: Validate file type and save it to the appropriate directory
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

            # Step 3: Collect and validate additional form data (if applicable)
            area = request.form.get('area')
            equipment_group = request.form.get('equipment_group')
            model = request.form.get('model')
            asset_number = request.form.get('asset_number')
            location = request.form.get('location')
            site_location = request.form.get('site_location')

            # Convert to IDs or default to None
            area_id = int(area) if area else None
            equipment_group_id = int(equipment_group) if equipment_group else None
            model_id = int(model) if model else None
            asset_number_id = int(asset_number) if asset_number else None
            location_id = int(location) if location else None
            site_location_id = int(site_location) if site_location else None

            logger.debug(f"Form data collected: area_id={area_id}, equipment_group_id={equipment_group_id}, "
                         f"model_id={model_id}, asset_number_id={asset_number_id}, location_id={location_id}, site_location_id={site_location_id}")

            # Create position ID (if necessary)
            position_id = create_position(area_id, equipment_group_id, model_id, asset_number_id, location_id, site_location_id)
            logger.info(f"Position ID created: {position_id}")

            # Step 4: Validate the image and generate the embedding
            try:
                logger.info(f"Opening image: {file_path}")
                image = PILImage.open(file_path).convert("RGB")

                # Load the current image model for validation
                current_image_model = load_image_model_config_from_db()
                model_handler = get_image_model_handler(current_image_model)
                logger.info(f"Using model handler: {model_handler.__class__.__name__}")

                # Check if the image meets the required dimensions or aspect ratio
                logger.info("Validating image dimensions and aspect ratio")
                if not model_handler.is_valid_image(image):
                    logger.warning(f"Image does not meet the required dimensions or aspect ratio: {file_path}")
                    return jsonify({'message': 'Image does not meet the required dimensions or aspect ratio.'}), 400
                else:
                    logger.info("Image passed validation.")
                    model_embedding = model_handler.get_image_embedding(image)
                    model_name = model_handler.__class__.__name__
                    logger.info(f"Image embedding generated using model: {model_name}")

            except Exception as e:
                logger.error(f"Unable to process the image file: {e}")
                return jsonify({'message': 'Unable to process the image file.', 'error': str(e)}), 400

            # Step 5: Pass data to add_image_to_db to handle the database entry
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

    logger.info("Received a POST request to /images/add_image")
    logger.info(f"Content-Type: {request.content_type}")

    complete_document_id = request.form.get('complete_document_id')
    completed_document_position_association_id = request.form.get('completed_document_position_association_id')
    position_id = None

    if 'image' not in request.files:
        return jsonify({'message': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        logger.info(f"Received file: {file.filename}")

        filename = secure_filename(file.filename)
        logger.info(f"Secure filename: {filename}")

        title = request.form.get('title')
        logger.info(f"Title from form: {title}")

        if not title:
            filename_without_extension = os.path.splitext(filename)[0]
            title = filename_without_extension.replace('_', '')  # Remove underscores
            logger.info(f"No title provided, using filename: {title}")

        description = request.form.get('description', '')
        logger.info(f"Description from form: {description}")

        # Ensure the directory exists
        if not os.path.exists(DATABASE_PATH_IMAGES_FOLDER):
            os.makedirs(DATABASE_PATH_IMAGES_FOLDER)

        # Save the file to the DATABASE_PATH_IMAGES_FOLDER 
        relative_path = os.path.join("DB_IMAGES", filename)
        file_path = os.path.join(DATABASE_PATH_IMAGES_FOLDER, filename)
        file.save(file_path)
        logger.debug(f'saved image to folder DATABASE_PATH_IMAGES_FOLDER, filename')
        try:
            # Attempt to open the image file
            logger.debug(f' Attempt to open the image file')
            with PILImage.open(file_path) as img:
                width, height = img.size
                aspect_ratio = width / height

                # Check if image meets minimum size requirement
                if width >= MINIMUM_SIZE[0] and height >= MINIMUM_SIZE[1]:
                    logger.info("Image meets minimum size requirement.")

                    # Check if image meets maximum aspect ratio requirement (4:1)
                    if aspect_ratio <= 5:
                        logger.info("Aspect ratio is within the allowed range. Proceeding...")

                        # Load the current AI model setting
                        current_ai_model, _ = load_config_from_db()
                        ai_model = load_ai_model(current_ai_model)
                        logger.info(f"Using AI model: {current_ai_model}")

                        # If no description provided, generate one using the AI model
                        logger.debug(f'If no description provided, generate one using the AI model')
                        if not description:
                            description = ai_model.generate_description(file_path)
                            logger.info(f"in image/add_image Generated description: {description}")

                        logger.debug(f'if complete_document_id and completed_document_position_association_id:')    
                        if complete_document_id and completed_document_position_association_id:
                            logger.info(f"Complete document ID: {complete_document_id}, Completed Document Position Association ID: {completed_document_position_association_id}")
                            
                            logger.info("Adding image to database with existing position association.")
                            add_image_to_db(title, file_path, position_id, None, None, description)
                            logger.info("Added image to database with existing position association.")
                            
                        elif complete_document_id and not completed_document_position_association_id:
                            logger.info(f'elif complete_document_id and not completed_document_position_association_id:')
                            logger.info(f"Complete document ID: {complete_document_id}")
                            
                            area = request.form.get('area')
                            equipment_group = request.form.get('equipment_group')
                            model = request.form.get('model')
                            asset_number = request.form.get('asset_number')
                            location = request.form.get('location')
                            area_id = int(area) if area else None
                            equipment_group_id = int(equipment_group) if equipment_group else None
                            model_id = int(model) if model else None
                            asset_number_id = int(asset_number) if asset_number else None
                            location_id = int(location) if location else None
                            
                            logger.debug(f'Creating Position.id')
                            position_id = create_position(area_id, equipment_group_id, model_id, asset_number_id, location_id)
                            logger.debug(f'Created Position.id')
                            completed_document_position_association = CompletedDocumentPositionAssociation(
                                complete_document_id=complete_document_id,
                                position_id=position_id
                            )
                            session = Session()
                            try:
                                session.add(completed_document_position_association)
                                session.commit()
                                completed_document_position_association_id = completed_document_position_association.id
                            except Exception as e:
                                session.rollback()
                                logger.error(f"Error committing to database: {e}")
                                return jsonify({'message': 'Error committing to database.', 'error': str(e)}), 500
                            finally:
                                session.close()
                            logger.info("entering add_image_to_db function")
                            add_image_to_db(title, file_path, position_id, None, None, description)
                            logger.info("Added image to database with new position association.")
                            
                        return redirect(url_for('image_bp.upload_image_page', filename=filename))
                    else:
                        logger.error(f"Aspect ratio {aspect_ratio:.2f} exceeds the maximum allowed (4:1). Image not processed.")
                        return jsonify({'message': f'Aspect ratio {aspect_ratio:.2f} exceeds the maximum allowed. Image not processed.'}), 400
                else:
                    logger.error("Image does not meet minimum size requirement.")
                    return jsonify({'message': 'Image does not meet minimum size requirement.'}), 400
        except Exception as e:
            logger.error(f"Unable to process the image file: {e}")
            return jsonify({'message': 'Unable to process the image file.', 'error': str(e)}), 400
    else:
        return jsonify({'message': 'Unsupported file type'}), 400
