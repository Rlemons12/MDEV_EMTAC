import logging
from flask import Blueprint, send_file, request, redirect, url_for, flash, render_template, session as flask_session
from emtacdb_fts import PartsPositionImageAssociation, Position, Part, Image
from config_env import DatabaseConfig
import json
from sqlalchemy import or_
import os

# Instantiate the database configuration
db_config = DatabaseConfig()

# Blueprint setup (define this before using it)
create_bill_of_material_bp = Blueprint('create_bill_of_material_bp', __name__)

# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a file handler for logging
file_handler = logging.FileHandler('create_bill_of_material.log')
file_handler.setLevel(logging.DEBUG)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)

# Function to serve an image from the database based on its ID
def serve_image(session, image_id):
    logger.info(f"Attempting to serve image with ID: {image_id}")
    try:
        image = session.query(Image).filter_by(id=image_id).first()
        if image:
            logger.debug(f"Image found: {image.title}, File path: {image.file_path}")
            file_path = os.path.join(os.getenv('DATABASE_DIR'), image.file_path)
            if os.path.exists(file_path):
                logger.info(f"Serving file: {file_path}")
                return send_file(file_path, mimetype='image/jpeg', as_attachment=False)
            else:
                logger.error(f"File not found: {file_path}")
                return "Image file not found", 404
        else:
            logger.error(f"Image not found with ID: {image_id}")
            return "Image not found", 404
    except Exception as e:
        logger.error(f"An error occurred while serving the image: {e}")
        return "Internal Server Error", 500

@create_bill_of_material_bp.route('/serve_image/<int:image_id>')
def serve_image_route(image_id):
    logger.debug(f"Request to serve image with ID: {image_id}")
    db_session = db_config.get_main_session()
    try:
        return serve_image(db_session, image_id)
    except Exception as e:
        logger.error(f"Error serving image {image_id}: {e}")
        flash(f"Error serving image {image_id}", "error")
        return "Image not found", 404


@create_bill_of_material_bp.route('/create_bill_of_material', methods=['POST'])
def create_bill_of_material():
    try:
        logger.info('Received request to create Bill of Material.')

        # Retrieve form data
        area_id = request.form.get('area')
        equipment_group_id = request.form.get('equipment_group')
        model_id = request.form.get('model')
        asset_number_id = request.form.get('asset_number')
        location_id = request.form.get('location')

        logger.debug(f'Received form data: area_id={area_id}, equipment_group_id={equipment_group_id}, '
                     f'model_id={model_id}, asset_number_id={asset_number_id}, location_id={location_id}')

        # Get a session from db_config
        db_session = db_config.get_main_session()

        # Start building the query
        query = db_session.query(Position)

        # Apply filters based on provided parameters
        if area_id:
            query = query.filter(Position.area_id == int(area_id))
        if equipment_group_id:
            query = query.filter(Position.equipment_group_id == int(equipment_group_id))
        if model_id:
            query = query.filter(Position.model_id == int(model_id))
        if asset_number_id:
            query = query.filter(Position.asset_number_id == int(asset_number_id))
        if location_id:
            query = query.filter(Position.location_id == int(location_id))

        # Execute the query to get all matching positions
        positions = query.all()

        if not positions:
            logger.warning('No matching positions found.')
            flash('No matching positions found for the provided input.', 'error')
            return render_template('bill_of_materials.html')

        logger.info(f'Found {len(positions)} matching positions.')

        # Aggregate all associated Part and Image IDs from the PartsPositionImageAssociation table
        associated_parts_images = []
        for position in positions:
            parts_images = db_session.query(PartsPositionImageAssociation).filter_by(position_id=position.id).all()
            associated_parts_images.extend(parts_images)

        if not associated_parts_images:
            logger.warning('No parts or images found for the selected positions.')
            flash('No parts or images found for the selected positions.', 'error')
            return render_template('bill_of_materials.html')

        logger.info(f'Found {len(associated_parts_images)} associated parts and images.')

        # Prepare data for display and serialize it for session storage
        results = [{'part_id': association.part_id, 'image_id': association.image_id} for association in associated_parts_images]
        flask_session['results'] = json.dumps(results)
        flask_session['model_id'] = model_id
        flask_session['asset_number_id'] = asset_number_id
        flask_session['location_id'] = location_id

        logger.info('Serialized results and stored them in session.')

        # Start with the first item (index 0)
        return redirect(url_for('create_bill_of_material_bp.view_bill_of_material', index=0))

    except Exception as e:
        logger.error(f'An error occurred: {str(e)}')
        flash(f'An error occurred: {str(e)}', 'error')
        return render_template('bill_of_materials.html')

@create_bill_of_material_bp.route('/view_bill_of_material', methods=['GET'])
def view_bill_of_material():
    try:
        # Get parameters from the request
        index = int(request.args.get('index', 0))
        per_page = int(request.args.get('per_page', 4))

        results = json.loads(flask_session['results'])

        if index < 0 or index >= len(results):
            index = 0  # Reset to start if out of range

        paginated_results = results[index:index + per_page]

        db_session = db_config.get_main_session()
        parts_and_images = [
            {
                'part': db_session.query(Part).filter_by(id=item['part_id']).first(),
                'image': db_session.query(Image).filter_by(id=item['image_id']).first()
            }
            for item in paginated_results
        ]

        model_name = db_session.query(Position.model).filter_by(id=flask_session.get('model_id')).first().model if flask_session.get('model_id') else None
        asset_number = db_session.query(Position.asset_number).filter_by(id=flask_session.get('asset_number_id')).first().asset_number if flask_session.get('asset_number_id') else None
        location_name = db_session.query(Position.location).filter_by(id=flask_session.get('location_id')).first().location if flask_session.get('location_id') else None

        next_index = index + per_page if index + per_page < len(results) else None
        prev_index = index - per_page if index - per_page >= 0 else None

        return render_template('bill_of_material_creation_results.html',
                               index=index,
                               parts_and_images=parts_and_images,
                               per_page=per_page,
                               total=len(results),
                               next_index=next_index,
                               prev_index=prev_index,
                               model_name=model_name,
                               asset_number=asset_number,
                               location_name=location_name)

    except Exception as e:
        logger.error(f'An error occurred: {str(e)}')
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('create_bill_of_material_bp.view_bill_of_material', index=max(0, index - per_page)))

@create_bill_of_material_bp.route('/bom_general_search', methods=['POST'])
def bom_general_search():
    try:
        # Get the search parameters from the form
        general_asset_number = request.form.get('general_asset_number', '').strip()
        general_location = request.form.get('general_location', '').strip()

        logger.info(f"General Search initiated with Asset Number: {general_asset_number} and Location: {general_location}")

        # Get a session from db_config
        db_session = db_config.get_main_session()

        # Start building the query
        query = db_session.query(Position)

        # Filter by asset number or location if provided
        if general_asset_number:
            query = query.filter(Position.asset_number_id == general_asset_number)
        if general_location:
            query = query.filter(Position.location_id == general_location)

        # Execute the query to get all matching positions
        positions = query.all()

        if not positions:
            flash('No results found for the given Asset Number or Location.', 'error')
            logger.warning('No matching positions found for the general search.')
            return render_template('bill_of_materials.html')

        logger.info(f'Found {len(positions)} matching positions for the general search.')

        # Aggregate all associated Part and Image IDs from the PartsPositionImageAssociation table
        associated_parts_images = []
        for position in positions:
            parts_images = db_session.query(PartsPositionImageAssociation).filter_by(position_id=position.id).all()
            associated_parts_images.extend(parts_images)

        if not associated_parts_images:
            flash('No parts or images found for the selected Asset Number or Location.', 'error')
            logger.warning('No parts or images found for the general search.')
            return render_template('bill_of_materials.html')

        logger.info(f'Found {len(associated_parts_images)} associated parts and images for the general search.')

        # Prepare data for display and serialize it for session storage
        results = [{'part_id': association.part_id, 'image_id': association.image_id} for association in associated_parts_images]
        flask_session['results'] = json.dumps(results)
        flask_session['general_asset_number'] = general_asset_number
        flask_session['general_location'] = general_location

        logger.info('Serialized general search results and stored them in session.')

        # Start with the first item (index 0)
        return redirect(url_for('create_bill_of_material_bp.view_bill_of_material', index=0))

    except Exception as e:
        logger.error(f'An error occurred during general search: {str(e)}')
        flash(f'An error occurred: {str(e)}', 'error')
        return render_template('bill_of_materials.html')
