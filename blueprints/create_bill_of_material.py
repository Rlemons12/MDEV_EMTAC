import logging
from flask import Blueprint, send_file, request, redirect, url_for, flash, render_template, session as flask_session
from emtacdb_fts import PartsPositionImageAssociation, Position, Part, Image, BOMResult, AssetNumber
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
        response = serve_image(db_session, image_id)
        return response
    except Exception as e:
        logger.error(f"Error serving image {image_id}: {e}")
        flash(f"Error serving image {image_id}", "error")
        return "Image not found", 404
    finally:
        db_session.close()


@create_bill_of_material_bp.route('/create_bill_of_material', methods=['POST'])
def create_bill_of_material():
    db_session = db_config.get_main_session()
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

        # Start building the query
        query = db_session.query(Position)

        # Apply filters based on provided parameters with validation
        try:
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
        except ValueError as ve:
            logger.error(f"Invalid input data: {ve}")
            flash('Invalid input data provided.', 'error')
            return render_template('bill_of_materials.html')

        # Execute the query to get all matching positions
        positions = query.all()

        if not positions:
            logger.warning('No matching positions found.')
            flash('No matching positions found for the provided input.', 'error')
            return render_template('bill_of_materials.html')

        logger.info(f'Found {len(positions)} matching positions.')

        # Prepare results list
        results = []

        # Process each position and associated parts/images within the active session
        for position in positions:
            # Query for parts and images while the session is active
            parts_images = db_session.query(PartsPositionImageAssociation).filter_by(position_id=position.id).all()

            for association in parts_images:
                # Access the `part_id` and `image_id` while the session is active
                part = db_session.query(Part).filter_by(id=association.part_id).first()
                if part:
                    store_bom_results(db_session, part_id=part.id, position_id=position.id,
                                      image_id=association.image_id, description="Sample description")
                    # Collect result data while session is active
                    results.append({'part_id': part.id, 'image_id': association.image_id})
                else:
                    logger.error(f"No part found with ID {association.part_id}")

        # Commit the results and finalize the session
        db_session.commit()

        # Store results in session
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
        db_session.rollback()  # Roll back in case of error
        return render_template('bill_of_materials.html')

    finally:
        db_session.close()  # Ensure session is closed after processing

@create_bill_of_material_bp.route('/view_bill_of_material', methods=['GET'])
def view_bill_of_material():
    db_session = db_config.get_main_session()
    try:
        # Get parameters from the request
        index = request.args.get('index', 0, type=int)  # Current page index
        per_page = request.args.get('per_page', 4, type=int)  # Items per page

        # Query the BOMResult table for paginated results
        query = db_session.query(BOMResult).offset(index).limit(per_page)
        results = query.all()

        # Total number of results for pagination
        total_results = db_session.query(BOMResult).count()

        # Extract part, position, and image information for display
        parts_and_images = []
        for result in results:
            # Fetch the part
            part = db_session.query(Part).filter_by(id=result.part_id).first()
            if not part:
                logger.error(f"No part found with ID {result.part_id}")
                continue  # Skip this result or handle as needed

            # Fetch the image if image_id is not None
            if result.image_id is not None:
                image = db_session.query(Image).filter_by(id=result.image_id).first()
                if not image:
                    logger.warning(f"No image found with ID {result.image_id}")
                    image = None
            else:
                image = None  # No image associated

            parts_and_images.append({
                'part': part,
                'image': image,
                'description': result.description
            })

        # Pagination logic
        next_index = index + per_page if index + per_page < total_results else None
        prev_index = index - per_page if index - per_page >= 0 else None

        return render_template(
            'bill_of_material_creation_results.html',
            index=index,
            parts_and_images=parts_and_images,
            per_page=per_page,
            total=total_results,
            next_index=next_index,
            prev_index=prev_index
        )

    except Exception as e:
        logger.error(f'An error occurred: {str(e)}')
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('create_bill_of_materials_bp.bill_of_materials'))

    finally:
        db_session.close()

@create_bill_of_material_bp.route('/bom_general_search', methods=['POST'])
def bom_general_search():
    db_session = db_config.get_main_session()
    try:
        clear_bom_results()
        # Get the search parameters from the form
        general_asset_number = request.form.get('general_asset_number', '').strip()
        general_location = request.form.get('general_location', '').strip()

        logger.info(
            f"General Search initiated with Asset Number: {general_asset_number} and Location: {general_location}")

        # Initialize filters
        asset_number_ids = []
        location_ids = []

        # Start building the Position query
        query = db_session.query(Position)

        # Apply asset number filtering
        if general_asset_number:
            # Retrieve asset_number_id(s) if general_asset_number is provided
            asset_number_records = db_session.query(AssetNumber).filter(AssetNumber.number == general_asset_number).all()
            if not asset_number_records:
                flash('No Asset Number found matching the provided input.', 'error')
                logger.warning('No AssetNumber records found matching the provided input.')
                return render_template('bill_of_materials.html')
            # Extract asset_number_ids
            asset_number_ids = [record.id for record in asset_number_records]
            logger.info(f"Retrieved AssetNumber IDs: {asset_number_ids}")

            # Apply asset number filter
            query = query.filter(Position.asset_number_id.in_(asset_number_ids))
        else:
            # Exclude positions where asset_number_id is NULL or empty
            query = query.filter(Position.asset_number_id.isnot(None))
            logger.info("Excluding positions with NULL asset_number_id.")

        # Apply location filtering
        if general_location:
            # Retrieve location_id(s) if general_location is provided
            location_records = db_session.query(Location).filter(Location.name == general_location).all()
            if not location_records:
                flash('No Location found matching the provided input.', 'error')
                logger.warning('No Location records found matching the provided input.')
                return render_template('bill_of_materials.html')
            # Extract location_ids
            location_ids = [record.id for record in location_records]
            logger.info(f"Retrieved Location IDs: {location_ids}")

            # Apply location filter
            query = query.filter(Position.location_id.in_(location_ids))
        else:
            # Exclude positions where location_id is NULL or empty
            query = query.filter(Position.location_id.isnot(None))
            logger.info("Excluding positions with NULL location_id.")

        # Execute the query to get all matching positions
        positions = query.all()

        if not positions:
            flash('No results found for the given Asset Number or Location.', 'error')
            logger.warning('No matching positions found for the general search.')
            return render_template('bill_of_materials.html')

        logger.info(f'Found {len(positions)} matching positions for the general search.')

        # Prepare results list
        results = []

        # Process each position and associated parts/images within the active session
        for position in positions:
            # Query for parts and images while the session is active
            parts_images = db_session.query(PartsPositionImageAssociation).filter_by(position_id=position.id).all()

            for association in parts_images:
                # Access the `part_id` and `image_id` while the session is active
                part = db_session.query(Part).filter_by(id=association.part_id).first()
                if part:
                    store_bom_results(db_session, part_id=part.id, position_id=position.id,
                                      image_id=association.image_id, description="General search result")
                    # Collect result data while session is active
                    results.append({'part_id': part.id, 'image_id': association.image_id})
                else:
                    logger.error(f"No part found with ID {association.part_id}")

        # Commit the results and finalize the session
        db_session.commit()

        # Store results in session
        flask_session['results'] = json.dumps(results)
        flask_session['general_asset_number'] = general_asset_number
        flask_session['general_location'] = general_location

        logger.info('Serialized general search results and stored them in session.')

        # Start with the first item (index 0)
        return redirect(url_for('create_bill_of_material_bp.view_bill_of_material', index=0))

    except Exception as e:
        logger.error(f'An error occurred during general search: {str(e)}')
        flash(f'An error occurred: {str(e)}', 'error')
        db_session.rollback()  # Roll back in case of error
        return render_template('bill_of_materials.html')

    finally:
        db_session.close()  # Ensure session is closed after processing

def store_bom_results(session, part_id, position_id, image_id=None, description=None):
    try:
        result = BOMResult(part_id=part_id, position_id=position_id, image_id=image_id, description=description)
        session.add(result)
        logging.info(f"Stored BOM result: Part ID: {part_id}, Position ID: {position_id}, Image ID: {image_id}")
    except Exception as e:
        logging.error(f"Failed to store BOM result: {e}")
        raise  # Propagate exception to caller

def clear_bom_results():
    """
    Clears the temporary BOM results stored in the BOMResult table.
    """
    session = db_config.get_main_session()
    try:
        session.query(BOMResult).delete()
        session.commit()
        logging.info("Cleared BOM results successfully.")
    except Exception as e:
        logging.error(f"Failed to clear BOM results: {e}")
        session.rollback()
    finally:
        session.close()
