from flask import Blueprint, jsonify, request, send_file, flash
from sqlalchemy.orm import sessionmaker, scoped_session, joinedload
from sqlalchemy import create_engine
import os
from modules.configuration.config import DATABASE_URL, DATABASE_DIR
from modules.emtacdb.emtacdb_fts import Image, Position, \
    ImagePositionAssociation  # Ensure this is the correct import path for your Image model
import logging

# Create a SQLAlchemy engine using the DATABASE_URL from your config
engine = create_engine(DATABASE_URL)

# Create a session factory
Session = scoped_session(sessionmaker(bind=engine))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

search_images_bp = Blueprint('search_images_bp', __name__)


# Function to serve an image from the database based on its ID
def serve_image(session, image_id):
    logger.info(f"Attempting to serve image with ID: {image_id}")
    try:
        image = session.query(Image).filter_by(id=image_id).first()
        if image:
            logger.debug(f"Image found: {image.title}, File path: {image.file_path}")
            file_path = os.path.join(DATABASE_DIR, image.file_path)
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


@search_images_bp.route('/serve_image/<int:image_id>')
def serve_image_route(image_id):
    logger.debug(f"Request to serve image with ID: {image_id}")
    with Session() as session:
        try:
            return serve_image(session, image_id)
        except Exception as e:
            logger.error(f"Error serving image {image_id}: {e}")
            flash(f"Error serving image {image_id}", "error")
            return "Image not found", 404


# Define the search_images route within the blueprint
@search_images_bp.route('/', methods=['GET'])
def search_images():
    with Session() as session:
        # Retrieve search parameters
        description = request.args.get('description', '')
        title = request.args.get('title', '')
        area = request.args.get('searchimage_area', '')
        equipment_group = request.args.get('searchimage_equipment_group', '')
        model = request.args.get('searchimage_model', '')
        asset_number = request.args.get('searchimage_asset_number', '')
        location = request.args.get('searchimage_location', '')

        logger.debug(
            f"Search parameters - Description: {description}, Title: {title}, Area: {area}, Equipment Group: {equipment_group}, Model: {model}, Asset Number: {asset_number}, Location: {location}")

        # Perform search using the updated function
        result = search_images_db(session, description=description, title=title, area=area,
                                  equipment_group=equipment_group, model=model, asset_number=asset_number,
                                  location=location)

        if 'images' in result:
            return jsonify(result)
        else:
            return jsonify({'error': 'No images found'}), 404


def search_images_db(session, description='', title='', area='', equipment_group='', model='', asset_number='', location=''):
    logger.info("Starting search_images_db")
    logger.debug(
        f"Search parameters - Description: {description}, Title: {title}, Area: {area}, Equipment Group: {equipment_group}, Model: {model}, Asset Number: {asset_number}, Location: {location}")

    try:
        # Create the base query with explicit joins
        query = (
            session.query(Image)
            .join(ImagePositionAssociation, Image.id == ImagePositionAssociation.image_id)
            .join(Position, ImagePositionAssociation.position_id == Position.id)
            .options(joinedload(Image.image_position_association).joinedload(ImagePositionAssociation.position))
        )

        # Apply filters based on provided parameters
        if description:
            query = query.filter(Image.description.ilike(f'%{description}%'))
        if title:
            query = query.filter(Image.title.ilike(f'%{title}%'))
        if area:
            query = query.filter(Position.area_id == int(area))
        if equipment_group:
            # Ensure that equipment_group is filtered only if area is also selected
            query = query.filter(Position.equipment_group_id == int(equipment_group))
            if area:
                query = query.filter(Position.area_id == int(area))
        if model:
            # Ensure that model is filtered only if equipment_group is also selected
            query = query.filter(Position.model_id == int(model))
            if equipment_group:
                query = query.filter(Position.equipment_group_id == int(equipment_group))
            if area:
                query = query.filter(Position.area_id == int(area))
        if asset_number:
            # Ensure that asset_number is filtered only if model is also selected
            query = query.filter(Position.asset_number_id == int(asset_number))
            if model:
                query = query.filter(Position.model_id == int(model))
            if equipment_group:
                query = query.filter(Position.equipment_group_id == int(equipment_group))
            if area:
                query = query.filter(Position.area_id == int(area))
        if location:
            # Ensure that location is filtered only if model is also selected
            query = query.filter(Position.location_id == int(location))
            if model:
                query = query.filter(Position.model_id == int(model))
            if equipment_group:
                query = query.filter(Position.equipment_group_id == int(equipment_group))
            if area:
                query = query.filter(Position.area_id == int(area))

        results = query.all()

        # Convert the results to a list of dictionaries for JSON response
        images = [
            {
                'id': img.id,
                'title': img.title,
                'description': img.description,
            }
            for img in results
        ]

        logger.info(f"Found {len(images)} images matching the criteria")
        return {"images": images}
    except Exception as e:
        session.rollback()
        logger.error(f"An error occurred while searching images: {e}")
        return {"error": str(e)}
