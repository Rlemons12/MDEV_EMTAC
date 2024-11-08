# pst_troubleshooting_guide_edit_update_bp.py
import os
import traceback
from flask import Blueprint, request, redirect, url_for, jsonify, flash, render_template
import logging
from config_env import DatabaseConfig
from emtacdb_fts import (CompletedDocumentPositionAssociation, ImagePositionAssociation, DrawingPositionAssociation,
                         Drawing, Problem, Task,
                         ImageProblemAssociation, ImageTaskAssociation, CompleteDocumentProblemAssociation,
                         Part, PartProblemAssociation, DrawingProblemAssociation,
                         PartsPositionImageAssociation, ProblemPositionAssociation,Area,EquipmentGroup,AssetNumber,
                         Model, Location, SiteLocation)
from sqlalchemy import or_
from sqlalchemy.exc import SQLAlchemyError
from logging.handlers import RotatingFileHandler

# Setup logging
def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    log_directory = 'logs'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(
        os.path.join(log_directory, 'app.log'), maxBytes=5 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

logger = setup_logger()

# Initialize DatabaseConfig
db_config = DatabaseConfig()

# Blueprint initialization
pst_troubleshooting_guide_edit_update_bp = Blueprint('pst_troubleshooting_guide_edit_update_bp', __name__)

# Helper function to update associations
def update_associations(session, model, filter_field, target_id, item_ids, assoc_field, assoc_data_func, assoc_name):
    logger.debug(f"Starting update_associations for {assoc_name} with target_id={target_id} and item_ids={item_ids}")

    if not target_id or not item_ids:
        logger.warning(f"Missing target_id or item_ids for {assoc_name} update. Skipping.")
        return

    try:
        current_assocs = session.query(model).filter(getattr(model, filter_field) == target_id).all()
        current_ids = {getattr(assoc, assoc_field) for assoc in current_assocs}

        to_delete = [assoc for assoc in current_assocs if getattr(assoc, assoc_field) not in item_ids]
        for assoc in to_delete:
            session.delete(assoc)
            logger.debug(f"Deleted {assoc_name} association with {assoc_field}={getattr(assoc, assoc_field)}")

        new_assocs = [assoc_data_func(target_id, item_id) for item_id in item_ids if item_id not in current_ids]
        if new_assocs:
            session.bulk_save_objects(new_assocs)
            logger.info(f"Added {len(new_assocs)} new {assoc_name} associations for {filter_field}={target_id}")
        else:
            logger.info(f"No new {assoc_name} associations to add for {filter_field}={target_id}")

    except Exception as e:
        logger.error(f"Error updating {assoc_name} associations: {traceback.format_exc()}")

# Route for editing/updating a Task
@pst_troubleshooting_guide_edit_update_bp.route('/troubleshooting_guide/edit_update_task', methods=['POST'])
def edit_update_task():
    logger.info("Accessed /troubleshooting_guide/edit_update_task route via POST method")

    form_data = request.form
    logger.debug(f"Form data received: {form_data}")

    # Retrieve task-specific data from the form
    task_id = form_data.get('task_id')
    task_name = form_data.get('task_name')
    task_description = form_data.get('task_description')
    logger.debug(f"Parsed Task Data - ID: {task_id}, Name: {task_name}, Description: {task_description}")

    # Collect associated IDs from the form related to the task
    selected_task_image_ids = form_data.getlist('edit_task_image[]')
    selected_document_ids = form_data.getlist('edit_document[]')
    selected_part_ids = form_data.getlist('edit_part[]')
    selected_drawing_ids = form_data.getlist('edit_drawing[]')
    logger.debug(f"Parsed Associated IDs - Image IDs: {selected_task_image_ids}, Document IDs: {selected_document_ids}, Part IDs: {selected_part_ids}, Drawing IDs: {selected_drawing_ids}")

    session = db_config.get_main_session()

    try:
        # Update Task details
        task = session.query(Task).filter_by(id=task_id).first()
        if task:
            task.name = task_name
            task.description = task_description
            logger.info(f"Updated Task ID={task_id} with new name and description")
        else:
            flash(f"Task with ID {task_id} not found", 'danger')
            return render_template('troubleshooting_guide.html'), 404

        # Update various task-specific associations
        update_associations(session, ImageTaskAssociation, 'task_id', task_id, selected_task_image_ids,
                            'image_id', lambda tid, iid: ImageTaskAssociation(task_id=tid, image_id=iid),
                            'ImageTaskAssociation')
        update_associations(session, CompleteDocumentTaskAssociation, 'task_id', task_id, selected_document_ids,
                            'complete_document_id', lambda tid, did: CompleteDocumentTaskAssociation(task_id=tid, complete_document_id=did),
                            'CompleteDocumentTaskAssociation')
        update_associations(session, PartTaskAssociation, 'task_id', task_id, selected_part_ids,
                            'part_id', lambda tid, partid: PartTaskAssociation(task_id=tid, part_id=partid),
                            'PartTaskAssociation')
        update_associations(session, DrawingTaskAssociation, 'task_id', task_id, selected_drawing_ids,
                            'drawing_id', lambda tid, drawingid: DrawingTaskAssociation(task_id=tid, drawing_id=drawingid),
                            'DrawingTaskAssociation')

        # Commit transaction
        session.commit()
        logger.info("Successfully committed updates to the database for Task")
        flash("Task updated successfully", 'success')
    except Exception as e:
        logger.error(f"Error updating Task: {traceback.format_exc()}")
        session.rollback()
        flash("An error occurred while updating", 'danger')
    finally:
        session.close()
        logger.debug("Database session closed")

    return redirect(url_for('pst_troubleshooting_guide_edit_update_bp.troubleshooting_guide'))

# Search route for Parts
@pst_troubleshooting_guide_edit_update_bp.route('/search_parts')
def search_parts():
    search_term = request.args.get('q', '')
    logger.info(f"Searching for parts with term '{search_term}'")

    session = db_config.get_main_session()
    parts = session.query(Part).filter(
        or_(
            Part.part_number.ilike(f'%{search_term}%'),
            Part.name.ilike(f'%{search_term}%'),
            Part.oem_mfg.ilike(f'%{search_term}%'),
            Part.model.ilike(f'%{search_term}%')
        )
    ).limit(10).all()

    results = [{'id': part.id, 'name': f"{part.part_number} - {part.name}"} for part in parts]
    logger.debug(f"Search results: {results}")
    return jsonify(results)

# Search route for Drawings
@pst_troubleshooting_guide_edit_update_bp.route('/search_drawings')
def search_drawings():
    search_term = request.args.get('q', '')
    logger.info(f"Searching for drawings with term '{search_term}'")

    session = db_config.get_main_session()
    drawings = session.query(Drawing).filter(
        or_(
            Drawing.drw_number.ilike(f'%{search_term}%'),
            Drawing.drw_name.ilike(f'%{search_term}%'),
            Drawing.drw_revision.ilike(f'%{search_term}%'),
            Drawing.drw_equipment_name.ilike(f'%{search_term}%')
        )
    ).limit(10).all()

    results = [{'id': drawing.id, 'text': f"{drawing.drw_number} - {drawing.drw_name}"} for drawing in drawings]
    logger.debug(f"Search results: {results}")
    return jsonify(results)


@pst_troubleshooting_guide_edit_update_bp.route('/search_images')
def search_images():
    # Get the search query parameter from the request
    search_term = request.args.get('q', '').strip()
    logger.info(f"Searching images with term: '{search_term}'")

    # Start a database session
    session = db_config.get_main_session()

    try:
        # Query to find images by title or description
        images = session.query(Image).filter(
            or_(
                Image.title.ilike(f'%{search_term}%'),
                Image.description.ilike(f'%{search_term}%')
            )
        ).limit(10).all()

        # Format results for JSON response
        results = [{'id': image.id, 'title': image.title, 'description': image.description} for image in images]
        logger.info(f"Found {len(results)} images matching search term '{search_term}'")

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error occurred while searching images: {traceback.format_exc()}")
        return jsonify({'error': 'An error occurred while searching for images'}), 500

    finally:
        session.close()

@pst_troubleshooting_guide_edit_update_bp.route('/get_areas', methods=['GET'])
def get_areas():
    session = db_config.get_main_session()
    try:
        areas = session.query(Area).all()
        areas_data = [{'id': area.id, 'name': area.name} for area in areas]
        logger.info(f"Fetched {len(areas_data)} areas.")
        return jsonify({"areas": areas_data}), 200
    except SQLAlchemyError as e:
        logger.error(f"Database error while fetching areas: {e}")
        return jsonify({"error": "Failed to fetch areas"}), 500
    finally:
        session.close()


@pst_troubleshooting_guide_edit_update_bp.route('/get_equipment_groups', methods=['GET'])
def get_equipment_groups():
    session = db_config.get_main_session()
    area_id = request.args.get('area_id')
    equipment_groups = session.query(EquipmentGroup).filter_by(area_id=area_id).all()
    data = [{'id': eg.id, 'name': eg.name} for eg in equipment_groups]
    return jsonify(data)

@pst_troubleshooting_guide_edit_update_bp.route('/get_models', methods=['GET'])
def get_models():
    session = db_config.get_main_session()
    equipment_group_id = request.args.get('equipment_group_id')
    models = session.query(Model).filter_by(equipment_group_id=equipment_group_id).all()
    data = [{'id': model.id, 'name': model.name} for model in models]
    return jsonify(data)

@pst_troubleshooting_guide_edit_update_bp.route('/get_asset_numbers', methods=['GET'])
def get_asset_numbers():
    session = db_config.get_main_session()
    model_id = request.args.get('model_id')
    asset_numbers = session.query(AssetNumber).filter_by(model_id=model_id).all()
    data = [{'id': asset.id, 'number': asset.number} for asset in asset_numbers]
    return jsonify(data)

@pst_troubleshooting_guide_edit_update_bp.route('/get_locations', methods=['GET'])
def get_locations():
    session = db_config.get_main_session()
    model_id = request.args.get('model_id')
    locations = session.query(Location).filter_by(model_id=model_id).all()
    data = [{'id': location.id, 'name': location.name} for location in locations]
    return jsonify(data)

@pst_troubleshooting_guide_edit_update_bp.route('/get_site_locations', methods=['GET'])
def get_site_locations():
    session = db_config.get_main_session()
    model_id = request.args.get('model_id')
    asset_number_id = request.args.get('asset_number_id')
    location_id = request.args.get('location_id')
    area_id = request.args.get('area_id')  # New parameter for area
    equipment_group_id = request.args.get('equipment_group_id')  # New parameter for equipment group

    logger.info(f"Received request to /get_site_locations with model_id: {model_id}, "
                f"asset_number_id: {asset_number_id}, location_id: {location_id}, "
                f"area_id: {area_id}, equipment_group_id: {equipment_group_id}")

    try:
        positions = session.query(Position).filter_by(
            model_id=model_id,
            asset_number_id=asset_number_id,
            location_id=location_id,
            area_id=area_id,
            equipment_group_id=equipment_group_id
        ).all()

        logger.info(f"Found {len(positions)} positions matching the filters.")

        site_locations = [
            {'id': pos.site_location.id, 'title': pos.site_location.title, 'room_number': pos.site_location.room_number}
            for pos in positions if pos.site_location
        ]

        logger.info(f"Extracted {len(site_locations)} site locations.")

        site_locations.append({'id': 'new', 'title': 'New Site Location', 'room_number': ''})

        return jsonify(site_locations)
    except Exception as e:
        logger.error(f"Error fetching site locations: {e}")
        return jsonify({"error": "An error occurred while fetching site locations"}), 500
