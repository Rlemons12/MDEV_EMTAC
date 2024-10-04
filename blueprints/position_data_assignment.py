from flask import Blueprint, request, redirect, flash, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
import os
import logging
from sqlalchemy.orm import joinedload
from emtacdb_fts import (Part,PartsPositionImageAssociation,DrawingPositionAssociation,CompletedDocumentPositionAssociation,
                Area, EquipmentGroup, Model, AssetNumber, Location, SiteLocation,CompleteDocument,
                         Image,Position, ImagePositionAssociation)
from config_env import DatabaseConfig
from config import ALLOWED_EXTENSIONS

# Configure logging
logger = logging.getLogger(__name__)

# Create the Blueprint
position_data_assignment_bp = Blueprint('position_data_assignment_bp', __name__, template_folder='templates')

# Initialize DatabaseConfig
db_config = DatabaseConfig()

# Utility function for allowed file types
def allowed_file(filename, allowed_extensions=ALLOWED_EXTENSIONS):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


# Route for displaying and submitting position data assignment form
@position_data_assignment_bp.route('/position_data_assignment', methods=['GET', 'POST'])
def position_data_assignment():
    db_session = db_config.get_main_session()

    position_id = request.args.get('position_id')  # Get position ID from query parameters

    if request.method == 'POST':
        try:
            # Handle form submission
            area_id = request.form.get('area_id')
            equipment_group_id = request.form.get('equipment_group_id')
            model_id = request.form.get('model_id')
            asset_number_id = request.form.get('asset_number_id') or None
            asset_number_input = request.form.get('asset_number_input') or None
            location_id = request.form.get('location_id') or None
            location_input = request.form.get('location_input') or None
            site_location_id = request.form.get('site_location_id')
            position_id = request.form.get('position_id')
            part_numbers = request.form.getlist('part_numbers[]')

            # Handle manual input for Asset Number and Location
            if not asset_number_id and asset_number_input:
                new_asset = AssetNumber(number=asset_number_input, model_id=model_id)
                db_session.add(new_asset)
                db_session.commit()
                asset_number_id = new_asset.id

            if not location_id and location_input:
                new_location = Location(name=location_input, model_id=model_id)
                db_session.add(new_location)
                db_session.commit()
                location_id = new_location.id

            # Handle file uploads
            images = request.files.getlist('images[]')
            drawings = request.files.getlist('drawings[]')

            saved_image_paths = []
            for image in images:
                if image and allowed_file(image.filename):
                    filename = secure_filename(image.filename)
                    image_path = os.path.join('static/uploads/images/', filename)
                    image.save(image_path)
                    saved_image_paths.append(image_path)

            saved_drawing_paths = []
            for drawing in drawings:
                if drawing and allowed_file(drawing.filename, ['pdf', 'jpg', 'jpeg', 'png']):
                    filename = secure_filename(drawing.filename)
                    drawing_path = os.path.join('static/uploads/drawings/', filename)
                    drawing.save(drawing_path)
                    saved_drawing_paths.append(drawing_path)

            # Check if we're updating an existing position or creating a new one
            if position_id:
                # Updating an existing position
                position = db_session.query(Position).filter_by(id=position_id).first()
                if not position:
                    flash('Position not found.', 'error')
                    return redirect(url_for('position_data_assignment_bp.position_data_assignment'))

                # Update position fields
                position.area_id = area_id
                position.equipment_group_id = equipment_group_id
                position.model_id = model_id
                position.asset_number_id = asset_number_id
                position.location_id = location_id
                position.site_location_id = site_location_id
                position.parts = part_numbers  # Assuming position has a parts relationship or column

                db_session.commit()
                flash('Position Data Updated Successfully!', 'success')

            else:
                # Creating a new position
                new_pda = PositionDataAssignment(
                    area_id=area_id,
                    equipment_group_id=equipment_group_id,
                    model_id=model_id,
                    asset_number_id=asset_number_id,
                    location_id=location_id,
                    site_location_id=site_location_id,
                    position_id=position_id,
                    parts=part_numbers,
                    images=saved_image_paths,
                    drawings=saved_drawing_paths
                )
                db_session.add(new_pda)
                db_session.commit()

                flash('Position Data Assigned Successfully!', 'success')

            return redirect(url_for('position_data_assignment_bp.position_data_assignment'))

        except Exception as e:
            db_session.rollback()
            logger.error(f"Error during form submission: {e}")
            flash('An error occurred while processing your request.', 'error')
            return redirect(url_for('position_data_assignment_bp.position_data_assignment'))

        finally:
            db_session.close()

    # If it's a GET request, load the form with initial data
    else:
        try:
            areas = db_session.query(Area).all()
            equipment_groups = db_session.query(EquipmentGroup).all()
            models = db_session.query(Model).all()
            asset_numbers = db_session.query(AssetNumber).all()
            locations = db_session.query(Location).all()
            site_locations = db_session.query(SiteLocation).all()

            position = None

            # Load the position if position_id is provided (for updating)
            if position_id:
                position = db_session.query(Position).filter_by(id=position_id).first()

            return render_template(
                'position_data_assignment.html',
                areas=areas,
                equipment_groups=equipment_groups,
                models=models,
                asset_numbers=asset_numbers,
                locations=locations,
                site_locations=site_locations,
                position=position
            )

        except Exception as e:
            logger.error(f"Error fetching areas or position: {e}")
            flash('Error loading the form', 'error')
            return redirect(url_for('position_data_assignment_bp.position_data_assignment'))

        finally:
            db_session.close()

# Additional routes for AJAX requests
@position_data_assignment_bp.route('/get_equipment_groups')
def get_equipment_groups():
    area_id = request.args.get('area_id')
    db_session = db_config.get_main_session()
    try:
        equipment_groups = db_session.query(EquipmentGroup).filter_by(area_id=area_id).all()
        data = [{'id': eg.id, 'name': eg.name} for eg in equipment_groups]
        return jsonify(data)  # Ensure data is in the correct format
    finally:
        db_session.close()

@position_data_assignment_bp.route('/get_models')
def get_models():
    equipment_group_id = request.args.get('equipment_group_id')
    db_session = db_config.get_main_session()
    try:
        models = db_session.query(Model).filter_by(equipment_group_id=equipment_group_id).all()
        data = [{'id': m.id, 'name': m.name} for m in models]
        return jsonify(data)
    finally:
        db_session.close()

@position_data_assignment_bp.route('/get_asset_numbers')
def get_asset_numbers():
    model_id = request.args.get('model_id')
    db_session = db_config.get_main_session()
    try:
        asset_numbers = db_session.query(AssetNumber).filter_by(model_id=model_id).all()
        data = [{'id': an.id, 'number': an.number} for an in asset_numbers]
        return jsonify(data)
    finally:
        db_session.close()

@position_data_assignment_bp.route('/get_locations')
def get_locations():
    model_id = request.args.get('model_id')
    db_session = db_config.get_main_session()
    try:
        locations = db_session.query(Location).filter_by(model_id=model_id).all()
        data = [{'id': loc.id, 'name': loc.name} for loc in locations]
        return jsonify(data)
    finally:
        db_session.close()

@position_data_assignment_bp.route('/get_site_locations')
def get_site_locations():
    model_id = request.args.get('model_id')
    asset_number_id = request.args.get('asset_number_id')
    location_id = request.args.get('location_id')
    db_session = db_config.get_main_session()
    try:
        positions = db_session.query(Position).filter_by(
            model_id=model_id,
            asset_number_id=asset_number_id,
            location_id=location_id
        ).options(joinedload(Position.site_location)).all()
        # Extract unique site locations
        site_location_set = {
            (p.site_location.id, p.site_location.title, p.site_location.room_number)
            for p in positions if p.site_location
        }
        data = [{'id': loc_id, 'title': title, 'room_number': room_number} for loc_id, title, room_number in site_location_set]
        return jsonify(data)
    finally:
        db_session.close()

@position_data_assignment_bp.route('/get_positions', methods=['GET'])
def get_positions():
    # Get the filter parameters from the request
    site_location_id = request.args.get('site_location_id')
    area_id = request.args.get('area_id')
    equipment_group_id = request.args.get('equipment_group_id')
    model_id = request.args.get('model_id')
    asset_number_id = request.args.get('asset_number_id')
    location_id = request.args.get('location_id')

    logger.info(f"Received GET request with filter parameters: "
                f"site_location_id={site_location_id}, area_id={area_id}, equipment_group_id={equipment_group_id}, "
                f"model_id={model_id}, asset_number_id={asset_number_id}, location_id={location_id}")

    db_session = db_config.get_main_session()
    try:
        # Step 1: Search for Positions with filters, only applying filters if they exist
        query = db_session.query(Position)
        logger.info("Initialized query for Position")

        if site_location_id:
            query = query.filter(Position.site_location_id == site_location_id)
            logger.info(f"Applied filter: site_location_id={site_location_id}")
        if area_id:
            query = query.filter(Position.area_id == area_id)
            logger.info(f"Applied filter: area_id={area_id}")
        if equipment_group_id:
            query = query.filter(Position.equipment_group_id == equipment_group_id)
            logger.info(f"Applied filter: equipment_group_id={equipment_group_id}")
        if model_id:
            query = query.filter(Position.model_id == model_id)
            logger.info(f"Applied filter: model_id={model_id}")
        if asset_number_id:
            query = query.filter(Position.asset_number_id == asset_number_id)
            logger.info(f"Applied filter: asset_number_id={asset_number_id}")
        if location_id:
            query = query.filter(Position.location_id == location_id)
            logger.info(f"Applied filter: location_id={location_id}")

        positions = query.all()
        logger.info(f"Query executed. Number of positions found: {len(positions)}")

        if not positions:
            logger.warning("No positions found for the provided filters")
            return jsonify({"message": "No positions found"}), 404

        # Step 2: Prepare a response with the associated data
        result_data = []

        for position in positions:
            position_data = {
                'position_id': position.id,
                'area': {
                    'id': position.area.id if position.area else None,
                    'name': position.area.name if position.area else None,
                    'description': position.area.description if position.area else None  # Add description for area
                },
                'equipment_group': {
                    'id': position.equipment_group.id if position.equipment_group else None,
                    'name': position.equipment_group.name if position.equipment_group else None
                },
                'model': {
                    'id': position.model.id if position.model else None,
                    'name': position.model.name if position.model else None,
                    'description': position.model.description if position.model else None  # Add description for model
                },
                'asset_number': {
                    'id': position.asset_number.id if position.asset_number else None,
                    'number': position.asset_number.number if position.asset_number else None,
                    'description': position.asset_number.description if position.asset_number else None  # Add description for asset number
                },
                'location': {
                    'id': position.location.id if position.location else None,
                    'name': position.location.name if position.location else None
                },
                'parts': [],
                'documents': [],
                'drawings': [],
                'images': []
            }

            logger.info(f"Processing position: {position.id}")

            # Fetch parts
            parts_position_image = db_session.query(PartsPositionImageAssociation).filter_by(
                position_id=position.id).all()
            part_ids = [ppi.part_id for ppi in parts_position_image]
            logger.info(f"Found {len(parts_position_image)} part associations for position {position.id}")

            if part_ids:
                parts = db_session.query(Part).filter(Part.id.in_(part_ids)).all()
                logger.info(f"Found {len(parts)} parts for position {position.id}")
                for part in parts:
                    position_data['parts'].append({
                        'part_id': part.id,
                        'part_number': part.part_number,
                        'name': part.name
                    })

            # Fetch complete documents
            complete_documents = db_session.query(CompletedDocumentPositionAssociation).filter_by(
                position_id=position.id).all()
            logger.info(f"Found {len(complete_documents)} complete document associations for position {position.id}")
            for cdpa in complete_documents:
                complete_doc = db_session.query(CompleteDocument).filter_by(id=cdpa.complete_document_id).first()
                if complete_doc:
                    position_data['documents'].append({
                        'complete_document_id': complete_doc.id,
                        'title': complete_doc.title,
                        'rev': complete_doc.rev
                    })

            # Fetch drawings
            drawing_positions = db_session.query(DrawingPositionAssociation).filter_by(position_id=position.id).all()
            logger.info(f"Found {len(drawing_positions)} drawing associations for position {position.id}")
            for dp in drawing_positions:
                drawing = db_session.query(Drawing).filter_by(id=dp.drawing_id).first()
                if drawing:
                    position_data['drawings'].append({
                        'drawing_id': drawing.id,
                        'drawing_name': drawing.drw_name,
                        'drawing_revision': drawing.drw_revision
                    })

            # Fetch images
            image_positions = db_session.query(ImagePositionAssociation).filter_by(position_id=position.id).all()
            logger.info(f"Found {len(image_positions)} image associations for position {position.id}")
            for ip in image_positions:
                image = db_session.query(Image).filter_by(id=ip.image_id).first()
                if image:
                    position_data['images'].append({
                        'image_id': image.id,
                        'image_title': image.title,
                        'description': image.description
                    })

            # Add position data to the result list
            result_data.append(position_data)

        logger.info(f"Returning result data for {len(result_data)} positions")

        # Step 8: Return the response as JSON
        return jsonify(result_data)

    except Exception as e:
        db_session.rollback()
        logger.error(f"Error during position search: {str(e)}", exc_info=True)
        return jsonify({"message": "Error occurred during search", "error": str(e)}), 500

    finally:
        db_session.close()
        logger.info("Database session closed")

@position_data_assignment_bp.route('/add_site_location', methods=['GET', 'POST'])
def add_site_location():
    db_session = db_config.get_main_session()

    if request.method == 'POST':
        try:
            # Get form data
            title = request.form.get('title')
            room_number = request.form.get('room_number')

            # Validate form data
            if not title or not room_number:
                flash("All fields are required.", "error")
                return redirect(url_for('position_data_assignment_bp.add_site_location'))

            # Create a new SiteLocation entry
            new_site_location = SiteLocation(title=title, room_number=room_number)
            db_session.add(new_site_location)
            db_session.commit()

            flash('Site Location added successfully!', 'success')
            return redirect(url_for('position_data_assignment_bp.position_data_assignment'))

        except Exception as e:
            db_session.rollback()
            flash('An error occurred while adding the site location.', 'error')
            return redirect(url_for('position_data_assignment_bp.add_site_location'))
        finally:
            db_session.close()

    # If GET request, render the form
    return render_template('add_site_location.html')

@position_data_assignment_bp.route('/update_position', methods=['POST'])
def update_position():
    db_session = db_config.get_main_session()
    try:
        # Retrieve form data from the update form
        position_id = request.form.get('position_id')
        area_id = request.form.get('area_id')
        area_name = request.form.get('area_name')
        area_description = request.form.get('area_description')

        equipment_group_id = request.form.get('equipment_group_id')
        equipment_group_name = request.form.get('equipment_group_name')
        equipment_group_description = request.form.get('equipment_group_description')

        model_id = request.form.get('model_id')
        model_name = request.form.get('model_name')
        model_description = request.form.get('model_description')

        asset_number_id = request.form.get('asset_number_id')
        asset_number_input = request.form.get('asset_number')
        asset_number_description = request.form.get('asset_number_description')

        location_id = request.form.get('location_id')
        location_name = request.form.get('location_name')
        location_description = request.form.get('location_description')

        site_id = request.form.get('site_id')
        site_title = request.form.get('site_title')
        site_room_number = request.form.get('site_room_number')

        # Log form data retrieval
        logger.info(f"Updating position with ID {position_id}")
        logger.debug(f"Form data - Area: {area_id}, Area Name: {area_name}, Description: {area_description}")
        logger.debug(f"Equipment Group: {equipment_group_id}, Name: {equipment_group_name}, Description: {equipment_group_description}")
        logger.debug(f"Model: {model_id}, Name: {model_name}, Description: {model_description}")
        logger.debug(f"Asset Number: {asset_number_id}, Number: {asset_number_input}, Description: {asset_number_description}")
        logger.debug(f"Location: {location_id}, Name: {location_name}")
        logger.debug(f"Location: {location_description}")
        logger.debug(f"Site: {site_id}, Name: {site_title}")
        logger.debug(f"Site: {site_room_number}")

        # Fetch the existing Position object from the database
        position = db_session.query(Position).filter_by(id=position_id).first()

        if not position:
            logger.error(f"Position with ID {position_id} not found.")
            flash('Position not found.', 'error')
            return redirect(url_for('position_data_assignment_bp.position_data_assignment'))

        logger.info(f"Position with ID {position_id} found. Proceeding with updates.")

        # Update the current data for the related entities
        # Update Area name and description
        if area_id:
            db_session.query(Area).filter_by(id=area_id).update({
                "name": area_name,
                "description": area_description
            })
            logger.info(f"Updated Area ID {area_id} with name '{area_name}' and description.")

        # Update EquipmentGroup name and description
        if equipment_group_id:
            db_session.query(EquipmentGroup).filter_by(id=equipment_group_id).update({
                "name": equipment_group_name,
                "description": equipment_group_description
            })
            logger.info(f"Updated Equipment Group ID {equipment_group_id} with name '{equipment_group_name}' and description.")

        # Update Model name and description
        if model_id:
            db_session.query(Model).filter_by(id=model_id).update({
                "name": model_name,
                "description": model_description
            })
            logger.info(f"Updated Model ID {model_id} with name '{model_name}' and description.")

        # Update AssetNumber
        if asset_number_id and asset_number_input:
            db_session.query(AssetNumber).filter_by(id=asset_number_id).update({
                "number": asset_number_input,
                "description": asset_number_description
            })
            logger.info(f"Updated Asset Number ID {asset_number_id} with number '{asset_number_input}' and description.")

        # Update Location name and description
        if location_id and location_name:
            db_session.query(Location).filter_by(id=location_id).update({
                "name": location_name,
                "description": location_description  # Make sure to use location_description from form data
            })
            logger.info(
                f"Updated Location ID {location_id} with name '{location_name}' and description '{location_description}'.")

        # Update Site Location Name and Description
        if site_id and site_title:
            db_session.query(SiteLocation).filter_by(id=site_id).update({
                "title": site_title,
                "room_number": site_room_number
            })
            logger.info(
                f'Updated Site Location ID {site_id} with title "{site_title}" and room_number "{site_room_number}".')

        # Save the updated data
        db_session.commit()
        logger.info(f"Position ID {position_id} and related entities updated successfully.")

        flash('Position data updated successfully!', 'success')
        return redirect(url_for('position_data_assignment_bp.position_data_assignment'))

    except Exception as e:
        db_session.rollback()
        logger.error(f"Error updating position data: {e}", exc_info=True)
        flash('An error occurred while updating position data.', 'error')
        return redirect(url_for('position_data_assignment_bp.position_data_assignment'))
    finally:
        db_session.close()
        logger.info(f"Database session for position ID {position_id} closed.")
