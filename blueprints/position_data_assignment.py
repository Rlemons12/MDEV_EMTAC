from flask import Blueprint, request, redirect, flash, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
import os
import logging
from sqlalchemy.orm import joinedload
from emtacdb_fts import Area, EquipmentGroup, Model, AssetNumber, Location, SiteLocation, Position
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


@position_data_assignment_bp.route('/position_data_assignment', methods=['GET', 'POST'])
def position_data_assignment():
    db_session = db_config.get_main_session()
    if request.method == 'POST':
        try:
            # Retrieve form data
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

            # Process and save files as needed
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

            # Assuming `PositionDataAssignment` is your model to save this data
            new_pda = PositionDataAssignment(
                area_id=area_id,
                equipment_group_id=equipment_group_id,
                model_id=model_id,
                asset_number_id=asset_number_id,
                location_id=location_id,
                site_location_id=site_location_id,
                position_id=position_id,
                parts=part_numbers,  # Adjust based on your model
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
    else:
        try:
            areas = db_session.query(Area).all()
            return render_template('position_data_assignment.html', areas=areas)
        except Exception as e:
            logger.error(f"Error fetching areas: {e}")
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
        return jsonify(data)
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
        ).options(joinedload('site_location')).all()
        # Extract unique site locations
        site_location_set = {
            (p.site_location.id, p.site_location.title, p.site_location.room_number)
            for p in positions if p.site_location
        }
        data = [{'id': loc_id, 'title': title, 'room_number': room_number} for loc_id, title, room_number in site_location_set]
        return jsonify(data)
    finally:
        db_session.close()

@position_data_assignment_bp.route('/get_positions')
def get_positions():
    site_location_id = request.args.get('site_location_id')
    asset_number_id = request.args.get('asset_number_id')
    location_id = request.args.get('location_id')
    db_session = db_config.get_main_session()
    try:
        positions = db_session.query(Position).filter_by(
            site_location_id=site_location_id,
            asset_number_id=asset_number_id,
            location_id=location_id
        ).all()
        data = [{'id': pos.id, 'name': f"Position {pos.id}"} for pos in positions]  # Adjust 'name' as needed
        return jsonify(data)
    finally:
        db_session.close()


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
        equipment_group_id = request.form.get('equipment_group_id')
        model_id = request.form.get('model_id')
        asset_number_id = request.form.get('asset_number_id') or None
        asset_number_input = request.form.get('edit_asset_number') or None
        location_id = request.form.get('location_id') or None
        location_input = request.form.get('edit_location_name') or None
        part_numbers = request.form.getlist('part_numbers[]')

        # Fetch the existing Position object from the database
        position = db_session.query(Position).filter_by(id=position_id).first()

        if not position:
            flash('Position not found.', 'error')
            return redirect(url_for('position_data_assignment_bp.position_data_assignment'))

        # Update the position data
        position.area_id = area_id
        position.equipment_group_id = equipment_group_id
        position.model_id = model_id

        # Handle manual input for Asset Number
        if asset_number_input:
            if not asset_number_id:
                # Create a new AssetNumber entry if none exists
                new_asset = AssetNumber(number=asset_number_input, model_id=model_id)
                db_session.add(new_asset)
                db_session.commit()
                asset_number_id = new_asset.id
            position.asset_number_id = asset_number_id

        # Handle manual input for Location
        if location_input:
            if not location_id:
                # Create a new Location entry if none exists
                new_location = Location(name=location_input, model_id=model_id)
                db_session.add(new_location)
                db_session.commit()
                location_id = new_location.id
            position.location_id = location_id

        # Update parts
        position.parts = part_numbers  # Adjust this based on your model structure

        # Save updated data
        db_session.commit()

        flash('Position data updated successfully!', 'success')
        return redirect(url_for('position_data_assignment_bp.position_data_assignment'))

    except Exception as e:
        db_session.rollback()
        logger.error(f"Error updating position data: {e}")
        flash('An error occurred while updating position data.', 'error')
        return redirect(url_for('position_data_assignment_bp.position_data_assignment'))
    finally:
        db_session.close()
