# blueprints/pst_troubleshoot_new_entry_bp.py

from flask import Blueprint, render_template, request, jsonify, flash
from config_env import DatabaseConfig
from emtacdb_fts import (
    Problem, Area, EquipmentGroup, Model, AssetNumber, Location, SiteLocation,
    Part, Solution
)
from sqlalchemy.exc import SQLAlchemyError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Blueprint
pst_troubleshoot_new_entry_bp = Blueprint(
    'pst_troubleshoot_new_entry_bp',
    __name__,
    template_folder='templates',
    static_folder='static',
    url_prefix='/pst_troubleshoot_new_entry'  # You can adjust the prefix as needed
)

# Initialize Database Config
db_config = DatabaseConfig()

@pst_troubleshoot_new_entry_bp.route('/', methods=['GET'])
def new_entry_form():
    """
    Render the New Problem Entry Form.
    """
    session = db_config.get_main_session()
    try:
        areas = session.query(Area).all()
        parts = session.query(Part).all()
        drawings = session.query(Drawing).all()
        return render_template('pst_troubleshoot_new_entry.html', areas=areas, parts=parts, drawings=drawings)
    except SQLAlchemyError as e:
        logger.error(f"Database error: {e}")
        flash('An error occurred while loading the new entry form.', 'danger')
        return render_template('error.html'), 500
    finally:
        session.close()

@pst_troubleshoot_new_entry_bp.route('/create_problem', methods=['POST'])
def create_new_problem():
    """
    Handle the creation of a new problem via AJAX.
    """
    session = db_config.get_main_session()
    try:
        # Extract form data
        problem_name = request.form.get('problem_name')
        problem_description = request.form.get('problem_description')
        area_id = request.form.get('area_id')
        equipment_group_id = request.form.get('equipment_group_id')
        model_id = request.form.get('model_id')
        asset_number_input = request.form.get('asset_number')  # Assuming 'asset_number' is the name
        location_input = request.form.get('location')          # Assuming 'location' is the name
        site_location_id = request.form.get('site_location_id')

        # Validation: Ensure all required fields are present
        required_fields = [problem_name, problem_description, area_id, equipment_group_id, model_id, asset_number_input, location_input, site_location_id]
        if not all(required_fields):
            return jsonify({'success': False, 'message': 'All fields are required.'}), 400

        # Handle Asset Number: Check if it exists; if not, create it
        asset_number = session.query(AssetNumber).filter_by(number=asset_number_input).first()
        if not asset_number:
            asset_number = AssetNumber(number=asset_number_input, model_id=model_id)
            session.add(asset_number)
            session.commit()
            logger.info(f"Created new Asset Number: {asset_number_input}")

        # Handle Location: Check if it exists; if not, create it
        location = session.query(Location).filter_by(name=location_input).first()
        if not location:
            location = Location(name=location_input, model_id=model_id)
            session.add(location)
            session.commit()
            logger.info(f"Created new Location: {location_input}")

        # Handle Site Location: If 'new', create a new Site Location
        if site_location_id == 'new':
            new_site_location_title = request.form.get('new_siteLocation_title')
            new_site_location_room_number = request.form.get('new_siteLocation_room_number')

            if not all([new_site_location_title, new_site_location_room_number]):
                return jsonify({'success': False, 'message': 'New Site Location title and room number are required.'}), 400

            site_location = SiteLocation(
                title=new_site_location_title,
                room_number=new_site_location_room_number
            )
            session.add(site_location)
            session.commit()
            logger.info(f"Created new Site Location: {new_site_location_title}")
        else:
            site_location = session.query(SiteLocation).filter_by(id=site_location_id).first()
            if not site_location:
                return jsonify({'success': False, 'message': 'Selected Site Location does not exist.'}), 400

        # Create the new Problem
        new_problem = Problem(
            name=problem_name,
            description=problem_description,
            area_id=area_id,
            equipment_group_id=equipment_group_id,
            model_id=model_id,
            asset_number_id=asset_number.id,
            location_id=location.id,
            site_location_id=site_location.id
        )
        session.add(new_problem)
        session.commit()
        logger.info(f"Created new Problem: {problem_name}")

        # Optionally, handle parts association
        parts_ids = request.form.getlist('parts[]')  # Assuming multiple parts can be selected
        if parts_ids:
            for part_id in parts_ids:
                part = session.query(Part).filter_by(id=part_id).first()
                if part:
                    new_problem.parts.append(part)
            session.commit()
            logger.info(f"Associated Parts: {parts_ids} with Problem ID: {new_problem.id}")

        # Handle new Solution if provided
        new_solution_name = request.form.get('new_solution_name')
        if new_solution_name:
            new_solution = Solution(name=new_solution_name, problem_id=new_problem.id)
            session.add(new_solution)
            session.commit()
            logger.info(f"Added new Solution: {new_solution_name} to Problem ID: {new_problem.id}")

        return jsonify({'success': True, 'message': 'Problem created successfully!', 'problem_id': new_problem.id}), 200

    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error during problem creation: {e}")
        return jsonify({'success': False, 'message': 'An error occurred while creating the problem.'}), 500
    except Exception as e:
        session.rollback()
        logger.error(f"Unexpected error during problem creation: {e}")
        return jsonify({'success': False, 'message': 'An unexpected error occurred.'}), 500
    finally:
        session.close()
