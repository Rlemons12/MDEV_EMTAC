from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from modules.configuration.config_env import DatabaseConfig  # Import your DatabaseConfig class
from modules.emtacdb.emtacdb_fts import Position, Area, EquipmentGroup, Model, AssetNumber, Location, SiteLocation
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db_config = DatabaseConfig()

# Get a session for the main database
session = db_config.get_main_session()

# Create a blueprint for the route
position_data_assignment_data_add_dependencies_bp = Blueprint('position_data_assignment_data_add_dependencies_bp', __name__)

@position_data_assignment_data_add_dependencies_bp.route('/add_position', methods=['POST'])
def add_position():
    # Extract data from the form
    position_id = request.form.get('position_id')

    area_ids = request.form.getlist('area_id[]')
    site_location_ids = request.form.getlist('site_location_id[]')
    equipment_group_ids = request.form.getlist('equipment_group_id[]')
    model_ids = request.form.getlist('model_id[]')
    asset_number_ids = request.form.getlist('asset_number_id[]')
    location_ids = request.form.getlist('location_id[]')

    # Track whether we create new records
    new_area_ids = []
    new_site_location_ids = []
    new_equipment_group_ids = []
    new_model_ids = []
    new_asset_number_ids = []
    new_location_ids = []

    # Handle multiple areas, including new ones
    for area_id in area_ids:
        if area_id != 'new':
            new_area_ids.append(area_id)
        else:
            new_area_names = request.form.getlist('new_area_name[]')
            new_area_descriptions = request.form.getlist('new_area_description[]')

            for name, description in zip(new_area_names, new_area_descriptions):
                if name:
                    new_area = Area(name=name, description=description)
                    session.add(new_area)
                    session.commit()
                    new_area_ids.append(new_area.id)

    # Handle multiple site locations, including new ones
    for site_location_id in site_location_ids:
        if site_location_id != 'new':
            new_site_location_ids.append(site_location_id)
        else:
            new_site_location_titles = request.form.getlist('new_siteLocation_title[]')
            new_site_location_room_numbers = request.form.getlist('new_siteLocation_room_number[]')

            for title, room_number in zip(new_site_location_titles, new_site_location_room_numbers):
                if title:
                    new_site_location = SiteLocation(title=title, room_number=room_number)
                    session.add(new_site_location)
                    session.commit()
                    new_site_location_ids.append(new_site_location.id)

    # Handle multiple equipment groups, including new ones
    for equipment_group_id in equipment_group_ids:
        if equipment_group_id != 'new':
            new_equipment_group_ids.append(equipment_group_id)
        else:
            new_equipment_group_names = request.form.getlist('new_equipmentGroup_name[]')
            new_equipment_group_descriptions = request.form.getlist('new_equipmentGroup_description[]')

            for name, description in zip(new_equipment_group_names, new_equipment_group_descriptions):
                if name:
                    # Ensure equipment group is related to an area
                    area_id = new_area_ids[0] if new_area_ids else None
                    new_equipment_group = EquipmentGroup(name=name, description=description, area_id=area_id)
                    session.add(new_equipment_group)
                    session.commit()
                    new_equipment_group_ids.append(new_equipment_group.id)

    # Handle models and relate them to equipment groups
    for model_id in model_ids:
        if model_id != 'new':
            new_model_ids.append(model_id)
        else:
            new_model_names = request.form.getlist('new_model_name[]')
            new_model_descriptions = request.form.getlist('new_model_description[]')

            for name, description in zip(new_model_names, new_model_descriptions):
                if name:
                    # Ensure model is related to an equipment group
                    equipment_group_id = new_equipment_group_ids[0] if new_equipment_group_ids else None
                    new_model = Model(name=name, description=description, equipment_group_id=equipment_group_id)
                    session.add(new_model)
                    session.commit()
                    new_model_ids.append(new_model.id)

    # Handle asset numbers and relate them to models
    for asset_number_id in asset_number_ids:
        if asset_number_id != 'new':
            new_asset_number_ids.append(asset_number_id)
        else:
            new_asset_numbers = request.form.getlist('new_assetNumber[]')
            new_asset_number_descriptions = request.form.getlist('new_assetNumber_description[]')

            for number, description in zip(new_asset_numbers, new_asset_number_descriptions):
                if number:
                    # Ensure asset number is related to a model
                    model_id = new_model_ids[0] if new_model_ids else None
                    new_asset_number = AssetNumber(number=number, description=description, model_id=model_id)
                    session.add(new_asset_number)
                    session.commit()
                    new_asset_number_ids.append(new_asset_number.id)

    # Handle locations and relate them to models
    for location_id in location_ids:
        if location_id != 'new':
            new_location_ids.append(location_id)
        else:
            new_location_names = request.form.getlist('new_location_name[]')
            new_location_descriptions = request.form.getlist('new_location_description[]')

            for name, description in zip(new_location_names, new_location_descriptions):
                if name:
                    # Ensure location is related to a model
                    model_id = new_model_ids[0] if new_model_ids else None
                    new_location = Location(name=name, description=description, model_id=model_id)
                    session.add(new_location)
                    session.commit()
                    new_location_ids.append(new_location.id)

    # Ensure at least one of Area or Site Location is provided
    if not new_area_ids and not new_site_location_ids:
        return jsonify({'error': 'At least one of Area or Site Location must be selected.'}), 400

    # Create the new position with the selected or created IDs
    new_position = Position(
        area_id=new_area_ids[0] if new_area_ids else None,  # Assuming one-to-one relationship
        site_location_id=new_site_location_ids[0] if new_site_location_ids else None,
        equipment_group_id=new_equipment_group_ids[0] if new_equipment_group_ids else None,
        model_id=new_model_ids[0] if new_model_ids else None,
        asset_number_id=new_asset_number_ids[0] if new_asset_number_ids else None,
        location_id=new_location_ids[0] if new_location_ids else None
    )

    # Save the new position to the database
    session.add(new_position)
    session.commit()

    return jsonify({'success': True, 'position_id': new_position.id})

@position_data_assignment_data_add_dependencies_bp.route('/add_site_location', methods=['GET', 'POST'])
def add_site_location():
    if request.method == 'POST':
        title = request.form.get('site_location_title')
        room_number = request.form.get('site_location_room_number')

        new_site_location = SiteLocation(title=title, room_number=room_number)
        session.add(new_site_location)
        session.commit()
        flash('New Site Location added successfully!')
        return redirect(url_for('position_data_assignment_data_add_dependencies_bp.add_site_location'))

    return render_template('add_site_location.html')

@position_data_assignment_data_add_dependencies_bp.route('/add_area', methods=['GET', 'POST'])
def add_area():
    if request.method == 'POST':
        area_name = request.form.get('area_name')
        area_description = request.form.get('area_description')

        new_area = Area(name=area_name, description=area_description)
        session.add(new_area)
        session.commit()
        flash('New Area added successfully!')
        return redirect(url_for('position_data_assignment_data_add_dependencies_bp.add_area'))

    return render_template('add_area.html')

@position_data_assignment_data_add_dependencies_bp.route('/add_equipment_group', methods=['GET', 'POST'])
def add_equipment_group():
    if request.method == 'POST':
        name = request.form.get('equipment_group_name')
        description = request.form.get('equipment_group_description')
        area_id = request.form.get('area_id')  # Link with Area

        new_equipment_group = EquipmentGroup(name=name, description=description, area_id=area_id)
        session.add(new_equipment_group)
        session.commit()
        flash('New Equipment Group added successfully!')
        return redirect(url_for('position_data_assignment_data_add_dependencies_bp.add_equipment_group'))

    areas = Area.query.all()
    return render_template('add_equipment_group.html', areas=areas)

@position_data_assignment_data_add_dependencies_bp.route('/add_model', methods=['GET', 'POST'])
def add_model():
    if request.method == 'POST':
        name = request.form.get('model_name')
        description = request.form.get('model_description')
        equipment_group_id = request.form.get('equipment_group_id')  # Link with EquipmentGroup

        new_model = Model(name=name, description=description, equipment_group_id=equipment_group_id)
        session.add(new_model)
        session.commit()
        flash('New Model added successfully!')
        return redirect(url_for('position_data_assignment_data_add_dependencies_bp.add_model'))

    equipment_groups = EquipmentGroup.query.all()
    return render_template('add_model.html', equipment_groups=equipment_groups)

@position_data_assignment_data_add_dependencies_bp.route('/add_asset_number', methods=['GET', 'POST'])
def add_asset_number():
    if request.method == 'POST':
        number = request.form.get('asset_number')
        description = request.form.get('asset_description')
        model_id = request.form.get('model_id')  # Link with Model

        new_asset_number = AssetNumber(number=number, description=description, model_id=model_id)
        session.add(new_asset_number)
        session.commit()
        flash('New Asset Number added successfully!')
        return redirect(url_for('position_data_assignment_data_add_dependencies_bp.add_asset_number'))

    models = Model.query.all()
    return render_template('add_asset_number.html', models=models)

@position_data_assignment_data_add_dependencies_bp.route('/add_location', methods=['GET', 'POST'])
def add_location():
    if request.method == 'POST':
        name = request.form.get('location_name')
        description = request.form.get('location_description')
        model_id = request.form.get('model_id')  # Link with Model

        new_location = Location(name=name, description=description, model_id=model_id)
        session.add(new_location)
        session.commit()
        flash('New Location added successfully!')
        return redirect(url_for('position_data_assignment_data_add_dependencies_bp.add_location'))

    models = Model.query.all()
    return render_template('add_location.html', models=models)

@position_data_assignment_data_add_dependencies_bp.route('/get_equipment_groups', methods=['GET'])
def get_equipment_groups():
    area_id = request.args.get('area_id')
    equipment_groups = session.query(EquipmentGroup).filter_by(area_id=area_id).all()
    data = [{'id': eg.id, 'name': eg.name} for eg in equipment_groups]
    return jsonify(data)

@position_data_assignment_data_add_dependencies_bp.route('/get_models', methods=['GET'])
def get_models():
    equipment_group_id = request.args.get('equipment_group_id')
    models = session.query(Model).filter_by(equipment_group_id=equipment_group_id).all()
    data = [{'id': model.id, 'name': model.name} for model in models]
    return jsonify(data)

@position_data_assignment_data_add_dependencies_bp.route('/get_asset_numbers', methods=['GET'])
def get_asset_numbers():
    model_id = request.args.get('model_id')
    asset_numbers = session.query(AssetNumber).filter_by(model_id=model_id).all()
    data = [{'id': asset.id, 'number': asset.number} for asset in asset_numbers]
    return jsonify(data)

@position_data_assignment_data_add_dependencies_bp.route('/get_locations', methods=['GET'])
def get_locations():
    model_id = request.args.get('model_id')
    locations = session.query(Location).filter_by(model_id=model_id).all()
    data = [{'id': location.id, 'name': location.name} for location in locations]
    return jsonify(data)

@position_data_assignment_data_add_dependencies_bp.route('/get_site_locations', methods=['GET'])
def get_site_locations():
    model_id = request.args.get('model_id')
    asset_number_id = request.args.get('asset_number_id')
    location_id = request.args.get('location_id')
    area_id = request.args.get('area_id')  # New parameter for area
    equipment_group_id = request.args.get('equipment_group_id')  # New parameter for equipment group

    # Log the incoming request parameters
    logger.info(f"Received request to /get_site_locations with model_id: {model_id}, "
                f"asset_number_id: {asset_number_id}, location_id: {location_id}, "
                f"area_id: {area_id}, equipment_group_id: {equipment_group_id}")

    try:
        # Filter positions by all the provided filters
        positions = session.query(Position).filter_by(
            model_id=model_id,
            asset_number_id=asset_number_id,
            location_id=location_id,
            area_id=area_id,
            equipment_group_id=equipment_group_id
        ).all()

        # Log the number of positions found
        logger.info(f"Found {len(positions)} positions matching the filters.")

        # Extract site locations from the filtered positions
        site_locations = [
            {'id': pos.site_location.id, 'title': pos.site_location.title, 'room_number': pos.site_location.room_number}
            for pos in positions if pos.site_location
        ]

        # Log the number of site locations found
        logger.info(f"Extracted {len(site_locations)} site locations.")

        # Add a default "New Site Location" option to the list
        site_locations.append({'id': 'new', 'title': 'New Site Location', 'room_number': ''})

        return jsonify(site_locations)
    except Exception as e:
        logger.error(f"Error fetching site locations: {e}")
        return jsonify({"error": "An error occurred while fetching site locations"}), 500


@position_data_assignment_data_add_dependencies_bp.route('/search_site_locations', methods=['GET'])
def search_site_locations():
    search_term = request.args.get('search', '')

    # Log the search term
    logger.info(f"Received request to /search_site_locations with search term: {search_term}")

    if search_term:
        try:
            # Search for site locations by the title (room number can also be included if needed)
            site_locations = session.query(SiteLocation).filter(SiteLocation.title.ilike(f'%{search_term}%')).all()

            # Log the number of search results found
            logger.info(f"Found {len(site_locations)} site locations matching the search term.")

            results = [
                {'id': location.id, 'title': location.title, 'room_number': location.room_number}
                for location in site_locations
            ]

            return jsonify(results)
        except Exception as e:
            logger.error(f"Error during site location search: {e}")
            return jsonify({"error": "An error occurred during site location search"}), 500

    logger.info("No search term provided, returning an empty list.")
    return jsonify([])  # Return an empty list if no search term is provided
