from flask import Blueprint, request, redirect, flash, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
import os
import logging
from sqlalchemy.orm import joinedload
from modules.emtacdb.emtacdb_fts import (Drawing, Part, PartsPositionImageAssociation, DrawingPositionAssociation,
                                         CompletedDocumentPositionAssociation,
                                         Area, EquipmentGroup, Model, AssetNumber, Location, SiteLocation,
                                         CompleteDocument,
                                         Image, Position, ImagePositionAssociation, Assembly, AssemblyView, SubAssembly,
                                         ToolCategory, ToolManufacturer)
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.config import ALLOWED_EXTENSIONS
from sqlalchemy import or_
from blueprints.position_data_assignment import position_data_assignment_bp
from modules.configuration.log_config import logger
from modules.tool_module.forms import ToolSearchForm


# Initialize DatabaseConfig
db_config = DatabaseConfig()

# Utility function for allowed file types
def allowed_file(filename, allowed_extensions=ALLOWED_EXTENSIONS):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# Route for displaying and submitting position data assignment form
@position_data_assignment_bp.route('/position_data_assignment', methods=['GET', 'POST'])
@position_data_assignment_bp.route('/position_data_assignment', methods=['GET', 'POST'])
def position_data_assignment():
    db_session = db_config.get_main_session()

    position_id = request.args.get('position_id')  # Get position ID from query parameters

    if request.method == 'POST':
        try:
            logger.info("Handling POST request for position_data_assignment.")

            # Handle form submission
            area_id = request.form.get('area_id')
            equipment_group_id = request.form.get('equipment_group_id')
            model_id = request.form.get('model_id')
            asset_number_id = request.form.get('asset_number_id') or None
            asset_number_input = request.form.get('asset_number_input') or None
            location_id = request.form.get('location_id') or None
            location_input = request.form.get('location_input') or None
            assembly_id = request.form.get('assembly_id') or None
            assembly_input = request.form.get('assembly_input') or None
            subassembly_id = request.form.get('subassembly_id') or None
            subassembly_input = request.form.get('subassembly_input') or None
            site_location_id = request.form.get('site_location_id')
            position_id = request.form.get('position_id')
            part_numbers = request.form.getlist('part_numbers[]')

            # Handle manual input for Asset Number and Location
            if not asset_number_id and asset_number_input:
                new_asset = AssetNumber(number=asset_number_input, model_id=model_id)
                db_session.add(new_asset)
                db_session.commit()
                asset_number_id = new_asset.id
                logger.info(f"Created new AssetNumber with ID {asset_number_id}.")

            if not location_id and location_input:
                new_location = Location(name=location_input, model_id=model_id)
                db_session.add(new_location)
                db_session.commit()
                location_id = new_location.id
                logger.info(f"Created new Location with ID {location_id}.")

            # Handle file uploads
            images = request.files.getlist('images[]')
            drawings = request.files.getlist('drawings[]')

            saved_image_paths = []
            for image in images:
                if image and allowed_file(image.filename, ALLOWED_IMAGE_EXTENSIONS):
                    filename = secure_filename(image.filename)
                    image_path = os.path.join('static/uploads/images/', filename)
                    image.save(image_path)
                    saved_image_paths.append(image_path)
                    logger.info(f"Saved image: {image_path}")

            saved_drawing_paths = []
            for drawing in drawings:
                if drawing and allowed_file(drawing.filename, ALLOWED_DRAWING_EXTENSIONS):
                    filename = secure_filename(drawing.filename)
                    drawing_path = os.path.join('static/uploads/drawings/', filename)
                    drawing.save(drawing_path)
                    saved_drawing_paths.append(drawing_path)
                    logger.info(f"Saved drawing: {drawing_path}")

            # Check if we're updating an existing position or creating a new one
            if position_id:
                # Updating an existing position
                position = db_session.query(Position).filter_by(id=position_id).first()
                if not position:
                    flash('Position not found.', 'error')
                    logger.error(f"Position with ID {position_id} not found for update.")
                    return redirect(url_for('position_data_assignment_bp.position_data_assignment'))

                # Update position fields
                position.area_id = area_id
                position.equipment_group_id = equipment_group_id
                position.model_id = model_id
                position.asset_number_id = asset_number_id
                position.location_id = location_id
                position.assembly_id = assembly_id
                position.subassembly_id = subassembly_id
                position.assembly_view_id = assembly_id
                position.site_location_id = site_location_id
                position.parts = part_numbers  # Ensure this is handled correctly

                db_session.commit()
                flash('Position Data Updated Successfully!', 'success')
                logger.info(f"Updated Position ID {position_id} successfully.")

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
                logger.info("Created new PositionDataAssignment successfully.")

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
            logger.info("Handling GET request for position_data_assignment.")
            areas = db_session.query(Area).all()
            equipment_groups = db_session.query(EquipmentGroup).all()
            models = db_session.query(Model).all()
            asset_numbers = db_session.query(AssetNumber).all()
            locations = db_session.query(Location).all()
            assemblies = db_session.query(Assembly).all()
            subassemblies = db_session.query(SubAssembly).all()
            assembly_views = db_session.query(AssemblyView).all()
            site_locations = db_session.query(SiteLocation).all()

            position = None

            # Load the position if position_id is provided (for updating)
            if position_id:
                position = db_session.query(Position).filter_by(id=position_id).first()
                if not position:
                    flash('Position not found.', 'error')
                    logger.error(f"Position with ID {position_id} not found.")
                    return redirect(url_for('position_data_assignment_bp.position_data_assignment'))
                logger.info(f"Loaded Position ID {position_id} for updating.")

            # Instantiate the ToolSearchForm
            tool_search_form = ToolSearchForm()
            logger.debug("Instantiated ToolSearchForm.")

            # Populate choices for SelectMultipleFields
            tool_search_form.tool_category.choices = [
                (c.id, c.name) for c in db_session.query(ToolCategory).order_by(ToolCategory.name).all()
            ]
            tool_search_form.tool_manufacturer.choices = [
                (m.id, m.name) for m in db_session.query(ToolManufacturer).order_by(ToolManufacturer.name).all()
            ]
            logger.debug("Populated ToolSearchForm choices for categories and manufacturers.")

            return render_template(
                'position_data_assignment/position_data_assignment.html',
                areas=areas,
                equipment_groups=equipment_groups,
                models=models,
                asset_numbers=asset_numbers,
                locations=locations,
                assemblies=assemblies,
                subassemblies=subassemblies,
                assembly_views=assembly_views,
                site_locations=site_locations,
                position=position,
                tool_search_form=tool_search_form  # Pass the form to the template
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

@position_data_assignment_bp.route('/get_assemblies')
def get_assemblies():
    location_id = request.args.get('location_id')
    db_session = db_config.get_main_session()
    try:
        assembly = db_session.query(Assembly).filter_by(location_id=location_id).all()
        data = [{'id': assembly.id, 'name': assembly.name, 'description': assembly.description} for assembly in assembly]
        return jsonify(data)
    finally:
        db_session.close()

@position_data_assignment_bp.route('/get_subassemblies')
def get_subassemblies():
    assembly_id = request.args.get('assembly_id')
    db_session = db_config.get_main_session()
    try:
        subassemblies = db_session.query(SubAssembly).filter(SubAssembly.assembly_id == assembly_id).all()
        data = [{'id': subassembly.id, 'name': subassembly.name, 'description': subassembly.description} for subassembly in subassemblies]
        return jsonify(data)
    finally:
        db_session.close()

@position_data_assignment_bp.route('/get_assembly_views')
def get_assembly_views():
    subassembly_id = request.args.get('subassembly_id')
    db_session = db_config.get_main_session()
    try:
        assembly_views = db_session.query(AssemblyView).filter(AssemblyView.subassembly_id == subassembly_id).all()
        data = [{'id': assembly_view.id, 'name': assembly_view.name} for assembly_view in assembly_views]
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
    try:
        # Get filter parameters from the request
        site_location_id = request.args.get('site_location_id')
        area_id = request.args.get('area_id')
        equipment_group_id = request.args.get('equipment_group_id')
        model_id = request.args.get('model_id')
        asset_number_id = request.args.get('asset_number_id')
        location_id = request.args.get('location_id')
        assembly_id = request.args.get('assembly_id')
        subassembly_id = request.args.get('subassembly_id')
        assembly_view_id = request.args.get('assembly_view_id')

        logger.info(f"Received GET request with filters: site_location_id={site_location_id}, area_id={area_id}, equipment_group_id={equipment_group_id}, model_id={model_id}, asset_number_id={asset_number_id}, location_id={location_id}")

        db_session = db_config.get_main_session()

        # Build the query based on filters
        query = db_session.query(Position)

        if site_location_id:
            query = query.filter(Position.site_location_id == site_location_id)
        if area_id:
            query = query.filter(Position.area_id == area_id)
        if equipment_group_id:
            query = query.filter(Position.equipment_group_id == equipment_group_id)
        if model_id:
            query = query.filter(Position.model_id == model_id)
        if asset_number_id:
            query = query.filter(Position.asset_number_id == asset_number_id)
        if location_id:
            query = query.filter(Position.location_id == location_id)
        if assembly_id:
            query = query.filter(Position.assembly_id == assembly_id)
        if subassembly_id:
            query = query.filter(Position.subassembly_id == subassembly_id)
        if assembly_view_id:
            query = query.filter(Position.assembly_view_id == assembly_view_id)

        positions = query.all()
        logger.info(f"Found {len(positions)} positions matching the filters.")

        if not positions:
            logger.warning("No positions found with the given filters.")
            return jsonify({"message": "No positions found"}), 404

        # Prepare the response data
        result_data = []

        for position in positions:
            position_data = {
                'position_id': position.id,
                'area': {
                    'id': position.area.id if position.area else None,
                    'name': position.area.name if position.area else None,
                    'description': position.area.description if position.area else None
                },
                'equipment_group': {
                    'id': position.equipment_group.id if position.equipment_group else None,
                    'name': position.equipment_group.name if position.equipment_group else None
                },
                'model': {
                    'id': position.model.id if position.model else None,
                    'name': position.model.name if position.model else None,
                    'description': position.model.description if position.model else None
                },
                'asset_number': {
                    'id': position.asset_number.id if position.asset_number else None,
                    'number': position.asset_number.number if position.asset_number else None,
                    'description': position.asset_number.description if position.asset_number else None
                },
                'location': {
                    'id': position.location.id if position.location else None,
                    'name': position.location.name if position.location else None,
                    'description': position.location.description if position.location else None
                },
                'site_location': {
                    'id': position.site_location.id if position.site_location else None,
                    'title': position.site_location.title if position.site_location else None,
                    'room_number': position.site_location.room_number if position.site_location else None
                },
                'parts': [],
                'documents': [],
                'drawings': [],
                'images': []
            }

            logger.info(f"Processing Position ID: {position.id}")

            # Fetch parts
            parts_associations = db_session.query(PartsPositionImageAssociation).filter_by(position_id=position.id).all()
            part_ids = [assoc.part_id for assoc in parts_associations]
            if part_ids:
                parts = db_session.query(Part).filter(Part.id.in_(part_ids)).all()
                for part in parts:
                    position_data['parts'].append({
                        'part_id': part.id,
                        'part_number': part.part_number,
                        'name': part.name
                    })

            # Fetch drawings
            drawing_associations = db_session.query(DrawingPositionAssociation).filter_by(position_id=position.id).all()
            drawing_ids = [assoc.drawing_id for assoc in drawing_associations]
            if drawing_ids:
                drawings = db_session.query(Drawing).filter(Drawing.id.in_(drawing_ids)).all()
                for drawing in drawings:
                    position_data['drawings'].append({
                        'drawing_id': drawing.id,
                        'drw_name': drawing.drw_name,
                        'drw_number': drawing.drw_number
                        # Include other fields if necessary
                    })

            # Fetch documents
            document_associations = db_session.query(CompletedDocumentPositionAssociation).filter_by(position_id=position.id).all()
            document_ids = [assoc.complete_document_id for assoc in document_associations]
            if document_ids:
                documents = db_session.query(CompleteDocument).filter(CompleteDocument.id.in_(document_ids)).all()
                for doc in documents:
                    position_data['documents'].append({
                        'document_id': doc.id,
                        'title': doc.title,
                        'rev': doc.rev,
                        'file_path': doc.file_path,
                        'content': doc.content  # Include if necessary
                    })
                logger.info(f"Added {len(documents)} documents to Position ID {position.id}")

            # Fetch images
            image_associations = db_session.query(ImagePositionAssociation).filter_by(position_id=position.id).all()
            image_ids = [assoc.image_id for assoc in image_associations]
            if image_ids:
                images = db_session.query(Image).filter(Image.id.in_(image_ids)).all()
                for image in images:
                    position_data['images'].append({
                        'image_id': image.id,
                        'title': image.title,
                        'description': image.description,
                        'file_path': image.file_path,
                    })
                logger.info(f"Added {len(images)} images to Position ID {position.id}")

            # Append the position data to the result list
            result_data.append(position_data)

        logger.info(f"Returning data for {len(result_data)} positions.")
        return jsonify(result_data), 200

    except Exception as e:
        logger.error(f"Error in /get_positions: {e}", exc_info=True)
        return jsonify({"message": "Error occurred during position search", "error": str(e)}), 500

    finally:
        db_session.close()
        logger.info("Database session closed for /get_positions.")

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
        logger.info(f"Initiating update form data from the updated form")
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

@position_data_assignment_bp.route('/remove_part_from_position', methods=['POST'])
def remove_part_from_position():
    try:
        part_id = request.json.get('part_id')
        position_id = request.json.get('position_id')

        if not part_id or not position_id:
            return jsonify({'message': 'Part ID and Position ID are required'}), 400

        with db_config.get_main_session() as session:
            # Find the association
            association = session.query(PartsPositionImageAssociation).filter_by(part_id=part_id, position_id=position_id).first()
            if not association:
                return jsonify({'message': 'Part is not associated with this position'}), 404

            # Remove the association
            session.delete(association)
            session.commit()

            logger.info(f"Removed Part ID {part_id} from Position ID {position_id}")
            return jsonify({
                'success': True,
                'message': 'Part successfully added to position',
                'part_id': part_id

            }), 200
    except Exception as e:
        logger.error(f"Error removing part from position: {e}")
        return jsonify({'message': 'Failed to remove part from position'}), 500

@position_data_assignment_bp.route('/add_part_to_position', methods=['POST'])
def add_part_to_position():
    try:
        part_id = request.json.get('part_id')
        position_id = request.json.get('position_id')

        if not part_id or not position_id:
            return jsonify({'message': 'Part ID and Position ID are required'}), 400

        with db_config.get_main_session() as session:
            # Check if the association already exists
            existing_association = session.query(PartsPositionImageAssociation).filter_by(part_id=part_id, position_id=position_id).first()
            if existing_association:
                return jsonify({'message': 'Part is already associated with this position'}), 409

            # Create a new association
            new_association = PartsPositionImageAssociation(part_id=part_id, position_id=position_id)
            session.add(new_association)

            # Fetch the part to get the part_number
            part = session.query(Part).filter_by(id=part_id).first()
            if not part:
                return jsonify({'message': 'Part not found'}), 404

            session.commit()

            logger.info(f"Added Part ID {part_id} to Position ID {position_id}")
            return jsonify({
                'message': 'Part successfully added to position',
                'part_id': part_id,
                'part_number': part.part_number
            }), 200
    except Exception as e:
        logger.error(f"Error adding part to position: {e}")
        return jsonify({'message': 'Failed to add part to position'}), 500

@position_data_assignment_bp.route('/pda_search_parts', methods=['GET'])
def search_parts():
    try:
        # Get the search query from the request
        search_term = request.args.get('query', '').strip()

        logging.info(f"Received search_term: '{search_term}' in position_data_assignment_bp")

        # Build the like pattern for part numbers only
        like_pattern = f"%{search_term}%"

        # Use 'ilike' to search only the part_number field
        with db_config.get_main_session() as session:
            parts = session.query(Part).filter(
                Part.part_number.ilike(like_pattern)
            ).limit(10).all()

            # Prepare the result list, including part ID
            result = []
            for part in parts:
                result.append({
                    'id': part.id,
                    'part_number': part.part_number,
                    'name': part.name,
                    'oem_mfg': part.oem_mfg,
                    'model': part.model
                })

        return jsonify(result), 200
    except Exception as e:
        logging.error(f"Error during part search: {e}", exc_info=True)
        return jsonify({"message": "Error occurred during part search"}), 500

@position_data_assignment_bp.route('/create_and_add_part', methods=['POST'])
def create_and_add_part():
    try:
        part_number = request.json.get('part_number')
        position_id = request.json.get('position_id')

        if not part_number or not position_id:
            return jsonify({'message': 'Part number and Position ID are required'}), 400

        with db_config.get_main_session() as session:
            # Check if the part already exists
            part = session.query(Part).filter_by(part_number=part_number).first()
            if not part:
                # Create a new Part
                part = Part(part_number=part_number)
                session.add(part)
                session.flush()  # Get the new part's ID

            # Check if the association already exists
            existing_association = session.query(PartsPositionImageAssociation).filter_by(part_id=part.id, position_id=position_id).first()
            if existing_association:
                return jsonify({'message': 'Part is already associated with this position'}), 409

            # Create a new association
            new_association = PartsPositionImageAssociation(part_id=part.id, position_id=position_id)
            session.add(new_association)
            session.commit()

            logger.info(f"Created Part ID {part.id} and added to Position ID {position_id}")
            return jsonify({
                'message': 'Part successfully created and added to position',
                'part_id': part.id,
                'part_number': part.part_number
            }), 200
    except Exception as e:
        logger.error(f"Error creating and adding part to position: {e}")
        return jsonify({'message': 'Failed to create and add part to position'}), 500

@position_data_assignment_bp.route('/create_and_add_image', methods=['POST'])
def create_and_add_image():
    try:
        title = request.json.get('title')
        description = request.json.get('description', '')  # Optional field
        file_path = request.json.get('file_path', '')      # Optional field
        position_id = request.json.get('position_id')

        if not title or not position_id:
            return jsonify({'message': 'Image title and Position ID are required'}), 400

        with db_config.get_main_session() as session:
            # Create a new Image
            image = Image(title=title, description=description, file_path=file_path)
            session.add(image)
            session.flush()  # Get the new image's ID

            # Verify that the position exists
            position = session.query(Position).filter_by(id=position_id).first()
            if not position:
                return jsonify({'message': 'Position not found'}), 404

            # Check if the association already exists
            existing_association = session.query(ImagePositionAssociation).filter_by(
                image_id=image.id, position_id=position_id
            ).first()
            if existing_association:
                return jsonify({'message': 'Image is already associated with this position'}), 409

            # Create a new association
            new_association = ImagePositionAssociation(
                image_id=image.id, position_id=position_id
            )
            session.add(new_association)
            session.commit()

            logging.info(f"Created Image ID {image.id} and added to Position ID {position_id}")
            return jsonify({
                'message': 'Image successfully created and added to position',
                'image_id': image.id,
                'title': image.title
            }), 200
    except Exception as e:
        logging.error(f"Error creating and adding image to position: {e}", exc_info=True)
        return jsonify({'message': 'Failed to create and add image to position'}), 500

@position_data_assignment_bp.route('/pda_search_images', methods=['GET'])
def search_images():
    try:
        search_term = request.args.get('query', '').strip()
        logging.info(f"Received search_term: '{search_term}' in position_data_assignment_bp for images")

        if not search_term:
            logging.warning("Search term is empty.")
            return jsonify([]), 200  # Return an empty list if no search term provided

        like_pattern = f"%{search_term}%"
        logging.info(f"Using like_pattern: '{like_pattern}'")

        with db_config.get_main_session() as session:
            images = session.query(Image).filter(
                or_(
                    Image.title.ilike(like_pattern),
                    Image.description.ilike(like_pattern)
                )
            ).limit(10).all()

            logging.info(f"Found {len(images)} images matching the search term.")

            result = []
            for image in images:
                logging.debug(f"Image found: ID={image.id}, Title='{image.title}'")
                result.append({
                    'id': image.id,
                    'title': image.title,
                    'description': image.description,
                    'file_path': image.file_path,
                })

        return jsonify(result), 200
    except Exception as e:
        logging.error(f"Error during image search: {e}", exc_info=True)
        return jsonify({"message": "Error occurred during image search"}), 500

@position_data_assignment_bp.route('/add_image_to_position', methods=['POST'])
def add_image_to_position():
    try:
        image_id = request.json.get('image_id')
        position_id = request.json.get('position_id')

        if not image_id or not position_id:
            return jsonify({'message': 'Image ID and Position ID are required'}), 400

        with db_config.get_main_session() as session:
            # Verify that the image and position exist
            image = session.query(Image).filter_by(id=image_id).first()
            if not image:
                return jsonify({'message': 'Image not found'}), 404

            position = session.query(Position).filter_by(id=position_id).first()
            if not position:
                return jsonify({'message': 'Position not found'}), 404

            # Check if the association already exists
            existing_association = session.query(ImagePositionAssociation).filter_by(
                image_id=image_id, position_id=position_id
            ).first()
            if existing_association:
                return jsonify({'message': 'Image is already associated with this position'}), 409

            # Create a new association
            new_association = ImagePositionAssociation(
                image_id=image_id, position_id=position_id
            )
            session.add(new_association)
            session.commit()

            logging.info(f"Added Image ID {image_id} to Position ID {position_id}")
            return jsonify({
                'message': 'Image successfully added to position',
                'image_id': image.id,
                'title': image.title  # Updated attribute name
            }), 200
    except Exception as e:
        logging.error(f"Error adding image to position: {e}", exc_info=True)
        return jsonify({'message': 'Failed to add image to position'}), 500

@position_data_assignment_bp.route('/remove_image_from_position', methods=['POST'])
def remove_image_from_position():
    try:
        image_id = request.json.get('image_id')
        position_id = request.json.get('position_id')

        if not image_id or not position_id:
            return jsonify({'message': 'Image ID and Position ID are required'}), 400

        with db_config.get_main_session() as session:
            # Find the association
            association = session.query(ImagePositionAssociation).filter_by(image_id=image_id, position_id=position_id).first()
            if not association:
                return jsonify({'message': 'Image is not associated with this position'}), 404

            # Remove the association
            session.delete(association)
            session.commit()

            logging.info(f"Removed Image ID {image_id} from Position ID {position_id}")
            return jsonify({'message': 'Image successfully removed from position'}), 200
    except Exception as e:
        logging.error(f"Error removing image from position: {e}", exc_info=True)
        return jsonify({'message': 'Failed to remove image from position'}), 500

@position_data_assignment_bp.route('/pda_search_drawings', methods=['GET'])
def search_drawings():
    try:
        search_term = request.args.get('query', '').strip()
        logging.info(f"Received search_term: '{search_term}' in position_data_assignment_bp for drawings")

        if not search_term:
            logging.warning("Search term is empty.")
            return jsonify([]), 200

        like_pattern = f"%{search_term}%"
        logging.info(f"Using like_pattern: '{like_pattern}'")

        with db_config.get_main_session() as session:
            drawings = session.query(Drawing).filter(
                or_(
                    Drawing.drw_equipment_name.ilike(like_pattern),
                    Drawing.drw_number.ilike(like_pattern),
                    Drawing.drw_name.ilike(like_pattern),
                    Drawing.drw_revision.ilike(like_pattern),
                    Drawing.drw_spare_part_number.ilike(like_pattern)
                )
            ).limit(10).all()

            logging.info(f"Found {len(drawings)} drawings matching the search term.")

            result = []
            for drawing in drawings:
                logging.debug(f"Drawing found: ID={drawing.id}, Name='{drawing.drw_name}'")
                result.append({
                    'id': drawing.id,
                    'drw_equipment_name': drawing.drw_equipment_name,
                    'drw_number': drawing.drw_number,
                    'drw_name': drawing.drw_name,
                    'drw_revision': drawing.drw_revision,
                    'drw_spare_part_number': drawing.drw_spare_part_number,
                    'file_path': drawing.file_path
                })

        return jsonify(result), 200
    except Exception as e:
        logging.error(f"Error during drawing search: {e}", exc_info=True)
        return jsonify({"message": "Error occurred during drawing search"}), 500

@position_data_assignment_bp.route('/pda_create_and_add_document', methods=['POST'])
def pda_create_and_add_document():
    db_session = db_config.get_main_session()
    try:
        title = request.form.get('title')
        description = request.form.get('description', '')
        position_id = request.form.get('position_id')
        file = request.files.get('file')

        if not title or not file or not position_id:
            return jsonify({'message': 'Title, file, and position ID are required.'}), 400

        if not allowed_file(file.filename):
            return jsonify({'message': 'File type not allowed.'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join('static/uploads/documents/', filename)
        file.save(file_path)

        # Create a new CompleteDocument
        new_document = CompleteDocument(
            title=title,
            description=description,
            rev='1.0',  # Or determine the revision as needed
            file_path=file_path,
            content=''  # Populate if necessary
        )
        db_session.add(new_document)
        db_session.flush()  # Get the new document's ID

        # Associate the document with the position
        new_association = CompletedDocumentPositionAssociation(
            complete_document_id=new_document.id,
            position_id=position_id
        )
        db_session.add(new_association)
        db_session.commit()

        return jsonify({
            'message': 'Document successfully created and added to position',
            'document_id': new_document.id,
            'title': new_document.title,
            'rev': new_document.rev
        }), 200

    except Exception as e:
        db_session.rollback()
        logger.error(f"Error creating and adding document to position: {e}", exc_info=True)
        return jsonify({'message': 'Failed to create and add document to position'}), 500

    finally:
        db_session.close()

@position_data_assignment_bp.route('/pda_add_drawing_to_position', methods=['POST'])
def add_drawing_to_position():
    try:
        # Extract JSON data
        data = request.get_json()
        logging.debug(f"Received data: {data}")

        drawing_id = data.get('drawing_id')
        position_id = data.get('position_id')

        logging.debug(f"drawing_id: {drawing_id}, position_id: {position_id}")

        if not drawing_id or not position_id:
            logging.warning("Missing drawing_id or position_id in the request.")
            return jsonify({'message': 'Drawing ID and Position ID are required'}), 400

        with db_config.get_main_session() as session:
            # Verify that the drawing and position exist
            drawing = session.query(Drawing).filter_by(id=drawing_id).first()
            if not drawing:
                logging.warning(f"Drawing with ID {drawing_id} not found.")
                return jsonify({'message': 'Drawing not found'}), 404

            position = session.query(Position).filter_by(id=position_id).first()
            if not position:
                logging.warning(f"Position with ID {position_id} not found.")
                return jsonify({'message': 'Position not found'}), 404

            # Check if the association already exists
            existing_association = session.query(DrawingPositionAssociation).filter_by(
                drawing_id=drawing_id, position_id=position_id
            ).first()
            if existing_association:
                logging.info(f"Association already exists between Drawing ID {drawing_id} and Position ID {position_id}.")
                return jsonify({'message': 'Drawing is already associated with this position'}), 409

            # Create a new association
            new_association = DrawingPositionAssociation(
                drawing_id=drawing_id, position_id=position_id
            )
            session.add(new_association)
            session.commit()

            logging.info(f"Added Drawing ID {drawing_id} to Position ID {position_id}")
            return jsonify({
                'message': 'Drawing successfully added to position',
                'drawing_id': drawing.id,
                'drw_name': drawing.drw_name  # Corrected attribute
            }), 200
    except Exception as e:
        logging.error(f"Error adding drawing to position: {e}", exc_info=True)
        return jsonify({'message': 'Failed to add drawing to position'}), 500

@position_data_assignment_bp.route('/pda_remove_drawing_from_position', methods=['POST'])
def remove_drawing_from_position():
    try:
        drawing_id = request.json.get('drawing_id')
        position_id = request.json.get('position_id')

        if not drawing_id or not position_id:
            return jsonify({'message': 'Drawing ID and Position ID are required'}), 400

        with db_config.get_main_session() as session:
            # Find the association
            association = session.query(DrawingPositionAssociation).filter_by(
                drawing_id=drawing_id, position_id=position_id
            ).first()
            if not association:
                return jsonify({'message': 'Drawing is not associated with this position'}), 404

            # Remove the association
            session.delete(association)
            session.commit()

            logging.info(f"Removed Drawing ID {drawing_id} from Position ID {position_id}")
            return jsonify({'message': 'Drawing successfully removed from position'}), 200
    except Exception as e:
        logging.error(f"Error removing drawing from position: {e}", exc_info=True)
        return jsonify({'message': 'Failed to remove drawing from position'}), 500

@position_data_assignment_bp.route('/pda_search_documents', methods=['GET'])
def search_documents():
    try:
        search_term = request.args.get('query', '').strip()
        logger.info(f"Received search_term: '{search_term}' in position_data_assignment_bp for documents")

        if not search_term:
            logger.warning("Search term is empty.")
            return jsonify([]), 200  # Return an empty list if no search term provided

        like_pattern = f"%{search_term}%"
        logger.info(f"Using like_pattern: '{like_pattern}'")

        with db_config.get_main_session() as session:
            documents = session.query(CompleteDocument).filter(
                or_(
                    CompleteDocument.title.ilike(like_pattern),
                    CompleteDocument.content.ilike(like_pattern),
                    CompleteDocument.rev.ilike(like_pattern)
                )
            ).limit(10).all()

            logger.info(f"Found {len(documents)} documents matching the search term.")

            result = []
            for doc in documents:
                logger.debug(f"Document found: ID={doc.id}, Title='{doc.title}'")
                result.append({
                    'id': doc.id,
                    'title': doc.title,
                    'rev': doc.rev,
                    'file_path': doc.file_path,
                    'content': doc.content  # Include if necessary
                })

        return jsonify(result), 200
    except Exception as e:
        logging.error(f"Error during document search: {e}", exc_info=True)
        return jsonify({"message": "Error occurred during document search"}), 500

@position_data_assignment_bp.route('/pda_add_document_to_position', methods=['POST'])
def add_document_to_position():
    try:
        # Extract JSON data
        data = request.get_json()
        logger.debug(f"Received data: {data}")

        document_id = data.get('document_id')
        position_id = data.get('position_id')

        logger.debug(f"document_id: {document_id}, position_id: {position_id}")

        if not document_id or not position_id:
            logger.warning("Missing document_id or position_id in the request.")
            return jsonify({'message': 'Document ID and Position ID are required'}), 400

        with db_config.get_main_session() as session:
            # Verify that the document and position exist
            document = session.query(CompleteDocument).filter_by(id=document_id).first()
            if not document:
                logger.warning(f"Document with ID {document_id} not found.")
                return jsonify({'message': 'Document not found'}), 404

            position = session.query(Position).filter_by(id=position_id).first()
            if not position:
                logger.warning(f"Position with ID {position_id} not found.")
                return jsonify({'message': 'Position not found'}), 404

            # Check if the association already exists
            existing_association = session.query(CompletedDocumentPositionAssociation).filter_by(
                complete_document_id=document_id, position_id=position_id
            ).first()
            if existing_association:
                logger.info(f"Association already exists between Document ID {document_id} and Position ID {position_id}.")
                return jsonify({'message': 'Document is already associated with this position'}), 409

            # Create a new association
            new_association = CompletedDocumentPositionAssociation(
                complete_document_id=document_id,
                position_id=position_id
            )
            session.add(new_association)
            session.commit()

            logger.info(f"Added Document ID {document_id} to Position ID {position_id}")
            return jsonify({
                'message': 'Document successfully added to position',
                'document_id': document.id,
                'title': document.title,
                'rev': document.rev
            }), 200

    except Exception as e:
        logger.error(f"Error adding document to position: {e}", exc_info=True)
        return jsonify({'message': 'Failed to add document to position'}), 500


@position_data_assignment_bp.route('/pda_remove_document_from_position', methods=['POST'])
def remove_document_from_position():
    try:
        document_id = request.json.get('document_id')
        position_id = request.json.get('position_id')

        if not document_id or not position_id:
            return jsonify({'message': 'Document ID and Position ID are required'}), 400

        with db_config.get_main_session() as session:
            # Find the association
            association = session.query(CompletedDocumentPositionAssociation).filter_by(
                complete_document_id=document_id,
                position_id=position_id
            ).first()
            if not association:
                return jsonify({'message': 'Document is not associated with this position'}), 404

            # Remove the association
            session.delete(association)
            session.commit()

            logger.info(f"Removed Document ID {document_id} from Position ID {position_id}")
            return jsonify({'message': 'Document successfully removed from position'}), 200
    except Exception as e:
        logger.error(f"Error removing document from position: {e}", exc_info=True)
        return jsonify({'message': 'Failed to remove document from position'}), 500

@position_data_assignment_bp.route('/pda_get_documents_by_position', methods=['GET'])
def pda_get_documents_by_position():
    try:
        position_id = request.args.get('position_id', '').strip()
        if not position_id:
            logger.warning("Position ID is missing in the request.")
            return jsonify({'message': 'Position ID is required.'}), 400

        db_session = db_config.get_main_session()

        # Verify that the position exists
        position = db_session.query(Position).filter_by(id=position_id).first()
        if not position:
            logger.warning(f"Position with ID {position_id} not found.")
            return jsonify({'message': 'Position not found.'}), 404

        # Fetch associated documents
        associations = db_session.query(CompletedDocumentPositionAssociation).filter_by(position_id=position_id).all()
        document_ids = [assoc.complete_document_id for assoc in associations]

        documents = db_session.query(CompleteDocument).filter(CompleteDocument.id.in_(document_ids)).all()

        # Prepare the response data
        result = []
        for doc in documents:
            result.append({
                'id': doc.id,
                'title': doc.title,
                'rev': doc.rev,
                'file_path': doc.file_path,
                'content': doc.content  # Include if necessary
            })

        logger.info(f"Fetched {len(result)} documents for Position ID {position_id}.")
        return jsonify({'documents': result}), 200

    except Exception as e:
        logger.error(f"Error fetching documents by position: {e}", exc_info=True)
        return jsonify({'message': 'Failed to fetch documents by position.'}), 500

    finally:
        db_session.close()


