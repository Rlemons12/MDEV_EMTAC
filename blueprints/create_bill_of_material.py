import logging
from flask import Blueprint, send_file, request, redirect, url_for, flash, render_template, session as flask_session
from modules.emtacdb.emtacdb_fts import (
    PartsPositionImageAssociation, Position, Part, Image, BOMResult, AssetNumber
)
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.config import BASE_DIR
from modules.configuration.log_config import logger
import json
import os

# Instantiate the database configuration
db_config = DatabaseConfig()

# Blueprint setup (define this before using it)
create_bill_of_material_bp = Blueprint('create_bill_of_material_bp', __name__)

def serve_bom_image(session, image_id):
    logger.info(f"Entered serve_bom_image with image_id: {image_id}")
    try:
        image = session.query(Image).filter_by(id=image_id).first()
        if image:
            logger.debug(f"Image found: {image.title} with file_path: {image.file_path}")
            file_path = os.path.join(BASE_DIR, image.file_path)
            logger.debug(f"Constructed file path: {file_path}")
            if os.path.exists(file_path):
                logger.info(f"File exists. Serving file: {file_path}")
                return send_file(file_path, mimetype='image/jpeg', as_attachment=False)
            else:
                logger.error(f"File not found at path: {file_path}")
                return "Image file not found", 404
        else:
            logger.error(f"No image found with ID: {image_id}")
            return "Image not found", 404
    except Exception as e:
        logger.exception(f"Exception in serve_bom_image: {e}")
        return "Internal Server Error", 500

@create_bill_of_material_bp.route('/bom_serve_image/<int:image_id>')
def bom_serve_image_route(image_id):
    logger.debug(f"Route /bom_serve_image accessed with image_id: {image_id}")
    db_session = db_config.get_main_session()
    try:
        response = serve_bom_image(db_session, image_id)
        logger.debug(f"Response from serve_bom_image: {response}")
        return response
    except Exception as e:
        logger.exception(f"Error in bom_serve_image_route for image_id {image_id}: {e}")
        flash(f"Error serving image {image_id}", "error")
        return "Image not found", 404
    finally:
        db_session.close()
        logger.debug("Database session closed in bom_serve_image_route.")

@create_bill_of_material_bp.route('/create_bill_of_material', methods=['POST'])
def create_bill_of_material():
    logger.info("Entered create_bill_of_material route.")
    db_session = db_config.get_main_session()
    try:
        logger.debug("Retrieving form data for create_bill_of_material.")
        area_id = request.form.get('area')
        equipment_group_id = request.form.get('equipment_group')
        model_id = request.form.get('model')
        asset_number_id = request.form.get('asset_number')
        location_id = request.form.get('location')
        logger.debug(f"Form data received: area_id={area_id}, equipment_group_id={equipment_group_id}, "
                     f"model_id={model_id}, asset_number_id={asset_number_id}, location_id={location_id}")

        # Start building the query
        query = db_session.query(Position)

        if area_id:
            query = query.filter(Position.area_id == int(area_id))
            logger.debug(f"Filtered by area_id: {area_id}")
        if equipment_group_id:
            query = query.filter(Position.equipment_group_id == int(equipment_group_id))
            logger.debug(f"Filtered by equipment_group_id: {equipment_group_id}")
        if model_id:
            query = query.filter(Position.model_id == int(model_id))
            logger.debug(f"Filtered by model_id: {model_id}")
        if asset_number_id:
            query = query.filter(Position.asset_number_id == int(asset_number_id))
            logger.debug(f"Filtered by asset_number_id: {asset_number_id}")
        if location_id:
            query = query.filter(Position.location_id == int(location_id))
            logger.debug(f"Filtered by location_id: {location_id}")

        positions = query.all()
        logger.info(f"Number of positions found: {len(positions)}")

        if not positions:
            logger.warning("No matching positions found for create_bill_of_material.")
            flash('No matching positions found for the provided input.', 'error')
            return render_template('bill_of_materials/bill_of_materials.html')

        results = []
        for position in positions:
            logger.debug(f"Processing position with ID: {position.id}")
            parts_images = db_session.query(PartsPositionImageAssociation).filter_by(position_id=position.id).all()
            logger.debug(f"Found {len(parts_images)} parts/images associations for position ID: {position.id}")
            for association in parts_images:
                part = db_session.query(Part).filter_by(id=association.part_id).first()
                if part:
                    store_bom_results(db_session, part_id=part.id, position_id=position.id,
                                      image_id=association.image_id, description="Sample description")
                    results.append({'part_id': part.id, 'image_id': association.image_id})
                    logger.debug(f"Stored BOM result for part ID: {part.id}, position ID: {position.id}, "
                                 f"image ID: {association.image_id}")
                else:
                    logger.error(f"Part not found with ID: {association.part_id}")

        db_session.commit()
        logger.info("BOM results committed successfully.")

        flask_session['results'] = json.dumps(results)
        flask_session['model_id'] = model_id
        flask_session['asset_number_id'] = asset_number_id
        flask_session['location_id'] = location_id
        logger.debug("BOM results stored in session.")

        logger.info("Redirecting to view_bill_of_material route with index 0.")
        return redirect(url_for('create_bill_of_material_bp.view_bill_of_material', index=0))
    except Exception as e:
        logger.exception(f"Exception in create_bill_of_material: {e}")
        flash(f'An error occurred: {str(e)}', 'error')
        db_session.rollback()
        return render_template('bill_of_materials/bill_of_materials.html')
    finally:
        db_session.close()
        logger.debug("Database session closed in create_bill_of_material.")

@create_bill_of_material_bp.route('/view_bill_of_material', methods=['GET'])
def view_bill_of_material():
    logger.info("Entered view_bill_of_material route.")
    db_session = db_config.get_main_session()
    try:
        index = request.args.get('index', 0, type=int)
        per_page = request.args.get('per_page', 4, type=int)
        logger.debug(f"Pagination parameters: index={index}, per_page={per_page}")

        query = db_session.query(BOMResult).offset(index).limit(per_page)
        results = query.all()
        total_results = db_session.query(BOMResult).count()
        logger.info(f"Retrieved {len(results)} BOM results out of total {total_results}")

        parts_and_images = []
        for result in results:
            logger.debug(f"Processing BOM result with part_id: {result.part_id}")
            part = db_session.query(Part).filter_by(id=result.part_id).first()
            if not part:
                logger.error(f"Part not found for BOM result with part_id: {result.part_id}")
                continue

            image = None
            if result.image_id is not None:
                image = db_session.query(Image).filter_by(id=result.image_id).first()
                if not image:
                    logger.warning(f"Image not found for BOM result with image_id: {result.image_id}")
            parts_and_images.append({
                'part': part,
                'image': image,
                'description': result.description
            })

        next_index = index + per_page if index + per_page < total_results else None
        prev_index = index - per_page if index - per_page >= 0 else None
        logger.info(f"Pagination calculated: next_index={next_index}, prev_index={prev_index}")

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
        logger.exception(f"Exception in view_bill_of_material: {e}")
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('create_bill_of_materials_bp.bill_of_materials'))
    finally:
        db_session.close()
        logger.debug("Database session closed in view_bill_of_material.")

@create_bill_of_material_bp.route('/bom_general_search', methods=['POST'])
def bom_general_search():
    logger.info("Entered bom_general_search route.")
    db_session = db_config.get_main_session()
    try:
        clear_bom_results()
        general_asset_number = request.form.get('general_asset_number', '').strip()
        general_location = request.form.get('general_location', '').strip()
        logger.debug(f"General search parameters: asset_number='{general_asset_number}', location='{general_location}'")

        query = db_session.query(Position)
        if general_asset_number:
            asset_number_records = db_session.query(AssetNumber).filter(AssetNumber.number == general_asset_number).all()
            if not asset_number_records:
                logger.warning("No AssetNumber records found for provided input.")
                flash('No Asset Number found matching the provided input.', 'error')
                return render_template('bill_of_materials/bill_of_materials.html')
            asset_number_ids = [record.id for record in asset_number_records]
            logger.debug(f"AssetNumber IDs retrieved: {asset_number_ids}")
            query = query.filter(Position.asset_number_id.in_(asset_number_ids))
        else:
            query = query.filter(Position.asset_number_id.isnot(None))
            logger.debug("Asset number not provided; excluding positions with NULL asset_number_id.")

        if general_location:
            # Note: Ensure you have imported Location from your models.
            from modules.emtacdb.emtacdb_fts import Location  # Adjust the import as necessary
            location_records = db_session.query(Location).filter(Location.name == general_location).all()
            if not location_records:
                logger.warning("No Location records found for provided input.")
                flash('No Location found matching the provided input.', 'error')
                return render_template('bill_of_materials/bill_of_materials.html')
            location_ids = [record.id for record in location_records]
            logger.debug(f"Location IDs retrieved: {location_ids}")
            query = query.filter(Position.location_id.in_(location_ids))
        else:
            query = query.filter(Position.location_id.isnot(None))
            logger.debug("Location not provided; excluding positions with NULL location_id.")

        positions = query.all()
        logger.info(f"Found {len(positions)} positions in general search.")
        if not positions:
            flash('No results found for the given Asset Number or Location.', 'error')
            return render_template('bill_of_materials/bill_of_materials.html')

        results = []
        for position in positions:
            logger.debug(f"Processing position ID: {position.id} in general search.")
            parts_images = db_session.query(PartsPositionImageAssociation).filter_by(position_id=position.id).all()
            for association in parts_images:
                part = db_session.query(Part).filter_by(id=association.part_id).first()
                if part:
                    store_bom_results(db_session, part_id=part.id, position_id=position.id,
                                      image_id=association.image_id, description="General search result")
                    results.append({'part_id': part.id, 'image_id': association.image_id})
                    logger.debug(f"Stored general search BOM result for part ID: {part.id}")
                else:
                    logger.error(f"Part not found with ID: {association.part_id}")
        db_session.commit()
        logger.info("General search BOM results committed successfully.")

        flask_session['results'] = json.dumps(results)
        flask_session['general_asset_number'] = general_asset_number
        flask_session['general_location'] = general_location
        logger.debug("General search results stored in session.")

        logger.info("Redirecting to view_bill_of_material route with index 0 for general search results.")
        return redirect(url_for('create_bill_of_material_bp.view_bill_of_material', index=0))
    except Exception as e:
        logger.exception(f"Exception in bom_general_search: {e}")
        flash(f'An error occurred during general search: {str(e)}', 'error')
        db_session.rollback()
        return render_template('bill_of_materials/bill_of_materials.html')
    finally:
        db_session.close()
        logger.debug("Database session closed in bom_general_search.")

def store_bom_results(session, part_id, position_id, image_id=None, description=None):
    try:
        result = BOMResult(part_id=part_id, position_id=position_id, image_id=image_id, description=description)
        session.add(result)
        logger.info(f"Stored BOM result: part_id={part_id}, position_id={position_id}, "
                    f"image_id={image_id}, description='{description}'")
    except Exception as e:
        logger.exception(f"Failed to store BOM result for part_id={part_id}: {e}")
        raise

def clear_bom_results():
    logger.info("Initiating clearing of BOM results.")
    session = db_config.get_main_session()
    try:
        deleted_count = session.query(BOMResult).delete()
        session.commit()
        logger.info(f"Cleared BOM results successfully, deleted {deleted_count} record(s).")
    except Exception as e:
        logger.exception(f"Failed to clear BOM results: {e}")
        session.rollback()
    finally:
        session.close()
        logger.debug("Database session closed in clear_bom_results.")
