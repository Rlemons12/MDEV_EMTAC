from flask import Blueprint, jsonify, request
from config_env import DatabaseConfig
from sqlalchemy.exc import SQLAlchemyError
from emtacdb_fts import (
    Task, Problem, Part, Drawing, Image, CompleteDocument, Position,
    Area, EquipmentGroup, Model, AssetNumber, Location, SiteLocation,
    PartTaskAssociation, DrawingTaskAssociation, ImageTaskAssociation, CompleteDocumentTaskAssociation
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db_config = DatabaseConfig()

# Blueprint for task-related routes
pst_troubleshooting_task_bp = Blueprint('pst_troubleshooting_task_bp', __name__)

@pst_troubleshooting_task_bp.route('/get_task_details/<int:task_id>', methods=['GET'])
def get_task_details(task_id):
    session = db_config.get_main_session()
    logger.info(f"Fetching details for task ID: {task_id}")

    try:
        # Query the Task table
        task = session.query(Task).filter_by(id=task_id).first()

        if not task:
            logger.warning(f"Task with ID {task_id} not found.")
            return jsonify({"error": "Task not found"}), 404

        # Compile a list of positions associated with the task, with detailed data
        positions = [{
            "area_id": assoc.position.area_id if assoc.position else None,
            "area_name": assoc.position.area.name if assoc.position and assoc.position.area else None,
            "equipment_group_id": assoc.position.equipment_group_id if assoc.position else None,
            "equipment_group_name": assoc.position.equipment_group.name if assoc.position and assoc.position.equipment_group else None,
            "model_id": assoc.position.model_id if assoc.position else None,
            "model_name": assoc.position.model.name if assoc.position and assoc.position.model else None,
            "asset_number_id": assoc.position.asset_number.id if assoc.position and assoc.position.asset_number else None,
            "asset_number": assoc.position.asset_number.number if assoc.position and assoc.position.asset_number else None,
            "location_id": assoc.position.location.id if assoc.position and assoc.position.location else None,
            "location_name": assoc.position.location.name if assoc.position and assoc.position.location else None,
            "site_location_id": assoc.position.site_location.id if assoc.position and assoc.position.site_location else None,
            "site_location_title": assoc.position.site_location.title if assoc.position and assoc.position.site_location else None,
            "site_location_room_number": assoc.position.site_location.room_number if assoc.position and assoc.position.site_location else None
        } for assoc in task.task_positions]

        task_data = {
            "id": task.id,
            "name": task.name,
            "description": task.description,
            "positions": positions,  # List of detailed positions
        }

        # Retrieve associated parts
        part_associations = session.query(PartTaskAssociation).filter_by(task_id=task_id).all()
        parts = [{
            "id": assoc.part.id,
            "part_number": assoc.part.part_number,
            "name": assoc.part.name
        } for assoc in part_associations if assoc.part]
        logger.info(f"Retrieved {len(parts)} parts for task ID {task_id}")

        # Retrieve associated drawings
        drawing_associations = session.query(DrawingTaskAssociation).filter_by(task_id=task_id).all()
        drawings = [{
            "id": assoc.drawing.id,
            "drw_number": assoc.drawing.drw_number,
            "drw_name": assoc.drawing.drw_name
        } for assoc in drawing_associations if assoc.drawing]
        logger.info(f"Retrieved {len(drawings)} drawings for task ID {task_id}")

        # Retrieve associated images
        image_associations = session.query(ImageTaskAssociation).filter_by(task_id=task_id).all()
        images = [{
            "id": assoc.image.id,
            "title": assoc.image.title,
            "description": assoc.image.description
        } for assoc in image_associations if assoc.image]
        logger.info(f"Retrieved {len(images)} images for task ID {task_id}")

        # Retrieve associated complete documents
        complete_document_associations = session.query(CompleteDocumentTaskAssociation).filter_by(task_id=task_id).all()
        complete_documents = [{
            "id": assoc.complete_document.id,
            "title": assoc.complete_document.title
        } for assoc in complete_document_associations if assoc.complete_document]
        logger.info(f"Retrieved {len(complete_documents)} complete documents for task ID {task_id}")

        # Combine all retrieved data
        task_data["associations"] = {
            "parts": parts,
            "drawings": drawings,
            "images": images,
            "complete_documents": complete_documents
        }

        return jsonify({"task": task_data}), 200

    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error for task ID {task_id}: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()
        logger.info(f"Session closed for task ID {task_id}")
