import os
from flask import Blueprint, request, redirect, url_for, flash, render_template
import logging
from config_env import DatabaseConfig  # Ensure this path is correct
from emtacdb_fts import (
    Problem, Solution,
    ImageProblemAssociation, ImageSolutionAssociation,
    CompleteDocumentProblemAssociation,
    PartProblemAssociation, DrawingProblemAssociation
)
import traceback
from logging.handlers import RotatingFileHandler

# Configure logging
def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Capture all levels

    # Create log directory if it doesn't exist
    log_directory = 'logs'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Adjust as needed

    # File Handler with rotation
    file_handler = RotatingFileHandler(
        os.path.join(log_directory, 'app.log'),
        maxBytes=5*1024*1024,  # 5 MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

logger = setup_logger()

# Initialize DatabaseConfig
db_config = DatabaseConfig()

# Create the Blueprint
troubleshooting_guide_edit_update_bp = Blueprint('troubleshooting_guide_edit_update_bp', __name__)

@troubleshooting_guide_edit_update_bp.route('/troubleshooting_guide_edit_update', methods=['POST'])
def edit_update_problem_solution():
    logger.info("Accessed /troubleshooting_guide_edit_update route via POST method")

    # Log the entire form data for debugging
    logger.debug(f"Entire Form Data: {request.form}")

    # Retrieve form fields
    problem_id = request.form.get('problem_id')
    problem_name = request.form.get('problem_name')
    solution_id = request.form.get('solution_id')
    problem_description = request.form.get('problem_description')
    solution_description = request.form.get('solution_description')

    # Associated data
    selected_problem_image_ids = request.form.getlist('edit_problem_image[]')
    selected_solution_image_ids = request.form.getlist('edit_solution_image[]')
    selected_document_ids = request.form.getlist('edit_document[]')
    selected_part_ids = request.form.getlist('edit_part[]')
    selected_drawing_ids = request.form.getlist('edit_drawing[]')

    # Log individual fields
    logger.info(f"Received form data - Problem ID: {problem_id}, Problem Name: {problem_name}, "
                f"Solution ID: {solution_id}, Problem Description: {problem_description}, "
                f"Solution Description: {solution_description}, Problem Images: {selected_problem_image_ids}, "
                f"Solution Images: {selected_solution_image_ids}, Documents: {selected_document_ids}, "
                f"Parts: {selected_part_ids}, Drawings: {selected_drawing_ids}")

    # Validate required fields
    if not all([problem_id, problem_name, solution_id, problem_description, solution_description]):
        logger.warning("Missing required form data")
        flash('Missing required fields', 'warning')
        return render_template('troubleshooting_guide.html'), 400  # Bad Request

    # Get session from DatabaseConfig
    session = db_config.get_main_session()

    try:
        logger.info(f"Starting update process for Problem ID: {problem_id} and Solution ID: {solution_id}")

        # Retrieve and update the problem record
        problem = session.query(Problem).filter_by(id=problem_id).first()
        if problem:
            logger.debug(f"Found Problem: ID={problem.id}, Name={problem.name}")
            problem.name = problem_name
            problem.description = problem_description
            logger.info(f"Updated Problem ID={problem.id} with new name and description")
        else:
            logger.warning(f"Problem with ID={problem_id} not found")
            flash(f'Problem with ID {problem_id} not found', 'warning')
            return render_template('troubleshooting_guide.html'), 404  # Not Found

        # Retrieve and update the solution record
        solution = session.query(Solution).filter_by(id=solution_id).first()
        if solution:
            logger.debug(f"Found Solution: ID={solution.id}")
            solution.description = solution_description
            logger.info(f"Updated Solution ID={solution.id} with new description")
        else:
            logger.warning(f"Solution with ID={solution_id} not found")
            flash(f'Solution with ID {solution_id} not found', 'warning')
            return render_template('troubleshooting_guide.html'), 404  # Not Found

        # Function to handle associations
        def update_associations(model, filter_field, filter_value, association_fields, selected_ids, association_type):
            logger.info(f"Updating {association_type} associations for ID={filter_value}")

            # Retrieve existing associations
            existing_associations = session.query(model).filter(getattr(model, filter_field) == filter_value).all()
            existing_ids = {getattr(assoc, 'image_id', None) or getattr(assoc, 'complete_document_id', None) or getattr(assoc, 'part_id', None) or getattr(assoc, 'drawing_id', None) for assoc in existing_associations}

            # Determine associations to remove (those that exist but are not in selected_ids)
            to_remove = [assoc for assoc in existing_associations if getattr(assoc, 'image_id', None) or getattr(assoc, 'complete_document_id', None) not in selected_ids]

            # Remove old associations
            for assoc in to_remove:
                session.delete(assoc)
                logger.debug(f"Deleted {association_type} association: {assoc}")
            logger.info(f"Removed {len(to_remove)} existing {association_type} associations for ID={filter_value}")

            # Determine associations to add (those in selected_ids but not already associated)
            to_add = [assoc_id for assoc_id in selected_ids if assoc_id not in existing_ids]

            # Add new associations
            new_associations = []
            for assoc_id in to_add:
                assoc = model(**association_fields(filter_value, assoc_id))
                new_associations.append(assoc)
                logger.debug(f"Prepared new {association_type} association: {assoc}")

            if new_associations:
                session.bulk_save_objects(new_associations)
                logger.info(f"Added {len(new_associations)} new {association_type} associations for ID={filter_value}")
            else:
                logger.info(f"No new {association_type} associations to add for ID={filter_value}")

        # Update ImageProblemAssociation
        update_associations(
            ImageProblemAssociation,
            'problem_id',
            problem_id,
            lambda pid, iid: {'problem_id': pid, 'image_id': iid},
            selected_problem_image_ids,
            'ImageProblemAssociation'
        )

        # Update ImageSolutionAssociation
        update_associations(
            ImageSolutionAssociation,
            'solution_id',
            solution_id,
            lambda sid, iid: {'solution_id': sid, 'image_id': iid},
            selected_solution_image_ids,
            'ImageSolutionAssociation'
        )

        # Update CompleteDocumentProblemAssociation
        update_associations(
            CompleteDocumentProblemAssociation,
            'problem_id',
            problem_id,
            lambda pid, did: {'problem_id': pid, 'complete_document_id': did},
            selected_document_ids,
            'CompleteDocumentProblemAssociation'
        )

        # Update PartProblemAssociation
        update_associations(
            PartProblemAssociation,
            'problem_id',
            problem_id,
            lambda pid, partid: {'problem_id': pid, 'part_id': partid},
            selected_part_ids,
            'PartProblemAssociation'
        )

        # Update DrawingProblemAssociation
        update_associations(
            DrawingProblemAssociation,
            'problem_id',
            problem_id,
            lambda pid, drawingid: {'problem_id': pid, 'drawing_id': drawingid},
            selected_drawing_ids,
            'DrawingProblemAssociation'
        )

        # Commit the changes to the database
        session.commit()
        logger.info(f"Successfully updated Problem ID={problem_id} and Solution ID={solution_id} with all associations")
        flash('Problem and Solution updated successfully', 'success')

    except Exception as e:
        logger.error(f"An error occurred while updating Problem ID={problem_id} and Solution ID={solution_id}: {e}", exc_info=True)
        session.rollback()
        flash(f'An error occurred: {e}', 'danger')

    finally:
        session.close()
        logger.debug("Database session closed")

    # Redirect back to the troubleshooting guide page after updating
    return render_template('troubleshooting_guide.html')
