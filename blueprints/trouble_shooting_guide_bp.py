from datetime import datetime
import logging
from flask import Blueprint, request, jsonify
from sqlalchemy.orm import sessionmaker
from emtacdb_fts import (
    Location, Problem, Solution, CompleteDocument, Image, engine, 
    ImageSolutionAssociation, ImageProblemAssociation, CompleteDocumentProblemAssociation, Part, Drawing, 
    ProblemPositionAssociation, Position, CompleteDocumentSolutionAssociation, PartsPositionAssociation, PartProblemAssociation, PartSolutionAssociation, 
    DrawingPositionAssociation, DrawingProblemAssociation, DrawingSolutionAssociation
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Session = sessionmaker(bind=engine)
session = Session()

trouble_shooting_guide_bp = Blueprint('trouble_shooting_guide_bp', __name__)

@trouble_shooting_guide_bp.route('/update_problem_solution', methods=['POST'])
def update_problem_solution():
    problem_name = request.form.get('problem_name')
    tsg_model_id = request.form.get('tsg_model')
    tsg_asset_number_id = request.form.get('tsg_asset_number')
    tsg_location_id = request.form.get('tsg_location')
    problem_description = request.form.get('problem_description')
    solution_description = request.form.get('solution_description')
    selected_document_ids = request.form.getlist('tsg_document_search')
    selected_problem_image_ids = request.form.getlist('tsg_problem_image_search')
    selected_solution_image_ids = request.form.getlist('tsg_solution_image_search')
    selected_part_ids = request.form.getlist('tsg_selected_part_search')
    selected_drawing_id = request.form.get('tsg_selected_drawing_search')

    # Debug: Log all form data
    logger.info(f"Form Data: {request.form}")

    if not selected_part_ids:
        logger.info("No selected part IDs provided")
        selected_parts_id = []
    else:
        logger.info(f"Selected part IDs: {selected_part_ids}")
        selected_parts = session.query(Part).filter(Part.id.in_(selected_part_ids)).all()
        selected_parts_id = [part.id for part in selected_parts]
        logger.info(f"Matching parts from database: {[part.part_number for part in selected_parts]}")
        logger.info(f"Selected parts ID: {selected_parts_id}")
        if not selected_parts_id:
            return jsonify({'error': 'No matching part found'}), 400

    if not selected_drawing_id:
        logger.info("No selected drawing ID provided")
        selected_drawing_id = []
    else:
        logger.info(f"Selected drawing ID: {selected_drawing_id}")
        selected_drawings = session.query(Drawing).filter(Drawing.id == selected_drawing_id).all()
        selected_drawing_id = [drawing.id for drawing in selected_drawings]
        logger.info(f"Matching drawings from database: {[drawing.drw_number for drawing in selected_drawings]}")
        logger.info(f"Selected drawing ID: {selected_drawing_id}")
        if not selected_drawing_id:
            return jsonify({'error': 'No matching drawing found'}), 400

    logger.info(f"Problem Name: {problem_name}")
    logger.info(f"Model ID: {tsg_model_id}")
    logger.info(f"Asset Number ID: {tsg_asset_number_id}")
    logger.info(f"Location: {tsg_location_id}")
    logger.info(f"Problem Description: {problem_description}")
    logger.info(f"Solution Description: {solution_description}")
    logger.info(f"Selected Document IDs: {selected_document_ids}")
    logger.info(f"Selected Problem Image IDs: {selected_problem_image_ids}")
    logger.info(f"Selected Solution Image IDs: {selected_solution_image_ids}")
    logger.info(f"Selected Part IDs: {selected_part_ids}")
    logger.info(f"Selected Drawing ID: {selected_drawing_id}")

    if not (problem_name and tsg_model_id and tsg_location_id and problem_description and solution_description):
        return jsonify({'error': 'All required fields are not provided'}), 400

    selected_document_ids = [int(doc_id) for doc_id in selected_document_ids]
    selected_problem_image_ids = [int(img_id) for img_id in selected_problem_image_ids]
    selected_solution_image_ids = [int(img_id) for img_id in selected_solution_image_ids]

    try:
        problem = Problem(name=problem_name, description=problem_description)
        session.add(problem)
        session.commit()

        position = Position(
            model_id=tsg_model_id,
            asset_number_id=tsg_asset_number_id,
            location_id=tsg_location_id
        )
        session.add(position)
        session.commit()

        problem_position_association = ProblemPositionAssociation(problem_id=problem.id, position_id=position.id)
        session.add(problem_position_association)

        for doc_id in selected_document_ids:
            document_association = CompleteDocumentProblemAssociation(problem_id=problem.id, complete_document_id=doc_id)
            session.add(document_association)

        solution = Solution(description=solution_description, problem=problem)
        session.add(solution)
        session.commit()

        for doc_id in selected_document_ids:
            document_association = CompleteDocumentSolutionAssociation(solution_id=solution.id, complete_document_id=doc_id)
            session.add(document_association)

        for img_id in selected_problem_image_ids:
            image_problem_association = ImageProblemAssociation(image_id=img_id, problem_id=problem.id)
            session.add(image_problem_association)

        for img_id in selected_solution_image_ids:
            image_solution_association = ImageSolutionAssociation(image_id=img_id, solution_id=solution.id)
            session.add(image_solution_association)

        session.commit()

        for part_id in selected_parts_id:
            logger.info(f"Processing part_id: {part_id}")
            part = session.query(Part).filter_by(id=part_id).first()
            if part:
                position_part_association = PartsPositionAssociation(part_id=part_id, position_id=position.id)
                session.add(position_part_association)
            else:
                logger.error("Error: Part not found for association")
                return jsonify({'error': 'Part not found for association'}), 400

        for drawing_id in selected_drawing_id:
            logger.info(f"Processing drawing_id: {drawing_id}")
            drawing = session.query(Drawing).filter_by(id=drawing_id).first()
            if drawing:
                position_drawing_association = DrawingPositionAssociation(drawing_id=drawing_id, position_id=position.id)
                session.add(position_drawing_association)
            else:
                logger.error("Error: Drawing not found for association")
                return jsonify({'error': 'Drawing not found for association'}), 400

        session.commit()

        return jsonify({'success': 'Problem and solution descriptions uploaded successfully'})
    except Exception as e:
        session.rollback()
        logger.error(f"Error: {str(e)}", exc_info=e)
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()
