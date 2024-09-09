from flask import Blueprint, request, flash, jsonify, url_for
from emtacdb_fts import (
    ImageProblemAssociation, ImageSolutionAssociation, CompleteDocument, get_total_images_count, create_thumbnail, 
    Image, serve_image, Problem, Solution, Document, Position, ProblemPositionAssociation, SiteLocation, CompleteDocumentProblemAssociation,
    Area, EquipmentGroup, Model, AssetNumber, Location, BillOfMaterial, Part, Drawing, DrawingProblemAssociation, DrawingSolutionAssociation,
    PartProblemAssociation, PartSolutionAssociation, PartsPositionImageAssociation
)
from blueprints import DATABASE_URL
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import logging

# Create a SQLAlchemy engine using the DATABASE_URL from your config
engine = create_engine(DATABASE_URL)

# Create a session factory
Session = sessionmaker(bind=engine)

search_problem_solution_bp = Blueprint('search_problem_solution_bp', __name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@search_problem_solution_bp.route('/search_problem_solution', methods=['GET'])
def search_problem_solution():
    session = Session()
    try:
        # Retrieve parameters from the request
        description = request.args.get('problem_description', '')
        location_id = request.args.get('problem_location', None)
        asset_number_id = request.args.get('problem_asset_number', None)
        model_id = request.args.get('problem_model', None)
        problem_title = request.args.get('problem_title', '')  # Retrieve problem title

        # Logging the parameters
        logger.info(f"Description: {description}")
        logger.info(f"Location ID: {location_id}")
        logger.info(f"Asset Number ID: {asset_number_id}")
        logger.info(f"Model ID: {model_id}")
        logger.info(f"Problem Title: {problem_title}")

        # Start the query with the Position model
        query = session.query(Position)

        # Apply filters based on provided parameters
        if location_id:
            query = query.filter(Position.location_id == int(location_id))
        if asset_number_id:
            query = query.filter(Position.asset_number_id == int(asset_number_id))
        if model_id:
            query = query.filter(Position.model_id == int(model_id))

        # Retrieve positions based on the filters
        positions = query.all()
        logger.info(f"Positions found: {positions}")

        if not positions:
            # Flash message indicating no positions found
            flash("No positions found", "error")
            return jsonify(problems=[])

        # Retrieve problems associated with the found positions
        problems = []
        for position in positions:
            problem_positions = session.query(ProblemPositionAssociation).filter_by(position_id=position.id).all()
            for problem_position in problem_positions:
                problem = session.query(Problem).filter_by(id=problem_position.problem_id).first()
                if problem:
                    problems.append(problem)

        logger.info(f"Problems found: {problems}")

        if not problems:
            # Flash message indicating no problems found
            flash("No problems found", "error")
            return jsonify(problems=[])

        # Collect all parts and drawings for display at the end
        all_parts = []
        all_drawings = []

        # Construct response containing the problems and associated solutions
        response = []
        for problem in problems:
            position = problem.problem_position[0].position if problem.problem_position else None
            problem_info = {
                'id': problem.id,
                'name': problem.name,
                'description': problem.description,
                'location': position.location.name if position and position.location else None,
                'asset_number': position.asset_number.number if position and position.asset_number else None,
                'model': position.model.name if position and position.model else None,
                'solutions': [],
                'documents': [],
                'images': [],  # Prepare to collect images at the problem level
                'drawings': [],  # Prepare to collect drawings at the problem level
                'parts': []  # Prepare to collect parts at the problem level
            }

            # Collect drawings related to each problem
            drawing_problem_associations = session.query(DrawingProblemAssociation).filter_by(problem_id=problem.id).all()
            logger.info(f"Drawing associations for problem {problem.id}: {drawing_problem_associations}")
            for drawing_association in drawing_problem_associations:
                drawing = session.query(Drawing).filter_by(id=drawing_association.drawing_id).first()
                if drawing:
                    drawing_info = {
                        'id': drawing.id,
                        'number': drawing.drw_number,
                        'name': drawing.drw_name
                    }
                    problem_info['drawings'].append(drawing_info)
                    all_drawings.append(drawing_info)

            # Collect parts related to each problem
            part_problem_associations = session.query(PartProblemAssociation).filter_by(problem_id=problem.id).all()
            logger.info(f"Part associations for problem {problem.id}: {part_problem_associations}")
            for part_association in part_problem_associations:
                part = session.query(Part).filter_by(id=part_association.part_id).first()
                if part:
                    part_info = {
                        'id': part.id,
                        'number': part.part_number,
                        'name': part.name,
                        'images': []
                    }
                    problem_info['parts'].append(part_info)
                    all_parts.append(part_info)

            # Collect images and drawings directly related to each solution
            for solution in problem.solution:
                solution_info = {
                    'id': solution.id,
                    'description': solution.description,
                    'drawings': [],
                    'parts': []
                }

                # Fetch drawings related to this solution
                drawing_solution_associations = session.query(DrawingSolutionAssociation).filter_by(solution_id=solution.id).all()
                logger.info(f"Drawing associations for solution {solution.id}: {drawing_solution_associations}")
                for drawing_association in drawing_solution_associations:
                    drawing = session.query(Drawing).filter_by(id=drawing_association.drawing_id).first()
                    if drawing:
                        drawing_info = {
                            'id': drawing.id,
                            'number': drawing.drw_number,
                            'name': drawing.drw_name
                        }
                        solution_info['drawings'].append(drawing_info)
                        problem_info['drawings'].append(drawing_info)  # Add drawing to the problem level
                        all_drawings.append(drawing_info)

                # Fetch parts related to this solution
                part_solution_associations = session.query(PartSolutionAssociation).filter_by(solution_id=solution.id).all()
                logger.info(f"Part associations for solution {solution.id}: {part_solution_associations}")
                for part_association in part_solution_associations:
                    part = session.query(Part).filter_by(id=part_association.part_id).first()
                    if part:
                        part_info = {
                            'id': part.id,
                            'number': part.part_number,
                            'name': part.name,
                            'images': []
                        }
                        solution_info['parts'].append(part_info)
                        problem_info['parts'].append(part_info)  # Add part to the problem level
                        all_parts.append(part_info)

                problem_info['solutions'].append(solution_info)

                # Query and add images associated with the solution
                image_solution_associations = session.query(ImageSolutionAssociation).filter_by(solution_id=solution.id).all()
                logger.info(f"Image associations for solution {solution.id}: {image_solution_associations}")
                for association in image_solution_associations:
                    image_id = association.image_id
                    image = session.query(Image).get(image_id)
                    if image:
                        image_info = {
                            'id': image.id,
                            'title': image.title,
                            'description': image.description
                        }
                        problem_info['images'].append(image_info)  # Add images to the problem level

            # Query and add parts associated with the position
            if position:
                # Query PartsPositionImageAssociation by position_id
                part_position_images = session.query(PartsPositionImageAssociation).filter_by(
                    position_id=position.id).all()
                logger.info(f"Parts and images for position {position.id}: {part_position_images}")

                for part_pos_image in part_position_images:
                    # Query the Part table for the matching part_id
                    part = session.query(Part).get(part_pos_image.part_id)
                    if part:
                        part_info = {
                            'id': part.id,
                            'number': part.part_number,
                            'name': part.name,
                            'images': []
                        }

                        # Query the Image table for the matching image_id
                        if part_pos_image.image_id:
                            image = session.query(Image).get(part_pos_image.image_id)
                            if image:
                                image_info = {
                                    'id': image.id,
                                    'title': image.title,
                                    'description': image.description
                                }
                                part_info['images'].append(image_info)

                        # Append the part information to the result
                        problem_info['parts'].append(part_info)
                        all_parts.append(part_info)

                # Logging the parts information
                logger.info(f"Parts and images associated with position {position.id}: {problem_info['parts']}")

            # Retrieve associated documents using the CompleteDocument model
            documents = session.query(CompleteDocument).join(CompleteDocumentProblemAssociation).filter(
                CompleteDocumentProblemAssociation.problem_id == problem.id).all()
            logger.info(f"Documents for problem {problem.id}: {documents}")

            # Add associated documents
            serialized_documents = []
            for document in documents:
                serialized_document = {
                    'id': document.id,
                    'title': document.title
                }
                serialized_documents.append(serialized_document)
            problem_info['documents'] = serialized_documents

            response.append(problem_info)

        html_content = ""
        for problem_info in response:
            # Add problem and solution
            html_content += f"<h3>Problem:</h3><p>{problem_info['description']}</p>"
            html_content += "<h3>Solutions:</h3>"
            for solution in problem_info['solutions']:
                html_content += f"<p>{solution['description']}</p>"

            # Add associated documents
            html_content += "<h3>Associated Documents:</h3><ul>"
            for document in problem_info['documents']:
                # Generate the document link using url_for
                document_link = url_for('search_documents_bp.view_document', document_id=document['id'])
                html_content += f"<li><a href='{document_link}'>{document['title']}</a></li>"
            html_content += "</ul>"

            # Add associated images
            html_content += "<h3>Associated Images:</h3>"
            for image in problem_info['images']:
                # Generate the image link using url_for
                image_link = url_for('serve_image_route', image_id=image['id'])
                html_content += f"""
                    <div class="image-details">
                        <a href="{image_link}">
                            <img class="thumbnail" src="{image_link}" alt="{image['title']}">
                        </a>
                        <div class="description">
                            <h2>{image['title']}</h2>
                            <p>{image['description']}</p>
                        </div>
                        <div style="clear: both;"></div>
                    </div>
                """
            # Add associated drawings
            html_content += "<h3>Associated Drawings:</h3>"
            for drawing in problem_info['drawings']:
                html_content += f"<p>Drawing Number: {drawing['number']}</p>"
                html_content += f"<p>Drawing Name: {drawing['name']}</p>"
                # Add any other drawing information you want to display
            html_content += "<hr>"  # Add a horizontal line to separate drawings

            # Add associated parts
            html_content += "<h3>Associated Parts:</h3>"
            for part in problem_info['parts']:
                html_content += f"<p>Part Number: {part['number']}</p>"
                # Check if there are associated images for the part
                if part['images']:
                    html_content += "<h4>Part Images:</h4>"
                    for part_image in part['images']:
                        part_image_link = url_for('serve_image_route', image_id=part_image['id'])
                        html_content += f"""
                            <div class="part-image-details">
                                <a href="{part_image_link}">
                                    <img class="thumbnail" src="{part_image_link}" alt="{part_image['title']}">
                                </a>
                                <div class="description">
                                    <h3>PART NAME: {part_image['title']}</h3>
                                    <p>PART DESCRIPTION: {part_image['description']}</p>
                                </div>
                                <div style="clear: both;"></div>
                            </div>
                        """
                html_content += "<hr>"  # Add a horizontal line to separate parts

            html_content += "<hr>"  # Add a horizontal line to separate problems

        # Display all collected parts and drawings at the end
        html_content += "<h3>All Associated Parts:</h3>"
        for part in all_parts:
            html_content += f"<p>Part Number: {part['number']}, Part Name: {part['name']}</p>"

        html_content += "<h3>All Associated Drawings:</h3>"
        for drawing in all_drawings:
            html_content += f"<p>Drawing Number: {drawing['number']}, Drawing Name: {drawing['name']}</p>"

        # Return the HTML content as a response
        return html_content

    except SQLAlchemyError as e:
        logger.error("An error occurred while retrieving problems: %s", e)
        flash("An error occurred while retrieving problems: {}".format(e), "error")
        return jsonify(error=str(e)), 500

    finally:
        session.close()  # Close the session in the finally block
