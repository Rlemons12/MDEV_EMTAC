from flask import Blueprint, jsonify, request, flash
from modules.emtacdb.emtacdb_fts import Image
from modules.configuration.log_config import logger
from modules.configuration.config_env import DatabaseConfig

db_config = DatabaseConfig()

search_images_bp = Blueprint('search_images_bp', __name__)


@search_images_bp.route('/serve_image/<int:image_id>')
def serve_image_route(image_id):
    """
    Route to serve an image file by ID using the Image.serve_file class method.
    """
    logger.debug(f"Request to serve image with ID: {image_id}")

    try:
        # Use the Image.serve_file class method
        success, response, status_code = Image.serve_file(image_id)

        if success:
            return response
        else:
            logger.error(f"Failed to serve image {image_id}: {response}")
            flash(f"Error serving image {image_id}", "error")
            return response, status_code

    except Exception as e:
        logger.error(f"Error serving image {image_id}: {e}")
        flash(f"Error serving image {image_id}", "error")
        return "Internal Server Error", 500


@search_images_bp.route('/', methods=['GET'])
def search_images():
    """
    Route to search for images using the Image.search_images class method.
    """
    with db_config.get_main_session() as session:
        try:
            # Extract search parameters from request
            search_params = _extract_search_parameters(request)

            logger.debug(f"Search parameters: {search_params}")

            # Use the Image.search_images class method
            images = Image.search_images(session, **search_params)

            if images:
                logger.info(f"Found {len(images)} images matching the criteria")
                return jsonify({
                    "images": images,
                    "count": len(images)
                })
            else:
                logger.info("No images found matching the criteria")
                return jsonify({
                    "images": [],
                    "count": 0,
                    "message": "No images found matching the search criteria"
                })

        except Exception as e:
            logger.error(f"Error in search_images route: {e}")
            return jsonify({
                "error": f"An error occurred while searching images: {str(e)}"
            }), 500


def _extract_search_parameters(request):
    """
    Helper function to extract and convert search parameters from Flask request.

    Args:
        request: Flask request object

    Returns:
        Dictionary of search parameters for Image.search_images
    """
    params = {}

    # Text-based searches
    description = request.args.get('description', '').strip()
    if description:
        params['description'] = description

    title = request.args.get('title', '').strip()
    if title:
        params['title'] = title

    # Direct ID searches
    position_id = request.args.get('position_id')
    if position_id:
        try:
            params['position_id'] = int(position_id)
        except ValueError:
            logger.warning(f"Invalid position_id: {position_id}")

    tool_id = request.args.get('tool_id')
    if tool_id:
        try:
            params['tool_id'] = int(tool_id)
        except ValueError:
            logger.warning(f"Invalid tool_id: {tool_id}")

    task_id = request.args.get('task_id')
    if task_id:
        try:
            params['task_id'] = int(task_id)
        except ValueError:
            logger.warning(f"Invalid task_id: {task_id}")

    problem_id = request.args.get('problem_id')
    if problem_id:
        try:
            params['problem_id'] = int(problem_id)
        except ValueError:
            logger.warning(f"Invalid problem_id: {problem_id}")

    completed_document_id = request.args.get('completed_document_id')
    if completed_document_id:
        try:
            params['completed_document_id'] = int(completed_document_id)
        except ValueError:
            logger.warning(f"Invalid completed_document_id: {completed_document_id}")

    # Hierarchy-based searches (supporting both old and new parameter names)
    area_id = request.args.get('area_id') or request.args.get('searchimage_area') or request.args.get('area')
    if area_id:
        try:
            params['area_id'] = int(area_id)
        except ValueError:
            logger.warning(f"Invalid area_id: {area_id}")

    equipment_group_id = (request.args.get('equipment_group_id') or
                          request.args.get('searchimage_equipment_group') or
                          request.args.get('equipment_group'))
    if equipment_group_id:
        try:
            params['equipment_group_id'] = int(equipment_group_id)
        except ValueError:
            logger.warning(f"Invalid equipment_group_id: {equipment_group_id}")

    model_id = request.args.get('model_id') or request.args.get('searchimage_model') or request.args.get('model')
    if model_id:
        try:
            params['model_id'] = int(model_id)
        except ValueError:
            logger.warning(f"Invalid model_id: {model_id}")

    asset_number_id = (request.args.get('asset_number_id') or
                       request.args.get('searchimage_asset_number') or
                       request.args.get('asset_number'))
    if asset_number_id:
        try:
            params['asset_number_id'] = int(asset_number_id)
        except ValueError:
            logger.warning(f"Invalid asset_number_id: {asset_number_id}")

    location_id = (request.args.get('location_id') or
                   request.args.get('searchimage_location') or
                   request.args.get('location'))
    if location_id:
        try:
            params['location_id'] = int(location_id)
        except ValueError:
            logger.warning(f"Invalid location_id: {location_id}")

    subassembly_id = request.args.get('subassembly_id')
    if subassembly_id:
        try:
            params['subassembly_id'] = int(subassembly_id)
        except ValueError:
            logger.warning(f"Invalid subassembly_id: {subassembly_id}")

    component_assembly_id = request.args.get('component_assembly_id')
    if component_assembly_id:
        try:
            params['component_assembly_id'] = int(component_assembly_id)
        except ValueError:
            logger.warning(f"Invalid component_assembly_id: {component_assembly_id}")

    assembly_view_id = request.args.get('assembly_view_id')
    if assembly_view_id:
        try:
            params['assembly_view_id'] = int(assembly_view_id)
        except ValueError:
            logger.warning(f"Invalid assembly_view_id: {assembly_view_id}")

    site_location_id = request.args.get('site_location_id')
    if site_location_id:
        try:
            params['site_location_id'] = int(site_location_id)
        except ValueError:
            logger.warning(f"Invalid site_location_id: {site_location_id}")

    # Limit parameter
    limit = request.args.get('limit', '50')
    try:
        params['limit'] = int(limit)
        # Ensure reasonable limits
        if params['limit'] > 1000:
            params['limit'] = 1000
        elif params['limit'] < 1:
            params['limit'] = 50
    except ValueError:
        logger.warning(f"Invalid limit: {limit}, using default 50")
        params['limit'] = 50

    return params


# Additional utility routes that might be useful

@search_images_bp.route('/image/<int:image_id>/details', methods=['GET'])
def get_image_details(image_id):
    """
    Route to get detailed information about a specific image including all associations.
    """
    with db_config.get_main_session() as session:
        try:
            # Search for the specific image
            images = Image.search_images(session, limit=1)

            # Filter by ID (since search_images doesn't have direct ID filter)
            image = session.query(Image).filter_by(id=image_id).first()

            if image:
                # Get associations using the helper method
                associations = Image._get_image_associations(session, image_id)

                image_details = {
                    "id": image.id,
                    "title": image.title,
                    "description": image.description,
                    "file_path": image.file_path,
                    "associations": associations
                }

                return jsonify(image_details)
            else:
                return jsonify({"error": "Image not found"}), 404

        except Exception as e:
            logger.error(f"Error getting image details for ID {image_id}: {e}")
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@search_images_bp.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint.
    """
    return jsonify({
        "status": "healthy",
        "service": "search_images_bp"
    })