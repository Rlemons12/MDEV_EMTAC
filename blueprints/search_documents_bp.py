from flask import Blueprint, jsonify, request, flash
from modules.emtacdb.emtacdb_fts import CompleteDocument
from modules.configuration.log_config import logger
from modules.configuration.config_env import DatabaseConfig

# Initialize DatabaseConfig
db_config = DatabaseConfig()

search_documents_bp = Blueprint('search_documents_bp', __name__)


@search_documents_bp.route('/view_document/<int:document_id>')
def view_document(document_id):
    """
    Route to serve a document file by ID using the CompleteDocument.serve_file class method.
    """
    logger.debug(f"Request to serve document with ID: {document_id}")

    try:
        # Use the CompleteDocument.serve_file class method
        success, response, status_code = CompleteDocument.serve_file(document_id)

        if success:
            return response
        else:
            logger.error(f"Failed to serve document {document_id}: {response}")
            flash(f"Error serving document {document_id}", "error")
            return response, status_code

    except Exception as e:
        logger.error(f"Error serving document {document_id}: {e}")
        flash(f"Error serving document {document_id}", "error")
        return "Internal Server Error", 500


@search_documents_bp.route('/', methods=['GET'])
def search_documents():
    """
    Route to search for documents using the CompleteDocument.search_documents_enhanced class method.
    """
    with db_config.main_session() as session:
        try:
            # Extract search parameters from request
            search_params = _extract_document_search_parameters(request)

            logger.debug(f"Document search parameters: {search_params}")

            # Use the CompleteDocument.search_documents_enhanced class method
            documents = CompleteDocument.search_documents_enhanced(session, **search_params)

            if documents:
                logger.info(f"Found {len(documents)} documents matching the criteria")
                return jsonify({
                    "documents": documents,
                    "count": len(documents)
                })
            else:
                logger.info("No documents found matching the criteria")
                return jsonify({
                    "documents": [],
                    "count": 0,
                    "message": "No documents found matching the search criteria"
                })

        except Exception as e:
            logger.error(f"Error in search_documents route: {e}")
            return jsonify({
                "error": f"An error occurred while searching documents: {str(e)}"
            }), 500


def _extract_document_search_parameters(request):
    """
    Helper function to extract and convert search parameters from Flask request.

    Args:
        request: Flask request object

    Returns:
        Dictionary of search parameters for CompleteDocument.search_documents_enhanced
    """
    params = {}

    # Text-based searches
    title = request.args.get('title', '').strip()
    if title:
        params['title'] = title

    content = request.args.get('content', '').strip()
    if content:
        params['content'] = content

    file_path = request.args.get('file_path', '').strip()
    if file_path:
        params['file_path'] = file_path

    # Direct ID searches
    position_id = request.args.get('position_id')
    if position_id:
        try:
            params['position_id'] = int(position_id)
        except ValueError:
            logger.warning(f"Invalid position_id: {position_id}")

    # Hierarchy-based searches (supporting both old and new parameter names)
    area_id = request.args.get('area_id') or request.args.get('searchdocument_area') or request.args.get('area')
    if area_id:
        try:
            params['area_id'] = int(area_id)
        except ValueError:
            logger.warning(f"Invalid area_id: {area_id}")

    equipment_group_id = (request.args.get('equipment_group_id') or
                          request.args.get('searchdocument_equipmentgroup') or
                          request.args.get('equipment_group'))
    if equipment_group_id:
        try:
            params['equipment_group_id'] = int(equipment_group_id)
        except ValueError:
            logger.warning(f"Invalid equipment_group_id: {equipment_group_id}")

    model_id = request.args.get('model_id') or request.args.get('searchdocument_model') or request.args.get('model')
    if model_id:
        try:
            params['model_id'] = int(model_id)
        except ValueError:
            logger.warning(f"Invalid model_id: {model_id}")

    asset_number_id = (request.args.get('asset_number_id') or
                       request.args.get('searchdocument_asset_number') or
                       request.args.get('asset_number'))
    if asset_number_id:
        try:
            params['asset_number_id'] = int(asset_number_id)
        except ValueError:
            logger.warning(f"Invalid asset_number_id: {asset_number_id}")

    location_id = (request.args.get('location_id') or
                   request.args.get('searchdocument_location') or
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


# Additional utility routes

@search_documents_bp.route('/document/<int:document_id>/details', methods=['GET'])
def get_document_details(document_id):
    """
    Route to get detailed information about a specific document.
    """
    with db_config.main_session() as session:
        try:
            # Get the specific document
            document = session.query(CompleteDocument).filter_by(id=document_id).first()

            if document:
                document_details = {
                    "id": document.id,
                    "title": document.title,
                    "file_path": document.file_path,
                    "rev": document.rev,
                    "content": document.content
                }

                # Check if file exists
                if document.file_path:
                    import os
                    from modules.configuration.config import DATABASE_DIR

                    potential_paths = [
                        document.file_path if os.path.isabs(document.file_path) else None,
                        os.path.join(os.getcwd(), document.file_path),
                        os.path.join(DATABASE_DIR, document.file_path),
                        os.path.join(os.getcwd(), "uploads", os.path.basename(document.file_path))
                    ]

                    file_exists = any(os.path.exists(path) for path in potential_paths if path)
                else:
                    file_exists = False

                document_details["file_exists"] = file_exists

                return jsonify(document_details)
            else:
                return jsonify({"error": "Document not found"}), 404

        except Exception as e:
            logger.error(f"Error getting document details for ID {document_id}: {e}")
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@search_documents_bp.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint.
    """
    return jsonify({
        "status": "healthy",
        "service": "search_documents_bp"
    })