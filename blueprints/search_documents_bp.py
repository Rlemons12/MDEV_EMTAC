from flask import Blueprint, jsonify, request, send_file, flash
from sqlalchemy.orm import joinedload
import os
from modules.configuration.config import DATABASE_DIR
from modules.emtacdb.emtacdb_fts import CompleteDocument, Position, CompletedDocumentPositionAssociation
from modules.configuration.log_config import logger
from modules.configuration.config_env import DatabaseConfig

# Initialize DatabaseConfig
db_config = DatabaseConfig()

search_documents_bp = Blueprint('search_documents_bp', __name__)

@search_documents_bp.route('/view_document/<int:document_id>')
def view_document(document_id):
    logger.debug("Inside view_document route")

    # Create an SQLAlchemy session
    session = db_config.get_main_session()

    try:
        # Fetch the document from the database based on the ID
        document = session.query(CompleteDocument).get(document_id)

        if document:
            logger.info(f"Found document with ID {document_id}")
            file_path = os.path.join(DATABASE_DIR, document.file_path)
            logger.info(f"File path: {file_path}")
            if os.path.exists(file_path):
                logger.info("File exists. Serving the document.")
                return send_file(file_path, as_attachment=True)
            else:
                logger.error("File not found.")
                return "File not found", 404
        else:
            logger.info("Document not found")
            return "Document not found", 404
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()

# Define the search_documents route within the blueprint
@search_documents_bp.route('/', methods=['GET'])
def search_documents():
    with db_config.get_main_session() as session:
        # Retrieve search parameters
        title = request.args.get('title', '')
        area = request.args.get('searchdocument_area', '')
        equipment_group = request.args.get('searchdocument_equipmentgroup', '')
        model = request.args.get('searchdocument_model', '')
        asset_number = request.args.get('searchdocument_asset_number', '')
        location = request.args.get('searchdocument_location', '')

        logger.info(f"Search parameters - Title: {title}, Area: {area}, Equipment Group: {equipment_group}, Model: {model}, Asset Number: {asset_number}, Location: {location}")

        # Perform search using the updated function
        result = search_documents_db(session, title=title, area=area, equipment_group=equipment_group, model=model, asset_number=asset_number, location=location)

        if 'documents' in result:
            return jsonify(result)
        else:
            return jsonify({'error': 'No documents found'}), 404

def search_documents_db(session, title='', area='', equipment_group='', model='', asset_number='', location=''):
    logger.info("Starting search_documents_db")
    logger.info(f"Search parameters - title: {title}, area: {area}, equipment_group: {equipment_group}, model: {model}, asset_number: {asset_number}, location: {location}")

    try:
        # Create the base query with explicit join using .select_from and .join
        query = (
            session.query(CompleteDocument)
            .select_from(CompleteDocument)
            .join(CompletedDocumentPositionAssociation, CompleteDocument.id == CompletedDocumentPositionAssociation.complete_document_id)
            .join(Position, CompletedDocumentPositionAssociation.position_id == Position.id)
            .options(joinedload(CompleteDocument.completed_document_position_association).joinedload(CompletedDocumentPositionAssociation.position))
        )

        # Apply filters based on provided parameters
        if title:
            query = query.filter(CompleteDocument.title.ilike(f'%{title}%'))
        if area:
            query = query.filter(Position.area_id == int(area))
        if equipment_group:
            query = query.filter(Position.equipment_group_id == int(equipment_group))
        if model:
            query = query.filter(Position.model_id == int(model))
        if asset_number:
            query = query.filter(Position.asset_number_id == int(asset_number))
        if location:
            query = query.filter(Position.location_id == int(location))

        results = query.all()

        # Convert the results to a list of dictionaries for JSON response
        documents = [
            {
                'id': doc.id,
                'title': doc.title,
            }
            for doc in results
        ]

        logger.info(f"Found {len(documents)} documents matching the criteria")
        return {"documents": documents}
    except Exception as e:
        logger.error(f"An error occurred while searching documents: {e}")
        return {"error": str(e)}
