# tool_search.py

from flask import request, jsonify
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import joinedload
from blueprints.tool_routes import tool_blueprint_bp
from modules.emtacdb.emtacdb_fts import Tool, ToolImageAssociation, Image
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import logger


db_config = DatabaseConfig()

@tool_blueprint_bp.route('/tools_search', methods=['GET'])
def tools_search():
    """
    Endpoint to search tools based on query parameters, including associated images.
    Implements pagination and optimized querying.
    """
    try:
        # Extract query parameters
        name = request.args.get('name', type=str, default=None)
        category_id = request.args.get('category_id', type=int, default=None)
        manufacturer_id = request.args.get('manufacturer_id', type=int, default=None)
        page = request.args.get('page', type=int, default=1)
        per_page = request.args.get('per_page', type=int, default=10)

        logger.debug(f"Search parameters - Name: {name}, Category ID: {category_id}, "
                     f"Manufacturer ID: {manufacturer_id}, Page: {page}, Per Page: {per_page}")

        with db_config.get_main_session() as session:
            # Build the base query with eager loading to prevent N+1 queries
            query = session.query(Tool).options(
                joinedload(Tool.tool_category),
                joinedload(Tool.tool_manufacturer),
                joinedload(Tool.tool_image_association).joinedload(ToolImageAssociation.image)
            )

            # Apply filters based on query parameters
            if name:
                query = query.filter(Tool.name.ilike(f'%{name}%'))
            if category_id:
                query = query.filter(Tool.tool_category_id == category_id)
            if manufacturer_id:
                query = query.filter(Tool.tool_manufacturer_id == manufacturer_id)

            # Implement pagination
            total = query.count()
            tools = query.offset((page - 1) * per_page).limit(per_page).all()

            logger.info(f"Found {total} tools matching the criteria. Returning page {page} with {len(tools)} tools.")

            # Prepare response data
            tool_data = []
            for tool in tools:
                images = [
                    {
                        'id': assoc.image.id,
                        'title': assoc.image.title,
                        'description': assoc.image.description,
                        'file_path': assoc.image.file_path,
                    }
                    for assoc in tool.tool_image_association if assoc.image
                ]
                tool_data.append({
                    'id': tool.id,
                    'name': tool.name,
                    'size': tool.size,
                    'type': tool.type,
                    'material': tool.material,
                    'description': tool.description,
                    'category': tool.tool_category.name if tool.tool_category else None,
                    'manufacturer': tool.tool_manufacturer.name if tool.tool_manufacturer else None,
                    'images': images,
                })

            # Return paginated response
            response = {
                'total': total,
                'page': page,
                'per_page': per_page,
                'tools': tool_data
            }

            return jsonify(response), 200

    except SQLAlchemyError as e:
        # Handle database errors
        logger.error(f"Database error occurred during tool search: {e}")
        return jsonify({"error": "Database error occurred.", "details": str(e)}), 500

    except Exception as e:
        # Handle generic errors
        logger.error(f"Unexpected error occurred during tool search: {e}")
        return jsonify({"error": "An unexpected error occurred.", "details": str(e)}), 500
