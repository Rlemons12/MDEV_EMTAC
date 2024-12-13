from flask import request, jsonify
from sqlalchemy.exc import SQLAlchemyError
from blueprints.tool_routes import tool_blueprint_bp
from modules.emtacdb.emtacdb_fts import Tool, ToolImageAssociation, Image
from modules.configuration.config_env import DatabaseConfig


db_config = DatabaseConfig()

@tool_blueprint_bp.route('/tool_search', methods=['GET'])
def tool_search():
    """
    Endpoint to search tools based on query parameters, including associated images.
    """
    try:
        with db_config.MainSession() as session:
            # Get query parameters
            name = request.args.get('name')
            category_id = request.args.get('category_id')
            manufacturer_id = request.args.get('manufacturer_id')

            # Query tools
            query = session.query(Tool).outerjoin(ToolImageAssociation).outerjoin(Image)

            if name:
                query = query.filter(Tool.name.ilike(f'%{name}%'))
            if category_id:
                query = query.filter(Tool.tool_category_id == category_id)
            if manufacturer_id:
                query = query.filter(Tool.tool_manufacturer_id == manufacturer_id)

            tools = query.all()

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

            return jsonify(tool_data)

    except SQLAlchemyError as e:
        # Handle database errors
        return jsonify({"error": "Database error occurred.", "details": str(e)}), 500

    except Exception as e:
        # Handle generic errors
        return jsonify({"error": "An unexpected error occurred.", "details": str(e)}), 500

