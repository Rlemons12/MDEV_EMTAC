#tool_get_data.py
from flask import request, jsonify
from sqlalchemy.exc import SQLAlchemyError
from blueprints.tool_routes import tool_blueprint_bp
from modules.emtacdb.emtacdb_fts import Position, ToolPackage, ToolManufacturer
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import logger  # Import the logger

db_config = DatabaseConfig()

@tool_blueprint_bp.route('/get_tool_positions', methods=['GET'])
def get_tool_positions():
    try:
        with db_config.MainSession() as session:
            positions = session.query(Position).all()  # Assuming `Position` is a defined model
            # Convert data to JSON format
            position_data = [{'id': position.id, 'name': position.name} for position in positions]
            return jsonify(position_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@tool_blueprint_bp.route('/get_tool_packages', methods=['GET'])
def get_tool_packages():
    try:
        with db_config.MainSession() as session:
            packages = session.query(ToolPackage).all()
            # Convert data to JSON format
            package_data = [{'id': package.id, 'name': package.name} for package in packages]
            return jsonify(package_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@tool_blueprint_bp.route('/get_tool_manufacturers', methods=['GET'])
def get_tool_manufacturer():
    """
    Route to fetch all tool manufacturers.
    Returns:
        JSON response containing a list of tool manufacturers with their IDs and names.
    """
    try:
        with db_config.MainSession() as session:
            # Query all ToolManufacturer records
            manufacturers = session.query(ToolManufacturer).all()
            manufacturers_data = [
                {'id': manufacturer.id, 'name': manufacturer.name}
                for manufacturer in manufacturers
            ]
            # Return the serialized data as JSON with a 200 OK status
            return jsonify(manufacturers_data), 200

    except SQLAlchemyError as e:
        # Log the database error
        logger.error(f"Database error occurred: {str(e)}")

        # Return a JSON error response with a 500 Internal Server Error status
        return jsonify({'error': 'Database error occurred while fetching tool manufacturers.'}), 500

    except Exception as e:
        # Log any unexpected errors
        logger.error(f"Unexpected error occurred: {str(e)}")

        # Return a JSON error response with a 500 Internal Server Error status
        return jsonify({'error': 'An unexpected error occurred while fetching tool manufacturers.'}), 500

@tool_blueprint_bp.route('/get_tool_categories', methods=['GET'])
def get_tool_categories():
    """
    Route to fetch all tool categories.
    Optional Query Parameters:
        - page: The page number for pagination (default: 1)
        - per_page: Number of items per page (default: 10)
    Returns:
        JSON response containing a list of tool categories with their details.
    """
    try:
        # Handle pagination parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))

        # Create a new database session
        with db_config.MainSession() as session:
            # Query all ToolCategory records
            query = session.query(ToolCategory)
            total = query.count()
            categories = query.offset((page - 1) * per_page).limit(per_page).all()

            # Serialize the data into a list of dictionaries
            categories_data = [
                {
                    'id': category.id,
                    'name': category.name,
                    'description': category.description,
                    'parent_id': category.parent_id,
                    'subcategories': [
                        {'id': sub.id, 'name': sub.name}
                        for sub in category.subcategories
                    ]
                }
                for category in categories
            ]

            # Log successful retrieval
            logger.info(f"Fetched {len(categories_data)} tool categories successfully (Page {page}).")

            # Return the serialized data as JSON with pagination info
            return jsonify({
                'total': total,
                'page': page,
                'per_page': per_page,
                'categories': categories_data
            }), 200

    except ValueError:
        # Log invalid pagination parameters
        logger.error("Invalid pagination parameters provided.")
        return jsonify({'error': 'Invalid pagination parameters.'}), 400

    except SQLAlchemyError as e:
        # Log the database error
        logger.error(f"Database error occurred while fetching tool categories: {str(e)}")

        # Return a JSON error response with a 500 Internal Server Error status
        return jsonify({'error': 'Database error occurred while fetching tool categories.'}), 500

    except Exception as e:
        # Log any unexpected errors
        logger.error(f"Unexpected error occurred while fetching tool categories: {str(e)}")

        # Return a JSON error response with a 500 Internal Server Error status
        return jsonify({'error': 'An unexpected error occurred while fetching tool categories.'}), 500

