# blueprints/tool_routes.py

from flask import Blueprint, jsonify, request, redirect, url_for, flash, render_template
from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import (
    ToolCategory,
    ToolManufacturer,
    Position,
    ToolPackage,
    Tool
)
from sqlalchemy.exc import SQLAlchemyError
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tool_bp = Blueprint('tool_bp', __name__)

db_config = DatabaseConfig()


@tool_bp.route('/get_tool_categories', methods=['GET'])
def get_tool_categories():
    try:
        with db_config.MainSession() as session:
            categories = session.query(ToolCategory).filter_by(parent_id=None).all()

            def serialize_category(category):
                return {
                    'id': category.id,
                    'name': category.name,
                    'description': category.description,
                    'subcategories': [serialize_category(sub) for sub in category.subcategories]
                }

            serialized_categories = [serialize_category(cat) for cat in categories]

        return jsonify({'categories': serialized_categories}), 200

    except SQLAlchemyError as e:
        logger.error(f"Database error while fetching categories: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500

    except Exception as e:
        logger.error(f"Unexpected error while fetching categories: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500


@tool_bp.route('/get_tool_manufacturers', methods=['GET'])
def get_tool_manufacturers():
    try:
        with db_config.MainSession() as session:
            manufacturers = session.query(ToolManufacturer).all()
            serialized_manufacturers = [
                {
                    'id': manufacturer.id,
                    'name': manufacturer.name,
                    'country': manufacturer.country,
                    'website': manufacturer.website
                }
                for manufacturer in manufacturers
            ]

        return jsonify({'manufacturers': serialized_manufacturers}), 200

    except SQLAlchemyError as e:
        logger.error(f"Database error while fetching manufacturers: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500

    except Exception as e:
        logger.error(f"Unexpected error while fetching manufacturers: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500


@tool_bp.route('/get_tool_positions', methods=['GET'])
def get_tool_positions():
    try:
        with db_config.MainSession() as session:
            positions = session.query(Position).all()
            serialized_positions = [
                {
                    'id': position.id,
                    'name': position.name,
                    'description': position.description
                }
                for position in positions
            ]

        return jsonify({'positions': serialized_positions}), 200

    except SQLAlchemyError as e:
        logger.error(f"Database error while fetching positions: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500

    except Exception as e:
        logger.error(f"Unexpected error while fetching positions: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500


@tool_bp.route('/get_tool_packages', methods=['GET'])
def get_tool_packages():
    try:
        with db_config.MainSession() as session:
            packages = session.query(ToolPackage).all()
            serialized_packages = [
                {
                    'id': package.id,
                    'name': package.name,
                    'description': package.description
                }
                for package in packages
            ]

        return jsonify({'packages': serialized_packages}), 200

    except SQLAlchemyError as e:
        logger.error(f"Database error while fetching packages: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500

    except Exception as e:
        logger.error(f"Unexpected error while fetching packages: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500