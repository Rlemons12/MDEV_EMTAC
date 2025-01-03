#tool_add.py
import os
from modules.configuration.log_config import logger
from flask import request, render_template, redirect, url_for, flash, jsonify, current_app
from blueprints.tool_routes import tool_blueprint_bp
from modules.emtacdb.emtacdb_fts import (Position,Tool, ToolCategory, ToolManufacturer, ToolImageAssociation, Image,
                                         ToolPackage)
from werkzeug.utils import secure_filename
from modules.configuration.config_env import DatabaseConfig
from sqlalchemy.exc import SQLAlchemyError
from modules.configuration.log_config import logger
from flask import request, render_template, flash, url_for, redirect
from blueprints.tool_routes import tool_blueprint_bp
from modules.emtacdb.emtacdb_fts import (
    Position, Tool, ToolCategory, ToolManufacturer, ToolPackage
)
from modules.configuration.config_env import DatabaseConfig
from sqlalchemy.exc import SQLAlchemyError

# Database configuration
db_config = DatabaseConfig()


# Route for adding tools
@tool_blueprint_bp.route('/submit_tool_data', methods=['GET', 'POST'])
def submit_tool_data():
    logger.debug(f"Request method: {request.method} - Path: {request.path}")

    if request.method == 'POST':
        # Extract form data
        tool_name = request.form.get('tool_name')
        tool_size = request.form.get('tool_size')
        tool_type = request.form.get('tool_type')
        tool_material = request.form.get('tool_material')
        tool_description = request.form.get('tool_description')
        tool_category_id = request.form.get('tool_category')
        tool_manufacturer_id = request.form.get('tool_manufacturer')
        tool_position_id = request.form.get('tool_position')
        position_description = request.form.get('position_description')
        tool_package_id = request.form.get('tool_package')

        # Validate required fields
        missing_fields = []
        if not tool_name:
            missing_fields.append('Tool Name')
        if not tool_category_id:
            missing_fields.append('Tool Category')
        if not tool_manufacturer_id:
            missing_fields.append('Tool Manufacturer')

        if missing_fields:
            flash(f"Missing required fields: {', '.join(missing_fields)}", 'error')
            return render_template('tool_search_entry.html')

        # Database transaction
        try:
            with db_config.get_main_session() as session:
                logger.debug("Starting database transaction to add a new tool.")

                new_tool = Tool(
                    name=tool_name,
                    size=tool_size,
                    type=tool_type,
                    material=tool_material,
                    description=tool_description,
                    tool_category_id=tool_category_id,
                    tool_manufacturer_id=tool_manufacturer_id
                )

                session.add(new_tool)
                session.commit()
                logger.info("Tool data submitted successfully.")
                flash('Tool data submitted successfully!', 'success')

                return redirect(url_for('tool_routes.submit_tool_data'))

        except SQLAlchemyError as e:
            logger.error(f"Database error while submitting tool data: {e}")
            flash('A database error occurred while submitting tool data.', 'error')
            return render_template('tool_search_entry.html')

        except Exception as e:
            logger.error(f"Unexpected error while submitting tool data: {e}")
            flash('An unexpected error occurred. Please try again.', 'error')
            return render_template('tool_search_entry.html')

    # GET Request: Render the tool form
    try:
        with db_config.get_main_session() as session:
            logger.debug("Loading tool form data from the database.")

            tool_categories = session.query(ToolCategory).all()
            tool_manufacturers = session.query(ToolManufacturer).all()
            tool_positions = session.query(Position).all()
            tool_packages = session.query(ToolPackage).all()

        return render_template(
            'tool_search_entry.html',
            tool_categories=tool_categories,
            tool_manufacturers=tool_manufacturers,
            tool_positions=tool_positions,
            tool_packages=tool_packages
        )

    except SQLAlchemyError as e:
        logger.error(f"Database error while loading form data: {e}")
        flash('An error occurred while loading the form. Please try again.', 'error')
        return render_template('tool_search_entry.html')

    except Exception as e:
        logger.error(f"Unexpected error while loading form data: {e}")
        flash('An unexpected error occurred while loading the form.', 'error')
        return render_template('tool_search_entry.html')
