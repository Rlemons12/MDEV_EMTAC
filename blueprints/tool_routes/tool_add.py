from modules.configuration.log_config import logger
from flask import request, render_template, redirect, url_for, flash, jsonify, current_app
from blueprints.tool_routes import tool_blueprint_bp
from modules.emtacdb.emtacdb_fts import Tool, ToolCategory, ToolManufacturer, ToolImageAssociation, Image
from werkzeug.utils import secure_filename
import os
from modules.configuration.config_env import DatabaseConfig
from sqlalchemy.exc import SQLAlchemyError

db_config = DatabaseConfig()

# Define a route for adding tools
@tool_bp.route('/submit_tool_data', methods=['GET', 'POST'])
def submit_tool_data():
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
        # Handle file upload if necessary

        # Validate required fields
        if not tool_name or not tool_category_id or not tool_manufacturer_id:
            flash('Please fill in all required fields.', 'error')
            return redirect(url_for('tool_bp.submit_tool_data'))

        try:
            with db_config.MainSession() as session:
                # Create a new Tool instance
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
                flash('Tool data submitted successfully!', 'success')
                return redirect(url_for('tool_bp.submit_tool_data'))

        except SQLAlchemyError as e:
            logger.error(f"Database error while submitting tool data: {e}")
            session.rollback()
            flash('An internal server error occurred while submitting tool data.', 'error')
            return redirect(url_for('tool_bp.submit_tool_data'))

        except Exception as e:
            logger.error(f"Unexpected error while submitting tool data: {e}")
            session.rollback()
            flash('An unexpected error occurred while submitting tool data.', 'error')
            return redirect(url_for('tool_bp.submit_tool_data'))

    else:
        # Handle GET request - render the form
        try:
            with db_config.MainSession() as session:
                # Fetch all tool categories
                tool_categories = session.query(ToolCategory).all()
                # Fetch other data like manufacturers, positions, packages
                tool_manufacturers = session.query(ToolManufacturer).all()
                tool_positions = session.query(Position).all()
                tool_packages = session.query(ToolPackage).all()

            return render_template(
                'tool_data_entry.html',
                tool_categories=tool_categories,
                tool_manufacturers=tool_manufacturers,
                tool_positions=tool_positions,
                tool_packages=tool_packages
            )

        except SQLAlchemyError as e:
            logger.error(f"Database error while loading the form: {e}")
            flash('An internal server error occurred while loading the form.', 'error')
            return redirect(url_for('tool_bp.submit_tool_data'))

        except Exception as e:
            logger.error(f"Unexpected error while loading the form: {e}")
            flash('An unexpected error occurred while loading the form.', 'error')
            return redirect(url_for('tool_bp.submit_tool_data'))