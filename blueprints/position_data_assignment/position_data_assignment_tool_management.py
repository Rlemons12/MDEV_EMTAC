# modules/position_data_assignment_tool_management.py

from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify
from modules.tool_module.forms import ToolSearchForm  # Adjust the import path as needed
from modules.emtacdb.emtacdb_fts import (
    Tool, ToolCategory, ToolManufacturer,
    Position, ToolPositionAssociation)
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import logger  # Import the logger

# Define the Blueprint (Ensure this matches your actual Blueprint definition)
position_data_assignment_bp = Blueprint('position_data_assignment_bp', __name__)

# Initialize DatabaseConfig
db_config = DatabaseConfig()

@position_data_assignment_bp.route('/manage_tools', methods=['GET', 'POST'])
def manage_tools():
    logger.info("Accessed /manage_tools route.")

    # Obtain a session from DatabaseConfig
    session = db_config.get_main_session()

    try:
        # Get 'position_id' from query parameters or form data
        position_id = request.args.get('position_id', type=int) or request.form.get('position_id', type=int)
        logger.debug(f"Retrieved position_id from request: {position_id}")

        if not position_id:
            logger.error("Position ID not provided in the request.")
            flash('Position ID is required.', 'danger')
            return redirect(url_for('main_bp.home'))  # Replace with your actual main route

        # Fetch the Position object
        position = session.query(Position).get(position_id)
        if not position:
            logger.error(f"Position with ID {position_id} not found.")
            flash('Position not found.', 'danger')
            return redirect(url_for('main_bp.home'))  # Replace with your actual main route

        logger.info(f"Managing tools for Position ID {position_id}: {position}")

        # Instantiate the ToolSearchForm
        tool_search_form = ToolSearchForm()
        logger.debug("Instantiated ToolSearchForm.")

        # Populate choices for SelectMultipleFields
        tool_search_form.tool_category.choices = [
            (c.id, c.name) for c in session.query(ToolCategory).order_by(ToolCategory.name).all()
        ]
        tool_search_form.tool_manufacturer.choices = [
            (m.id, m.name) for m in session.query(ToolManufacturer).order_by(ToolManufacturer.name).all()
        ]
        logger.debug("Populated ToolSearchForm choices for categories and manufacturers.")

        # Initialize an empty list for searched tools
        searched_tools = []
        logger.info("Initialized an empty list for searched tools.")

        if request.method == 'POST':
            if tool_search_form.validate_on_submit():
                logger.info("ToolSearchForm submitted and validated successfully.")

                # Extract form data
                search_name = tool_search_form.tool_name.data
                search_size = tool_search_form.tool_size.data
                search_type = tool_search_form.tool_type.data
                search_material = tool_search_form.tool_material.data
                search_categories = tool_search_form.tool_category.data
                search_manufacturers = tool_search_form.tool_manufacturer.data

                logger.debug(
                    f"Form Data - Name: {search_name}, Size: {search_size}, Type: {search_type}, "
                    f"Material: {search_material}, Categories: {search_categories}, Manufacturers: {search_manufacturers}"
                )

                # Build the query based on search criteria
                logger.info('Building the query based on search criteria.')
                query = session.query(Tool).filter(
                    Tool.tool_position_association.any(position_id=position_id)
                )

                # Apply additional filters
                if search_name:
                    query = query.filter(Tool.name.ilike(f'%{search_name}%'))
                    logger.debug(f"Applied filter: Tool.name LIKE '%{search_name}%'")

                if search_size:
                    query = query.filter(Tool.size.ilike(f'%{search_size}%'))
                    logger.debug(f"Applied filter: Tool.size LIKE '%{search_size}%'")

                if search_type:
                    query = query.filter(Tool.type.ilike(f'%{search_type}%'))
                    logger.debug(f"Applied filter: Tool.type LIKE '%{search_type}%'")

                if search_material:
                    query = query.filter(Tool.material.ilike(f'%{search_material}%'))
                    logger.debug(f"Applied filter: Tool.material LIKE '%{search_material}%'")

                if search_categories:
                    query = query.filter(Tool.tool_category_id.in_(search_categories))
                    logger.debug(f"Applied filter: Tool.tool_category_id IN {search_categories}")

                if search_manufacturers:
                    query = query.filter(Tool.tool_manufacturer_id.in_(search_manufacturers))
                    logger.debug(f"Applied filter: Tool.tool_manufacturer_id IN {search_manufacturers}")

                # Execute the query once after all filters are applied
                searched_tools = query.all()
                logger.info(f"Query executed. Number of tools found: {len(searched_tools)}")

                if not searched_tools:
                    logger.info("No tools found matching the criteria.")
                    flash('No tools found matching the criteria.', 'info')

                # Check if the request is AJAX
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    logger.debug("Detected AJAX request. Preparing JSON response.")

                    # Serialize the tools into JSON-friendly format
                    tools_data = []
                    for tool in searched_tools:
                        tools_data.append({
                            'id': tool.id,
                            'name': tool.name,
                            'size': tool.size or 'N/A',
                            'type': tool.type or 'N/A',
                            'material': tool.material or 'N/A',
                            'tool_category': tool.tool_category.name if tool.tool_category else 'N/A',
                            'tool_manufacturer': tool.tool_manufacturer.name if tool.tool_manufacturer else 'N/A',
                            'edit_url': url_for('position_data_assignment_bp.edit_tool', tool_id=tool.id),
                            'delete_url': url_for('position_data_assignment_bp.delete_tool', tool_id=tool.id)
                        })

                    logger.debug("Returning JSON response for AJAX request.")
                    return jsonify({'tools': tools_data})

            else:
                logger.warning("ToolSearchForm validation failed.")
                flash('Please correct the errors in the form.', 'danger')

        # For GET request or non-AJAX POST request, render the full page
        return render_template(
            'position_data_assignment/position_data_assignment.html',
            tool_search_form=tool_search_form,
            searched_tools=searched_tools,
            position=position
        )

    except Exception as e:
        session.rollback()
        logger.exception(f"An exception occurred in /manage_tools route: {e}")
        flash('An unexpected error occurred.', 'error')
        return redirect(url_for('main_bp.home'))  # Replace with your actual main route

    finally:
        session.close()
        logger.debug("Database session closed.")

@position_data_assignment_bp.route('/edit_tool/<int:tool_id>', methods=['GET', 'POST'])
def edit_tool(tool_id):
    """
    Edit the details of a specific tool.
    """
    logger.info(f'Edit the details of tool with ID: {tool_id}')

    db_session = db_config.get_main_session()
    tool = db_session.query(Tool).filter_by(id=tool_id).first()
    if not tool:
        flash('Tool not found.', 'error')
        return redirect(url_for('position_data_assignment_bp.manage_tools'))  # Adjust endpoint as needed

    if request.method == 'POST':
        # Extract form data and update tool
        tool.name = request.form.get('tool_name')
        tool.size = request.form.get('tool_size')
        tool.type = request.form.get('tool_type')
        tool.material = request.form.get('tool_material')
        tool.description = request.form.get('tool_description')
        tool.tool_manufacturer_id = request.form.get('manufacturer_id')
        tool.tool_category_id = request.form.get('tool_category_id')  # Corrected attribute

        try:
            db_session.commit()
            flash('Tool updated successfully.', 'success')
            return redirect(url_for('position_data_assignment_bp.manage_tools', position_id=tool.tool_position_association[0].position_id))
        except Exception as e:
            db_session.rollback()
            flash('An error occurred while updating the tool.', 'error')
            logger.exception(f"Error updating tool with ID {tool_id}: {e}")
            return redirect(url_for('position_data_assignment_bp.manage_tools', position_id=tool.tool_position_association[0].position_id))

    # For GET request, render an edit form
    manufacturers = db_session.query(ToolManufacturer).all()
    categories = db_session.query(ToolCategory).all()

    return render_template('position_data_assignment/pda_partials/edit_tool.html', tool=tool,
                           manufacturers=manufacturers, categories=categories)
