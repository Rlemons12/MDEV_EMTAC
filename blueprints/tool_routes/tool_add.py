# blueprints/tool_routes/tool_add.py
import os
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, jsonify
from werkzeug.utils import secure_filename
from sqlalchemy.orm import joinedload  # Import for eager loading
from modules.tool_module.forms import ToolForm, ToolCategoryForm, ToolManufacturerForm, ToolSearchForm
from modules.emtacdb.emtacdb_fts import (Tool, ToolCategory, ToolManufacturer, ToolPositionAssociation, Position,
                                         Area, EquipmentGroup, Model, AssetNumber, ToolImageAssociation,
                                         Location, Subassembly, ComponentAssembly, AssemblyView, SiteLocation, Image)
from modules.configuration.log_config import logger
from modules.emtacdb.forms.position_form import PositionForm

# Initialize Blueprint
tool_blueprint_bp = Blueprint('tool_routes', __name__)

# Allowed extensions for tool images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@tool_blueprint_bp.route('/submit_tool_data', methods=['GET', 'POST'])
def submit_tool_data():
    logger.info("Accessed /submit_tool_data route.")
    logger.info("inside submit_tool_data route")

    # Access db_config
    db_config = current_app.config.get('db_config')
    if not db_config:
        error_msg = "Database configuration not found."
        logger.error(error_msg)
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'message': 'Something went wrong'}), 500
        else:
            flash(error_msg, 'danger')
            return render_template(
                'tool_templates/tool_search_entry.html',
                tool_form=None,
                category_form=None,
                manufacturer_form=None,
                position_form=None,
                search_tool_form=None,
                manufacturers=[],
                categories=[],
                positions=[]
            )

    main_session = db_config.get_main_session()

    logger.info("Instantiating forms...")
    tool_form = ToolForm()
    category_form = ToolCategoryForm()
    manufacturer_form = ToolManufacturerForm()
    position_form = PositionForm()
    tool_search_form = ToolSearchForm()
    logger.info("Forms instantiated successfully.")

    try:
        # Populate choices for various forms.
        logger.info("Populating tool_form.tool_category.choices...")
        tool_form.tool_category.choices = [
            (c.id, c.name)
            for c in main_session.query(ToolCategory).order_by(ToolCategory.name)
        ]
        logger.info("tool_form.tool_category.choices populated.")

        logger.info("Populating tool_form.tool_manufacturer.choices...")
        tool_form.tool_manufacturer.choices = [
            (m.id, m.name)
            for m in main_session.query(ToolManufacturer).order_by(ToolManufacturer.name)
        ]
        logger.info("tool_form.tool_manufacturer.choices populated.")

        logger.info("Populating category_form.parent_id.choices...")
        category_form.parent_id.choices = [(0, 'None')] + [
            (c.id, c.name)
            for c in main_session.query(ToolCategory).order_by(ToolCategory.name)
        ]
        logger.info("category_form.parent_id.choices populated.")

        # Populate PositionForm choices (similar queries for area, equipment_group, etc.)
        logger.info("Populating position_form.area.choices...")
        position_form.area.choices = [
            (a.id, a.name)
            for a in main_session.query(Area).order_by(Area.name)
        ]
        logger.info("position_form.area.choices populated.")

        # (Additional population for other position fields omitted for brevity)

        logger.info("Populating tool_search_form.tool_category.choices...")
        tool_search_form.tool_category.choices = [
            (c.id, c.name)
            for c in main_session.query(ToolCategory).order_by(ToolCategory.name)
        ]
        logger.info("tool_search_form.tool_category.choices populated.")

        logger.info("Populating tool_search_form.tool_manufacturer.choices...")
        tool_search_form.tool_manufacturer.choices = [
            (m.id, m.name)
            for m in main_session.query(ToolManufacturer).order_by(ToolManufacturer.name)
        ]
        logger.info("tool_search_form.tool_manufacturer.choices populated.")

        logger.info("Finished populating all form choices successfully.")

    except Exception as e:
        error_msg = f"Error populating form choices: {e}"
        logger.info(error_msg)
        logger.error(error_msg, exc_info=True)
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'message': error_msg}), 500
        else:
            flash(error_msg, 'danger')
            return render_template(
                'tool_templates/tool_search_entry.html',
                tool_form=tool_form,
                category_form=category_form,
                manufacturer_form=manufacturer_form,
                position_form=position_form,
                search_tool_form=tool_search_form,
                manufacturers=[],
                categories=[],
                positions=[]
            )

    is_ajax = (request.headers.get('X-Requested-With') == 'XMLHttpRequest')
    logger.debug(f"request.form data: {request.form.to_dict(flat=False)}")
    logger.info("Reached the point before checking request.method...")

    if request.method == 'POST':
        logger.info("Inside POST handling...")
        # Determine which form was submitted
        if 'submit_manufacturer' in request.form:
            form = manufacturer_form
            form_name = 'manufacturer'
            logger.info("Detected 'submit_manufacturer'")
        elif 'submit_category' in request.form:
            form = category_form
            form_name = 'category'
            logger.info("Detected category submission!")
            logger.info(f"Category Form Submitted: Name - {form.name.data}, Description - {form.description.data}")
        elif 'submit_tool' in request.form:
            form = tool_form
            form_name = 'tool'
            logger.info("Detected 'submit_tool'")
        elif 'submit_search' in request.form:
            form = tool_search_form
            form_name = 'search'
            logger.info("Detected 'submit_search'")
        else:
            form = None
            form_name = 'unknown'
            logger.info("No recognized form submission found.")

        logger.debug(f"Form submission detected: {form_name}")
        if form and form.validate_on_submit():
            logger.info(f"Form '{form_name}' validated successfully. Entering try block...")
            try:
                if form_name == 'manufacturer':
                    logger.info("Handling 'manufacturer' form logic...")
                    # ... manufacturer logic ...
                    pass

                elif form_name == 'category':
                    logger.info("Handling 'category' form logic...")
                    # ... category logic ...
                    pass

                elif form_name == 'tool':
                    logger.info("Handling 'tool' form logic now... (image upload, new Tool creation, etc.)")
                    # Process multiple file uploads from tool_images
                    uploaded_file_paths = []
                    if form.tool_images.data:
                        for file in form.tool_images.data:
                            if file and allowed_file(file.filename):
                                logger.info("File is present and allowed. Saving file...")
                                filename = secure_filename(file.filename)
                                upload_folder = os.path.join(current_app.root_path, 'static', 'uploads', 'tools')
                                os.makedirs(upload_folder, exist_ok=True)
                                file_path = os.path.join(upload_folder, filename)
                                file.save(file_path)
                                uploaded_file_path = os.path.join('uploads', 'tools', filename)
                                logger.info(f"File saved as {uploaded_file_path}")
                                uploaded_file_paths.append(uploaded_file_path)
                            else:
                                logger.warning("Skipping invalid or empty file.")

                    logger.info("Creating new_tool instance (without an `image` kwarg)...")
                    new_tool = Tool(
                        name=form.tool_name.data.strip(),
                        size=form.tool_size.data.strip() if form.tool_size.data else None,
                        type=form.tool_type.data.strip() if form.tool_type.data else None,
                        material=form.tool_material.data.strip() if form.tool_material.data else None,
                        description=form.tool_description.data.strip() if form.tool_description.data else None,
                        tool_category_id=form.tool_category.data,
                        tool_manufacturer_id=form.tool_manufacturer.data
                    )
                    logger.info(f"new_tool created: {new_tool.name}")
                    main_session.add(new_tool)

                    # Process each uploaded file: create an Image and associate via ToolImageAssociation
                    for path in uploaded_file_paths:
                        logger.info("Creating Image row for uploaded file...")
                        new_image = Image(
                            title="Main Tool Image",
                            description=form.image_description.data.strip() if form.image_description.data else "Uploaded via the tool form",
                            file_path=path
                        )
                        main_session.add(new_image)
                        main_session.flush()  # Flush to assign new_image.id without committing

                        logger.info("Creating ToolImageAssociation to link Tool & Image...")
                        tool_image_assoc = ToolImageAssociation(
                            tool=new_tool,
                            image=new_image,
                            description="Primary uploaded tool image"
                        )
                        main_session.add(tool_image_assoc)

                    # Process other associations (positions) as before...
                    logger.info("Processing selected position IDs...")
                    selected_areas = request.form.getlist('area')
                    selected_equipment_groups = request.form.getlist('equipment_group')
                    selected_models = request.form.getlist('model')
                    selected_asset_numbers = request.form.getlist('asset_number')
                    selected_locations = request.form.getlist('location')
                    selected_assemblies = request.form.getlist('subassembly')
                    selected_subassemblies = request.form.getlist('component_assembly')
                    selected_assembly_views = request.form.getlist('assembly_view')
                    selected_site_locations = request.form.getlist('site_location')

                    position_fields = [
                        ('Area', selected_areas),
                        ('Equipment Group', selected_equipment_groups),
                        ('Model', selected_models),
                        ('Asset Number', selected_asset_numbers),
                        ('Location', selected_locations),
                        ('Subassembly', selected_assemblies),
                        ('Subassembly', selected_subassemblies),
                        ('Subassembly View', selected_assembly_views),
                        ('Site Location', selected_site_locations)
                    ]

                    for category, selected_ids in position_fields:
                        for pos_id in selected_ids:
                            try:
                                pos_id_int = int(pos_id)
                            except ValueError:
                                logger.error(f"Invalid position ID: {pos_id}")
                                logger.info(f"Skipping invalid position ID: {pos_id}")
                                continue

                            position = main_session.query(Position).get(pos_id_int)
                            if not position:
                                logger.error(f"Position with ID {pos_id_int} does not exist.")
                                logger.info(f"Skipping non-existent position ID: {pos_id_int}")
                                continue

                            logger.info(f"Associating position ID {pos_id_int} ({category}) with Tool '{new_tool.name}'")
                            association = ToolPositionAssociation(
                                tool=new_tool,
                                position_id=pos_id_int,
                                description=f"{category} Description"
                            )
                            main_session.add(association)

                    logger.info("Adding new_tool and any image associations to DB...")
                    main_session.add(new_tool)
                    main_session.commit()

                    message = 'Tool added successfully with position associations and images!'
                    logger.info(message)
                    if is_ajax:
                        return jsonify({'success': True, 'message': message}), 200
                    else:
                        flash(message, 'success')
                        return redirect(url_for('tool_routes.submit_tool_data'))

                elif form_name == 'search':
                    logger.info("Handling 'search' form logic now...")
                    # ... search logic ...
                    pass

                logger.info(f"Completed {form_name} form logic without exceptions.")
            except Exception as e:
                main_session.rollback()
                error_msg = f"Error processing {form_name} form: {str(e)}"
                logger.info(error_msg)
                logger.error(error_msg, exc_info=True)
                if is_ajax:
                    return jsonify({'success': False, 'message': error_msg}), 500
                else:
                    flash(error_msg, 'danger')
                    try:
                        manufacturers = main_session.query(ToolManufacturer).order_by(ToolManufacturer.name).all()
                        categories = main_session.query(ToolCategory).order_by(ToolCategory.name).all()
                        positions = main_session.query(Position).order_by(Position.id).all()
                    except Exception as e2:
                        logger.error(f"Error fetching data during error handling: {e2}", exc_info=True)
                        manufacturers = []
                        categories = []
                        positions = []

                    return render_template(
                        'tool_templates/tool_search_entry.html',
                        tool_form=tool_form,
                        category_form=category_form,
                        manufacturer_form=manufacturer_form,
                        position_form=position_form,
                        search_tool_form=tool_search_form,
                        tools=[],
                        page=1,
                        per_page=20,
                        total_pages=0,
                        manufacturers=manufacturers,
                        categories=categories,
                        positions=positions
                    )
        else:
            logger.info("No recognized form found or request.method != 'POST'")
            error_msg = "No valid form submission detected."
            logger.error(error_msg)
            if is_ajax:
                return jsonify({'success': False, 'message': error_msg}), 400
            else:
                flash(error_msg, 'danger')
                try:
                    manufacturers = main_session.query(ToolManufacturer).order_by(ToolManufacturer.name).all()
                    categories = main_session.query(ToolCategory).order_by(ToolCategory.name).all()
                    positions = main_session.query(Position).order_by(Position.id).all()
                except Exception as e:
                    logger.error(f"Error fetching data during unknown form submission: {e}", exc_info=True)
                    manufacturers = []
                    categories = []
                    positions = []

                return render_template(
                    'tool_templates/tool_search_entry.html',
                    tool_form=tool_form,
                    category_form=category_form,
                    manufacturer_form=manufacturer_form,
                    position_form=position_form,
                    search_tool_form=tool_search_form,
                    tools=[],
                    page=1,
                    per_page=20,
                    total_pages=0,
                    manufacturers=manufacturers,
                    categories=categories,
                    positions=positions
                )

    elif request.method == 'GET' and is_ajax:
        logger.info("Handling AJAX GET request for tool search.")
        try:
            query_params = request.args.to_dict()
            page = int(query_params.get('page', 1))
            per_page = int(query_params.get('per_page', 10))
            filters = []
            if 'tool_name' in query_params and query_params['tool_name']:
                filters.append(Tool.name.ilike(f"%{query_params['tool_name']}%"))
            if 'tool_type' in query_params and query_params['tool_type']:
                filters.append(Tool.type.ilike(f"%{query_params['tool_type']}%"))
            if 'tool_category' in query_params and query_params['tool_category']:
                try:
                    tool_category_id = int(query_params['tool_category'])
                    filters.append(Tool.tool_category_id == tool_category_id)
                except ValueError:
                    logger.error(f"Invalid tool_category ID: {query_params['tool_category']}")
            if 'tool_manufacturer' in query_params and query_params['tool_manufacturer']:
                try:
                    tool_manufacturer_id = int(query_params['tool_manufacturer'])
                    filters.append(Tool.tool_manufacturer_id == tool_manufacturer_id)
                except ValueError:
                    logger.error(f"Invalid tool_manufacturer ID: {query_params['tool_manufacturer']}")
            query = main_session.query(Tool).options(
                joinedload(Tool.tool_category),
                joinedload(Tool.tool_manufacturer)
            )
            if filters:
                query = query.filter(*filters)
            total = query.count()
            tools = query.offset((page - 1) * per_page).limit(per_page).all()

            tools_data = []
            for tool in tools:
                tools_data.append({
                    'name': tool.name,
                    'size': tool.size or 'N/A',
                    'type': tool.type or 'N/A',
                    'material': tool.material or 'N/A',
                    'category': tool.tool_category.name if tool.tool_category else 'N/A',
                    'manufacturer': tool.tool_manufacturer.name if tool.tool_manufacturer else 'N/A',
                    'description': tool.description or 'N/A',
                    'image': tool.get_main_image_url() if hasattr(tool, 'get_main_image_url') else None
                })

            response = {
                'success': True,
                'tools': tools_data,
                'total': total,
                'page': page,
                'per_page': per_page,
                'total_pages': (total + per_page - 1) // per_page
            }
            logger.info(f"Search successful: {len(tools_data)} tools found.")
            return jsonify(response), 200

        except Exception as e:
            error_msg = f"Error processing AJAX search: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return jsonify({'success': False, 'message': error_msg}), 500

    return render_template(
        'tool_templates/tool_search_entry.html',
        tool_form=tool_form,
        category_form=category_form,
        manufacturer_form=manufacturer_form,
        position_form=position_form,
        search_tool_form=tool_search_form,
        manufacturers=[],
        categories=[],
        positions=[]
    )

@tool_blueprint_bp.teardown_app_request
def remove_session(exception=None):
    try:
        db_config = current_app.config.get('db_config')
        if db_config:
            main_session_registry = db_config.get_main_session_registry()
            main_session_registry.remove()
            logger.info("SQLAlchemy session removed successfully.")
        else:
            logger.warning("Database configuration not found during teardown.")
    except Exception as e:
        logger.error(f"Error removing SQLAlchemy session: {e}")

