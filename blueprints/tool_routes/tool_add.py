# blueprints/tool_routes/tool_add.py

import os
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, jsonify
from werkzeug.utils import secure_filename
from modules.tool_module.forms import ToolForm, ToolCategoryForm, ToolManufacturerForm, ToolSearchForm
from modules.emtacdb.emtacdb_fts import (Tool, ToolCategory, ToolManufacturer, ToolPositionAssociation, Position,
                                        Area,EquipmentGroup,Model,AssetNumber,
                                        Location,Assembly,SubAssembly,AssemblyView,SiteLocation )
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
    print(f'inside submit_tool_data route')
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
    print(f'Instantiate forms')
    # Instantiate forms
    tool_form = ToolForm()
    category_form = ToolCategoryForm()
    manufacturer_form = ToolManufacturerForm()
    position_form = PositionForm()
    tool_search_form = ToolSearchForm()

    print("Instantiating forms...")
    tool_form = ToolForm()
    category_form = ToolCategoryForm()
    manufacturer_form = ToolManufacturerForm()
    position_form = PositionForm()
    tool_search_form = ToolSearchForm()
    print("Forms instantiated successfully.")

    try:
        # For the tool form
        print("Populating tool_form.tool_category.choices...")
        tool_form.tool_category.choices = [
            (c.id, c.name)
            for c in main_session.query(ToolCategory).order_by(ToolCategory.name)
        ]
        print("tool_form.tool_category.choices populated.")

        print("Populating tool_form.tool_manufacturer.choices...")
        tool_form.tool_manufacturer.choices = [
            (m.id, m.name)
            for m in main_session.query(ToolManufacturer).order_by(ToolManufacturer.name)
        ]
        print("tool_form.tool_manufacturer.choices populated.")

        # For the category form
        print("Populating category_form.parent_id.choices...")
        category_form.parent_id.choices = [(0, 'None')] + [
            (c.id, c.name)
            for c in main_session.query(ToolCategory).order_by(ToolCategory.name)
        ]
        print("category_form.parent_id.choices populated.")

        # ----- Populate PositionForm from separate tables ------
        # For "area" -> from the 'Area' table
        print("Populating position_form.area.choices...")
        position_form.area.choices = [
            (a.id, a.name)
            for a in main_session.query(Area).order_by(Area.name)
        ]
        print("position_form.area.choices populated.")

        print("Populating position_form.equipment_group.choices...")
        position_form.equipment_group.choices = [
            (eg.id, eg.name)
            for eg in main_session.query(EquipmentGroup).order_by(EquipmentGroup.name)
        ]
        print("position_form.equipment_group.choices populated.")

        print("Populating position_form.model.choices...")
        position_form.model.choices = [
            (md.id, md.name)
            for md in main_session.query(Model).order_by(Model.name)
        ]
        print("position_form.model.choices populated.")

        print("Populating position_form.asset_number.choices...")
        position_form.asset_number.choices = [
            (an.id, an.number)
            for an in main_session.query(AssetNumber).order_by(AssetNumber.number)
        ]
        print("position_form.asset_number.choices populated.")

        print("Populating position_form.location.choices...")
        position_form.location.choices = [
            (loc.id, loc.name)
            for loc in main_session.query(Location).order_by(Location.name)
        ]
        print("position_form.location.choices populated.")

        print("Populating position_form.assembly.choices...")
        position_form.assembly.choices = [
            (asm.id, asm.name)
            for asm in main_session.query(Assembly).order_by(Assembly.name)
        ]
        print("position_form.assembly.choices populated.")

        print("Populating position_form.subassembly.choices...")
        position_form.subassembly.choices = [
            (sasm.id, sasm.name)
            for sasm in main_session.query(SubAssembly).order_by(SubAssembly.name)
        ]
        print("position_form.subassembly.choices populated.")

        print("Populating position_form.assembly_view.choices...")
        position_form.assembly_view.choices = [
            (av.id, av.name)
            for av in main_session.query(AssemblyView).order_by(AssemblyView.name)
        ]
        print("position_form.assembly_view.choices populated.")

        print("Populating position_form.site_location.choices...")
        position_form.site_location.choices = [
            (sl.id, sl.title)
            for sl in main_session.query(SiteLocation).order_by(SiteLocation.title)
        ]
        print("position_form.site_location.choices populated.")

        # For the tool search form
        print("Populating tool_search_form.tool_category.choices...")
        tool_search_form.tool_category.choices = [
            (c.id, c.name)
            for c in main_session.query(ToolCategory).order_by(ToolCategory.name)
        ]
        print("tool_search_form.tool_category.choices populated.")

        print("Populating tool_search_form.tool_manufacturer.choices...")
        tool_search_form.tool_manufacturer.choices = [
            (m.id, m.name)
            for m in main_session.query(ToolManufacturer).order_by(ToolManufacturer.name)
        ]
        print("tool_search_form.tool_manufacturer.choices populated.")

        print("Finished populating all form choices successfully.")

    except Exception as e:
        error_msg = f"Error populating form choices: {e}"
        print(error_msg)
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
    print("Reached the point before checking request.method...")

    if request.method == 'POST':
        print("Inside POST handling...")

        # figure out which form was submitted
        print("Checking which submit button was used...")
        if 'submit_manufacturer' in request.form:
            form = manufacturer_form
            form_name = 'manufacturer'
            print("Detected 'submit_manufacturer'")
        elif 'submit_category' in request.form:
            form = category_form
            form_name = 'category'
            print("Detected 'submit_category'")
        elif 'submit_tool' in request.form:
            form = tool_form
            form_name = 'tool'
            print("Detected 'submit_tool'")
        elif 'submit_search' in request.form:
            form = tool_search_form
            form_name = 'search'
            print("Detected 'submit_search'")
        else:
            form = None
            form_name = 'unknown'
            print("No recognized form submission found.")

        logger.debug(f"Form submission detected: {form_name}")
        print(f"Form submission detected: {form_name}")

        if form:
            print(f"About to validate form: {form_name}")
            # Check validation
            if form.validate_on_submit():
                print(f"Form '{form_name}' validated successfully. Entering try block...")
                try:
                    if form_name == 'manufacturer':
                        print("Handling 'manufacturer' form logic...")
                        # ... same logic as before ...
                        pass

                    elif form_name == 'category':
                        print("Handling 'category' form logic...")
                        # ... same logic as before ...
                        pass


                    elif form_name == 'tool':

                        print("Handling 'tool' form logic now... (image upload, new Tool creation, etc.)")

                        # Handle image upload

                        uploaded_file_path = None

                        if form.tool_image.data:

                            file = form.tool_image.data

                            if file and allowed_file(file.filename):

                                print("File is present and allowed. Saving file...")

                                filename = secure_filename(file.filename)

                                upload_folder = os.path.join(current_app.root_path, 'static', 'uploads', 'tools')

                                os.makedirs(upload_folder, exist_ok=True)

                                file_path = os.path.join(upload_folder, filename)

                                file.save(file_path)

                                uploaded_file_path = os.path.join('uploads', 'tools', filename)

                                print(f"File saved as {uploaded_file_path}")

                            else:

                                error_msg = 'Invalid image file.'

                                print(error_msg)

                                logger.error(error_msg)

                                if is_ajax:

                                    return jsonify({'success': False, 'message': error_msg}), 400

                                else:

                                    flash(error_msg, 'danger')

                                    return redirect(url_for('tool_routes.submit_tool_data'))

                        print("Creating new_tool instance (without an `image` kwarg)...")

                        new_tool = Tool(

                            name=form.tool_name.data.strip(),

                            size=form.tool_size.data.strip() if form.tool_size.data else None,

                            type=form.tool_type.data.strip() if form.tool_type.data else None,

                            material=form.tool_material.data.strip() if form.tool_material.data else None,

                            description=form.tool_description.data.strip() if form.tool_description.data else None,

                            # NOTE: We do NOT pass `image=...` because Tool model doesn't have that column

                            tool_category_id=form.tool_category.data,  # or category_id if that matches your model

                            tool_manufacturer_id=form.tool_manufacturer.data

                        )

                        print(f"new_tool created: {new_tool.name}")

                        # If a file was uploaded, create a new Image row & ToolImageAssociation

                        if uploaded_file_path:
                            print("Creating `Image` row to store file_path...")

                            new_image = Image(

                                title="Main Tool Image",  # or some other title from the form

                                description="Uploaded via the tool form",  # or from the form

                                file_path=uploaded_file_path

                            )

                            main_session.add(new_image)

                            main_session.commit()  # get new_image.id

                            # Now create a ToolImageAssociation row to link new_tool and new_image

                            print("Creating `ToolImageAssociation` to link Tool & Image...")

                            tool_image_assoc = ToolImageAssociation(

                                tool=new_tool,

                                image=new_image,

                                description="Primary uploaded tool image"  # or from the form

                            )

                            main_session.add(tool_image_assoc)

                            # We'll commit later along with new_tool

                        # Now handle multiple selected IDs from each field (as you already do)

                        print("Processing selected position IDs...")

                        selected_areas = request.form.getlist('area')

                        selected_equipment_groups = request.form.getlist('equipment_group')

                        selected_models = request.form.getlist('model')

                        selected_asset_numbers = request.form.getlist('asset_number')

                        selected_locations = request.form.getlist('location')

                        selected_assemblies = request.form.getlist('assembly')

                        selected_subassemblies = request.form.getlist('subassembly')

                        selected_assembly_views = request.form.getlist('assembly_view')

                        selected_site_locations = request.form.getlist('site_location')

                        position_fields = [

                            ('Area', selected_areas),

                            ('Equipment Group', selected_equipment_groups),

                            ('Model', selected_models),

                            ('Asset Number', selected_asset_numbers),

                            ('Location', selected_locations),

                            ('Assembly', selected_assemblies),

                            ('Subassembly', selected_subassemblies),

                            ('Assembly View', selected_assembly_views),

                            ('Site Location', selected_site_locations)

                        ]

                        for category, selected_ids in position_fields:

                            for pos_id in selected_ids:

                                try:

                                    pos_id_int = int(pos_id)

                                except ValueError:

                                    logger.error(f"Invalid position ID: {pos_id}")

                                    print(f"Skipping invalid position ID: {pos_id}")

                                    continue

                                position = main_session.query(Position).get(pos_id_int)

                                if not position:
                                    logger.error(f"Position with ID {pos_id_int} does not exist.")

                                    print(f"Skipping non-existent position ID: {pos_id_int}")

                                    continue

                                print(f"Associating position ID {pos_id_int} ({category}) with Tool '{new_tool.name}'")

                                association = ToolPositionAssociation(

                                    tool=new_tool,

                                    position_id=pos_id_int,

                                    description=f"{category} Description"

                                )

                                main_session.add(association)

                        print("Adding new_tool and any image associations to DB...")

                        main_session.add(new_tool)

                        main_session.commit()

                        message = 'Tool added successfully with position associations and image!'

                        logger.info(message)

                        print(message)

                        if is_ajax:

                            return jsonify({'success': True, 'message': message}), 200

                        else:

                            flash(message, 'success')

                            return redirect(url_for('tool_routes.submit_tool_data'))


                    elif form_name == 'search':
                        print("Handling 'search' form logic now...")
                        # ... same logic as before ...
                        pass

                    print(f"Completed {form_name} form logic without exceptions.")
                except Exception as e:
                    main_session.rollback()
                    error_msg = f"Error processing {form_name} form: {str(e)}"
                    print(error_msg)
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
                print(f"Form '{form_name}' did NOT validate or form was missing.")
                if is_ajax:
                    errors = {field: errs for field, errs in form.errors.items()}
                    logger.error(f"Form validation failed for {form_name}: {errors}")
                    return jsonify({'success': False, 'errors': errors}), 400
                else:
                    for field, errors in form.errors.items():
                        for error in errors:
                            flash(f"Error in {getattr(form, field).label.text}: {error}", 'danger')

                    try:
                        manufacturers = main_session.query(ToolManufacturer).order_by(ToolManufacturer.name).all()
                        categories = main_session.query(ToolCategory).order_by(ToolCategory.name).all()
                        positions = main_session.query(Position).order_by(Position.id).all()
                    except Exception as e:
                        logger.error(f"Error fetching data during validation failure: {e}", exc_info=True)
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
            print("No recognized form found or request.method != 'POST'")
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

    # end if request.method == 'POST'

    # If GET request or no post
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