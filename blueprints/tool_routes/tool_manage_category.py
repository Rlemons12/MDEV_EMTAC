# tool_manage_category.py
from blueprints.tool_routes import tool_blueprint_bp
from flask import render_template, redirect, url_for, flash, request
from modules.configuration.config_env import DatabaseConfig
from modules.tool_module.forms.tool_category_forms import ToolCategoryForm
from modules.emtacdb.emtacdb_fts import ToolCategory
from modules.configuration.log_config import logger

db_config = DatabaseConfig()


@tool_blueprint_bp.route('/tool_category/add', methods=['GET', 'POST'])

@tool_blueprint_bp.route('/tool_category/add_category', methods=['GET', 'POST'])
def add_tool_category():
    logger.info("Accessing /tool_category/add_category route for adding a new category.")

    # Instantiate the form and get a database session.
    category_form = ToolCategoryForm()
    session = db_config.get_main_session()
    logger.info("Database session acquired successfully.")

    # Initialize categories for use in both parent choices and existing list.
    categories = []

    # Populate parent category choices and retrieve all categories.
    try:
        logger.info("Querying database for existing tool categories to populate parent choices.")
        categories = session.query(ToolCategory).order_by(ToolCategory.name).all()
        num_categories = len(categories)
        logger.info(f"Retrieved {num_categories} categories from the database.")
        category_form.parent_id.choices = [(0, 'None')] + [(c.id, c.name) for c in categories]
        logger.info("Parent category choices populated successfully.")
    except Exception as e:
        logger.error(f"Error fetching category choices: {e}", exc_info=True)
        flash("An error occurred while fetching category choices.", "danger")

    # Process the form submission when validated.
    if category_form.validate_on_submit():
        logger.info("Form validation successful. Processing form data.")
        category_name = category_form.name.data.strip()
        category_description = category_form.description.data.strip()
        parent_choice = category_form.parent_id.data
        logger.info(
            f"Form Data - Name: '{category_name}', Description: '{category_description}', Parent ID: {parent_choice}")

        # Create new ToolCategory instance.
        new_category = ToolCategory(
            name=category_name,
            description=category_description,
            parent_id=parent_choice if parent_choice != 0 else None
        )
        logger.info(f"New ToolCategory instance created: {new_category}")

        # Add the new category to the session.
        session.add(new_category)
        logger.info("New category added to the database session.")

        try:
            logger.info("Attempting to commit the new category to the database.")
            session.commit()
            logger.info(f"Category '{new_category.name}' saved successfully with ID: {new_category.id}!")
            flash("Category saved successfully!", "success")
            return redirect(url_for("tool_routes.add_tool_category"))
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving category: {e}. Rolled back the transaction.", exc_info=True)
            flash("An error occurred while saving the category. Please try again.", "danger")
    else:
        if request.method == 'POST':
            logger.info("Form submission failed validation. Errors: %s", category_form.errors)

    logger.info("Rendering add category template.")
    return render_template("tool_templates/tool_category.html", category_form=category_form, category=None,
                           categories=categories)

@tool_blueprint_bp.route('/tool_category/edit_tool_category', methods=['GET', 'POST'])
@tool_blueprint_bp.route('/tool_category/edit_tool_category/<int:category_id>', methods=['GET', 'POST'])
def edit_tool_category(category_id=None):
    logger.info("Accessing /tool_category/edit_tool_category route.")

    category_form = ToolCategoryForm()
    session = db_config.get_main_session()
    category = None

    # If category_id is not provided in the URL, try to get it from form data.
    if not category_id:
        category_id_from_form = request.form.get('category_id')
        if category_id_from_form:
            try:
                category_id = int(category_id_from_form)
            except ValueError:
                flash("Invalid category ID.", "danger")
                return redirect(url_for("tool_routes.add_tool_category"))

    if category_id:
        logger.info(f"Editing existing category with ID: {category_id}")
        category = session.query(ToolCategory).get(category_id)
        if not category:
            logger.warning(f"Category with ID {category_id} not found!")
            flash("Category not found.", "danger")
            return redirect(url_for("tool_routes.add_tool_category"))

        if request.method == 'GET':
            logger.info(f"Pre-filling form with existing category data: {category.name}")
            category_form.name.data = category.name
            category_form.description.data = category.description
            category_form.parent_id.data = category.parent_id

    try:
        logger.info("Populating parent category choices.")
        all_categories = session.query(ToolCategory).order_by(ToolCategory.name).all()
        category_form.parent_id.choices = [(0, 'None')] + [
            (c.id, c.name) for c in all_categories if not category or c.id != category.id
        ]
        logger.info("Parent category choices populated successfully.")
    except Exception as e:
        logger.error(f"Error fetching category choices: {e}", exc_info=True)

    if category_form.validate_on_submit():
        logger.info(f"Form submitted. Name: {category_form.name.data}, Description: {category_form.description.data}, Parent ID: {category_form.parent_id.data}")
        if category:
            logger.info(f"Updating category ID {category_id}.")
            category.name = category_form.name.data.strip()
            category.description = category_form.description.data.strip()
            category.parent_id = category_form.parent_id.data if category_form.parent_id.data != 0 else None
        else:
            logger.info("Creating new category (should not occur in edit route).")
            category = ToolCategory(
                name=category_form.name.data.strip(),
                description=category_form.description.data.strip(),
                parent_id=category_form.parent_id.data if category_form.parent_id.data != 0 else None
            )
            session.add(category)

        try:
            session.commit()
            logger.info(f"Category '{category.name}' saved successfully!")
            flash("Category saved successfully!", "success")
            return redirect(url_for("tool_routes.add_tool_category"))
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving category: {e}", exc_info=True)
            flash("An error occurred while saving the category. Please try again.", "danger")

    try:
        categories = session.query(ToolCategory).order_by(ToolCategory.name).all()
    except Exception as e:
        logger.error(f"Error retrieving categories for display: {e}", exc_info=True)
        categories = []

    logger.info("Rendering category management template.")
    return render_template("tool_templates/manage_tool_category.html", category_form=category_form, category=category, categories=categories)


@tool_blueprint_bp.route('/tool_category/delete_tool_category', methods=['GET', 'POST'])
def delete_tool_category(category_id=None):
    pass



