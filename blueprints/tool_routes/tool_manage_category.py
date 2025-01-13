# tool_manage_category.py
from blueprints.tool_routes import tool_blueprint_bp
from flask import render_template, redirect, url_for, flash, request
from modules.configuration.config_env import DatabaseConfig
from modules.tool_module.forms.tool_category_forms import ToolCategoryForm
from modules.configuration.log_config import logger

db_config = DatabaseConfig

@tool_blueprint_bp.route('/tool_category/add', methods=['GET', 'POST'])
@tool_blueprint_bp.route('/tool_category/edit_tool_category/<int:category_id>', methods=['GET', 'POST'])
def edit_tool_category(category_id=None):
    form = ToolCategoryForm()
    category = None
    session = db_config.get_main_session()
    
    # If editing an existing category
    if category_id:
        category = ToolCategory.query.get_or_404(category_id)
        if request.method == 'GET':
            form.name.data = category.name
            form.description.data = category.description
            form.parent_id.data = category.parent_id

    # Populate parent category choices
    form.parent_id.choices = [(0, 'None')] + [
        (c.id, c.name) for c in ToolCategory.query.order_by(ToolCategory.name).all()
        if not category or c.id != category.id  # Exclude self from parent list
    ]

    # Handle form submission
    if form.validate_on_submit():
        if category_id:
            # Update existing category
            category.name = form.name.data.strip()
            category.description = form.description.data.strip()
            category.parent_id = form.parent_id.data if form.parent_id.data != 0 else None
        else:
            # Create a new category
            category = ToolCategory(
                name=form.name.data.strip(),
                description=form.description.data.strip(),
                parent_id=form.parent_id.data if form.parent_id.data != 0 else None
            )
            session.add(category)

        try:
            session.commit()
            flash('Category saved successfully!', 'success')
            return redirect(url_for('manage_tool_category'))
        except Exception as e:
            session.rollback()
            flash('An error occurred while saving the category. Please try again.', 'danger')
            app.logger.error(f"Error saving category: {e}")

    return render_template('tool_templates/manage_tool_category.html', form=form, category=category)

@tool_blueprint_bp.route('/tool_category/delete_tool_category', methods=['GET', 'POST'])
def delete_tool_category(category_id=None):
    pass



