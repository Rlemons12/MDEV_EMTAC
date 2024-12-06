from flask import Blueprint, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SubmitField, SelectField, FileField
from wtforms.validators import DataRequired
from flask_wtf.file import FileAllowed
from sqlalchemy.orm import sessionmaker
from tool_model import Base, Category, Manufacturer, Tool, Image, ToolForm
import os
from werkzeug.utils import secure_filename

# Create the Blueprint instance
tool_bp = Blueprint('tool_bp', __name__, template_folder='templates')

# Folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads/'

# Define the route for adding a tool using the blueprint
@tool_bp.route('/add_tool', methods=['GET', 'POST'])
def add_tool():
    form = ToolForm()

    # Assuming 'db' and 'session' are already accessible
    form.category.choices = [(c.id, c.name) for c in db.session.query(Category).all()]
    form.manufacturer.choices = [(m.id, m.name) for m in db.session.query(Manufacturer).all()]

    if form.validate_on_submit():
        # Handle the form submission logic
        new_tool = Tool(
            name=form.name.data,
            size=form.size.data,
            type=form.type.data,
            material=form.material.data,
            description=form.description.data,
            category_id=form.category.data,
            manufacturer_id=form.manufacturer.data
        )

        # Handle image upload if it exists
        if form.image.data:
            image_file = form.image.data
            filename = secure_filename(image_file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)

            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)

            image_file.save(file_path)

            # Create an Image instance and link it to the Tool
            new_image = Image(title=filename, file_path=file_path)
            session.add(new_image)
            session.flush()  # Ensure new_image.id is generated
            new_tool.images.append(new_image)

        session.add(new_tool)
        session.commit()

        return redirect(url_for('tool_bp.add_tool'))

    return render_template('add_tool.html', form=form)
