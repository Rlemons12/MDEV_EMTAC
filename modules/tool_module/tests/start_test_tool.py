from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SubmitField, SelectField
from wtforms.validators import DataRequired
from sqlalchemy.orm import sessionmaker
from tool_model import Base, Category, Manufacturer, Tool, Image, ToolPackage, tool_package_association
import webbrowser
import threading
import time
from wtforms import SelectMultipleField


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tools_inventory.db'
app.config['SECRET_KEY'] = '12345'
db = SQLAlchemy(app)

# Bind the existing Base to the Flask app's metadata and create all tables
with app.app_context():
    Base.metadata.create_all(db.engine)  # Creates all tables if they don't exist
    Base.metadata.bind = db.engine
    Session = sessionmaker(bind=db.engine)
    session = Session()



class ToolForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    size = StringField('Size')
    type = StringField('Type')
    material = StringField('Material')
    description = TextAreaField('Description')
    category = SelectField('Category', coerce=int, validators=[DataRequired()])
    manufacturer = SelectField('Manufacturer', coerce=int, validators=[DataRequired()])
    image_ids = SelectMultipleField('Images', coerce=int)  # New field for selecting images
    submit = SubmitField('Add Tool')


@app.route('/add_tool', methods=['GET', 'POST'])
def add_tool():
    form = ToolForm()
    with app.app_context():
        # Load available choices for categories, manufacturers, and images
        form.category.choices = [(c.id, c.name) for c in session.query(Category).all()]
        form.manufacturer.choices = [(m.id, m.name) for m in session.query(Manufacturer).all()]
        form.image_ids.choices = [(i.id, i.title) for i in session.query(Image).all()]  # Populate images

    if form.validate_on_submit():
        # Add a new tool to the database if form is submitted
        new_tool = Tool(
            name=form.name.data,
            size=form.size.data,
            type=form.type.data,
            material=form.material.data,
            description=form.description.data,
            category_id=form.category.data,
            manufacturer_id=form.manufacturer.data
        )
        session.add(new_tool)

        # Associate selected images with the tool
        for image_id in form.image_ids.data:
            image = session.query(Image).get(image_id)
            new_tool.images.append(image)

        session.commit()
        return redirect(url_for('add_tool'))

    return render_template('add_tool.html', form=form)

def open_browser():
    """Open the default web browser after a delay to give the server time to start"""
    time.sleep(1)  # Delay to ensure the server has started
    webbrowser.open_new('http://127.0.0.1:5001/add_tool')

if __name__ == '__main__':
    # Start a thread to open the web browser
    threading.Thread(target=open_browser).start()

    # Run the Flask application
    app.run(debug=True, port=5001)
