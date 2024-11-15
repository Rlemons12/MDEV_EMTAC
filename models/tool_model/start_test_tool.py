from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SubmitField, SelectField
from wtforms.validators import DataRequired
from sqlalchemy.orm import sessionmaker
from tool_model import Base, Category, Manufacturer, Tool, ToolPackage, tool_package_association

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tools_inventory.db'
app.config['SECRET_KEY'] = '12345'
db = SQLAlchemy(app)

# Bind the existing Base to the Flask app's metadata
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
    submit = SubmitField('Add Tool')

@app.route('/add_tool', methods=['GET', 'POST'])
def add_tool():
    form = ToolForm()
    form.category.choices = [(c.id, c.name) for c in session.query(Category).all()]
    form.manufacturer.choices = [(m.id, m.name) for m in session.query(Manufacturer).all()]

    if form.validate_on_submit():
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
        session.commit()
        return redirect(url_for('add_tool'))

    return render_template('add_tool.html', form=form)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
