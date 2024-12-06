import sys
import os

if getattr(sys, 'frozen', False):  # Check if running as an executable
    current_dir = os.path.dirname(sys.executable)  # Use the executable directory
else:
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Use the script directory

sys.path.append(current_dir)

from sqlalchemy import Column, Integer, String, Text, ForeignKey, Table, create_engine
from sqlalchemy.orm import relationship, declarative_base, sessionmaker
from flask import Flask, render_template, request, redirect, url_for
from wtforms import StringField, TextAreaField, SubmitField, SelectField
from wtforms.validators import DataRequired
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import SelectMultipleField
# Add the path of AuMaintdb to sys.path
from emtacdb_fts import Position, Image
from config import DATABASE_URL
from base import Base




# Association table for tools and tool packages (many-to-many)
tool_package_association = Table(
    'tool_package_association',
    Base.metadata,
    Column('tool_id', Integer, ForeignKey('tool.id'), primary_key=True),
    Column('package_id', Integer, ForeignKey('tool_package.id'), primary_key=True),
    Column('quantity', Integer, nullable=False, default=1)
)

class ToolForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    size = StringField('Size')
    type = StringField('Type')
    material = StringField('Material')
    description = TextAreaField('Description')
    category = SelectField('Category', coerce=int, validators=[DataRequired()])
    manufacturer = SelectField('Manufacturer', coerce=int, validators=[DataRequired()])
    image_id = FileField('Image', validators=[FileAllowed(['jpg', 'png'], 'Images only!')])
    submit = SubmitField('Add Tool')

# Association class for tools and images (many-to-many)
class ToolImageAssociation(Base):
    __tablename__ = 'tool_image_association'
    id = Column(Integer, primary_key=True)
    tool_id = Column(Integer, ForeignKey('tool.id'))
    image_id = Column(Integer, ForeignKey('image.id'))
    description = Column(Text, nullable=True)

    # Relationships
    tool = relationship('Tool', back_populates='tool_image_associations')
    image = relationship('Image', back_populates='tool_image_associations')

class ToolPositionAssociation(Base):
    __tablename__ = 'tool_position_association'
    id = Column(Integer, primary_key=True)
    tool_id = Column(Integer, ForeignKey('tool.id'))
    position_id = Column(Integer, ForeignKey('position.id'))
    description = Column(Text, nullable=True)
    tool = relationship('Tool', back_populates='tool_position_associations')
    position = relationship('Position', back_populates='tool_position_associations')

# Category model for hierarchical tool categories
class Category(Base):
    __tablename__ = 'category'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(Text)
    parent_id = Column(Integer, ForeignKey('category.id'), nullable=True)

    # Self-referential relationships for hierarchy
    parent = relationship('Category', remote_side=[id], back_populates='subcategories')
    subcategories = relationship('Category', back_populates='parent', cascade="all, delete-orphan")
    tools = relationship('Tool', back_populates='category', cascade="all, delete-orphan")

# Manufacturer model for tracking tool manufacturers
class Manufacturer(Base):
    __tablename__ = 'manufacturer'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    country = Column(String, nullable=True)
    website = Column(String, nullable=True)

    tools = relationship('Tool', back_populates='manufacturer')

# Tool model representing individual tools
class Tool(Base):
    __tablename__ = 'tool'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    size = Column(String, nullable=True)
    type = Column(String, nullable=True)
    material = Column(String, nullable=True)
    description = Column(Text)
    category_id = Column(Integer, ForeignKey('category.id'))
    manufacturer_id = Column(Integer, ForeignKey('manufacturer.id'))

    # Relationships
    category = relationship('Category', back_populates='tools')
    manufacturer = relationship('Manufacturer', back_populates='tools')
    packages = relationship('ToolPackage', secondary=tool_package_association, back_populates='tools')
    tool_image_associations = relationship(
        'ToolImageAssociation',
        back_populates='image',
        cascade="all, delete-orphan",
        overlaps="tools"
    )
    tools = relationship(
        'Tool',
        secondary='tool_image_association',
        back_populates='images',
        overlaps="tool_image_associations"
    )

# ToolPackage model for representing collections of tools
class ToolPackage(Base):
    __tablename__ = 'tool_package'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)

    tools = relationship('Tool', secondary=tool_package_association, back_populates='packages')

class ToolUsed(Base):
    __tablename__ = 'tool_used'
    id = Column(Integer, primary_key=True)
    tool_id = Column(Integer, ForeignKey('tool.id'))
    position_id = Column(Integer, ForeignKey('position.id'))



# Database setup
DATABASE_URL = DATABASE_URL
engine = create_engine(DATABASE_URL, echo=True)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()


# Adding example data
def populate_example_data(session):
    # Check if "Hand Tools" already exists
    if not session.query(Category).filter_by(name="Hand Tools").first():
        # Create categories
        hand_tools = Category(name="Hand Tools", description="Manual tools operated by hand.")
        session.add(hand_tools)
        session.commit()

    # Check if manufacturers exist before adding
    if not session.query(Manufacturer).filter_by(name="Manufacturer A").first():
        manufacturer_a = Manufacturer(name="Manufacturer A", country="USA", website="https://www.manufacturera.com")
        session.add(manufacturer_a)

    if not session.query(Manufacturer).filter_by(name="Manufacturer B").first():
        manufacturer_b = Manufacturer(name="Manufacturer B", country="Germany", website="https://www.manufacturerb.de")
        session.add(manufacturer_b)

    # Similarly, add checks for other entries if necessary
    session.commit()

# Populate database with example data
populate_example_data(session)
print("Database populated with example data.")
