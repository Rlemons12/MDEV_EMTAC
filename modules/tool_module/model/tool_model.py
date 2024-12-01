
import os
import sys
from sqlalchemy import Column, Integer, String, Text, ForeignKey, Table, create_engine
from sqlalchemy.orm import relationship, declarative_base, sessionmaker

# Add the path of AuMaintdb to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from emtacdb_fts import ImagePositionAssociation

Base = declarative_base()

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
    image = FileField('Image', validators=[FileAllowed(['jpg', 'png'], 'Images only!')])
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


class ImagePositionAssociation(Base):
    __tablename__ = 'image_position_association'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('image.id'))
    position_id = Column(Integer, ForeignKey('position.id'))

    image = relationship("Image", back_populates="image_position_association")
    position = relationship("Position", back_populates="image_position_association")

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
    tool_image_associations = relationship('ToolImageAssociation', back_populates='tool', cascade="all, delete-orphan")
    images = relationship('Image', secondary='tool_image_association', back_populates='tools')

# ToolPackage model for representing collections of tools
class ToolPackage(Base):
    __tablename__ = 'tool_package'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)

    tools = relationship('Tool', secondary=tool_package_association, back_populates='packages')

# Image model representing tool images
class Image(Base):
    __tablename__ = 'image'
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(String, nullable=False)
    file_path = Column(String, nullable=False)

    # Relationships
    tool_image_associations = relationship('ToolImageAssociation', back_populates='image', cascade="all, delete-orphan")
    tools = relationship('Tool', secondary='tool_image_association', back_populates='images')
    #parts_position_image = relationship("PartsPositionImageAssociation", back_populates="image")
    #image_problem = relationship("ImageProblemAssociation", back_populates="image")
    #image_task = relationship("ImageTaskAssociation", back_populates="image")
    #image_completed_document_association = relationship("ImageCompletedDocumentAssociation", back_populates="image")
    #image_embedding = relationship("ImageEmbedding", back_populates="image")
    #image_position_association = relationship("ImagePositionAssociation", back_populates="image")

# Main Tables
class SiteLocation(Base):
    __tablename__ = 'site_location'
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    room_number = Column(String, nullable=False)

    position = relationship('Position', back_populates="site_location")

class Position(Base):
    __tablename__ = 'position'
    id = Column(Integer, primary_key=True)
    area_id = Column(Integer, ForeignKey('area.id'), nullable=True)
    equipment_group_id = Column(Integer, ForeignKey('equipment_group.id'), nullable=True)
    model_id = Column(Integer, ForeignKey('model.id'), nullable=True)
    asset_number_id = Column(Integer, ForeignKey('asset_number.id'), nullable=True)
    location_id = Column(Integer, ForeignKey('location.id'), nullable=True)
    site_location_id = Column(Integer, ForeignKey('site_location.id'), nullable=True)

    area = relationship("Area", back_populates="position")
    equipment_group = relationship("EquipmentGroup", back_populates="position")
    model = relationship("Model", back_populates="position")
    asset_number = relationship("AssetNumber", back_populates="position")
    location = relationship("Location", back_populates="position")
    image_position_association = relationship("ImagePositionAssociation", back_populates="position")
    site_location = relationship("SiteLocation", back_populates="position")
    """bill_of_material = relationship("BillOfMaterial", back_populates="position")
    part_position_image = relationship("PartsPositionImageAssociation", back_populates="position")
   
    drawing_position = relationship("DrawingPositionAssociation", back_populates="position")
    problem_position = relationship("ProblemPositionAssociation", back_populates="position")
    completed_document_position_association = relationship("CompletedDocumentPositionAssociation",
                                                           back_populates="position")
    position_tasks = relationship("TaskPositionAssociation", back_populates=
        "position", cascade="all, delete-orphan")"""

class ToolUsed(Base):
    __tablename__ = 'tool_used'
    id = Column(Integer, primary_key=True)
    tool_id = Column(Integer, ForeignKey('tool.id'))
    position_id = Column(Integer, ForeignKey('position.id'))

class Area(Base):
    __tablename__ = 'area'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String)

    equipment_group = relationship("EquipmentGroup", back_populates="area")
    position = relationship("Position", back_populates="area")

class EquipmentGroup(Base):
    __tablename__ = 'equipment_group'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    area_id = Column(Integer, ForeignKey('area.id'))
    description = Column(String, nullable=True)

    area = relationship("Area", back_populates="equipment_group")
    model = relationship("Model", back_populates="equipment_group")
    position = relationship("Position", back_populates="equipment_group")

class Model(Base):
    __tablename__ = 'model'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String)
    equipment_group_id = Column(Integer, ForeignKey('equipment_group.id'))

    equipment_group = relationship("EquipmentGroup", back_populates="model")
    asset_number = relationship("AssetNumber", back_populates="model")
    location = relationship("Location", back_populates="model")
    position = relationship("Position", back_populates="model")

class AssetNumber(Base):
    __tablename__ = 'asset_number'

    id = Column(Integer, primary_key=True)
    number = Column(String, nullable=False)
    description = Column(String)
    model_id = Column(Integer, ForeignKey('model.id'))

    model = relationship("Model", back_populates="asset_number")
    position = relationship("Position", back_populates="asset_number")

class Location(Base):
    __tablename__ = 'location'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    model_id = Column(Integer, ForeignKey('model.id'))
    description = Column(String, nullable=True)

    model = relationship("Model", back_populates="location")
    position = relationship("Position", back_populates="location")

# Database setup
DATABASE_URL = "sqlite:///tools_inventory.db"
engine = create_engine(DATABASE_URL, echo=True)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Adding example data
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
