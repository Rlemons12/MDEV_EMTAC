from sqlalchemy import Column, Integer, String, Text, ForeignKey, Table, create_engine
from sqlalchemy.orm import relationship, declarative_base, sessionmaker

Base = declarative_base()

# Association table for tools and tool packages (many-to-many)
tool_package_association = Table(
    'tool_package_association',
    Base.metadata,
    Column('tool_id', Integer, ForeignKey('tool.id'), primary_key=True),
    Column('package_id', Integer, ForeignKey('tool_package.id'), primary_key=True),
    Column('quantity', Integer, nullable=False, default=1)
)

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

# ToolPackage model for representing collections of tools
class ToolPackage(Base):
    __tablename__ = 'tool_package'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)

    tools = relationship('Tool', secondary=tool_package_association, back_populates='packages')

# Database setup
DATABASE_URL = "sqlite:///tools_inventory.db"
engine = create_engine(DATABASE_URL, echo=True)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Adding example data
def populate_example_data(session):
    # Create manufacturers
    manufacturer_a = Manufacturer(name="Manufacturer A", country="USA", website="https://www.manufacturera.com")
    manufacturer_b = Manufacturer(name="Manufacturer B", country="Germany", website="https://www.manufacturerb.de")

    # Create categories
    hand_tools = Category(name="Hand Tools", description="Manual tools operated by hand.")
    wrenches = Category(name="Wrenches", description="Tools used for gripping and turning objects.", parent=hand_tools)
    adjustable_wrenches = Category(name="Adjustable Wrenches", description="Wrenches with adjustable jaws.", parent=wrenches)

    # Create tools
    wrench_a = Tool(
        name="Adjustable Wrench",
        size="3/8 inch",
        type="Adjustable",
        material="Steel",
        description="3/8 inch adjustable wrench made of high-quality steel.",
        category=adjustable_wrenches,
        manufacturer=manufacturer_a
    )

    wrench_b = Tool(
        name="Adjustable Wrench",
        size="3/8 inch",
        type="Adjustable",
        material="Alloy",
        description="3/8 inch adjustable wrench made of durable alloy.",
        category=adjustable_wrenches,
        manufacturer=manufacturer_b
    )

    # Create a tool package containing the two wrenches
    wrench_set = ToolPackage(
        name="3/8 Inch Adjustable Wrench Set",
        description="Set of two 3/8 inch adjustable wrenches from different manufacturers."
    )
    wrench_set.tools = [wrench_a, wrench_b]

    # Add and commit everything
    session.add_all([manufacturer_a, manufacturer_b, hand_tools, wrenches, adjustable_wrenches, wrench_a, wrench_b, wrench_set])
    session.commit()

# Populate database with example data
populate_example_data(session)
print("Database populated with example data.")
