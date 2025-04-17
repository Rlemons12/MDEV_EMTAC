import os
import logging
from datetime import datetime
import openai
import spacy
from fuzzywuzzy import process
from werkzeug.security import check_password_hash, generate_password_hash
from sqlalchemy import (DateTime, Column, ForeignKey, Integer, JSON, LargeBinary, Enum as SqlEnum,
                        String, create_engine, text, Float, Text, UniqueConstraint, Table)
from enum import Enum as PyEnum  # Import Enum and alias it as PyEnum
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.orm import declarative_base, configure_mappers, relationship, scoped_session, sessionmaker
from modules.configuration.config import (OPENAI_API_KEY, BASE_DIR, COPY_FILES, DATABASE_URL,DATABASE_PATH)
from modules.configuration.base import Base
from modules.configuration.log_config import logger

# Configure mappers (must be called after all ORM classes are defined)
configure_mappers()

# Load the English language model
nlp = spacy.load('en_core_web_sm')

# Set your OpenAI API key
openai.api_key = OPENAI_API_KEY

# Constants for chunk size and model name
CHUNK_SIZE = 8000
MODEL_NAME = "text-embedding-ada-002"

# Check if the database file exists in the specified directory and create it if not
if not os.path.exists(DATABASE_PATH):
    open(DATABASE_PATH, 'w').close()

# Database setup
engine = create_engine(
    DATABASE_URL, 
    pool_size=10, 
    max_overflow=20, 
    connect_args={"check_same_thread": False}
)

Session = scoped_session(sessionmaker(bind=engine))
session = Session


# Revision control database configuration
"""REVISION_CONTROL_DB_PATH = os.path.join(DATABASE_DIR, 'emtac_revision_control_db.db')
revision_control_engine = create_engine(
    f'sqlite:///{REVISION_CONTROL_DB_PATH}',
    pool_size=10,            # Set a small pool size
    max_overflow=20,         # Allow up to 10 additional connections
    connect_args={"check_same_thread": False}  # Needed for SQLite when using threading
)

RevisionControlBase = declarative_base()  # This is correctly defined
RevisionControlSession = scoped_session(sessionmaker(bind=revision_control_engine))  # Use distinct name
revision_control_session = RevisionControlSession()"""

class VersionInfo(Base):
    __tablename__ = 'version_info'
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True, autoincrement=True)
    version_number = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    description = Column(String, nullable=True)

# Main Tables
class SiteLocation(Base):
    __tablename__ = 'site_location'
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    room_number = Column(String, nullable=False)
    site_area = Column(String, nullable=False)
    
    position = relationship('Position', back_populates="site_location")

class Position(Base):
    __tablename__ = 'position'
    id = Column(Integer, primary_key=True)
    area_id = Column(Integer, ForeignKey('area.id'), nullable=True)
    equipment_group_id = Column(Integer, ForeignKey('equipment_group.id'), nullable=True)
    model_id = Column(Integer, ForeignKey('model.id'), nullable=True)
    asset_number_id = Column(Integer, ForeignKey('asset_number.id'), nullable=True)
    location_id = Column(Integer, ForeignKey('location.id'), nullable=True)
    subassembly_id = Column(Integer, ForeignKey('subassembly.id'), nullable=True)
    component_assembly_id = Column(Integer, ForeignKey('component_assembly.id'), nullable=True)
    assembly_view_id = Column(Integer, ForeignKey('assembly_view.id'), nullable=True)
    site_location_id = Column(Integer, ForeignKey('site_location.id'), nullable=True)

    area = relationship("Area", back_populates="position")
    equipment_group = relationship("EquipmentGroup", back_populates="position")
    model = relationship("Model", back_populates="position")
    asset_number = relationship("AssetNumber", back_populates="position")
    location = relationship("Location", back_populates="position")
    """bill_of_material = relationship("BillOfMaterial", back_populates="position")"""
    part_position_image = relationship("PartsPositionImageAssociation", back_populates="position")
    image_position_association = relationship("ImagePositionAssociation", back_populates="position")
    drawing_position = relationship("DrawingPositionAssociation", back_populates="position")
    problem_position = relationship("ProblemPositionAssociation", back_populates="position")
    completed_document_position_association = relationship("CompletedDocumentPositionAssociation", back_populates="position")
    site_location = relationship("SiteLocation", back_populates="position")
    position_tasks = relationship("TaskPositionAssociation", back_populates="position", cascade="all, delete-orphan")
    tool_position_association = relationship("ToolPositionAssociation", back_populates="position")
    subassembly = relationship("Subassembly", back_populates="position")
    component_assembly = relationship("ComponentAssembly", back_populates="position")
    assembly_view = relationship("AssemblyView", back_populates="position")

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
    description = Column(String,nullable=True)
    
    area = relationship("Area", back_populates="equipment_group") 
    model = relationship("Model", back_populates="equipment_group")
    position = relationship("Position", back_populates="equipment_group")
    
class Model(Base):
    __tablename__ = 'model'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String,nullable=True)
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

    @classmethod
    def get_ids_by_number(cls, session, number):
        """Retrieve all AssetNumber IDs that match the given number."""
        logger.info(f"========== ASSET NUMBER SEARCH ==========")
        logger.debug(f"Querying AssetNumber IDs for number: '{number}'")

        try:
            # Log the search pattern being used
            logger.debug(f"Using exact match search pattern for number: '{number}'")

            # Execute the query
            results = session.query(cls.id).filter(cls.number == number).all()

            # Extract IDs from the results
            ids = [id_ for (id_,) in results]

            # Log detailed information about the results
            if ids:
                logger.info(f"Found {len(ids)} AssetNumbers with number '{number}': {ids}")
                for i, asset_id in enumerate(ids):
                    try:
                        # Get more details about each asset found
                        asset = session.query(cls).filter(cls.id == asset_id).first()
                        if asset:
                            logger.debug(f"Asset #{i + 1}: ID={asset_id}, Number={asset.number}, " +
                                         f"Description={asset.description or 'None'}, Model ID={asset.model_id}")

                            # Get model info if available
                            if asset.model_id:
                                model = session.query(Model).filter(Model.id == asset.model_id).first()
                                if model:
                                    logger.debug(f"  -> Model: ID={model.id}, Name={model.name}")
                    except Exception as e:
                        logger.warning(f"Error getting details for asset ID {asset_id}: {e}")
            else:
                logger.warning(f"No AssetNumbers found with number '{number}'")

            return ids
        except Exception as e:
            logger.error(f"Error querying AssetNumbers by number '{number}': {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        finally:
            logger.info(f"========== ASSET NUMBER SEARCH COMPLETE ==========")

    @classmethod
    def get_model_id_by_asset_number_id(cls, session, asset_number_id):
        """
        Given an asset_number_id, returns the associated model_id.

        Parameters:
            session: SQLAlchemy session object used for querying.
            asset_number_id: The id of the AssetNumber record.

        Returns:
            The model_id associated with the asset_number, or None if not found.
        """
        logger.info(f"========== GETTING MODEL FOR ASSET ID {asset_number_id} ==========")
        logger.debug(f"Querying AssetNumber for asset_number_id: {asset_number_id}")

        try:
            # First try to get the full asset record for more detailed logging
            asset = session.query(cls).filter(cls.id == asset_number_id).first()
            if asset:
                logger.debug(f"Found asset: ID={asset.id}, Number={asset.number}, " +
                             f"Description={asset.description or 'None'}, Model ID={asset.model_id}")
                model_id = asset.model_id
            else:
                # Fallback to just getting the model_id directly
                logger.debug(f"Asset not found, querying only for the model_id")
                model_id = session.query(cls.model_id).filter(cls.id == asset_number_id).scalar()

            if model_id is not None:
                logger.info(f"Found model_id: {model_id} for asset_number_id: {asset_number_id}")

                # Get model details for better logging
                try:
                    model = session.query(Model).filter(Model.id == model_id).first()
                    if model:
                        logger.debug(f"Model details: ID={model.id}, Name={model.name}, " +
                                     f"Equipment Group ID={model.equipment_group_id}")
                except Exception as e:
                    logger.warning(f"Error getting model details: {e}")
            else:
                logger.warning(f"No AssetNumber found with id: {asset_number_id}")

            return model_id
        except Exception as e:
            logger.error(f"Error getting model_id for asset_number_id {asset_number_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        finally:
            logger.info(f"========== MODEL SEARCH COMPLETE ==========")

    @classmethod
    def get_equipment_group_id_by_asset_number_id(cls, session, asset_number_id):
        """
        Given an asset_number_id, retrieves the equipment_group id that is associated with its model.

        This method works in two steps:
          1. It joins the AssetNumber table with the Model table (using AssetNumber.model_id).
          2. It selects the 'equipment_group_id' field from Model, which holds the id of the associated EquipmentGroup.

        Parameters:
            session: SQLAlchemy session object used for querying.
            asset_number_id: The id of the AssetNumber record.

        Returns:
            The equipment_group id if found, otherwise None.
        """
        logger.info(f"========== GETTING EQUIPMENT GROUP FOR ASSET ID {asset_number_id} ==========")
        logger.debug(f"Querying for equipment_group id using asset_number_id: {asset_number_id}")

        try:
            # Try to get the model_id first for more detailed logging
            model_id = cls.get_model_id_by_asset_number_id(session, asset_number_id)

            if model_id is not None:
                logger.debug(f"Found model_id: {model_id} for asset_number_id: {asset_number_id}")

                # Query directly using the model ID for better performance
                equipment_group_id = session.query(Model.equipment_group_id).filter(Model.id == model_id).scalar()

                if equipment_group_id is not None:
                    logger.info(f"Found equipment_group_id: {equipment_group_id} via model_id {model_id}")

                    # Get equipment group details for better logging
                    try:
                        group = session.query(EquipmentGroup).filter(EquipmentGroup.id == equipment_group_id).first()
                        if group:
                            logger.debug(f"Equipment Group details: ID={group.id}, Name={group.name}, " +
                                         f"Area ID={group.area_id}")
                    except Exception as e:
                        logger.warning(f"Error getting equipment group details: {e}")
                else:
                    logger.warning(f"No equipment_group_id found for model_id: {model_id}")

                    # Fall back to the join method
                    logger.debug(f"Falling back to join query method")
                    equipment_group_id = (
                        session.query(Model.equipment_group_id)
                        .join(Model, Model.id == cls.model_id)
                        .filter(cls.id == asset_number_id)
                        .scalar()
                    )
            else:
                # If we couldn't get the model_id, use the join method directly
                logger.debug(f"No model_id found, using join query method directly")
                equipment_group_id = (
                    session.query(Model.equipment_group_id)
                    .join(Model, Model.id == cls.model_id)
                    .filter(cls.id == asset_number_id)
                    .scalar()
                )

            if equipment_group_id is not None:
                logger.info(
                    f"Final result: Found equipment_group_id: {equipment_group_id} for asset_number_id: {asset_number_id}")
            else:
                logger.warning(f"Final result: No EquipmentGroup found for asset_number_id: {asset_number_id}")

            return equipment_group_id
        except Exception as e:
            logger.error(f"Error getting equipment_group_id for asset_number_id {asset_number_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        finally:
            logger.info(f"========== EQUIPMENT GROUP SEARCH COMPLETE ==========")

    @classmethod
    def get_area_id_by_asset_number_id(cls, session, asset_number_id):
        """
        Given an asset_number_id, retrieves the associated area_id.

        This method performs a series of joins:
          1. Join Area to EquipmentGroup on Area.id equals EquipmentGroup.area_id.
          2. Join EquipmentGroup to Model on EquipmentGroup.id equals Model.equipment_group_id.
          3. Join Model to AssetNumber on Model.id equals AssetNumber.model_id.
          4. Filter by the specified asset_number_id to ultimately extract the Area.id.

        Parameters:
            session: SQLAlchemy session object used for querying.
            asset_number_id: The id of the AssetNumber record.

        Returns:
            The area_id associated with the asset_number, or None if no matching record is found.
        """
        logger.info(f"========== GETTING AREA FOR ASSET ID {asset_number_id} ==========")
        logger.debug(f"Querying for area_id using asset_number_id: {asset_number_id}")

        try:
            # Try to get the equipment_group_id first for more detailed logging
            equipment_group_id = cls.get_equipment_group_id_by_asset_number_id(session, asset_number_id)

            if equipment_group_id is not None:
                logger.debug(f"Found equipment_group_id: {equipment_group_id} for asset_number_id: {asset_number_id}")

                # Query directly using the equipment group ID for better performance
                area_id = session.query(EquipmentGroup.area_id).filter(EquipmentGroup.id == equipment_group_id).scalar()

                if area_id is not None:
                    logger.info(f"Found area_id: {area_id} via equipment_group_id {equipment_group_id}")

                    # Get area details for better logging
                    try:
                        area = session.query(Area).filter(Area.id == area_id).first()
                        if area:
                            logger.debug(f"Area details: ID={area.id}, Name={area.name}")
                    except Exception as e:
                        logger.warning(f"Error getting area details: {e}")
                else:
                    logger.warning(f"No area_id found for equipment_group_id: {equipment_group_id}")

                    # Fall back to the join method
                    logger.debug(f"Falling back to join query method")
                    area_id = (
                        session.query(Area.id)
                        .join(EquipmentGroup, EquipmentGroup.area_id == Area.id)
                        .join(Model, Model.equipment_group_id == EquipmentGroup.id)
                        .join(AssetNumber, AssetNumber.model_id == Model.id)
                        .filter(AssetNumber.id == asset_number_id)
                        .scalar()
                    )
            else:
                # If we couldn't get the equipment_group_id, use the join method directly
                logger.debug(f"No equipment_group_id found, using join query method directly")
                area_id = (
                    session.query(Area.id)
                    .join(EquipmentGroup, EquipmentGroup.area_id == Area.id)
                    .join(Model, Model.equipment_group_id == EquipmentGroup.id)
                    .join(AssetNumber, AssetNumber.model_id == Model.id)
                    .filter(AssetNumber.id == asset_number_id)
                    .scalar()
                )

            if area_id is not None:
                logger.info(f"Final result: Found area_id: {area_id} for asset_number_id: {asset_number_id}")
            else:
                logger.warning(f"Final result: No area found for asset_number_id: {asset_number_id}")

            return area_id
        except Exception as e:
            logger.error(f"Error getting area_id for asset_number_id {asset_number_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        finally:
            logger.info(f"========== AREA SEARCH COMPLETE ==========")

    @classmethod
    def get_position_ids_by_asset_number_id(cls, session, asset_number_id):
        """
        Given an asset_number_id, retrieves all Position IDs that reference this asset_number.

        This method performs a query on the Position table where the asset_number_id
        matches the provided value. It returns a list of Position.id values.

        Parameters:
            session: SQLAlchemy session object used for querying.
            asset_number_id: The id value of the AssetNumber record.

        Returns:
            A list of Position IDs associated with the given asset_number_id.
            If no matching positions are found, an empty list is returned.
        """
        logger.info(f"========== GETTING POSITIONS FOR ASSET ID {asset_number_id} ==========")
        logger.debug(f"Querying for all Position IDs with asset_number_id: {asset_number_id}")

        try:
            # Get the asset details for more context in logging
            asset = session.query(cls).filter(cls.id == asset_number_id).first()
            if asset:
                logger.debug(f"Asset details: ID={asset.id}, Number={asset.number}, " +
                             f"Description={asset.description or 'None'}, Model ID={asset.model_id}")

            # Execute the query to get positions
            results = session.query(Position.id).filter(Position.asset_number_id == asset_number_id).all()
            position_ids = [pos_id for (pos_id,) in results]

            # Log detailed information about the results
            if position_ids:
                logger.info(
                    f"Found {len(position_ids)} Position(s) for asset_number_id: {asset_number_id}: {position_ids}")

                # Log details about each position
                for i, pos_id in enumerate(position_ids):
                    try:
                        position = session.query(Position).filter(Position.id == pos_id).first()
                        if position:
                            logger.debug(f"Position #{i + 1}: ID={pos_id}, " +
                                         f"Area ID={position.area_id}, " +
                                         f"Group ID={position.equipment_group_id}, " +
                                         f"Model ID={position.model_id}, " +
                                         f"Location ID={position.location_id}")

                            # Try to get location name for more context
                            if position.location_id:
                                location = session.query(Location).filter(Location.id == position.location_id).first()
                                if location:
                                    logger.debug(f"  -> Location: ID={location.id}, Name={location.name}")
                    except Exception as e:
                        logger.warning(f"Error getting details for position ID {pos_id}: {e}")
            else:
                logger.warning(f"No Positions found for asset_number_id: {asset_number_id}")

            return position_ids
        except Exception as e:
            logger.error(f"Error getting position_ids for asset_number_id {asset_number_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        finally:
            logger.info(f"========== POSITION SEARCH COMPLETE ==========")

    @classmethod
    def search_asset_numbers(cls, session, query, limit=10):
        """
        Searches for asset numbers that match the provided query using
        a case-insensitive partial match. Useful for autocomplete or dynamic
        search interfaces.

        Parameters:
            session: SQLAlchemy session object used for querying.
            query: The partial asset number string input by the user.
            limit: Maximum number of results to return (default is 10).

        Returns:
            A list of dictionaries, each containing details about an asset:
              - id: The asset's unique identifier.
              - number: The asset number.
              - description: The asset description.
              - model_id: The associated model ID.
            If no records match, an empty list is returned.
        """
        logger.info("========== ASSET NUMBER AUTOCOMPLETE SEARCH ==========")
        logger.debug(f"Initiating search for asset numbers with query: '{query}'")

        try:
            # If the query is empty, just return an empty list early
            if not query:
                logger.debug("Empty query received; returning empty result set.")
                return []

            # Create a search pattern for a partial, case-insensitive match.
            search_pattern = f"%{query}%"
            logger.debug(f"Using search pattern: '{search_pattern}'")

            # Query for matching asset numbers; you can adjust the limit as needed.
            results = session.query(cls).filter(cls.number.ilike(search_pattern)).limit(limit).all()

            if results:
                assets = []
                # Loop through the found results to build a structured list with detailed logging.
                for asset in results:
                    asset_details = {
                        "id": asset.id,
                        "number": asset.number,
                        "description": asset.description,
                        "model_id": asset.model_id
                    }
                    assets.append(asset_details)
                    logger.debug(f"Found asset: {asset_details}")

                logger.info(f"Found {len(assets)} asset(s) matching query '{query}'.")
                return assets
            else:
                logger.warning(f"No assets found matching query '{query}'.")
                return []
        except Exception as e:
            logger.error(f"Error searching for asset numbers with query '{query}': {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        finally:
            logger.info("========== ASSET NUMBER AUTOCOMPLETE SEARCH COMPLETE ==========")
    
class Location(Base):
    __tablename__ = 'location'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)    
    model_id = Column(Integer, ForeignKey('model.id'))
    description = Column(String, nullable=True)
    
    model = relationship("Model", back_populates="location")
    position = relationship("Position", back_populates="location")
    subassembly = relationship("Subassembly", back_populates="location")

#class's dealing with machine subassemblies.

class Subassembly(Base):
    __tablename__ = 'subassembly'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=True)
    location_id = Column(Integer, ForeignKey('location.id'))
    description = Column(String, nullable=True)  # CHANGED to allow NULL
    # Relationships
    location = relationship("Location", back_populates="subassembly")
    component_assembly = relationship("ComponentAssembly", back_populates="subassembly")
    position = relationship("Position", back_populates="subassembly")

class ComponentAssembly(Base):
    # specific group of components of a subassembly
    __tablename__ = 'component_assembly'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=True)
    description = Column(String, nullable=True)
    subassembly_id = Column(Integer, ForeignKey('subassembly.id'), nullable=False)

    # Relationships
    subassembly = relationship("Subassembly", back_populates="component_assembly")
    assembly_view = relationship("AssemblyView", back_populates="component_assembly")
    position = relationship("Position", back_populates="component_assembly")

class AssemblyView(Base): # # TODO Rename to ComponentView
    __tablename__ = 'assembly_view'
    # location within component_assembly. example front,back,right-side top left ect...
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=True)
    description = Column(String, nullable=True)
    component_assembly_id = Column(Integer, ForeignKey('component_assembly.id'), nullable=False)
    # Relationships
    component_assembly = relationship("ComponentAssembly", back_populates="assembly_view")
    position = relationship("Position", back_populates="assembly_view")

class Part(Base):
    __tablename__ = 'part'

    id = Column(Integer, primary_key=True)
    part_number = Column(String, unique=True)  # MP2=ITEMNUM, SPC= Item Number
    name = Column(String)  # MP2=DESCRIPTION, SPC= Description
    oem_mfg = Column(String)  # MP2=OEMMFG, SPC= Manufacturer
    model = Column(String)  # MP2=MODEL, SPC= MFG Part Number
    class_flag = Column(String) # MP2=Class Flag SPC= Category
    ud6 = Column(String)  # MP2=UD6
    type = Column(String)  # MP2=TYPE
    notes = Column(String)  # MP2=Notes, SPC= Long Description
    documentation = Column(String)  # MP2=Specifications

    part_position_image = relationship("PartsPositionImageAssociation", back_populates="part")
    part_problem = relationship("PartProblemAssociation", back_populates="part")
    part_task = relationship("PartTaskAssociation", back_populates="part")
    drawing_part = relationship("DrawingPartAssociation", back_populates="part")

    __table_args__ = (UniqueConstraint('part_number', name='_part_number_uc'),)

# region
# Todo: create method for serving
# todo: create method for searching image
class Image(Base):
    __tablename__ = 'image'

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(String, nullable=False)
    file_path = Column(String, nullable=False)

    parts_position_image = relationship("PartsPositionImageAssociation", back_populates="image")
    image_problem = relationship("ImageProblemAssociation", back_populates="image")
    image_task = relationship("ImageTaskAssociation", back_populates="image")
    """bill_of_material = relationship("BillOfMaterial", back_populates="image")"""
    image_completed_document_association = relationship("ImageCompletedDocumentAssociation", back_populates="image")
    image_embedding = relationship("ImageEmbedding", back_populates="image")
    image_position_association = relationship("ImagePositionAssociation", back_populates= "image")
    tool_image_association = relationship("ToolImageAssociation", back_populates="image",cascade="all, delete-orphan")
# end region
class ImageEmbedding(Base):
    __tablename__ = 'image_embedding'

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('image.id'))
    model_name = Column(String, nullable=False)
    model_embedding = Column(LargeBinary, nullable=False)

    image = relationship("Image", back_populates="image_embedding")

class Drawing(Base):
    __tablename__ = 'drawing'

    id = Column(Integer, primary_key=True)
    drw_equipment_name = Column(String)
    drw_number = Column(String)
    drw_name = Column(String)
    drw_revision = Column(String)
    drw_spare_part_number = Column(String)
    file_path = Column(String, nullable=False)
    
    drawing_position = relationship("DrawingPositionAssociation", back_populates="drawing")
    drawing_problem = relationship("DrawingProblemAssociation", back_populates="drawing")
    drawing_task = relationship("DrawingTaskAssociation", back_populates="drawing")
    drawing_part = relationship("DrawingPartAssociation", back_populates="drawing")

class Document(Base):
    __tablename__ = 'document'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    content = Column(String)
    complete_document_id = Column(Integer, ForeignKey('complete_document.id'))
    embedding = Column(LargeBinary)
    rev = Column(String, nullable=False, default="R0")

    
    embeddings = relationship("DocumentEmbedding", back_populates="document")
    complete_document = relationship("CompleteDocument", back_populates="document")

class DocumentEmbedding(Base):
    __tablename__ = 'document_embedding'

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('document.id'))
    model_name = Column(String, nullable=False)
    model_embedding = Column(LargeBinary, nullable=False)

    document = relationship("Document", back_populates="embeddings")

class CompleteDocument(Base):
    __tablename__ = 'complete_document'

    id = Column(Integer, primary_key=True)
    title = Column(String)
    file_path = Column(String)
    content = Column(Text)
    rev = Column(String, nullable=False, default="R0")


    document = relationship("Document", back_populates="complete_document")
    completed_document_position_association = relationship("CompletedDocumentPositionAssociation", back_populates="complete_document")
    powerpoint = relationship("PowerPoint", back_populates="complete_document")
    image_completed_document_association = relationship("ImageCompletedDocumentAssociation", back_populates="complete_document")
    complete_document_problem = relationship("CompleteDocumentProblemAssociation", back_populates="complete_document")
    complete_document_task = relationship("CompleteDocumentTaskAssociation", back_populates="complete_document")

class Problem(Base):
    __tablename__ = 'problem'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=False)

    # Relationships
    solutions = relationship("Solution", back_populates="problem")  # One-to-many with solutions
    problem_position = relationship("ProblemPositionAssociation", back_populates="problem")
    image_problem = relationship("ImageProblemAssociation", back_populates="problem")
    complete_document_problem = relationship("CompleteDocumentProblemAssociation", back_populates="problem")
    drawing_problem = relationship("DrawingProblemAssociation", back_populates="problem")
    part_problem = relationship("PartProblemAssociation", back_populates="problem")

class Solution(Base):
    __tablename__ = 'solution'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)  # Solution name
    description = Column(String, nullable=False)
    problem_id = Column(Integer, ForeignKey('problem.id'))

    # Relationships
    problem = relationship("Problem", back_populates="solutions")
    task_solutions = relationship("TaskSolutionAssociation", back_populates="solution", cascade="all, delete-orphan")

class Task(Base):
    __tablename__ = 'task'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=False)

    # Relationships
    task_positions = relationship("TaskPositionAssociation", back_populates="task", cascade="all, delete-orphan")
    task_solutions = relationship("TaskSolutionAssociation", back_populates="task", cascade="all, delete-orphan")
    image_task = relationship("ImageTaskAssociation", back_populates="task")
    complete_document_task = relationship("CompleteDocumentTaskAssociation", back_populates="task")
    drawing_task = relationship("DrawingTaskAssociation", back_populates="task")
    part_task = relationship("PartTaskAssociation", back_populates="task")
    tool_tasks = relationship("TaskToolAssociation", back_populates="task", cascade="all, delete-orphan")

class TaskSolutionAssociation(Base):
    __tablename__ = 'task_solution_association'

    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, ForeignKey('task.id'))
    solution_id = Column(Integer, ForeignKey('solution.id'))

    # Relationships
    task = relationship("Task", back_populates="task_solutions")
    solution = relationship("Solution", back_populates="task_solutions")

class PowerPoint(Base):
    __tablename__ = 'powerpoint'

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    ppt_file_path = Column(String, nullable=False)
    pdf_file_path = Column(String, nullable=False)
    description = Column(String, nullable=True)
    complete_document_id = Column(Integer, ForeignKey('complete_document.id'))
    rev = Column(String, nullable=True)

    complete_document = relationship("CompleteDocument", back_populates="powerpoint")

    def __init__(self, title, ppt_file_path, pdf_file_path, complete_document_id, description=None):
        self.title = title
        self.ppt_file_path = ppt_file_path
        self.pdf_file_path = pdf_file_path
        self.complete_document_id = complete_document_id
        self.description = description

# Junction Classes 
class DrawingPartAssociation(Base):
    __tablename__ = 'drawing_part'
    id = Column(Integer, primary_key=True)
    drawing_id = Column(Integer, ForeignKey('drawing.id'))
    part_id = Column(Integer, ForeignKey('part.id'))
    
    drawing = relationship("Drawing", back_populates="drawing_part")
    part = relationship("Part", back_populates="drawing_part")
    
class PartProblemAssociation(Base):
    __tablename__ = 'part_problem'
    id = Column(Integer, primary_key=True)
    part_id = Column(Integer, ForeignKey('part.id'))
    problem_id = Column(Integer, ForeignKey('problem.id'))
    
    part = relationship("Part", back_populates="part_problem")
    problem = relationship("Problem", back_populates="part_problem")

class TaskPositionAssociation(Base):
    __tablename__ = 'task_position'

    id = Column(Integer, primary_key=True)
    task_id = Column(Integer, ForeignKey('task.id'), nullable=False)
    position_id = Column(Integer, ForeignKey('position.id'), nullable=False)

    # Define relationships
    task = relationship("Task", back_populates="task_positions")
    position = relationship("Position", back_populates="position_tasks")

class PartTaskAssociation(Base):
    __tablename__ = 'part_task'

    id = Column(Integer, primary_key=True)
    part_id = Column(Integer, ForeignKey('part.id'))
    task_id = Column(Integer, ForeignKey('task.id'))  # Corrected foreign key

    part = relationship("Part", back_populates="part_task")
    task = relationship("Task", back_populates="part_task")

class DrawingTaskAssociation(Base):
    __tablename__ = 'drawing_task'
    id = Column(Integer, primary_key=True)
    drawing_id = Column(Integer, ForeignKey('drawing.id'))
    task_id = Column(Integer, ForeignKey('task.id'))

    drawing = relationship("Drawing", back_populates="drawing_task")
    task = relationship("Task", back_populates="drawing_task")

class ImageTaskAssociation(Base):
    __tablename__ = 'image_task'

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('image.id'))
    task_id = Column(Integer, ForeignKey('task.id'))  # Corrected foreign key

    image = relationship("Image", back_populates="image_task")
    task = relationship("Task", back_populates="image_task")

class TaskToolAssociation(Base):
    __tablename__ = 'tool_task'

    id = Column(Integer, primary_key=True)
    tool_id = Column(Integer, ForeignKey('tool.id'), nullable=False)
    task_id = Column(Integer, ForeignKey('task.id'), nullable=False)

    # Relationships
    tool = relationship("Tool", back_populates="tool_tasks")
    task = relationship("Task", back_populates="tool_tasks")

class DrawingProblemAssociation(Base):
    __tablename__ = 'drawing_problem'
    id = Column(Integer, primary_key=True)
    drawing_id = Column(Integer, ForeignKey('drawing.id'))
    problem_id = Column(Integer, ForeignKey('problem.id'))
    
    drawing = relationship("Drawing", back_populates="drawing_problem")
    problem = relationship("Problem", back_populates="drawing_problem")

class BillOfMaterial(Base):
    __tablename__ = 'bill_of_material'
    id = Column(Integer, primary_key=True)
    part_position_image_id = Column(Integer, ForeignKey('part_position_image.id'))  # Corrected line
    """part_id = Column(Integer, ForeignKey('part.id'))
    position_id = Column(Integer, ForeignKey('position.id'))"""
    quantity = Column(Float, nullable=False)  # Corrected to Float
    comment = Column(String)

    part_position_image = relationship("PartsPositionImageAssociation", back_populates="bill_of_material")
    """part = relationship("Part", back_populates="bill_of_material")
    position = relationship("Position", back_populates="bill_of_material")
    image = relationship("Image", back_populates="bill_of_material")"""

class ProblemPositionAssociation(Base):
    __tablename__ = 'problem_position'
    id = Column(Integer, primary_key=True)
    problem_id = Column(Integer, ForeignKey('problem.id'))
    position_id = Column(Integer, ForeignKey('position.id'))
    
    problem = relationship("Problem", back_populates="problem_position")
    position = relationship("Position", back_populates="problem_position")

class CompleteDocumentProblemAssociation(Base):
    __tablename__ = 'complete_document_problem'
    
    id = Column(Integer, primary_key=True)
    complete_document_id = Column(Integer, ForeignKey('complete_document.id'))
    problem_id = Column(Integer, ForeignKey('problem.id'))
    
    complete_document = relationship("CompleteDocument", back_populates="complete_document_problem")
    problem = relationship("Problem", back_populates="complete_document_problem")
    
class CompleteDocumentTaskAssociation(Base):
    __tablename__ = 'complete_document_task'
    
    id = Column(Integer, primary_key=True)
    complete_document_id = Column(Integer, ForeignKey('complete_document.id'))
    task_id = Column(Integer, ForeignKey('task.id'))
    
    complete_document = relationship("CompleteDocument", back_populates="complete_document_task")
    task = relationship("Task", back_populates="complete_document_task")

class ImageProblemAssociation(Base):
    __tablename__ = 'image_problem'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('image.id'))
    problem_id = Column(Integer, ForeignKey('problem.id'))
    
    image = relationship("Image", back_populates="image_problem")
    problem = relationship("Problem", back_populates="image_problem")

class PartsPositionImageAssociation(Base):
    __tablename__ = 'part_position_image'
    id = Column(Integer, primary_key=True)
    part_id = Column(Integer, ForeignKey('part.id'))
    position_id = Column(Integer, ForeignKey('position.id'))
    image_id = Column(Integer, ForeignKey('image.id'))

    part = relationship("Part", back_populates="part_position_image")
    position = relationship("Position", back_populates="part_position_image")
    image = relationship("Image", back_populates="parts_position_image")
    bill_of_material = relationship("BillOfMaterial", back_populates="part_position_image")

class ImagePositionAssociation(Base):
    __tablename__ = 'image_position_association'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('image.id'))
    position_id = Column(Integer, ForeignKey('position.id'))
    
    image = relationship("Image", back_populates="image_position_association")
    position = relationship("Position", back_populates="image_position_association")

class DrawingPositionAssociation(Base):
    __tablename__ = 'drawing_position'
    id = Column(Integer, primary_key=True)
    drawing_id = Column(Integer, ForeignKey('drawing.id'))
    position_id = Column(Integer, ForeignKey('position.id'))
    
    drawing = relationship("Drawing", back_populates="drawing_position")
    position = relationship("Position", back_populates="drawing_position")

class CompletedDocumentPositionAssociation(Base):
    __tablename__ = 'completed_document_position_association'
    id = Column(Integer, primary_key=True)
    complete_document_id = Column(Integer, ForeignKey('complete_document.id'))
    position_id = Column(Integer, ForeignKey('position.id'))

    complete_document = relationship("CompleteDocument", back_populates="completed_document_position_association")
    position = relationship("Position", back_populates="completed_document_position_association")

class ImageCompletedDocumentAssociation(Base):
    __tablename__ = 'image_completed_document_association'

    id = Column(Integer, primary_key=True)
    complete_document_id = Column(Integer, ForeignKey('complete_document.id'))
    image_id = Column(Integer, ForeignKey('image.id'))
    
    complete_document = relationship("CompleteDocument", back_populates="image_completed_document_association")
    image = relationship("Image", back_populates="image_completed_document_association")
    
# Process Classes
class FileLog(Base):
    __tablename__ = 'file_logs'
    log_id = Column(Integer, primary_key=True, autoincrement=True)
    session = Column(Integer, nullable=False)
    session_datetime = Column(DateTime, nullable=False)
    file_processed = Column(String)  # Added column for file processed
    total_time = Column(String)

class KeywordAction(Base):
    __tablename__ = 'keyword_actions'

    id = Column(Integer, primary_key=True)
    keyword = Column(String, unique=True)
    action = Column(String)

    # Manually define the query attribute
    query = scoped_session(session).query_property()

    @classmethod
    def find_best_match(cls, user_input, session):
        try:
            # Retrieve all keywords from the database
            all_keywords = [keyword.keyword for keyword in session.query(cls).all()]

            # Use fuzzy string matching to find the best matching keyword
            logger.debug("All keywords: %s", all_keywords)
            matched_keyword, similarity_score = process.extractOne(user_input, all_keywords)
            logger.debug("Matched keyword: %s", matched_keyword)
            logger.debug("Similarity score: %s", similarity_score)

            # Set a threshold for the minimum similarity score
            threshold = 50

            # If the similarity score exceeds the threshold, return the matched keyword and its associated action
            if similarity_score >= threshold:
                # Extract keyword and details using spaCy
                keyword, details = extract_keyword_and_details(user_input)
                if keyword:
                    # Retrieve the associated action from the database using the matched keyword
                    keyword_entry = session.query(cls).filter_by(keyword=keyword).first()
                    if keyword_entry:
                        action = keyword_entry.action
                        logger.debug("Associated action: %s", action)
                        return keyword, action, None  # No need to extract details

            # If no matching keyword is found or similarity score is below threshold, return None
            logger.debug("No matching keyword found or similarity score is below threshold.")
            return None, None, None

        except SQLAlchemyError as e:
            # Handle SQLAlchemy errors
            logger.error("Database error: %s", e)
            return None, None, None

        except Exception as e:
            # Handle other unexpected errors
            logger.error("Unexpected error: %s", e)
            return None, None, None

class ChatSession(Base):
    __tablename__ = 'chat_sessions'
    session_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False)
    start_time = Column(String, nullable=False)
    last_interaction = Column(String, nullable=False)
    session_data = Column(MutableList.as_mutable(JSON), default=[])  # Initialize as empty list
    conversation_summary = Column(MutableList.as_mutable(JSON), default=[])  # New column for conversation summary

    def __init__(self, user_id, start_time, last_interaction, session_data=None, conversation_summary=None):
        self.user_id = user_id
        self.start_time = start_time
        self.last_interaction = last_interaction
        if session_data is None:
            session_data = []  # Initialize session_data as empty list if not provided
        self.session_data = session_data
        if conversation_summary is None:
            conversation_summary = []  # Initialize conversation_summary as empty list if not provided
        self.conversation_summary = conversation_summary
      
class QandA(Base):
    __tablename__ = 'qanda'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String)
    question = Column(String)
    answer = Column(String)
    comment = Column(String)
    rating = Column(String)
    timestamp = Column(String, nullable=False)
    
    def __init__(self, user_id, question, answer, timestamp, rating=None, comment=None):
        self.user_id = user_id
        self.question = question
        self.answer = answer
        self.timestamp = timestamp
        self.rating = rating
        self.comment = comment      

class UserLevel(PyEnum):
    ADMIN = 'ADMIN'
    LEVEL_III = 'LEVEL_III'
    LEVEL_II = 'LEVEL_II'
    STANDARD = 'STANDARD'

class AIModelConfig(Base):
    __tablename__ = 'ai_model_config'

    id = Column(Integer, primary_key=True)
    key = Column(String, unique=True, nullable=False)
    value = Column(String, nullable=False)

class ImageModelConfig(Base):
    __tablename__ = 'image_model_config'

    id = Column(Integer, primary_key=True)
    key = Column(String, unique=True, nullable=False)
    value = Column(String, nullable=False)

# Define the User model
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    employee_id = Column(String, unique=True, nullable=False)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    current_shift = Column(String, nullable=True)
    primary_area = Column(String, nullable=True)
    age = Column(Integer, nullable=True)
    education_level = Column(String, nullable=True)
    start_date = Column(DateTime, nullable=True)
    hashed_password = Column(String, nullable=False)

    # Store enum as string in the database
    user_level = Column(SqlEnum(UserLevel, values_callable=lambda obj: [e.value for e in obj]),
                        default=UserLevel.STANDARD, nullable=False)

    # Relationship to comments
    comments = relationship("UserComments", back_populates="user")

    def set_password(self, password):
        self.hashed_password = generate_password_hash(password)

    def check_password_hash(self, password):
        return check_password_hash(self.hashed_password, password)

# Define the UserComments model
class UserComments(Base):
    __tablename__ = 'user_comments'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    comment = Column(Text, nullable=False)
    page_url = Column(String, nullable=False)
    screenshot_path = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationship to User
    user = relationship("User", back_populates="comments")

class BOMResult(Base):
    __tablename__ = 'bom_result'
    id = Column(Integer, primary_key=True)
    part_id = Column(Integer, ForeignKey('part.id'), nullable=False)
    position_id = Column(Integer, ForeignKey('position.id'), nullable=False)
    image_id = Column(Integer, ForeignKey('image.id'), nullable=True)
    description = Column(String)

    part = relationship('Part', lazy='joined')
    image = relationship('Image', lazy='joined')

#class's dealing with tools
class ToolImageAssociation(Base):
    __tablename__ = 'tool_image_association'
    id = Column(Integer, primary_key=True)
    tool_id = Column(Integer, ForeignKey('tool.id'))
    image_id = Column(Integer, ForeignKey('image.id'))
    description = Column(Text, nullable=True)

    # Relationships
    tool = relationship('Tool', back_populates='tool_image_association')
    image = relationship('Image', back_populates='tool_image_association')

class ToolPositionAssociation(Base):
    __tablename__ = 'tool_position_association'
    id = Column(Integer, primary_key=True)
    tool_id = Column(Integer, ForeignKey('tool.id'))
    position_id = Column(Integer, ForeignKey('position.id'))
    description = Column(Text, nullable=True)
    tool = relationship('Tool', back_populates='tool_position_association')
    position = relationship('Position', back_populates='tool_position_association')

class ToolCategory(Base):
    __tablename__ = 'tool_category'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    description = Column(Text)
    parent_id = Column(Integer, ForeignKey('tool_category.id'), nullable=True)

    # Self-referential relationships for hierarchy
    parent = relationship('ToolCategory', remote_side=[id], back_populates='subcategories')
    subcategories = relationship('ToolCategory', back_populates='parent', cascade="all, delete-orphan")
    tools = relationship('Tool', back_populates='tool_category', cascade="all, delete-orphan")

tool_package_association = Table(
    'tool_package_association',
    Base.metadata,
    Column('tool_id', Integer, ForeignKey('tool.id'), primary_key=True),
    Column('package_id', Integer, ForeignKey('tool_package.id'), primary_key=True),
    Column('quantity', Integer, nullable=False, default=1)
)

class ToolManufacturer(Base):
    __tablename__ = 'tool_manufacturer'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    description =Column(String, nullable=True)
    country = Column(String, nullable=True)
    website = Column(String, nullable=True)

    tools = relationship('Tool', back_populates='tool_manufacturer')

class Tool(Base):
    __tablename__ = 'tool'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    size = Column(String, nullable=True)
    type = Column(String, nullable=True)
    material = Column(String, nullable=True)
    description = Column(Text)
    tool_category_id = Column(Integer, ForeignKey('tool_category.id'))
    tool_manufacturer_id = Column(Integer, ForeignKey('tool_manufacturer.id'))

    # Relationships
    tool_category = relationship('ToolCategory', back_populates='tools')
    tool_manufacturer = relationship('ToolManufacturer', back_populates='tools')
    tool_packages = relationship('ToolPackage', secondary=tool_package_association, back_populates='tools')
    tool_image_association = relationship('ToolImageAssociation', back_populates='tool')
    tool_position_association = relationship('ToolPositionAssociation',back_populates='tool',)
    tool_tasks = relationship('TaskToolAssociation', back_populates='tool', cascade="all, delete-orphan")

class ToolPackage(Base):
    __tablename__ = 'tool_package'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)

    tools = relationship('Tool', secondary=tool_package_association, back_populates='tool_packages')

# Bind the engine to the Base class
Base.metadata.bind = engine

# Then call create_all()
Base.metadata.create_all(engine, checkfirst=True)

# Function to load config from the database
def load_config_from_db():
    session = Session()
    ai_model_config = session.query(AIModelConfig).filter_by(key="CURRENT_AI_MODEL").first()
    embedding_model_config = session.query(AIModelConfig).filter_by(key="CURRENT_EMBEDDING_MODEL").first()
    session.close()

    current_ai_model = ai_model_config.value if ai_model_config else "NoAIModel"
    current_embedding_model = embedding_model_config.value if embedding_model_config else "NoEmbeddingModel"

    return current_ai_model, current_embedding_model

def load_image_model_config_from_db():
    session = Session()
    image_model_config = session.query(ImageModelConfig).filter_by(key="CURRENT_IMAGE_MODEL").first()
    session.close()

    current_image_model = image_model_config.value if image_model_config else "no_model"

    return current_image_model

# Create the 'documents_fts' table for full-text search
with Session() as session:
    sql_statement = text("CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING FTS5(title, content)")
    session.execute(sql_statement)
    session.commit()

# Ask for API key if it's empty
if not OPENAI_API_KEY:
    OPENAI_API_KEY = input("Enter your OpenAI API key: ")
    with open('../configuration/config.py', 'w') as config_file:
        config_file.write(f'BASE_DIR = "{BASE_DIR}"\n')
        config_file.write(f'copy_files = {COPY_FILES}\n')
        config_file.write(f'OPENAI_API_KEY = "{OPENAI_API_KEY}"\n')
