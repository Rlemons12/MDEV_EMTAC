import os
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
import logging
import shutil
from datetime import datetime
from typing import Optional,List
from sqlalchemy import or_, and_
from sqlalchemy.orm import Session
from typing import List, Optional
from typing import Dict, Any, List, Optional, Tuple, Union
from flask import send_file, jsonify, request, abort, flash, redirect, url_for, render_template
from sqlalchemy import desc, asc
from werkzeug.utils import secure_filename
import mimetypes
from PIL import Image as PILImage


import openai
import spacy
from fuzzywuzzy import process
from werkzeug.security import check_password_hash, generate_password_hash
from sqlalchemy import (DateTime, Column, ForeignKey, Integer, JSON, LargeBinary, Enum as SqlEnum, Boolean,
                        String, create_engine, text, Float, Text, UniqueConstraint, and_, Table)
from enum import Enum as PyEnum  # Import Enum and alias it as PyEnum
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.orm import declarative_base, configure_mappers, relationship, scoped_session, sessionmaker
from modules.configuration.config import (OPENAI_API_KEY, BASE_DIR, COPY_FILES, DATABASE_URL,DATABASE_PATH)
from modules.configuration.base import Base
from modules.configuration.log_config import *
from modules.configuration.config import DATABASE_DIR
from modules.configuration.config_env import DatabaseConfig
from flask import g  # Required for access to g.request_id in the methods
from functools import wraps  # Required if you need to recreate with_request_id

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

    @classmethod
    @with_request_id
    def add_site_location(cls, session, title, room_number, site_area, request_id=None):
        """
        Add a new site location to the database.

        Args:
            session: SQLAlchemy database session
            title (str): Title of the site location
            room_number (str): Room number of the site location
            site_area (str): Site area of the site location
            request_id (str, optional): Unique identifier for the request

        Returns:
            SiteLocation: The newly created site location object
        """
        new_site_location = cls(
            title=title,
            room_number=room_number,
            site_area=site_area
        )

        session.add(new_site_location)
        session.commit()

        logger.info(f"Created new site location: '{title}' in room {room_number}, area {site_area}")
        return new_site_location

    @classmethod
    @with_request_id
    def delete_site_location(cls, session, site_location_id, request_id=None):
        """
        Delete a site location from the database.

        Args:
            session: SQLAlchemy database session
            site_location_id (int): ID of the site location to delete
            request_id (str, optional): Unique identifier for the request

        Returns:
            bool: True if deletion was successful, False if site location not found
        """
        site_location = session.query(cls).filter(cls.id == site_location_id).first()

        if site_location:
            session.delete(site_location)
            session.commit()
            logger.info(f"Deleted site location ID {site_location_id}")
            return True
        else:
            logger.warning(f"Failed to delete site location ID {site_location_id} - not found")
            return False

    @classmethod
    @with_request_id
    def find_related_entities(cls, session, identifier, is_id=True, request_id=None):
        """
        Find all related entities for a site location.

        Args:
            session: SQLAlchemy database session
            identifier: Either site location ID (int) or title (str)
            is_id (bool): If True, identifier is an ID, otherwise it's a title
            request_id (str, optional): Unique identifier for the request

        Returns:
            dict: Dictionary containing:
                - 'site_location': The found site location object
                - 'downward': Dictionary containing:
                    - 'positions': List of all positions at this site location
        """
        # Find the site location
        if is_id:
            site_location = session.query(cls).filter(cls.id == identifier).first()
        else:
            site_location = session.query(cls).filter(cls.title == identifier).first()

        if not site_location:
            logger.warning(f"Site location not found for identifier: {identifier}")
            return None

        # Going downward in the hierarchy
        downward = {
            'positions': site_location.position
        }

        logger.info(f"Found related entities for site location ID {site_location.id}")
        return {
            'site_location': site_location,
            'downward': downward
        }

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

    # Hierarchy definition
    # Define HIERARCHY using string names instead of direct class references
    HIERARCHY = {
        'area': {
            'model': 'EquipmentGroup',
            'filter_field': 'area_id',
            'order_field': 'name',
            'next_level': 'equipment_group'
        },
        'equipment_group': {
            'model': 'Model',
            'filter_field': 'equipment_group_id',
            'order_field': 'name',
            'next_level': 'model'
        },
        'model': {
            # Models have two potential child types - asset_number and location
            'child_types': [
                {
                    'model': 'AssetNumber',
                    'filter_field': 'model_id',
                    'order_field': 'number',
                    'next_level': 'asset_number'
                },
                {
                    'model': 'Location',
                    'filter_field': 'model_id',
                    'order_field': 'name',
                    'next_level': 'location'
                }
            ]
        },
        'location': {
            'model': 'Subassembly',
            'filter_field': 'location_id',
            'order_field': 'name',
            'next_level': 'subassembly'
        },
        'subassembly': {
            'model': 'ComponentAssembly',
            'filter_field': 'subassembly_id',
            'order_field': 'name',
            'next_level': 'component_assembly'
        },
        'component_assembly': {
            'model': 'AssemblyView',
            'filter_field': 'component_assembly_id',
            'order_field': 'name',
            'next_level': 'assembly_view'
        }
    }

    # Model mapping - defined once for efficiency
    MODELS_MAP = None

    @classmethod
    @with_request_id
    def get_dependent_items(cls, session, parent_type, parent_id, child_type=None):
        """
        Generic method to get dependent items based on parent type and ID.

        Args:
            session: SQLAlchemy session
            parent_type: The type of the parent (e.g., 'area', 'equipment_group')
            parent_id: The ID of the parent
            child_type: Optional, to specify which child type to return when parent has multiple child types

        Returns:
            List of dependent items
        """
        if not parent_id:
            return []

        # Get parent configuration from hierarchy
        parent_config = cls.HIERARCHY.get(parent_type)
        if not parent_config:
            return []

        # Handle parents with multiple child types
        if 'child_types' in parent_config:
            if child_type:
                # Find the specific child type configuration
                for child_config in parent_config['child_types']:
                    if child_config.get('next_level') == child_type:
                        return cls._fetch_dependent_items(session, child_config, parent_id)
                return []
            else:
                # Return the first child type by default
                return cls._fetch_dependent_items(session, parent_config['child_types'][0], parent_id)
        else:
            # Standard single child type
            return cls._fetch_dependent_items(session, parent_config, parent_id)

    @staticmethod
    def _fetch_dependent_items(session, config, parent_id):
        """
        Helper method to fetch dependent items based on configuration.

        Args:
            session: SQLAlchemy session
            config: Configuration dictionary with model, filter_field, order_field
            parent_id: The ID of the parent

        Returns:
            List of dependent items
        """
        model_name = config.get('model')
        filter_field = config.get('filter_field')
        order_field = config.get('order_field')

        if not all([model_name, filter_field, order_field]):
            return []

        # Get the actual model class from its name
        if isinstance(model_name, str):
            # Use globals() to find the class by name
            model = globals().get(model_name)
            if not model:
                # Alternative approach - if globals() doesn't work, you can use a mapping
                models_map = {
                    'EquipmentGroup': EquipmentGroup,
                    'Model': Model,
                    'AssetNumber': AssetNumber,
                    'Location': Location,
                    'Subassembly': Subassembly,
                    'ComponentAssembly': ComponentAssembly,
                    'AssemblyView': AssemblyView,
                    'SiteLocation': SiteLocation
                }
                model = models_map.get(model_name)
                if not model:
                    return []
        else:
            model = model_name  # Already a class

        query = session.query(model).filter_by(**{filter_field: parent_id})

        # Apply ordering
        order_attr = getattr(model, order_field)
        query = query.order_by(order_attr)

        return query.all()

    @classmethod
    @with_request_id
    def get_next_level_type(cls, current_level):
        """Get the next level type in the hierarchy"""
        config = cls.HIERARCHY.get(current_level)
        if not config:
            return None

        if 'child_types' in config:
            # Return the first child type by default
            return config['child_types'][0].get('next_level')
        else:
            return config.get('next_level')

    @classmethod
    @with_request_id
    def add_to_db(cls, session=None, area_id=None, equipment_group_id=None, model_id=None, asset_number_id=None,
                  location_id=None, subassembly_id=None, component_assembly_id=None, assembly_view_id=None,
                  site_location_id=None,):
        """
        Get-or-create a Position with exactly these FK values.
        If `session` is None, uses DatabaseConfig().get_main_session().
        Returns the Position instance (new or existing).
        """
        # 1) ensure we have a session
        if session is None:
            session = DatabaseConfig().get_main_session()

        # 2) log input parameters
        debug_id(
            "add_to_db called with "
            "area_id=%s, equipment_group_id=%s, model_id=%s, "
            "asset_number_id=%s, location_id=%s, subassembly_id=%s, "
            "component_assembly_id=%s, assembly_view_id=%s, site_location_id=%s",
            area_id, equipment_group_id, model_id,
            asset_number_id, location_id, subassembly_id,
            component_assembly_id, assembly_view_id, site_location_id,
        )

        # 3) build filter dict
        filters = {
            "area_id": area_id,
            "equipment_group_id": equipment_group_id,
            "model_id": model_id,
            "asset_number_id": asset_number_id,
            "location_id": location_id,
            "subassembly_id": subassembly_id,
            "component_assembly_id": component_assembly_id,
            "assembly_view_id": assembly_view_id,
            "site_location_id": site_location_id,
        }

        try:
            # 4) try to find an existing row
            existing = session.query(cls).filter_by(**filters).first()
            if existing:
                info_id("Found existing Position id=%s", existing.id)
                return existing

            # 5) not found → create new
            position = cls(**filters)
            session.add(position)
            session.commit()
            info_id("Created new Position id=%s", position.id)
            return position

        except SQLAlchemyError as e:
            session.rollback()
            error_id("Failed to add_or_get Position: %s", e, exc_info=True)
            raise

    @classmethod
    @with_request_id
    def get_corresponding_position_ids(cls, session=None, area_id=None, equipment_group_id=None,
                                       model_id=None, asset_number_id=None, location_id=None,
                                       request_id='no_request_id'):
        """
        Search for corresponding Position IDs based on the provided filters with request ID logging.

        Args:
            session: SQLAlchemy session (Optional)
            area_id: ID of the area (optional)
            equipment_group_id: ID of the equipment group (optional)
            model_id: ID of the model (optional)
            asset_number_id: ID of the asset number (optional)
            location_id: ID of the location (optional)
            request_id: Unique identifier for the request

        Returns:
            List of Position IDs that match the criteria
        """
        # Ensure a session is available, if not use DatabaseConfig to get it
        if session is None:
            session = DatabaseConfig().get_main_session()

        # Log input parameters with request ID
        logging.info(
            f"[{request_id}] get_corresponding_position_ids called with "
            f"area_id={area_id}, equipment_group_id={equipment_group_id}, "
            f"model_id={model_id}, asset_number_id={asset_number_id}, "
            f"location_id={location_id}"
        )

        try:
            # Start by fetching the root-level positions based on hierarchy
            positions = cls._get_positions_by_hierarchy(
                session,
                area_id=area_id,
                equipment_group_id=equipment_group_id,
                model_id=model_id,
                asset_number_id=asset_number_id,
                location_id=location_id,
                request_id=request_id
            )

            # Extract Position IDs
            position_ids = [position.id for position in positions]

            # Log the result
            logging.info(f"[{request_id}] Retrieved {len(position_ids)} Position IDs")
            return position_ids

        except SQLAlchemyError as e:
            # Log any errors encountered during the query
            logging.error(
                f"[{request_id}] Error in get_corresponding_position_ids: {str(e)}",
                exc_info=True
            )
            raise

    @classmethod
    @with_request_id
    def _get_positions_by_hierarchy(cls, session, area_id=None, equipment_group_id=None, model_id=None,
                                    asset_number_id=None, location_id=None):
        """
        Helper method to fetch positions based on hierarchical filters.

        Args:
            session: SQLAlchemy session
            area_id, equipment_group_id, model_id, asset_number_id, location_id: IDs for filtering

        Returns:
            List of Position objects that match the criteria
        """
        # Building the filter dynamically based on input parameters
        filters = {}
        if area_id:
            filters['area_id'] = area_id
        if equipment_group_id:
            filters['equipment_group_id'] = equipment_group_id
        if model_id:
            filters['model_id'] = model_id
        if asset_number_id:
            filters['asset_number_id'] = asset_number_id
        if location_id:
            filters['location_id'] = location_id

        # Log the filter parameters
        debug_id(f"Filtering Positions with filters: {filters}", request_id=g.request_id)

        try:
            # Query the Position table based on the filters
            query = session.query(Position).filter_by(**filters)

            # Log the query execution
            info_id(f"Executing query for positions with {len(filters)} filters.", request_id=g.request_id)

            # Return the positions matching the filter
            positions = query.all()

            # Log the result
            info_id(f"Retrieved {len(positions)} positions.", request_id=g.request_id)
            return positions

        except SQLAlchemyError as e:
            # Log any errors encountered during the query
            error_id(f"Error in _get_positions_by_hierarchy: {str(e)}", exc_info=True, request_id=g.request_id)
            raise

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

    @classmethod
    @with_request_id
    def add_equipment_group(cls, session, name, area_id, description=None, request_id=None):
        """
        Add a new equipment group to the database.

        Args:
            session: SQLAlchemy database session
            name (str): Name of the equipment group
            area_id (int): ID of the area this equipment group belongs to
            description (str, optional): Description of the equipment group
            request_id (str, optional): Unique identifier for the request

        Returns:
            EquipmentGroup: The newly created equipment group object
        """
        new_equipment_group = cls(
            name=name,
            area_id=area_id,
            description=description
        )

        session.add(new_equipment_group)
        session.commit()

        return new_equipment_group

    @classmethod
    @with_request_id
    def delete_equipment_group(cls, session, equipment_group_id, request_id=None):
        """
        Delete an equipment group from the database.

        Args:
            session: SQLAlchemy database session
            equipment_group_id (int): ID of the equipment group to delete
            request_id (str, optional): Unique identifier for the request

        Returns:
            bool: True if deletion was successful, False if equipment group not found
        """
        equipment_group = session.query(cls).filter(cls.id == equipment_group_id).first()

        if equipment_group:
            session.delete(equipment_group)
            session.commit()
            return True
        else:
            return False

    @classmethod
    @with_request_id
    def find_related_entities(cls, session, identifier, is_id=True, request_id=None):
        """
        Find all related entities for an equipment group, traversing both up and down
        the hierarchy: Area → EquipmentGroup → Model → (AssetNumber, Location, Position).

        Args:
            session: SQLAlchemy database session
            identifier: Either equipment_group ID (int) or name (str)
            is_id (bool): If True, identifier is an ID, otherwise it's a name
            request_id (str, optional): Unique identifier for the request

        Returns:
            dict: Dictionary containing:
                - 'equipment_group': The found equipment group object
                - 'upward': Dictionary containing 'area' the equipment group belongs to
                - 'downward': Dictionary containing:
                    - 'models': List of all models belonging to this equipment group
                    - 'positions': List of all positions directly related to this equipment group
        """
        # Find the equipment group
        if is_id:
            equipment_group = session.query(cls).filter(cls.id == identifier).first()
        else:
            equipment_group = session.query(cls).filter(cls.name == identifier).first()

        if not equipment_group:
            return None

        # Going upward in the hierarchy
        upward = {
            'area': equipment_group.area
        }

        # Going downward in the hierarchy
        downward = {
            'models': equipment_group.model,
            'positions': equipment_group.position
        }

        # Collecting more detailed information from models if needed
        model_details = []
        for model in equipment_group.model:
            model_info = {
                'id': model.id,
                'name': model.name,
                'description': model.description,
                'asset_numbers': model.asset_number,
                'locations': model.location,
                'positions': model.position
            }
            model_details.append(model_info)

        downward['model_details'] = model_details

        return {
            'equipment_group': equipment_group,
            'upward': upward,
            'downward': downward
        }

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

    @classmethod
    @with_request_id
    def search_models(cls, session, query, limit=10):
        """
        Searches for models that match the provided query using a case-insensitive
        partial match on the name field. Useful for autocomplete or dynamic search interfaces.

        Parameters:
            session: SQLAlchemy session object used for querying.
            query: The partial model name input by the user.
            limit: Maximum number of results to return (default is 10).

        Returns:
            A list of dictionaries, each containing details about a model:
              - id: The model's unique identifier.
              - name: The model's name.
              - description: The model's description.
              - equipment_group_id: The associated equipment group ID.
            If no records match, an empty list is returned.
        """
        logger.info("========== MODEL AUTOCOMPLETE SEARCH ==========")
        logger.debug(f"Initiating search for models with query: '{query}'")

        try:
            if not query:
                logger.debug("Empty query received; returning empty result set.")
                return []

            search_pattern = f"%{query}%"
            logger.debug(f"Using search pattern: '{search_pattern}'")

            results = session.query(cls).filter(cls.name.ilike(search_pattern)).limit(limit).all()

            if results:
                models = []
                for model in results:
                    model_details = {
                        "id": model.id,
                        "name": model.name,
                        "description": model.description,
                        "equipment_group_id": model.equipment_group_id
                    }
                    models.append(model_details)
                    logger.debug(f"Found model: {model_details}")
                logger.info(f"Found {len(models)} model(s) matching query '{query}'.")
                return models
            else:
                logger.warning(f"No models found matching query '{query}'.")
                return []
        except Exception as e:
            logger.error(f"Error searching for models with query '{query}': {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        finally:
            logger.info("========== MODEL AUTOCOMPLETE SEARCH COMPLETE ==========")

    @classmethod
    @with_request_id
    def add_model(cls, session, name, equipment_group_id, description=None, request_id=None):
        """
        Add a new model to the database.

        Args:
            session: SQLAlchemy database session
            name (str): Name of the model
            equipment_group_id (int): ID of the equipment group this model belongs to
            description (str, optional): Description of the model
            request_id (str, optional): Unique identifier for the request

        Returns:
            Model: The newly created model object
        """
        new_model = cls(
            name=name,
            equipment_group_id=equipment_group_id,
            description=description
        )

        session.add(new_model)
        session.commit()

        return new_model

    @classmethod
    @with_request_id
    def delete_model(cls, session, model_id, request_id=None):
        """
        Delete a model from the database.

        Args:
            session: SQLAlchemy database session
            model_id (int): ID of the model to delete
            request_id (str, optional): Unique identifier for the request

        Returns:
            bool: True if deletion was successful, False if model not found
        """
        model = session.query(cls).filter(cls.id == model_id).first()

        if model:
            session.delete(model)
            session.commit()
            return True
        else:
            return False

    @classmethod
    @with_request_id
    def find_related_entities(cls, session, identifier, is_id=True, request_id=None):
        """
        Find all related entities for a model, traversing both up and down
        the hierarchy: Area → EquipmentGroup → Model → (AssetNumber, Location, Position).

        Args:
            session: SQLAlchemy database session
            identifier: Either model ID (int) or name (str)
            is_id (bool): If True, identifier is an ID, otherwise it's a name
            request_id (str, optional): Unique identifier for the request

        Returns:
            dict: Dictionary containing:
                - 'model': The found model object
                - 'upward': Dictionary containing 'equipment_group' and 'area'
                - 'downward': Dictionary containing:
                    - 'asset_numbers': List of all asset numbers belonging to this model
                    - 'locations': List of all locations for this model
                    - 'positions': List of all positions related to this model
        """
        # Find the model
        if is_id:
            model = session.query(cls).filter(cls.id == identifier).first()
        else:
            model = session.query(cls).filter(cls.name == identifier).first()

        if not model:
            return None

        # Going upward in the hierarchy
        upward = {
            'equipment_group': model.equipment_group,
            'area': model.equipment_group.area if model.equipment_group else None
        }

        # Going downward in the hierarchy
        downward = {
            'asset_numbers': model.asset_number,
            'locations': model.location,
            'positions': model.position
        }

        return {
            'model': model,
            'upward': upward,
            'downward': downward
        }

class AssetNumber(Base):
    __tablename__ = 'asset_number'

    id = Column(Integer, primary_key=True)
    number = Column(String, nullable=False)
    description = Column(String)
    model_id = Column(Integer, ForeignKey('model.id'))

    model = relationship("Model", back_populates="asset_number")
    position = relationship("Position", back_populates="asset_number")

    @classmethod
    @with_request_id
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
    @with_request_id
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
    @with_request_id
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
    @with_request_id
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
    @with_request_id
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
    @with_request_id
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

    @classmethod
    @with_request_id
    def add_asset_number(cls, session, number, model_id, description=None, request_id=None):
        """
        Add a new asset number to the database.

        Args:
            session: SQLAlchemy database session
            number (str): Asset number
            model_id (int): ID of the model this asset number belongs to
            description (str, optional): Description of the asset number
            request_id (str, optional): Unique identifier for the request

        Returns:
            AssetNumber: The newly created asset number object
        """
        new_asset_number = cls(
            number=number,
            model_id=model_id,
            description=description
        )

        session.add(new_asset_number)
        session.commit()

        logger.info(f"Created new asset number: {number} for model ID {model_id}")
        return new_asset_number

    @classmethod
    @with_request_id
    def delete_asset_number(cls, session, asset_number_id, request_id=None):
        """
        Delete an asset number from the database.

        Args:
            session: SQLAlchemy database session
            asset_number_id (int): ID of the asset number to delete
            request_id (str, optional): Unique identifier for the request

        Returns:
            bool: True if deletion was successful, False if asset number not found
        """
        asset_number = session.query(cls).filter(cls.id == asset_number_id).first()

        if asset_number:
            session.delete(asset_number)
            session.commit()
            logger.info(f"Deleted asset number ID {asset_number_id}")
            return True
        else:
            logger.warning(f"Failed to delete asset number ID {asset_number_id} - not found")
            return False

    @classmethod
    @with_request_id
    def find_related_entities(cls, session, identifier, is_id=True, request_id=None):
        """
        Find all related entities for an asset number, traversing both up and down
        the hierarchy: Area → EquipmentGroup → Model → AssetNumber → Position.

        Args:
            session: SQLAlchemy database session
            identifier: Either asset_number ID (int) or number (str)
            is_id (bool): If True, identifier is an ID, otherwise it's a number
            request_id (str, optional): Unique identifier for the request

        Returns:
            dict: Dictionary containing:
                - 'asset_number': The found asset number object
                - 'upward': Dictionary containing 'model', 'equipment_group', and 'area'
                - 'downward': Dictionary containing:
                    - 'positions': List of all positions related to this asset number
        """
        # Find the asset number
        if is_id:
            asset_number = session.query(cls).filter(cls.id == identifier).first()
        else:
            asset_number = session.query(cls).filter(cls.number == identifier).first()

        if not asset_number:
            logger.warning(f"Asset number not found for identifier: {identifier}")
            return None

        # Going upward in the hierarchy
        upward = {
            'model': asset_number.model,
            'equipment_group': asset_number.model.equipment_group if asset_number.model else None,
            'area': asset_number.model.equipment_group.area if asset_number.model and asset_number.model.equipment_group else None
        }

        # Going downward in the hierarchy
        downward = {
            'positions': asset_number.position
        }

        logger.info(f"Found related entities for asset number ID {asset_number.id}")
        return {
            'asset_number': asset_number,
            'upward': upward,
            'downward': downward
        }
    
class Location(Base):
    __tablename__ = 'location'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)    
    model_id = Column(Integer, ForeignKey('model.id'))
    description = Column(String, nullable=True)
    
    model = relationship("Model", back_populates="location")
    position = relationship("Position", back_populates="location")
    subassembly = relationship("Subassembly", back_populates="location")

    @classmethod
    @with_request_id
    def add_location(cls, session, name, model_id, description=None, request_id=None):
        """
        Add a new location to the database.

        Args:
            session: SQLAlchemy database session
            name (str): Name of the location
            model_id (int): ID of the model this location belongs to
            description (str, optional): Description of the location
            request_id (str, optional): Unique identifier for the request

        Returns:
            Location: The newly created location object
        """
        new_location = cls(
            name=name,
            model_id=model_id,
            description=description
        )

        session.add(new_location)
        session.commit()

        logger.info(f"Created new location: {name} for model ID {model_id}")
        return new_location

    @classmethod
    @with_request_id
    def delete_location(cls, session, location_id, request_id=None):
        """
        Delete a location from the database.

        Args:
            session: SQLAlchemy database session
            location_id (int): ID of the location to delete
            request_id (str, optional): Unique identifier for the request

        Returns:
            bool: True if deletion was successful, False if location not found
        """
        location = session.query(cls).filter(cls.id == location_id).first()

        if location:
            session.delete(location)
            session.commit()
            logger.info(f"Deleted location ID {location_id}")
            return True
        else:
            logger.warning(f"Failed to delete location ID {location_id} - not found")
            return False

    @classmethod
    @with_request_id
    def find_related_entities(cls, session, identifier, is_id=True, request_id=None):
        """
        Find all related entities for a location, traversing both up and down
        the hierarchy: Area → EquipmentGroup → Model → Location → (Position, Subassembly).

        Args:
            session: SQLAlchemy database session
            identifier: Either location ID (int) or name (str)
            is_id (bool): If True, identifier is an ID, otherwise it's a name
            request_id (str, optional): Unique identifier for the request

        Returns:
            dict: Dictionary containing:
                - 'location': The found location object
                - 'upward': Dictionary containing 'model', 'equipment_group', and 'area'
                - 'downward': Dictionary containing:
                    - 'positions': List of all positions related to this location
                    - 'subassemblies': List of all subassemblies related to this location
        """
        # Find the location
        if is_id:
            location = session.query(cls).filter(cls.id == identifier).first()
        else:
            location = session.query(cls).filter(cls.name == identifier).first()

        if not location:
            logger.warning(f"Location not found for identifier: {identifier}")
            return None

        # Going upward in the hierarchy
        upward = {
            'model': location.model,
            'equipment_group': location.model.equipment_group if location.model else None,
            'area': location.model.equipment_group.area if location.model and location.model.equipment_group else None
        }

        # Going downward in the hierarchy
        downward = {
            'positions': location.position,
            'subassemblies': location.subassembly
        }

        logger.info(f"Found related entities for location ID {location.id}")
        return {
            'location': location,
            'upward': upward,
            'downward': downward
        }

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
    class_flag = Column(String)  # MP2=Class Flag SPC= Category
    ud6 = Column(String)  # MP2=UD6
    type = Column(String)  # MP2=TYPE
    notes = Column(String)  # MP2=Notes, SPC= Long Description
    documentation = Column(String)  # MP2=Specifications

    part_position_image = relationship("PartsPositionImageAssociation", back_populates="part")
    part_problem = relationship("PartProblemAssociation", back_populates="part")
    part_task = relationship("PartTaskAssociation", back_populates="part")
    drawing_part = relationship("DrawingPartAssociation", back_populates="part")

    __table_args__ = (UniqueConstraint('part_number', name='_part_number_uc'),)

    @classmethod
    @with_request_id
    def search(cls,
               search_text: Optional[str] = None,
               fields: Optional[List[str]] = None,
               exact_match: bool = False,
               part_id: Optional[int] = None,
               part_number: Optional[str] = None,
               name: Optional[str] = None,
               oem_mfg: Optional[str] = None,
               model: Optional[str] = None,
               class_flag: Optional[str] = None,
               ud6: Optional[str] = None,
               type_: Optional[str] = None,
               notes: Optional[str] = None,
               documentation: Optional[str] = None,
               limit: int = 100,
               request_id: Optional[str] = None,
               session: Optional[Session] = None) -> List['Part']:
        """
        Comprehensive search method for Part objects with flexible search options.

        Args:
            search_text: Text to search for across specified fields
            fields: List of field names to search in. If None, searches in default fields
                   (part_number, name, oem_mfg, model)
            exact_match: If True, performs exact matching instead of partial matching
            part_id: Optional ID to filter by
            part_number: Optional part_number to filter by
            name: Optional name to filter by
            oem_mfg: Optional oem_mfg to filter by
            model: Optional model to filter by
            class_flag: Optional class_flag to filter by
            ud6: Optional ud6 to filter by
            type_: Optional type to filter by (using type_ to avoid Python keyword conflict)
            notes: Optional notes to filter by
            documentation: Optional documentation to filter by
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Part objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for Part.search", rid)

        # Log the search operation with request ID
        search_params = {
            'search_text': search_text,
            'fields': fields,
            'exact_match': exact_match,
            'part_id': part_id,
            'part_number': part_number,
            'name': name,
            'oem_mfg': oem_mfg,
            'model': model,
            'class_flag': class_flag,
            'ud6': ud6,
            'type_': type_,
            'notes': notes,
            'documentation': documentation,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting Part.search with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("Part.search", rid):
                # Start with the base query
                query = session.query(cls)
                filters = []

                # Process search_text across multiple fields if provided
                if search_text:
                    search_text = search_text.strip()
                    if search_text:
                        # Default fields to search in if none specified
                        if fields is None or len(fields) == 0:
                            fields = ['part_number', 'name', 'oem_mfg', 'model']

                        debug_id(f"Searching for text '{search_text}' in fields: {fields}", rid)

                        # Create field-specific filters
                        text_filters = []
                        for field_name in fields:
                            if hasattr(cls, field_name):
                                field = getattr(cls, field_name)
                                if exact_match:
                                    text_filters.append(field == search_text)
                                else:
                                    text_filters.append(field.ilike(f"%{search_text}%"))

                        # Add the combined text search filter if we have any
                        if text_filters:
                            filters.append(or_(*text_filters))

                # Add filters for specific fields if provided
                if part_id is not None:
                    debug_id(f"Adding filter for part_id: {part_id}", rid)
                    filters.append(cls.id == part_id)

                if part_number is not None:
                    debug_id(f"Adding filter for part_number: {part_number}", rid)
                    if exact_match:
                        filters.append(cls.part_number == part_number)
                    else:
                        filters.append(cls.part_number.ilike(f"%{part_number}%"))

                if name is not None:
                    debug_id(f"Adding filter for name: {name}", rid)
                    if exact_match:
                        filters.append(cls.name == name)
                    else:
                        filters.append(cls.name.ilike(f"%{name}%"))

                if oem_mfg is not None:
                    debug_id(f"Adding filter for oem_mfg: {oem_mfg}", rid)
                    if exact_match:
                        filters.append(cls.oem_mfg == oem_mfg)
                    else:
                        filters.append(cls.oem_mfg.ilike(f"%{oem_mfg}%"))

                if model is not None:
                    debug_id(f"Adding filter for model: {model}", rid)
                    if exact_match:
                        filters.append(cls.model == model)
                    else:
                        filters.append(cls.model.ilike(f"%{model}%"))

                if class_flag is not None:
                    debug_id(f"Adding filter for class_flag: {class_flag}", rid)
                    if exact_match:
                        filters.append(cls.class_flag == class_flag)
                    else:
                        filters.append(cls.class_flag.ilike(f"%{class_flag}%"))

                if ud6 is not None:
                    debug_id(f"Adding filter for ud6: {ud6}", rid)
                    if exact_match:
                        filters.append(cls.ud6 == ud6)
                    else:
                        filters.append(cls.ud6.ilike(f"%{ud6}%"))

                if type_ is not None:
                    debug_id(f"Adding filter for type: {type_}", rid)
                    if exact_match:
                        filters.append(cls.type == type_)
                    else:
                        filters.append(cls.type.ilike(f"%{type_}%"))

                if notes is not None:
                    debug_id(f"Adding filter for notes: {notes}", rid)
                    if exact_match:
                        filters.append(cls.notes == notes)
                    else:
                        filters.append(cls.notes.ilike(f"%{notes}%"))

                if documentation is not None:
                    debug_id(f"Adding filter for documentation: {documentation}", rid)
                    if exact_match:
                        filters.append(cls.documentation == documentation)
                    else:
                        filters.append(cls.documentation.ilike(f"%{documentation}%"))

                # Apply all filters with AND logic if we have any
                if filters:
                    query = query.filter(and_(*filters))

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"Part.search completed, found {len(results)} results", rid)
                return results

        except Exception as e:
            error_id(f"Error in Part.search: {str(e)}", rid)
            # Re-raise the exception after logging it
            raise
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for Part.search", rid)

    @classmethod
    @with_request_id
    def get_by_id(cls, part_id: int, request_id: Optional[str] = None, session: Optional[Session] = None) -> Optional[
        'Part']:
        """
        Get a part by its ID.

        Args:
            part_id: ID of the part to retrieve
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            Part object if found, None otherwise
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for Part.get_by_id", rid)

        debug_id(f"Getting part with ID: {part_id}", rid)
        try:
            part = session.query(cls).filter(cls.id == part_id).first()
            if part:
                debug_id(f"Found part: {part.part_number} (ID: {part_id})", rid)
            else:
                debug_id(f"No part found with ID: {part_id}", rid)
            return part
        except Exception as e:
            error_id(f"Error retrieving part with ID {part_id}: {str(e)}", rid)
            return None
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for Part.get_by_id", rid)

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
    image_position_association = relationship("ImagePositionAssociation", back_populates="image")
    tool_image_association = relationship("ToolImageAssociation", back_populates="image", cascade="all, delete-orphan")

    @classmethod
    @with_request_id
    def add_to_db(cls, session, title, file_path, description="", position_id=None, complete_document_id=None,
                  clean_title=True):
        """
        Add an image to the database, handling file copying and deduplication.

        Args:
            session: SQLAlchemy session
            title: Image title
            file_path: Path to image file
            description: Optional image description
            position_id: Optional position ID to associate with the image
            complete_document_id: Optional completed document ID to associate with the image
            clean_title: Whether to clean image extensions from the title (default: True)

        Returns:
            Image object
        """
        # Import locally to avoid circular dependencies
        import os
        import shutil
        import re
        from sqlalchemy import and_
        from PIL import Image as PILImage
        from plugins.image_modules.image_models import CLIPModelHandler, get_image_model_handler

        # Common image extensions to remove from title
        COMMON_EXTENSIONS = [
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp',
            '.svg', '.ico', '.heic', '.raw', '.cr2', '.nef', '.arw'
        ]

        # Clean the title if requested
        original_title = title
        if clean_title:
            # Remove any image extensions embedded in the title
            pattern = '|'.join([re.escape(ext) for ext in COMMON_EXTENSIONS])
            regex = re.compile(f"({pattern})", re.IGNORECASE)

            # Iteratively remove extensions until none are found
            while True:
                new_title = regex.sub('', title)
                if new_title == title:
                    break
                title = new_title

            # Clean up any dots that might be left at the end
            title = title.rstrip('.')

            # If we've removed everything, use a default name
            if not title:
                title = "image"

            # Log if title was changed
            if title != original_title:
                logger.info(f"Cleaned title from '{original_title}' to '{title}'")

        logger.info(f"Processing image: {title}")

        # Clean quotes from file path first
        src = file_path
        if src.startswith('"') and src.endswith('"'):
            src = src[1:-1]
        elif src.startswith("'") and src.endswith("'"):
            src = src[1:-1]

        # Check if image with same title and description already exists
        existing_image = session.query(cls).filter(
            and_(cls.title == title, cls.description == description)
        ).first()

        # If it exists and has the same file path, return it
        if existing_image is not None and existing_image.file_path == src:
            logger.info(f"Image already exists: {title}")
            new_image = existing_image
        else:
            # Determine file extension and create unique filename from cleaned path
            _, ext = os.path.splitext(src)
            dest_name = f"{title}{ext}"
            dest_rel = os.path.join("DB_IMAGES", dest_name)
            dest_abs = os.path.join(DATABASE_DIR, "DB_IMAGES", dest_name)

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(dest_abs), exist_ok=True)

            # Copy file using cleaned path
            shutil.copy2(src, dest_abs)
            logger.debug(f"Copied '{src}' -> '{dest_abs}'")

            # Create image record
            new_image = cls(
                title=title,
                description=description,
                file_path=dest_rel
            )
            session.add(new_image)
            session.flush()  # Get ID without committing transaction

        try:
            # Create a model handler directly to avoid circular imports
            model_handler = CLIPModelHandler()
            logger.info(f"Using model handler: {model_handler.__class__.__name__}")

            # Convert the file path to an absolute path for processing
            if os.path.isabs(new_image.file_path):
                absolute_file_path = new_image.file_path
            else:
                absolute_file_path = os.path.join(DATABASE_DIR, new_image.file_path)

            logger.info(f"Opening image: {absolute_file_path}")

            # Open the image using PIL
            image = PILImage.open(absolute_file_path).convert("RGB")

            # Validate the image
            logger.info("Calling model_handler.is_valid_image()")
            if not model_handler.is_valid_image(image):
                logger.info(
                    f"Skipping {absolute_file_path}: Image does not meet the required dimensions or aspect ratio.")
            else:
                # Generate embedding if image is valid
                logger.info("Image passed validation.")
                model_embedding = model_handler.get_image_embedding(image)
                model_name = model_handler.__class__.__name__

                # Add embedding if it doesn't exist already
                if model_name and model_embedding is not None:
                    logger.debug("Checking if the image embedding already exists")
                    existing_embedding = session.query(ImageEmbedding).filter(
                        and_(ImageEmbedding.image_id == new_image.id, ImageEmbedding.model_name == model_name)
                    ).first()

                    if existing_embedding is None:
                        logger.info("Creating a new ImageEmbedding entry")
                        image_embedding = ImageEmbedding(
                            image_id=new_image.id,
                            model_name=model_name,
                            model_embedding=model_embedding.tobytes()
                        )
                        session.add(image_embedding)
                        logger.info(f"Created ImageEmbedding with image ID {new_image.id}, model name {model_name}")

                # Handle position association if applicable
                if position_id:
                    logger.debug("Checking if ImagePositionAssociation already exists")
                    existing_association = session.query(ImagePositionAssociation).filter(
                        and_(ImagePositionAssociation.image_id == new_image.id,
                             ImagePositionAssociation.position_id == position_id)
                    ).first()

                    if existing_association is None:
                        logger.info("Creating a new ImagePositionAssociation entry")
                        image_position_association = ImagePositionAssociation(
                            image_id=new_image.id,
                            position_id=position_id
                        )
                        session.add(image_position_association)
                        logger.info(
                            f"Created ImagePositionAssociation with image ID {new_image.id} and position ID {position_id}")

                # Handle completed document association if applicable
                if complete_document_id:
                    logger.debug("Checking if ImageCompletedDocumentAssociation already exists")
                    existing_association = session.query(ImageCompletedDocumentAssociation).filter(
                        and_(ImageCompletedDocumentAssociation.image_id == new_image.id,
                             ImageCompletedDocumentAssociation.complete_document_id == complete_document_id)
                    ).first()

                    if existing_association is None:
                        logger.info("Creating a new ImageCompletedDocumentAssociation entry")
                        image_completed_document_association = ImageCompletedDocumentAssociation(
                            image_id=new_image.id,
                            complete_document_id=complete_document_id
                        )
                        session.add(image_completed_document_association)
                        logger.info(
                            f"Created ImageCompletedDocumentAssociation with image ID {new_image.id} and complete document ID {complete_document_id}")

        except Exception as e:
            # Log the error
            logger.error(f"An error occurred while processing the image: {e}", exc_info=True)
            error_file_path = os.path.join(DATABASE_DIR, "DB_IMAGES", 'failed_uploads.txt')
            os.makedirs(os.path.dirname(error_file_path), exist_ok=True)
            with open(error_file_path, 'a') as error_file:
                error_file.write(f"Error processing image with title '{title}': {e}\n")

        # Return the image object
        return new_image

    @classmethod
    @with_request_id
    def create_with_tool_association(cls, session, title, file_path, tool, description=""):
        """
        Create an image and associate it with a tool

        Args:
            session: SQLAlchemy session
            title: Image title
            file_path: Path to the image file
            tool: Tool instance to associate with the image
            description: Image description (default: "")

        Returns:
            Tuple of (Image instance, ToolImageAssociation instance)
        """
        # Create the image
        new_image = cls.add_to_db(session, title, file_path, description)

        # Create the association
        from modules.emtacdb.emtacdb_fts import ToolImageAssociation

        # Check for existing association
        existing_assoc = session.query(ToolImageAssociation).filter(
            and_(ToolImageAssociation.tool_id == tool.id,
                 ToolImageAssociation.image_id == new_image.id)
        ).first()

        if existing_assoc:
            logger.info(f"Tool-image association already exists for tool ID {tool.id} and image ID {new_image.id}")
            return new_image, existing_assoc

        # Create new association
        tool_image_assoc = ToolImageAssociation(
            tool_id=tool.id,
            image_id=new_image.id,
            description="Primary uploaded tool image"
        )
        session.add(tool_image_assoc)

        return new_image, tool_image_assoc

    def generate_embedding(self, session, model_handler):
        """
        Generate and store embedding for this image

        Args:
            session: SQLAlchemy session
            model_handler: The image model handler to use

        Returns:
            True if embedding was generated successfully, False otherwise
        """
        try:
            from modules.emtacdb.emtacdb_fts import ImageEmbedding

            # Convert the relative file path back to an absolute path if needed
            if os.path.isabs(self.file_path):
                absolute_file_path = self.file_path
            else:
                absolute_file_path = os.path.join(BASE_DIR, self.file_path)

            logger.info(f"Opening image: {absolute_file_path}")

            # Open the image using the absolute file path
            image = PILImage.open(absolute_file_path).convert("RGB")

            logger.info("Checking if image is valid for the model")
            if not model_handler.is_valid_image(image):
                logger.info(f"Image does not meet the required dimensions or aspect ratio.")
                return False

            logger.info("Generating image embedding")
            model_embedding = model_handler.get_image_embedding(image)
            model_name = model_handler.__class__.__name__

            if model_embedding is None:
                logger.error("Failed to generate embedding")
                return False

            # Check if the embedding already exists
            existing_embedding = session.query(ImageEmbedding).filter(
                and_(ImageEmbedding.image_id == self.id, ImageEmbedding.model_name == model_name)
            ).first()

            if existing_embedding is None:
                # Create new embedding
                logger.info(f"Creating a new ImageEmbedding entry for image ID {self.id}")
                image_embedding = ImageEmbedding(
                    image_id=self.id,
                    model_name=model_name,
                    model_embedding=model_embedding.tobytes()
                )
                session.add(image_embedding)

            return True

        except Exception as e:
            logger.error(f"Error generating embedding: {e}", exc_info=True)

            # Log the error to failed_uploads.txt
            error_file_path = os.path.join(DATABASE_DIR, 'images', 'failed_uploads.txt')
            os.makedirs(os.path.join(DATABASE_DIR, 'images'), exist_ok=True)
            with open(error_file_path, 'a') as error_file:
                error_file.write(f"Error generating embedding for image ID {self.id}: {e}\n")

            return False

    @classmethod
    @with_request_id
    def serve_image(cls,
                    image_id: Optional[int] = None,
                    title: Optional[str] = None,
                    file_path: Optional[str] = None,
                    request_id: Optional[str] = None,
                    session: Optional[Session] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve an image from the database and construct file information for serving.

        Args:
            image_id: Optional ID of the image to retrieve
            title: Optional title to search for
            file_path: Optional file path to search for
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            Dictionary containing image data if found, None otherwise. Dictionary includes:
            - id: Image ID
            - title: Image title
            - description: Image description
            - file_path: Relative path to the file
            - absolute_path: Absolute path to the file on disk
            - exists: Boolean indicating if the file exists on disk
            - content_type: MIME type of the image
            - modified_time: Last modified time of the file
            - size: File size in bytes
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for Image.serve_image", rid)

        # Log the operation with request ID
        debug_id(
            f"Starting Image.serve_image with parameters: image_id={image_id}, title={title}, file_path={file_path}",
            rid)

        try:
            import os
            import mimetypes
            from datetime import datetime

            # Build a query to find the image
            query = session.query(cls)
            if image_id is not None:
                query = query.filter(cls.id == image_id)
            if title is not None:
                query = query.filter(cls.title == title)
            if file_path is not None:
                query = query.filter(cls.file_path == file_path)

            # Get the first matching image
            image = query.first()
            if not image:
                debug_id(f"No image found with parameters: image_id={image_id}, title={title}, file_path={file_path}",
                         rid)
                return None

            # Construct the absolute file path
            relative_path = image.file_path

            # Try to resolve the absolute path using different possibilities
            file_exists = False
            absolute_path = None

            # Potential paths to check - in order of most likely to least likely
            potential_paths = [
                # Path is directly stored in DB as absolute path
                relative_path if os.path.isabs(relative_path) else None,

                # Path using DATABASE_DIR as base
                os.path.join(DATABASE_DIR, relative_path),

                # Path explicitly using DATABASE_PATH_IMAGES_FOLDER
                os.path.join(DATABASE_PATH_IMAGES_FOLDER, os.path.basename(relative_path)),

                # Try just the filename in DATABASE_PATH_IMAGES_FOLDER
                os.path.join(DATABASE_PATH_IMAGES_FOLDER, os.path.basename(relative_path))
                if os.path.basename(relative_path) != relative_path else None,

                # Try using BASE_DIR instead of DATABASE_DIR
                os.path.join(BASE_DIR, relative_path)
            ]

            # Filter out None values
            potential_paths = [p for p in potential_paths if p is not None]

            # Check each path
            for path in potential_paths:
                if os.path.isfile(path):
                    absolute_path = path
                    file_exists = True
                    debug_id(f"Found image at path: {absolute_path}", rid)
                    break

            # If file wasn't found, use the first potential path for error reporting
            if not file_exists and potential_paths:
                absolute_path = potential_paths[0]
                error_id(f"Image file not found at any of the potential paths. "
                         f"First path checked: {absolute_path}", rid)
            elif not file_exists:
                error_id(f"Image file not found and no potential paths to check", rid)
                absolute_path = relative_path  # Default fallback

            # Get file stats if the file exists
            file_size = 0
            modified_time = None
            if file_exists:
                file_stats = os.stat(absolute_path)
                file_size = file_stats.st_size
                modified_time = datetime.fromtimestamp(file_stats.st_mtime)

            # Determine content type
            content_type = mimetypes.guess_type(absolute_path)[0] or 'application/octet-stream'

            # Build the response data
            image_data = {
                'id': image.id,
                'title': image.title,
                'description': image.description,
                'file_path': relative_path,
                'absolute_path': absolute_path,
                'exists': file_exists,
                'content_type': content_type,
                'modified_time': modified_time,
                'size': file_size
            }

            debug_id(f"Image.serve_image completed successfully for image ID: {image.id}", rid)
            return image_data

        except Exception as e:
            error_id(f"Error in Image.serve_image: {str(e)}", rid, exc_info=True)
            return None
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for Image.serve_image", rid)

    @classmethod
    @with_request_id
    def search(cls,
               search_text: Optional[str] = None,
               fields: Optional[List[str]] = None,
               image_id: Optional[int] = None,
               title: Optional[str] = None,
               description: Optional[str] = None,
               file_path: Optional[str] = None,
               position_id: Optional[int] = None,
               tool_id: Optional[int] = None,
               complete_document_id: Optional[int] = None,
               exact_match: bool = False,
               limit: int = 100,
               offset: int = 0,
               sort_by: str = 'id',
               sort_order: str = 'asc',
               request_id: Optional[str] = None,
               session: Optional[Session] = None) -> List['Image']:
        """
        Comprehensive search method for Image objects with flexible search options.

        Args:
            search_text: Text to search for across specified fields
            fields: List of field names to search in. If None, searches in title and description
            image_id: Optional image ID to filter by
            title: Optional title to filter by
            description: Optional description to filter by
            file_path: Optional file path to filter by
            position_id: Optional position ID to filter by (will search through associations)
            tool_id: Optional tool ID to filter by (will search through associations)
            complete_document_id: Optional complete document ID to filter by (will search through associations)
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            offset: Number of results to skip (for pagination)
            sort_by: Field to sort by (default 'id')
            sort_order: Sort direction: 'asc' or 'desc' (default 'asc')
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Image objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for Image.search", rid)

        # Log the search operation with request ID
        search_params = {
            'search_text': search_text,
            'fields': fields,
            'image_id': image_id,
            'title': title,
            'description': description,
            'file_path': file_path,
            'position_id': position_id,
            'tool_id': tool_id,
            'complete_document_id': complete_document_id,
            'exact_match': exact_match,
            'limit': limit,
            'offset': offset,
            'sort_by': sort_by,
            'sort_order': sort_order
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting Image.search with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("Image.search", rid):
                from sqlalchemy import or_, and_, desc, asc

                # Start with a base query
                query = session.query(cls)
                joins_needed = set()
                filters = []

                # Process search_text across multiple fields if provided
                if search_text and search_text.strip():
                    search_text = search_text.strip()

                    # Default fields to search in if none specified
                    if fields is None or len(fields) == 0:
                        fields = ['title', 'description']

                    debug_id(f"Searching for text '{search_text}' in fields: {fields}", rid)

                    # Create field-specific filters
                    text_filters = []
                    for field_name in fields:
                        if hasattr(cls, field_name):
                            field = getattr(cls, field_name)
                            if exact_match:
                                text_filters.append(field == search_text)
                            else:
                                text_filters.append(field.ilike(f"%{search_text}%"))

                    # Add the combined text search filter if we have any
                    if text_filters:
                        filters.append(or_(*text_filters))

                # Add filters for specific fields if provided
                if image_id is not None:
                    filters.append(cls.id == image_id)

                if title is not None:
                    if exact_match:
                        filters.append(cls.title == title)
                    else:
                        filters.append(cls.title.ilike(f"%{title}%"))

                if description is not None:
                    if exact_match:
                        filters.append(cls.description == description)
                    else:
                        filters.append(cls.description.ilike(f"%{description}%"))

                if file_path is not None:
                    if exact_match:
                        filters.append(cls.file_path == file_path)
                    else:
                        filters.append(cls.file_path.ilike(f"%{file_path}%"))

                # Apply association-based filters
                if position_id is not None:
                    joins_needed.add('position')

                if tool_id is not None:
                    joins_needed.add('tool')

                if complete_document_id is not None:
                    joins_needed.add('complete_document')

                # Perform necessary joins
                if 'position' in joins_needed:
                    from models import ImagePositionAssociation, Position
                    query = query.join(ImagePositionAssociation, cls.id == ImagePositionAssociation.image_id)
                    query = query.join(Position, Position.id == ImagePositionAssociation.position_id)
                    filters.append(Position.id == position_id)

                if 'tool' in joins_needed:
                    from models import ToolImageAssociation, Tool
                    query = query.join(ToolImageAssociation, cls.id == ToolImageAssociation.image_id)
                    query = query.join(Tool, Tool.id == ToolImageAssociation.tool_id)
                    filters.append(Tool.id == tool_id)

                if 'complete_document' in joins_needed:
                    from models import ImageCompletedDocumentAssociation, CompleteDocument
                    query = query.join(ImageCompletedDocumentAssociation,
                                       cls.id == ImageCompletedDocumentAssociation.image_id)
                    query = query.join(CompleteDocument,
                                       CompleteDocument.id == ImageCompletedDocumentAssociation.complete_document_id)
                    filters.append(CompleteDocument.id == complete_document_id)

                # Apply all filters with AND logic if we have any
                if filters:
                    query = query.filter(and_(*filters))

                # Make results distinct to avoid duplicates from joins
                if joins_needed:
                    query = query.distinct()

                # Apply sorting
                sort_column = getattr(cls, sort_by, cls.id)
                if sort_order.lower() == 'desc':
                    query = query.order_by(desc(sort_column))
                else:
                    query = query.order_by(asc(sort_column))

                # Apply pagination
                query = query.offset(offset).limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"Image.search completed, found {len(results)} results", rid)
                return results

        except Exception as e:
            error_id(f"Error in Image.search: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for Image.search", rid)

class ImageEmbedding(Base):
    __tablename__ = 'image_embedding'

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('image.id'))
    model_name = Column(String, nullable=False)
    model_embedding = Column(LargeBinary, nullable=False)

    image = relationship("Image", back_populates="image_embedding")

# TODO: add Drawing type ex: Electrical, Mechanical....
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

    @classmethod
    @with_request_id
    def search(cls,
               search_text: Optional[str] = None,
               fields: Optional[List[str]] = None,
               exact_match: bool = False,
               drawing_id: Optional[int] = None,
               drw_equipment_name: Optional[str] = None,
               drw_number: Optional[str] = None,
               drw_name: Optional[str] = None,
               drw_revision: Optional[str] = None,
               drw_spare_part_number: Optional[str] = None,
               file_path: Optional[str] = None,
               limit: int = 100,
               request_id: Optional[str] = None,
               session: Optional[Session] = None) -> List['Drawing']:
        """
        Comprehensive search method for Drawing objects with flexible search options.

        Args:
            search_text: Text to search for across specified fields
            fields: List of field names to search in. If None, searches in default fields
                   (drw_number, drw_name, drw_equipment_name, drw_spare_part_number)
            exact_match: If True, performs exact matching instead of partial matching
            drawing_id: Optional ID to filter by
            drw_equipment_name: Optional equipment name to filter by
            drw_number: Optional drawing number to filter by
            drw_name: Optional drawing name to filter by
            drw_revision: Optional revision to filter by
            drw_spare_part_number: Optional spare part number to filter by
            file_path: Optional file path to filter by
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Drawing objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for Drawing.search", rid)

        # Log the search operation with request ID
        search_params = {
            'search_text': search_text,
            'fields': fields,
            'exact_match': exact_match,
            'drawing_id': drawing_id,
            'drw_equipment_name': drw_equipment_name,
            'drw_number': drw_number,
            'drw_name': drw_name,
            'drw_revision': drw_revision,
            'drw_spare_part_number': drw_spare_part_number,
            'file_path': file_path,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting Drawing.search with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("Drawing.search", rid):
                # Start with the base query
                query = session.query(cls)
                filters = []

                # Process search_text across multiple fields if provided
                if search_text:
                    search_text = search_text.strip()
                    if search_text:
                        # Default fields to search in if none specified
                        if fields is None or len(fields) == 0:
                            fields = ['drw_number', 'drw_name', 'drw_equipment_name', 'drw_spare_part_number']

                        debug_id(f"Searching for text '{search_text}' in fields: {fields}", rid)

                        # Create field-specific filters
                        text_filters = []
                        for field_name in fields:
                            if hasattr(cls, field_name):
                                field = getattr(cls, field_name)
                                if exact_match:
                                    text_filters.append(field == search_text)
                                else:
                                    text_filters.append(field.ilike(f"%{search_text}%"))

                        # Add the combined text search filter if we have any
                        if text_filters:
                            filters.append(or_(*text_filters))

                # Add filters for specific fields if provided
                if drawing_id is not None:
                    debug_id(f"Adding filter for drawing_id: {drawing_id}", rid)
                    filters.append(cls.id == drawing_id)

                if drw_equipment_name is not None:
                    debug_id(f"Adding filter for drw_equipment_name: {drw_equipment_name}", rid)
                    if exact_match:
                        filters.append(cls.drw_equipment_name == drw_equipment_name)
                    else:
                        filters.append(cls.drw_equipment_name.ilike(f"%{drw_equipment_name}%"))

                if drw_number is not None:
                    debug_id(f"Adding filter for drw_number: {drw_number}", rid)
                    if exact_match:
                        filters.append(cls.drw_number == drw_number)
                    else:
                        filters.append(cls.drw_number.ilike(f"%{drw_number}%"))

                if drw_name is not None:
                    debug_id(f"Adding filter for drw_name: {drw_name}", rid)
                    if exact_match:
                        filters.append(cls.drw_name == drw_name)
                    else:
                        filters.append(cls.drw_name.ilike(f"%{drw_name}%"))

                if drw_revision is not None:
                    debug_id(f"Adding filter for drw_revision: {drw_revision}", rid)
                    if exact_match:
                        filters.append(cls.drw_revision == drw_revision)
                    else:
                        filters.append(cls.drw_revision.ilike(f"%{drw_revision}%"))

                if drw_spare_part_number is not None:
                    debug_id(f"Adding filter for drw_spare_part_number: {drw_spare_part_number}", rid)
                    if exact_match:
                        filters.append(cls.drw_spare_part_number == drw_spare_part_number)
                    else:
                        filters.append(cls.drw_spare_part_number.ilike(f"%{drw_spare_part_number}%"))

                if file_path is not None:
                    debug_id(f"Adding filter for file_path: {file_path}", rid)
                    if exact_match:
                        filters.append(cls.file_path == file_path)
                    else:
                        filters.append(cls.file_path.ilike(f"%{file_path}%"))

                # Apply all filters with AND logic if we have any
                if filters:
                    query = query.filter(and_(*filters))

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"Drawing.search completed, found {len(results)} results", rid)
                return results

        except Exception as e:
            error_id(f"Error in Drawing.search: {str(e)}", rid)
            # Re-raise the exception after logging it
            raise
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for Drawing.search", rid)

    @classmethod
    @with_request_id
    def get_by_id(cls, drawing_id: int, request_id: Optional[str] = None, session: Optional[Session] = None) -> \
    Optional['Drawing']:
        """
        Get a drawing by its ID.

        Args:
            drawing_id: ID of the drawing to retrieve
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            Drawing object if found, None otherwise
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for Drawing.get_by_id", rid)

        debug_id(f"Getting drawing with ID: {drawing_id}", rid)
        try:
            drawing = session.query(cls).filter(cls.id == drawing_id).first()
            if drawing:
                debug_id(f"Found drawing: {drawing.drw_number} (ID: {drawing_id})", rid)
            else:
                debug_id(f"No drawing found with ID: {drawing_id}", rid)
            return drawing
        except Exception as e:
            error_id(f"Error retrieving drawing with ID {drawing_id}: {str(e)}", rid)
            return None
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for Drawing.get_by_id", rid)

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

    @classmethod
    @with_request_id
    def get_parts_by_drawing(cls,
                             drawing_id: Optional[int] = None,
                             drw_equipment_name: Optional[str] = None,
                             drw_number: Optional[str] = None,
                             drw_name: Optional[str] = None,
                             drw_revision: Optional[str] = None,
                             drw_spare_part_number: Optional[str] = None,
                             part_id: Optional[int] = None,
                             part_number: Optional[str] = None,
                             part_name: Optional[str] = None,
                             oem_mfg: Optional[str] = None,
                             model: Optional[str] = None,
                             class_flag: Optional[str] = None,
                             exact_match: bool = False,
                             limit: int = 100,
                             request_id: Optional[str] = None,
                             session: Optional[Session] = None) -> List['Part']:
        """
        Get parts associated with drawings based on flexible search criteria.

        Args:
            drawing_id: Optional drawing ID to filter by
            drw_equipment_name, drw_number, drw_name, drw_revision, drw_spare_part_number:
                Optional drawing attributes to filter by
            part_id: Optional part ID to filter by
            part_number, part_name, oem_mfg, model, class_flag:
                Optional part attributes to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Part objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for Drawing.get_parts_by_drawing", rid)

        # Log the search operation with request ID
        search_params = {
            'drawing_id': drawing_id,
            'drw_equipment_name': drw_equipment_name,
            'drw_number': drw_number,
            'drw_name': drw_name,
            'drw_revision': drw_revision,
            'drw_spare_part_number': drw_spare_part_number,
            'part_id': part_id,
            'part_number': part_number,
            'part_name': part_name,
            'oem_mfg': oem_mfg,
            'model': model,
            'class_flag': class_flag,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting Drawing.get_parts_by_drawing with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("Drawing.get_parts_by_drawing", rid):
                from models import Part, DrawingPartAssociation

                # Start with a query that joins Part and DrawingPartAssociation
                query = session.query(Part).join(DrawingPartAssociation).join(cls)

                # Apply drawing filters
                if drawing_id is not None:
                    query = query.filter(cls.id == drawing_id)

                if drw_equipment_name is not None:
                    if exact_match:
                        query = query.filter(cls.drw_equipment_name == drw_equipment_name)
                    else:
                        query = query.filter(cls.drw_equipment_name.ilike(f"%{drw_equipment_name}%"))

                if drw_number is not None:
                    if exact_match:
                        query = query.filter(cls.drw_number == drw_number)
                    else:
                        query = query.filter(cls.drw_number.ilike(f"%{drw_number}%"))

                if drw_name is not None:
                    if exact_match:
                        query = query.filter(cls.drw_name == drw_name)
                    else:
                        query = query.filter(cls.drw_name.ilike(f"%{drw_name}%"))

                if drw_revision is not None:
                    if exact_match:
                        query = query.filter(cls.drw_revision == drw_revision)
                    else:
                        query = query.filter(cls.drw_revision.ilike(f"%{drw_revision}%"))

                if drw_spare_part_number is not None:
                    if exact_match:
                        query = query.filter(cls.drw_spare_part_number == drw_spare_part_number)
                    else:
                        query = query.filter(cls.drw_spare_part_number.ilike(f"%{drw_spare_part_number}%"))

                # Apply part filters
                if part_id is not None:
                    query = query.filter(Part.id == part_id)

                if part_number is not None:
                    if exact_match:
                        query = query.filter(Part.part_number == part_number)
                    else:
                        query = query.filter(Part.part_number.ilike(f"%{part_number}%"))

                if part_name is not None:
                    if exact_match:
                        query = query.filter(Part.name == part_name)
                    else:
                        query = query.filter(Part.name.ilike(f"%{part_name}%"))

                if oem_mfg is not None:
                    if exact_match:
                        query = query.filter(Part.oem_mfg == oem_mfg)
                    else:
                        query = query.filter(Part.oem_mfg.ilike(f"%{oem_mfg}%"))

                if model is not None:
                    if exact_match:
                        query = query.filter(Part.model == model)
                    else:
                        query = query.filter(Part.model.ilike(f"%{model}%"))

                if class_flag is not None:
                    if exact_match:
                        query = query.filter(Part.class_flag == class_flag)
                    else:
                        query = query.filter(Part.class_flag.ilike(f"%{class_flag}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"Drawing.get_parts_by_drawing completed, found {len(results)} parts", rid)
                return results

        except Exception as e:
            error_id(f"Error in Drawing.get_parts_by_drawing: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for Drawing.get_parts_by_drawing", rid)

    @classmethod
    @with_request_id
    def get_drawings_by_part(cls,
                             part_id: Optional[int] = None,
                             part_number: Optional[str] = None,
                             part_name: Optional[str] = None,
                             oem_mfg: Optional[str] = None,
                             model: Optional[str] = None,
                             class_flag: Optional[str] = None,
                             drawing_id: Optional[int] = None,
                             drw_equipment_name: Optional[str] = None,
                             drw_number: Optional[str] = None,
                             drw_name: Optional[str] = None,
                             drw_revision: Optional[str] = None,
                             drw_spare_part_number: Optional[str] = None,
                             exact_match: bool = False,
                             limit: int = 100,
                             request_id: Optional[str] = None,
                             session: Optional[Session] = None) -> List['Drawing']:
        """
        Get drawings associated with parts based on flexible search criteria.

        Args:
            part_id: Optional part ID to filter by
            part_number, part_name, oem_mfg, model, class_flag:
                Optional part attributes to filter by
            drawing_id: Optional drawing ID to filter by
            drw_equipment_name, drw_number, drw_name, drw_revision, drw_spare_part_number:
                Optional drawing attributes to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Drawing objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for Part.get_drawings_by_part", rid)

        # Log the search operation with request ID
        search_params = {
            'part_id': part_id,
            'part_number': part_number,
            'part_name': part_name,
            'oem_mfg': oem_mfg,
            'model': model,
            'class_flag': class_flag,
            'drawing_id': drawing_id,
            'drw_equipment_name': drw_equipment_name,
            'drw_number': drw_number,
            'drw_name': drw_name,
            'drw_revision': drw_revision,
            'drw_spare_part_number': drw_spare_part_number,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting Part.get_drawings_by_part with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("Part.get_drawings_by_part", rid):
                from models import Drawing, DrawingPartAssociation

                # Start with a query that joins Drawing and DrawingPartAssociation
                query = session.query(Drawing).join(DrawingPartAssociation).join(cls)

                # Apply part filters
                if part_id is not None:
                    query = query.filter(cls.id == part_id)

                if part_number is not None:
                    if exact_match:
                        query = query.filter(cls.part_number == part_number)
                    else:
                        query = query.filter(cls.part_number.ilike(f"%{part_number}%"))

                if part_name is not None:
                    if exact_match:
                        query = query.filter(cls.name == part_name)
                    else:
                        query = query.filter(cls.name.ilike(f"%{part_name}%"))

                if oem_mfg is not None:
                    if exact_match:
                        query = query.filter(cls.oem_mfg == oem_mfg)
                    else:
                        query = query.filter(cls.oem_mfg.ilike(f"%{oem_mfg}%"))

                if model is not None:
                    if exact_match:
                        query = query.filter(cls.model == model)
                    else:
                        query = query.filter(cls.model.ilike(f"%{model}%"))

                if class_flag is not None:
                    if exact_match:
                        query = query.filter(cls.class_flag == class_flag)
                    else:
                        query = query.filter(cls.class_flag.ilike(f"%{class_flag}%"))

                # Apply drawing filters
                if drawing_id is not None:
                    query = query.filter(Drawing.id == drawing_id)

                if drw_equipment_name is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_equipment_name == drw_equipment_name)
                    else:
                        query = query.filter(Drawing.drw_equipment_name.ilike(f"%{drw_equipment_name}%"))

                if drw_number is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_number == drw_number)
                    else:
                        query = query.filter(Drawing.drw_number.ilike(f"%{drw_number}%"))

                if drw_name is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_name == drw_name)
                    else:
                        query = query.filter(Drawing.drw_name.ilike(f"%{drw_name}%"))

                if drw_revision is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_revision == drw_revision)
                    else:
                        query = query.filter(Drawing.drw_revision.ilike(f"%{drw_revision}%"))

                if drw_spare_part_number is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_spare_part_number == drw_spare_part_number)
                    else:
                        query = query.filter(Drawing.drw_spare_part_number.ilike(f"%{drw_spare_part_number}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"Part.get_drawings_by_part completed, found {len(results)} drawings", rid)
                return results

        except Exception as e:
            error_id(f"Error in Part.get_drawings_by_part: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for Part.get_drawings_by_part", rid)
    
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

    @classmethod
    @with_request_id
    def get_positions_by_task_id(cls, session=None, task_id=None, name=None, description=None,
                                 area_id=None, equipment_group_id=None, model_id=None,
                                 asset_number_id=None, location_id=None, subassembly_id=None,
                                 component_assembly_id=None, assembly_view_id=None, site_location_id=None):
        """
        Get all positions associated with a specific task or set of task criteria.

        Args:
            session: SQLAlchemy session (Optional)
            task_id: ID of the task (Optional)
            name: Filter by task name (Optional)
            description: Filter by task description (Optional)
            area_id, equipment_group_id, etc.: Position hierarchy filters (Optional)

        Returns:
            List of Position objects matching the criteria
        """
        if session is None:
            session = DatabaseConfig().get_main_session()

        try:
            from models import TaskPositionAssociation

            # Start with a query to get task(s)
            task_query = session.query(cls)

            # Apply task filters if provided
            if task_id is not None:
                task_query = task_query.filter(cls.id == task_id)
            if name is not None:
                task_query = task_query.filter(cls.name.like(f"%{name}%"))
            if description is not None:
                task_query = task_query.filter(cls.description.like(f"%{description}%"))

            tasks = task_query.all()
            if not tasks:
                return []

            task_ids = [t.id for t in tasks]

            # Start with a query that joins Position and TaskPositionAssociation
            query = session.query(Position).join(TaskPositionAssociation)

            # Filter by task IDs
            query = query.filter(TaskPositionAssociation.task_id.in_(task_ids))

            # Apply position hierarchy filters if provided
            if area_id is not None:
                query = query.filter(Position.area_id == area_id)
            if equipment_group_id is not None:
                query = query.filter(Position.equipment_group_id == equipment_group_id)
            if model_id is not None:
                query = query.filter(Position.model_id == model_id)
            if asset_number_id is not None:
                query = query.filter(Position.asset_number_id == asset_number_id)
            if location_id is not None:
                query = query.filter(Position.location_id == location_id)
            if subassembly_id is not None:
                query = query.filter(Position.subassembly_id == subassembly_id)
            if component_assembly_id is not None:
                query = query.filter(Position.component_assembly_id == component_assembly_id)
            if assembly_view_id is not None:
                query = query.filter(Position.assembly_view_id == assembly_view_id)
            if site_location_id is not None:
                query = query.filter(Position.site_location_id == site_location_id)

            # Make results distinct in case multiple tasks point to same position
            query = query.distinct()

            debug_id(f"Getting positions with filters: task_id={task_id}, name={name}, description={description}")
            positions = query.all()
            info_id(f"Found {len(positions)} positions matching the criteria")

            return positions

        except SQLAlchemyError as e:
            error_id(f"Error getting positions: {str(e)}", exc_info=True)
            return []

    @classmethod
    @with_request_id
    def get_tasks_by_position_id(cls, session=None, position_id=None, name=None, description=None,
                                 area_id=None, equipment_group_id=None, model_id=None,
                                 asset_number_id=None, location_id=None, subassembly_id=None,
                                 component_assembly_id=None, assembly_view_id=None, site_location_id=None):
        """
        Get all tasks associated with a specific position or set of position criteria.

        Args:
            session: SQLAlchemy session (Optional)
            position_id: ID of the position (Optional)
            name: Filter tasks by name (Optional)
            description: Filter tasks by description (Optional)
            area_id, equipment_group_id, etc.: Position hierarchy filters (Optional)

        Returns:
            List of Task objects matching the criteria
        """
        if session is None:
            session = DatabaseConfig().get_main_session()

        try:
            from models import TaskPositionAssociation, Task

            # Start with a query that joins Task and TaskPositionAssociation
            query = session.query(Task).join(TaskPositionAssociation)

            # If position_id is provided, filter by that specific position
            if position_id:
                query = query.filter(TaskPositionAssociation.position_id == position_id)
            else:
                # If no position_id but position hierarchy filters are provided
                position_filters = {}
                if area_id is not None:
                    position_filters['area_id'] = area_id
                if equipment_group_id is not None:
                    position_filters['equipment_group_id'] = equipment_group_id
                if model_id is not None:
                    position_filters['model_id'] = model_id
                if asset_number_id is not None:
                    position_filters['asset_number_id'] = asset_number_id
                if location_id is not None:
                    position_filters['location_id'] = location_id
                if subassembly_id is not None:
                    position_filters['subassembly_id'] = subassembly_id
                if component_assembly_id is not None:
                    position_filters['component_assembly_id'] = component_assembly_id
                if assembly_view_id is not None:
                    position_filters['assembly_view_id'] = assembly_view_id
                if site_location_id is not None:
                    position_filters['site_location_id'] = site_location_id

                if position_filters:
                    # Get position IDs matching the criteria
                    positions = session.query(cls).filter_by(**position_filters).all()
                    position_ids = [p.id for p in positions]

                    if not position_ids:
                        return []  # No positions match the criteria

                    query = query.filter(TaskPositionAssociation.position_id.in_(position_ids))

            # Apply task-specific filters if provided
            if name is not None:
                query = query.filter(Task.name.like(f"%{name}%"))
            if description is not None:
                query = query.filter(Task.description.like(f"%{description}%"))

            # Make results distinct in case same task appears in multiple positions
            query = query.distinct()

            debug_id(f"Getting tasks with filters: position_id={position_id}, name={name}, description={description}")
            tasks = query.all()
            info_id(f"Found {len(tasks)} tasks matching the criteria")

            return tasks

        except SQLAlchemyError as e:
            error_id(f"Error getting tasks: {str(e)}", exc_info=True)
            return []

    @classmethod
    @with_request_id
    def associate_task_position(cls,
                                task_id: int,
                                position_id: int,
                                request_id: Optional[str] = None,
                                session: Optional[Session] = None) -> Optional['TaskPositionAssociation']:
        """
        Associate a task with a position.

        Args:
            task_id: ID of the task to associate
            position_id: ID of the position to associate
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            The created TaskPositionAssociation object if successful, None otherwise
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for TaskPositionAssociation.associate_task_position", rid)

        # Log the operation with request ID
        debug_id(
            f"Starting TaskPositionAssociation.associate_task_position with parameters: task_id={task_id}, position_id={position_id}",
            rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("TaskPositionAssociation.associate_task_position", rid):
                from models import Task, Position

                # Check if task exists
                task = session.query(Task).filter(Task.id == task_id).first()
                if not task:
                    error_id(
                        f"Error in TaskPositionAssociation.associate_task_position: Task with ID {task_id} not found",
                        rid)
                    return None

                # Check if position exists
                position = session.query(Position).filter(Position.id == position_id).first()
                if not position:
                    error_id(
                        f"Error in TaskPositionAssociation.associate_task_position: Position with ID {position_id} not found",
                        rid)
                    return None

                # Check if association already exists
                existing = session.query(cls).filter(
                    cls.task_id == task_id,
                    cls.position_id == position_id
                ).first()

                if existing:
                    debug_id(f"Association between task {task_id} and position {position_id} already exists", rid)
                    return existing

                # Create new association
                association = cls(task_id=task_id, position_id=position_id)
                session.add(association)

                # Commit if we created the session
                if not session_provided:
                    session.commit()
                    debug_id(f"Committed new association between task {task_id} and position {position_id}", rid)

                return association

        except Exception as e:
            error_id(f"Error in TaskPositionAssociation.associate_task_position: {str(e)}", rid, exc_info=True)
            if not session_provided:
                session.rollback()
                debug_id(f"Rolled back transaction in TaskPositionAssociation.associate_task_position", rid)
            return None
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for TaskPositionAssociation.associate_task_position", rid)

    @classmethod
    @with_request_id
    def dissociate_task_position(cls,
                                 task_id: int,
                                 position_id: int,
                                 request_id: Optional[str] = None,
                                 session: Optional[Session] = None) -> bool:
        """
        Remove an association between a task and a position.

        Args:
            task_id: ID of the task to dissociate
            position_id: ID of the position to dissociate
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            True if the association was removed, False otherwise
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for TaskPositionAssociation.dissociate_task_position", rid)

        # Log the operation with request ID
        debug_id(
            f"Starting TaskPositionAssociation.dissociate_task_position with parameters: task_id={task_id}, position_id={position_id}",
            rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("TaskPositionAssociation.dissociate_task_position", rid):
                # Find the association
                association = session.query(cls).filter(
                    cls.task_id == task_id,
                    cls.position_id == position_id
                ).first()

                if not association:
                    debug_id(f"No association found between task {task_id} and position {position_id}", rid)
                    return False

                # Delete the association
                session.delete(association)

                # Commit if we created the session
                if not session_provided:
                    session.commit()
                    debug_id(f"Removed association between task {task_id} and position {position_id}", rid)

                return True

        except Exception as e:
            error_id(f"Error in TaskPositionAssociation.dissociate_task_position: {str(e)}", rid, exc_info=True)
            if not session_provided:
                session.rollback()
                debug_id(f"Rolled back transaction in TaskPositionAssociation.dissociate_task_position", rid)
            return False
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for TaskPositionAssociation.dissociate_task_position", rid)

    @classmethod
    @with_request_id
    def associate_multiple_tasks_to_position(cls,
                                             task_ids: List[int],
                                             position_id: int,
                                             request_id: Optional[str] = None,
                                             session: Optional[Session] = None) -> Dict[int, bool]:
        """
        Associate multiple tasks with a single position.

        Args:
            task_ids: List of task IDs to associate
            position_id: ID of the position to associate with all tasks
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            Dictionary mapping task IDs to success status (True if associated, False if failed)
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for TaskPositionAssociation.associate_multiple_tasks_to_position",
                     rid)

        # Log the operation with request ID
        debug_id(f"Starting TaskPositionAssociation.associate_multiple_tasks_to_position with parameters: "
                 f"task_ids={task_ids}, position_id={position_id}", rid)

        results = {}
        try:
            # Check if position exists
            from models import Position
            position = session.query(Position).filter(Position.id == position_id).first()
            if not position:
                error_id(f"Error in TaskPositionAssociation.associate_multiple_tasks_to_position: "
                         f"Position with ID {position_id} not found", rid)
                return {task_id: False for task_id in task_ids}

            # Process each task
            for task_id in task_ids:
                try:
                    association = cls.associate_task_position(
                        task_id=task_id,
                        position_id=position_id,
                        request_id=rid,
                        session=session
                    )
                    results[task_id] = association is not None
                except Exception as e:
                    error_id(f"Error associating task {task_id} with position {position_id}: {str(e)}", rid)
                    results[task_id] = False

            # Commit if we created the session
            if not session_provided:
                session.commit()
                debug_id(f"Committed all associations in associate_multiple_tasks_to_position", rid)

            # Log summary
            success_count = sum(1 for success in results.values() if success)
            debug_id(
                f"Successfully associated {success_count} out of {len(task_ids)} tasks with position {position_id}",
                rid)

            return results

        except Exception as e:
            error_id(f"Error in TaskPositionAssociation.associate_multiple_tasks_to_position: {str(e)}", rid,
                     exc_info=True)
            if not session_provided:
                session.rollback()
                debug_id(f"Rolled back transaction in TaskPositionAssociation.associate_multiple_tasks_to_position",
                         rid)
            return {task_id: False for task_id in task_ids}
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for TaskPositionAssociation.associate_multiple_tasks_to_position",
                         rid)

    @classmethod
    @with_request_id
    def associate_task_to_multiple_positions(cls,
                                             task_id: int,
                                             position_ids: List[int],
                                             request_id: Optional[str] = None,
                                             session: Optional[Session] = None) -> Dict[int, bool]:
        """
        Associate a single task with multiple positions.

        Args:
            task_id: ID of the task to associate
            position_ids: List of position IDs to associate with the task
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            Dictionary mapping position IDs to success status (True if associated, False if failed)
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for TaskPositionAssociation.associate_task_to_multiple_positions",
                     rid)

        # Log the operation with request ID
        debug_id(f"Starting TaskPositionAssociation.associate_task_to_multiple_positions with parameters: "
                 f"task_id={task_id}, position_ids={position_ids}", rid)

        results = {}
        try:
            # Check if task exists
            from models import Task
            task = session.query(Task).filter(Task.id == task_id).first()
            if not task:
                error_id(f"Error in TaskPositionAssociation.associate_task_to_multiple_positions: "
                         f"Task with ID {task_id} not found", rid)
                return {position_id: False for position_id in position_ids}

            # Process each position
            for position_id in position_ids:
                try:
                    association = cls.associate_task_position(
                        task_id=task_id,
                        position_id=position_id,
                        request_id=rid,
                        session=session
                    )
                    results[position_id] = association is not None
                except Exception as e:
                    error_id(f"Error associating task {task_id} with position {position_id}: {str(e)}", rid)
                    results[position_id] = False

            # Commit if we created the session
            if not session_provided:
                session.commit()
                debug_id(f"Committed all associations in associate_task_to_multiple_positions", rid)

            # Log summary
            success_count = sum(1 for success in results.values() if success)
            debug_id(
                f"Successfully associated task {task_id} with {success_count} out of {len(position_ids)} positions",
                rid)

            return results

        except Exception as e:
            error_id(f"Error in TaskPositionAssociation.associate_task_to_multiple_positions: {str(e)}", rid,
                     exc_info=True)
            if not session_provided:
                session.rollback()
                debug_id(f"Rolled back transaction in TaskPositionAssociation.associate_task_to_multiple_positions",
                         rid)
            return {position_id: False for position_id in position_ids}
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for TaskPositionAssociation.associate_task_to_multiple_positions",
                         rid)

class PartTaskAssociation(Base):
    __tablename__ = 'part_task'

    id = Column(Integer, primary_key=True)
    part_id = Column(Integer, ForeignKey('part.id'))
    task_id = Column(Integer, ForeignKey('task.id'))  # Corrected foreign key

    part = relationship("Part", back_populates="part_task")
    task = relationship("Task", back_populates="part_task")

    @classmethod
    @with_request_id
    def get_tasks_by_part(cls,
                          part_id: Optional[int] = None,
                          part_number: Optional[str] = None,
                          name: Optional[str] = None,
                          oem_mfg: Optional[str] = None,
                          model: Optional[str] = None,
                          task_id: Optional[int] = None,
                          task_name: Optional[str] = None,
                          task_description: Optional[str] = None,
                          exact_match: bool = False,
                          limit: int = 100,
                          request_id: Optional[str] = None,
                          session: Optional[Session] = None) -> List['Task']:
        """
        Get tasks associated with parts based on flexible search criteria.

        Args:
            part_id: Optional part ID to filter by
            part_number, name, oem_mfg, model: Optional part attributes to filter by
            task_id: Optional task ID to filter by
            task_name: Optional task name to filter by
            task_description: Optional task description to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Task objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for Part.get_tasks_by_part", rid)

        # Log the search operation with request ID
        search_params = {
            'part_id': part_id,
            'part_number': part_number,
            'name': name,
            'oem_mfg': oem_mfg,
            'model': model,
            'task_id': task_id,
            'task_name': task_name,
            'task_description': task_description,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting Part.get_tasks_by_part with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("Part.get_tasks_by_part", rid):
                from models import Task, PartTaskAssociation

                # Start with a query that joins Task and PartTaskAssociation
                query = session.query(Task).join(PartTaskAssociation).join(Part)

                # Apply part filters
                if part_id is not None:
                    query = query.filter(Part.id == part_id)

                if part_number is not None:
                    if exact_match:
                        query = query.filter(Part.part_number == part_number)
                    else:
                        query = query.filter(Part.part_number.ilike(f"%{part_number}%"))

                if name is not None:
                    if exact_match:
                        query = query.filter(Part.name == name)
                    else:
                        query = query.filter(Part.name.ilike(f"%{name}%"))

                if oem_mfg is not None:
                    if exact_match:
                        query = query.filter(Part.oem_mfg == oem_mfg)
                    else:
                        query = query.filter(Part.oem_mfg.ilike(f"%{oem_mfg}%"))

                if model is not None:
                    if exact_match:
                        query = query.filter(Part.model == model)
                    else:
                        query = query.filter(Part.model.ilike(f"%{model}%"))

                # Apply task filters
                if task_id is not None:
                    query = query.filter(Task.id == task_id)

                if task_name is not None:
                    if exact_match:
                        query = query.filter(Task.name == task_name)
                    else:
                        query = query.filter(Task.name.ilike(f"%{task_name}%"))

                if task_description is not None:
                    if exact_match:
                        query = query.filter(Task.description == task_description)
                    else:
                        query = query.filter(Task.description.ilike(f"%{task_description}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"Part.get_tasks_by_part completed, found {len(results)} tasks", rid)
                return results

        except Exception as e:
            error_id(f"Error in Part.get_tasks_by_part: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for Part.get_tasks_by_part", rid)

    @classmethod
    @with_request_id
    def get_parts_by_task(cls,
                          task_id: Optional[int] = None,
                          task_name: Optional[str] = None,
                          task_description: Optional[str] = None,
                          part_id: Optional[int] = None,
                          part_number: Optional[str] = None,
                          part_name: Optional[str] = None,
                          oem_mfg: Optional[str] = None,
                          model: Optional[str] = None,
                          class_flag: Optional[str] = None,
                          exact_match: bool = False,
                          limit: int = 100,
                          request_id: Optional[str] = None,
                          session: Optional[Session] = None) -> List['Part']:
        """
        Get parts associated with tasks based on flexible search criteria.

        Args:
            task_id: Optional task ID to filter by
            task_name: Optional task name to filter by
            task_description: Optional task description to filter by
            part_id: Optional part ID to filter by
            part_number, part_name, oem_mfg, model, class_flag: Optional part attributes to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Part objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for Task.get_parts_by_task", rid)

        # Log the search operation with request ID
        search_params = {
            'task_id': task_id,
            'task_name': task_name,
            'task_description': task_description,
            'part_id': part_id,
            'part_number': part_number,
            'part_name': part_name,
            'oem_mfg': oem_mfg,
            'model': model,
            'class_flag': class_flag,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting Task.get_parts_by_task with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("Task.get_parts_by_task", rid):
                from models import Part, PartTaskAssociation

                # Start with a query that joins Part and PartTaskAssociation
                query = session.query(Part).join(PartTaskAssociation).join(Task)

                # Apply task filters
                if task_id is not None:
                    query = query.filter(Task.id == task_id)

                if task_name is not None:
                    if exact_match:
                        query = query.filter(Task.name == task_name)
                    else:
                        query = query.filter(Task.name.ilike(f"%{task_name}%"))

                if task_description is not None:
                    if exact_match:
                        query = query.filter(Task.description == task_description)
                    else:
                        query = query.filter(Task.description.ilike(f"%{task_description}%"))

                # Apply part filters
                if part_id is not None:
                    query = query.filter(Part.id == part_id)

                if part_number is not None:
                    if exact_match:
                        query = query.filter(Part.part_number == part_number)
                    else:
                        query = query.filter(Part.part_number.ilike(f"%{part_number}%"))

                if part_name is not None:
                    if exact_match:
                        query = query.filter(Part.name == part_name)
                    else:
                        query = query.filter(Part.name.ilike(f"%{part_name}%"))

                if oem_mfg is not None:
                    if exact_match:
                        query = query.filter(Part.oem_mfg == oem_mfg)
                    else:
                        query = query.filter(Part.oem_mfg.ilike(f"%{oem_mfg}%"))

                if model is not None:
                    if exact_match:
                        query = query.filter(Part.model == model)
                    else:
                        query = query.filter(Part.model.ilike(f"%{model}%"))

                if class_flag is not None:
                    if exact_match:
                        query = query.filter(Part.class_flag == class_flag)
                    else:
                        query = query.filter(Part.class_flag.ilike(f"%{class_flag}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"Task.get_parts_by_task completed, found {len(results)} parts", rid)
                return results

        except Exception as e:
            error_id(f"Error in Task.get_parts_by_task: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for Task.get_parts_by_task", rid)

    @classmethod
    @with_request_id
    def associate_part_with_task(cls, session=None, part_id=None, task_id=None, request_id=None):
        """
        Create an association between a part and a task if it doesn't already exist.

        Args:
            session: SQLAlchemy session (optional)
            part_id: ID of the part to associate
            task_id: ID of the task to associate
            request_id: Optional request ID for tracking

        Returns:
            The PartTaskAssociation instance (existing or new)
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for PartTaskAssociation.associate_part_with_task", rid)

        debug_id(f"Associating part ID {part_id} with task ID {task_id}", rid)

        try:
            from models import Part, Task
            from sqlalchemy import and_

            # Check if the part and task exist
            part = session.query(Part).get(part_id)
            task = session.query(Task).get(task_id)

            if not part:
                error_id(f"Part with ID {part_id} not found", rid)
                return None

            if not task:
                error_id(f"Task with ID {task_id} not found", rid)
                return None

            # Check if the association already exists
            existing = session.query(cls).filter(
                and_(
                    cls.part_id == part_id,
                    cls.task_id == task_id
                )
            ).first()

            if existing:
                debug_id(f"Association between part ID {part_id} and task ID {task_id} already exists", rid)
                return existing

            # Create new association
            association = cls(
                part_id=part_id,
                task_id=task_id
            )
            session.add(association)
            session.flush()

            debug_id(f"Created new association between part ID {part_id} and task ID {task_id}", rid)
            return association

        except Exception as e:
            error_id(f"Error associating part with task: {str(e)}", rid, exc_info=True)
            return None
        finally:
            # Close the session if we created it
            if not session_provided and session:
                session.close()
                debug_id(f"Closed database session for PartTaskAssociation.associate_part_with_task", rid)

class DrawingTaskAssociation(Base):
    __tablename__ = 'drawing_task'
    id = Column(Integer, primary_key=True)
    drawing_id = Column(Integer, ForeignKey('drawing.id'))
    task_id = Column(Integer, ForeignKey('task.id'))

    drawing = relationship("Drawing", back_populates="drawing_task")
    task = relationship("Task", back_populates="drawing_task")

    @classmethod
    def get_tasks_by_drawing(cls,
                             drawing_id: Optional[int] = None,
                             drw_equipment_name: Optional[str] = None,
                             drw_number: Optional[str] = None,
                             drw_name: Optional[str] = None,
                             drw_revision: Optional[str] = None,
                             drw_spare_part_number: Optional[str] = None,
                             file_path: Optional[str] = None,
                             task_id: Optional[int] = None,
                             task_name: Optional[str] = None,
                             task_description: Optional[str] = None,
                             exact_match: bool = False,
                             limit: int = 100,
                             request_id: Optional[str] = None,
                             session: Optional[Session] = None) -> List['Task']:
        """
        Get tasks associated with drawings based on flexible search criteria.

        Args:
            drawing_id: Optional drawing ID to filter by
            drw_equipment_name, drw_number, drw_name, drw_revision, drw_spare_part_number, file_path:
                Optional drawing attributes to filter by
            task_id: Optional task ID to filter by
            task_name: Optional task name to filter by
            task_description: Optional task description to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Task objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for Drawing.get_tasks_by_drawing", rid)

        # Log the search operation with request ID
        search_params = {
            'drawing_id': drawing_id,
            'drw_equipment_name': drw_equipment_name,
            'drw_number': drw_number,
            'drw_name': drw_name,
            'drw_revision': drw_revision,
            'drw_spare_part_number': drw_spare_part_number,
            'file_path': file_path,
            'task_id': task_id,
            'task_name': task_name,
            'task_description': task_description,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting Drawing.get_tasks_by_drawing with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("Drawing.get_tasks_by_drawing", rid):
                from models import Task, DrawingTaskAssociation

                # Start with a query that joins Task and DrawingTaskAssociation
                query = session.query(Task).join(DrawingTaskAssociation).join(cls)

                # Apply drawing filters
                if drawing_id is not None:
                    query = query.filter(cls.id == drawing_id)

                if drw_equipment_name is not None:
                    if exact_match:
                        query = query.filter(cls.drw_equipment_name == drw_equipment_name)
                    else:
                        query = query.filter(cls.drw_equipment_name.ilike(f"%{drw_equipment_name}%"))

                if drw_number is not None:
                    if exact_match:
                        query = query.filter(cls.drw_number == drw_number)
                    else:
                        query = query.filter(cls.drw_number.ilike(f"%{drw_number}%"))

                if drw_name is not None:
                    if exact_match:
                        query = query.filter(cls.drw_name == drw_name)
                    else:
                        query = query.filter(cls.drw_name.ilike(f"%{drw_name}%"))

                if drw_revision is not None:
                    if exact_match:
                        query = query.filter(cls.drw_revision == drw_revision)
                    else:
                        query = query.filter(cls.drw_revision.ilike(f"%{drw_revision}%"))

                if drw_spare_part_number is not None:
                    if exact_match:
                        query = query.filter(cls.drw_spare_part_number == drw_spare_part_number)
                    else:
                        query = query.filter(cls.drw_spare_part_number.ilike(f"%{drw_spare_part_number}%"))

                if file_path is not None:
                    if exact_match:
                        query = query.filter(cls.file_path == file_path)
                    else:
                        query = query.filter(cls.file_path.ilike(f"%{file_path}%"))

                # Apply task filters
                if task_id is not None:
                    query = query.filter(Task.id == task_id)

                if task_name is not None:
                    if exact_match:
                        query = query.filter(Task.name == task_name)
                    else:
                        query = query.filter(Task.name.ilike(f"%{task_name}%"))

                if task_description is not None:
                    if exact_match:
                        query = query.filter(Task.description == task_description)
                    else:
                        query = query.filter(Task.description.ilike(f"%{task_description}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"Drawing.get_tasks_by_drawing completed, found {len(results)} tasks", rid)
                return results

        except Exception as e:
            error_id(f"Error in Drawing.get_tasks_by_drawing: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for Drawing.get_tasks_by_drawing", rid)

    @classmethod
    def get_drawings_by_task(cls,
                             task_id: Optional[int] = None,
                             task_name: Optional[str] = None,
                             task_description: Optional[str] = None,
                             drawing_id: Optional[int] = None,
                             drw_equipment_name: Optional[str] = None,
                             drw_number: Optional[str] = None,
                             drw_name: Optional[str] = None,
                             drw_revision: Optional[str] = None,
                             drw_spare_part_number: Optional[str] = None,
                             file_path: Optional[str] = None,
                             exact_match: bool = False,
                             limit: int = 100,
                             request_id: Optional[str] = None,
                             session: Optional[Session] = None) -> List['Drawing']:
        """
        Get drawings associated with tasks based on flexible search criteria.

        Args:
            task_id: Optional task ID to filter by
            task_name: Optional task name to filter by
            task_description: Optional task description to filter by
            drawing_id: Optional drawing ID to filter by
            drw_equipment_name, drw_number, drw_name, drw_revision, drw_spare_part_number, file_path:
                Optional drawing attributes to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Drawing objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for Task.get_drawings_by_task", rid)

        # Log the search operation with request ID
        search_params = {
            'task_id': task_id,
            'task_name': task_name,
            'task_description': task_description,
            'drawing_id': drawing_id,
            'drw_equipment_name': drw_equipment_name,
            'drw_number': drw_number,
            'drw_name': drw_name,
            'drw_revision': drw_revision,
            'drw_spare_part_number': drw_spare_part_number,
            'file_path': file_path,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting Task.get_drawings_by_task with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("Task.get_drawings_by_task", rid):
                from models import Drawing, DrawingTaskAssociation

                # Start with a query that joins Drawing and DrawingTaskAssociation
                query = session.query(Drawing).join(DrawingTaskAssociation).join(cls)

                # Apply task filters
                if task_id is not None:
                    query = query.filter(cls.id == task_id)

                if task_name is not None:
                    if exact_match:
                        query = query.filter(cls.name == task_name)
                    else:
                        query = query.filter(cls.name.ilike(f"%{task_name}%"))

                if task_description is not None:
                    if exact_match:
                        query = query.filter(cls.description == task_description)
                    else:
                        query = query.filter(cls.description.ilike(f"%{task_description}%"))

                # Apply drawing filters
                if drawing_id is not None:
                    query = query.filter(Drawing.id == drawing_id)

                if drw_equipment_name is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_equipment_name == drw_equipment_name)
                    else:
                        query = query.filter(Drawing.drw_equipment_name.ilike(f"%{drw_equipment_name}%"))

                if drw_number is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_number == drw_number)
                    else:
                        query = query.filter(Drawing.drw_number.ilike(f"%{drw_number}%"))

                if drw_name is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_name == drw_name)
                    else:
                        query = query.filter(Drawing.drw_name.ilike(f"%{drw_name}%"))

                if drw_revision is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_revision == drw_revision)
                    else:
                        query = query.filter(Drawing.drw_revision.ilike(f"%{drw_revision}%"))

                if drw_spare_part_number is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_spare_part_number == drw_spare_part_number)
                    else:
                        query = query.filter(Drawing.drw_spare_part_number.ilike(f"%{drw_spare_part_number}%"))

                if file_path is not None:
                    if exact_match:
                        query = query.filter(Drawing.file_path == file_path)
                    else:
                        query = query.filter(Drawing.file_path.ilike(f"%{file_path}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"Task.get_drawings_by_task completed, found {len(results)} drawings", rid)
                return results

        except Exception as e:
            error_id(f"Error in Task.get_drawings_by_task: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for Task.get_drawings_by_task", rid)

    @classmethod
    @with_request_id
    def associate_drawing_with_task(cls, session, drawing_id, task_id, request_id=None):
        """
        Create an association between a drawing and a task if it doesn't already exist.

        Args:
            session: SQLAlchemy session
            drawing_id: ID of the drawing to associate
            task_id: ID of the task to associate
            request_id: Optional request ID for tracking

        Returns:
            The DrawingTaskAssociation instance (existing or new)
        """
        from models import DrawingTaskAssociation, Drawing, Task
        from sqlalchemy import and_

        # Get or use the provided request_id
        rid = request_id or get_request_id()

        debug_id(f"Associating drawing ID {drawing_id} with task ID {task_id}", rid)

        try:
            # Check if the drawing and task exist
            drawing = session.query(Drawing).get(drawing_id)
            task = session.query(cls).get(task_id)

            if not drawing:
                error_id(f"Drawing with ID {drawing_id} not found", rid)
                return None

            if not task:
                error_id(f"Task with ID {task_id} not found", rid)
                return None

            # Check if the association already exists
            existing = session.query(DrawingTaskAssociation).filter(
                and_(
                    DrawingTaskAssociation.drawing_id == drawing_id,
                    DrawingTaskAssociation.task_id == task_id
                )
            ).first()

            if existing:
                debug_id(f"Association between drawing ID {drawing_id} and task ID {task_id} already exists", rid)
                return existing

            # Create new association
            association = DrawingTaskAssociation(
                drawing_id=drawing_id,
                task_id=task_id
            )
            session.add(association)
            session.flush()

            debug_id(f"Created new association between drawing ID {drawing_id} and task ID {task_id}", rid)
            return association

        except Exception as e:
            error_id(f"Error associating drawing with task: {str(e)}", rid, exc_info=True)
            return None

class ImageTaskAssociation(Base):

    __tablename__ = 'image_task'

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('image.id'))
    task_id = Column(Integer, ForeignKey('task.id'))  # Corrected foreign key

    image = relationship("Image", back_populates="image_task")
    task = relationship("Task", back_populates="image_task")

    @classmethod
    @with_request_id
    def get_tasks_by_image(cls,
                           image_id: Optional[int] = None,
                           title: Optional[str] = None,
                           description: Optional[str] = None,
                           file_path: Optional[str] = None,
                           task_id: Optional[int] = None,
                           task_name: Optional[str] = None,
                           task_description: Optional[str] = None,
                           exact_match: bool = False,
                           limit: int = 100,
                           request_id: Optional[str] = None,
                           session: Optional[Session] = None) -> List['Task']:
        """
        Get tasks associated with images based on flexible search criteria.

        Args:
            image_id: Optional image ID to filter by
            title: Optional image title to filter by
            description: Optional image description to filter by
            file_path: Optional file path to filter by
            task_id: Optional task ID to filter by
            task_name: Optional task name to filter by
            task_description: Optional task description to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Task objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for Image.get_tasks_by_image", rid)

        # Log the search operation with request ID
        search_params = {
            'image_id': image_id,
            'title': title,
            'description': description,
            'file_path': file_path,
            'task_id': task_id,
            'task_name': task_name,
            'task_description': task_description,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting Image.get_tasks_by_image with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("Image.get_tasks_by_image", rid):
                from models import Task, ImageTaskAssociation

                # Start with a query that joins Task and ImageTaskAssociation
                query = session.query(Task).join(ImageTaskAssociation).join(cls)

                # Apply image filters
                if image_id is not None:
                    query = query.filter(cls.id == image_id)

                if title is not None:
                    if exact_match:
                        query = query.filter(cls.title == title)
                    else:
                        query = query.filter(cls.title.ilike(f"%{title}%"))

                if description is not None:
                    if exact_match:
                        query = query.filter(cls.description == description)
                    else:
                        query = query.filter(cls.description.ilike(f"%{description}%"))

                if file_path is not None:
                    if exact_match:
                        query = query.filter(cls.file_path == file_path)
                    else:
                        query = query.filter(cls.file_path.ilike(f"%{file_path}%"))

                # Apply task filters
                if task_id is not None:
                    query = query.filter(Task.id == task_id)

                if task_name is not None:
                    if exact_match:
                        query = query.filter(Task.name == task_name)
                    else:
                        query = query.filter(Task.name.ilike(f"%{task_name}%"))

                if task_description is not None:
                    if exact_match:
                        query = query.filter(Task.description == task_description)
                    else:
                        query = query.filter(Task.description.ilike(f"%{task_description}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"Image.get_tasks_by_image completed, found {len(results)} tasks", rid)
                return results

        except Exception as e:
            error_id(f"Error in Image.get_tasks_by_image: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for Image.get_tasks_by_image", rid)

    @classmethod
    @with_request_id
    def get_images_by_task(cls,
                           task_id: Optional[int] = None,
                           task_name: Optional[str] = None,
                           task_description: Optional[str] = None,
                           image_id: Optional[int] = None,
                           title: Optional[str] = None,
                           description: Optional[str] = None,
                           file_path: Optional[str] = None,
                           exact_match: bool = False,
                           limit: int = 100,
                           request_id: Optional[str] = None,
                           session: Optional[Session] = None) -> List['Image']:
        """
        Get images associated with tasks based on flexible search criteria.

        Args:
            task_id: Optional task ID to filter by
            task_name: Optional task name to filter by
            task_description: Optional task description to filter by
            image_id: Optional image ID to filter by
            title: Optional image title to filter by
            description: Optional image description to filter by
            file_path: Optional file path to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Image objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for Task.get_images_by_task", rid)

        # Log the search operation with request ID
        search_params = {
            'task_id': task_id,
            'task_name': task_name,
            'task_description': task_description,
            'image_id': image_id,
            'title': title,
            'description': description,
            'file_path': file_path,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting Task.get_images_by_task with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("Task.get_images_by_task", rid):
                from models import Image, ImageTaskAssociation

                # Start with a query that joins Image and ImageTaskAssociation
                query = session.query(Image).join(ImageTaskAssociation).join(cls)

                # Apply task filters
                if task_id is not None:
                    query = query.filter(cls.id == task_id)

                if task_name is not None:
                    if exact_match:
                        query = query.filter(cls.name == task_name)
                    else:
                        query = query.filter(cls.name.ilike(f"%{task_name}%"))

                if task_description is not None:
                    if exact_match:
                        query = query.filter(cls.description == task_description)
                    else:
                        query = query.filter(cls.description.ilike(f"%{task_description}%"))

                # Apply image filters
                if image_id is not None:
                    query = query.filter(Image.id == image_id)

                if title is not None:
                    if exact_match:
                        query = query.filter(Image.title == title)
                    else:
                        query = query.filter(Image.title.ilike(f"%{title}%"))

                if description is not None:
                    if exact_match:
                        query = query.filter(Image.description == description)
                    else:
                        query = query.filter(Image.description.ilike(f"%{description}%"))

                if file_path is not None:
                    if exact_match:
                        query = query.filter(Image.file_path == file_path)
                    else:
                        query = query.filter(Image.file_path.ilike(f"%{file_path}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"Task.get_images_by_task completed, found {len(results)} images", rid)
                return results

        except Exception as e:
            error_id(f"Error in Task.get_images_by_task: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for Task.get_images_by_task", rid)

    @classmethod
    @with_request_id
    def associate_image_task(cls,
                             image_id: int,
                             task_id: int,
                             request_id: Optional[str] = None,
                             session: Optional[Session] = None) -> Optional['ImageTaskAssociation']:
        """
        Associate an image with a task.

        Args:
            image_id: ID of the image to associate
            task_id: ID of the task to associate
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            The created ImageTaskAssociation object if successful, None otherwise
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for ImageTaskAssociation.associate_image_task", rid)

        # Log the operation with request ID
        debug_id(
            f"Starting ImageTaskAssociation.associate_image_task with parameters: image_id={image_id}, task_id={task_id}",
            rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("ImageTaskAssociation.associate_image_task", rid):
                from models import Image, Task

                # Check if image exists
                image = session.query(Image).filter(Image.id == image_id).first()
                if not image:
                    error_id(f"Error in ImageTaskAssociation.associate_image_task: Image with ID {image_id} not found",
                             rid)
                    return None

                # Check if task exists
                task = session.query(Task).filter(Task.id == task_id).first()
                if not task:
                    error_id(f"Error in ImageTaskAssociation.associate_image_task: Task with ID {task_id} not found",
                             rid)
                    return None

                # Check if association already exists
                existing = session.query(cls).filter(
                    cls.image_id == image_id,
                    cls.task_id == task_id
                ).first()

                if existing:
                    debug_id(f"Association between image {image_id} and task {task_id} already exists", rid)
                    return existing

                # Create new association
                association = cls(image_id=image_id, task_id=task_id)
                session.add(association)

                # Commit if we created the session
                if not session_provided:
                    session.commit()
                    debug_id(f"Committed new association between image {image_id} and task {task_id}", rid)

                return association

        except Exception as e:
            error_id(f"Error in ImageTaskAssociation.associate_image_task: {str(e)}", rid, exc_info=True)
            if not session_provided:
                session.rollback()
                debug_id(f"Rolled back transaction in ImageTaskAssociation.associate_image_task", rid)
            return None
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for ImageTaskAssociation.associate_image_task", rid)

class TaskToolAssociation(Base):
    __tablename__ = 'tool_task'

    id = Column(Integer, primary_key=True)
    tool_id = Column(Integer, ForeignKey('tool.id'), nullable=False)
    task_id = Column(Integer, ForeignKey('task.id'), nullable=False)

    # Relationships
    tool = relationship("Tool", back_populates="tool_tasks")
    task = relationship("Task", back_populates="tool_tasks")

    @classmethod
    @with_request_id
    def associate_task_tool(cls,
                            task_id: int,
                            tool_id: int,
                            request_id: Optional[str] = None,
                            session: Optional[Session] = None) -> Optional['TaskToolAssociation']:
        """
        Associate a task with a tool.

        Args:
            task_id: ID of the task to associate
            tool_id: ID of the tool to associate
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            The created TaskToolAssociation object if successful, None otherwise
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for TaskToolAssociation.associate_task_tool", rid)

        # Log the operation with request ID
        debug_id(
            f"Starting TaskToolAssociation.associate_task_tool with parameters: task_id={task_id}, tool_id={tool_id}",
            rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("TaskToolAssociation.associate_task_tool", rid):
                from models import Task, Tool

                # Check if task exists
                task = session.query(Task).filter(Task.id == task_id).first()
                if not task:
                    error_id(f"Error in TaskToolAssociation.associate_task_tool: Task with ID {task_id} not found", rid)
                    return None

                # Check if tool exists
                tool = session.query(Tool).filter(Tool.id == tool_id).first()
                if not tool:
                    error_id(f"Error in TaskToolAssociation.associate_task_tool: Tool with ID {tool_id} not found", rid)
                    return None

                # Check if association already exists
                existing = session.query(cls).filter(
                    cls.task_id == task_id,
                    cls.tool_id == tool_id
                ).first()

                if existing:
                    debug_id(f"Association between task {task_id} and tool {tool_id} already exists", rid)
                    return existing

                # Create new association
                association = cls(task_id=task_id, tool_id=tool_id)
                session.add(association)

                # Commit if we created the session
                if not session_provided:
                    session.commit()
                    debug_id(f"Committed new association between task {task_id} and tool {tool_id}", rid)

                return association

        except Exception as e:
            error_id(f"Error in TaskToolAssociation.associate_task_tool: {str(e)}", rid, exc_info=True)
            if not session_provided:
                session.rollback()
                debug_id(f"Rolled back transaction in TaskToolAssociation.associate_task_tool", rid)
            return None
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for TaskToolAssociation.associate_task_tool", rid)

    @classmethod
    @with_request_id
    def dissociate_task_tool(cls,
                             task_id: int,
                             tool_id: int,
                             request_id: Optional[str] = None,
                             session: Optional[Session] = None) -> bool:
        """
        Remove an association between a task and a tool.

        Args:
            task_id: ID of the task to dissociate
            tool_id: ID of the tool to dissociate
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            True if the association was removed, False otherwise
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for TaskToolAssociation.dissociate_task_tool", rid)

        # Log the operation with request ID
        debug_id(
            f"Starting TaskToolAssociation.dissociate_task_tool with parameters: task_id={task_id}, tool_id={tool_id}",
            rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("TaskToolAssociation.dissociate_task_tool", rid):
                # Find the association
                association = session.query(cls).filter(
                    cls.task_id == task_id,
                    cls.tool_id == tool_id
                ).first()

                if not association:
                    debug_id(f"No association found between task {task_id} and tool {tool_id}", rid)
                    return False

                # Delete the association
                session.delete(association)

                # Commit if we created the session
                if not session_provided:
                    session.commit()
                    debug_id(f"Removed association between task {task_id} and tool {tool_id}", rid)

                return True

        except Exception as e:
            error_id(f"Error in TaskToolAssociation.dissociate_task_tool: {str(e)}", rid, exc_info=True)
            if not session_provided:
                session.rollback()
                debug_id(f"Rolled back transaction in TaskToolAssociation.dissociate_task_tool", rid)
            return False
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for TaskToolAssociation.dissociate_task_tool", rid)

    @classmethod
    @with_request_id
    def get_tools_by_task(cls,
                          task_id: Optional[int] = None,
                          task_name: Optional[str] = None,
                          task_description: Optional[str] = None,
                          tool_id: Optional[int] = None,
                          tool_name: Optional[str] = None,
                          tool_type: Optional[str] = None,
                          tool_material: Optional[str] = None,
                          tool_size: Optional[str] = None,
                          exact_match: bool = False,
                          limit: int = 100,
                          request_id: Optional[str] = None,
                          session: Optional[Session] = None) -> List['Tool']:
        """
        Get tools associated with tasks based on flexible search criteria.

        Args:
            task_id: Optional task ID to filter by
            task_name: Optional task name to filter by
            task_description: Optional task description to filter by
            tool_id: Optional tool ID to filter by
            tool_name: Optional tool name to filter by
            tool_type: Optional tool type to filter by
            tool_material: Optional tool material to filter by
            tool_size: Optional tool size to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Tool objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for TaskToolAssociation.get_tools_by_task", rid)

        # Log the search operation with request ID
        search_params = {
            'task_id': task_id,
            'task_name': task_name,
            'task_description': task_description,
            'tool_id': tool_id,
            'tool_name': tool_name,
            'tool_type': tool_type,
            'tool_material': tool_material,
            'tool_size': tool_size,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting TaskToolAssociation.get_tools_by_task with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("TaskToolAssociation.get_tools_by_task", rid):
                from models import Task, Tool

                # Start with a query that joins Tool and TaskToolAssociation
                query = session.query(Tool).join(cls, Tool.id == cls.tool_id).join(Task, Task.id == cls.task_id)

                # Apply task filters
                if task_id is not None:
                    query = query.filter(Task.id == task_id)

                if task_name is not None:
                    if exact_match:
                        query = query.filter(Task.name == task_name)
                    else:
                        query = query.filter(Task.name.ilike(f"%{task_name}%"))

                if task_description is not None:
                    if exact_match:
                        query = query.filter(Task.description == task_description)
                    else:
                        query = query.filter(Task.description.ilike(f"%{task_description}%"))

                # Apply tool filters
                if tool_id is not None:
                    query = query.filter(Tool.id == tool_id)

                if tool_name is not None:
                    if exact_match:
                        query = query.filter(Tool.name == tool_name)
                    else:
                        query = query.filter(Tool.name.ilike(f"%{tool_name}%"))

                if tool_type is not None:
                    if exact_match:
                        query = query.filter(Tool.type == tool_type)
                    else:
                        query = query.filter(Tool.type.ilike(f"%{tool_type}%"))

                if tool_material is not None:
                    if exact_match:
                        query = query.filter(Tool.material == tool_material)
                    else:
                        query = query.filter(Tool.material.ilike(f"%{tool_material}%"))

                if tool_size is not None:
                    if exact_match:
                        query = query.filter(Tool.size == tool_size)
                    else:
                        query = query.filter(Tool.size.ilike(f"%{tool_size}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"TaskToolAssociation.get_tools_by_task completed, found {len(results)} tools", rid)
                return results

        except Exception as e:
            error_id(f"Error in TaskToolAssociation.get_tools_by_task: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for TaskToolAssociation.get_tools_by_task", rid)

    @classmethod
    @with_request_id
    def get_tasks_by_tool(cls,
                          tool_id: Optional[int] = None,
                          tool_name: Optional[str] = None,
                          tool_type: Optional[str] = None,
                          tool_material: Optional[str] = None,
                          tool_size: Optional[str] = None,
                          task_id: Optional[int] = None,
                          task_name: Optional[str] = None,
                          task_description: Optional[str] = None,
                          exact_match: bool = False,
                          limit: int = 100,
                          request_id: Optional[str] = None,
                          session: Optional[Session] = None) -> List['Task']:
        """
        Get tasks associated with tools based on flexible search criteria.

        Args:
            tool_id: Optional tool ID to filter by
            tool_name: Optional tool name to filter by
            tool_type: Optional tool type to filter by
            tool_material: Optional tool material to filter by
            tool_size: Optional tool size to filter by
            task_id: Optional task ID to filter by
            task_name: Optional task name to filter by
            task_description: Optional task description to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Task objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for TaskToolAssociation.get_tasks_by_tool", rid)

        # Log the search operation with request ID
        search_params = {
            'tool_id': tool_id,
            'tool_name': tool_name,
            'tool_type': tool_type,
            'tool_material': tool_material,
            'tool_size': tool_size,
            'task_id': task_id,
            'task_name': task_name,
            'task_description': task_description,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting TaskToolAssociation.get_tasks_by_tool with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("TaskToolAssociation.get_tasks_by_tool", rid):
                from models import Task, Tool

                # Start with a query that joins Task and TaskToolAssociation
                query = session.query(Task).join(cls, Task.id == cls.task_id).join(Tool, Tool.id == cls.tool_id)

                # Apply tool filters
                if tool_id is not None:
                    query = query.filter(Tool.id == tool_id)

                if tool_name is not None:
                    if exact_match:
                        query = query.filter(Tool.name == tool_name)
                    else:
                        query = query.filter(Tool.name.ilike(f"%{tool_name}%"))

                if tool_type is not None:
                    if exact_match:
                        query = query.filter(Tool.type == tool_type)
                    else:
                        query = query.filter(Tool.type.ilike(f"%{tool_type}%"))

                if tool_material is not None:
                    if exact_match:
                        query = query.filter(Tool.material == tool_material)
                    else:
                        query = query.filter(Tool.material.ilike(f"%{tool_material}%"))

                if tool_size is not None:
                    if exact_match:
                        query = query.filter(Tool.size == tool_size)
                    else:
                        query = query.filter(Tool.size.ilike(f"%{tool_size}%"))

                # Apply task filters
                if task_id is not None:
                    query = query.filter(Task.id == task_id)

                if task_name is not None:
                    if exact_match:
                        query = query.filter(Task.name == task_name)
                    else:
                        query = query.filter(Task.name.ilike(f"%{task_name}%"))

                if task_description is not None:
                    if exact_match:
                        query = query.filter(Task.description == task_description)
                    else:
                        query = query.filter(Task.description.ilike(f"%{task_description}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"TaskToolAssociation.get_tasks_by_tool completed, found {len(results)} tasks", rid)
                return results

        except Exception as e:
            error_id(f"Error in TaskToolAssociation.get_tasks_by_tool: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for TaskToolAssociation.get_tasks_by_tool", rid)

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

    @classmethod
    @with_request_id
    def add_to_db(cls, session=None, problem_id=None, position_id=None):
        """
        Get-or-create a ProblemPositionAssociation with the specified problem_id and position_id.
        If `session` is None, uses DatabaseConfig().get_main_session().
        Returns the ProblemPositionAssociation instance (new or existing).
        """
        # 1) ensure we have a session
        if session is None:
            session = DatabaseConfig().get_main_session()

        # 2) log input parameters
        debug_id(
            "add_to_db called with problem_id=%s, position_id=%s",
            problem_id, position_id
        )

        # Check for required parameters
        if problem_id is None or position_id is None:
            error_id("Both problem_id and position_id must be provided")
            raise ValueError("Both problem_id and position_id must be provided")

        # 3) build filter dict
        filters = {
            "problem_id": problem_id,
            "position_id": position_id,
        }

        try:
            # 4) try to find an existing row
            existing = session.query(cls).filter_by(**filters).first()
            if existing:
                info_id("Found existing ProblemPositionAssociation id=%s", existing.id)
                return existing

            # 5) not found → create new
            association = cls(**filters)
            session.add(association)
            session.commit()
            info_id("Created new ProblemPositionAssociation id=%s", association.id)
            return association

        except SQLAlchemyError as e:
            session.rollback()
            error_id("Failed to add_or_get ProblemPositionAssociation: %s", e, exc_info=True)
            raise

    @classmethod
    @with_request_id
    def get_positions_for_problem(cls, session, problem_id):
        """
        Get all positions associated with a specific problem.

        Args:
            session: SQLAlchemy session
            problem_id: ID of the problem

        Returns:
            List of Position objects associated with the problem
        """
        if session is None:
            session = DatabaseConfig().get_main_session()

        if problem_id is None:
            error_id("problem_id must be provided")
            return []

        try:
            # Query for all associations with this problem_id
            associations = session.query(cls).filter_by(problem_id=problem_id).all()

            # Extract the positions
            positions = [assoc.position for assoc in associations if assoc.position]

            info_id(f"Retrieved {len(positions)} positions for problem_id={problem_id}")
            return positions

        except SQLAlchemyError as e:
            error_id(f"Error retrieving positions for problem_id={problem_id}: {str(e)}", exc_info=True)
            return []

    @classmethod
    @with_request_id
    def get_problems_for_position(cls, session, position_id):
        """
        Get all problems associated with a specific position.

        Args:
            session: SQLAlchemy session
            position_id: ID of the position

        Returns:
            List of Problem objects associated with the position
        """
        if session is None:
            session = DatabaseConfig().get_main_session()

        if position_id is None:
            error_id("position_id must be provided")
            return []

        try:
            # Query for all associations with this position_id
            associations = session.query(cls).filter_by(position_id=position_id).all()

            # Extract the problems
            problems = [assoc.problem for assoc in associations if assoc.problem]

            info_id(f"Retrieved {len(problems)} problems for position_id={position_id}")
            return problems

        except SQLAlchemyError as e:
            error_id(f"Error retrieving problems for position_id={position_id}: {str(e)}", exc_info=True)
            return []

    @classmethod
    @with_request_id
    def delete_association(cls, session, problem_id=None, position_id=None, association_id=None):
        """
        Delete a problem-position association.
        Can delete by providing either the association_id or both problem_id and position_id.

        Args:
            session: SQLAlchemy session
            problem_id: ID of the problem (optional if association_id is provided)
            position_id: ID of the position (optional if association_id is provided)
            association_id: ID of the association (optional if both problem_id and position_id are provided)

        Returns:
            Boolean indicating success
        """
        if session is None:
            session = DatabaseConfig().get_main_session()

        try:
            if association_id:
                # Delete by association ID
                association = session.query(cls).filter_by(id=association_id).first()
                if association:
                    session.delete(association)
                    session.commit()
                    info_id(f"Deleted ProblemPositionAssociation id={association_id}")
                    return True
                else:
                    warn_id(f"No ProblemPositionAssociation found with id={association_id}")
                    return False
            elif problem_id and position_id:
                # Delete by problem_id and position_id
                association = session.query(cls).filter_by(
                    problem_id=problem_id, position_id=position_id).first()
                if association:
                    session.delete(association)
                    session.commit()
                    info_id(
                        f"Deleted ProblemPositionAssociation with problem_id={problem_id}, position_id={position_id}")
                    return True
                else:
                    warn_id(
                        f"No ProblemPositionAssociation found with problem_id={problem_id}, position_id={position_id}")
                    return False
            else:
                error_id("Either association_id or both problem_id and position_id must be provided")
                return False

        except SQLAlchemyError as e:
            session.rollback()
            error_id(f"Error deleting ProblemPositionAssociation: {str(e)}", exc_info=True)
            return False

    @classmethod
    @with_request_id
    def get_positions_for_problem_by_hierarchy(cls, session, problem_id, level_filters=None):
        """
        Get positions associated with a problem filtered by hierarchy levels.

        Args:
            session: SQLAlchemy session
            problem_id: ID of the problem
            level_filters: Dictionary with level names as keys and IDs as values
                           e.g., {'area_id': 1, 'equipment_group_id': 2}

        Returns:
            List of Position objects matching the criteria
        """
        if session is None:
            session = DatabaseConfig().get_main_session()

        if problem_id is None:
            error_id("problem_id must be provided")
            return []

        try:
            # Start with base query for the problem
            query = session.query(Position).join(
                cls, Position.id == cls.position_id
            ).filter(cls.problem_id == problem_id)

            # Apply hierarchy filters if provided
            if level_filters and isinstance(level_filters, dict):
                for field, value in level_filters.items():
                    if hasattr(Position, field) and value is not None:
                        query = query.filter(getattr(Position, field) == value)

            positions = query.all()
            info_id(f"Retrieved {len(positions)} positions for problem_id={problem_id} with hierarchy filters")
            return positions

        except SQLAlchemyError as e:
            error_id(f"Error in get_positions_for_problem_by_hierarchy: {str(e)}", exc_info=True)
            return []

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

    @classmethod
    @with_request_id
    def associate_complete_document_task(cls,
                                         complete_document_id: int,
                                         task_id: int,
                                         request_id: Optional[str] = None,
                                         session: Optional[Session] = None) -> Optional[
        'CompleteDocumentTaskAssociation']:
        """
        Associate a complete document with a task.

        Args:
            complete_document_id: ID of the complete document to associate
            task_id: ID of the task to associate
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            The created CompleteDocumentTaskAssociation object if successful, None otherwise
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(
                f"Created new database session for CompleteDocumentTaskAssociation.associate_complete_document_task",
                rid)

        # Log the operation with request ID
        debug_id(
            f"Starting CompleteDocumentTaskAssociation.associate_complete_document_task with parameters: complete_document_id={complete_document_id}, task_id={task_id}",
            rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("CompleteDocumentTaskAssociation.associate_complete_document_task", rid):
                from models import CompleteDocument, Task

                # Check if complete document exists
                complete_document = session.query(CompleteDocument).filter(
                    CompleteDocument.id == complete_document_id).first()
                if not complete_document:
                    error_id(
                        f"Error in CompleteDocumentTaskAssociation.associate_complete_document_task: CompleteDocument with ID {complete_document_id} not found",
                        rid)
                    return None

                # Check if task exists
                task = session.query(Task).filter(Task.id == task_id).first()
                if not task:
                    error_id(
                        f"Error in CompleteDocumentTaskAssociation.associate_complete_document_task: Task with ID {task_id} not found",
                        rid)
                    return None

                # Check if association already exists
                existing = session.query(cls).filter(
                    cls.complete_document_id == complete_document_id,
                    cls.task_id == task_id
                ).first()

                if existing:
                    debug_id(
                        f"Association between complete document {complete_document_id} and task {task_id} already exists",
                        rid)
                    return existing

                # Create new association
                association = cls(complete_document_id=complete_document_id, task_id=task_id)
                session.add(association)

                # Commit if we created the session
                if not session_provided:
                    session.commit()
                    debug_id(
                        f"Committed new association between complete document {complete_document_id} and task {task_id}",
                        rid)

                return association

        except Exception as e:
            error_id(f"Error in CompleteDocumentTaskAssociation.associate_complete_document_task: {str(e)}", rid,
                     exc_info=True)
            if not session_provided:
                session.rollback()
                debug_id(f"Rolled back transaction in CompleteDocumentTaskAssociation.associate_complete_document_task",
                         rid)
            return None
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(
                    f"Closed database session for CompleteDocumentTaskAssociation.associate_complete_document_task",
                    rid)

    @classmethod
    @with_request_id
    def dissociate_complete_document_task(cls,
                                          complete_document_id: int,
                                          task_id: int,
                                          request_id: Optional[str] = None,
                                          session: Optional[Session] = None) -> bool:
        """
        Remove an association between a complete document and a task.

        Args:
            complete_document_id: ID of the complete document to dissociate
            task_id: ID of the task to dissociate
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            True if the association was removed, False otherwise
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(
                f"Created new database session for CompleteDocumentTaskAssociation.dissociate_complete_document_task",
                rid)

        # Log the operation with request ID
        debug_id(
            f"Starting CompleteDocumentTaskAssociation.dissociate_complete_document_task with parameters: complete_document_id={complete_document_id}, task_id={task_id}",
            rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("CompleteDocumentTaskAssociation.dissociate_complete_document_task", rid):
                # Find the association
                association = session.query(cls).filter(
                    cls.complete_document_id == complete_document_id,
                    cls.task_id == task_id
                ).first()

                if not association:
                    debug_id(
                        f"No association found between complete document {complete_document_id} and task {task_id}",
                        rid)
                    return False

                # Delete the association
                session.delete(association)

                # Commit if we created the session
                if not session_provided:
                    session.commit()
                    debug_id(f"Removed association between complete document {complete_document_id} and task {task_id}",
                             rid)

                return True

        except Exception as e:
            error_id(f"Error in CompleteDocumentTaskAssociation.dissociate_complete_document_task: {str(e)}", rid,
                     exc_info=True)
            if not session_provided:
                session.rollback()
                debug_id(
                    f"Rolled back transaction in CompleteDocumentTaskAssociation.dissociate_complete_document_task",
                    rid)
            return False
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(
                    f"Closed database session for CompleteDocumentTaskAssociation.dissociate_complete_document_task",
                    rid)

    @classmethod
    @with_request_id
    def get_tasks_by_complete_document(cls,
                                       complete_document_id: Optional[int] = None,
                                       title: Optional[str] = None,
                                       file_path: Optional[str] = None,
                                       rev: Optional[str] = None,
                                       task_id: Optional[int] = None,
                                       task_name: Optional[str] = None,
                                       task_description: Optional[str] = None,
                                       exact_match: bool = False,
                                       limit: int = 100,
                                       request_id: Optional[str] = None,
                                       session: Optional[Session] = None) -> List['Task']:
        """
        Get tasks associated with complete documents based on flexible search criteria.

        Args:
            complete_document_id: Optional complete document ID to filter by
            title: Optional document title to filter by
            file_path: Optional file path to filter by
            rev: Optional revision to filter by
            task_id: Optional task ID to filter by
            task_name: Optional task name to filter by
            task_description: Optional task description to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Task objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for CompleteDocumentTaskAssociation.get_tasks_by_complete_document",
                     rid)

        # Log the search operation with request ID
        search_params = {
            'complete_document_id': complete_document_id,
            'title': title,
            'file_path': file_path,
            'rev': rev,
            'task_id': task_id,
            'task_name': task_name,
            'task_description': task_description,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(
            f"Starting CompleteDocumentTaskAssociation.get_tasks_by_complete_document with parameters: {logged_params}",
            rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("CompleteDocumentTaskAssociation.get_tasks_by_complete_document", rid):
                from models import Task, CompleteDocument

                # Start with a query that joins Task and CompleteDocumentTaskAssociation
                query = session.query(Task).join(cls, Task.id == cls.task_id).join(CompleteDocument,
                                                                                   CompleteDocument.id == cls.complete_document_id)

                # Apply complete document filters
                if complete_document_id is not None:
                    query = query.filter(CompleteDocument.id == complete_document_id)

                if title is not None:
                    if exact_match:
                        query = query.filter(CompleteDocument.title == title)
                    else:
                        query = query.filter(CompleteDocument.title.ilike(f"%{title}%"))

                if file_path is not None:
                    if exact_match:
                        query = query.filter(CompleteDocument.file_path == file_path)
                    else:
                        query = query.filter(CompleteDocument.file_path.ilike(f"%{file_path}%"))

                if rev is not None:
                    if exact_match:
                        query = query.filter(CompleteDocument.rev == rev)
                    else:
                        query = query.filter(CompleteDocument.rev.ilike(f"%{rev}%"))

                # Apply task filters
                if task_id is not None:
                    query = query.filter(Task.id == task_id)

                if task_name is not None:
                    if exact_match:
                        query = query.filter(Task.name == task_name)
                    else:
                        query = query.filter(Task.name.ilike(f"%{task_name}%"))

                if task_description is not None:
                    if exact_match:
                        query = query.filter(Task.description == task_description)
                    else:
                        query = query.filter(Task.description.ilike(f"%{task_description}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(
                    f"CompleteDocumentTaskAssociation.get_tasks_by_complete_document completed, found {len(results)} tasks",
                    rid)
                return results

        except Exception as e:
            error_id(f"Error in CompleteDocumentTaskAssociation.get_tasks_by_complete_document: {str(e)}", rid,
                     exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for CompleteDocumentTaskAssociation.get_tasks_by_complete_document",
                         rid)

    @classmethod
    @with_request_id
    def get_complete_documents_by_task(cls,
                                       task_id: Optional[int] = None,
                                       task_name: Optional[str] = None,
                                       task_description: Optional[str] = None,
                                       complete_document_id: Optional[int] = None,
                                       title: Optional[str] = None,
                                       file_path: Optional[str] = None,
                                       rev: Optional[str] = None,
                                       exact_match: bool = False,
                                       limit: int = 100,
                                       request_id: Optional[str] = None,
                                       session: Optional[Session] = None) -> List['CompleteDocument']:
        """
        Get complete documents associated with tasks based on flexible search criteria.

        Args:
            task_id: Optional task ID to filter by
            task_name: Optional task name to filter by
            task_description: Optional task description to filter by
            complete_document_id: Optional complete document ID to filter by
            title: Optional document title to filter by
            file_path: Optional file path to filter by
            rev: Optional revision to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of CompleteDocument objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for CompleteDocumentTaskAssociation.get_complete_documents_by_task",
                     rid)

        # Log the search operation with request ID
        search_params = {
            'task_id': task_id,
            'task_name': task_name,
            'task_description': task_description,
            'complete_document_id': complete_document_id,
            'title': title,
            'file_path': file_path,
            'rev': rev,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(
            f"Starting CompleteDocumentTaskAssociation.get_complete_documents_by_task with parameters: {logged_params}",
            rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("CompleteDocumentTaskAssociation.get_complete_documents_by_task", rid):
                from models import Task, CompleteDocument

                # Start with a query that joins CompleteDocument and CompleteDocumentTaskAssociation
                query = session.query(CompleteDocument).join(cls, CompleteDocument.id == cls.complete_document_id).join(
                    Task, Task.id == cls.task_id)

                # Apply task filters
                if task_id is not None:
                    query = query.filter(Task.id == task_id)

                if task_name is not None:
                    if exact_match:
                        query = query.filter(Task.name == task_name)
                    else:
                        query = query.filter(Task.name.ilike(f"%{task_name}%"))

                if task_description is not None:
                    if exact_match:
                        query = query.filter(Task.description == task_description)
                    else:
                        query = query.filter(Task.description.ilike(f"%{task_description}%"))

                # Apply complete document filters
                if complete_document_id is not None:
                    query = query.filter(CompleteDocument.id == complete_document_id)

                if title is not None:
                    if exact_match:
                        query = query.filter(CompleteDocument.title == title)
                    else:
                        query = query.filter(CompleteDocument.title.ilike(f"%{title}%"))

                if file_path is not None:
                    if exact_match:
                        query = query.filter(CompleteDocument.file_path == file_path)
                    else:
                        query = query.filter(CompleteDocument.file_path.ilike(f"%{file_path}%"))

                if rev is not None:
                    if exact_match:
                        query = query.filter(CompleteDocument.rev == rev)
                    else:
                        query = query.filter(CompleteDocument.rev.ilike(f"%{rev}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(
                    f"CompleteDocumentTaskAssociation.get_complete_documents_by_task completed, found {len(results)} complete documents",
                    rid)
                return results

        except Exception as e:
            error_id(f"Error in CompleteDocumentTaskAssociation.get_complete_documents_by_task: {str(e)}", rid,
                     exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for CompleteDocumentTaskAssociation.get_complete_documents_by_task",
                         rid)

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

    @classmethod
    @with_request_id
    def search(cls, session=None, **filters):
        """
        Search the 'part_position_image' table based on the provided filters.

        Args:
            session: SQLAlchemy session (optional).
            filters: A dictionary of filter parameters (e.g., part_id, position_id, image_id).

        Returns:
            List of matching 'PartPositionImageAssociation' objects.
        """
        if session is None:
            session = DatabaseConfig().get_main_session()

        # Get the request ID for logging
        request_id = get_request_id()

        # Log the start of the search operation
        info_id(f"Starting search with filters: {filters}", request_id=request_id)

        # Start with the base query
        query = session.query(cls)

        try:
            # Apply filters dynamically
            if filters:
                for field, value in filters.items():
                    if value is not None:  # Only apply non-None filters
                        query = query.filter(getattr(cls, field) == value)

            # Execute the query and log the result
            results = query.all()

            # Log the number of results found
            info_id(f"Search returned {len(results)} result(s) for filters: {filters}", request_id=request_id)

            return results
        except SQLAlchemyError as e:
            # Log the error
            error_id(f"Error during search operation with filters {filters}: {e}", request_id=request_id, exc_info=True)
            raise

    @classmethod
    @with_request_id
    def get_corresponding_position_ids(cls, session, area_id=None, equipment_group_id=None, model_id=None,
                                       asset_number_id=None, location_id=None):
        """
        Search for corresponding Position IDs based on the provided filters.
        Traverses the hierarchy and retrieves matching Position IDs.

        Args:
            session: SQLAlchemy session
            area_id: ID of the area (optional)
            equipment_group_id: ID of the equipment group (optional)
            model_id: ID of the model (optional)
            asset_number_id: ID of the asset number (optional)
            location_id: ID of the location (optional)

        Returns:
            List of Position IDs that match the criteria
        """
        # Get the request ID for logging
        request_id = get_request_id()

        # Log the start of the operation
        info_id(f"Starting get_corresponding_position_ids with filters: "
                f"area_id={area_id}, equipment_group_id={equipment_group_id}, "
                f"model_id={model_id}, asset_number_id={asset_number_id}, "
                f"location_id={location_id}", request_id=request_id)

        # Start by fetching the root-level positions based on area_id (or first level in hierarchy)
        try:
            positions = cls._get_positions_by_hierarchy(session, area_id=area_id,
                                                        equipment_group_id=equipment_group_id,
                                                        model_id=model_id,
                                                        asset_number_id=asset_number_id,
                                                        location_id=location_id)
            position_ids = [position.id for position in positions]

            # Log the number of Position IDs found
            info_id(f"Found {len(position_ids)} Position IDs for the given filters", request_id=request_id)

            return position_ids
        except SQLAlchemyError as e:
            error_id(f"Error during get_corresponding_position_ids with filters "
                     f"area_id={area_id}, equipment_group_id={equipment_group_id}, "
                     f"model_id={model_id}, asset_number_id={asset_number_id}, "
                     f"location_id={location_id}: {e}", request_id=request_id, exc_info=True)
            raise

    @classmethod
    @with_request_id
    def _get_positions_by_hierarchy(cls, session, area_id=None, equipment_group_id=None, model_id=None,
                                    asset_number_id=None, location_id=None):
        """
        Helper method to fetch positions based on hierarchical filters.

        Args:
            session: SQLAlchemy session
            area_id, equipment_group_id, model_id, asset_number_id, location_id: IDs for filtering

        Returns:
            List of Position objects that match the criteria
        """
        # Get the request ID for logging
        request_id = get_request_id()

        # Building the filter dynamically based on input parameters
        filters = {}
        if area_id:
            filters['area_id'] = area_id
        if equipment_group_id:
            filters['equipment_group_id'] = equipment_group_id
        if model_id:
            filters['model_id'] = model_id
        if asset_number_id:
            filters['asset_number_id'] = asset_number_id
        if location_id:
            filters['location_id'] = location_id

        try:
            # Log the filter being applied
            info_id(f"Applying filters to query: {filters}", request_id=request_id)

            # Query the Position table based on the filters
            query = session.query(Position).filter_by(**filters)

            # Execute and return the results
            positions = query.all()

            # Log the number of results
            info_id(f"Found {len(positions)} positions for the given filters", request_id=request_id)

            return positions
        except SQLAlchemyError as e:
            error_id(f"Error during _get_positions_by_hierarchy with filters {filters}: {e}", request_id=request_id,
                     exc_info=True)
            raise

class ImagePositionAssociation(Base):
    __tablename__ = 'image_position_association'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('image.id'))
    position_id = Column(Integer, ForeignKey('position.id'))
    
    image = relationship("Image", back_populates="image_position_association")
    position = relationship("Position", back_populates="image_position_association")

    @classmethod
    @with_request_id
    def associate_image_position(cls,
                                 image_id: int,
                                 position_id: int,
                                 request_id: Optional[str] = None,
                                 session: Optional[Session] = None) -> Optional['ImagePositionAssociation']:
        """
        Associate an image with a position.

        Args:
            image_id: ID of the image to associate
            position_id: ID of the position to associate
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            The created ImagePositionAssociation object if successful, None otherwise
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for ImagePositionAssociation.associate_image_position", rid)

        # Log the operation with request ID
        debug_id(
            f"Starting ImagePositionAssociation.associate_image_position with parameters: image_id={image_id}, position_id={position_id}",
            rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("ImagePositionAssociation.associate_image_position", rid):
                from models import Image, Position

                # Check if image exists
                image = session.query(Image).filter(Image.id == image_id).first()
                if not image:
                    error_id(
                        f"Error in ImagePositionAssociation.associate_image_position: Image with ID {image_id} not found",
                        rid)
                    return None

                # Check if position exists
                position = session.query(Position).filter(Position.id == position_id).first()
                if not position:
                    error_id(
                        f"Error in ImagePositionAssociation.associate_image_position: Position with ID {position_id} not found",
                        rid)
                    return None

                # Check if association already exists
                existing = session.query(cls).filter(
                    cls.image_id == image_id,
                    cls.position_id == position_id
                ).first()

                if existing:
                    debug_id(f"Association between image {image_id} and position {position_id} already exists", rid)
                    return existing

                # Create new association
                association = cls(image_id=image_id, position_id=position_id)
                session.add(association)

                # Commit if we created the session
                if not session_provided:
                    session.commit()
                    debug_id(f"Committed new association between image {image_id} and position {position_id}", rid)

                return association

        except Exception as e:
            error_id(f"Error in ImagePositionAssociation.associate_image_position: {str(e)}", rid, exc_info=True)
            if not session_provided:
                session.rollback()
                debug_id(f"Rolled back transaction in ImagePositionAssociation.associate_image_position", rid)
            return None
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for ImagePositionAssociation.associate_image_position", rid)

    @classmethod
    @with_request_id
    def dissociate_image_position(cls,
                                  image_id: int,
                                  position_id: int,
                                  request_id: Optional[str] = None,
                                  session: Optional[Session] = None) -> bool:
        """
        Remove an association between an image and a position.

        Args:
            image_id: ID of the image to dissociate
            position_id: ID of the position to dissociate
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            True if the association was removed, False otherwise
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for ImagePositionAssociation.dissociate_image_position", rid)

        # Log the operation with request ID
        debug_id(
            f"Starting ImagePositionAssociation.dissociate_image_position with parameters: image_id={image_id}, position_id={position_id}",
            rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("ImagePositionAssociation.dissociate_image_position", rid):
                # Find the association
                association = session.query(cls).filter(
                    cls.image_id == image_id,
                    cls.position_id == position_id
                ).first()

                if not association:
                    debug_id(f"No association found between image {image_id} and position {position_id}", rid)
                    return False

                # Delete the association
                session.delete(association)

                # Commit if we created the session
                if not session_provided:
                    session.commit()
                    debug_id(f"Removed association between image {image_id} and position {position_id}", rid)

                return True

        except Exception as e:
            error_id(f"Error in ImagePositionAssociation.dissociate_image_position: {str(e)}", rid, exc_info=True)
            if not session_provided:
                session.rollback()
                debug_id(f"Rolled back transaction in ImagePositionAssociation.dissociate_image_position", rid)
            return False
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for ImagePositionAssociation.dissociate_image_position", rid)

    @classmethod
    @with_request_id
    def get_positions_by_image(cls,
                               image_id: Optional[int] = None,
                               title: Optional[str] = None,
                               description: Optional[str] = None,
                               file_path: Optional[str] = None,
                               position_id: Optional[int] = None,
                               area_id: Optional[int] = None,
                               equipment_group_id: Optional[int] = None,
                               model_id: Optional[int] = None,
                               asset_number_id: Optional[int] = None,
                               location_id: Optional[int] = None,
                               subassembly_id: Optional[int] = None,
                               component_assembly_id: Optional[int] = None,
                               assembly_view_id: Optional[int] = None,
                               site_location_id: Optional[int] = None,
                               exact_match: bool = False,
                               limit: int = 100,
                               request_id: Optional[str] = None,
                               session: Optional[Session] = None) -> List['Position']:
        """
        Get positions associated with images based on flexible search criteria.

        Args:
            image_id: Optional image ID to filter by
            title: Optional image title to filter by
            description: Optional image description to filter by
            file_path: Optional file path to filter by
            position_id: Optional position ID to filter by
            area_id: Optional area ID to filter by
            equipment_group_id: Optional equipment group ID to filter by
            model_id: Optional model ID to filter by
            asset_number_id: Optional asset number ID to filter by
            location_id: Optional location ID to filter by
            subassembly_id: Optional subassembly ID to filter by
            component_assembly_id: Optional component assembly ID to filter by
            assembly_view_id: Optional assembly view ID to filter by
            site_location_id: Optional site location ID to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Position objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for ImagePositionAssociation.get_positions_by_image", rid)

        # Log the search operation with request ID
        search_params = {
            'image_id': image_id,
            'title': title,
            'description': description,
            'file_path': file_path,
            'position_id': position_id,
            'area_id': area_id,
            'equipment_group_id': equipment_group_id,
            'model_id': model_id,
            'asset_number_id': asset_number_id,
            'location_id': location_id,
            'subassembly_id': subassembly_id,
            'component_assembly_id': component_assembly_id,
            'assembly_view_id': assembly_view_id,
            'site_location_id': site_location_id,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting ImagePositionAssociation.get_positions_by_image with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("ImagePositionAssociation.get_positions_by_image", rid):
                from models import Image, Position

                # Start with a query that joins Position and ImagePositionAssociation
                query = session.query(Position).join(cls, Position.id == cls.position_id).join(Image,
                                                                                               Image.id == cls.image_id)

                # Apply image filters
                if image_id is not None:
                    query = query.filter(Image.id == image_id)

                if title is not None:
                    if exact_match:
                        query = query.filter(Image.title == title)
                    else:
                        query = query.filter(Image.title.ilike(f"%{title}%"))

                if description is not None:
                    if exact_match:
                        query = query.filter(Image.description == description)
                    else:
                        query = query.filter(Image.description.ilike(f"%{description}%"))

                if file_path is not None:
                    if exact_match:
                        query = query.filter(Image.file_path == file_path)
                    else:
                        query = query.filter(Image.file_path.ilike(f"%{file_path}%"))

                # Apply position filters
                if position_id is not None:
                    query = query.filter(Position.id == position_id)

                if area_id is not None:
                    query = query.filter(Position.area_id == area_id)

                if equipment_group_id is not None:
                    query = query.filter(Position.equipment_group_id == equipment_group_id)

                if model_id is not None:
                    query = query.filter(Position.model_id == model_id)

                if asset_number_id is not None:
                    query = query.filter(Position.asset_number_id == asset_number_id)

                if location_id is not None:
                    query = query.filter(Position.location_id == location_id)

                if subassembly_id is not None:
                    query = query.filter(Position.subassembly_id == subassembly_id)

                if component_assembly_id is not None:
                    query = query.filter(Position.component_assembly_id == component_assembly_id)

                if assembly_view_id is not None:
                    query = query.filter(Position.assembly_view_id == assembly_view_id)

                if site_location_id is not None:
                    query = query.filter(Position.site_location_id == site_location_id)

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"ImagePositionAssociation.get_positions_by_image completed, found {len(results)} positions",
                         rid)
                return results

        except Exception as e:
            error_id(f"Error in ImagePositionAssociation.get_positions_by_image: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for ImagePositionAssociation.get_positions_by_image", rid)

    @classmethod
    @with_request_id
    def get_images_by_position(cls,
                               position_id: Optional[int] = None,
                               area_id: Optional[int] = None,
                               equipment_group_id: Optional[int] = None,
                               model_id: Optional[int] = None,
                               asset_number_id: Optional[int] = None,
                               location_id: Optional[int] = None,
                               subassembly_id: Optional[int] = None,
                               component_assembly_id: Optional[int] = None,
                               assembly_view_id: Optional[int] = None,
                               site_location_id: Optional[int] = None,
                               image_id: Optional[int] = None,
                               title: Optional[str] = None,
                               description: Optional[str] = None,
                               file_path: Optional[str] = None,
                               exact_match: bool = False,
                               limit: int = 100,
                               request_id: Optional[str] = None,
                               session: Optional[Session] = None) -> List['Image']:
        """
        Get images associated with positions based on flexible search criteria.

        Args:
            position_id: Optional position ID to filter by
            area_id: Optional area ID to filter by
            equipment_group_id: Optional equipment group ID to filter by
            model_id: Optional model ID to filter by
            asset_number_id: Optional asset number ID to filter by
            location_id: Optional location ID to filter by
            subassembly_id: Optional subassembly ID to filter by
            component_assembly_id: Optional component assembly ID to filter by
            assembly_view_id: Optional assembly view ID to filter by
            site_location_id: Optional site location ID to filter by
            image_id: Optional image ID to filter by
            title: Optional image title to filter by
            description: Optional image description to filter by
            file_path: Optional file path to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Image objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for ImagePositionAssociation.get_images_by_position", rid)

        # Log the search operation with request ID
        search_params = {
            'position_id': position_id,
            'area_id': area_id,
            'equipment_group_id': equipment_group_id,
            'model_id': model_id,
            'asset_number_id': asset_number_id,
            'location_id': location_id,
            'subassembly_id': subassembly_id,
            'component_assembly_id': component_assembly_id,
            'assembly_view_id': assembly_view_id,
            'site_location_id': site_location_id,
            'image_id': image_id,
            'title': title,
            'description': description,
            'file_path': file_path,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting ImagePositionAssociation.get_images_by_position with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("ImagePositionAssociation.get_images_by_position", rid):
                from models import Image, Position

                # Start with a query that joins Image and ImagePositionAssociation
                query = session.query(Image).join(cls, Image.id == cls.image_id).join(Position,
                                                                                      Position.id == cls.position_id)

                # Apply position filters
                if position_id is not None:
                    query = query.filter(Position.id == position_id)

                if area_id is not None:
                    query = query.filter(Position.area_id == area_id)

                if equipment_group_id is not None:
                    query = query.filter(Position.equipment_group_id == equipment_group_id)

                if model_id is not None:
                    query = query.filter(Position.model_id == model_id)

                if asset_number_id is not None:
                    query = query.filter(Position.asset_number_id == asset_number_id)

                if location_id is not None:
                    query = query.filter(Position.location_id == location_id)

                if subassembly_id is not None:
                    query = query.filter(Position.subassembly_id == subassembly_id)

                if component_assembly_id is not None:
                    query = query.filter(Position.component_assembly_id == component_assembly_id)

                if assembly_view_id is not None:
                    query = query.filter(Position.assembly_view_id == assembly_view_id)

                if site_location_id is not None:
                    query = query.filter(Position.site_location_id == site_location_id)

                # Apply image filters
                if image_id is not None:
                    query = query.filter(Image.id == image_id)

                if title is not None:
                    if exact_match:
                        query = query.filter(Image.title == title)
                    else:
                        query = query.filter(Image.title.ilike(f"%{title}%"))

                if description is not None:
                    if exact_match:
                        query = query.filter(Image.description == description)
                    else:
                        query = query.filter(Image.description.ilike(f"%{description}%"))

                if file_path is not None:
                    if exact_match:
                        query = query.filter(Image.file_path == file_path)
                    else:
                        query = query.filter(Image.file_path.ilike(f"%{file_path}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(f"ImagePositionAssociation.get_images_by_position completed, found {len(results)} images", rid)
                return results

        except Exception as e:
            error_id(f"Error in ImagePositionAssociation.get_images_by_position: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for ImagePositionAssociation.get_images_by_position", rid)

class DrawingPositionAssociation(Base):
    __tablename__ = 'drawing_position'
    id = Column(Integer, primary_key=True)
    drawing_id = Column(Integer, ForeignKey('drawing.id'))
    position_id = Column(Integer, ForeignKey('position.id'))
    
    drawing = relationship("Drawing", back_populates="drawing_position")
    position = relationship("Position", back_populates="drawing_position")

    @classmethod
    @with_request_id
    def associate_drawing_position(cls,
                                   drawing_id: int,
                                   position_id: int,
                                   request_id: Optional[str] = None,
                                   session: Optional[Session] = None) -> Optional['DrawingPositionAssociation']:
        """
        Associate a drawing with a position.

        Args:
            drawing_id: ID of the drawing to associate
            position_id: ID of the position to associate
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            The created DrawingPositionAssociation object if successful, None otherwise
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for DrawingPositionAssociation.associate_drawing_position", rid)

        # Log the operation with request ID
        debug_id(
            f"Starting DrawingPositionAssociation.associate_drawing_position with parameters: drawing_id={drawing_id}, position_id={position_id}",
            rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("DrawingPositionAssociation.associate_drawing_position", rid):
                from models import Drawing, Position

                # Check if drawing exists
                drawing = session.query(Drawing).filter(Drawing.id == drawing_id).first()
                if not drawing:
                    error_id(
                        f"Error in DrawingPositionAssociation.associate_drawing_position: Drawing with ID {drawing_id} not found",
                        rid)
                    return None

                # Check if position exists
                position = session.query(Position).filter(Position.id == position_id).first()
                if not position:
                    error_id(
                        f"Error in DrawingPositionAssociation.associate_drawing_position: Position with ID {position_id} not found",
                        rid)
                    return None

                # Check if association already exists
                existing = session.query(cls).filter(
                    cls.drawing_id == drawing_id,
                    cls.position_id == position_id
                ).first()

                if existing:
                    debug_id(f"Association between drawing {drawing_id} and position {position_id} already exists", rid)
                    return existing

                # Create new association
                association = cls(drawing_id=drawing_id, position_id=position_id)
                session.add(association)

                # Commit if we created the session
                if not session_provided:
                    session.commit()
                    debug_id(f"Committed new association between drawing {drawing_id} and position {position_id}", rid)

                return association

        except Exception as e:
            error_id(f"Error in DrawingPositionAssociation.associate_drawing_position: {str(e)}", rid, exc_info=True)
            if not session_provided:
                session.rollback()
                debug_id(f"Rolled back transaction in DrawingPositionAssociation.associate_drawing_position", rid)
            return None
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for DrawingPositionAssociation.associate_drawing_position", rid)

    @classmethod
    @with_request_id
    def dissociate_drawing_position(cls,
                                    drawing_id: int,
                                    position_id: int,
                                    request_id: Optional[str] = None,
                                    session: Optional[Session] = None) -> bool:
        """
        Remove an association between a drawing and a position.

        Args:
            drawing_id: ID of the drawing to dissociate
            position_id: ID of the position to dissociate
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            True if the association was removed, False otherwise
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for DrawingPositionAssociation.dissociate_drawing_position", rid)

        # Log the operation with request ID
        debug_id(
            f"Starting DrawingPositionAssociation.dissociate_drawing_position with parameters: drawing_id={drawing_id}, position_id={position_id}",
            rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("DrawingPositionAssociation.dissociate_drawing_position", rid):
                # Find the association
                association = session.query(cls).filter(
                    cls.drawing_id == drawing_id,
                    cls.position_id == position_id
                ).first()

                if not association:
                    debug_id(f"No association found between drawing {drawing_id} and position {position_id}", rid)
                    return False

                # Delete the association
                session.delete(association)

                # Commit if we created the session
                if not session_provided:
                    session.commit()
                    debug_id(f"Removed association between drawing {drawing_id} and position {position_id}", rid)

                return True

        except Exception as e:
            error_id(f"Error in DrawingPositionAssociation.dissociate_drawing_position: {str(e)}", rid, exc_info=True)
            if not session_provided:
                session.rollback()
                debug_id(f"Rolled back transaction in DrawingPositionAssociation.dissociate_drawing_position", rid)
            return False
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for DrawingPositionAssociation.dissociate_drawing_position", rid)

    @classmethod
    @with_request_id
    def get_positions_by_drawing(cls,
                                 drawing_id: Optional[int] = None,
                                 drw_equipment_name: Optional[str] = None,
                                 drw_number: Optional[str] = None,
                                 drw_name: Optional[str] = None,
                                 drw_revision: Optional[str] = None,
                                 drw_spare_part_number: Optional[str] = None,
                                 file_path: Optional[str] = None,
                                 position_id: Optional[int] = None,
                                 area_id: Optional[int] = None,
                                 equipment_group_id: Optional[int] = None,
                                 model_id: Optional[int] = None,
                                 asset_number_id: Optional[int] = None,
                                 location_id: Optional[int] = None,
                                 subassembly_id: Optional[int] = None,
                                 component_assembly_id: Optional[int] = None,
                                 assembly_view_id: Optional[int] = None,
                                 site_location_id: Optional[int] = None,
                                 exact_match: bool = False,
                                 limit: int = 100,
                                 request_id: Optional[str] = None,
                                 session: Optional[Session] = None) -> List['Position']:
        """
        Get positions associated with drawings based on flexible search criteria.

        Args:
            drawing_id: Optional drawing ID to filter by
            drw_equipment_name: Optional equipment name to filter by
            drw_number: Optional drawing number to filter by
            drw_name: Optional drawing name to filter by
            drw_revision: Optional revision to filter by
            drw_spare_part_number: Optional spare part number to filter by
            file_path: Optional file path to filter by
            position_id: Optional position ID to filter by
            area_id: Optional area ID to filter by
            equipment_group_id: Optional equipment group ID to filter by
            model_id: Optional model ID to filter by
            asset_number_id: Optional asset number ID to filter by
            location_id: Optional location ID to filter by
            subassembly_id: Optional subassembly ID to filter by
            component_assembly_id: Optional component assembly ID to filter by
            assembly_view_id: Optional assembly view ID to filter by
            site_location_id: Optional site location ID to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Position objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for DrawingPositionAssociation.get_positions_by_drawing", rid)

        # Log the search operation with request ID
        search_params = {
            'drawing_id': drawing_id,
            'drw_equipment_name': drw_equipment_name,
            'drw_number': drw_number,
            'drw_name': drw_name,
            'drw_revision': drw_revision,
            'drw_spare_part_number': drw_spare_part_number,
            'file_path': file_path,
            'position_id': position_id,
            'area_id': area_id,
            'equipment_group_id': equipment_group_id,
            'model_id': model_id,
            'asset_number_id': asset_number_id,
            'location_id': location_id,
            'subassembly_id': subassembly_id,
            'component_assembly_id': component_assembly_id,
            'assembly_view_id': assembly_view_id,
            'site_location_id': site_location_id,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting DrawingPositionAssociation.get_positions_by_drawing with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("DrawingPositionAssociation.get_positions_by_drawing", rid):
                from models import Drawing, Position

                # Start with a query that joins Position and DrawingPositionAssociation
                query = session.query(Position).join(cls, Position.id == cls.position_id).join(Drawing,
                                                                                               Drawing.id == cls.drawing_id)

                # Apply drawing filters
                if drawing_id is not None:
                    query = query.filter(Drawing.id == drawing_id)

                if drw_equipment_name is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_equipment_name == drw_equipment_name)
                    else:
                        query = query.filter(Drawing.drw_equipment_name.ilike(f"%{drw_equipment_name}%"))

                if drw_number is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_number == drw_number)
                    else:
                        query = query.filter(Drawing.drw_number.ilike(f"%{drw_number}%"))

                if drw_name is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_name == drw_name)
                    else:
                        query = query.filter(Drawing.drw_name.ilike(f"%{drw_name}%"))

                if drw_revision is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_revision == drw_revision)
                    else:
                        query = query.filter(Drawing.drw_revision.ilike(f"%{drw_revision}%"))

                if drw_spare_part_number is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_spare_part_number == drw_spare_part_number)
                    else:
                        query = query.filter(Drawing.drw_spare_part_number.ilike(f"%{drw_spare_part_number}%"))

                if file_path is not None:
                    if exact_match:
                        query = query.filter(Drawing.file_path == file_path)
                    else:
                        query = query.filter(Drawing.file_path.ilike(f"%{file_path}%"))

                # Apply position filters
                if position_id is not None:
                    query = query.filter(Position.id == position_id)

                if area_id is not None:
                    query = query.filter(Position.area_id == area_id)

                if equipment_group_id is not None:
                    query = query.filter(Position.equipment_group_id == equipment_group_id)

                if model_id is not None:
                    query = query.filter(Position.model_id == model_id)

                if asset_number_id is not None:
                    query = query.filter(Position.asset_number_id == asset_number_id)

                if location_id is not None:
                    query = query.filter(Position.location_id == location_id)

                if subassembly_id is not None:
                    query = query.filter(Position.subassembly_id == subassembly_id)

                if component_assembly_id is not None:
                    query = query.filter(Position.component_assembly_id == component_assembly_id)

                if assembly_view_id is not None:
                    query = query.filter(Position.assembly_view_id == assembly_view_id)

                if site_location_id is not None:
                    query = query.filter(Position.site_location_id == site_location_id)

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(
                    f"DrawingPositionAssociation.get_positions_by_drawing completed, found {len(results)} positions",
                    rid)
                return results

        except Exception as e:
            error_id(f"Error in DrawingPositionAssociation.get_positions_by_drawing: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for DrawingPositionAssociation.get_positions_by_drawing", rid)

    @classmethod
    @with_request_id
    def get_drawings_by_position(cls,
                                 position_id: Optional[int] = None,
                                 area_id: Optional[int] = None,
                                 equipment_group_id: Optional[int] = None,
                                 model_id: Optional[int] = None,
                                 asset_number_id: Optional[int] = None,
                                 location_id: Optional[int] = None,
                                 subassembly_id: Optional[int] = None,
                                 component_assembly_id: Optional[int] = None,
                                 assembly_view_id: Optional[int] = None,
                                 site_location_id: Optional[int] = None,
                                 drawing_id: Optional[int] = None,
                                 drw_equipment_name: Optional[str] = None,
                                 drw_number: Optional[str] = None,
                                 drw_name: Optional[str] = None,
                                 drw_revision: Optional[str] = None,
                                 drw_spare_part_number: Optional[str] = None,
                                 file_path: Optional[str] = None,
                                 exact_match: bool = False,
                                 limit: int = 100,
                                 request_id: Optional[str] = None,
                                 session: Optional[Session] = None) -> List['Drawing']:
        """
        Get drawings associated with positions based on flexible search criteria.

        Args:
            position_id: Optional position ID to filter by
            area_id: Optional area ID to filter by
            equipment_group_id: Optional equipment group ID to filter by
            model_id: Optional model ID to filter by
            asset_number_id: Optional asset number ID to filter by
            location_id: Optional location ID to filter by
            subassembly_id: Optional subassembly ID to filter by
            component_assembly_id: Optional component assembly ID to filter by
            assembly_view_id: Optional assembly view ID to filter by
            site_location_id: Optional site location ID to filter by
            drawing_id: Optional drawing ID to filter by
            drw_equipment_name: Optional equipment name to filter by
            drw_number: Optional drawing number to filter by
            drw_name: Optional drawing name to filter by
            drw_revision: Optional revision to filter by
            drw_spare_part_number: Optional spare part number to filter by
            file_path: Optional file path to filter by
            exact_match: If True, performs exact matching instead of partial matching for string fields
            limit: Maximum number of results to return (default 100)
            request_id: Optional request ID for tracking this operation in logs
            session: Optional SQLAlchemy session. If None, a new session will be created

        Returns:
            List of Drawing objects matching the search criteria
        """
        # Get or use the provided request_id
        rid = request_id or get_request_id()

        # Get a database session if one wasn't provided
        db_config = DatabaseConfig()
        session_provided = session is not None
        if not session_provided:
            session = db_config.get_main_session()
            debug_id(f"Created new database session for DrawingPositionAssociation.get_drawings_by_position", rid)

        # Log the search operation with request ID
        search_params = {
            'position_id': position_id,
            'area_id': area_id,
            'equipment_group_id': equipment_group_id,
            'model_id': model_id,
            'asset_number_id': asset_number_id,
            'location_id': location_id,
            'subassembly_id': subassembly_id,
            'component_assembly_id': component_assembly_id,
            'assembly_view_id': assembly_view_id,
            'site_location_id': site_location_id,
            'drawing_id': drawing_id,
            'drw_equipment_name': drw_equipment_name,
            'drw_number': drw_number,
            'drw_name': drw_name,
            'drw_revision': drw_revision,
            'drw_spare_part_number': drw_spare_part_number,
            'file_path': file_path,
            'exact_match': exact_match,
            'limit': limit
        }
        # Filter out None values for cleaner logging
        logged_params = {k: v for k, v in search_params.items() if v is not None}
        debug_id(f"Starting DrawingPositionAssociation.get_drawings_by_position with parameters: {logged_params}", rid)

        try:
            # Use the timed operation context manager with request ID
            with log_timed_operation("DrawingPositionAssociation.get_drawings_by_position", rid):
                from models import Drawing, Position

                # Start with a query that joins Drawing and DrawingPositionAssociation
                query = session.query(Drawing).join(cls, Drawing.id == cls.drawing_id).join(Position,
                                                                                            Position.id == cls.position_id)

                # Apply position filters
                if position_id is not None:
                    query = query.filter(Position.id == position_id)

                if area_id is not None:
                    query = query.filter(Position.area_id == area_id)

                if equipment_group_id is not None:
                    query = query.filter(Position.equipment_group_id == equipment_group_id)

                if model_id is not None:
                    query = query.filter(Position.model_id == model_id)

                if asset_number_id is not None:
                    query = query.filter(Position.asset_number_id == asset_number_id)

                if location_id is not None:
                    query = query.filter(Position.location_id == location_id)

                if subassembly_id is not None:
                    query = query.filter(Position.subassembly_id == subassembly_id)

                if component_assembly_id is not None:
                    query = query.filter(Position.component_assembly_id == component_assembly_id)

                if assembly_view_id is not None:
                    query = query.filter(Position.assembly_view_id == assembly_view_id)

                if site_location_id is not None:
                    query = query.filter(Position.site_location_id == site_location_id)

                # Apply drawing filters
                if drawing_id is not None:
                    query = query.filter(Drawing.id == drawing_id)

                if drw_equipment_name is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_equipment_name == drw_equipment_name)
                    else:
                        query = query.filter(Drawing.drw_equipment_name.ilike(f"%{drw_equipment_name}%"))

                if drw_number is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_number == drw_number)
                    else:
                        query = query.filter(Drawing.drw_number.ilike(f"%{drw_number}%"))

                if drw_name is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_name == drw_name)
                    else:
                        query = query.filter(Drawing.drw_name.ilike(f"%{drw_name}%"))

                if drw_revision is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_revision == drw_revision)
                    else:
                        query = query.filter(Drawing.drw_revision.ilike(f"%{drw_revision}%"))

                if drw_spare_part_number is not None:
                    if exact_match:
                        query = query.filter(Drawing.drw_spare_part_number == drw_spare_part_number)
                    else:
                        query = query.filter(Drawing.drw_spare_part_number.ilike(f"%{drw_spare_part_number}%"))

                if file_path is not None:
                    if exact_match:
                        query = query.filter(Drawing.file_path == file_path)
                    else:
                        query = query.filter(Drawing.file_path.ilike(f"%{file_path}%"))

                # Make results distinct to avoid duplicates
                query = query.distinct()

                # Apply limit
                query = query.limit(limit)

                # Execute query and log results
                results = query.all()
                debug_id(
                    f"DrawingPositionAssociation.get_drawings_by_position completed, found {len(results)} drawings",
                    rid)
                return results

        except Exception as e:
            error_id(f"Error in DrawingPositionAssociation.get_drawings_by_position: {str(e)}", rid, exc_info=True)
            return []
        finally:
            # Close the session if we created it
            if not session_provided:
                session.close()
                debug_id(f"Closed database session for DrawingPositionAssociation.get_drawings_by_position", rid)

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
    @with_request_id
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
    LEVEL_I = 'LEVEL_I'
    STANDARD = 'STANDARD'

# region Todo: Create and Refactor class's to a new class called ModelsConfig

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

# endregion

# Define the User model
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    type = Column(String(50))  # This column is needed for SQLAlchemy inheritance
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
    logins = relationship('UserLogin', back_populates='user')

    # Add mapper arguments for inheritance
    __mapper_args__ = {
        'polymorphic_identity': 'user',
        'polymorphic_on': type
    }

    def set_password(self, password):
        self.hashed_password = generate_password_hash(password)

    def check_password_hash(self, password):
        return check_password_hash(self.hashed_password, password)

    @classmethod
    @with_request_id
    def create_new_user(cls, employee_id, first_name, last_name, password, current_shift=None,
                        primary_area=None, age=None, education_level=None, start_date=None,
                        text_to_voice="default", voice_to_text="default"):
        """
        Creates a new user with comprehensive error handling and proper session management.
        """
        logger = logging.getLogger('ematac_logger')
        logger.info(f"============ CREATE_NEW_USER STARTED for {employee_id} ============")

        from modules.configuration.config_env import DatabaseConfig
        from sqlalchemy.exc import IntegrityError, SQLAlchemyError
        import traceback

        # Get database session
        try:
            logger.info("Getting database session...")
            db_config = DatabaseConfig()
            session = db_config.get_main_session()
            logger.debug(f"Got database session: {session}")
        except Exception as e:
            logger.error(f"ERROR GETTING DATABASE SESSION: {e}")
            logger.error(traceback.format_exc())
            return False, f"Database connection error: {str(e)}"

        try:
            # Create new user object
            logger.info(f"Creating User object with: {employee_id}, {first_name}, {last_name}")
            new_user = User(
                employee_id=employee_id,
                first_name=first_name,
                last_name=last_name,
                current_shift=current_shift,
                primary_area=primary_area,
                age=age,
                education_level=education_level,
                start_date=start_date,
                user_level=UserLevel.STANDARD
            )
            logger.debug("Created User object successfully")

            # Set password
            logger.debug("Setting password...")
            new_user.set_password(password)
            logger.debug("Password set successfully")

            # Add to session
            logger.debug("Adding user to database session...")
            session.add(new_user)
            logger.debug("User added to session")

            # Commit changes
            logger.info("Committing session...")
            session.commit()
            logger.info("Session committed successfully")
            logger.info(f"User created successfully: {employee_id}")

            return True, "User created successfully"

        except IntegrityError as e:
            logger.error(f"INTEGRITY ERROR: {str(e)}")
            session.rollback()
            error_msg = str(e)
            logger.error(f"IntegrityError creating user: {error_msg}")

            if "UNIQUE constraint failed" in error_msg:
                return False, f"A user with employee ID {employee_id} already exists."
            else:
                return False, f"Database integrity error: {error_msg}"

        except SQLAlchemyError as e:
            logger.error(f"SQL ALCHEMY ERROR: {str(e)}")
            session.rollback()
            error_msg = str(e)
            logger.error(f"SQLAlchemy error creating user: {error_msg}")
            return False, f"Database error: {error_msg}"

        except Exception as e:
            logger.error(f"UNEXPECTED ERROR: {str(e)}")
            logger.error(traceback.format_exc())
            session.rollback()
            error_msg = str(e)
            logger.error(f"Unexpected error creating user: {error_msg}")
            return False, f"An unexpected error occurred: {error_msg}"

        finally:
            # Always close the session
            try:
                logger.debug("Closing database session")
                session.close()
                logger.debug("Database session closed")
            except Exception as e:
                logger.error(f"ERROR CLOSING SESSION: {e}")

            logger.info(f"============ CREATE_NEW_USER FINISHED for {employee_id} ============")

class UserLogin(Base):
    __tablename__ = 'user_logins'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    session_id = Column(String, nullable=False)  # Flask session ID
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)  # Browser/client info
    login_time = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    logout_time = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)

    # Relationship to User
    user = relationship("User", back_populates="logins")

    def __init__(self, user_id, session_id, ip_address=None, user_agent=None):
        self.user_id = user_id
        self.session_id = session_id
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.login_time = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.is_active = True

class KivyUser(User):
    __tablename__ = 'kivy_users'

    id = Column(Integer, ForeignKey('users.id'), primary_key=True)

    # Relationship to layouts
    layouts = relationship("UserLayout", back_populates="user", cascade="all, delete-orphan")

    __mapper_args__ = {
        'polymorphic_identity': 'kivy_user',
    }

    def get_layout(self, layout_name):
        """Get a specific layout by name"""
        from sqlalchemy.orm.session import object_session

        session = object_session(self)
        if not session:
            raise ValueError("User is not attached to a session")

        layout = session.query(UserLayout).filter_by(
            user_id=self.id,
            layout_name=layout_name
        ).first()

        if layout:
            import json
            return json.loads(layout.layout_data)
        return None

    def save_layout(self, layout_name, layout_data):
        """Save a layout with a specific name"""
        from sqlalchemy.orm.session import object_session
        import json

        session = object_session(self)
        if not session:
            raise ValueError("User is not attached to a session")

        layout = session.query(UserLayout).filter_by(
            user_id=self.id,
            layout_name=layout_name
        ).first()

        if layout:
            # Update existing layout
            layout.layout_data = json.dumps(layout_data)
        else:
            # Create new layout
            layout = UserLayout(
                user_id=self.id,
                layout_name=layout_name,
                layout_data=json.dumps(layout_data)
            )
            session.add(layout)

        session.commit()

    def get_all_layouts(self):
        """Get all layouts for this user"""
        from sqlalchemy.orm.session import object_session
        import json

        session = object_session(self)
        if not session:
            raise ValueError("User is not attached to a session")

        layouts = session.query(UserLayout).filter_by(user_id=self.id).all()

        return {layout.layout_name: json.loads(layout.layout_data) for layout in layouts}

    def delete_layout(self, layout_name):
        """Delete a layout by name"""
        from sqlalchemy.orm.session import object_session

        session = object_session(self)
        if not session:
            raise ValueError("User is not attached to a session")

        layout = session.query(UserLayout).filter_by(
            user_id=self.id,
            layout_name=layout_name
        ).first()

        if layout:
            session.delete(layout)
            session.commit()
            return True
        return False

    @classmethod
    @with_request_id
    def ensure_kivy_user(cls, session, user_or_id):
        """
        Ensures a KivyUser record exists for a given User or user ID.
        Args:
            session: SQLAlchemy session
            user_or_id: A User instance or user ID
        Returns:
            KivyUser instance if successful, None if failed
        """
        # Import logger and SQLAlchemy text
        import logging
        from sqlalchemy import text
        from sqlalchemy.exc import SQLAlchemyError

        logger = logging.getLogger(__name__)

        if user_or_id is None:
            logger.error("Cannot ensure KivyUser for None user")
            return None

        # Get user ID
        user_id = user_or_id.id if hasattr(user_or_id, 'id') else user_or_id

        # Try to get existing KivyUser
        kivy_user = session.query(cls).filter(cls.id == user_id).first()

        if kivy_user:
            logger.debug(f"Found existing KivyUser for ID {user_id}")
            return kivy_user

        # No KivyUser found, check if the User exists and has type='kivy_user'
        user = None
        if hasattr(user_or_id, 'id'):
            user = user_or_id
        else:
            user = session.query(User).filter(User.id == user_id).first()

        if not user:
            logger.error(f"No User found with ID {user_id}")
            return None

        # Check if the user is already marked as a KivyUser
        if user.type != 'kivy_user':
            # Update the user type to 'kivy_user'
            logger.info(f"Updating User {user.employee_id} (ID: {user.id}) type to 'kivy_user'")
            user.type = 'kivy_user'
            try:
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Error updating User type: {e}")
                return None

        # Create the KivyUser record
        try:
            logger.info(f"Creating KivyUser record for User {user.employee_id} (ID: {user.id})")
            session.execute(
                text("INSERT INTO kivy_users (id) VALUES (:id)"),
                {"id": user.id}
            )
            session.commit()

            # Fetch the newly created KivyUser
            kivy_user = session.query(cls).filter(cls.id == user.id).first()

            if kivy_user:
                logger.info(f"Successfully created KivyUser for {user.employee_id} (ID: {user.id})")
                return kivy_user
            else:
                logger.error(f"Failed to retrieve created KivyUser for {user.employee_id} (ID: {user.id})")
                return None

        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error creating KivyUser record: {e}")
            return None

class UserLayout(Base):
    __tablename__ = 'user_layouts'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('kivy_users.id'), nullable=False)
    layout_name = Column(String, nullable=False)
    layout_data = Column(Text, nullable=False)  # Store JSON layout data

    # Relationship to KivyUser
    user = relationship("KivyUser", back_populates="layouts")

    # Create a unique constraint to prevent duplicate layout names for a user
    __table_args__ = (
        UniqueConstraint('user_id', 'layout_name', name='uix_user_layout_name'),
    )

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

    @classmethod
    @with_request_id
    def associate_with_tool(cls, session, image_id, tool_id, description=None):
        """Associate an existing image with a tool in the database.

        Args:
            session: The database session
            image_id: ID of the existing image to associate
            tool_id: ID of the tool to associate with the image
            description: Optional description for this specific association

        Returns:
            The created ToolImageAssociation object or existing one if found
        """
        # Import locally to avoid circular dependencies
        from sqlalchemy import and_

        logger.info(f"Associating image ID {image_id} with tool ID {tool_id}")

        # Check if association already exists
        existing_association = session.query(cls).filter(
            and_(
                cls.image_id == image_id,
                cls.tool_id == tool_id
            )
        ).first()

        if existing_association is not None:
            logger.info(f"Association already exists between image ID {image_id} and tool ID {tool_id}")

            # Update description if provided and different
            if description is not None and existing_association.description != description:
                existing_association.description = description
                logger.info(f"Updated description for existing association")

            return existing_association
        else:
            # Create new association
            logger.info(f"Creating new association between image ID {image_id} and tool ID {tool_id}")
            new_association = cls(
                image_id=image_id,
                tool_id=tool_id,
                description=description
            )
            session.add(new_association)
            session.flush()  # Get ID without committing transaction

            logger.info(f"Created ToolImageAssociation with ID {new_association.id}")
            return new_association

    @classmethod
    @with_request_id
    def add_and_associate_with_tool(cls, session, title, file_path, tool_id, description="",
                                    association_description=None):
        """Add an image to the database and associate it with a tool in one operation.

        Args:
            session: The database session
            title: Title for the image
            file_path: Path to the image file
            tool_id: ID of the tool to associate with the image
            description: Description for the image itself
            association_description: Optional description for the tool-image association

        Returns:
            Tuple of (Image object, ToolImageAssociation object)
        """
        # Import the Image class locally to avoid circular imports
        from modules.emtacdb.emtacdb_fts import Image

        # First add the image to the database
        image = Image.add_to_db(session, title, file_path, description)

        # Then create the association
        association = cls.associate_with_tool(
            session,
            image_id=image.id,
            tool_id=tool_id,
            description=association_description
        )

        return image, association

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

# region Todo: Refactor to class AIModelConfig, which will be refactor to "ModelsConfig(Bas)"

def load_config_from_db():
    """
    Load AI model configuration from the database.

    Returns:
        Tuple of (current_ai_model, current_embedding_model)
    """
    from modules.configuration.config_env import DatabaseConfig

    db_config = DatabaseConfig()
    session = db_config.get_main_session()
    try:
        ai_model_config = session.query(AIModelConfig).filter_by(key="CURRENT_AI_MODEL").first()
        embedding_model_config = session.query(AIModelConfig).filter_by(key="CURRENT_EMBEDDING_MODEL").first()

        current_ai_model = ai_model_config.value if ai_model_config else "NoAIModel"
        current_embedding_model = embedding_model_config.value if embedding_model_config else "NoEmbeddingModel"

        return current_ai_model, current_embedding_model
    finally:
        session.close()


def load_image_model_config_from_db():
    """
    Load image model configuration from the database.

    Returns:
        String representing the current image model
    """
    from modules.configuration.config_env import DatabaseConfig

    db_config = DatabaseConfig()
    session = db_config.get_main_session()
    try:
        image_model_config = session.query(ImageModelConfig).filter_by(key="CURRENT_IMAGE_MODEL").first()
        current_image_model = image_model_config.value if image_model_config else "no_model"

        return current_image_model
    finally:
        session.close()

# endregion


# Ask for API key if it's empty
if not OPENAI_API_KEY:
    OPENAI_API_KEY = input("Enter your OpenAI API key: ")
    with open('../configuration/config.py', 'w') as config_file:
        config_file.write(f'BASE_DIR = "{BASE_DIR}"\n')
        config_file.write(f'copy_files = {COPY_FILES}\n')
        config_file.write(f'OPENAI_API_KEY = "{OPENAI_API_KEY}"\n')
