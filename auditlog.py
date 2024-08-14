import os
import logging
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime,insert
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from snapshot_utils import (
    get_latest_version_info, add_version_info, create_sitlocation_snapshot, 
    create_position_snapshot, create_area_snapshot, create_equipment_group_snapshot, 
    create_model_snapshot, create_asset_number_snapshot, create_part_snapshot, 
    create_image_snapshot, create_image_embedding_snapshot, create_drawing_snapshot, 
    create_document_snapshot, create_complete_document_snapshot, create_problem_snapshot, 
    create_solution_snapshot, create_drawing_part_association_snapshot, 
    create_part_problem_association_snapshot, create_part_solution_association_snapshot, 
    create_drawing_problem_association_snapshot, create_drawing_solution_association_snapshot, 
    create_problem_position_association_snapshot, create_complete_document_problem_association_snapshot, 
    create_complete_document_solution_association_snapshot, create_image_problem_association_snapshot, 
    create_image_solution_association_snapshot, create_image_position_association_snapshot, 
    create_drawing_position_association_snapshot, create_completed_document_position_association_snapshot, 
    create_image_completed_document_association_snapshot, create_snapshot, 
    create_parts_position_association_snapshot
)
from config import DATABASE_DIR, REVISION_CONTROL_DB_PATH 
from emtac_revision_control_db import (
    VersionInfo, RevisionControlBase, revision_control_engine, LocationSnapshot, 
    SiteLocationSnapshot, PositionSnapshot, AreaSnapshot, EquipmentGroupSnapshot, ModelSnapshot, 
    AssetNumberSnapshot, PartSnapshot, ImageSnapshot, ImageEmbeddingSnapshot, DrawingSnapshot, 
    DocumentSnapshot, CompleteDocumentSnapshot, ProblemSnapshot, SolutionSnapshot, 
    DrawingPartAssociationSnapshot, PartProblemAssociationSnapshot, PartSolutionAssociationSnapshot, 
    PartsPositionAssociationSnapshot, DrawingProblemAssociationSnapshot, DrawingSolutionAssociationSnapshot, 
    ProblemPositionAssociationSnapshot, CompleteDocumentProblemAssociationSnapshot, 
    CompleteDocumentSolutionAssociationSnapshot, ImageProblemAssociationSnapshot, 
    ImageSolutionAssociationSnapshot, ImagePositionAssociationSnapshot, DrawingPositionAssociationSnapshot, 
    CompletedDocumentPositionAssociationSnapshot, ImageCompletedDocumentAssociationSnapshot
)
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Revision control database configuration
REVISION_CONTROL_DB_PATH = os.path.join(DATABASE_DIR, 'emtac_revision_control_db.db')
revision_control_engine = create_engine(f'sqlite:///{REVISION_CONTROL_DB_PATH}')
RevisionControlBase = declarative_base()
RevisionControlSession = scoped_session(sessionmaker(bind=revision_control_engine))  # Use distinct name

def get_serializable_data(instance):
    """
    Returns a dictionary of serializable fields from the SQLAlchemy instance.
    Excludes the '_sa_instance_state' and other non-serializable attributes.
    """
    data = instance.__dict__.copy()
    data.pop('_sa_instance_state', None)
    return data

class AuditLog(RevisionControlBase):
    __tablename__ = 'audit_log'

    id = Column(Integer, primary_key=True, autoincrement=True)
    table_name = Column(String, nullable=False)
    operation = Column(String, nullable=False)
    record_id = Column(Integer, nullable=False)
    old_data = Column(JSON, nullable=True)
    new_data = Column(JSON, nullable=True)
    changed_at = Column(DateTime, default=datetime.utcnow, nullable=False)

def log_insert(mapper, connection, target, SnapshotClass=None, session=None):
    # Log the call stack
    stack = traceback.format_stack()
    logger.info("log_insert called by:\n" + "".join(stack))
    
    session = session or RevisionControlSession()  # Ensure a session is always available
    table_name = target.__tablename__
    new_data = {c.name: getattr(target, c.name) for c in target.__table__.columns}

    if table_name == 'complete_document':
        target.rev = 'R0'  # Set initial revision number

    try:
        audit_log_data = {
            'table_name': table_name,
            'operation': 'INSERT',
            'record_id': new_data.get('id'),
            'new_data': new_data
        }
        
        connection.execute(insert(AuditLog).values(audit_log_data))

        if SnapshotClass:
            create_snapshot(target, session, SnapshotClass)

        logger.info(f"Inserted record into audit_log and created snapshot for {table_name} with data: {new_data}")
    except Exception as e:
        logger.error(f"An error occurred during log_insert: {e}")
        session.rollback()
    finally:
        session.close()

def log_update(mapper, connection, target, SnapshotClass=None, session=None):
    # Log the call stack
    stack = traceback.format_stack()
    logger.info("log_update called by:\n" + "".join(stack))
    
    session = session or RevisionControlSession()
    table_name = target.__tablename__
    old_instance = session.query(mapper.class_).get(mapper.primary_key_from_instance(target))
    old_data = get_serializable_data(old_instance)
    new_data = {c.name: getattr(target, c.name) for c in target.__table__.columns}

    if table_name == 'complete_document':
        current_rev = target.rev
        if current_rev:
            rev_number = int(current_rev[1:])  
            target.rev = f'R{rev_number + 1}'

    try:
        audit_log = AuditLog(
            table_name=table_name,
            operation='UPDATE',
            record_id=new_data.get('id'),
            old_data=old_data,
            new_data=new_data
        )
        session.add(audit_log)
        session.commit()
        
        if SnapshotClass:
            create_snapshot(target, session, SnapshotClass)
            
        logger.info(f"Updated record in audit_log and created snapshot for {table_name} with data: {new_data}")
    except Exception as e:
        logger.error(f"An error occurred during log_update: {e}")
        session.rollback()
    finally:
        session.close()

def log_delete(mapper, connection, target, SnapshotClass=None, session=None):
    # Log the call stack
    stack = traceback.format_stack()
    logger.info("log_delete called by:\n" + "".join(stack))
    
    session = session or RevisionControlSession()
    table_name = target.__tablename__
    old_data = {c.name: getattr(target, c.name) for c in target.__table__.columns}

    try:
        audit_log = AuditLog(
            table_name=table_name,
            operation='DELETE',
            record_id=old_data.get('id'),
            old_data=old_data
        )
        session.add(audit_log)
        session.commit()
        
        if SnapshotClass:
            create_snapshot(target, session, SnapshotClass)
            
        logger.info(f"Deleted record from audit_log and created snapshot for {table_name} with data: {old_data}")
    except Exception as e:
        logger.error(f"An error occurred during log_delete: {e}")
        session.rollback()
    finally:
        session.close()

def log_event_listeners(entity_name):
    """
    Logs the setup of event listeners for a given entity.
    """
    logger.info(f"Setting up event listeners for {entity_name}.")