import os
import pandas as pd
from sqlalchemy import create_engine, func, inspect
from sqlalchemy.orm import sessionmaker, scoped_session
import sys
from datetime import datetime
from config import BASE_DIR, DATABASE_URL, DB_LOADSHEET, DB_LOADSHEETS_BACKUP, REVISION_CONTROL_DB_PATH
from emtacdb_fts import Area, EquipmentGroup, Model, AssetNumber, Location, Base
from auditlog import AuditLog, log_delete, log_insert, log_update
from emtac_revision_control_db import (
    VersionInfo, revision_control_engine, RevisionControlSession, SiteLocationSnapshot, PositionSnapshot, AreaSnapshot, EquipmentGroupSnapshot, ModelSnapshot,
    AssetNumberSnapshot, PartSnapshot, ImageSnapshot, ImageEmbeddingSnapshot, DrawingSnapshot, LocationSnapshot,
    DocumentSnapshot, CompleteDocumentSnapshot, ProblemSnapshot, SolutionSnapshot,
    DrawingPartAssociationSnapshot, PartProblemAssociationSnapshot, PartSolutionAssociationSnapshot,
    PartsPositionAssociationSnapshot, DrawingProblemAssociationSnapshot, DrawingSolutionAssociationSnapshot,
    ProblemPositionAssociationSnapshot, CompleteDocumentProblemAssociationSnapshot,
    CompleteDocumentSolutionAssociationSnapshot, ImageProblemAssociationSnapshot,
    ImageSolutionAssociationSnapshot, ImagePositionAssociationSnapshot, DrawingPositionAssociationSnapshot,
    CompletedDocumentPositionAssociationSnapshot, ImageCompletedDocumentAssociationSnapshot
)
from snapshot_utils import (
    get_latest_version_info, add_version_info, create_sitlocation_snapshot, create_position_snapshot, create_area_snapshot, create_equipment_group_snapshot,
    create_model_snapshot, create_asset_number_snapshot, create_part_snapshot, create_image_snapshot,
    create_image_embedding_snapshot, create_drawing_snapshot, create_document_snapshot,
    create_complete_document_snapshot, create_problem_snapshot, create_solution_snapshot,
    create_drawing_part_association_snapshot, create_part_problem_association_snapshot,
    create_part_solution_association_snapshot, create_drawing_problem_association_snapshot,
    create_drawing_solution_association_snapshot, create_problem_position_association_snapshot,
    create_complete_document_problem_association_snapshot, create_complete_document_solution_association_snapshot,
    create_image_problem_association_snapshot, create_image_solution_association_snapshot,
    create_image_position_association_snapshot, create_drawing_position_association_snapshot,
    create_completed_document_position_association_snapshot, create_image_completed_document_association_snapshot,
    create_parts_position_association_snapshot
)

# Initialize logging
import logging

# Ensure the directory for the log file exists
log_directory = os.path.join(BASE_DIR, "logs")
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_file_path = os.path.join(log_directory, "script.log")

logging.basicConfig(level=logging.DEBUG,  # Set to DEBUG to capture all log levels
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file_path),
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger(__name__)

# Test logging
logger.info("Logging is configured correctly and this is a test log entry.")
logger.debug(f"Log file path: {log_file_path}")

# Create SQLAlchemy engine for main database
try:
    engine = create_engine(DATABASE_URL, echo=True)  # Enable SQL logging
    Session = scoped_session(sessionmaker(bind=engine))
    logger.info(f"Main database engine created with URL: {DATABASE_URL}")
except Exception as e:
    logger.error(f"Failed to create main database engine: {e}")

# Ensure tables exist in the main database
try:
    Base.metadata.create_all(engine)
    logger.info("Main database tables created (if not exist).")
except Exception as e:
    logger.error(f"Failed to create main database tables: {e}")

# Inspect the main database and log table names
try:
    main_inspector = inspect(engine)
    main_table_names = main_inspector.get_table_names()
    logger.info(f"Tables in the main database: {main_table_names}")
except Exception as e:
    logger.error(f"Failed to inspect main database: {e}")

# Create SQLAlchemy engine for revision control session
try:
    revision_control_engine = create_engine(f'sqlite:///{REVISION_CONTROL_DB_PATH}', echo=True)  # Enable SQL logging
    RevisionControlSession = scoped_session(sessionmaker(bind=revision_control_engine))
    logger.info(f"Revision control database engine created with URL: sqlite:///{REVISION_CONTROL_DB_PATH}")
except Exception as e:
    logger.error(f"Failed to create revision control database engine: {e}")

# Ensure tables exist in the revision control database
try:
    RevisionControlBase.metadata.create_all(revision_control_engine)
    logger.info("Revision control database tables created (if not exist).")
except Exception as e:
    logger.error(f"Failed to create revision control database tables: {e}")

# Inspect the revision control engine and log table names
try:
    inspector = inspect(revision_control_engine)
    table_names = inspector.get_table_names()
    logger.info(f"Tables in the revision control database: {table_names}")
    if 'version_info' not in table_names:
        logger.error("version_info table does not exist in the revision control database.")
except Exception as e:
    logger.error(f"Failed to inspect revision control database: {e}")

def delete_duplicates(session, model, attribute):
    try:
        # Find duplicate records based on the specified attribute
        duplicates = session.query(getattr(model, attribute), func.count()).group_by(getattr(model, attribute)).having(func.count() > 1)
        
        # Iterate over duplicate records and keep one instance while deleting the rest
        for attr_value, count in duplicates:
            records = session.query(model).filter(getattr(model, attribute) == attr_value).all()
            for record in records[1:]:  # Keep the first instance, delete the rest
                session.delete(record)
        logger.info(f"Deleted duplicates for model {model.__name__} based on attribute {attribute}")
    except Exception as e:
        logger.error(f"Failed to delete duplicates for model {model.__name__}: {e}")

def backup_database_relationships(session):
    """
    Function to create a backup of the database.
    """
    try:
        # Define the directory to store backup Excel files
        backup_directory = os.path.join(BASE_DIR, "Database", "DB_LOADSHEETS_BACKUP")
        
        # Create the backup directory if it doesn't exist
        if not os.path.exists(backup_directory):
            os.makedirs(backup_directory)

        # Get the current date and time for the timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create the Excel file name with the timestamp
        excel_file_name = f"equipment_relationships_table_data_database_backup_{timestamp}.xlsx"
        excel_file_path = os.path.join(backup_directory, excel_file_name)

        # Extract data from each table and create DataFrames
        area_data = [(area.name, area.description) for area in session.query(Area).all()]
        equipment_group_data = [(group.name, group.area_id) for group in session.query(EquipmentGroup).all()]
        model_data = [(model.name, model.description, model.equipment_group_id) for model in session.query(Model).all()]
        asset_number_data = [(asset.number, asset.model_id, asset.description) for asset in session.query(AssetNumber).all()]
        location_data = [(location.name, location.model_id) for location in session.query(Location).all()]

        # Create DataFrames from the extracted data
        df_area = pd.DataFrame(area_data, columns=['name', 'description'])
        df_equipment_group = pd.DataFrame(equipment_group_data, columns=['name', 'area_id'])
        df_model = pd.DataFrame(model_data, columns=['name', 'description', 'equipment_group_id'])
        df_asset_number = pd.DataFrame(asset_number_data, columns=['number', 'model_id', 'description'])
        df_location = pd.DataFrame(location_data, columns=['name', 'model_id'])

        # Write DataFrames to the Excel file
        with pd.ExcelWriter(excel_file_path) as writer:
            df_area.to_excel(writer, sheet_name='Area', index=False)
            df_equipment_group.to_excel(writer, sheet_name='EquipmentGroup', index=False)
            df_model.to_excel(writer, sheet_name='Model', index=False)
            df_asset_number.to_excel(writer, sheet_name='AssetNumber', index=False)
            df_location.to_excel(writer, sheet_name='Location', index=False)

        logger.info(f"Database backup created successfully: {excel_file_name}")
    except Exception as e:
        logger.error(f"Error creating database backup: {e}")

def upload_data_from_excel(file_path, engine):
    # Load Excel file into pandas DataFrame
    logger.info("Loading 'Area' DataFrame...")
    df_area = pd.read_excel(file_path, sheet_name='Area')
    logger.info(f"Number of rows in 'Area' DataFrame: {len(df_area)}")
    logger.info(f"Number of columns in 'Area' DataFrame: {len(df_area.columns)}")

    logger.info("Loading 'EquipmentGroup' DataFrame...")
    df_equipment_group = pd.read_excel(file_path, sheet_name='EquipmentGroup')
    logger.info(f"Number of rows in 'EquipmentGroup' DataFrame: {len(df_equipment_group)}")
    logger.info(f"Number of columns in 'EquipmentGroup' DataFrame: {len(df_equipment_group.columns)}")

    logger.info("Loading 'Model' DataFrame...")
    df_model = pd.read_excel(file_path, sheet_name='Model')
    logger.info(f"Number of rows in 'Model' DataFrame: {len(df_model)}")
    logger.info(f"Number of columns in 'Model' DataFrame: {len(df_model.columns)}")
    logger.info(f"Column names in 'Model' DataFrame: {df_model.columns}")

    logger.info("Loading 'AssetNumber' DataFrame...")
    df_asset_number = pd.read_excel(file_path, sheet_name='AssetNumber')
    logger.info(f"Number of rows in 'AssetNumber' DataFrame: {len(df_asset_number)}")
    logger.info(f"Number of columns in 'AssetNumber' DataFrame: {len(df_asset_number.columns)}")

    logger.info("Loading 'Location' DataFrame...")
    df_location = pd.read_excel(file_path, sheet_name='Location')
    logger.info(f"Number of rows in 'Location' DataFrame: {len(df_location)}")
    logger.info(f"Number of columns in 'Location' DataFrame: {len(df_location.columns)}")
    logger.info(f"Column names in 'Location' DataFrame: {df_location.columns}")

    # Create session
    session = Session()

    try:
        # Backup the database before making any changes
        backup_database_relationships(session)
        
        # Insert or update data into 'Area' table
        for _, row in df_area.iterrows():
            # Strip leading and trailing spaces from the area name
            area_name = row['name'].strip()
            area = session.query(Area).filter_by(name=area_name).first()
            if area:
                area.description = row['description']
            else:
                area = Area(name=area_name, description=row['description'])
                session.add(area)

        # Insert or update data into 'EquipmentGroup' table
        for _, row in df_equipment_group.iterrows():
            if 'area_id' in df_equipment_group.columns:
                area_id = row['area_id']
            else:
                area_id = None
            # Strip leading and trailing spaces from the equipment group name
            equipment_group_name = row['name'].strip()
            equipment_group = session.query(EquipmentGroup).filter_by(name=equipment_group_name).first()
            if equipment_group:
                equipment_group.area_id = area_id
            else:
                equipment_group = EquipmentGroup(name=equipment_group_name, area_id=area_id)
                session.add(equipment_group)

        # Insert or update data into 'Model' table
        for _, row in df_model.iterrows():
            if 'equipment_group_id' in df_model.columns:
                equipment_group_id = row['equipment_group_id']
            else:
                equipment_group_id = None
            # Strip leading and trailing spaces from the model name
            model_name = row['name'].strip()
            model = session.query(Model).filter_by(name=model_name).first()
            if model:
                model.description = row['description']
                model.equipment_group_id = equipment_group_id
            else:
                equipment_group = session.query(EquipmentGroup).filter_by(id=equipment_group_id).first()
                model = Model(name=model_name, description=row['description'], equipment_group=equipment_group)
                session.add(model)
                
        # Insert or update data into 'AssetNumber' table
        for _, row in df_asset_number.iterrows():
            if 'model_id' in df_asset_number.columns:
                model_id = row['model_id']
            else:
                model_id = None
            # Strip leading and trailing spaces from the asset number
            asset_number_name = row['number'].strip()
            asset_number = session.query(AssetNumber).filter_by(number=asset_number_name).first()
            if asset_number:
                asset_number.model_id = model_id
                asset_number.description = row['description']
            else:
                asset_number = AssetNumber(number=asset_number_name, model_id=model_id, description=row['description'])
                session.add(asset_number)

        # Insert or update data into 'Location' table
        logger.info(f'inserting into location table')
        for index, row in df_location.iterrows():
            logger.info(f"Processing row: {index}")
            if 'model_id' in df_location.columns:
                model_id = row['model_id']
            else:
                model_id = None
            # Strip leading and trailing spaces from the location name
            location_name = row['name'].strip()
            logger.info(f"Location name: {location_name}")
            location = session.query(Location).filter_by(name=location_name).first()
            if location:
                location.model_id = model_id
            else:
                location = Location(name=location_name, model_id=model_id)
                session.add(location)

        delete_duplicates(session, Area, 'name')
        delete_duplicates(session, EquipmentGroup, 'name')
        delete_duplicates(session, Model, 'name')
        delete_duplicates(session, AssetNumber, 'number')
        delete_duplicates(session, Location, 'name')

        # Commit the session
        session.commit()
        logger.info("Data uploaded successfully!")

        # Add version info and create snapshots
        try:
            rev_session = RevisionControlSession()
            new_version = VersionInfo(version_number=1, description="Initial version")
            rev_session.add(new_version)
            rev_session.commit()

            for area in session.query(Area).all():
                create_snapshot(area, rev_session, AreaSnapshot)
            for equipment_group in session.query(EquipmentGroup).all():
                create_snapshot(equipment_group, rev_session, EquipmentGroupSnapshot)
            for model in session.query(Model).all():
                create_snapshot(model, rev_session, ModelSnapshot)
            for asset_number in session.query(AssetNumber).all():
                create_snapshot(asset_number, rev_session, AssetNumberSnapshot)
            for location in session.query(Location).all():
                create_snapshot(location, rev_session, LocationSnapshot)

            rev_session.commit()
        except Exception as e:
            logger.error(f"An error occurred while adding version info and creating snapshots: {e}")
            rev_session.rollback()
        finally:
            rev_session.close()

        # Querying the version_info table
        try:
            revision_session = RevisionControlSession()
            logger.info("Querying version_info table in revision control database.")
            version_info = revision_session.query(VersionInfo).order_by(VersionInfo.id.desc()).first()
            if version_info:
                logger.info(f"Latest version_info: {version_info.version_number}")
            else:
                logger.warning("No version_info found.")
        except Exception as e:
            logger.error(f"Error querying version_info table: {e}")
        finally:
            revision_session.close()

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    # Debugging print statement to verify DB_LOADSHEET value
    logger.info(f"DB_LOADSHEET: {DB_LOADSHEET}")
    
    # Provide the new name for your Excel file
    excel_file_path = os.path.join(DB_LOADSHEET, "load_equipment_relationships_table_data.xlsx")

    # Call the function to upload data
    upload_data_from_excel(excel_file_path, engine)
