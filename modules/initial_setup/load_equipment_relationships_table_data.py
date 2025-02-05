import os
import sys
import pandas as pd
from sqlalchemy import create_engine, func, inspect
from sqlalchemy.orm import sessionmaker, scoped_session
from datetime import datetime
from sqlalchemy.ext.declarative import declarative_base

from modules.configuration.config import (
    BASE_DIR,
    DATABASE_URL,
    DB_LOADSHEET,
    REVISION_CONTROL_DB_PATH
)
from modules.emtacdb.emtacdb_fts import (
    Area,
    EquipmentGroup,
    Model,
    AssetNumber,
    Location,
    Base
)
from modules.emtacdb.emtac_revision_control_db import (
    VersionInfo,
    revision_control_engine,
    RevisionControlSession,
    AreaSnapshot,
    EquipmentGroupSnapshot,
    ModelSnapshot,
    AssetNumberSnapshot,
    LocationSnapshot
)
from modules.emtacdb.utlity.revision_database.snapshot_utils import (
    create_snapshot
)
from modules.initial_setup.initializer_logger import (
    initializer_logger,
    close_initializer_logger
)

# Define the base for the revision control database models
RevisionControlBase = declarative_base()

def setup_logging():
    """
    Ensure the directory for the log file exists and perform any additional logging setup if necessary.
    """
    log_directory = os.path.join(BASE_DIR, "logs")
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    # Additional logging configurations can be added here if needed
    initializer_logger.info("Logging setup complete.")

def create_main_database_engine():
    """
    Create and return the SQLAlchemy engine and session for the main database.
    """
    try:
        engine = create_engine(DATABASE_URL, echo=True)  # Enable SQL logging
        Session = scoped_session(sessionmaker(bind=engine))
        initializer_logger.info(f"Main database engine created with URL: {DATABASE_URL}")
        return engine, Session
    except Exception as e:
        initializer_logger.error(f"Failed to create main database engine: {e}")
        raise

def create_revision_control_engine():
    """
    Create and return the SQLAlchemy engine and session for the revision control database.
    """
    try:
        revision_engine = create_engine(f'sqlite:///{REVISION_CONTROL_DB_PATH}', echo=True)  # Enable SQL logging
        RevisionControlSessionLocal = scoped_session(sessionmaker(bind=revision_engine))
        initializer_logger.info(f"Revision control database engine created with URL: sqlite:///{REVISION_CONTROL_DB_PATH}")
        return revision_engine, RevisionControlSessionLocal
    except Exception as e:
        initializer_logger.error(f"Failed to create revision control database engine: {e}")
        raise

def create_tables(engine, base, db_type="Main"):
    """
    Create tables in the specified database.
    """
    try:
        base.metadata.create_all(engine)
        initializer_logger.info(f"{db_type} database tables created (if not exist).")
    except Exception as e:
        initializer_logger.error(f"Failed to create {db_type} database tables: {e}")
        raise

def inspect_database(engine, db_type="Main"):
    """
    Inspect the specified database and log the table names.
    """
    try:
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        initializer_logger.info(f"Tables in the {db_type} database: {table_names}")
        if db_type == "Revision Control" and 'version_info' not in table_names:
            initializer_logger.error("version_info table does not exist in the revision control database.")
    except Exception as e:
        initializer_logger.error(f"Failed to inspect {db_type} database: {e}")
        raise

def delete_duplicates(session, model, attribute):
    """
    Delete duplicate records in a given model based on a specified attribute.
    Keeps the first occurrence and deletes the rest.
    """
    try:
        # Find duplicate records based on the specified attribute
        duplicates = session.query(getattr(model, attribute), func.count()).group_by(getattr(model, attribute)).having(func.count() > 1)

        # Iterate over duplicate records and keep one instance while deleting the rest
        for attr_value, count in duplicates:
            records = session.query(model).filter(getattr(model, attribute) == attr_value).all()
            for record in records[1:]:  # Keep the first instance, delete the rest
                session.delete(record)
        initializer_logger.info(f"Deleted duplicates for model {model.__name__} based on attribute '{attribute}'.")
    except Exception as e:
        initializer_logger.error(f"Failed to delete duplicates for model {model.__name__}: {e}")

def backup_database_relationships(session):
    """
    Create a backup of the database by exporting tables to an Excel file.
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

        initializer_logger.info(f"Database backup created successfully: {excel_file_name}")
    except Exception as e:
        initializer_logger.error(f"Error creating database backup: {e}")

def upload_data_from_excel(file_path, engine, Session):
    """
    Load data from an Excel file and insert/update it into the main database.
    Also handles versioning and snapshot creation in the revision control database.
    """
    # Load Excel file into pandas DataFrame
    try:
        initializer_logger.info("Loading 'Area' DataFrame...")
        df_area = pd.read_excel(file_path, sheet_name='Area')
        initializer_logger.info(f"Number of rows in 'Area' DataFrame: {len(df_area)}")
        initializer_logger.info(f"Number of columns in 'Area' DataFrame: {len(df_area.columns)}")

        initializer_logger.info("Loading 'EquipmentGroup' DataFrame...")
        df_equipment_group = pd.read_excel(file_path, sheet_name='EquipmentGroup')
        initializer_logger.info(f"Number of rows in 'EquipmentGroup' DataFrame: {len(df_equipment_group)}")
        initializer_logger.info(f"Number of columns in 'EquipmentGroup' DataFrame: {len(df_equipment_group.columns)}")

        initializer_logger.info("Loading 'Model' DataFrame...")
        df_model = pd.read_excel(file_path, sheet_name='Model')
        initializer_logger.info(f"Number of rows in 'Model' DataFrame: {len(df_model)}")
        initializer_logger.info(f"Number of columns in 'Model' DataFrame: {len(df_model.columns)}")
        initializer_logger.info(f"Column names in 'Model' DataFrame: {df_model.columns}")

        initializer_logger.info("Loading 'AssetNumber' DataFrame...")
        df_asset_number = pd.read_excel(file_path, sheet_name='AssetNumber')
        initializer_logger.info(f"Number of rows in 'AssetNumber' DataFrame: {len(df_asset_number)}")
        initializer_logger.info(f"Number of columns in 'AssetNumber' DataFrame: {len(df_asset_number.columns)}")

        initializer_logger.info("Loading 'Location' DataFrame...")
        df_location = pd.read_excel(file_path, sheet_name='Location')
        initializer_logger.info(f"Number of rows in 'Location' DataFrame: {len(df_location)}")
        initializer_logger.info(f"Number of columns in 'Location' DataFrame: {len(df_location.columns)}")
        initializer_logger.info(f"Column names in 'Location' DataFrame: {df_location.columns}")

        # Create session
        session = Session()

        try:
            # Backup the database before making any changes
            backup_database_relationships(session)

            # Insert or update data into 'Area' table
            for _, row in df_area.iterrows():
                area_name = row['name'].strip()
                area = session.query(Area).filter_by(name=area_name).first()
                if area:
                    area.description = row['description']
                else:
                    area = Area(name=area_name, description=row['description'])
                    session.add(area)

            # Insert or update data into 'EquipmentGroup' table
            for _, row in df_equipment_group.iterrows():
                area_id = row.get('area_id', None)
                equipment_group_name = row['name'].strip()
                equipment_group = session.query(EquipmentGroup).filter_by(name=equipment_group_name).first()
                if equipment_group:
                    equipment_group.area_id = area_id
                else:
                    equipment_group = EquipmentGroup(name=equipment_group_name, area_id=area_id)
                    session.add(equipment_group)

            # Insert or update data into 'Model' table
            for _, row in df_model.iterrows():
                equipment_group_id = row.get('equipment_group_id', None)
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
                model_id = row.get('model_id', None)
                asset_number_name = row['number'].strip()
                asset_number = session.query(AssetNumber).filter_by(number=asset_number_name).first()
                if asset_number:
                    asset_number.model_id = model_id
                    asset_number.description = row['description']
                else:
                    asset_number = AssetNumber(number=asset_number_name, model_id=model_id, description=row['description'])
                    session.add(asset_number)

            # Insert or update data into 'Location' table
            initializer_logger.info("Inserting into 'Location' table...")
            for index, row in df_location.iterrows():
                initializer_logger.info(f"Processing row: {index + 1}")
                model_id = row.get('model_id', None)
                location_name = row['name'].strip()
                initializer_logger.info(f"Location name: {location_name}")
                location = session.query(Location).filter_by(name=location_name).first()
                if location:
                    location.model_id = model_id
                else:
                    location = Location(name=location_name, model_id=model_id)
                    session.add(location)

            # Delete duplicates
            delete_duplicates(session, Area, 'name')
            delete_duplicates(session, EquipmentGroup, 'name')
            delete_duplicates(session, Model, 'name')
            delete_duplicates(session, AssetNumber, 'number')
            delete_duplicates(session, Location, 'name')

            # Commit the session
            session.commit()
            initializer_logger.info("Data uploaded successfully!")

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
                initializer_logger.error(f"An error occurred while adding version info and creating snapshots: {e}")
                rev_session.rollback()
            finally:
                rev_session.close()

            # Querying the version_info table
            try:
                revision_session = RevisionControlSession()
                initializer_logger.info("Querying 'version_info' table in revision control database.")
                version_info = revision_session.query(VersionInfo).order_by(VersionInfo.id.desc()).first()
                if version_info:
                    initializer_logger.info(f"Latest version_info: {version_info.version_number}")
                else:
                    initializer_logger.warning("No version_info found.")
            except Exception as e:
                initializer_logger.error(f"Error querying 'version_info' table: {e}")
            finally:
                revision_session.close()

        except Exception as e:
            initializer_logger.error(f"An error occurred during data upload: {e}")
            session.rollback()
        finally:
            session.close()

    except Exception as e:
        initializer_logger.error(f"An unexpected error occurred: {e}")
        raise

def main():
    """
    Main entry point for the script.
    """
    try:
        setup_logging()

        # Create engines and sessions
        main_engine, MainSession = create_main_database_engine()
        revision_engine, RevisionControlSessionLocal = create_revision_control_engine()

        # Create tables
        create_tables(main_engine, Base, db_type="Main")
        create_tables(revision_engine, RevisionControlBase, db_type="Revision Control")

        # Inspect databases
        inspect_database(main_engine, db_type="Main")
        inspect_database(revision_engine, db_type="Revision Control")

        # Define the Excel file path
        excel_file_path = os.path.join(DB_LOADSHEET, "load_equipment_relationships_table_data.xlsx")

        # Upload data from Excel
        upload_data_from_excel(excel_file_path, main_engine, MainSession)

    except Exception as e:
        initializer_logger.error(f"An error occurred in the main execution flow: {e}")
    finally:
        # Close the logger
        close_initializer_logger()
        initializer_logger.info("Program execution completed successfully.")

if __name__ == "__main__":
    main()
