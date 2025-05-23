import os
import sys
import pandas as pd
import numpy as np
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
    SiteLocation,
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
        initializer_logger.info(
            f"Revision control database engine created with URL: sqlite:///{REVISION_CONTROL_DB_PATH}")
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


def clean_dataframe(df, required_columns):
    """
    Clean a DataFrame by removing empty columns and ensuring required columns exist.
    """
    # Remove completely empty columns
    df = df.dropna(axis=1, how='all')

    # Remove columns that are just empty strings
    for col in df.columns:
        if df[col].astype(str).str.strip().eq('').all():
            df = df.drop(columns=[col])

    # Check if all required columns exist
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Replace NaN values with None for database compatibility
    df = df.replace({np.nan: None})

    return df[required_columns]  # Return only required columns in the correct order


def delete_duplicates(session, model, attribute):
    """
    Delete duplicate records in a given model based on a specified attribute.
    Keeps the first occurrence and deletes the rest.
    """
    try:
        # Find duplicate records based on the specified attribute
        duplicates = session.query(getattr(model, attribute), func.count()).group_by(getattr(model, attribute)).having(
            func.count() > 1)

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
        asset_number_data = [(asset.number, asset.model_id, asset.description) for asset in
                             session.query(AssetNumber).all()]
        location_data = [(location.name, location.model_id) for location in session.query(Location).all()]
        site_location_data = [(site.id, site.title, site.room_number, site.site_area) for site in
                              session.query(SiteLocation).all()]

        # Create DataFrames from the extracted data
        df_area = pd.DataFrame(area_data, columns=['name', 'description'])
        df_equipment_group = pd.DataFrame(equipment_group_data, columns=['name', 'area_id'])
        df_model = pd.DataFrame(model_data, columns=['name', 'description', 'equipment_group_id'])
        df_asset_number = pd.DataFrame(asset_number_data, columns=['number', 'model_id', 'description'])
        df_location = pd.DataFrame(location_data, columns=['name', 'model_id'])
        df_site_location = pd.DataFrame(site_location_data, columns=['id', 'title', 'room_number', 'site_area'])

        # Write DataFrames to the Excel file
        with pd.ExcelWriter(excel_file_path) as writer:
            df_area.to_excel(writer, sheet_name='Area', index=False)
            df_equipment_group.to_excel(writer, sheet_name='EquipmentGroup', index=False)
            df_model.to_excel(writer, sheet_name='Model', index=False)
            df_asset_number.to_excel(writer, sheet_name='AssetNumber', index=False)
            df_location.to_excel(writer, sheet_name='Location', index=False)
            df_site_location.to_excel(writer, sheet_name='SiteLocation', index=False)

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
        df_area_raw = pd.read_excel(file_path, sheet_name='Area')
        df_area = clean_dataframe(df_area_raw, ['name', 'description'])
        initializer_logger.info(f"Number of rows in 'Area' DataFrame: {len(df_area)}")
        initializer_logger.info(f"Column names in cleaned 'Area' DataFrame: {list(df_area.columns)}")

        initializer_logger.info("Loading 'EquipmentGroup' DataFrame...")
        df_equipment_group_raw = pd.read_excel(file_path, sheet_name='EquipmentGroup')
        df_equipment_group = clean_dataframe(df_equipment_group_raw, ['name', 'area_id'])
        initializer_logger.info(f"Number of rows in 'EquipmentGroup' DataFrame: {len(df_equipment_group)}")
        initializer_logger.info(
            f"Column names in cleaned 'EquipmentGroup' DataFrame: {list(df_equipment_group.columns)}")

        initializer_logger.info("Loading 'Model' DataFrame...")
        df_model_raw = pd.read_excel(file_path, sheet_name='Model')
        df_model = clean_dataframe(df_model_raw, ['name', 'description', 'equipment_group_id'])
        initializer_logger.info(f"Number of rows in 'Model' DataFrame: {len(df_model)}")
        initializer_logger.info(f"Column names in cleaned 'Model' DataFrame: {list(df_model.columns)}")

        initializer_logger.info("Loading 'AssetNumber' DataFrame...")
        df_asset_number_raw = pd.read_excel(file_path, sheet_name='AssetNumber')
        # Only use the required columns for AssetNumber, ignoring equipment_group_id
        df_asset_number = clean_dataframe(df_asset_number_raw, ['number', 'description', 'model_id'])
        initializer_logger.info(f"Number of rows in 'AssetNumber' DataFrame: {len(df_asset_number)}")
        initializer_logger.info(f"Column names in cleaned 'AssetNumber' DataFrame: {list(df_asset_number.columns)}")

        initializer_logger.info("Loading 'Location' DataFrame...")
        df_location_raw = pd.read_excel(file_path, sheet_name='Location')
        df_location = clean_dataframe(df_location_raw, ['name', 'model_id'])
        initializer_logger.info(f"Number of rows in 'Location' DataFrame: {len(df_location)}")
        initializer_logger.info(f"Column names in cleaned 'Location' DataFrame: {list(df_location.columns)}")

        initializer_logger.info("Loading 'SiteLocation' DataFrame...")
        df_site_location_raw = pd.read_excel(file_path, sheet_name='SiteLocation')
        df_site_location = clean_dataframe(df_site_location_raw, ['id', 'title', 'room_number', 'site_area'])
        initializer_logger.info(f"Number of rows in 'SiteLocation' DataFrame: {len(df_site_location)}")
        initializer_logger.info(f"Column names in cleaned 'SiteLocation' DataFrame: {list(df_site_location.columns)}")

        # Create session
        session = Session()

        try:
            # Backup the database before making any changes
            backup_database_relationships(session)

            # Insert or update data into 'Area' table
            initializer_logger.info("Processing 'Area' data...")
            for _, row in df_area.iterrows():
                area_name = str(row['name']).strip() if pd.notna(row['name']) else ''
                if not area_name:
                    continue
                area = session.query(Area).filter_by(name=area_name).first()
                if area:
                    area.description = row['description'] if pd.notna(row['description']) else None
                else:
                    area = Area(name=area_name,
                                description=row['description'] if pd.notna(row['description']) else None)
                    session.add(area)

            # Insert or update data into 'EquipmentGroup' table
            initializer_logger.info("Processing 'EquipmentGroup' data...")
            for _, row in df_equipment_group.iterrows():
                area_id = row.get('area_id', None)
                if pd.notna(area_id):
                    area_id = int(area_id)
                else:
                    area_id = None
                equipment_group_name = str(row['name']).strip() if pd.notna(row['name']) else ''
                if not equipment_group_name:
                    continue
                equipment_group = session.query(EquipmentGroup).filter_by(name=equipment_group_name).first()
                if equipment_group:
                    equipment_group.area_id = area_id
                else:
                    equipment_group = EquipmentGroup(name=equipment_group_name, area_id=area_id)
                    session.add(equipment_group)

            # Insert or update data into 'Model' table
            initializer_logger.info("Processing 'Model' data...")
            for _, row in df_model.iterrows():
                equipment_group_id = row.get('equipment_group_id', None)
                if pd.notna(equipment_group_id):
                    equipment_group_id = int(equipment_group_id)
                else:
                    equipment_group_id = None
                model_name = str(row['name']).strip() if pd.notna(row['name']) else ''
                if not model_name:
                    continue
                model = session.query(Model).filter_by(name=model_name).first()
                if model:
                    model.description = row['description'] if pd.notna(row['description']) else None
                    model.equipment_group_id = equipment_group_id
                else:
                    equipment_group = None
                    if equipment_group_id:
                        equipment_group = session.query(EquipmentGroup).filter_by(id=equipment_group_id).first()
                    model = Model(name=model_name,
                                  description=row['description'] if pd.notna(row['description']) else None,
                                  equipment_group=equipment_group)
                    session.add(model)

            # Insert or update data into 'AssetNumber' table (ignoring equipment_group_id column)
            initializer_logger.info("Processing 'AssetNumber' data...")
            for _, row in df_asset_number.iterrows():
                model_id = row.get('model_id', None)
                if pd.notna(model_id):
                    model_id = int(model_id)
                else:
                    model_id = None
                asset_number_name = str(row['number']).strip() if pd.notna(row['number']) else ''
                if not asset_number_name:
                    continue
                asset_number = session.query(AssetNumber).filter_by(number=asset_number_name).first()
                if asset_number:
                    asset_number.model_id = model_id
                    asset_number.description = row['description'] if pd.notna(row['description']) else None
                else:
                    asset_number = AssetNumber(number=asset_number_name, model_id=model_id,
                                               description=row['description'] if pd.notna(row['description']) else None)
                    session.add(asset_number)

            # Insert or update data into 'Location' table
            initializer_logger.info("Processing 'Location' data...")
            for index, row in df_location.iterrows():
                if index % 10 == 0:  # Log progress every 10 rows
                    initializer_logger.info(f"Processing location row: {index + 1}")
                model_id = row.get('model_id', None)
                if pd.notna(model_id):
                    model_id = int(model_id)
                else:
                    model_id = None
                location_name = str(row['name']).strip() if pd.notna(row['name']) else ''
                if not location_name:
                    continue
                location = session.query(Location).filter_by(name=location_name).first()
                if location:
                    location.model_id = model_id
                else:
                    location = Location(name=location_name, model_id=model_id)
                    session.add(location)

            # Insert or update data into 'SiteLocation' table
            initializer_logger.info("Processing 'SiteLocation' data...")
            for index, row in df_site_location.iterrows():
                if index % 100 == 0:  # Log progress every 100 rows
                    initializer_logger.info(f"Processing site location row: {index + 1}")

                site_location_id = row.get('id', None)
                if pd.notna(site_location_id):
                    site_location_id = int(site_location_id)
                else:
                    site_location_id = None

                title = str(row['title']).strip() if pd.notna(row['title']) else ''
                room_number = str(row['room_number']).strip() if pd.notna(row['room_number']) else ''
                site_area = str(row['site_area']).strip() if pd.notna(row['site_area']) else ''

                if not title:
                    continue

                # Check if site location already exists by ID first, then by title
                site_location = None
                if site_location_id:
                    site_location = session.query(SiteLocation).filter_by(id=site_location_id).first()

                if not site_location:
                    site_location = session.query(SiteLocation).filter_by(title=title).first()

                if site_location:
                    # Update existing site location
                    site_location.title = title
                    site_location.room_number = room_number
                    site_location.site_area = site_area
                else:
                    # Create new site location
                    site_location = SiteLocation(
                        id=site_location_id,  # This will be None if not provided, allowing auto-increment
                        title=title,
                        room_number=room_number,
                        site_area=site_area
                    )
                    session.add(site_location)

            # Delete duplicates
            initializer_logger.info("Removing duplicates...")
            delete_duplicates(session, Area, 'name')
            delete_duplicates(session, EquipmentGroup, 'name')
            delete_duplicates(session, Model, 'name')
            delete_duplicates(session, AssetNumber, 'number')
            delete_duplicates(session, Location, 'name')
            delete_duplicates(session, SiteLocation, 'title')

            # Commit the session
            session.commit()
            initializer_logger.info("Data uploaded successfully!")

            # Add version info and create snapshots
            try:
                rev_session = RevisionControlSession()
                new_version = VersionInfo(version_number=1, description="Initial version with updated column structure")
                rev_session.add(new_version)
                rev_session.commit()

                # Create snapshots for all entities
                initializer_logger.info("Creating snapshots...")
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

                # Create SiteLocation snapshots if SiteLocationSnapshot class exists
                try:
                    from modules.emtacdb.emtac_revision_control_db import SiteLocationSnapshot
                    for site_location in session.query(SiteLocation).all():
                        create_snapshot(site_location, rev_session, SiteLocationSnapshot)
                    initializer_logger.info("SiteLocation snapshots created successfully!")
                except ImportError:
                    initializer_logger.warning("SiteLocationSnapshot class not found - skipping SiteLocation snapshots")
                except Exception as e:
                    initializer_logger.error(f"Error creating SiteLocation snapshots: {e}")

                rev_session.commit()
                initializer_logger.info("Snapshots created successfully!")
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

        # Check if Excel file exists
        if not os.path.exists(excel_file_path):
            initializer_logger.error(f"Excel file not found: {excel_file_path}")
            raise FileNotFoundError(f"Excel file not found: {excel_file_path}")

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