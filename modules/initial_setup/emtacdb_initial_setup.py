import os
import sys
import subprocess
from sqlalchemy import create_engine, inspect
from modules.configuration.config import (
    DATABASE_URL, REVISION_CONTROL_DB_PATH,
    TEMPLATE_FOLDER_PATH, DATABASE_DIR, UPLOAD_FOLDER,
    IMAGES_FOLDER, DATABASE_PATH_IMAGES_FOLDER,
    PDF_FOR_EXTRACTION_FOLDER, IMAGES_EXTRACTED,
    TEMPORARY_FILES, PPT2PDF_PPT_FILES_PROCESS,
    PPT2PDF_PDF_FILES_PROCESS, DATABASE_DOC,
    TEMPORARY_UPLOAD_FILES, DB_LOADSHEET,
    DB_LOADSHEETS_BACKUP, DB_LOADSHEET_BOMS,
    BACKUP_DIR, Utility_tools, UTILITIES
)
from modules.initial_setup.initializer_logger import (
    initializer_logger, compress_logs_except_most_recent, close_initializer_logger, LOG_DIRECTORY
)
from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import Base as MainBase
from modules.emtacdb.emtac_revision_control_db import RevisionControlBase

# List of directories to check and create
directories_to_check = [
    TEMPLATE_FOLDER_PATH,
    DATABASE_DIR,
    UPLOAD_FOLDER,
    IMAGES_FOLDER,
    DATABASE_PATH_IMAGES_FOLDER,
    PDF_FOR_EXTRACTION_FOLDER,
    IMAGES_EXTRACTED,
    TEMPORARY_FILES,
    PPT2PDF_PPT_FILES_PROCESS,
    PPT2PDF_PDF_FILES_PROCESS,
    DATABASE_DOC,
    TEMPORARY_UPLOAD_FILES,
    DB_LOADSHEET,
    DB_LOADSHEETS_BACKUP,
    DB_LOADSHEET_BOMS,
    BACKUP_DIR,
    Utility_tools,
    UTILITIES
]
def check_and_install_requirements():
    """
    Checks if `requirements.txt` exists and installs missing dependencies.
    """
    requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")

    if os.path.isfile(requirements_file):
        initializer_logger.info("📦 Checking and installing dependencies from requirements.txt...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_file], check=True)
            initializer_logger.info("✅ All dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            initializer_logger.error(f"❌ Failed to install dependencies: {e}")
            sys.exit(1)
    else:
        initializer_logger.warning("⚠️ No requirements.txt file found. Skipping dependency installation.")

def create_directories():
    """
    Ensures all required directories exist, creating them if necessary.
    """
    initializer_logger.info("Checking and creating required directories...")

    for directory in directories_to_check:
        if not os.path.exists(directory):
            os.makedirs(directory)
            initializer_logger.info(f"✅ Created directory: {directory}")
        else:
            initializer_logger.info(f"✔️ Directory already exists: {directory}")

def check_and_create_database():
    """
    Ensures that the main and revision control databases exist and have all required tables.
    """
    initializer_logger.info("Checking if databases and tables exist...")

    db_config = DatabaseConfig()
    main_engine = db_config.main_engine
    revision_engine = db_config.revision_control_engine

    try:
        # Check if main database tables exist
        main_inspector = inspect(main_engine)
        main_tables = main_inspector.get_table_names()
        if not main_tables:
            initializer_logger.warning("⚠️ No tables found in the main database. Creating tables...")
            MainBase.metadata.create_all(main_engine)
            initializer_logger.info("✅ Main database tables created successfully.")
        else:
            initializer_logger.info(f"✔️ Main database is ready with tables: {main_tables}")

        # Check if revision control database tables exist
        revision_inspector = inspect(revision_engine)
        revision_tables = revision_inspector.get_table_names()
        if not revision_tables:
            initializer_logger.warning("⚠️ No tables found in the revision control database. Creating tables...")
            RevisionControlBase.metadata.create_all(revision_engine)
            initializer_logger.info("✅ Revision control database tables created successfully.")
        else:
            initializer_logger.info(f"✔️ Revision control database is ready with tables: {revision_tables}")

    except Exception as e:
        initializer_logger.error(f"❌ Database setup failed: {e}", exc_info=True)
        sys.exit(1)

def run_setup_scripts():
    """
    Runs each of the setup scripts in sequence.
    """
    scripts_to_run = [
        "load_equipment_relationships_table_data.py",
        "initial_admin.py",
        "load_parts_sheet.py",
        "load_active_drawing_list.py",
        "load_image_folder.py",
        "load_bom_loadsheet.py",
    ]

    this_dir = os.path.dirname(os.path.abspath(__file__))

    for script in scripts_to_run:
        script_path = os.path.join(this_dir, script)

        print(f"\n--- Now running: {script} ---\n")
        try:
            subprocess.run([sys.executable, script_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR: Script {script} failed with: {e}")
            initializer_logger.error(f"❌ Script {script} failed with: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print(f"❌ ERROR: Could not find the script {script_path}")
            initializer_logger.error(f"❌ Could not find the script {script_path}")
            sys.exit(1)

    print("\n✅ All setup scripts have run successfully!")
    initializer_logger.info("✅ All setup scripts have run successfully!")

    logs_directory = os.path.join(this_dir, "logs")
    if os.path.exists(logs_directory):
        initializer_logger.info("🗜️ Compressing old initializer logs...")
        compress_logs_except_most_recent(logs_directory)
        initializer_logger.info("✔️ Log compression completed.")
    else:
        initializer_logger.warning(f"⚠️ No logs directory found at {logs_directory}.")

def main():
    # Ensure all required directories exist before running the database setup
    create_directories()

    # Ensure dependencies are installed
    check_and_install_requirements()

    # Ensure database and tables are ready
    check_and_create_database()

    # Run all setup scripts
    run_setup_scripts()

    # Compress logs
    compress_logs_except_most_recent(LOG_DIRECTORY)

    initializer_logger.info("✅ All setup scripts completed successfully. Exiting now.")
    close_initializer_logger()

if __name__ == "__main__":
    main()