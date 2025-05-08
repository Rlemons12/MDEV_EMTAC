import os
import sys
import subprocess
import shutil
from sqlalchemy import inspect

# Initialize logger first
from modules.initial_setup.initializer_logger import (
    initializer_logger, compress_logs_except_most_recent, close_initializer_logger, LOG_DIRECTORY
)

# Global variables that will be populated after directory creation
DatabaseConfig = None
MainBase = None
RevisionControlBase = None
directories_to_check = []


def create_base_directories():
    """
    Creates the essential base directories needed before importing modules
    """
    initializer_logger.info("Creating essential base directories...")

    # Create the project root directory path
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    # Essential directories that must exist before importing modules
    essential_dirs = [
        os.path.join(project_root, "Database"),
        os.path.join(project_root, "logs")
    ]

    for directory in essential_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            initializer_logger.info(f"✅ Created essential directory: {directory}")
        else:
            initializer_logger.info(f"✔️ Essential directory already exists: {directory}")


def import_modules_after_directory_setup():
    """
    Imports modules that require base directories to exist
    """
    global DatabaseConfig, MainBase, RevisionControlBase, directories_to_check

    try:
        initializer_logger.info("Importing configuration and database modules...")

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

        initializer_logger.info("✅ Successfully imported all required modules")
        return True
    except Exception as e:
        initializer_logger.error(f"❌ Error importing modules: {e}", exc_info=True)
        return False


def setup_virtual_environment_and_install_requirements():
    """
    Creates a virtual environment if desired and installs requirements.txt dependencies.
    """
    # Calculate the project root directory
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    requirements_file = os.path.join(project_root, "requirements.txt")
    venv_dir = os.path.join(project_root, "venv")

    # Ask user if they want to create a virtual environment
    create_venv = input("Would you like to create a virtual environment? (Recommended) (y/n): ").strip().lower()

    if create_venv == 'y' or create_venv == 'yes':
        # Check if venv already exists
        if os.path.exists(venv_dir):
            overwrite = input(f"Virtual environment already exists at {venv_dir}. Overwrite? (y/n): ").strip().lower()
            if overwrite == 'y' or overwrite == 'yes':
                initializer_logger.info(f"Removing existing virtual environment at {venv_dir}")
                shutil.rmtree(venv_dir)
            else:
                initializer_logger.info("Using existing virtual environment.")

        # Create virtual environment if it doesn't exist or was removed
        if not os.path.exists(venv_dir):
            initializer_logger.info(f"Creating virtual environment at {venv_dir}...")
            try:
                subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
                initializer_logger.info("✅ Virtual environment created successfully.")
            except subprocess.CalledProcessError as e:
                initializer_logger.error(f"❌ Failed to create virtual environment: {e}")
                sys.exit(1)

        # Get the path to the Python executable in the virtual environment
        if sys.platform == "win32":
            venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
        else:
            venv_python = os.path.join(venv_dir, "bin", "python")

        # Upgrade pip in the virtual environment
        try:
            initializer_logger.info("Upgrading pip in virtual environment...")
            subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        except subprocess.CalledProcessError as e:
            initializer_logger.warning(f"⚠️ Failed to upgrade pip: {e}")

        # Use the virtual environment's Python to install requirements
        python_executable = venv_python

        # Print activation instructions for the user
        if sys.platform == "win32":
            activate_cmd = os.path.join(venv_dir, "Scripts", "activate")
            print(f"\n📝 To activate this virtual environment in the future, run: {activate_cmd}")
        else:
            activate_cmd = f"source {os.path.join(venv_dir, 'bin', 'activate')}"
            print(f"\n📝 To activate this virtual environment in the future, run: {activate_cmd}")

        print(
            "⚠️ Note: This script will continue using the virtual environment, but you'll need to activate it manually for future sessions.\n")
    else:
        # Use the current Python if not creating a virtual environment
        initializer_logger.info("Skipping virtual environment creation.")
        python_executable = sys.executable

    # Install requirements
    if os.path.isfile(requirements_file):
        initializer_logger.info(f"📦 Installing dependencies from {requirements_file}...")
        try:
            subprocess.run([python_executable, "-m", "pip", "install", "-r", requirements_file], check=True)
            initializer_logger.info("✅ All dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            initializer_logger.error(f"❌ Failed to install dependencies: {e}")
            sys.exit(1)
    else:
        initializer_logger.warning(
            f"⚠️ No requirements.txt file found at {requirements_file}. Skipping dependency installation.")


def create_directories():
    """
    Ensures all required directories exist, creating them if necessary.
    """
    global directories_to_check

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
    global DatabaseConfig, MainBase, RevisionControlBase

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
    Runs each of the setup scripts in sequence, prompting the user before each one.
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

        # Get a description of what the script does (if available)
        script_description = {
            "load_equipment_relationships_table_data.py": "Loads equipment relationship data into the database",
            "initial_admin.py": "Creates the initial admin user",
            "load_parts_sheet.py": "Imports parts data from spreadsheets",
            "load_active_drawing_list.py": "Loads active drawing list information",
            "load_image_folder.py": "Imports images from the image folder",
            "load_bom_loadsheet.py": "Imports bill of materials data"
        }.get(script, "No description available")

        # Prompt the user
        print(f"\n--- {script} ---")
        print(f"Description: {script_description}")

        user_input = input(f"Run {script}? (y/n): ").strip().lower()

        if user_input == 'y' or user_input == 'yes':
            print(f"\nRunning: {script}...\n")
            try:
                subprocess.run([sys.executable, script_path], check=True)
                print(f"✅ {script} completed successfully.")
                initializer_logger.info(f"✅ {script} completed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"❌ ERROR: Script {script} failed with: {e}")
                initializer_logger.error(f"❌ Script {script} failed with: {e}")

                # Ask if the user wants to continue despite the error
                continue_input = input("Continue with the next script despite the error? (y/n): ").strip().lower()
                if continue_input != 'y' and continue_input != 'yes':
                    initializer_logger.info("Setup aborted by user after script failure.")
                    sys.exit(1)
            except FileNotFoundError:
                print(f"❌ ERROR: Could not find the script {script_path}")
                initializer_logger.error(f"❌ Could not find the script {script_path}")

                # Ask if the user wants to continue despite the missing script
                continue_input = input(
                    "Continue with the next script despite the missing file? (y/n): ").strip().lower()
                if continue_input != 'y' and continue_input != 'yes':
                    initializer_logger.info("Setup aborted by user after missing script file.")
                    sys.exit(1)
        else:
            print(f"Skipping {script}...")
            initializer_logger.info(f"User chose to skip {script}")

    print("\n✅ Setup script sequence completed!")
    initializer_logger.info("✅ Setup script sequence completed!")

    # Ask about log compression
    print("\nWould you like to compress old initializer logs?")
    compress_input = input("Compress logs? (y/n): ").strip().lower()

    if compress_input == 'y' or compress_input == 'yes':
        logs_directory = os.path.join(this_dir, "logs")
        if os.path.exists(logs_directory):
            print("🗜️ Compressing old initializer logs...")
            initializer_logger.info("🗜️ Compressing old initializer logs...")
            compress_logs_except_most_recent(logs_directory)
            print("✔️ Log compression completed.")
            initializer_logger.info("✔️ Log compression completed.")
        else:
            print(f"⚠️ No logs directory found at {logs_directory}.")
            initializer_logger.warning(f"⚠️ No logs directory found at {logs_directory}.")


def main():
    # First, create essential base directories (Database and logs)
    create_base_directories()

    # Import modules that depend on directories existing
    if not import_modules_after_directory_setup():
        initializer_logger.error("❌ Failed to import required modules. Exiting.")
        sys.exit(1)

    # Create remaining directories defined in config
    create_directories()

    # Create virtual environment and ensure dependencies are installed
    setup_virtual_environment_and_install_requirements()

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