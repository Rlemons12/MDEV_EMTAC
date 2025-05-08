# setup_manager.py (in the root directory)
import os
import sys
import shutil
import logging
import subprocess
from datetime import datetime


class SetupManager:
    """
    Manages the setup process by controlling the order of operations
    and handling dependencies in a methodical way.
    """

    def __init__(self):
        """Initialize with paths and basic logging"""
        # Set up basic paths
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = self.script_dir

        # Set up basic logging - this doesn't depend on any other modules
        self.setup_basic_logging()

        # Define essential directories
        self.essential_dirs = [
            os.path.join(self.project_root, "Database"),
            os.path.join(self.project_root, "logs"),
            os.path.join(self.project_root, "Database", "db_backup"),
            os.path.join(self.project_root, "Database", "DB_DOC"),
            os.path.join(self.project_root, "Database", "DB_IMAGES"),
            os.path.join(self.project_root, "Database", "DB_LOADSHEETS"),
            os.path.join(self.project_root, "Database", "DB_LOADSHEETS_BACKUP"),
            os.path.join(self.project_root, "Database", "logs"),
            os.path.join(self.project_root, "Database", "PDF_FILES"),
            os.path.join(self.project_root, "Database", "PPT_FILES")
        ]

        # Directory for application logs
        self.log_directory = os.path.join(self.project_root, "logs")

        # Track if certain steps have been completed
        self.directories_created = False
        self.virtual_env_created = False
        self.database_initialized = False

    def setup_basic_logging(self):
        """Set up basic logging without depending on any custom modules"""
        # Ensure logs directory exists
        log_dir = os.path.join(self.project_root, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Set up logging
        log_file = os.path.join(log_dir, f"setup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger('SetupManager')
        self.logger.info("Setup process started")

    def create_essential_directories(self):
        """Create all essential directories needed before database initialization"""
        self.logger.info("Creating essential directories...")

        for directory in self.essential_dirs:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                self.logger.info(f"Created directory: {directory}")
            else:
                self.logger.info(f"Directory already exists: {directory}")

        # Create .gitkeep files to maintain directory structure in git
        for directory in self.essential_dirs:
            if directory != os.path.join(self.project_root, "Database") and directory != os.path.join(self.project_root,
                                                                                                      "logs"):
                gitkeep_file = os.path.join(directory, ".gitkeep")
                if not os.path.exists(gitkeep_file):
                    with open(gitkeep_file, "w") as f:
                        pass  # Create empty file
                    self.logger.info(f"Created .gitkeep file in: {directory}")

        self.directories_created = True
        self.logger.info("Essential directories created successfully")

    def setup_virtual_environment(self):
        """Set up a virtual environment if requested"""
        # Only import what we need for this method
        import subprocess

        self.logger.info("Setting up virtual environment...")

        venv_dir = os.path.join(self.project_root, "venv")
        requirements_file = os.path.join(self.project_root, "requirements.txt")

        # Ask user if they want to create a virtual environment
        create_venv = input("Would you like to create a virtual environment? (Recommended) (y/n): ").strip().lower()

        if create_venv == 'y' or create_venv == 'yes':
            # Check if venv already exists
            if os.path.exists(venv_dir):
                overwrite = input(
                    f"Virtual environment already exists at {venv_dir}. Overwrite? (y/n): ").strip().lower()
                if overwrite == 'y' or overwrite == 'yes':
                    self.logger.info(f"Removing existing virtual environment at {venv_dir}")
                    shutil.rmtree(venv_dir)
                else:
                    self.logger.info("Using existing virtual environment.")

            # Create virtual environment if it doesn't exist or was removed
            if not os.path.exists(venv_dir):
                self.logger.info(f"Creating virtual environment at {venv_dir}...")
                try:
                    subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
                    self.logger.info("Virtual environment created successfully.")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Failed to create virtual environment: {e}")
                    sys.exit(1)

            # Get the path to the Python executable in the virtual environment
            if sys.platform == "win32":
                venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
            else:
                venv_python = os.path.join(venv_dir, "bin", "python")

            # Upgrade pip in the virtual environment
            try:
                self.logger.info("Upgrading pip in virtual environment...")
                subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], check=True)
            except subprocess.CalledProcessError as e:
                self.logger.warning(f"Failed to upgrade pip: {e}")

            python_executable = venv_python

            # Print activation instructions for the user
            if sys.platform == "win32":
                activate_cmd = os.path.join(venv_dir, "Scripts", "activate")
                print(f"\nTo activate this virtual environment in the future, run: {activate_cmd}")
            else:
                activate_cmd = f"source {os.path.join(venv_dir, 'bin', 'activate')}"
                print(f"\nTo activate this virtual environment in the future, run: {activate_cmd}")

            print(
                "Note: This script will continue using the virtual environment, but you'll need to activate it manually for future sessions.\n")

            self.virtual_env_created = True
        else:
            # Use the current Python if not creating a virtual environment
            self.logger.info("Skipping virtual environment creation.")
            python_executable = sys.executable

        # Install requirements
        if os.path.isfile(requirements_file):
            self.logger.info(f"Installing dependencies from {requirements_file}...")
            try:
                subprocess.run([python_executable, "-m", "pip", "install", "-r", requirements_file], check=True)
                self.logger.info("All dependencies installed successfully.")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to install dependencies: {e}")
                sys.exit(1)
        else:
            self.logger.warning(
                f"No requirements.txt file found at {requirements_file}. Skipping dependency installation.")

        # Return the Python executable to use for subsequent steps
        return python_executable

    def initialize_database(self):
        """Initialize the database and create required tables"""
        # Only import database modules after directories are created
        if not self.directories_created:
            self.logger.error("Cannot initialize database before creating directories.")
            return False

        self.logger.info("Initializing database...")

        try:
            # Import database modules here, after directories are created
            from sqlalchemy import create_engine, inspect
            from modules.configuration.config_env import DatabaseConfig

            # Import Base classes
            self.logger.info("Importing database models...")
            from modules.emtacdb.emtacdb_fts import Base as MainBase
            from modules.emtacdb.emtac_revision_control_db import RevisionControlBase

            # Initialize database config
            db_config = DatabaseConfig()
            main_engine = db_config.main_engine
            revision_engine = db_config.revision_control_engine

            # Check if main database tables exist
            main_inspector = inspect(main_engine)
            main_tables = main_inspector.get_table_names()
            if not main_tables:
                self.logger.warning("No tables found in the main database. Creating tables...")
                MainBase.metadata.create_all(main_engine)
                self.logger.info("Main database tables created successfully.")
            else:
                self.logger.info(f"Main database is ready with tables: {main_tables}")

            # Check if revision control database tables exist
            revision_inspector = inspect(revision_engine)
            revision_tables = revision_inspector.get_table_names()
            if not revision_tables:
                self.logger.warning("No tables found in the revision control database. Creating tables...")
                RevisionControlBase.metadata.create_all(revision_engine)
                self.logger.info("Revision control database tables created successfully.")
            else:
                self.logger.info(f"Revision control database is ready with tables: {revision_tables}")

            self.database_initialized = True
            return True

        except Exception as e:
            self.logger.error(f"Database setup failed: {e}", exc_info=True)
            return False

    def run_setup_scripts(self, python_executable=None):
        """Run the individual setup scripts with user prompts"""
        if not self.database_initialized:
            self.logger.error("Cannot run setup scripts before initializing database.")
            return False

        if python_executable is None:
            python_executable = sys.executable

        self.logger.info("Running setup scripts...")

        # Define scripts and their descriptions
        scripts_to_run = [
            {
                "name": "load_equipment_relationships_table_data.py",
                "description": "Loads equipment relationship data into the database",
                "path": os.path.join(self.project_root, "modules", "initial_setup",
                                     "load_equipment_relationships_table_data.py")
            },
            {
                "name": "initial_admin.py",
                "description": "Creates the initial admin user",
                "path": os.path.join(self.project_root, "modules", "initial_setup", "initial_admin.py")
            },
            {
                "name": "load_parts_sheet.py",
                "description": "Imports parts data from spreadsheets",
                "path": os.path.join(self.project_root, "modules", "initial_setup", "load_parts_sheet.py")
            },
            {
                "name": "load_active_drawing_list.py",
                "description": "Loads active drawing list information",
                "path": os.path.join(self.project_root, "modules", "initial_setup", "load_active_drawing_list.py")
            },
            {
                "name": "load_image_folder.py",
                "description": "Imports images from the image folder",
                "path": os.path.join(self.project_root, "modules", "initial_setup", "load_image_folder.py")
            },
            {
                "name": "load_bom_loadsheet.py",
                "description": "Imports bill of materials data",
                "path": os.path.join(self.project_root, "modules", "initial_setup", "load_bom_loadsheet.py")
            }
        ]

        for script in scripts_to_run:
            script_name = script["name"]
            script_description = script["description"]
            script_path = script["path"]

            # Prompt the user
            print(f"\n--- {script_name} ---")
            print(f"Description: {script_description}")

            user_input = input(f"Run {script_name}? (y/n): ").strip().lower()

            if user_input == 'y' or user_input == 'yes':
                print(f"\nRunning: {script_name}...\n")
                try:
                    subprocess.run([python_executable, script_path], check=True)
                    print(f"✅ {script_name} completed successfully.")
                    self.logger.info(f"{script_name} completed successfully.")
                except subprocess.CalledProcessError as e:
                    print(f"❌ ERROR: Script {script_name} failed with: {e}")
                    self.logger.error(f"Script {script_name} failed with: {e}")

                    # Ask if the user wants to continue despite the error
                    continue_input = input("Continue with the next script despite the error? (y/n): ").strip().lower()
                    if continue_input != 'y' and continue_input != 'yes':
                        self.logger.info("Setup aborted by user after script failure.")
                        return False
                except FileNotFoundError:
                    print(f"❌ ERROR: Could not find the script {script_path}")
                    self.logger.error(f"Could not find the script {script_path}")

                    # Ask if the user wants to continue despite the missing script
                    continue_input = input(
                        "Continue with the next script despite the missing file? (y/n): ").strip().lower()
                    if continue_input != 'y' and continue_input != 'yes':
                        self.logger.info("Setup aborted by user after missing script file.")
                        return False
            else:
                print(f"Skipping {script_name}...")
                self.logger.info(f"User chose to skip {script_name}")

        print("\n✅ Setup script sequence completed!")
        self.logger.info("Setup script sequence completed!")

        # Ask about log compression
        print("\nWould you like to compress old logs?")
        compress_input = input("Compress logs? (y/n): ").strip().lower()

        if compress_input == 'y' or compress_input == 'yes':
            self.compress_logs()

        return True

    def compress_logs(self):
        """Compress old log files"""
        print("🗜️ Compressing old logs...")
        self.logger.info("Compressing old logs...")

        try:
            # Import the log compression function
            from modules.initial_setup.initializer_logger import compress_logs_except_most_recent

            # Compress logs
            compress_logs_except_most_recent(self.log_directory)

            print("✔️ Log compression completed.")
            self.logger.info("Log compression completed.")
        except Exception as e:
            print(f"⚠️ Error compressing logs: {e}")
            self.logger.warning(f"Error compressing logs: {e}")

    def run_setup(self):
        """Run the complete setup process"""
        try:
            # Step 1: Create essential directories
            self.create_essential_directories()

            # Step 2: Set up virtual environment and install dependencies
            python_executable = self.setup_virtual_environment()

            # Step 3: Initialize the database
            if self.initialize_database():
                # Step 4: Run setup scripts
                self.run_setup_scripts(python_executable)

                print("\n✅ Setup completed successfully!")
                self.logger.info("Setup completed successfully!")
            else:
                print("\n❌ Setup failed during database initialization.")
                self.logger.error("Setup failed during database initialization.")

        except Exception as e:
            print(f"\n❌ An unexpected error occurred during setup: {e}")
            self.logger.error(f"Unexpected error during setup: {e}", exc_info=True)
            return False

        return True


# Run the setup if this file is executed directly
if __name__ == "__main__":
    setup_manager = SetupManager()
    setup_manager.run_setup()