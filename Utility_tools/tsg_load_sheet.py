import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sys
import os
import shutil
from datetime import datetime
import logging
import json  # Import json module
# Ensure the parent directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATABASE_DIR, BASE_DIR, DATABASE_URL, DB_LOADSHEET, DB_LOADSHEETS_BACKUP
from emtacdb_fts import (Area, EquipmentGroup, Model, AssetNumber, Location, SiteLocation, Problem, Solution, Base, Position, ProblemPositionAssociation, 
    split_text_into_chunks, extract_images_from_pdf, create_position, Document, load_config_from_db)
from plugins.ai_models import generate_embedding
from config_env import DatabaseConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize DatabaseConfig
db_config = DatabaseConfig()
MainSession = db_config.get_main_session()

# Load the current AI and embedding model configurations from the database
current_ai_model, current_embedding_model = load_config_from_db()

def backup_database():
    session = MainSession  # Directly use MainSession as it is a scoped session
    try:
        # Define the directory to store backup Excel files
        backup_directory = os.path.join(DB_LOADSHEETS_BACKUP)

        # Create the backup directory if it doesn't exist
        if not os.path.exists(backup_directory):
            os.makedirs(backup_directory)

        # Get the current date and time for the timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create the Excel file name with the timestamp
        excel_file_name = f"tsg_database_backup_{timestamp}.xlsx"
        excel_file_path = os.path.join(backup_directory, excel_file_name)

        # Extract data from each table and create DataFrames
        area_data = [(area.name, area.description) for area in session.query(Area).all()]
        equipment_group_data = [(group.name, group.area_id) for group in session.query(EquipmentGroup).all()]
        model_data = [(model.name, model.description, model.equipment_group_id) for model in session.query(Model).all()]
        asset_number_data = [(asset.number, asset.model_id, asset.description) for asset in session.query(AssetNumber).all()]
        location_data = [(location.name, location.model_id) for location in session.query(Location).all()]
        problem_data = [(problem.name, problem.description) for problem in session.query(Problem).all()]
        solution_data = [(solution.description, solution.problem_id) for solution in session.query(Solution).all()]

        # Create DataFrames from the extracted data
        df_area = pd.DataFrame(area_data, columns=['name', 'description'])
        df_equipment_group = pd.DataFrame(equipment_group_data, columns=['name', 'area_id'])
        df_model = pd.DataFrame(model_data, columns=['name', 'description', 'equipment_group_id'])
        df_asset_number = pd.DataFrame(asset_number_data, columns=['number', 'model_id', 'description'])
        df_location = pd.DataFrame(location_data, columns=['name', 'model_id'])
        df_problem = pd.DataFrame(problem_data, columns(['name', 'description']))
        df_solution = pd.DataFrame(solution_data, columns=['description', 'problem_id'])

        # Write DataFrames to the Excel file
        with pd.ExcelWriter(excel_file_path) as writer:
            df_area.to_excel(writer, sheet_name='Area', index=False)
            df_equipment_group.to_excel(writer, sheet_name='EquipmentGroup', index=False)
            df_model.to_excel(writer, sheet_name='Model', index=False)
            df_asset_number.to_excel(writer, sheet_name='AssetNumber', index=False)
            df_location.to_excel(writer, sheet_name='Location', index=False)
            df_problem.to_excel(writer, sheet_name='Problem', index=False)
            df_solution.to_excel(writer, sheet_name='Solution', index=False)

        logger.info("Database backup created successfully: %s", excel_file_name)
    except Exception as e:
        logger.error("Error creating database backup: %s", e)

def add_tsg_loadsheet_to_document_table_db(file_path, area_data, equipment_group_data, model_data, asset_number_data,
                                           location_data, problem_data, solution_data):
    session = MainSession  # Directly use MainSession as it is a scoped session
    try:
        logger.info(f"Reading Excel file: {file_path}")
        # Read the Excel file
        df = pd.read_excel(file_path)

        # Log the column names found in the Excel file
        logger.info(f"Columns in the Excel file: {df.columns}")

        # Check if the necessary columns are present
        required_columns = ['area', 'model', 'location', 'problem', 'solution']
        if not all(column in df.columns for column in required_columns):
            logger.error("Excel file is missing required columns.")
            return None, False

        # Initialize document_id for return value
        document_id = None

        # Iterate through each row in the dataframe
        for index, row in df.iterrows():
            # Extract data from the row
            problem_desc = row['problem']
            location_name = row['location'].strip()  # Strip leading and trailing spaces from location name
            model_name = row['model'].strip()  # Strip leading and trailing spaces from model name
            solution_desc = row['solution']  # Extract solution description

            logger.info(
                f"Processing row {index + 1} - Problem: {problem_desc}, Location: {location_name}, Model: {model_name}, Solution: {solution_desc}")

            # Query Location table to find matching location_id
            location = session.query(Location).filter(Location.name == location_name).first()
            logger.info("Checking location: %s", location_name)  # Log the location being checked
            if location:
                location_id = location.id
                logger.info("Location found: %s", location_name)  # Log when a location is found
            else:
                logger.warning("Location '%s' not found.", location_name)  # Log when a location is not found
                continue

            # Query Model table to find matching model_id
            model = session.query(Model).filter(Model.name == model_name).first()
            logger.info("Checking model: %s", model_name)  # Log the model being checked
            if model:
                model_id = model.id
                logger.info("Model found: %s", model_name)  # Log when a model is found
            else:
                logger.warning("Model '%s' not found.", model_name)  # Log when a model is not found
                continue

            # Create a new position if required
            position_id = create_position(None, None, model_id, None, location_id, None,
                                          session)  # Adjust parameters as needed
            if not position_id:
                logger.error("Failed to create or retrieve position.")
                continue

            # Check if the problem description already exists in the Problem table
            existing_problem = session.query(Problem).filter(Problem.description == problem_desc).first()
            if existing_problem:
                logger.info("Problem with description '%s' already exists.", problem_desc)
                continue

            # Create a new entry in the Problem table
            new_problem = Problem(name=problem_desc, description=problem_desc)
            session.add(new_problem)
            session.commit()

            # Insert solution into Solution table if it exists and is valid
            if not pd.isna(solution_desc) and solution_desc.strip():
                new_solution = Solution(description=solution_desc, problem_id=new_problem.id)
                session.add(new_solution)
                session.commit()
            else:
                logger.warning("Invalid solution description for problem '%s'. Skipping solution insertion.",
                               problem_desc)

            # Create a new entry in the ProblemPositionAssociation table
            problem_position_association = ProblemPositionAssociation(problem_id=new_problem.id,
                                                                      position_id=position_id)
            session.add(problem_position_association)
            session.commit()

            # Concatenate extracted text for embedding
            extracted_text = f"{problem_desc} {location_name} {model_name} {solution_desc}"

            logger.info("Splitting extracted text into chunks.")
            # Split text into chunks and process each chunk
            text_chunks = split_text_into_chunks(extracted_text)
            for i, chunk in enumerate(text_chunks):
                padded_chunk = ' '.join(split_text_into_chunks(chunk, pad_token="", max_words=150))

                if current_embedding_model != "NoEmbeddingModel":
                    embeddings = generate_embedding(padded_chunk, current_embedding_model)
                    if embeddings is None:
                        logger.warning(f"Failed to generate embedding for chunk {i + 1} of document: {file_path}")
                    else:
                        store_embedding(document_id, embeddings, current_embedding_model)
                        logger.info(f"Generated and stored embedding for chunk {i + 1} of document: {file_path}")
                else:
                    logger.info(
                        f"No embedding generated for chunk {i + 1} of document: {file_path} because no model is selected.")

        logger.info(f"Successfully processed file: {file_path}")
        return document_id, True
    except Exception as e:
        logger.error(f"An error occurred while adding document from Excel: {e}")
        return None, False

# Main script logic

# Backup the database
backup_database()

# Extract data from each table and create DataFrames
session = MainSession  # Directly use MainSession as it is a scoped session
try:
    area_data = [(area.name, area.description) for area in session.query(Area).all()]
    equipment_group_data = [(group.name, group.area_id) for group in session.query(EquipmentGroup).all()]
    model_data = [(model.name, model.description, model.equipment_group_id) for model in session.query(Model).all()]
    asset_number_data = [(asset.number, asset.model_id, asset.description) for asset in session.query(AssetNumber).all()]
    location_data = [(location.name, location.model_id) for location in session.query(Location).all()]
    problem_data = [(problem.name, problem.description) for problem in session.query(Problem).all()]
    solution_data = [(solution.description, solution.problem_id) for solution in session.query(Solution).all()]
finally:
    session.close()  # Ensure the session is closed after the operation

# Load data from load sheet
load_sheet_path = os.path.join(DB_LOADSHEET, "tsg_load_sheet.xlsx")

# Add the load sheet data to the Document table
document_id, success = add_tsg_loadsheet_to_document_table_db(load_sheet_path, area_data, equipment_group_data,
                                                              model_data, asset_number_data, location_data,
                                                              problem_data, solution_data)
if success:
    logger.info(f"Document added with ID: {document_id}")
else:
    logger.error("Failed to add document from Excel")