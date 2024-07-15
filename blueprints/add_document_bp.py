from flask import Blueprint, request, jsonify, redirect, url_for
from emtacdb_fts import create_position, SiteLocation, add_document_to_db, add_docx_to_db, add_text_file_to_db, add_csv_data_to_db, Position, create_position
from blueprints import DATABASE_DOC,DATABASE_URL
import os
from werkzeug.utils import secure_filename
import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import logging

# Database setup (ensure DATABASE_URI is defined in your config)'
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a blueprint for the add_document route
add_document_bp = Blueprint("add_document_bp", __name__)

# Define the route for adding documents
@add_document_bp.route("/add_document", methods=["POST"])
def add_document():
    # Check if the request contains any files
    if "files" not in request.files:
        return jsonify({"message": "No files uploaded"}), 400

    files = request.files.getlist("files")

    for file in files:
        # Check if the file is empty
        if file.filename == "":
            continue  # Skip empty files

        try:
            # Get the title from the request
            title = request.form.get("title")
            # If title is not provided, use the filename
            if not title:
                filename = secure_filename(file.filename)
                file_name_without_extension = os.path.splitext(filename)[0]
                title = file_name_without_extension.replace('_', '')

            # Get other variables from the request
            area = request.form.get("area")
            print(f'{area}')
            equipment_group = request.form.get("equipment_group")
            print(f'{equipment_group}')
            model = request.form.get("model")
            asset_number = request.form.get("asset_number")
            location = request.form.get("location")
            site_location_title = request.form.get("site_location")

            # Ensure a secure filename and save to UPLOAD_FOLDER
            filename = secure_filename(file.filename)
            file_path = os.path.join(DATABASE_DOC, filename)
            file.save(file_path)

            # Process the site location
            site_location = None
            if site_location_title:
                with Session() as session:
                    site_location = session.query(SiteLocation).filter_by(title=site_location_title).first()
                    if not site_location:
                        site_location = SiteLocation(title=site_location_title, room_number="Unknown")
                        session.add(site_location)
                        session.commit()

            # Process the position
            position_id = create_position(area, equipment_group, model, asset_number, location, site_location.id if site_location else None)

            # Process the file with the associated position
            added = process_file(title, file_path, position_id)

        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            return jsonify({"message": str(e)}), 500

    # After successfully processing the files, redirect to the 'upload_success' endpoint
    return redirect(url_for('upload_success'))

def process_file(title, file_path, position_id):
    logger.info(f"Processing file: {file_path}")

    # Extract the file name from the file path
    file_name = os.path.basename(file_path)

    logger.info(f"File name: {file_name}")

    # Determine the file type (pdf, docx, txt, csv, xlsx) and process accordingly
    if file_path.endswith(".pdf"):
        logger.info("Detected PDF file.")
        added = add_document_to_db(title, file_path, position_id)
    elif file_path.endswith(".docx"):
        logger.info("Detected Word document file.")
        added = add_docx_to_db(title, file_path, position_id)
    elif file_path.endswith(".txt"):
        logger.info("Detected TXT file.")
        added = add_text_file_to_db(title, file_path, position_id)
    elif file_path.endswith(".csv"):
        logger.info("Detected CSV file.")
        added = add_csv_data_to_db(file_path, position_id)
    elif file_path.endswith(".xlsx"):
        logger.info("Detected Excel file.")
        # Convert Excel to CSV
        csv_file_path = os.path.splitext(file_path)[0] + ".csv"  # New CSV file path
        if excel_to_csv(file_path, csv_file_path):
            logger.info("Converted Excel to CSV successfully.")
            added = add_csv_data_to_db(csv_file_path, position_id)
        else:
            logger.error("Failed to convert Excel to CSV.")
            added = False
    else:
        logger.error("Unsupported file format.")
        added = False

    if added:
        logger.info(f"Document added successfully: {file_path}")
        # Now, move the uploaded file to the DATABASE_DOC directory
        destination_path = os.path.join(DATABASE_DOC, file_name)
        os.rename(file_path, destination_path)
        logger.info(f"File moved to DATABASE_DOC directory: {destination_path}")
    else:
        logger.error(f"Failed to add the document: {file_path}")
        logger.debug(f"Position ID: {position_id}")

def excel_to_csv(excel_file_path, csv_file_path):
    try:
        # Read Excel file
        df = pd.read_excel(excel_file_path)
        # Save as CSV
        df.to_csv(csv_file_path, index=False)
        return True
    except Exception as e:
        logger.error(f"Failed to convert Excel to CSV: {e}")
        return False
