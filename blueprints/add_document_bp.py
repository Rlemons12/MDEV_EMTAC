from flask import Blueprint, request, jsonify, redirect, url_for
from emtacdb_fts import (
    create_position, SiteLocation, extract_text_from_pdf, extract_text_from_txt, 
    CompleteDocument, Document, split_text_into_chunks, generate_embedding, 
    store_embedding, CURRENT_EMBEDDING_MODEL, Area, EquipmentGroup, AssetNumber, 
    Model, Location, ChatSession, Position, Image, Drawing, Problem, Solution, 
    Part, ImageEmbedding, PowerPoint, PartsPositionAssociation, 
    ImagePositionAssociation, DrawingPositionAssociation, 
    CompletedDocumentPositionAssociation, ImageCompletedDocumentAssociation, 
    ProblemPositionAssociation, ImageProblemAssociation, 
    CompleteDocumentProblemAssociation, ImageSolutionAssociation
)

from config import DATABASE_URL, DATABASE_DOC, DATABASE_DIR, TEMPORARY_UPLOAD_FILES, REVISION_CONTROL_DB_PATH
import os
from werkzeug.utils import secure_filename
import pandas as pd
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import create_engine, text, event
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from threading import Lock
from datetime import datetime
import fitz
import time
import requests
from emtac_revision_control_db import (
    VersionInfo, RevisionControlBase, RevisionControlSession, SiteLocationSnapshot, 
    PositionSnapshot, AreaSnapshot, EquipmentGroupSnapshot, ModelSnapshot, 
    AssetNumberSnapshot, PartSnapshot, ImageSnapshot, ImageEmbeddingSnapshot, 
    DrawingSnapshot, DocumentSnapshot, CompleteDocumentSnapshot, ProblemSnapshot, 
    SolutionSnapshot, DrawingPartAssociationSnapshot, PartProblemAssociationSnapshot, 
    PartSolutionAssociationSnapshot, DrawingProblemAssociationSnapshot, 
    DrawingSolutionAssociationSnapshot, ProblemPositionAssociationSnapshot, 
    CompleteDocumentProblemAssociationSnapshot, CompleteDocumentSolutionAssociationSnapshot, 
    ImageProblemAssociationSnapshot, ImageSolutionAssociationSnapshot, 
    ImagePositionAssociationSnapshot, DrawingPositionAssociationSnapshot, 
    CompletedDocumentPositionAssociationSnapshot, ImageCompletedDocumentAssociationSnapshot, 
    LocationSnapshot
)
from snapshot_utils import (
    create_sitlocation_snapshot, create_position_snapshot, create_snapshot, 
    create_area_snapshot, create_equipment_group_snapshot, create_model_snapshot, 
    create_asset_number_snapshot, create_part_snapshot, create_image_snapshot, 
    create_image_embedding_snapshot, create_drawing_snapshot, 
    create_document_snapshot, create_complete_document_snapshot, 
    create_problem_snapshot, create_solution_snapshot, 
    create_drawing_part_association_snapshot, create_part_problem_association_snapshot, 
    create_part_solution_association_snapshot, create_drawing_problem_association_snapshot, 
    create_drawing_solution_association_snapshot, create_problem_position_association_snapshot, 
    create_complete_document_problem_association_snapshot, 
    create_complete_document_solution_association_snapshot, 
    create_image_problem_association_snapshot, create_image_solution_association_snapshot, 
    create_image_position_association_snapshot, create_drawing_position_association_snapshot, 
    create_completed_document_position_association_snapshot, 
    create_image_completed_document_association_snapshot, 
    create_parts_position_association_snapshot
)
from auditlog import AuditLog

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure logging to write to a file with timestamps
log_file = f'logs/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'))

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Also add a stream handler for console output
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

# Database setup
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
Session = scoped_session(sessionmaker(bind=engine))

# Revision control database configuration
REVISION_CONTROL_DB_PATH = os.path.join(DATABASE_DIR, 'emtac_revision_control_db.db')
revision_control_engine = create_engine(f'sqlite:///{REVISION_CONTROL_DB_PATH}')
RevisionControlBase = declarative_base()
RevisionControlSession = scoped_session(sessionmaker(bind=revision_control_engine))  # Use distinct name
revision_control_session = RevisionControlSession()

# Apply PRAGMA settings to SQLite database
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA synchronous = OFF")
    cursor.execute("PRAGMA journal_mode = MEMORY")
    cursor.execute("PRAGMA cache_size = -64000")
    cursor.execute("PRAGMA temp_store = MEMORY")
    cursor.close()

# Temporary list to store audit log entries
audit_log_entries = []
audit_log_lock = Lock()

def add_audit_log_entry(table_name, operation, record_id, old_data=None, new_data=None, commit_to_db=False):
    entry = {
        'table_name': table_name,
        'operation': operation,
        'record_id': record_id,
        'old_data': old_data,
        'new_data': new_data,
        'changed_at': datetime.utcnow()
    }

    with audit_log_lock:
        # Append to the temporary in-memory list
        audit_log_entries.append(entry)
        logger.info(f"Audit log entry added to memory: {entry}")

        if commit_to_db:
            try:
                # Optionally commit this entry to the database immediately
                with RevisionControlSession() as session:
                    audit_log = AuditLog(
                        table_name=entry['table_name'],
                        operation=entry['operation'],
                        record_id=entry['record_id'],
                        old_data=entry['old_data'],
                        new_data=entry['new_data'],
                        changed_at=entry['changed_at']
                    )
                    session.add(audit_log)
                    session.commit()
                    logger.info(f"Audit log entry committed to database: {entry}")
            except Exception as e:
                logger.error(f"Failed to commit audit log entry to database: {e}")

def commit_audit_logs():
    with audit_log_lock:
        try:
            with RevisionControlSession() as session:
                for entry in audit_log_entries:
                    audit_log = AuditLog(
                        table_name=entry['table_name'],
                        operation=entry['operation'],
                        record_id=entry['record_id'],
                        old_data=entry['old_data'],
                        new_data=entry['new_data'],
                        changed_at=entry['changed_at']
                    )
                    session.add(audit_log)
                session.commit()
                logger.info(f"All audit log entries committed to database: {len(audit_log_entries)} entries.")
            audit_log_entries.clear()  # Clear the list after committing
            logger.info("Cleared in-memory audit log entries after committing to database.")
        except Exception as e:
            logger.error(f"Failed to commit audit log entries to database: {e}")

# Create a blueprint for the add_document route
add_document_bp = Blueprint("add_document_bp", __name__)

# Define the route for adding documents
@add_document_bp.route("/add_document", methods=["POST"])
def add_document():
    logger.info("Received a request to add documents")

    if "files" not in request.files:
        logger.error("No files uploaded")
        return jsonify({"message": "No files uploaded"}), 400

    files = request.files.getlist("files")
    logger.info(f"Number of files received: {len(files)}")

    # Collect general metadata from the request
    area = request.form.get("area")
    equipment_group = request.form.get("equipment_group")
    model = request.form.get("model")
    asset_number = request.form.get("asset_number")
    location = request.form.get("location")
    site_location_title = request.form.get("site_location")

    try:
        site_location = None
        with Session() as session:
            if site_location_title:
                site_location = session.query(SiteLocation).filter_by(title=site_location_title).first()
                if not site_location:
                    site_location = SiteLocation(title=site_location_title, room_number="Unknown")
                    session.add(site_location)
                    session.commit()
                logger.info(f"Processed site location: {site_location_title}")

            position_id = create_position(area, equipment_group, model, asset_number, location, site_location.id if site_location else None)
            logger.info(f"Processed position ID: {position_id}")

        cpu_count = os.cpu_count()
        max_workers = max(1, cpu_count - 2) 
        logger.info(f"Using {max_workers} workers based on CPU count.")

        num_files = len(files)
        file_processing_workers = min(num_files, max_workers)
        remaining_workers = max_workers - file_processing_workers

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for file in files:
                if file.filename == "":
                    logger.warning("Skipped empty file")
                    continue

                title = request.form.get("title")
                if not title:
                    filename = secure_filename(file.filename)
                    file_name_without_extension = os.path.splitext(filename)[0]
                    title = file_name_without_extension.replace('_', ' ')

                filename = secure_filename(file.filename)
                file_path = os.path.join(DATABASE_DOC, filename)
                file.save(file_path)
                logger.info(f"Saved file: {file_path}")

                with Session() as session:
                    existing_document = session.query(CompleteDocument).filter_by(title=title).order_by(CompleteDocument.rev.desc()).first()
                    if existing_document:
                        new_rev = f"R{int(existing_document.rev[1:]) + 1}"
                        logger.info(f"Existing document found. Incrementing revision to: {new_rev}")
                    else:
                        new_rev = "R0"
                        logger.info("No existing document found. Starting with revision R0.")

                    # Create and add CompleteDocument
                    complete_document = CompleteDocument(
                        title=title,
                        file_path=os.path.relpath(file_path, DATABASE_DIR),
                        content=None,  # Text will be added later
                        rev=new_rev
                    )
                    session.add(complete_document)
                    session.commit()
                    complete_document_id = complete_document.id
                    logger.info(f"Created CompleteDocument with ID: {complete_document_id}")

                    # Create and add CompletedDocumentPositionAssociation
                    completed_document_position_association = CompletedDocumentPositionAssociation(
                        complete_document_id=complete_document_id,
                        position_id=position_id
                    )
                    session.add(completed_document_position_association)
                    session.commit()
                    completed_document_position_association_id = completed_document_position_association.id
                    logger.info(f"Created CompletedDocumentPositionAssociation with ID: {completed_document_position_association_id}")

                # Submit the file processing task to the executor
                futures.append(executor.submit(
                    add_document_to_db_multithread,
                    title,
                    file_path,
                    position_id,
                    new_rev,
                    remaining_workers,
                    complete_document_id,
                    completed_document_position_association_id
                ))

            for future in futures:
                result = future.result()
                if not result:
                    logger.error("One of the file processing tasks failed")
                    add_audit_log_entry(
                        table_name="complete_document",
                        operation="ERROR",
                        record_id=None,
                        new_data={"error": "One of the file processing tasks failed"},
                        commit_to_db=False
                    )
                    raise Exception("One of the file processing tasks failed")

        # At the end, save audit log entries to the database
        commit_audit_logs()

    except Exception as e:
        logger.error(f"Error processing files: {e}")
        return jsonify({"message": str(e)}), 500

    logger.info("Successfully processed all files")
    return redirect(url_for('upload_success'))

def add_document_to_db_multithread(title, file_path, position_id, revision, remaining_workers, complete_document_id, completed_document_position_association_id):
    thread_id = threading.get_ident()
    logger.info(f"[Thread {thread_id}] Started processing file: {file_path}")
    
    try:
        extracted_text = None
        with Session() as session:
            logger.info(f"[Thread {thread_id}] Session started for file: {file_path}")

            # Thread function to extract text
            def extract_text():
                nonlocal extracted_text
                if file_path.endswith(".pdf"):
                    logger.info(f"[Thread {thread_id}] Extracting text from PDF...")
                    extracted_text = extract_text_from_pdf(file_path)
                    logger.info(f"[Thread {thread_id}] Text extracted from PDF.")
                elif file_path.endswith(".txt"):
                    logger.info(f"[Thread {thread_id}] Extracting text from TXT...")
                    extracted_text = extract_text_from_txt(file_path)
                    logger.info(f"[Thread {thread_id}] Text extracted from TXT.")
                else:
                    logger.error(f"[Thread {thread_id}] Unsupported file format: {file_path}")
                    return None

            # Thread function to extract images
            def extract_images():
                if file_path.endswith(".pdf"):
                    logger.info(f"[Thread {thread_id}] Extracting images from PDF...")
                    extract_images_from_pdf(file_path, session, complete_document_id, completed_document_position_association_id)
                    logger.info(f"[Thread {thread_id}] Images extracted from PDF.")

            # Use ThreadPoolExecutor to run extract_text and extract_images concurrently
            with ThreadPoolExecutor(max_workers=max(1, remaining_workers)) as executor:
                futures = []
                futures.append(executor.submit(extract_text))
                futures.append(executor.submit(extract_images))
                for future in futures:
                    future.result()  # Wait for all threads to complete

            if extracted_text:
                # Insert the CompleteDocument and get the ID
                complete_document = CompleteDocument(
                    title=title,
                    file_path=os.path.relpath(file_path, DATABASE_DIR),
                    content=extracted_text,
                    rev=revision
                )
                session.add(complete_document)
                session.commit()
                complete_document_id = complete_document.id  # Retrieve the newly created document ID
                
                # Log the INSERT operation
                add_audit_log_entry(
                    table_name="complete_document",
                    operation="INSERT",
                    record_id=complete_document_id,  # Use the correct record_id
                    new_data={"title": title, "file_path": file_path}
                )

                # Insert the CompletedDocumentPositionAssociation and get the ID
                completed_document_position_association = CompletedDocumentPositionAssociation(
                    complete_document_id=complete_document_id,
                    position_id=position_id
                )
                session.add(completed_document_position_association)
                session.commit()
                completed_document_position_association_id = completed_document_position_association.id

                # Log the INSERT operation for the association
                add_audit_log_entry(
                    table_name="completed_document_position_association",
                    operation="INSERT",
                    record_id=completed_document_position_association_id,  # Use the correct record_id
                    new_data={"complete_document_id": complete_document_id, "position_id": position_id}
                )

                # Insert the document into the FTS table
                insert_query_fts = "INSERT INTO documents_fts (title, content) VALUES (:title, :content)"
                session.execute(text(insert_query_fts), {"title": title, "content": extracted_text})
                session.commit()

                # Log the INSERT operation for the FTS entry
                add_audit_log_entry(
                    table_name="documents_fts",
                    operation="INSERT",
                    record_id=complete_document_id,  # Assuming this uses the same document ID
                    new_data={"title": title, "content": extracted_text}
                )

            else:
                logger.error(f"[Thread {thread_id}] No text extracted from the document.")
                return None, False

            logger.info(f"[Thread {thread_id}] Successfully processed file: {file_path}")
            return complete_document_id, True

    except Exception as e:
        logger.error(f"[Thread {thread_id}] An error occurred in add_document_to_db_multithread: {e}")
        return None, False

def extract_images_from_pdf(file_path, session, complete_document_id, completed_document_position_association_id):
    # Log the received arguments to ensure they are being passed correctly
    logger.info(f"extract_images_from_pdf called with arguments:")
    logger.info(f"file_path: {file_path}")
    logger.info(f"complete_document_id: {complete_document_id}")
    logger.info(f"completed_document_position_association_id: {completed_document_position_association_id}")
    logger.info(f"Session info: {session}")

    logger.info(f"Opening PDF file from: {file_path}")
    doc = fitz.open(file_path)
    total_pages = len(doc)
    logger.info(f"Total pages in the PDF: {total_pages}")
    logger.info(f"Inside MT extract_images_from_pdf, complete_document_id: {complete_document_id}")
    logger.info(f"CompletedDocumentPositionAssociation ID: {completed_document_position_association_id}")
    extracted_images = []

    file_name = os.path.splitext(os.path.basename(file_path))[0].replace("_", " ")

    if not os.path.exists(TEMPORARY_UPLOAD_FILES):
        os.makedirs(TEMPORARY_UPLOAD_FILES)

    def process_image(page_num, img_index, image_bytes):
        temp_path = os.path.join(TEMPORARY_UPLOAD_FILES, f"{file_name}_page{page_num + 1}_image{img_index + 1}.jpg")  
        
        with open(temp_path, 'wb') as temp_file:
            temp_file.write(image_bytes)
        
        logger.info("Preparing to send POST request to http://localhost:5000/image/add_image")
        logger.info(f"File path: {temp_path}")
        logger.info(f"complete_document_id: {complete_document_id}")
        logger.info(f"completed_document_position_association_id: {completed_document_position_association_id}")
        
        try:
            response = requests.post('http://localhost:5000/image/add_image', 
                                    files={'image': open(temp_path, 'rb')}, 
                                    data={'complete_document_id': complete_document_id,
                                          'completed_document_position_association_id': completed_document_position_association_id})

            if response.status_code == 200:
                logger.info("Image processed successfully")
                logger.info(f"Response from server: {response.text}")  
            else:
                logger.error(f"Failed to process image: {response.text}")
                logger.error(f"HTTP Status Code: {response.status_code}")
        except Exception as e:
            logger.error(f"Exception occurred while processing image: {e}")

        extracted_images.append(temp_path)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for page_num in range(total_pages):
            page = doc[page_num]
            img_list = page.get_images(full=True)

            logger.info(f"Processing page {page_num + 1}/{total_pages} with {len(img_list)} images.")

            for img_index, img in enumerate(img_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                futures.append(executor.submit(process_image, page_num, img_index, image_bytes))

        for future in futures:
            future.result()

    # Log the final state before returning
    logger.info(f"extract_images_from_pdf completed. Extracted images: {extracted_images}")

    return extracted_images

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
