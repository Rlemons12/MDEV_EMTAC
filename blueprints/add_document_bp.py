from flask import Blueprint, request, jsonify, redirect, url_for
from emtacdb_fts import (create_position, SiteLocation, extract_text_from_pdf, extract_text_from_txt, 
                         extract_images_from_pdf, CompleteDocument, CompletedDocumentPositionAssociation,
                         split_text_into_chunks, Document, generate_embedding, store_embedding, CURRENT_EMBEDDING_MODEL)
from config import DATABASE_URL, DATABASE_DOC, DATABASE_DIR
import os
from werkzeug.utils import secure_filename
import pandas as pd
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import create_engine, text
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime

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

# Create a blueprint for the add_document route
add_document_bp = Blueprint("add_document_bp", __name__)

# Define the route for adding documents
@add_document_bp.route("/add_document", methods=["POST"])
def add_document():
    logger.info("Received a request to add documents")

    # Check if the request contains any files
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
        # Process the site location
        site_location = None
        with Session() as session:
            if site_location_title:
                site_location = session.query(SiteLocation).filter_by(title=site_location_title).first()
                if not site_location:
                    site_location = SiteLocation(title=site_location_title, room_number="Unknown")
                    session.add(site_location)
                    session.commit()
                logger.info(f"Processed site location: {site_location_title}")

            # Process the position
            position_id = create_position(area, equipment_group, model, asset_number, location, site_location.id if site_location else None)
            logger.info(f"Processed position ID: {position_id}")

        # Use ThreadPoolExecutor to process files concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for file in files:
                # Check if the file is empty
                if file.filename == "":
                    logger.warning("Skipped empty file")
                    continue  # Skip empty files

                # Get the title from the request
                title = request.form.get("title")
                # If title is not provided, use the filename
                if not title:
                    filename = secure_filename(file.filename)
                    file_name_without_extension = os.path.splitext(filename)[0]
                    title = file_name_without_extension.replace('_', ' ')

                # Ensure a secure filename and save to UPLOAD_FOLDER
                filename = secure_filename(file.filename)
                file_path = os.path.join(DATABASE_DOC, filename)
                file.save(file_path)
                logger.info(f"Saved file: {file_path}")

                # Submit the file processing task to the executor
                futures.append(executor.submit(add_document_to_db_multithread, title, file_path, position_id))

            # Wait for all futures to complete
            for future in futures:
                result = future.result()
                if not result:
                    logger.error("One of the file processing tasks failed")
                    raise Exception("One of the file processing tasks failed")

    except Exception as e:
        logger.error(f"Error processing files: {e}")
        return jsonify({"message": str(e)}), 500

    # After successfully processing the files, redirect to the 'upload_success' endpoint
    logger.info("Successfully processed all files")
    return redirect(url_for('upload_success'))

def add_document_to_db_multithread(title, file_path, position_id):
    thread_id = threading.get_ident()
    logger.info(f"[Thread {thread_id}] Started processing file: {file_path}")

    try:
        extracted_text = None
        complete_document_id = None
        completed_document_position_association_id = None
        with Session() as session:
            logger.info(f"[Thread {thread_id}] Session started for file: {file_path}")
            
            # Thread function to extract text
            def extract_text():
                nonlocal extracted_text
                thread_id = threading.get_ident()
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
                thread_id = threading.get_ident()
                if file_path.endswith(".pdf"):
                    logger.info(f"[Thread {thread_id}] Extracting images from PDF...")
                    extract_images_from_pdf(file_path, session, complete_document_id, completed_document_position_association_id)
                    logger.info(f"[Thread {thread_id}] Images extracted from PDF.")

            # Use ThreadPoolExecutor to run extract_text and extract_images concurrently
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                futures.append(executor.submit(extract_text))
                futures.append(executor.submit(extract_images))
                for future in futures:
                    future.result()  # Wait for all threads to complete

            if extracted_text:
                complete_document = CompleteDocument(
                    title=title,
                    file_path=os.path.relpath(file_path, DATABASE_DIR),
                    content=extracted_text
                )
                session.add(complete_document)
                session.commit()
                complete_document_id = complete_document.id
                logger.info(f"[Thread {thread_id}] Added complete document: {title}, ID: {complete_document_id}")

                completed_document_position_association = CompletedDocumentPositionAssociation(
                    complete_document_id=complete_document_id,
                    position_id=position_id
                )
                session.add(completed_document_position_association)
                session.commit()
                completed_document_position_association_id = completed_document_position_association.id
                logger.info(f"[Thread {thread_id}] Added CompletedDocumentPositionAssociation for complete document ID: {complete_document_id}, position ID: {position_id}")

                insert_query_fts = "INSERT INTO documents_fts (title, content) VALUES (:title, :content)"
                session.execute(text(insert_query_fts), {"title": title, "content": extracted_text})
                session.commit()
                logger.info(f"[Thread {thread_id}] Added document to the FTS table.")

                text_chunks = split_text_into_chunks(extracted_text)
                for i, chunk in enumerate(text_chunks):
                    padded_chunk = ' '.join(split_text_into_chunks(chunk, pad_token="", max_words=150))
                    document = Document(
                        name=f"{title} - Chunk {i+1}",
                        file_path=os.path.relpath(file_path, DATABASE_DIR),
                        content=padded_chunk,
                        complete_document_id=complete_document_id,
                    )
                    session.add(document)
                    session.commit()
                    logger.info(f"[Thread {thread_id}] Added chunk {i+1} of document: {title}")

                    if CURRENT_EMBEDDING_MODEL != "NoEmbeddingModel":
                        embeddings = generate_embedding(padded_chunk, CURRENT_EMBEDDING_MODEL)
                        if embeddings is None:
                            logger.warning(f"[Thread {thread_id}] Failed to generate embedding for chunk {i+1} of document: {title}")
                        else:
                            store_embedding(document.id, embeddings, CURRENT_EMBEDDING_MODEL)
                            logger.info(f"[Thread {thread_id}] Generated and stored embedding for chunk {i+1} of document: {title}")
                    else:
                        logger.info(f"[Thread {thread_id}] No embedding generated for chunk {i+1} of document: {title} because no model is selected.")
            else:
                logger.error(f"[Thread {thread_id}] No text extracted from the document.")
                return None, False

            logger.info(f"[Thread {thread_id}] Successfully processed file: {file_path}")
            return complete_document_id, True
    except Exception as e:
        logger.error(f"[Thread {thread_id}] An error occurred in add_document_to_db_multithread: {e}")
        logger.error(f"[Thread {thread_id}] Attempted Processed file: {file_path}")
        return None, False

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
