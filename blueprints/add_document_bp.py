# blueprints/add_document_bp.py
import psutil
from flask import Blueprint, request, jsonify, redirect, url_for
from plugins.ai_modules import generate_embedding
from modules.emtacdb.emtacdb_fts import (SiteLocation, CompleteDocument, Document, CompletedDocumentPositionAssociation)
from modules.emtacdb.utlity.main_database.database import (create_position, split_text_into_chunks,
                                                           extract_images_from_pdf,
                                                           extract_text_from_pdf,extract_text_from_txt, add_docx_to_db,
                                                           store_embedding)
from modules.configuration.config import DATABASE_DOC, DATABASE_DIR,CURRENT_EMBEDDING_MODEL
import os
from werkzeug.utils import secure_filename
from sqlalchemy import text
from concurrent.futures import ThreadPoolExecutor
import threading
from modules.emtacdb.emtac_revision_control_db import (VersionInfo, RevisionControlSession)
from modules.emtacdb.utlity.revision_database.auditlog import commit_audit_logs, add_audit_log_entry
from modules.configuration.config_env import DatabaseConfig
import traceback
from modules.configuration.log_config import (
    get_request_id, set_request_id, clear_request_id,
    log_with_id, debug_id, info_id, warning_id, error_id, critical_id,
    with_request_id, log_timed_operation
)


# region why: should we move this to a cenetral location?
POST_URL = os.getenv('IMAGE_POST_URL', 'http://localhost:5000/image/add_image')
REQUEST_DELAY = float(os.getenv('REQUEST_DELAY', '1.0'))  # in seconds
# endregion

db_config = DatabaseConfig()

# Create a blueprint for the add_document route
add_document_bp = Blueprint("add_document_bp", __name__)

# Define the route for adding documents
@add_document_bp.route("/add_document", methods=["POST"])
@with_request_id  # Add the decorator to track request IDs
def add_document():
    request_id = get_request_id()
    info_id("Received a request to add documents", request_id)

    if "files" not in request.files:
        error_id("No files uploaded", request_id)
        return jsonify({"message": "No files uploaded"}), 400

    files = request.files.getlist("files")
    info_id(f"Number of files received: {len(files)}", request_id)

    # Collect general metadata from the request
    area = request.form.get("area")
    equipment_group = request.form.get("equipment_group")
    model = request.form.get("model")
    asset_number = request.form.get("asset_number")
    location = request.form.get("location")
    site_location_title = request.form.get("site_location")

    try:
        with log_timed_operation("Processing site location", request_id):
            site_location = None
            with db_config.get_main_session() as session:
                if site_location_title:
                    site_location = session.query(SiteLocation).filter_by(title=site_location_title).first()
                    if not site_location:
                        site_location = SiteLocation(title=site_location_title, room_number="Unknown")
                        session.add(site_location)
                        session.commit()
                    info_id(f"Processed site location: {site_location_title}", request_id)

                position_id = create_position(area, equipment_group, model, asset_number, location,
                                              site_location.id if site_location else None, session,)
                info_id(f"Processed position ID: {position_id}", request_id)

        cpu_count = os.cpu_count()
        max_workers = max(1, cpu_count - 2)
        info_id(f"Using {max_workers} workers based on CPU count.", request_id)

        num_files = len(files)
        file_processing_workers = min(num_files, max_workers)
        remaining_workers = max_workers - file_processing_workers

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for file in files:
                if file.filename == "":
                    warning_id("Skipped empty file", request_id)
                    continue

                title = request.form.get("title")
                if not title:
                    filename = secure_filename(file.filename)
                    file_name_without_extension = os.path.splitext(filename)[0]
                    title = file_name_without_extension.replace('_', ' ')

                filename = secure_filename(file.filename)
                file_path = os.path.join(DATABASE_DOC, filename)
                file.save(file_path)
                info_id(f"Saved file: {file_path}", request_id)

                if file_path.endswith(".docx"):
                    # Process DOCX files by converting them to PDF
                    info_id(f"Processing DOCX file: {file_path}", request_id)
                    success = add_docx_to_db(title, file_path, position_id)
                    if not success:
                        error_id(f"Failed to process DOCX file: {file_path}", request_id)
                        raise Exception(f"Failed to process DOCX file: {file_path}")
                    # Skip the rest of the processing for DOCX files since they're already handled
                    continue  # This is the key change - skip further processing for DOCX files

                with db_config.get_main_session() as session:
                    existing_document = session.query(CompleteDocument)\
                        .filter_by(title=title)\
                        .order_by(CompleteDocument.rev.desc())\
                        .first()
                    if existing_document:
                        new_rev = f"R{int(existing_document.rev[1:]) + 1}"
                        info_id(f"Existing document found. Incrementing revision to: {new_rev}", request_id)
                    else:
                        new_rev = "R0"
                        info_id("No existing document found. Starting with revision R0.", request_id)

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
                    info_id(f"Created CompleteDocument with ID: {complete_document_id}", request_id)

                    # Create and add CompletedDocumentPositionAssociation
                    completed_document_position_association = CompletedDocumentPositionAssociation(
                        complete_document_id=complete_document_id,
                        position_id=position_id
                    )
                    session.add(completed_document_position_association)
                    session.commit()
                    completed_document_position_association_id = completed_document_position_association.id
                    info_id(f"Created CompletedDocumentPositionAssociation with ID: {completed_document_position_association_id}", request_id)

                # Submit the file processing task to the executor
                futures.append(executor.submit(
                    add_document_to_db_multithread,
                    title,
                    file_path,
                    position_id,
                    new_rev,
                    remaining_workers,
                    complete_document_id,
                    completed_document_position_association_id,
                    request_id  # Pass request_id to the thread function
                ))

            for future in futures:
                result = future.result()
                if not result:
                    error_id("One of the file processing tasks failed", request_id)
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
        error_id(f"Error processing files: {e}", request_id)
        error_id(traceback.format_exc(), request_id)
        return jsonify({"message": str(e)}), 500

    info_id("Successfully processed all files", request_id)
    return redirect(request.referrer or url_for("index"))

def add_document_to_db_multithread(title, file_path, position_id, revision, remaining_workers,
                                   complete_document_id, completed_document_position_association_id, request_id=None):
    thread_id = threading.get_ident()

    # If no request_id is provided, generate a new one for this thread
    if request_id is None:
        request_id = set_request_id()

    info_id(f"[Thread {thread_id}] Started processing file: {file_path} with revision: {revision}", request_id)

    try:
        extracted_text = None

        # Correctly obtain a new session using get_main_session()
        with db_config.get_main_session() as session:
            info_id(f"[Thread {thread_id}] Session started for file: {file_path}", request_id)

            # Extract text from the document
            if file_path.endswith(".pdf"):
                with log_timed_operation(f"Extracting text from PDF: {file_path}", request_id):
                    info_id(f"[Thread {thread_id}] Extracting text from PDF...", request_id)
                    extracted_text = extract_text_from_pdf(file_path)
                    info_id(f"[Thread {thread_id}] Text extracted from PDF.", request_id)
            elif file_path.endswith(".txt"):
                with log_timed_operation(f"Extracting text from TXT: {file_path}", request_id):
                    info_id(f"[Thread {thread_id}] Extracting text from TXT...", request_id)
                    extracted_text = extract_text_from_txt(file_path)
                    info_id(f"[Thread {thread_id}] Text extracted from TXT.", request_id)
            else:
                error_id(f"[Thread {thread_id}] Unsupported file format: {file_path}", request_id)
                return None, False

            # Proceed only if text was successfully extracted
            if extracted_text:
                # Check for an existing CompleteDocument with the same title and revision
                existing_document = session.query(CompleteDocument).filter_by(title=title, rev=revision).first()

                if existing_document:
                    info_id(
                        f"[Thread {thread_id}] Found existing document: ID {existing_document.id} with title '{title}' and revision '{revision}'",
                        request_id)
                    # Update the existing document with the extracted content
                    existing_document.content = extracted_text
                    session.commit()
                    complete_document_id = existing_document.id
                    info_id(
                        f"[Thread {thread_id}] Updated existing document with content for ID: {complete_document_id}",
                        request_id)
                else:
                    # Create a new CompleteDocument with the extracted content
                    complete_document = CompleteDocument(
                        title=title,
                        file_path=os.path.relpath(file_path, DATABASE_DIR),
                        content=extracted_text,
                        rev=revision  # Use the passed revision number
                    )
                    session.add(complete_document)
                    session.commit()
                    complete_document_id = complete_document.id
                    info_id(
                        f"[Thread {thread_id}] Added new complete document: {title}, ID: {complete_document_id}, Rev: {revision}",
                        request_id)

                # Create or update the CompletedDocumentPositionAssociation
                completed_document_position_association = session.query(CompletedDocumentPositionAssociation).filter_by(
                    complete_document_id=complete_document_id, position_id=position_id).first()

                if not completed_document_position_association:
                    completed_document_position_association = CompletedDocumentPositionAssociation(
                        complete_document_id=complete_document_id,
                        position_id=position_id
                    )
                    session.add(completed_document_position_association)
                    session.commit()
                    completed_document_position_association_id = completed_document_position_association.id
                    info_id(
                        f"[Thread {thread_id}] Added CompletedDocumentPositionAssociation for complete document ID: {complete_document_id}, position ID: {position_id}",
                        request_id)

                # Add the document to the FTS table
                with log_timed_operation("Adding document to FTS table", request_id):
                    insert_query_fts = "INSERT INTO documents_fts (title, content) VALUES (:title, :content)"
                    session.execute(text(insert_query_fts), {"title": title, "content": extracted_text})
                    session.commit()
                    info_id(f"[Thread {thread_id}] Added document to the FTS table.", request_id)

                # Now that the IDs are created, extract images
                if file_path.endswith(".pdf"):
                    with log_timed_operation(f"Extracting images from PDF: {file_path}", request_id):
                        info_id(f"[Thread {thread_id}] Extracting images from PDF...", request_id)
                        extract_images_from_pdf(file_path, complete_document_id,
                                                completed_document_position_association_id, position_id)
                        info_id(f"[Thread {thread_id}] Images extracted from PDF.", request_id)

            else:
                error_id(f"[Thread {thread_id}] No text extracted from the document.", request_id)
                return None, False

            # Process document chunks and generate embeddings
            with log_timed_operation("Processing document chunks and embeddings", request_id):
                text_chunks = split_text_into_chunks(extracted_text)
                for i, chunk in enumerate(text_chunks):
                    padded_chunk = ' '.join(split_text_into_chunks(chunk, pad_token="", max_words=150))
                    document = Document(
                        name=f"{title} - Chunk {i + 1}",
                        file_path=os.path.relpath(file_path, DATABASE_DIR),
                        content=padded_chunk,
                        complete_document_id=complete_document_id,
                        rev=revision  # Same revision number for document chunk
                    )
                    session.add(document)
                    session.commit()
                    info_id(f"[Thread {thread_id}] Added chunk {i + 1} of document: {title}, Rev: {document.rev}",
                            request_id)

                    if CURRENT_EMBEDDING_MODEL != "NoEmbeddingModel":
                        embeddings = generate_embedding(padded_chunk, CURRENT_EMBEDDING_MODEL)
                        if embeddings is None:
                            warning_id(
                                f"[Thread {thread_id}] Failed to generate embedding for chunk {i + 1} of document: {title}",
                                request_id)
                        else:
                            store_embedding(document.id, embeddings, CURRENT_EMBEDDING_MODEL)
                            info_id(
                                f"[Thread {thread_id}] Generated and stored embedding for chunk {i + 1} of document: {title}",
                                request_id)
                    else:
                        info_id(
                            f"[Thread {thread_id}] No embedding generated for chunk {i + 1} of document: {title} because no model is selected.",
                            request_id)

            # Querying the version_info table in the revision control database
            try:
                with db_config.get_revision_control_session() as revision_session:
                    info_id(f"[Thread {thread_id}] Querying version_info table in revision control database.",
                            request_id)
                    version_info = revision_session.query(VersionInfo).order_by(VersionInfo.id.desc()).first()
                    if version_info:
                        info_id(f"[Thread {thread_id}] Latest version_info: {version_info.version_number}", request_id)
                    else:
                        warning_id(f"[Thread {thread_id}] No version_info found.", request_id)
            except Exception as e:
                error_id(f"[Thread {thread_id}] Error querying version_info table: {e}", request_id)

            info_id(f"[Thread {thread_id}] Successfully processed file: {file_path}", request_id)
            return complete_document_id, True

    except Exception as e:
        error_id(f"[Thread {thread_id}] An error occurred in add_document_to_db_multithread: {e}", request_id)
        error_id(f"[Thread {thread_id}] Attempted Processed file: {file_path}", request_id)
        return None, False

def calculate_optimal_workers(memory_threshold=0.5, max_workers=None, request_id=None):
    """Calculate optimal number of workers based on available memory."""
    available_memory = psutil.virtual_memory().available
    memory_per_thread = 100 * 1024 * 1024  # Example: assume each thread uses 100MB
    max_memory_workers = available_memory // memory_per_thread

    if max_workers is None:
        max_workers = os.cpu_count()

    # Limit workers based on memory and CPU availability
    optimal_workers = min(max_memory_workers, max_workers)

    # Apply a memory threshold to avoid using all available memory
    result = max(1, int(optimal_workers * memory_threshold))

    info_id(f"Calculated optimal workers: {result} (available memory: {available_memory / (1024 * 1024):.2f} MB, "
            f"max workers: {max_workers}, memory threshold: {memory_threshold})", request_id)

    return result

# region Todo: remove from routes once its established that it dost have any knockdown effects
'''def extract_images_from_pdf(file_path, complete_document_id, completed_document_position_association_id, position_id=None):
    """
    Extracts images from a PDF file, uploads them, and creates associations in the database.

    Args:
        file_path (str): Path to the PDF file.
        complete_document_id (int): ID of the completed document.
        completed_document_position_association_id (int): ID of the completed document position association.
        position_id (int, optional): ID of the position. Defaults to None.

    Returns:
        list: List of paths to the extracted images.
    """
    try:
        # Obtain a session using the getter method
        with db_config.get_main_session() as session:
            logger.info("Starting image extraction from PDF.")
            logger.info(f"file_path: {file_path}")
            logger.info(f"complete_document_id: {complete_document_id}")
            logger.info(f"completed_document_position_association_id: {completed_document_position_association_id}")
            logger.info(f"position_id: {position_id}")
            logger.info(f"Session info: {session}")

            if not os.path.exists(TEMPORARY_UPLOAD_FILES):
                os.makedirs(TEMPORARY_UPLOAD_FILES)
                logger.debug(f"Created temporary upload directory at {TEMPORARY_UPLOAD_FILES}")

            doc = fitz.open(file_path)
            total_pages = len(doc)
            logger.info(f"Opened PDF file. Total pages: {total_pages}")

            extracted_images = []
            file_name = os.path.splitext(os.path.basename(file_path))[0].replace("_", " ")

            for page_num in range(total_pages):
                page = doc[page_num]
                img_list = page.get_images(full=True)
                logger.info(f"Processing page {page_num + 1}/{total_pages} with {len(img_list)} images.")

                for img_index, img in enumerate(img_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image.get("ext", "jpg")  # Default to jpg if extension not found
                    temp_path = os.path.join(
                        TEMPORARY_UPLOAD_FILES,
                        f"{file_name}_page{page_num + 1}_image{img_index + 1}.{image_ext}"
                    )

                    with open(temp_path, 'wb') as temp_file:
                        temp_file.write(image_bytes)
                    logger.debug(f"Saved image to {temp_path}")

                    try:
                        with open(temp_path, 'rb') as img_file:
                            response = requests.post(
                                POST_URL,
                                files={'image': img_file},
                                data={
                                    'complete_document_id': complete_document_id,
                                    'completed_document_position_association_id': completed_document_position_association_id,
                                    'position_id': position_id
                                }
                            )

                        if response.status_code == 200:
                            logger.info(f"Successfully processed image {img_index + 1} on page {page_num + 1}.")
                            try:
                                response_data = response.json()
                                image_id = response_data.get('image_id')
                                if image_id:
                                    association = create_image_completed_document_association(
                                        image_id=image_id,
                                        complete_document_id=complete_document_id,
                                        session=session
                                    )
                                    logger.info(f"Created ImageCompletedDocumentAssociation with ID: {association.id}")
                                else:
                                    logger.error(f"'image_id' not found in response for image {img_index + 1} on page {page_num + 1}.")
                            except json.JSONDecodeError:
                                logger.error(f"Invalid JSON response for image {img_index + 1} on page {page_num + 1}: {response.text}")
                        else:
                            logger.error(f"Failed to process image {img_index + 1} on page {page_num + 1}: {response.text} (Status Code: {response.status_code})")
                    except requests.RequestException as req_err:
                        logger.error(f"HTTP request failed for image {img_index + 1} on page {page_num + 1}: {req_err}")
                    except Exception as e:
                        logger.error(f"Unexpected error processing image {img_index + 1} on page {page_num + 1}: {e}")

                    extracted_images.append(temp_path)

                    # Introduce a delay to prevent overwhelming the server
                    time.sleep(REQUEST_DELAY)

            logger.info(f"Image extraction completed. Total images extracted: {len(extracted_images)}")
            return extracted_images

    except Exception as e:
        logger.error(f"An error occurred in extract_images_from_pdf: {e}")
        try:
            session.rollback()
            logger.debug("Database session rolled back due to error.")
        except Exception as rollback_e:
            logger.error(f"Failed to rollback the session: {rollback_e}")
        return []

'''
# endregion