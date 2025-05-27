# blueprints/add_document_bp.py
import psutil
from flask import Blueprint, request, jsonify, redirect, url_for
from sqlalchemy import text
from modules.emtacdb.emtacdb_fts import CompleteDocument, Document
from modules.emtacdb.utlity.revision_database.auditlog import commit_audit_logs, add_audit_log_entry
from modules.configuration.config_env import DatabaseConfig
import traceback
from modules.configuration.log_config import (
    get_request_id, set_request_id, clear_request_id,
    log_with_id, debug_id, info_id, warning_id, error_id, critical_id,
    with_request_id, log_timed_operation
)

db_config = DatabaseConfig()

# Create a blueprint for the add_document route
add_document_bp = Blueprint("add_document_bp", __name__)


# Initialize FTS table - this should be called once when setting up the application
def initialize_fts_table():
    """Initialize the FTS table when the app starts."""
    try:
        success = Document.create_fts_table()
        if success:
            info_id("FTS table initialized successfully")
        else:
            warning_id("FTS table initialization failed")
    except Exception as e:
        error_id(f"Error initializing FTS table: {e}")


@add_document_bp.route("/add_document", methods=["POST"])
@with_request_id
def add_document():
    """
    Enhanced document upload route using PostgreSQL-optimized CompleteDocument class.
    Handles file uploads with concurrent processing and proper error handling.
    """
    request_id = get_request_id()
    info_id("Received a request to add documents", request_id)

    # Validate request
    if "files" not in request.files:
        error_id("No files uploaded", request_id)
        return jsonify({"message": "No files uploaded"}), 400

    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        error_id("No valid files provided", request_id)
        return jsonify({"message": "No valid files provided"}), 400

    info_id(f"Processing {len(files)} files: {[f.filename for f in files]}", request_id)

    # Collect and validate metadata
    metadata = {
        'title': request.form.get("title", "").strip(),
        'area': request.form.get("area", "").strip(),
        'equipment_group': request.form.get("equipment_group", "").strip(),
        'model': request.form.get("model", "").strip(),
        'asset_number': request.form.get("asset_number", "").strip(),
        'location': request.form.get("location", "").strip(),
        'site_location': request.form.get("site_location", "").strip(),
        'room_number': request.form.get("room_number", "Unknown").strip(),
        'department': request.form.get("department", "").strip(),
        'tags': request.form.get("tags", "").strip(),
        'priority': request.form.get("priority", "normal").strip(),
    }

    info_id(f"Metadata: {metadata}", request_id)

    try:
        # Ensure FTS table exists before processing
        _ensure_fts_table_exists(request_id)

        # Process upload using the PostgreSQL-optimized method
        with log_timed_operation("Document processing", request_id):
            success = CompleteDocument.process_upload(files, metadata, request_id)

        if success:
            info_id("Successfully processed all files", request_id)
            _log_successful_processing(files, metadata, request_id)
            return redirect(request.referrer or url_for("index"))
        else:
            error_id("Document processing failed", request_id)
            _log_failed_processing(files, "Document processing failed", request_id)
            return jsonify({"message": "Document processing failed"}), 500

    except Exception as e:
        error_id(f"Error processing files: {e}", request_id)
        error_id(traceback.format_exc(), request_id)
        _log_failed_processing(files, str(e), request_id)
        return jsonify({"message": f"Error processing files: {str(e)}"}), 500

def _ensure_fts_table_exists(request_id):
    """Ensure the FTS table exists before processing documents."""
    try:
        # Check if FTS table exists
        with db_config.get_main_session() as session:
            result = session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'documents_fts'
                )
            """)).fetchone()

            if not result[0]:  # Table doesn't exist
                info_id("FTS table missing, creating it now", request_id)
                success = Document.create_fts_table()
                if success:
                    info_id("FTS table created successfully", request_id)
                else:
                    warning_id("FTS table creation failed", request_id)
    except Exception as e:
        warning_id(f"Error checking/creating FTS table: {e}", request_id)

def _log_successful_processing(files, metadata, request_id):
    """Log successful processing to audit trail."""
    try:
        add_audit_log_entry(
            table_name="complete_document",
            operation="CREATE",
            record_id=None,  # Multiple documents created
            new_data={
                "files_processed": len(files),
                "filenames": [f.filename for f in files],
                "metadata": metadata,
                "processing_method": "concurrent_postgresql"
            },
            commit_to_db=False
        )
        commit_audit_logs()
    except Exception as audit_e:
        # Don't fail the whole operation if audit logging fails
        warning_id(f"Audit logging failed: {audit_e}", request_id)

def _log_failed_processing(files, error_message, request_id):
    """Log failed processing to audit trail."""
    try:
        add_audit_log_entry(
            table_name="complete_document",
            operation="ERROR",
            record_id=None,
            new_data={
                "error": error_message,
                "files_attempted": [f.filename for f in files],
                "processing_method": "concurrent_postgresql"
            },
            commit_to_db=False
        )
        commit_audit_logs()
    except Exception:
        pass  # Ignore audit logging errors

@add_document_bp.route("/initialize_fts", methods=["POST"])
@with_request_id
def initialize_fts():
    """
    Administrative endpoint to manually initialize or recreate the FTS table.
    Useful for maintenance and troubleshooting.
    """
    request_id = get_request_id()
    info_id("Manual FTS table initialization requested", request_id)

    try:
        success = Document.create_fts_table()
        if success:
            info_id("FTS table initialized successfully via admin endpoint", request_id)
            return jsonify({"message": "FTS table initialized successfully"}), 200
        else:
            error_id("FTS table initialization failed via admin endpoint", request_id)
            return jsonify({"message": "FTS table initialization failed"}), 500
    except Exception as e:
        error_id(f"Error in manual FTS initialization: {e}", request_id)
        return jsonify({"message": f"Error initializing FTS table: {str(e)}"}), 500


@add_document_bp.route("/health", methods=["GET"])
@with_request_id
def health_check():
    """
    Health check endpoint for the document processing system.
    Checks database connectivity and FTS table status.
    """
    request_id = get_request_id()
    health_status = {
        "status": "healthy",
        "database": "unknown",
        "fts_table": "unknown",
        "timestamp": None
    }

    try:
        # Check database connectivity
        with db_config.get_main_session() as session:
            session.execute(text("SELECT 1")).fetchone()
            health_status["database"] = "connected"

            # Check FTS table
            result = session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'documents_fts'
                )
            """)).fetchone()

            health_status["fts_table"] = "exists" if result[0] else "missing"

        from datetime import datetime
        health_status["timestamp"] = datetime.now().isoformat()

        debug_id(f"Health check completed: {health_status}", request_id)
        return jsonify(health_status), 200

    except Exception as e:
        error_id(f"Health check failed: {e}", request_id)
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
        return jsonify(health_status), 503


def calculate_optimal_workers(memory_threshold=0.5, max_workers=None, request_id=None):
    """
    Calculate optimal number of workers based on available memory.
    This function is kept for compatibility but the CompleteDocument class
    now handles worker optimization automatically.
    """
    available_memory = psutil.virtual_memory().available
    memory_per_thread = 100 * 1024 * 1024  # 100MB per thread estimate
    max_memory_workers = available_memory // memory_per_thread

    if max_workers is None:
        max_workers = psutil.cpu_count()

    # Limit workers based on memory and CPU availability
    optimal_workers = min(max_memory_workers, max_workers)

    # Apply memory threshold to avoid using all available memory
    result = max(1, int(optimal_workers * memory_threshold))

    if request_id:
        info_id(
            f"Calculated optimal workers: {result} "
            f"(available memory: {available_memory / (1024 * 1024):.2f} MB, "
            f"max workers: {max_workers}, memory threshold: {memory_threshold})",
            request_id
        )

    return result


# ==========================================
# DEPRECATED FUNCTIONS (Kept for compatibility)
# These functions are no longer used but kept to avoid breaking imports
# TODO: Remove after confirming no external dependencies
# ==========================================

def add_document_to_db_multithread(*args, **kwargs):
    """
    DEPRECATED: This function has been replaced by CompleteDocument.process_upload()
    Kept for backward compatibility. Consider updating calling code.
    """
    warning_id("add_document_to_db_multithread is deprecated, use CompleteDocument.process_upload()")
    return None, False