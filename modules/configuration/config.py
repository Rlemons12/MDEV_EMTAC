"""Project configuration file"""
#modules/configuration/config.py

import os
import sys
from dotenv import load_dotenv

# Determine the root directory based on whether the code is frozen (e.g., PyInstaller .exe)
if getattr(sys, 'frozen', False):  # Check if running as an executable
    BASE_DIR = os.path.dirname(sys.executable)  # Use the directory of the executable
else:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # Use the AuMaintdb root directory

# Add the current directory to the Python module search path for flexibility
sys.path.append(BASE_DIR)

#os.getenv
# Get the current directory path

#Important pathways
load_dotenv()
DATABASE_URL = "postgresql://postgres:emtac123@localhost:5432/emtac"
ENABLE_REVISION_CONTROL = False   # or True once you wire it up
TEMPLATE_FOLDER_PATH = os.path.join(BASE_DIR, 'templates')
LOAD_FOLDER = os.path.join(BASE_DIR, 'load_process')
LOAD_FOLDER_REFERENCE = os.path.join(BASE_DIR, 'load_process', 'load_reference')
LOAD_FOLDER_INTAKE = os.path.join(BASE_DIR, 'load_process', 'load_intake_sheets')
LOAD_FOLDER_OUTPUT = os.path.join(BASE_DIR, 'load_process', 'load_output')
KEYWORDS_FILE_PATH = os.path.join(BASE_DIR,"static", 'keywords_file.xlsx')  # Update with the actual filename or path
DATABASE_DIR = os.path.join(BASE_DIR, 'Database')
DATABASE_PATH = os.path.join(DATABASE_DIR, 'emtac_db.db')
REVISION_CONTROL_DB_PATH = os.path.join(DATABASE_DIR, 'emtac_revision_control_db.db')
CSV_DIR = DATABASE_DIR
COMMENT_IMAGES_FOLDER = os.path.join(BASE_DIR,'static', 'comment_images')
UPLOAD_FOLDER = os.path.join(BASE_DIR,"static", "uploads")
IMAGES_FOLDER = os.path.join(BASE_DIR,"static", "images")
DATABASE_PATH_IMAGES_FOLDER = os.path.join(DATABASE_DIR, 'DB_IMAGES')
PDF_FOR_EXTRACTION_FOLDER = os.path.join("../../static", "image_extraction")
IMAGES_EXTRACTED = os.path.join("../../static", "extracted_pdf_images")
COPY_FILES = False
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}  # Allowed image file extensions
TEMPORARY_FILES = os.path.join(DATABASE_DIR, 'temp_files')
PPT2PDF_PPT_FILES_PROCESS = os.path.join(DATABASE_DIR, 'PPT_FILES')
PPT2PDF_PDF_FILES_PROCESS = os.path.join(DATABASE_DIR, 'PDF_FILES')
DATABASE_DOC = os.path.join(DATABASE_DIR, 'DB_DOC')
TEMPORARY_UPLOAD_FILES = os.path.join(DATABASE_DIR, 'temp_upload_files')
DB_LOADSHEET = os.path.join(DATABASE_DIR, "DB_LOADSHEETS")
DB_LOADSHEETS_BACKUP = os.path.join(DATABASE_DIR, "DB_LOADSHEETS_BACKUP")
DB_LOADSHEET_BOMS = os.path.join(DATABASE_DIR, "DB_LOADSHEET_BOMS")
DRAWING_IMPORT_DATA_DIR = os.path.join(DB_LOADSHEET, "drawing_import_data")
BACKUP_DIR = os.path.join(DATABASE_DIR, "db_backup")
Utility_tools = os.path.join(BASE_DIR, "utility_tools")
UTILITIES = os.path.join(BASE_DIR,'utilities')
OPENAI_MODEL_NAME = "text-embedding-3-small"
NUM_VERSIONS_TO_KEEP = 3
ADMIN_CREATION_PASSWORD= "12345"
# Base directory for all local AI models
GPT4ALL_MODELS_PATH = os.path.join(BASE_DIR, 'plugins', 'ai_modules', 'gpt4all')
# Add this line too
SENTENCE_TRANSFORMERS_MODELS_PATH = os.path.join(BASE_DIR, 'plugins', 'huggingface')
CURRENT_AI_MODEL="OpenAIModel"
CURRENT_EMBEDDING_MODEL="OpenAIEmbeddingModel"


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
HUGGINGFACE_API_KEY="..."
Visual_Code_api = os.getenv('Visual Code api')


ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
# NLP model setup
nlp_model_name = os.getenv('nlp_model_name')
auth_token = os.getenv('auth_token')



#List of directories to check and create

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
    BACKUP_DIR,
    Utility_tools,
    UTILITIES
]


# --- Orchestrator base (single source of truth) ---
# Allow override by env var, but default to modules/emtac_ai under the project
ORC_BASE_DIR = os.getenv(
    "ORCHESTRATOR_BASE_DIR",
    os.path.join(BASE_DIR, "modules", "emtac_ai")
)

# (Optional) Keep this only if something else uses it. Otherwise you can remove ORC_PROJECT_ROOT.
ORC_PROJECT_ROOT = BASE_DIR  # same as project root

# Models directory
ORC_MODELS_DIR = os.path.join(ORC_BASE_DIR, "models")

# Training data directory
ORC_TRAINING_DATA_DIR = os.path.join(ORC_BASE_DIR, "training_data", "datasets")
ORC_TRAINING_DATA_LOADSHEET = os.path.join(ORC_TRAINING_DATA_DIR, "loadsheet")

# Main training data directory (parent of datasets)
ORC_TRAINING_DATA_ROOT = os.path.join(ORC_BASE_DIR, "training_data")


# Specific model directories
ORC_INTENT_MODEL_DIR = os.path.join(ORC_MODELS_DIR, "intent_classifier")
ORC_PARTS_MODEL_DIR = os.path.join(ORC_MODELS_DIR, "parts")
ORC_IMAGES_MODEL_DIR = os.path.join(ORC_MODELS_DIR, "images")
ORC_DOCUMENTS_MODEL_DIR = os.path.join(ORC_MODELS_DIR, "documents")
ORC_DRAWINGS_MODEL_DIR = os.path.join(ORC_MODELS_DIR, "drawings")
ORC_TOOLS_MODEL_DIR = os.path.join(ORC_MODELS_DIR, "tools")
ORC_TROUBLESHOOTING_MODEL_DIR = os.path.join(ORC_MODELS_DIR, "troubleshooting")

# Specific training data directories
ORC_INTENT_TRAIN_DATA_DIR = os.path.join(ORC_TRAINING_DATA_DIR, "intent_classifier")
ORC_PARTS_TRAIN_DATA_DIR = os.path.join(ORC_TRAINING_DATA_DIR, "parts")
ORC_IMAGES_TRAIN_DATA_DIR = os.path.join(ORC_TRAINING_DATA_DIR, "images")
ORC_DOCUMENTS_TRAIN_DATA_DIR = os.path.join(ORC_TRAINING_DATA_DIR, "documents")
ORC_DRAWINGS_TRAIN_DATA_DIR = os.path.join(ORC_TRAINING_DATA_DIR, "drawings")
ORC_TOOLS_TRAIN_DATA_DIR = os.path.join(ORC_TRAINING_DATA_DIR, "tools")
ORC_TROUBLESHOOTING_TRAIN_DATA_DIR = os.path.join(ORC_TRAINING_DATA_DIR, "troubleshooting")

# Query template directories - These paths are CORRECT and match your actual structure
ORC_QUERY_TEMPLATES_TRAIN_DATA_DIR = os.path.join(ORC_BASE_DIR, "training_data", "query_templates")
ORC_QUERY_TEMPLATE_PARTS = os.path.join(ORC_BASE_DIR, "training_data", "query_templates", "parts")
ORC_QUERY_TEMPLATE_DRAWINGS = os.path.join(ORC_BASE_DIR, "training_data", "query_templates", "drawings")

# NEW: Additional directories found in your project
ORC_ORCHESTRATOR_DIR = os.path.join(ORC_BASE_DIR, "orchestrator")
ORC_ORCHESTRATOR_TEST_SCRIPTS_DIR = os.path.join(ORC_ORCHESTRATOR_DIR, "test_scripts_orchestrator")

ORC_TEST_SCRIPTS_DIR = os.path.join(ORC_BASE_DIR, "test_scripts")

ORC_TRAINING_MODULE_DIR = os.path.join(ORC_BASE_DIR, "training_module")

# load sheets
ORC_TRAINING_DATA_DRAWINGS_LOADSHEET_PATH= os.path.join(ORC_TRAINING_DATA_LOADSHEET,"drawing_loadsheet.xlsx")
ORC_TRAINING_DATA_PARTS_LOADSHEET_PATH= os.path.join(ORC_TRAINING_DATA_LOADSHEET,"parts_loadsheet.xlsx")

ORC_TRAINING_SCRIPTS_DIR = os.path.join(ORC_BASE_DIR, "training_scripts")
ORC_TRAINING_SCRIPTS_DATASET_GEN_DIR = os.path.join(ORC_TRAINING_SCRIPTS_DIR, "dataset_gen")
ORC_TRAINING_SCRIPTS_INTENT_TRAIN_DIR = os.path.join(ORC_TRAINING_SCRIPTS_DIR, "dataset_intent_train")
ORC_TRAINING_SCRIPTS_PERFORMANCE_DIR = os.path.join(ORC_TRAINING_SCRIPTS_DIR, "performance_tst_model")
ORC_TRAINING_SCRIPTS_TST_DIR = os.path.join(ORC_TRAINING_SCRIPTS_DIR, "tst")

ORC_UTIL_SCRIPTS_DIR = os.path.join(ORC_BASE_DIR, "util_scripts")

# Optional: A dictionary to easily iterate over model directories
MODEL_DIRS = {
    "intent_classifier": ORC_INTENT_MODEL_DIR,
    "parts": ORC_PARTS_MODEL_DIR,
    "images": ORC_IMAGES_MODEL_DIR,
    "documents": ORC_DOCUMENTS_MODEL_DIR,
    "drawings": ORC_DRAWINGS_MODEL_DIR,
    "tools": ORC_TOOLS_MODEL_DIR,
    "troubleshooting": ORC_TROUBLESHOOTING_MODEL_DIR,
}

# Training data directories dictionary
TRAIN_DATA_DIRS = {
    "intent_classifier": ORC_INTENT_TRAIN_DATA_DIR,
    "parts": ORC_PARTS_TRAIN_DATA_DIR,
    "images": ORC_IMAGES_TRAIN_DATA_DIR,
    "documents": ORC_DOCUMENTS_TRAIN_DATA_DIR,
    "drawings": ORC_DRAWINGS_TRAIN_DATA_DIR,
    "tools": ORC_TOOLS_TRAIN_DATA_DIR,
    "troubleshooting": ORC_TROUBLESHOOTING_TRAIN_DATA_DIR,
}

# NEW: Additional directories dictionary for complete project structure
PROJECT_DIRS = {
    "base": ORC_BASE_DIR,
    "models": ORC_MODELS_DIR,
    "training_data_root": ORC_TRAINING_DATA_ROOT,
    "training_data_datasets": ORC_TRAINING_DATA_DIR,
    "training_data_loadsheet": ORC_TRAINING_DATA_LOADSHEET,
    "query_templates": ORC_QUERY_TEMPLATES_TRAIN_DATA_DIR,
    "query_template_parts": ORC_QUERY_TEMPLATE_PARTS,
    "query_template_drawings": ORC_QUERY_TEMPLATE_DRAWINGS,
    "orchestrator": ORC_ORCHESTRATOR_DIR,
    "orchestrator_test_scripts": ORC_ORCHESTRATOR_TEST_SCRIPTS_DIR,
    "test_scripts": ORC_TEST_SCRIPTS_DIR,
    "training_module": ORC_TRAINING_MODULE_DIR,
    "training_scripts": ORC_TRAINING_SCRIPTS_DIR,
    "training_scripts_dataset_gen": ORC_TRAINING_SCRIPTS_DATASET_GEN_DIR,
    "training_scripts_intent_train": ORC_TRAINING_SCRIPTS_INTENT_TRAIN_DIR,
    "training_scripts_performance": ORC_TRAINING_SCRIPTS_PERFORMANCE_DIR,
    "training_scripts_tst": ORC_TRAINING_SCRIPTS_TST_DIR,
    "util_scripts": ORC_UTIL_SCRIPTS_DIR,
}

# Complete list of all directories for setup scripts
ALL_DIRS = [
    ORC_BASE_DIR,
    ORC_MODELS_DIR,
    ORC_TRAINING_DATA_ROOT,
    ORC_TRAINING_DATA_DIR,
    ORC_TRAINING_DATA_LOADSHEET,
    ORC_QUERY_TEMPLATES_TRAIN_DATA_DIR,
    ORC_QUERY_TEMPLATE_PARTS,
    ORC_QUERY_TEMPLATE_DRAWINGS,
    ORC_ORCHESTRATOR_DIR,
    ORC_ORCHESTRATOR_TEST_SCRIPTS_DIR,
    ORC_TEST_SCRIPTS_DIR,
    ORC_TRAINING_MODULE_DIR,
    ORC_TRAINING_SCRIPTS_DIR,
    ORC_TRAINING_SCRIPTS_DATASET_GEN_DIR,
    ORC_TRAINING_SCRIPTS_INTENT_TRAIN_DIR,
    ORC_TRAINING_SCRIPTS_PERFORMANCE_DIR,
    ORC_TRAINING_SCRIPTS_TST_DIR,
    ORC_UTIL_SCRIPTS_DIR,
] + list(MODEL_DIRS.values()) + list(TRAIN_DATA_DIRS.values())