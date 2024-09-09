import os 
# API key for OpenAI
OPENAI_API_KEY = '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' #was API_KEY 
HUGGINGFACE_API_KEY="..."
#Visual Code api = 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

# Get the current directory path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
#Important pathways
TEMPLATE_FOLDER_PATH = os.path.join(BASE_DIR, 'templates')
KEYWORDS_FILE_PATH = os.path.join(BASE_DIR,"static", 'keywords_file.xlsx')  # Update with the actual filename or path
DATABASE_DIR = os.path.join(BASE_DIR, 'Database')
DATABASE_PATH = os.path.join(DATABASE_DIR, 'emtac_db.db')
REVISION_CONTROL_DB_PATH = os.path.join(DATABASE_DIR, 'emtac_revision_control_db.db')
CSV_DIR = DATABASE_DIR
COMMENT_IMAGES_FOLDER = os.path.join(BASE_DIR,'static', 'comment_images')
UPLOAD_FOLDER = os.path.join(BASE_DIR,"static", "uploads")
IMAGES_FOLDER = os.path.join(BASE_DIR,"static", "images")
DATABASE_PATH_IMAGES_FOLDER = os.path.join(DATABASE_DIR, 'DB_IMAGES')
PDF_FOR_EXTRATION_FOLDER = os.path.join("static", "image_extraction")
IMAGES_EXTRACTED = os.path.join("static","extracted_pdf_images")
COPY_FILES = False
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}  # Allowed image file extensions
TEMPORARY_FILES = os.path.join(DATABASE_DIR, 'temp_files')
PPT2PDF_PPT_FILES_PROCESS = os.path.join(DATABASE_DIR, 'PPT_FILES')
PPT2PDF_PDF_FILES_PROCESS = os.path.join(DATABASE_DIR, 'PDF_FILES')
DATABASE_URL = f'sqlite:///{os.path.join(BASE_DIR, "Database", "emtac_db.db")}'
DATABASE_DOC = os.path.join(DATABASE_DIR, 'DB_DOC')
TEMPORARY_UPLOAD_FILES = os.path.join(DATABASE_DIR, 'temp_upload_files')
DB_LOADSHEET = os.path.join(DATABASE_DIR, "DB_LOADSHEETS")
DB_LOADSHEETS_BACKUP = os.path.join(DATABASE_DIR, "DB_LOADSHEETS_BACKUP")
DB_LOADSHEET_BOMS = os.path.join(DATABASE_DIR, "DB_LOADSHEET_BOMS")
BACKUP_DIR = os.path.join(DATABASE_DIR, "db_backup")
Utility_tools = os.path.join(BASE_DIR, "Utility_tools")
OPENAI_MODEL_NAME = "text-embedding-3-small"
NUM_VERSIONS_TO_KEEP = 3
ADMIN_CREATION_PASSWORD= "12345"

CURRENT_AI_MODEL="OpenAIModel"
CURRENT_EMBEDDING_MODEL="OpenAIEmbeddingModel" 




# NLP model setup
nlp_model_name = "mrm8488/t5-base-finetuned-question-generation-ap"
auth_token = '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' #huggingface_token


#List of directories to check and create
directories_to_check = [
    TEMPLATE_FOLDER_PATH,
    DATABASE_DIR,
    UPLOAD_FOLDER,
    IMAGES_FOLDER,
    DATABASE_PATH_IMAGES_FOLDER,
    PDF_FOR_EXTRATION_FOLDER,
    IMAGES_EXTRACTED,
    TEMPORARY_FILES,
    PPT2PDF_PPT_FILES_PROCESS,
    PPT2PDF_PDF_FILES_PROCESS,
    DATABASE_DOC,
    TEMPORARY_UPLOAD_FILES,
    DB_LOADSHEET,
    DB_LOADSHEETS_BACKUP,
    BACKUP_DIR,
    Utility_tools
]