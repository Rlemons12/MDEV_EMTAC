import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from emtac_revision_control_db import (VersionInfo, RevisionControlBase, SiteLocationSnapshot, PositionSnapshot, AreaSnapshot,
                                       EquipmentGroupSnapshot, ModelSnapshot, AssetNumberSnapshot, PartSnapshot, ImageSnapshot,
                                       ImageEmbeddingSnapshot, DrawingSnapshot, DocumentSnapshot, CompleteDocumentSnapshot,
                                       ProblemSnapshot, TaskSnapshot, DrawingPartAssociationSnapshot, PartProblemAssociationSnapshot,
                                       PartTaskAssociationSnapshot, DrawingProblemAssociationSnapshot, DrawingTaskAssociationSnapshot,
                                       ProblemPositionAssociationSnapshot, CompleteDocumentProblemAssociationSnapshot,
                                       CompleteDocumentTaskAssociationSnapshot, ImageProblemAssociationSnapshot, ImageTaskAssociationSnapshot,
                                       ImagePositionAssociationSnapshot, DrawingPositionAssociationSnapshot, CompletedDocumentPositionAssociationSnapshot,
                                       ImageCompletedDocumentAssociationSnapshot
                                       )
from emtacdb_fts import (split_text_into_chunks, AIModelConfig, ImageEmbedding, ImageModelConfig, ChatSession, User, engine, search_documents_fts, search_images_by_keyword, find_keyword_and_extract_detail,
                         load_keywords_to_db, perform_action_based_on_keyword, load_keywords_and_patterns,
                         find_most_relevant_document, create_session, update_session, get_session, QandA,
                         ChatSession, Area, EquipmentGroup, Model, AssetNumber, Location, SiteLocation, Position,
                         Document, Image, Drawing, Problem, Task, CompleteDocument, PowerPoint,
                         PartsPositionImageAssociation, ImagePositionAssociation, DrawingPositionAssociation,
                         CompletedDocumentPositionAssociation, ImageCompletedDocumentAssociation,
                         ProblemPositionAssociation, ImageProblemAssociation, CompleteDocumentProblemAssociation,
                         ImageTaskAssociation, UserLevel, User, AIModelConfig, load_config_from_db, load_image_model_config_from_db)

from snapshot_utils import(
    create_sitlocation_snapshot, create_position_snapshot, create_snapshot,
    create_area_snapshot, create_equipment_group_snapshot, create_model_snapshot, create_asset_number_snapshot,
    create_part_snapshot, create_image_snapshot, create_image_embedding_snapshot, create_drawing_snapshot,
    create_document_snapshot, create_complete_document_snapshot, create_problem_snapshot, create_task_snapshot,
    create_drawing_part_association_snapshot, create_part_problem_association_snapshot, create_part_task_association_snapshot,
    create_drawing_problem_association_snapshot, create_drawing_task_association_snapshot, create_problem_position_association_snapshot,
    create_complete_document_problem_association_snapshot, create_complete_document_task_association_snapshot,
    create_image_problem_association_snapshot, create_image_task_association_snapshot, create_image_position_association_snapshot,
    create_drawing_position_association_snapshot, create_completed_document_position_association_snapshot, create_image_completed_document_association_snapshot,
    create_parts_position_association_snapshot)

from auditlog import AuditLog, log_insert, log_update, log_delete  # Ensure this is the correct module for these functions

from config import (TEMPORARY_FILES, OPENAI_API_KEY, DATABASE_PATH_IMAGES_FOLDER,
                    DATABASE_DOC, DATABASE_URL, DATABASE_DIR, PPT2PDF_PPT_FILES_PROCESS,
                    PPT2PDF_PDF_FILES_PROCESS,UPLOAD_FOLDER)
from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, configure_mappers, relationship, scoped_session, sessionmaker
import log_config
from plugins import load_ai_model, load_embedding_model
logger = log_config.logger
from flask_bcrypt import Bcrypt
from flask import Flask, Blueprint, request, jsonify, current_app, url_for, redirect

# Log that the application is starting
logger.info("Starting the Flask application")

# Import blueprints
from pst_troubleshooting_task_bp import pst_troubleshooting_task_bp
from pst_troubleshooting_guide_edit_update_bp import pst_troubleshooting_guide_edit_update_bp
from pst_troubleshooting_solution_bp import pst_troubleshooting_solution_bp
from blueprints.pst_troubleshooting_position_update_bp import pst_troubleshooting_position_update_bp
from blueprints.pst_troubleshooting_new_entry_bp import pst_troubleshoot_new_entry_bp
from blueprints.pst_troubleshooting_bp import pst_troubleshooting_bp
from blueprints.tsg_search_parts_bp import tsg_search_parts_bp
from blueprints.search_drawing_by_number_bp import search_drawing_by_number_bp
from blueprints.tsg_search_drawing_bp import tsg_search_drawing_bp
from blueprints.update_problem_solution_bp import update_problem_solution_bp
from blueprints.tsg_search_images_bp import tsg_search_images_bp
from blueprints.get_drawing_data_bp import get_drawing_data_bp
from blueprints.get_search_troubleshooting_guide_data_bp import get_search_troubleshooting_guide_data_bp
from blueprints.get_search_tsg_list_data_bp import get_search_tsg_list_data_bp
from blueprints.search_problem_solution_bp import search_problem_solution_bp
from blueprints.get_tsg_search_image_list_data_bp import get_tsg_search_image_list_data_bp
from blueprints.trouble_shooting_guide_bp import trouble_shooting_guide_bp
from blueprints.get_troubleshooting_guide_data_bp import get_troubleshooting_guide_data_bp
from blueprints.get_search_powerpoint_list_data_bp import get_search_powerpoint_list_data_bp
from blueprints.get_search_image_list_data_bp import get_search_image_list_data_bp
from blueprints.get_completedocument_list_data_bp import get_completedocument_list_data_bp
from blueprints.get_powerpoint_list_data_bp import get_powerpoint_list_data_bp
from blueprints.get_image_list_data_bp import get_image_list_data_bp
from blueprints.get_list_data_bp import get_list_data_bp
from blueprints.get_batch_list_data_bp import get_batch_list_data_bp
from blueprints.chatbot_bp import chatbot_bp
from blueprints.image_bp import image_bp
from blueprints.add_document_bp import add_document_bp
from blueprints.upload_powerpoint_bp import upload_powerpoint_bp
from blueprints.search_images_bp import search_images_bp
from blueprints.search_powerpoint_bp import search_powerpoint_bp
from blueprints.display_pdf_bp import display_pdf_bp
from blueprints.search_documents_bp import search_documents_bp
from blueprints.search_powerpoint_fts_bp import search_powerpoint_fts_bp
from blueprints.login_bp import login_bp
from blueprints.create_user_bp import create_user_bp
from blueprints.search_documents_fts_bp import search_documents_fts_bp
from blueprints.logout_bp import logout_bp
from blueprints.batch_processing_bp import batch_processing_bp
from blueprints.admin_bp import admin_bp
from blueprints.image_compare_bp import image_compare_bp
from blueprints.folder_image_embedding_bp import folder_image_embedding_bp
from blueprints.bill_of_materials_bp import bill_of_materials_bp  # Newly added blueprint
from blueprints.bill_of_materials_data_bp import bill_of_materials_data_bp
from blueprints.get_bill_of_material_query_data import get_bill_of_material_query_data_bp
from blueprints.create_bill_of_material import create_bill_of_material_bp
from blueprints.enter_new_part import enter_new_part_bp
from blueprints.get_troubleshooting_guide_edit_data_bp import get_troubleshooting_guide_edit_data_bp
from blueprints.comment_pop_up_bp import comment_pop_up_bp
from blueprints.update_part_bp import update_part_bp
from blueprints.position_data_assignment import position_data_assignment_bp
from blueprints.position_data_assignment_data_add_dependencies_bp import position_data_assignment_data_add_dependencies_bp

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

# Function to load config from the database
def load_config_from_db():
    session = Session()
    ai_model_config = session.query(AIModelConfig).filter_by(key="CURRENT_AI_MODEL").first()
    embedding_model_config = session.query(AIModelConfig).filter_by(key="CURRENT_EMBEDDING_MODEL").first()
    session.close()

    current_ai_model = ai_model_config.value if ai_model_config else "NoAIModel"
    current_embedding_model = embedding_model_config.value if embedding_model_config else "NoEmbeddingModel"

    return current_ai_model, current_embedding_model

# Load the current AI and embedding models from the database
current_ai_model, current_embedding_model = load_config_from_db()

# Load the models
ai_model = load_ai_model(current_ai_model)
embedding_model = load_embedding_model(current_embedding_model)

def register_blueprints(app):
    app.register_blueprint(pst_troubleshooting_guide_edit_update_bp, url_prefix='/pst_troubleshooting_guide_edit_update')
    app.register_blueprint(pst_troubleshooting_task_bp,url_prefix='/pst_troubleshooting_task')
    app.register_blueprint(pst_troubleshooting_solution_bp, url_prefix='/pst_troubleshooting_solution')
    app.register_blueprint(pst_troubleshooting_position_update_bp, url_prefix='/pst_troubleshooting_position_update')
    app.register_blueprint(pst_troubleshoot_new_entry_bp, url_prefix='/pst_troubleshoot_new_entry')
    app.register_blueprint(pst_troubleshooting_bp, url_prefix='/pst_troubleshooting')
    app.register_blueprint(chatbot_bp, url_prefix='/chatbot')
    app.register_blueprint(image_bp, url_prefix='/image')
    app.register_blueprint(add_document_bp, url_prefix='/documents')
    app.register_blueprint(search_images_bp, url_prefix='/search_images')
    app.register_blueprint(upload_powerpoint_bp, url_prefix='/powerpoints')
    app.register_blueprint(search_powerpoint_bp, url_prefix='/search_powerpoint')
    app.register_blueprint(display_pdf_bp, url_prefix='/')
    app.register_blueprint(search_documents_bp, url_prefix='/search_documents')
    app.register_blueprint(search_documents_fts_bp, url_prefix='/search_documents_fts')
    app.register_blueprint(search_powerpoint_fts_bp, url_prefix='/search_powerpoint_fts')
    app.register_blueprint(login_bp,url_prefix='/')
    app.register_blueprint(create_user_bp, url_prefix='/')
    app.register_blueprint(logout_bp)
    app.register_blueprint(batch_processing_bp,url_prefix='/')
    app.register_blueprint(get_list_data_bp, url_prefix='/')
    app.register_blueprint(get_batch_list_data_bp, url_prefix='/batch')
    app.register_blueprint(get_image_list_data_bp, url_prefix='/')
    app.register_blueprint(get_powerpoint_list_data_bp, url_prefix='/')
    app.register_blueprint(get_completedocument_list_data_bp, url_prefix='/')
    app.register_blueprint(get_search_image_list_data_bp,url_prefix='/')
    app.register_blueprint(get_search_powerpoint_list_data_bp, url_prefix='/')
    app.register_blueprint(get_troubleshooting_guide_data_bp, url_prefix='/')
    app.register_blueprint(trouble_shooting_guide_bp, url_prefix='/')
    app.register_blueprint(get_tsg_search_image_list_data_bp, url_prefix='/')
    app.register_blueprint(tsg_search_images_bp, url_prefix= '/tsg_search_images')
    app.register_blueprint(search_problem_solution_bp, url_prefix= '/')
    app.register_blueprint(get_search_tsg_list_data_bp,url_prefix='/')
    app.register_blueprint(get_search_troubleshooting_guide_data_bp, url_prefix='/')
    app.register_blueprint(get_drawing_data_bp,url_prefix='/')
    app.register_blueprint(update_problem_solution_bp, url_prefix='/')
    app.register_blueprint(tsg_search_drawing_bp,url_prefix='/')
    app.register_blueprint(search_drawing_by_number_bp,url_prefix='/')
    app.register_blueprint(tsg_search_parts_bp,url_prefix='/')
    app.register_blueprint(admin_bp, url_prefix='/')
    app.register_blueprint(image_compare_bp, url_prefix='/')
    app.register_blueprint(folder_image_embedding_bp, url_prefix='/folder_image_embedding')
    app.register_blueprint(bill_of_materials_bp,url_prefix='/')  # Registering the bill_of_materials blueprint
    app.register_blueprint(bill_of_materials_data_bp,url_prefix='/')
    app.register_blueprint(get_bill_of_material_query_data_bp)
    app.register_blueprint(create_bill_of_material_bp)
    app.register_blueprint(enter_new_part_bp)
    app.register_blueprint(comment_pop_up_bp)
    app.register_blueprint(update_part_bp)
    app.register_blueprint(position_data_assignment_bp)
    app.register_blueprint(position_data_assignment_data_add_dependencies_bp)


app = Flask(__name__)
app.secret_key = '1234'

# Register blueprints
register_blueprints(app)

if __name__ == '__main__':
    app.run(debug=True)