import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import logging
from flask import Blueprint, jsonify, request, render_template
from PIL import Image as PILImage, ImageFile
from config import DATABASE_PATH_IMAGES_FOLDER
from plugins.image_models import ImageHandler  # Update import to the new location

ImageFile.LOAD_TRUNCATED_IMAGES = True

folder_image_embedding_bp = Blueprint('folder_image_embedding_bp', __name__)

image_handler = ImageHandler()  # Initialize the ImageHandler

def process_and_store_images(folder, model_name="no_model"):
    session = image_handler.Session()
    for filename in os.listdir(folder):
        if image_handler.allowed_file(filename, model_name):
            source_file_path = os.path.join(folder, filename)
            dest_file_path = os.path.join(DATABASE_PATH_IMAGES_FOLDER, filename)
            
            try:
                image = PILImage.open(source_file_path).convert("RGB")
                if not image_handler.is_valid_image(image, model_name):
                    logging.info(f"Skipping {filename}: Image does not meet the required dimensions or aspect ratio.")
                    continue
                
                embedding = image_handler.get_image_embedding(image, model_name)
                if embedding is not None:
                    # Save the image to the destination folder
                    os.makedirs(DATABASE_PATH_IMAGES_FOLDER, exist_ok=True)
                    image.save(dest_file_path)
                    image_handler.store_image_metadata(session, filename, "Auto-generated description", dest_file_path, embedding, model_name)
            except Exception as e:
                logging.error(f"Failed to process {filename}: {e}")
    session.close()

@folder_image_embedding_bp.route('/process_folder', methods=['POST'])
def process_folder():
    folder_path = request.form.get('folder_path')
    if not folder_path or not os.path.isdir(folder_path):
        return jsonify({'error': 'Invalid folder path.'}), 400

    model_name = request.form.get('model_name', 'no_model')  # Default to 'no_model'
    process_and_store_images(folder_path, model_name)
    return jsonify({'success': 'Images processed and stored.'}), 200

@folder_image_embedding_bp.route('/compare_image', methods=['GET'])
def upload_and_compare_form():
    return render_template('upload_and_compare.html')
