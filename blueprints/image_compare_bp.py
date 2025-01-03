from flask import Blueprint, jsonify, request, send_from_directory, send_file, flash
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import create_engine
import os
import numpy as np
from PIL import Image as PILImage
from werkzeug.utils import secure_filename
from modules.emtacdb.emtacdb_fts import Image, ImageEmbedding, load_image_model_config_from_db
from plugins.image_modules.image_models import get_image_model_handler
from modules.configuration.config import DATABASE_URL, DATABASE_PATH_IMAGES_FOLDER, DATABASE_DIR
import logging

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create a session factory
Session = scoped_session(sessionmaker(bind=engine))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

image_compare_bp = Blueprint('image_compare_bp', __name__)

# Load the current image model configuration from the database
CURRENT_IMAGE_MODEL = load_image_model_config_from_db()

# Instantiate the appropriate handler using the function from image_modules.py
image_handler = get_image_model_handler(CURRENT_IMAGE_MODEL)

@image_compare_bp.route('/upload_and_compare', methods=['POST'])
def upload_and_compare():
    logger.info('Received request to upload and compare image.')

    if 'query_image' not in request.files:
        logger.error('No file part in the request.')
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['query_image']
    if file.filename == '':
        logger.error('No selected file.')
        return jsonify({'error': 'No selected file'}), 400

    if file and image_handler.allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_folder = os.path.join(DATABASE_PATH_IMAGES_FOLDER, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)
        logger.info(f'File saved to {file_path}')

        try:
            image = PILImage.open(file_path).convert("RGB")
            embedding = image_handler.get_image_embedding(image)
        except Exception as e:
            logger.exception('Failed to process the uploaded image.')
            return jsonify({'error': 'Failed to process the uploaded image.'}), 500

        if embedding is None:
            logger.error('Failed to get image embedding.')
            return jsonify({'error': 'Failed to process the uploaded image.'}), 500

        logger.info('Comparing uploaded image with stored images.')

        try:
            session = Session()
            stored_images = session.query(Image).join(ImageEmbedding).all()
            similarities = []
            for stored_image in stored_images:
                stored_embedding = session.query(ImageEmbedding).filter_by(image_id=stored_image.id).first()
                stored_embedding_vector = np.frombuffer(stored_embedding.model_embedding, dtype=np.float32)
                similarity = float(np.dot(embedding, stored_embedding_vector) / (np.linalg.norm(embedding) * np.linalg.norm(stored_embedding_vector)))
                similarities.append({
                    'id': stored_image.id,
                    'title': stored_image.title,
                    'description': stored_image.description,
                    'file_path': os.path.basename(stored_image.file_path),
                    'similarity': similarity
                })

            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            top_matches = similarities[:5]

            results = [
                {
                    'id': match['id'],
                    'title': match['title'],
                    'description': match['description'],
                    'file_path': match['file_path'],
                    'similarity': match['similarity']
                }
                for match in top_matches
            ]

            logger.info('Successfully processed and compared the image.')
            return jsonify({'image_similarity_search': results})
        except Exception as e:
            logger.exception('An error occurred during the comparison process.')
            return jsonify({'error': 'An error occurred during the comparison process.'}), 500
        finally:
            session.close()
    else:
        logger.error('File type not allowed.')
        return jsonify({'error': 'File type not allowed'}), 400

@image_compare_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    logger.info(f'Serving file {filename} from uploads.')
    return send_from_directory(DATABASE_PATH_IMAGES_FOLDER, filename)

@image_compare_bp.route('/serve_image/<int:image_id>')
def serve_image_route(image_id):
    logger.debug(f"Request to serve image with ID: {image_id}")
    with Session() as session:
        try:
            return serve_image(session, image_id)
        except Exception as e:
            logger.error(f"Error serving image {image_id}: {e}")
            flash(f"Error serving image {image_id}", "error")
            return "Image not found", 404

def serve_image(session, image_id):
    logger.info(f"Attempting to serve image with ID: {image_id}")
    try:
        image = session.query(Image).filter_by(id=image_id).first()
        if image:
            logger.debug(f"Image found: {image.title}, File path: {image.file_path}")
            file_path = os.path.join(DATABASE_DIR, image.file_path)
            if os.path.exists(file_path):
                logger.info(f"Serving file: {file_path}")
                return send_file(file_path, mimetype='image/jpeg', as_attachment=False)
            else:
                logger.error(f"File not found: {file_path}")
                return "Image file not found", 404
        else:
            logger.error(f"Image not found with ID: {image_id}")
            return "Image not found", 404
    except Exception as e:
        logger.error(f"An error occurred while serving the image: {e}")
        return "Internal Server Error", 500
