import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import numpy as np
from flask import Blueprint, request, jsonify, render_template, send_from_directory
from PIL import Image as PILImage, ImageFile
from werkzeug.utils import secure_filename
from config import DATABASE_PATH_IMAGES_FOLDER
from plugins.image_models import ImageHandler  # Update import to the new location
from emtacdb_fts import Image, ImageEmbedding  # Import ImageEmbedding

image_compare_bp = Blueprint('image_compare_bp', __name__)
image_handler = ImageHandler()  # Initialize the ImageHandler

# image_compare_bp.py
@image_compare_bp.route('/upload_and_compare', methods=['POST'])
def upload_and_compare():
    if 'query_image' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['query_image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    model_name = request.form.get('model_name', 'clip')  # Default to 'clip'

    if file and image_handler.allowed_file(file.filename, model_name):
        filename = secure_filename(file.filename)
        upload_folder = os.path.join('static', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)

        # Process the uploaded image
        image = PILImage.open(file_path).convert("RGB")
        embedding = image_handler.get_image_embedding(image, model_name)
        
        if embedding is None:
            return jsonify({'error': 'Failed to process the uploaded image.'}), 500

        # Compare with stored images
        session = image_handler.Session()
        stored_images = session.query(Image).all()
        similarities = []
        for stored_image in stored_images:
            stored_embedding = session.query(ImageEmbedding).filter_by(image_id=stored_image.id).first()
            if stored_embedding and stored_embedding.model_name == model_name:
                stored_embedding = np.frombuffer(stored_embedding.model_embedding, dtype=np.float32)
                similarity = np.dot(embedding, stored_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(stored_embedding))
                similarities.append({
                    'id': stored_image.id,
                    'title': stored_image.title,
                    'description': stored_image.description,
                    'file_path': os.path.basename(stored_image.file_path),  # Use only the filename
                    'similarity': similarity
                })
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_matches = similarities[:5]  # Get top 5 matches

        return render_template('upload_and_compare.html', results=top_matches)
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@image_compare_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(DATABASE_PATH_IMAGES_FOLDER, filename)

@image_compare_bp.route('/compare_image', methods=['GET'])
def upload_and_compare_form():
    return render_template('upload_and_compare.html')
