import logging
import os
from werkzeug.utils import secure_filename
from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file
from modules.emtacdb.emtacdb_fts import Part  # Assuming you have a session manager
from sqlalchemy.exc import IntegrityError
from modules.configuration.config_env import DatabaseConfig
from modules.emtacdb.emtacdb_fts import PartsPositionImageAssociation, Image, Position
from modules.configuration.config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS,DATABASE_DIR, BASE_DIR
from blueprints.bill_of_materials import update_part_bp
from modules.configuration.log_config import logger


@update_part_bp.route('/edit_part/<int:part_id>', methods=['GET', 'POST'])
def edit_part(part_id):
    logger.info(f'Starting edit_part function for part_id: {part_id}')
    db_session = DatabaseConfig().get_main_session()
    logger.debug(f'Database session obtained for editing part: {part_id}')

    # Fetch the part based on part_id
    part = db_session.query(Part).filter_by(id=part_id).first()
    if part:
        logger.info(f'Found part for editing: {part.part_number} (ID: {part_id})')
    else:
        logger.warning(f'Part with ID {part_id} not found')
        flash("Part not found.", "error")
        return redirect(url_for('update_part_bp.search_part'))

    # Get existing images associated with this part
    try:
        part_images = db_session.query(Image).join(
            PartsPositionImageAssociation,
            PartsPositionImageAssociation.image_id == Image.id
        ).filter(
            PartsPositionImageAssociation.part_id == part_id
        ).all()
        logger.info(f'Retrieved {len(part_images)} images associated with part {part_id}')
    except Exception as e:
        logger.error(f'Error retrieving images for part {part_id}: {str(e)}')
        part_images = []

    # Get all positions for dropdown
    try:
        positions = db_session.query(Position).all()
        logger.debug(f'Retrieved {len(positions)} positions for dropdown')
    except Exception as e:
        logger.error(f'Error retrieving positions: {str(e)}')
        positions = []

    if request.method == 'POST':
        logger.info(f'Processing POST request for part {part_id}')

        # Log form data (excluding file contents for security)
        form_data = {k: v for k, v in request.form.items() if k != 'part_image'}
        logger.debug(f'Form data received: {form_data}')

        # Update part attributes from form input
        old_values = {
            'part_number': part.part_number,
            'name': part.name,
            'oem_mfg': part.oem_mfg,
            'model': part.model,
            'class_flag': part.class_flag,
            'ud6': part.ud6,
            'type': part.type
        }

        part.part_number = request.form.get('part_number')
        part.name = request.form.get('name')
        part.oem_mfg = request.form.get('oem_mfg')
        part.model = request.form.get('model')
        part.class_flag = request.form.get('class_flag')
        part.ud6 = request.form.get('ud6')
        part.type = request.form.get('type')
        part.notes = request.form.get('notes')
        part.documentation = request.form.get('documentation')

        # Log changes to part attributes
        new_values = {
            'part_number': part.part_number,
            'name': part.name,
            'oem_mfg': part.oem_mfg,
            'model': part.model,
            'class_flag': part.class_flag,
            'ud6': part.ud6,
            'type': part.type
        }

        for key, old_value in old_values.items():
            new_value = new_values[key]
            if old_value != new_value:
                logger.info(f'Updated part {part_id} {key}: "{old_value}" -> "{new_value}"')

        try:
            # Handle image upload if a file was submitted
            if 'part_image' in request.files and request.files['part_image'].filename != '':
                uploaded_file = request.files['part_image']
                logger.info(f'Image upload detected for part {part_id}: {uploaded_file.filename}')

                # Check if the file extension is allowed
                if not '.' in uploaded_file.filename or \
                        uploaded_file.filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
                    logger.warning(f'Invalid file type attempted: {uploaded_file.filename}')
                    flash("File type not allowed. Please upload jpg, jpeg, png, or gif files only.", "error")
                    return render_template('bill_of_materials/bom_partials/edit_part.html', part=part,
                                           part_images=part_images,
                                           positions=positions)

                # Ensure the filename is secure
                filename = secure_filename(uploaded_file.filename)
                logger.debug(f'Secured filename: {filename}')

                # Create upload folder for parts - use a folder structure compatible with your existing code
                upload_folder = os.path.join(UPLOAD_FOLDER, 'parts')
                logger.debug(f'Upload folder path: {upload_folder}')

                # Create the directory if it doesn't exist
                if not os.path.exists(upload_folder):
                    logger.info(f'Creating upload directory: {upload_folder}')
                    os.makedirs(upload_folder)

                # Save the file with absolute path
                abs_file_path = os.path.join(upload_folder, filename)
                uploaded_file.save(abs_file_path)
                logger.info(f'Image saved to: {abs_file_path}')

                # Calculate relative path for storage in database (consistent with add_image_to_db)
                rel_file_path = os.path.relpath(abs_file_path, BASE_DIR)
                logger.debug(f'Relative file path for database: {rel_file_path}')

                # Create a new Image record with relative path
                image_title = request.form.get('image_title', f"Image for {part.part_number}")
                image_description = request.form.get('image_description', f"Image for part {part.part_number}")

                # Check if an image with the same title and description already exists
                existing_image = db_session.query(Image).filter(
                    and_(Image.title == image_title, Image.description == image_description)
                ).first()

                if existing_image is not None and existing_image.file_path == rel_file_path:
                    logger.info(f"Image with the same title, description, and file path already exists: {image_title}")
                    new_image = existing_image
                else:
                    # Create new image
                    new_image = Image(
                        title=image_title,
                        description=image_description,
                        file_path=rel_file_path  # Store relative path in the database
                    )
                    db_session.add(new_image)
                    db_session.flush()  # Flush to get the image ID
                    logger.info(f'Created new image record with ID: {new_image.id}')

                # Get the position ID from the form
                position_id = request.form.get('position_id')
                if position_id:
                    logger.debug(f'Position selected for image: {position_id}')
                else:
                    logger.debug('No position selected for image')

                # Create the association
                association = PartsPositionImageAssociation(
                    part_id=part_id,
                    position_id=position_id,
                    image_id=new_image.id
                )

                db_session.add(association)
                logger.info(
                    f'Created association between part {part_id}, position {position_id}, and image {new_image.id}')

            # Handle image removal if requested
            if 'remove_image' in request.form:
                image_ids_to_remove = request.form.getlist('remove_image')
                logger.info(f'Request to remove {len(image_ids_to_remove)} images: {image_ids_to_remove}')

                for image_id in image_ids_to_remove:
                    # Find the association
                    association = db_session.query(PartsPositionImageAssociation).filter_by(
                        part_id=part_id,
                        image_id=image_id
                    ).first()

                    if association:
                        logger.debug(f'Found association for part {part_id} and image {image_id}')
                        db_session.delete(association)
                        logger.info(f'Deleted association for part {part_id} and image {image_id}')

                        # Optionally, also delete the image if it's not associated with any other parts
                        image_associations = db_session.query(PartsPositionImageAssociation).filter_by(
                            image_id=image_id
                        ).count()

                        if image_associations <= 1:  # 1 because we haven't committed the deletion yet
                            image = db_session.query(Image).filter_by(id=image_id).first()
                            if image:
                                logger.debug(f'Image {image_id} has no other associations, preparing to delete')
                                # Delete the file from the filesystem - handle absolute path conversion
                                abs_file_path = os.path.join(BASE_DIR, image.file_path)
                                if os.path.exists(abs_file_path):
                                    logger.debug(f'Deleting image file: {abs_file_path}')
                                    os.remove(abs_file_path)
                                db_session.delete(image)
                                logger.info(f'Deleted image record with ID: {image_id}')
                    else:
                        logger.warning(f'No association found for part {part_id} and image {image_id}')

            db_session.commit()
            logger.info(f'Successfully committed all changes for part {part_id}')
            flash("Part updated successfully!", "success")
            return redirect(url_for('update_part_bp.search_part'))

        except IntegrityError as ie:
            db_session.rollback()
            logger.error(f'IntegrityError during part update: {str(ie)}')
            flash("Part number must be unique.", "error")
        except Exception as e:
            db_session.rollback()
            logger.error(f'Unexpected error during part update: {str(e)}', exc_info=True)
            flash(f"An error occurred: {str(e)}", "error")

    logger.debug(f'Rendering edit_part template for part {part_id}')
    return render_template('bill_of_materials/bom_partials/edit_part.html', part=part, part_images=part_images,
                           positions=positions)


# Add route to serve images directly if needed
@update_part_bp.route('/part_image/<int:image_id>')
def serve_part_image(image_id):
    logger.info(f"Attempting to serve image with ID: {image_id}")
    db_session = DatabaseConfig().get_main_session()

    try:
        image = db_session.query(Image).filter_by(id=image_id).first()
        if image:
            logger.debug(f"Image found: {image.title}, File path: {image.file_path}")
            file_path = os.path.join(BASE_DIR, image.file_path)
            if os.path.exists(file_path):
                logger.info(f"Serving file: {file_path}")
                return send_file(file_path, mimetype='image/jpeg')
            else:
                logger.error(f"File not found: {file_path}")
                return "Image file not found", 404
        else:
            logger.error(f"Image not found with ID: {image_id}")
            return "Image not found", 404
    except Exception as e:
        logger.error(f"An error occurred while serving the image: {e}")
        return "Internal Server Error", 500

# Route: Search Part
@update_part_bp.route('/search_part', methods=['GET'])
def search_part():
    logger.info(f'start to search part')
    db_session = DatabaseConfig().get_main_session()
    search_query = request.args.get('search_query', '')

    if search_query:
        # Search for parts matching the query
        part = db_session.query(Part).filter(
            (Part.part_number.like(f'%{search_query}%')) |
            (Part.name.like(f'%{search_query}%')) |
            (Part.oem_mfg.like(f'%{search_query}%')) |
            (Part.model.like(f'%{search_query}%'))
        ).first()  # Get the first matching part

        if part:
            # Get images associated with this part
            part_images = db_session.query(Image).join(
                PartsPositionImageAssociation,
                PartsPositionImageAssociation.image_id == Image.id
            ).filter(
                PartsPositionImageAssociation.part_id == part.id
            ).all()

            # Get positions for dropdown
            positions = db_session.query(Position).all()

            return render_template('bill_of_materials/bom_partials/edit_part.html',
                                   part=part,
                                   part_images=part_images,
                                   positions=positions)
        else:
            flash("No parts found matching your search criteria.", "info")

    # If no query or no results, render the template with empty values
    # Initialize these variables to avoid the NameError
    part = None
    part_images = []
    positions = db_session.query(Position).all()

    return render_template('bill_of_materials/bom_partials/edit_part.html',
                           part=part,
                           part_images=part_images,
                           positions=positions)
