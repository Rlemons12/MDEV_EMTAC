import logging
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from modules.emtacdb.emtacdb_fts import Session, User, UserComments, UserLevel, AIModelConfig, ImageModelConfig, load_config_from_db, \
    load_image_model_config_from_db
from sqlalchemy.orm import subqueryload
from modules.configuration import config

# Set up logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for detailed logging
logger = logging.getLogger(__name__)

admin_bp = Blueprint('admin_bp', __name__)


@admin_bp.route('/admin_dashboard')
def admin_dashboard():
    logger.debug("Admin dashboard accessed by user: %s", session.get('user_id'))

    if session.get('user_level') != UserLevel.ADMIN.name:
        logger.warning("Unauthorized access attempt by user: %s", session.get('user_id'))
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('login_bp.login'))

    session_db = None
    try:
        # Fetch all users and comments from the database
        session_db = Session()
        logger.debug("Fetching users and comments from the database.")

        users = session_db.query(User).all()
        comments = session_db.query(UserComments).options(subqueryload(UserComments.user)).all()

        # Fetch current model configurations
        current_ai_model, current_embedding_model = load_config_from_db()
        current_image_model = load_image_model_config_from_db()

        logger.info("Fetched %d users and %d comments. AI Model: %s, Embedding Model: %s, Image Model: %s",
                    len(users), len(comments), current_ai_model, current_embedding_model, current_image_model)

    except Exception as e:
        logger.error("Error loading admin dashboard data: %s", e)
        flash(f'Error loading dashboard: {e}', 'error')
        return redirect(url_for('admin_bp.admin_dashboard'))

    finally:
        if session_db:
            session_db.close()
            logger.debug("Database session closed.")

    return render_template(
        'admin_dashboard.html',
        users=users,
        comments=comments,
        current_ai_model=current_ai_model,
        current_embedding_model=current_embedding_model,
        current_image_model=current_image_model
    )


@admin_bp.route('/change_user_level', methods=['POST'])
def change_user_level():
    logger.debug("User level change requested by user: %s", session.get('user_id'))

    if session.get('user_level') != UserLevel.ADMIN.name:
        logger.warning("Unauthorized attempt to change user level by user: %s", session.get('user_id'))
        flash('You do not have permission to perform this action.', 'error')
        return redirect(url_for('login_bp.login'))

    user_id = request.form['user_id']
    new_user_level = request.form['user_level']
    session_db = None

    try:
        session_db = Session()
        logger.debug("Fetching user with ID: %s", user_id)

        user = session_db.query(User).filter_by(id=user_id).first()
        if user:
            user.user_level = UserLevel(new_user_level)
            session_db.commit()
            flash('User level updated successfully.', 'success')
            logger.info("User level changed for user: %s to %s", user_id, new_user_level)
        else:
            flash('User not found.', 'error')
            logger.warning("User not found for user ID: %s", user_id)

    except Exception as e:
        if session_db:
            session_db.rollback()
        logger.error("Error changing user level: %s", e)
        flash(f'Error changing user level: {e}', 'error')

    finally:
        if session_db:
            session_db.close()
            logger.debug("Database session closed.")

    return redirect(url_for('admin_bp.admin_dashboard'))


@admin_bp.route('/set_models', methods=['POST'])
def set_models():
    logger.debug("Model change requested by user: %s", session.get('user_id'))

    if session.get('user_level') != UserLevel.ADMIN.name:
        logger.warning("Unauthorized attempt to change models by user: %s", session.get('user_id'))
        flash('You do not have permission to perform this action.', 'error')
        return redirect(url_for('login_bp.login'))

    ai_model = request.form['ai_model']
    embedding_model = request.form['embedding_model']
    image_model = request.form['image_model']
    session_db = None

    try:
        # Update the global configuration
        config.CURRENT_AI_MODEL = ai_model
        config.CURRENT_EMBEDDING_MODEL = embedding_model
        config.CURRENT_IMAGE_MODEL = image_model
        logger.debug("New model configurations: AI Model: %s, Embedding Model: %s, Image Model: %s",
                     ai_model, embedding_model, image_model)

        # Save the new configuration to the database
        session_db = Session()

        ai_model_config = session_db.query(AIModelConfig).filter_by(key='CURRENT_AI_MODEL').first()
        embedding_model_config = session_db.query(AIModelConfig).filter_by(key='CURRENT_EMBEDDING_MODEL').first()
        image_model_config = session_db.query(ImageModelConfig).filter_by(key='CURRENT_IMAGE_MODEL').first()

        if ai_model_config:
            ai_model_config.value = ai_model
        else:
            ai_model_config = AIModelConfig(key='CURRENT_AI_MODEL', value=ai_model)
            session_db.add(ai_model_config)

        if embedding_model_config:
            embedding_model_config.value = embedding_model
        else:
            embedding_model_config = AIModelConfig(key='CURRENT_EMBEDDING_MODEL', value=embedding_model)
            session_db.add(embedding_model_config)

        if image_model_config:
            image_model_config.value = image_model
        else:
            image_model_config = ImageModelConfig(key='CURRENT_IMAGE_MODEL', value=image_model)
            session_db.add(image_model_config)

        session_db.commit()
        logger.info("Model configurations updated successfully.")
        flash('Models updated successfully.', 'success')

    except Exception as e:
        if session_db:
            session_db.rollback()
        logger.error("Error updating models: %s", e)
        flash(f'Error updating models: {e}', 'error')

    finally:
        if session_db:
            session_db.close()
            logger.debug("Database session closed.")

    return redirect(url_for('admin_bp.admin_dashboard'))
