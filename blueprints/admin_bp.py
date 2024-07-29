from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from emtacdb_fts import Session, User, UserLevel, AIModelConfig, ImageModelConfig, load_config_from_db, load_image_model_config_from_db
import config  # Assuming your configurations are in a config.py module

admin_bp = Blueprint('admin_bp', __name__)

@admin_bp.route('/admin')
def admin_dashboard():
    # Ensure only admins can access this page
    if session.get('user_level') != UserLevel.ADMIN.name:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('login_bp.login'))

    # Fetch all users from the database
    session_db = Session()
    users = session_db.query(User).all()
    session_db.close()

    # Fetch current model configurations from the database
    current_ai_model, current_embedding_model = load_config_from_db()
    current_image_model = load_image_model_config_from_db()

    return render_template('admin_dashboard.html', users=users, current_ai_model=current_ai_model, current_embedding_model=current_embedding_model, current_image_model=current_image_model)

@admin_bp.route('/change_user_level', methods=['POST'])
def change_user_level():
    # Ensure only admins can perform this action
    if session.get('user_level') != UserLevel.ADMIN.name:
        flash('You do not have permission to perform this action.', 'error')
        return redirect(url_for('login_bp.login'))

    user_id = request.form['user_id']
    new_user_level = request.form['user_level']

    session_db = Session()
    user = session_db.query(User).filter_by(id=user_id).first()
    if user:
        user.user_level = UserLevel(new_user_level)
        session_db.commit()
        flash('User level updated successfully.', 'success')
    else:
        flash('User not found.', 'error')

    session_db.close()
    return redirect(url_for('admin_bp.admin_dashboard'))

@admin_bp.route('/set_models', methods=['POST'])
def set_models():
    # Ensure only admins can perform this action
    if session.get('user_level') != UserLevel.ADMIN.name:
        flash('You do not have permission to perform this action.', 'error')
        return redirect(url_for('login_bp.login'))

    ai_model = request.form['ai_model']
    embedding_model = request.form['embedding_model']
    image_model = request.form['image_model']

    try:
        # Update the global configuration
        config.CURRENT_AI_MODEL = ai_model
        config.CURRENT_EMBEDDING_MODEL = embedding_model
        config.CURRENT_IMAGE_MODEL = image_model

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
        session_db.close()

        flash('Models updated successfully.', 'success')
    except Exception as e:
        flash(f'Error updating models: {e}', 'error')

    return redirect(url_for('admin_bp.admin_dashboard'))
