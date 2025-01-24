# ai_emtac.py

import sys
import os
import logging
import webbrowser
from threading import Timer
from flask import Flask, render_template, flash
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

# Add current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import models, functions, and utilities from your application modules
from modules.emtacdb.emtacdb_fts import (
    UserLevel, load_config_from_db
)
from modules.emtacdb.utlity.main_database.database import serve_image
from blueprints import register_blueprints
from modules.emtacdb.utlity.revision_database.event_listeners import register_event_listeners
from modules.configuration.config import UPLOAD_FOLDER, DATABASE_URL, REVISION_CONTROL_DB_PATH
from utilities.auth_utils import requires_roles
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import initial_log_cleanup

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize database engines
engine = create_engine(DATABASE_URL)
revision_control_engine = create_engine(f'sqlite:///{REVISION_CONTROL_DB_PATH}')


# Log SQL queries executed by SQLAlchemy
@event.listens_for(engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    logger.debug(f"Executing query: {statement}")
    logger.debug(f"With parameters: {parameters}")


# Create session factories
SessionFactory = sessionmaker(bind=engine)
RevisionControlSessionFactory = sessionmaker(bind=revision_control_engine)


def create_app():
    app = Flask(__name__)

    # Set the secret key for session encryption
    app.secret_key = '1234'  # You should use a more secure and random secret key in production

    # Set the upload folder in the app's configuration
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    # Initialize and set db_config
    db_config = DatabaseConfig()
    app.config['db_config'] = db_config

    # Set the session lifetime (e.g., 1 day)
    #app.permanent_session_lifetime = timedelta(days=1)

    # Register blueprints and event listeners
    register_blueprints(app)
    register_event_listeners()
    # Track if session has been cleared after app start
    app.has_cleared_session = True

    from flask import session, request, redirect, url_for

    @app.before_request
    def global_login_check():
        # List of routes that do not require login
        allowed_routes = [
            'login_bp.login',
            'login_bp.logout',
            'static',
            'create_user_bp.create_user',
            'create_user_bp.submit_user_creation',
            'tsg_search_parts_bp.tsg_search_parts',
            'trouble_shooting_guide_bp.update_problem_solution',
            'trouble_shooting_guide_edit_update.troubleshooting_guide_edit_update',
            'trouble_shooting_guide_bp.search_documents',
            'pst_troubleshooting_solution.get_solutions',
            'tool_routes.tool_search',
            'tool_routes.get_tool_positions',
            'tool_routes.get_tool_packages',
            'tool_routes.submit_tool_data',
            'tool_routes.get_manufacturers',
            'tool_routes.get_categories',
            'image_bp.add_image',
            'image_bp.upload_image',
        ]

        # Check if request.endpoint is None (which can happen for invalid requests)
        if request.endpoint is None:
            return

        # If user is not logged in and the endpoint is not in the allowed routes, redirect to login
        if 'user_id' not in session and request.endpoint not in allowed_routes:
            return redirect(url_for('login_bp.login'))

        # Make the session permanent to extend it on every request (useful for maintaining sessions)
        session.permanent = True

    # Define routes
    @app.route('/')
    def index():
        session.permanent = False  # Make the session non-permanent
        user_id = session.get('user_id', '')

        # Ensure the session retrieves the correct user_level value
        user_level = session.get('user_level', UserLevel.STANDARD.value)

        if not user_id:
            return redirect(url_for('login_bp.login'))  # Redirect to login page if user is not logged in

        # Load the current AI and embedding models from the database
        current_ai_model, current_embedding_model = load_config_from_db()

        return render_template('index.html',
                               current_ai_model=current_ai_model,
                               current_embedding_model=current_embedding_model,
                               user_level=user_level)  # Pass user_level to the template

    @app.route('/upload_search_database')
    def upload_image_page():
        session.permanent = False  # Make the session non-permanent
        total_pages = 1
        page = 1
        return render_template('upload_search_database.html', total_pages=total_pages, page=page)

    @app.route('/success')
    def upload_success():
        session.permanent = False  # Make the session non-permanent
        return render_template('success.html')

    @app.route('/view_pdf_by_title/<string:title>')
    def view_pdf_by_title_route(title):
        session.permanent = False  # Make the session non-permanent
        return view_pdf_by_title(title)

    @app.route('/serve_image/<int:image_id>')
    def serve_image_route(image_id):
        logger.debug(f"Request to serve image with ID: {image_id}")
        with SessionFactory() as session:
            try:
                return serve_image(session, image_id)
            except Exception as e:
                logger.error(f"Error serving image {image_id}: {e}")
                flash(f"Error serving image {image_id}", "error")
                return "Image not found", 404

    @app.route('/document_success')
    def document_upload_success():
        session.permanent = False  # Make the session non-permanent
        return render_template('success.html')

    @app.route('/troubleshooting_guide')
    @requires_roles(UserLevel.ADMIN.value,UserLevel.LEVEL_III.value)
    def troubleshooting_guide():
        session.permanent = False  # Make the session non-permanent
        return render_template('troubleshooting_guide.html')

    @app.route('/tsg_search_problems')
    def tsg_search_problems():
        session.permanent = False  # Make the session non-permanent
        return render_template('tsg_search_problems.html')

    @app.route('/search_bill_of_material', methods=['GET'])
    def search_bill_of_material():
        return render_template('search_bill_of_material.html')

    @app.route('/bill_of_materials')
    def bill_of_materials():
        logger.debug("Rendering bill_of_materials page.")
        return render_template('bill_of_materials.html')

    @app.route('/position_data_assignment')
    def position_data_assignment():
        logger.debug("Rendering position_data_assignment page.")
        return render_template('position_data_assignment/position_data_assignment.html')

    """@app.route('/pst_troubleshooting')
    def pst_troubleshooting_page():
        session.permanent = False
        logger.debug("Rendering pst Troubleshooting page.")
        return render_template('pst_troubleshooting.html')"""

    @app.errorhandler(403)
    def forbidden(e):
        return render_template('403.html'), 403

    for rule in app.url_map.iter_rules():
        print(rule)

    return app

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')


if __name__ == '__main__':
    print(f'Perform initial log cleanup (compress old logs and delete old backups)')# Perform initial log cleanup (compress old logs and delete old backups)
    initial_log_cleanup()

    # Optional: Open the browser after a slight delay (1 second)
    Timer(1, open_browser).start()

    # Create and run your application
    app = create_app()
    app.run(debug=True)
