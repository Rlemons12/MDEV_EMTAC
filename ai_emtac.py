# ai_emtac.py
from datetime import datetime
import sys
import os
import webbrowser
import socket
from threading import Timer
from flask import Flask, session, request, redirect, url_for, current_app, render_template
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from modules.emtacdb.emtacdb_fts import UserLogin
from utilities.custom_jinja_filters import register_jinja_filters

# Add current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import models, functions, and utilities from your application modules
from modules.emtacdb.emtacdb_fts import (UserLevel)
from modules.emtacdb.utlity.main_database.database import serve_image
from blueprints import register_blueprints
from modules.emtacdb.utlity.revision_database.event_listeners import register_event_listeners
from modules.configuration.config import UPLOAD_FOLDER, DATABASE_URL, REVISION_CONTROL_DB_PATH
from utilities.auth_utils import requires_roles
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import initial_log_cleanup, logger

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

def get_local_ip():
    """
    Dynamically retrieves the local IP address by creating a temporary socket
    to a public DNS server (8.8.8.8).
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
    except Exception:
        local_ip = '127.0.0.1'
    finally:
        s.close()
    return local_ip

def open_browser():
    port = int(os.environ.get('PORT', 5000))
    ip = get_local_ip()
    url = f'http://{ip}:{port}/'
    logger.info(f"Opening browser at {url}")
    webbrowser.open_new(url)

def create_app():
    app = Flask(__name__)

    # Set the secret key for session encryption
    app.secret_key = '1234'  # Replace with a secure secret key for production

    # Register custom Jinja filters
    register_jinja_filters(app)

    # Set the upload folder in the app's configuration
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    # Initialize and set db_config
    db_config = DatabaseConfig()
    app.config['db_config'] = db_config

    # Initialize models configuration
    from plugins.ai_modules import ModelsConfig
    ModelsConfig.initialize_models_config_table()
    logger.info("Models configuration initialized during app startup")
    print("Models configuration initialized during app startup")

    # Register blueprints and event listeners
    register_blueprints(app)
    register_event_listeners()
    app.has_cleared_session = True

    from flask import session, request, redirect, url_for

    @app.before_request
    def global_login_check():
        endpoint = request.endpoint
        current_app.logger.debug(f"→ incoming endpoint: {endpoint}")

        # Only proceed with the activity tracking if user is logged in and has a login record
        if 'user_id' in session and 'login_record_id' in session:
            try:
                # Use your existing SessionFactory to create a session
                with SessionFactory() as session_db:
                    login_record = session_db.query(UserLogin).get(session['login_record_id'])
                    if login_record and login_record.is_active:
                        login_record.last_activity = datetime.utcnow()
                        session_db.commit()
            except Exception as e:
                logger.error(f"Error updating activity timestamp: {e}")

        # Continue with login check
        allowed_routes = [
            'login_bp.login',
            'login_bp.logout',
            'static',  # Allow static files
            'create_user_bp.create_user',
            'create_user_bp.submit_user_creation'
        ]

        if request.endpoint is None:
            return

        if 'user_id' not in session and request.endpoint not in allowed_routes:
            return redirect(url_for('login_bp.login'))

        session.permanent = True

    @app.route('/')
    def index():
        session.permanent = False
        user_id = session.get('user_id', '')
        user_level = session.get('user_level', UserLevel.STANDARD.value)
        if not user_id:
            return redirect(url_for('login_bp.login'))

        # Update this line to use ModelsConfig
        from plugins.ai_modules import ModelsConfig
        current_ai_model, current_embedding_model = ModelsConfig.load_config_from_db()

        return render_template('index.html',
                               current_ai_model=current_ai_model,
                               current_embedding_model=current_embedding_model,
                               user_level=user_level)

    @app.route('/upload_search_database')
    def upload_image_page():
        session.permanent = False
        total_pages = 1
        page = 1
        return render_template('upload_search_database/upload_search_database.html', total_pages=total_pages, page=page)

    @app.route('/success')
    def upload_success():
        session.permanent = False
        return render_template('success.html')

    @app.route('/view_pdf_by_title/<string:title>')
    def view_pdf_by_title_route(title):
        session.permanent = False
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
        session.permanent = False
        return render_template('success.html')

    @app.route('/troubleshooting_guide')
    @requires_roles(UserLevel.ADMIN.value, UserLevel.LEVEL_III.value)
    def troubleshooting_guide():
        session.permanent = False
        return render_template('troubleshooting_guide.html')

    @app.route('/tsg_search_problems')
    def tsg_search_problems():
        session.permanent = False
        return render_template('tsg_search_problems.html')

    @app.route('/search_bill_of_material', methods=['GET'])
    def search_bill_of_material():
        return render_template('search_bill_of_material.html')

    @app.route('/bill_of_materials')
    def bill_of_materials():
        logger.debug("Rendering bill_of_materials page.")
        return render_template('bill_of_materials/bill_of_materials.html')

    @app.route('/position_data_assignment')
    def position_data_assignment():
        logger.debug("Rendering position_data_assignment page.")
        return render_template('position_data_assignment/position_data_assignment.html')

    @app.errorhandler(403)
    def forbidden(e):
        return render_template('403.html'), 403

    for rule in app.url_map.iter_rules():
        print(rule)

    # Log configuration details after the app is created.
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    ip = get_local_ip()
    url = f'http://{ip}:{port}/'
    logger.info(f"Starting application on host: {host}, port: {port}")
    logger.info(f"Accessible at: {url}")
    print(f"Starting application on host: {host}, port: {port}")
    print(f"Accessible at: {url}")

    return app

if __name__ == '__main__':
    """Must runt in terminal python ai_emtac.py to allow remote access to local network"""

    print('Perform initial log cleanup (compress old logs and delete old backups)')
    initial_log_cleanup()

    # Read configuration from environment variables
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', '1') == '1'  # Default to debug mode if not set

    ip = get_local_ip()
    url = f'http://{ip}:{port}/'

    # Log the configuration details when running the script directly
    logger.info(f"Starting application on host: {host}, port: {port}")
    logger.info(f"Accessible at: {url}")
    print(f"Starting application on host: {host}, port: {port}")
    print(f"Accessible at: {url}")

    # Optional: Open the browser after a slight delay (1 second)
    Timer(1, open_browser).start()

    # Create and run the application
    app = create_app()
    app.run(host=host, port=port, debug=debug_mode)
    # To disable the auto-reloader (if duplicate processes occur), use:
    # app.run(host=host, port=port, debug=debug_mode, use_reloader=False)
