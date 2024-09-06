import logging
from flask import Blueprint, render_template, request, redirect, flash, session, url_for
from emtacdb_fts import ChatSession, User, engine, UserLevel
from datetime import datetime
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base
from flask_bcrypt import Bcrypt
from sqlalchemy import create_engine
from blueprints import DATABASE_URL
from werkzeug.security import check_password_hash

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create SQLAlchemy engine for the main database
logger.info(f"Creating SQLAlchemy engine with DATABASE_URL: {DATABASE_URL}")
engine = create_engine(DATABASE_URL)
Base = declarative_base()
db_session = scoped_session(sessionmaker(bind=engine))

login_bp = Blueprint('login_bp', __name__)

bcrypt = Bcrypt()  # Create a Flask-Bcrypt instance


@login_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        employee_id = request.form['employee_id']
        password = request.form['password']

        logger.info(f"Login attempt for employee_id: {employee_id}")

        try:
            # Check if a user with the provided employee_id exists
            user = db_session.query(User).filter_by(employee_id=employee_id).first()
            logger.debug(f"User found: {user}")

            if user:
                logger.debug(f"User {user.employee_id} found. Checking password.")
                if user.check_password_hash(password):
                    logger.info(f"User {user.employee_id} authenticated successfully.")

                    # Create a chat session
                    new_chat_session = ChatSession(
                        user_id=user.id,
                        start_time=datetime.now(),
                        last_interaction=datetime.now(),
                        session_data=[]
                    )
                    db_session.add(new_chat_session)
                    db_session.commit()

                    # Store user information in Flask session
                    session['user_id'] = user.id
                    session['employee_id'] = user.employee_id
                    session['first_name'] = user.first_name
                    session['last_name'] = user.last_name
                    session['primary_area'] = user.primary_area
                    session['age'] = user.age
                    session['education_level'] = user.education_level
                    session['start_date'] = user.start_date
                    session['user_level'] = user.user_level.name  # Store user level as name

                    # Redirect based on user level
                    if user.user_level == UserLevel.ADMIN:
                        logger.info(f"Redirecting admin user {user.employee_id} to admin dashboard.")
                        return redirect(url_for('admin_bp.admin_dashboard'))
                    elif user.user_level == UserLevel.STANDARD:
                        logger.info(f"Redirecting standard user {user.employee_id} to tsg_search_problems.")
                        return redirect(url_for('tsg_search_problems'))

                    # Redirect to the main index route
                    logger.info(f"Redirecting user {user.employee_id} to the index page.")
                    return redirect(url_for('index'))  # Redirect instead of rendering directly
                else:
                    logger.warning(f"Failed login attempt for user {employee_id}: Incorrect password.")
                    flash("Invalid username or password", 'error')
            else:
                logger.warning(f"Failed login attempt: User {employee_id} not found.")
                flash("Invalid username or password", 'error')

        except Exception as e:
            logger.error(f"An error occurred during login attempt for user {employee_id}: {e}")
            flash(f"An error occurred: {e}", 'error')

        finally:
            db_session.remove()  # Remove the SQLAlchemy session from the scoped_session
            logger.debug("SQLAlchemy session removed.")

    return render_template('login.html')


@login_bp.route('/logout')
def logout():
    logger.info("Logging out user.")

    # Clear all user-related session data
    session.pop('user_id', None)
    session.pop('employee_id', None)
    session.pop('first_name', None)
    session.pop('last_name', None)
    session.pop('primary_area', None)
    session.pop('age', None)
    session.pop('education_level', None)
    session.pop('start_date', None)
    session.pop('user_level', None)  # Clear user level

    logger.info("User session cleared. Redirecting to login page.")
    return redirect(url_for('login_bp.login'))

