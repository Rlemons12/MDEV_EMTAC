# auth_utils.py
from flask import session, redirect, url_for, render_template, flash
from functools import wraps
import logging

# Set up logging for the auth_utils module
logger = logging.getLogger(__name__)

# Define login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            logger.debug("User not logged in, redirecting to login page.")
            flash("Please log in to access this page.", "warning")
            return redirect(url_for('login_bp.login'))
        logger.debug(f"User {session.get('user_id')} is logged in.")
        return f(*args, **kwargs)
    return decorated_function

from flask import redirect, url_for

from flask import redirect, url_for

def logout():
    # Clear session variables related to user authentication
    session.pop('user_id', None)
    session.pop('employee_id', None)
    session.pop('name', None)
    session.pop('primary_area', None)
    session.pop('age', None)
    session.pop('education_level', None)
    session.pop('start_date', None)

    # Debug statements
    print("Session variables cleared.")

    # Redirect the user to the login page
    print("Redirecting to login page.")
    return redirect(url_for('login_bp.login'))



