# auth_utils.py
from flask import session, redirect, url_for, render_template
from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_bp.login'))  # Adjust to your actual login route
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



