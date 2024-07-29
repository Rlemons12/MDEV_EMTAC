from flask import Blueprint, render_template, request, redirect, url_for
from emtacdb_fts import Session, User, UserLevel  # Import the User model and Session
from datetime import datetime
from sqlalchemy import Column, Integer, String, Date
from werkzeug.security import generate_password_hash


create_user_bp = Blueprint('create_user_bp', __name__)

@create_user_bp.route('/create_user', methods=['GET'])
def create_user():
    return render_template('create_user.html')

@create_user_bp.route('/submit_user_creation', methods=['POST'])
def submit_user_creation():
    # Extract form data
    employee_id = request.form['employee_id']
    first_name = request.form['first_name']
    last_name = request.form['last_name']
    current_shift = request.form['current_shift']
    primary_area = request.form['primary_area']
    age = request.form.get('age', None)  # Optional
    education_level = request.form.get('education_level', None)  # Optional
    start_date_str = request.form.get('start_date', None)  # Optional

    # Convert the start_date to a Python date object or None if not provided
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d') if start_date_str else None

    # Create a new User object with default user level STANDARD
    new_user = User(
        employee_id=employee_id,
        first_name=first_name,
        last_name=last_name,
        current_shift=current_shift,
        primary_area=primary_area,
        age=age,
        education_level=education_level,
        start_date=start_date,
        user_level=UserLevel.STANDARD  # Default user level
    )

    # Set the password using the set_password method
    new_user.set_password(request.form['password'])

    # Add the new user to the session
    session = Session()
    session.add(new_user)

    # Commit the session to save the user data to the database
    session.commit()
    session.close()

    return redirect(url_for('create_user_bp.create_user'))
