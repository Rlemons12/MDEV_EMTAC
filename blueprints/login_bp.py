from flask import Blueprint, render_template, request, redirect, flash, session, url_for
from emtacdb_fts import ChatSession, User, engine, UserLevel
from datetime import datetime
from sqlalchemy.orm import sessionmaker, scoped_session
from flask_bcrypt import Bcrypt
from sqlalchemy import create_engine
from blueprints import DATABASE_URL
from werkzeug.security import check_password_hash

# Create an SQLAlchemy engine with the corrected URL
engine = create_engine(DATABASE_URL)

# Create an SQLAlchemy session using scoped_session
db_session = scoped_session(sessionmaker(bind=engine))

login_bp = Blueprint('login_bp', __name__)

bcrypt = Bcrypt()  # Create a Flask-Bcrypt instance

@login_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        employee_id = request.form['employee_id']
        password = request.form['password']

        try:
            # Check if a user with the provided employee_id exists
            user = db_session.query(User).filter_by(employee_id=employee_id).first()

            if user and user.check_password_hash(password):
                # User is found and password is correct
                # Proceed to create a session
                new_chat_session = ChatSession(
                    user_id=user.id,
                    start_time=datetime.now(),
                    last_interaction=datetime.now(),
                    session_data=[]  # Or any initial data you want to include
                )
                db_session.add(new_chat_session)
                db_session.commit()

                # Store user information in the Flask session
                session['user_id'] = user.id
                session['employee_id'] = user.employee_id
                session['first_name'] = user.first_name
                session['last_name'] = user.last_name
                session['primary_area'] = user.primary_area
                session['age'] = user.age
                session['education_level'] = user.education_level
                session['start_date'] = user.start_date
                session['user_level'] = user.user_level.name

                # Redirect based on user level
                if user.user_level == UserLevel.ADMIN:
                    return redirect(url_for('admin_bp.admin_dashboard'))

                # Redirect to chat interface or home page within login_bp
                return render_template('index.html', employee_id=user.employee_id, name=user.first_name)

            else:
                flash("Invalid username or password", 'error')

        except Exception as e:
            flash(f"An error occurred: {e}", 'error')

        finally:
            db_session.remove()  # Remove the SQLAlchemy session from the scoped_session

    return render_template('login.html')

@login_bp.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('employee_id', None)
    session.pop('first_name', None)
    session.pop('last_name', None)
    session.pop('primary_area', None)
    session.pop('age', None)
    session.pop('education_level', None)
    session.pop('start_date', None)
    return redirect(url_for('login_bp.login'))
