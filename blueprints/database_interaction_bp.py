# database_interaction_bp.py
from flask import Blueprint, render_template

# Create a Blueprint for database interaction
database_interaction_bp = Blueprint('database_interaction_bp', __name__)

@database_interaction_bp.route('/')
def database_interaction():
    return render_template('database_interaction.html')
