from flask import Blueprint

# Create the blueprint
tool_blueprint_bp = Blueprint('tool_routes', __name__, template_folder='../../templates/tool_templates')

# Import routes from submodules
from .tool_add import *
from .tool_search import *
from .tool_get_data import *  # Import the renamed module
