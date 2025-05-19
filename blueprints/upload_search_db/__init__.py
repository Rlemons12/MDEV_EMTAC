# Import necessary modules
from flask import Blueprint

# Import the blueprint from the module file
from .search_drawing import drawing_routes

# If you have other blueprints in this directory, import them here
# from .other_module import other_blueprint

# Export the blueprints to make them available when importing from this package
__all__ = ['drawing_routes']

# Optional: Any initialization code for the package can go here