# modules/emtac_ai/__init__.py

# Basic imports for your module
from . import training_scripts
from . import training_data
from . import models
from . import orchestrator
from . import util_scripts

# Optionally, expose key classes or functions for convenience
# For example:
# from .some_core_module import SomeMainClass, some_main_function

__version__ = "0.1.0"

def get_version():
    return __version__
