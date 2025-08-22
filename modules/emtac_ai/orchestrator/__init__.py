"""
Orchestrator package initializer.

This makes it easy to import the Orchestrator and its utilities:
    from modules.emtac_ai.orchestrator import Orchestrator, to_abs_path
"""

from .orchestrator import Orchestrator, to_abs_path

__all__ = [
    "Orchestrator",
    "to_abs_path",
]

