# tool_manage_manufacturer.py

from blueprints.tool_routes import tool_blueprint_bp
from flask import render_template, redirect, url_for, flash, request
from modules.configuration.config_env import DatabaseConfig
from modules.tool_module.forms import ToolManufacturerForm
from modules.configuration.log_config import logger

db_config = DatabaseConfig


@tool_blueprint_bp.route('/tool_manufacturer/add', methods=['GET', 'POST'])
@tool_blueprint_bp.route('/tool_manufacturer/edit_manufacturer/<int:manufacturer_id>', methods=['GET', 'POST'])
def edit_manufacturer(manufacturer_id=None):
    pass
@tool_blueprint_bp.route('/tool_manufacturer/delete', methods=['GET', 'POST'])
def delete_manufacturer(manufacturer_id=None):
    pass