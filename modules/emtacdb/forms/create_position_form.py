# modules/emtacdb/forms/create_position_form.py

from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField, TextAreaField, MultipleFileField
from wtforms.validators import Optional, DataRequired
from wtforms_sqlalchemy.fields import QuerySelectField
from modules.emtacdb.emtacdb_fts import (
    Area, EquipmentGroup, Model, AssetNumber, Location,
    Assembly, SubAssembly, AssemblyView, SiteLocation
)
from flask import current_app  # To access the app's config


class CreatePositionForm(FlaskForm):
    # Existing QuerySelectFields
    area = QuerySelectField(
        label="Area",
        query_factory=lambda: current_app.config['db_config'].get_main_session().query(Area).order_by(Area.name).all(),
        allow_blank=True,
        blank_text="Select an Area",
        validators=[Optional()],
        get_label="name"
    )

    equipment_group = QuerySelectField(
        label="Equipment Group",
        query_factory=lambda: current_app.config['db_config'].get_main_session().query(EquipmentGroup).order_by(
            EquipmentGroup.name).all(),
        allow_blank=True,
        blank_text="Select an Equipment Group",
        validators=[Optional()],
        get_label="name"
    )

    model = QuerySelectField(
        label="Model",
        query_factory=lambda: current_app.config['db_config'].get_main_session().query(Model).order_by(
            Model.name).all(),
        allow_blank=True,
        blank_text="Select a Model",
        validators=[Optional()],
        get_label="name"
    )

    asset_number = QuerySelectField(
        label="Asset Number",
        query_factory=lambda: current_app.config['db_config'].get_main_session().query(AssetNumber).order_by(
            AssetNumber.number).all(),
        allow_blank=True,
        blank_text="Select an Asset Number",
        validators=[Optional()],
        get_label="number"
    )

    asset_number_input = StringField(
        label="New Asset Number",
        validators=[Optional()],
        render_kw={"placeholder": "Enter new Asset Number if not listed"}
    )

    location = QuerySelectField(
        label="Location",
        query_factory=lambda: current_app.config['db_config'].get_main_session().query(Location).order_by(
            Location.name).all(),
        allow_blank=True,
        blank_text="Select a Location",
        validators=[Optional()],
        get_label="name"
    )

    location_input = StringField(
        label="New Location",
        validators=[Optional()],
        render_kw={"placeholder": "Enter new Location if not listed"}
    )

    assembly = QuerySelectField(
        label="Assembly",
        query_factory=lambda: current_app.config['db_config'].get_main_session().query(Assembly).order_by(
            Assembly.name).all(),
        allow_blank=True,
        blank_text="Select an Assembly",
        validators=[Optional()],
        get_label="name"
    )

    assembly_input = StringField(
        label="New Assembly",
        validators=[Optional()],
        render_kw={"placeholder": "Enter new Assembly if not listed"}
    )

    subassembly = QuerySelectField(
        label="Subassembly",
        query_factory=lambda: current_app.config['db_config'].get_main_session().query(SubAssembly).order_by(
            SubAssembly.name).all(),
        allow_blank=True,
        blank_text="Select a Subassembly",
        validators=[Optional()],
        get_label="name"
    )

    subassembly_input = StringField(
        label="New Subassembly",
        validators=[Optional()],
        render_kw={"placeholder": "Enter new Subassembly if not listed"}
    )

    assembly_view = QuerySelectField(
        label="Assembly View",
        query_factory=lambda: current_app.config['db_config'].get_main_session().query(AssemblyView).order_by(
            AssemblyView.name).all(),
        allow_blank=True,
        blank_text="Select an Assembly View",
        validators=[Optional()],
        get_label="name"
    )

    site_location = QuerySelectField(
        label="Site Location",
        query_factory=lambda: current_app.config['db_config']
            .get_main_session()
            .query(SiteLocation)
            .order_by(SiteLocation.title, SiteLocation.room_number)
            .all(),
        allow_blank=True,
        blank_text="Select a Site Location",
        validators=[Optional()],
        get_label=lambda site_location: f"{site_location.title} - Room {site_location.room_number}"
    )

    # New Fields
    position = StringField(
        label="Position Name",
        validators=[DataRequired(message="Position name is required.")],
        render_kw={"placeholder": "Enter Position Name"}
    )

    part_numbers = TextAreaField(
        label="Part Numbers",
        validators=[Optional()],
        render_kw={"placeholder": "Enter Part Numbers separated by commas"}
    )

    images = MultipleFileField(
        label="Upload Images",
        validators=[Optional()],
        render_kw={"multiple": True}
    )

    drawings = MultipleFileField(
        label="Upload Drawings",
        validators=[Optional()],
        render_kw={"multiple": True}
    )

    submit = SubmitField("Create Position")
