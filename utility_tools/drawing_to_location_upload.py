import pandas as pd
import logging
import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
# Import database configuration
from modules.configuration.config_env import DatabaseConfig
# Import custom logging functions
from modules.configuration.log_config import (
    debug_id, info_id, warning_id, error_id, critical_id,
    with_request_id, get_request_id, set_request_id, log_timed_operation
)

# Import application configuration
from modules.configuration.config import (
    BASE_DIR, DATABASE_DIR, DB_LOADSHEET, DATABASE_URL
)

# Assuming these models are imported from your existing module
from modules.emtacdb.emtacdb_fts import Area, EquipmentGroup, Model, AssetNumber, Location, Position, Drawing, Part, DrawingPositionAssociation, DrawingPartAssociation

# Use the module's logger instead of creating a new one
from modules.configuration.log_config import logger

# Add these imports at the top of drawing_to_location_upload.py if they don't already exist
import pandas as pd
import os
from sqlalchemy.exc import SQLAlchemyError
from modules.configuration.config import BASE_DIR, DATABASE_DIR, DB_LOADSHEET
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    debug_id, info_id, warning_id, error_id, critical_id,
    with_request_id, get_request_id, set_request_id, log_timed_operation
)


# Replace the existing ExcelToDbMapper class with this updated version
class ExcelToDbMapper:
    """
    A class to map Excel data to database models according to the hierarchical structure.
    """

    def __init__(self, db_url=None):
        """
        Initialize with database connection using DatabaseConfig.

        Args:
            db_url (str, optional): SQLAlchemy database URL (kept for compatibility, not used)
        """
        # Set a request ID for this instance
        self.request_id = set_request_id()
        info_id(f"Initializing ExcelToDbMapper with DatabaseConfig", self.request_id)

        # For backward compatibility, log the provided db_url if present
        if db_url:
            debug_id(f"Note: Provided db_url: {db_url} will be ignored in favor of DatabaseConfig")

        # Create DB config and get a session
        self.db_config = DatabaseConfig()
        self.session = self.db_config.get_main_session()
        self.excel_data = None

    @with_request_id
    def load_excel(self, file_path):
        """
        Load data from Excel file.

        Args:
            file_path (str): Path to the Excel file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            info_id(f"Loading Excel file: {file_path}")

            # Verify file exists
            if not os.path.exists(file_path):
                error_id(f"Excel file not found: {file_path}")
                return False

            self.excel_data = pd.read_excel(file_path)
            info_id(f"Successfully loaded {len(self.excel_data)} rows")

            # Log column headers for verification
            headers = list(self.excel_data.columns)
            debug_id(f"Excel headers: {headers}")

            return True
        except Exception as e:
            error_id(f"Error loading Excel file: {e}", exc_info=True)
            return False

    @with_request_id
    def process_data(self):
        """
        Process Excel data and map to database models.
        Ensures all required fields are present and prevents duplicates.

        Returns:
            bool: True if successful, False otherwise
        """
        if self.excel_data is None:
            error_id("No Excel data loaded")
            return False

        required_fields = ['area', 'equipment_group', 'model', 'asset_number', 'stations']

        try:
            # Start transaction
            info_id(f"Starting to process {len(self.excel_data)} rows of data")

            for index, row in self.excel_data.iterrows():
                with log_timed_operation(f"Processing row {index + 1}"):
                    info_id(f"Processing row {index + 1}")

                    # 1. Validate all required fields are present
                    missing_fields = [field for field in required_fields if pd.isna(row.get(field))]
                    if missing_fields:
                        warning_id(f"Skipping row {index + 1}: Missing required fields: {', '.join(missing_fields)}")
                        continue

                    # 2. Check for existing drawing first
                    drawing = None
                    drawing_number = row.get('DRAWING NUMBER')
                    if pd.notna(drawing_number):
                        with log_timed_operation(f"Looking up drawing {drawing_number}"):
                            drawing = (
                                self.session.query(Drawing)
                                .filter(Drawing.drw_number == drawing_number)
                                .first()
                            )
                            if drawing:
                                info_id(f"Found existing drawing with number {drawing_number} (ID: {drawing.id})")

                    # 3a. Check for existing area or create new one
                    area_name = str(row.get('area')).strip()
                    with log_timed_operation(f"Processing area {area_name}"):
                        area = self.session.query(Area).filter(Area.name == area_name).first()
                        if not area:
                            info_id(f"Creating new Area: {area_name}")
                            area = Area(name=area_name)
                            self.session.add(area)
                            self.session.flush()

                    # 3b. Check for existing equipment_group or create new one
                    equipment_group_name = str(row.get('equipment_group')).strip()
                    with log_timed_operation(f"Processing equipment group {equipment_group_name}"):
                        equipment_group = (
                            self.session.query(EquipmentGroup)
                            .filter(
                                EquipmentGroup.name == equipment_group_name,
                                EquipmentGroup.area_id == area.id
                            )
                            .first()
                        )
                        if not equipment_group:
                            info_id(f"Creating new EquipmentGroup: {equipment_group_name} in area {area.id}")
                            equipment_group = EquipmentGroup(name=equipment_group_name, area_id=area.id)
                            self.session.add(equipment_group)
                            self.session.flush()

                    # 3c. Check for existing model or create new one
                    model_name = str(row.get('model')).strip()
                    with log_timed_operation(f"Processing model {model_name}"):
                        model = (
                            self.session.query(Model)
                            .filter(
                                Model.name == model_name,
                                Model.equipment_group_id == equipment_group.id
                            )
                            .first()
                        )
                        if not model:
                            info_id(f"Creating new Model: {model_name} in equipment group {equipment_group.id}")
                            model = Model(name=model_name, equipment_group_id=equipment_group.id)
                            self.session.add(model)
                            self.session.flush()

                    # 3d. Check for existing location or create new one
                    stations_value = str(row.get('stations')).strip()
                    # Extract only the first part before any comma
                    if ',' in stations_value:
                        location_name = stations_value.split(',')[0].strip()
                        info_id(f"Using only first location from comma-separated list: {location_name}")
                    else:
                        location_name = stations_value

                    with log_timed_operation(f"Processing location {location_name}"):
                        location = (
                            self.session.query(Location)
                            .filter(
                                Location.name == location_name,
                                Location.model_id == model.id
                            )
                            .first()
                        )
                        if not location:
                            info_id(f"Creating new Location: {location_name} for model {model.id}")
                            location = Location(name=location_name, model_id=model.id)
                            self.session.add(location)
                            self.session.flush()

                    # 4. Check for existing position with these hierarchy elements (excluding asset_number)
                    with log_timed_operation("Processing position"):
                        existing_position = (
                            self.session.query(Position)
                            .filter(
                                Position.area_id == area.id,
                                Position.equipment_group_id == equipment_group.id,
                                Position.model_id == model.id,
                                Position.location_id == location.id
                            )
                            .first()
                        )

                        if existing_position:
                            info_id(f"Found existing Position with ID={existing_position.id}")
                            position = existing_position
                        else:
                            # Create a new position
                            position = Position(
                                area_id=area.id,
                                equipment_group_id=equipment_group.id,
                                model_id=model.id,
                                location_id=location.id
                            )
                            self.session.add(position)
                            self.session.flush()
                            info_id(f"Created new Position with ID={position.id}")

                    # 5. If we don't have a drawing yet, create one if needed
                    if not drawing and pd.notna(drawing_number):
                        with log_timed_operation(f"Creating new drawing {drawing_number}"):
                            drawing_name = row.get('DRAWING NAME', '')
                            revision = row.get('REVISION', '')
                            equipment_name = row.get('EQUIPMENT NAME', '')
                            spare_part_number = row.get('SPARE PART NUMBER')

                            info_id(f"Creating new Drawing: {drawing_number} - {drawing_name} (Rev: {revision})")
                            drawing = Drawing(
                                drw_number=drawing_number,
                                drw_name=drawing_name,
                                drw_revision=revision,
                                drw_equipment_name=equipment_name,
                                drw_spare_part_number=spare_part_number
                            )
                            self.session.add(drawing)
                            self.session.flush()

                    # 6. Associate drawing with position if we have both
                    if drawing and position:
                        with log_timed_operation(f"Associating drawing {drawing.id} with position {position.id}"):
                            info_id(f"About to associate drawing {drawing.id} with position {position.id}")
                            try:
                                # First check if association already exists to better understand the situation
                                existing_assoc = (
                                    self.session.query(DrawingPositionAssociation)
                                    .filter(
                                        DrawingPositionAssociation.drawing_id == drawing.id,
                                        DrawingPositionAssociation.position_id == position.id
                                    )
                                    .first()
                                )

                                if existing_assoc:
                                    info_id(
                                        f"Association between drawing {drawing.id} and position {position.id} already exists (ID: {existing_assoc.id})")
                                else:
                                    info_id(f"No existing association found, creating new association")

                                    # Try first with the helper method
                                    try:
                                        result = DrawingPositionAssociation.associate_drawing_position(
                                            drawing_id=drawing.id,
                                            position_id=position.id,
                                            session=self.session
                                        )

                                        if result:
                                            info_id(
                                                f"Successfully associated drawing {drawing.id} with position {position.id} using helper method")
                                        else:
                                            warning_id(
                                                f"Helper method returned None when associating drawing {drawing.id} with position {position.id}")

                                            # Fall back to direct creation
                                            info_id(f"Trying direct association creation as fallback")
                                            drawing_position = DrawingPositionAssociation(
                                                drawing_id=drawing.id,
                                                position_id=position.id
                                            )
                                            self.session.add(drawing_position)
                                            self.session.flush()
                                            info_id(f"Created association directly with ID: {drawing_position.id}")
                                    except Exception as e:
                                        error_id(f"Error using helper method: {e}", exc_info=True)

                                        # Fall back to direct creation
                                        info_id(f"Trying direct association creation as fallback after error")
                                        drawing_position = DrawingPositionAssociation(
                                            drawing_id=drawing.id,
                                            position_id=position.id
                                        )
                                        self.session.add(drawing_position)
                                        self.session.flush()
                                        info_id(f"Created association directly with ID: {drawing_position.id}")

                            except Exception as e:
                                error_id(f"Error associating drawing with position: {e}", exc_info=True)
                                # Continue processing other rows even if this association fails

                    # 7. Handle spare part if present
                    if drawing and pd.notna(row.get('SPARE PART NUMBER')):
                        with log_timed_operation("Processing spare part"):
                            spare_part_number = str(row.get('SPARE PART NUMBER')).strip()

                            # Check if part exists
                            part = (
                                self.session.query(Part)
                                .filter(Part.part_number == spare_part_number)
                                .first()
                            )

                            if not part:
                                info_id(f"Creating new Part: {spare_part_number}")
                                part = Part(
                                    part_number=spare_part_number,
                                    name=f"Part {spare_part_number}"
                                )
                                self.session.add(part)
                                self.session.flush()

                            # Check if association already exists
                            existing_part_assoc = (
                                self.session.query(DrawingPartAssociation)
                                .filter(
                                    DrawingPartAssociation.drawing_id == drawing.id,
                                    DrawingPartAssociation.part_id == part.id
                                )
                                .first()
                            )

                            if not existing_part_assoc:
                                try:
                                    drawing_part_assoc = DrawingPartAssociation(
                                        drawing_id=drawing.id,
                                        part_id=part.id
                                    )
                                    self.session.add(drawing_part_assoc)
                                    self.session.flush()
                                    info_id(f"Associated drawing {drawing.id} with part {part.id}")
                                except Exception as e:
                                    error_id(f"Error associating drawing with part: {e}", exc_info=True)
                            else:
                                info_id(f"Drawing {drawing.id} already associated with part {part.id}")

            # Commit transaction
            with log_timed_operation("Committing transaction"):
                self.session.commit()
                info_id(f"Successfully processed all Excel data.")
            return True

        except SQLAlchemyError as e:
            self.session.rollback()
            error_id(f"Database error while processing data: {e}", exc_info=True)
            return False
        except Exception as e:
            self.session.rollback()
            error_id(f"Unexpected error while processing data: {e}", exc_info=True)
            return False

    @with_request_id
    def close(self):
        """Close database session."""
        debug_id("Closing database session")
        if self.session:
            self.session.close()


@with_request_id
def main():
    """Main entry point for the script."""
    # Use the database URL from the config module
    db_url = DATABASE_URL

    # Use the correct path for the Excel file
    excel_file = os.path.join(DB_LOADSHEET, "Active Drawing List breakdown.xlsx")
    info_id(f"Excel file path: {excel_file}")

    # Log information about execution environment
    info_id(f"Running from directory: {os.getcwd()}")
    info_id(f"Base directory: {BASE_DIR}")
    info_id(f"Database directory: {DATABASE_DIR}")
    info_id(f"DB_LOADSHEET directory: {DB_LOADSHEET}")

    # Create mapper and process data
    mapper = ExcelToDbMapper(db_url)
    try:
        with log_timed_operation("Excel Import Process"):
            info_id(f"Starting Excel import process for file: {excel_file}")
            if mapper.load_excel(excel_file):
                success = mapper.process_data()
                if success:
                    info_id("Data mapping completed successfully")
                else:
                    error_id("Data mapping failed")
            else:
                error_id("Failed to load Excel file")
    except Exception as e:
        error_id(f"Unhandled exception in main: {e}", exc_info=True)
    finally:
        mapper.close()


if __name__ == "__main__":
    main()