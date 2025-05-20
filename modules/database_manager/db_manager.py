from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import debug_id, info_id, error_id, get_request_id,logger
from modules.emtacdb.emtacdb_fts import Part, Image, PartsPositionImageAssociation, Drawing, DrawingPartAssociation
from sqlalchemy import and_
import pandas as pd
import sqlite3
import json
from datetime import datetime


class DatabaseManager:
    """Base class for database management operations."""

    def __init__(self, session=None, request_id=None):
        self.session_provided = session is not None
        self.session = session or DatabaseConfig().get_main_session()
        self.request_id = request_id or get_request_id()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.session_provided:
            self.session.close()
            debug_id("Closed database session", self.request_id)

    def commit(self):
        """Commit the current transaction."""
        try:
            self.session.commit()
            debug_id("Transaction committed", self.request_id)
        except Exception as e:
            self.session.rollback()
            error_id(f"Transaction failed, rolled back: {str(e)}", self.request_id, exc_info=True)
            raise

class RelationshipManager(DatabaseManager):
    """Manages relationships and associations between different database entities."""

    def associate_parts_with_images_by_title(self, part_ids=None, position_id=None):
        """
        Associates parts with images that have matching titles.

        Args:
            part_ids: List of part IDs to process (None for all parts)
            position_id: Optional position ID to include in associations

        Returns:
            Dictionary mapping part IDs to lists of created associations
        """
        info_id("Starting part-image association process", self.request_id)
        result = {}

        try:
            # Get parts to process
            if part_ids is None:
                parts = self.session.query(Part).all()
                part_ids = [part.id for part in parts]
            else:
                parts = self.session.query(Part).filter(Part.id.in_(part_ids)).all()
                part_ids = [part.id for part in parts]

            info_id(f"Processing {len(part_ids)} parts for image associations", self.request_id)

            # Process each part
            for part in parts:
                associations = self._associate_single_part(part, position_id)
                result[part.id] = associations

            return result
        except Exception as e:
            error_id(f"Error in part-image association: {str(e)}", self.request_id, exc_info=True)
            raise

    def _associate_single_part(self, part, position_id=None):
        """Helper method to associate a single part with matching images."""
        created = []

        # Find images with matching titles
        matching_images = self.session.query(Image).filter(Image.title == part.part_number).all()
        info_id(f"Found {len(matching_images)} images matching part {part.part_number}", self.request_id)

        for image in matching_images:
            # Check for existing association
            existing = self.session.query(PartsPositionImageAssociation).filter(
                and_(
                    PartsPositionImageAssociation.part_id == part.id,
                    PartsPositionImageAssociation.image_id == image.id,
                    PartsPositionImageAssociation.position_id == position_id if position_id else True
                )
            ).first()

            if not existing:
                # Create new association
                assoc = PartsPositionImageAssociation(
                    part_id=part.id,
                    image_id=image.id,
                    position_id=position_id
                )
                self.session.add(assoc)
                self.session.flush()
                created.append(assoc)

        return created

    def associate_drawings_with_parts_by_number(self):
        """
        Associates drawings with parts based on spare part numbers.
        Handles multiple comma-separated part numbers per drawing.

        Returns:
            Dict mapping drawing_id to list of created associations
        """
        # Import the logging functions with correct parameter order
        from modules.configuration.log_config import debug_id, info_id

        # Use info_id with correct parameter order (message, request_id)
        info_id("Associating drawings with parts based on spare part numbers", self.request_id)

        # Get all drawings with spare part numbers
        drawings = self.session.query(Drawing).filter(Drawing.drw_spare_part_number.isnot(None))

        # Count the drawings
        drawing_count = drawings.count()
        info_id(f"Found {drawing_count} drawings with spare part numbers", self.request_id)

        # Track associations for each drawing
        associations_by_drawing = {}

        for drawing in drawings:
            # Skip if drawing has no spare part number
            if not drawing.drw_spare_part_number or not drawing.drw_spare_part_number.strip():
                continue

            # Debug info about the drawing
            debug_id(
                f"Processing drawing {drawing.drw_number} with spare part number(s): {drawing.drw_spare_part_number}",
                self.request_id)

            # Split by comma and clean up each part number
            part_numbers = [pn.strip() for pn in drawing.drw_spare_part_number.split(',') if pn.strip()]

            drawing_associations = []

            # Process each part number for this drawing
            for part_number in part_numbers:
                # Find parts matching this part number
                matching_parts = self.session.query(Part).filter(
                    Part.part_number == part_number
                ).all()

                # Debug info about matching parts
                debug_id(f"Found {len(matching_parts)} parts matching number '{part_number}'", self.request_id)

                # Create association for each matching part
                for part in matching_parts:
                    # Check if association already exists
                    existing = self.session.query(DrawingPartAssociation).filter(
                        DrawingPartAssociation.drawing_id == drawing.id,
                        DrawingPartAssociation.part_id == part.id
                    ).first()

                    if not existing:
                        # Create new association - only use fields that exist in your model
                        association = DrawingPartAssociation(
                            drawing_id=drawing.id,
                            part_id=part.id
                            # Removed created_by and association_type as they don't exist in your model
                        )
                        self.session.add(association)

                        # We need to flush to get the ID
                        self.session.flush()

                        drawing_associations.append(association)
                        debug_id(f"Created association between drawing {drawing.drw_number} "
                                 f"and part {part.part_number}", self.request_id)
                    else:
                        debug_id(f"Association already exists between drawing {drawing.drw_number} "
                                 f"and part {part.part_number}", self.request_id)

            # Store associations for this drawing
            if drawing_associations:
                associations_by_drawing[drawing.id] = drawing_associations

        # Final debug info
        info_id(f"Created new associations for {len(associations_by_drawing)} drawings", self.request_id)

        return associations_by_drawing

    def _associate_single_drawing(self, drawing):
        """Helper method to associate a single drawing with matching parts."""
        created = []

        # Skip drawings without a spare part number
        if not drawing.drw_spare_part_number:
            info_id(f"Drawing {drawing.id} has no spare part number, skipping", self.request_id)
            return created

        # Find parts with matching part numbers
        matching_parts = self.session.query(Part).filter(Part.part_number == drawing.drw_spare_part_number).all()
        info_id(
            f"Found {len(matching_parts)} parts matching drawing {drawing.drw_number} (spare part: {drawing.drw_spare_part_number})",
            self.request_id)

        for part in matching_parts:
            # Check for existing association
            existing = self.session.query(DrawingPartAssociation).filter(
                and_(
                    DrawingPartAssociation.drawing_id == drawing.id,
                    DrawingPartAssociation.part_id == part.id
                )
            ).first()

            if not existing:
                # Create new association
                assoc = DrawingPartAssociation(
                    drawing_id=drawing.id,
                    part_id=part.id
                )
                self.session.add(assoc)
                self.session.flush()
                created.append(assoc)

        return created

class DuplicateManager(DatabaseManager):
    """Manages detection and handling of duplicate entities in the database."""

    def find_duplicate_parts(self, threshold=0.9):
        """
        Finds potential duplicate parts based on similarity metrics.

        Args:
            threshold: Similarity threshold (0.0-1.0)

        Returns:
            List of tuples containing potential duplicate part pairs
        """
        # Implementation for duplicate detection
        pass

    def merge_duplicate_parts(self, source_id, target_id, fields_to_merge=None):
        """
        Merges two duplicate parts.

        Args:
            source_id: ID of the source part (will be merged into target)
            target_id: ID of the target part (will be kept)
            fields_to_merge: List of fields to merge (None for all)

        Returns:
            The updated target part
        """
        # Implementation for merging entities
        pass


import pandas as pd
import json
from datetime import datetime
import os
import sys

# Import logging and database configurations
from modules.configuration.log_config import (
    logger, debug_id, info_id, warning_id, error_id,
    set_request_id, get_request_id, log_timed_operation,
    with_request_id
)
from modules.configuration.config_env import DatabaseConfig


class EnhancedExcelToSQLiteMapper:
    def __init__(self, excel_path, db_config=None):
        """
        Initialize the mapper with an Excel file path and database configuration.

        Args:
            excel_path: Path to the Excel file
            db_config: DatabaseConfig instance. If None, a new one will be created.
        """
        # Generate a request ID for tracking operations
        self.request_id = set_request_id()
        info_id("Initializing EnhancedExcelToSQLiteMapper", self.request_id)

        self.excel_path = excel_path

        # Use the provided db_config or create a new one
        self.db_config = db_config if db_config else DatabaseConfig()

        debug_id(f"Mapper initialized with Excel file: {excel_path}", self.request_id)

    def infer_sqlite_type(self, pandas_dtype):
        """Infer SQLite type from pandas dtype."""
        if pd.api.types.is_integer_dtype(pandas_dtype):
            return 'INTEGER'
        elif pd.api.types.is_float_dtype(pandas_dtype):
            return 'REAL'
        elif pd.api.types.is_bool_dtype(pandas_dtype):
            return 'INTEGER'
        elif pd.api.types.is_datetime64_any_dtype(pandas_dtype):
            return 'TEXT'
        else:
            return 'TEXT'

    @with_request_id
    def prompt_for_mapping(self, df):
        """
        Prompt user for column mapping and type overrides.
        Decorated with request ID tracking.
        """
        mapping = {}
        type_overrides = {}

        info_id("Excel columns found:", self.request_id)
        for i, col in enumerate(df.columns):
            dtype = self.infer_sqlite_type(df[col].dtype)
            debug_id(f"  {i + 1}. '{col}' (suggested SQLite type: {dtype})", self.request_id)

        info_id("For each Excel column, specify the SQLite column name to map to (leave blank to skip):",
                self.request_id)
        for col in df.columns:
            dtype = self.infer_sqlite_type(df[col].dtype)
            mapped_col = input(f"Map Excel column '{col}' to SQLite column (or blank to skip): ").strip()
            if mapped_col:
                type_choice = input(
                    f" - Data type for '{mapped_col}'? [INTEGER/REAL/TEXT, default: {dtype}]: ").strip().upper()
                type_overrides[mapped_col] = type_choice if type_choice in ['INTEGER', 'REAL', 'TEXT'] else dtype
                mapping[col] = mapped_col
                debug_id(f"Mapped '{col}' to '{mapped_col}' with type {type_overrides[mapped_col]}", self.request_id)

        return mapping, type_overrides

    def create_mapping_table(self, session):
        """Create the mapping table if it doesn't exist using SQLAlchemy session."""
        with log_timed_operation("create_mapping_table", self.request_id):
            try:
                session.execute("""
                CREATE TABLE IF NOT EXISTS excel_sqlite_mapping (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mapping_name TEXT,
                    excel_file TEXT,
                    excel_sheet TEXT,
                    sqlite_table TEXT,
                    column_mapping TEXT,
                    column_types TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """)
                session.commit()
                debug_id("Mapping table created or already exists", self.request_id)
            except Exception as e:
                session.rollback()
                error_id(f"Error creating mapping table: {str(e)}", self.request_id)
                raise

    def store_mapping(self, session, mapping_name, excel_file, excel_sheet, sqlite_table, mapping, type_overrides):
        """Store the mapping information using SQLAlchemy session."""
        with log_timed_operation("store_mapping", self.request_id):
            try:
                sql = """
                INSERT INTO excel_sqlite_mapping (mapping_name, excel_file, excel_sheet, sqlite_table, column_mapping, column_types, created_at)
                VALUES (:name, :file, :sheet, :table, :mapping, :types, :created_at)
                """
                session.execute(sql, {
                    'name': mapping_name,
                    'file': excel_file,
                    'sheet': excel_sheet,
                    'table': sqlite_table,
                    'mapping': json.dumps(mapping),
                    'types': json.dumps(type_overrides),
                    'created_at': datetime.now().isoformat()
                })
                session.commit()
                info_id(f"Mapping information stored for '{mapping_name}'", self.request_id)
            except Exception as e:
                session.rollback()
                error_id(f"Error storing mapping information: {str(e)}", self.request_id)
                raise

    def create_table(self, session, table_name, mapping, type_overrides):
        """Create the target table using SQLAlchemy session."""
        with log_timed_operation(f"create_table_{table_name}", self.request_id):
            try:
                columns = []
                for excel_col, sqlite_col in mapping.items():
                    col_type = type_overrides[sqlite_col]
                    columns.append(f'"{sqlite_col}" {col_type}')

                col_defs = ", ".join(columns)
                sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({col_defs});'

                session.execute(sql)
                session.commit()
                info_id(f"Created table '{table_name}' with columns: {columns}", self.request_id)
            except Exception as e:
                session.rollback()
                error_id(f"Error creating table '{table_name}': {str(e)}", self.request_id)
                raise

    def insert_data(self, session, table_name, df, mapping):
        """Insert data into the target table using SQLAlchemy session."""
        with log_timed_operation(f"insert_data_{table_name}", self.request_id):
            try:
                mapped_cols = [col for col in mapping.keys()]
                sqlite_cols = [mapping[col] for col in mapped_cols]

                insert_cols = ', '.join(f'"{col}"' for col in sqlite_cols)
                placeholders = ', '.join(['?'] * len(sqlite_cols))
                insert_sql = f'INSERT INTO "{table_name}" ({insert_cols}) VALUES ({placeholders})'

                values = df[mapped_cols].values.tolist()

                # Converting to raw SQL execution for batch insertion
                connection = session.connection().connection
                cursor = connection.cursor()

                # Using executemany for more efficient insertion
                cursor.executemany(insert_sql, values)
                connection.commit()

                info_id(f"Inserted {len(values)} rows into '{table_name}'.", self.request_id)
            except Exception as e:
                session.rollback()
                error_id(f"Error inserting data into '{table_name}': {str(e)}", self.request_id)
                raise

    @with_request_id
    def run(self, sheet_name=None):
        """Main execution method with request ID tracking."""
        try:
            # Read Excel file
            with log_timed_operation("read_excel", self.request_id):
                if sheet_name:
                    info_id(f"Reading Excel sheet: {sheet_name}", self.request_id)
                    df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
                else:
                    xls = pd.ExcelFile(self.excel_path)
                    info_id(f"Sheets found: {xls.sheet_names}", self.request_id)
                    sheet_name = input("Enter sheet name to import: ").strip()
                    df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
                    info_id(f"Read {len(df)} rows from sheet '{sheet_name}'", self.request_id)

            # Table name
            default_table = sheet_name.replace(' ', '_')
            table_name = input(f"SQLite table name? (default: {default_table}): ").strip() or default_table
            info_id(f"Using table name: {table_name}", self.request_id)

            # Mapping name
            mapping_name = input("Name this mapping (for future use): ").strip() or f"{sheet_name}_to_{table_name}"
            info_id(f"Using mapping name: {mapping_name}", self.request_id)

            # Column mapping
            mapping, type_overrides = self.prompt_for_mapping(df)

            if not mapping:
                warning_id("No columns mapped! Exiting.", self.request_id)
                return

            # Get a session from the main database
            info_id("Establishing database connection", self.request_id)
            session = self.db_config.get_main_session()

            try:
                with log_timed_operation("database_operations", self.request_id):
                    # Create mapping table if needed
                    self.create_mapping_table(session)

                    # Store mapping information
                    self.store_mapping(session, mapping_name, self.excel_path, sheet_name, table_name, mapping,
                                       type_overrides)

                    # Create data table
                    self.create_table(session, table_name, mapping, type_overrides)

                    # Insert data
                    self.insert_data(session, table_name, df, mapping)

                info_id(f"All operations completed successfully for mapping '{mapping_name}'", self.request_id)
            finally:
                # Ensure session is properly closed
                debug_id("Closing database session", self.request_id)
                session.close()

        except Exception as e:
            error_id(f"Error in Excel to SQLite mapping process: {str(e)}", self.request_id)
            raise

        info_id(f"Done! Mapping info and data stored for '{mapping_name}'.", self.request_id)





