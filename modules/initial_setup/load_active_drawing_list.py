import os
import sys
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from modules.initial_setup.initializer_logger import initializer_logger, close_initializer_logger
from modules.configuration.config import DB_LOADSHEET, DATABASE_URL
from modules.emtacdb.emtacdb_fts import Drawing

def main():
    """
    Loads data from 'active drawing list.xlsx' into the Drawing table.
    Intended to run manually or from an orchestrator script that calls main().
    """

    # Log the start of the process
    initializer_logger.info("Starting active drawing list data insertion process.")

    try:
        # Create a SQLAlchemy engine using the DATABASE_URL from your config
        initializer_logger.debug(f"Creating SQLAlchemy engine with DATABASE_URL: {DATABASE_URL}")
        engine = create_engine(DATABASE_URL)

        # Create a session
        Session = sessionmaker(bind=engine)
        session = Session()
        initializer_logger.debug("SQLAlchemy session created successfully.")

        # Load data from the "active drawing list" sheet
        load_sheet_path = os.path.join(DB_LOADSHEET, "active drawing list.xlsx")
        initializer_logger.info(f"Loading Excel file: {load_sheet_path}")
        df = pd.read_excel(load_sheet_path)

        # Drop the extra column if it exists
        if 'Unnamed: 7' in df.columns:
            df = df.drop(columns=['Unnamed: 7'])
            initializer_logger.debug("Dropped extra column 'Unnamed: 7'.")

        # Rename columns if needed
        column_mapping = {
            'EQUIPMENT NUMBER': 'EQUIPMENT NUMBER',
            'EQUIPMENT NAME': 'EQUIPMENT NAME',
            'DRAWING NUMBER': 'DRAWING NUMBER',
            'DRAWING NAME': 'DRAWING NAME',
            'REVISION': 'REVISION',
            'CC REQUIRED': 'CC REQUIRED',
            'SPARE PART NUMBER': 'SPARE PART NUMBER'
        }
        df.columns = [column_mapping.get(col, col) for col in df.columns]
        initializer_logger.debug(f"Renamed columns: {df.columns.tolist()}")

        # Strip leading/trailing spaces from each column
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        initializer_logger.info("Stripped leading and trailing spaces from columns.")

        # Iterate through each row in the dataframe
        for index, row in df.iterrows():
            # Extract data from the row
            equipment_name = row['EQUIPMENT NAME']
            drawing_number = row['DRAWING NUMBER']
            drawing_name = row['DRAWING NAME']
            revision = row['REVISION']
            spare_part_number = row['SPARE PART NUMBER']

            # Provide a default file_path or derive logically
            file_path = "default/path"
            initializer_logger.debug(f"Processing row {index + 1}: {row.to_dict()}")

            # Create a new Drawing entry
            new_drawing = Drawing(
                drw_equipment_name=equipment_name,
                drw_number=drawing_number,
                drw_name=drawing_name,
                drw_revision=revision,
                drw_spare_part_number=spare_part_number,
                file_path=file_path
            )
            session.add(new_drawing)
            initializer_logger.info(f"Added new drawing: {drawing_number} - {drawing_name}")

        # Commit all changes
        session.commit()
        initializer_logger.info("All data successfully inserted into the database.")

    except Exception as e:
        # Rollback in case of error
        initializer_logger.error(f"An error occurred: {e}", exc_info=True)
        session.rollback()
    finally:
        # Close session
        session.close()
        initializer_logger.debug("SQLAlchemy session closed.")

    initializer_logger.info("Active drawing list data insertion process completed.")

if __name__ == "__main__":
    try:
        main()
    finally:
        # Close out the logger if it does something fancy (e.g., rotating file)
        close_initializer_logger()
