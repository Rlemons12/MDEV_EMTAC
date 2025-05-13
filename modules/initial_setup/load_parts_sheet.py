import os
import pandas as pd
from modules.emtacdb.emtacdb_fts import Part
from modules.configuration.config import DB_LOADSHEET
from modules.configuration.config_env import DatabaseConfig
from modules.initial_setup.initializer_logger import initializer_logger
from modules.database_manager.db_manager import RelationshipManager  # Import the RelationshipManager


def load_equip_boms():
    """
    Main function to load EQUIP_BOMS data into the database and associate parts with matching images.
    """
    initializer_logger.info("Starting EQUIP_BOMS data loading process.")

    try:
        # Initialize the database configuration
        db_config = DatabaseConfig()
        initializer_logger.debug("Database configuration initialized.")

        # Specify the file name and path
        load_sheet_filename = "load_MP2_ITEMS_BOMS.xlsx"
        file_path = os.path.join(DB_LOADSHEET, load_sheet_filename)
        initializer_logger.debug(f"Load sheet file path: {file_path}")

        # Check if the file exists
        if not os.path.isfile(file_path):
            initializer_logger.error(f"The file {file_path} does not exist.")
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Create a session using the DatabaseConfig class
        session = db_config.get_main_session()
        initializer_logger.debug("Database session created.")

        # Specify the sheet name to be processed
        sheet_name = "EQUIP_BOMS"  # Specify the desired sheet name
        initializer_logger.info(f"Processing sheet: {sheet_name}")

        # Load the Excel file using the path from the config file, specifying the 'EQUIP_BOMS' sheet
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        initializer_logger.info(f"Loaded Excel sheet '{sheet_name}' with {len(df)} rows.")

        # Fetch all existing part numbers at once
        existing_parts = session.query(Part.part_number).all()
        existing_part_numbers = set([part[0] for part in existing_parts])
        initializer_logger.debug(f"Fetched {len(existing_part_numbers)} existing part numbers.")

        # Prepare the list of new parts to insert
        new_parts = []
        total_created = 0
        total_duplicates = 0

        for index, row in df.iterrows():
            part_number = row['ITEMNUM']
            if part_number not in existing_part_numbers and part_number not in {p['part_number'] for p in new_parts}:
                new_parts.append({
                    'part_number': part_number,
                    'name': row['DESCRIPTION'],
                    'oem_mfg': row['OEMMFG'],
                    'model': row['MODEL'],
                    'class_flag': row['Class Flag'],
                    'ud6': row['UD6'],
                    'type': row['TYPE'],
                    'notes': row['Notes'],
                    'documentation': row['Specifications']
                })
                initializer_logger.debug(f"Added new part: {part_number}")
                total_created += 1
            else:
                initializer_logger.warning(f"Duplicate part number found: {part_number}. Skipping this entry.")
                total_duplicates += 1

        # Use bulk_insert_mappings for faster insertion
        if new_parts:
            session.bulk_insert_mappings(Part, new_parts)
            session.commit()
            initializer_logger.info(f"Inserted {total_created} new parts into the database.")

            # Get the part numbers of newly inserted parts
            new_part_numbers = [p['part_number'] for p in new_parts]

            # Query the database to get the IDs of the newly inserted parts
            newly_inserted_parts = session.query(Part.id).filter(Part.part_number.in_(new_part_numbers)).all()
            new_part_ids = [part.id for part in newly_inserted_parts]
            initializer_logger.info(f"Retrieved {len(new_part_ids)} IDs for newly inserted parts.")

            # Associate parts with matching images using RelationshipManager
            initializer_logger.info("Starting automatic part-image association process...")
            try:
                with RelationshipManager(session=session, request_id="load-equip-boms") as manager:
                    result = manager.associate_parts_with_images_by_title(part_ids=new_part_ids)
                    manager.commit()

                    # Count total associations created
                    total_associations = sum(len(assocs) for assocs in result.values())
                    initializer_logger.info(f"Created {total_associations} part-image associations automatically.")
            except Exception as e:
                initializer_logger.error(f"Error during part-image association: {e}", exc_info=True)
                # Note: We don't want to roll back the main transaction if only the association fails
        else:
            initializer_logger.info("No new parts to insert.")

        # Log the results
        initializer_logger.info(f"Total items created: {total_created}")
        initializer_logger.info(f"Total duplicates found: {total_duplicates}")

        initializer_logger.info("EQUIP_BOMS data loading process completed successfully.")
        print("Data has been loaded successfully")

    except Exception as e:
        # Rollback in case of an error
        initializer_logger.error(f"An error occurred during the process: {e}", exc_info=True)
        session.rollback()
    finally:
        # Close the session after completion
        db_config.MainSession.remove()
        initializer_logger.debug("Database session closed.")


if __name__ == "__main__":
    load_equip_boms()