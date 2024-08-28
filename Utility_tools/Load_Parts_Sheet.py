import os
import pandas as pd
from emtacdb_fts import Part  # Ensure this import matches your actual file structure
from config import DB_LOADSHEET
from config_env import DatabaseConfig  # Ensure this import matches your actual file structure

# Initialize the database configuration
db_config = DatabaseConfig()

# Specify the file name and path
load_sheet_filename = "load_MP2_ITEMS_BOMS.xlsx"
file_path = os.path.join(DB_LOADSHEET, load_sheet_filename)

# Check if the file exists
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")

# Create a session using the DatabaseConfig class
session = db_config.get_main_session()

# Load the Excel file using the path from the config file
df = pd.read_excel(file_path)

# Fetch all existing part numbers at once
existing_parts = session.query(Part.part_number).all()
existing_part_numbers = set([part[0] for part in existing_parts])

# Prepare the list of new parts to insert
new_parts = []
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
    else:
        print(f"Duplicate part number found: {part_number}. Skipping this entry.")

# Use bulk_insert_mappings for faster insertion
if new_parts:
    session.bulk_insert_mappings(Part, new_parts)
    session.commit()

print("Data has been loaded successfully")

# Close the session after completion
db_config.MainSession.remove()  # Correctly call remove() on the scoped session
