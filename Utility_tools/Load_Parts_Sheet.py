import os
import sys
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from emtacdb_fts import Base, Part  # Ensure this import matches your actual file structure
from config import DATABASE_URL, DB_LOADSHEET

# Specify the file name
load_sheet_filename = "load_MP2_ITEMS_BOMS.xlsx"
file_path = os.path.join(DB_LOADSHEET, load_sheet_filename)

# Check if the file exists
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")

# Create the database engine and session
engine = create_engine(DATABASE_URL)
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()

# Load the Excel file using the path from the config file
df = pd.read_excel(file_path)

# Function to check if part_number already exists
def part_exists(part_number):
    return session.query(Part).filter_by(part_number=part_number).first() is not None

# Map the Excel columns to the Part attributes and check for duplicates
for index, row in df.iterrows():
    if not part_exists(row['ITEMNUM']):
        part = Part(
            part_number=row['ITEMNUM'],
            name=row['DESCRIPTION'],
            oem_mfg=row['OEMMFG'],
            model=row['MODEL'],
            class_flag=row['Class Flag'],
            ud6=row['UD6'],
            type=row['TYPE'],
            notes=row['Notes'],
            documentation=row['Specifications']
        )
        session.add(part)
    else:
        print(f"Duplicate part number found: {row['ITEMNUM']}. Skipping this entry.")

# Commit the session to save the parts in the database
session.commit()

print("Data has been loaded successfully")
