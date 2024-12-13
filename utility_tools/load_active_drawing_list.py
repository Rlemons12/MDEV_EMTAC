import pandas as pd
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sys

# Ensure the parent directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.configuration.config import DB_LOADSHEET, DATABASE_URL
from modules.emtacdb.emtacdb_fts import Drawing  # Assuming you have defined the Drawing class

# Create a SQLAlchemy engine using the DATABASE_URL from your config
engine = create_engine(DATABASE_URL)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Load data from the "active drawing list" sheet
load_sheet_path = os.path.join(DB_LOADSHEET, "active drawing list.xlsx")
df = pd.read_excel(load_sheet_path)

# Drop the extra column
df = df.drop(columns=['Unnamed: 7'])

# Rename the remaining columns
df.columns = ['EQUIPMENT NUMBER', 'EQUIPMENT NAME', 'DRAWING NUMBER', 'DRAWING NAME', 'REVISION', 'CC REQUIRED', 'SPARE PART NUMBER']

# Strip leading and trailing spaces from each column
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Iterate through each row in the dataframe
for index, row in df.iterrows():
    # Extract data from the row
    equipment_name = row['EQUIPMENT NAME']
    drawing_number = row['DRAWING NUMBER']
    drawing_name = row['DRAWING NAME']
    revision = row['REVISION']
    spare_part_number = row['SPARE PART NUMBER']

    # Provide a default file_path value or derive it logically
    file_path = "default/path"  # Replace with an appropriate default value or logic to derive the file path

    # Create a new instance of the Drawing class and add it to the session
    new_drawing = Drawing(drw_equipment_name=equipment_name, drw_number=drawing_number,
                           drw_name=drawing_name, drw_revision=revision,
                           drw_spare_part_number=spare_part_number, file_path=file_path)
    session.add(new_drawing)

# Commit the session after adding all new entries
session.commit()

print("Data insertion completed.")
