import os
import sys
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from emtacdb_fts import User, UserLevel, Base  # Import your User model and UserLevel enum
from config import DATABASE_URL, ADMIN_CREATION_PASSWORD
from werkzeug.security import generate_password_hash

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure ADMIN_CREATION_PASSWORD is treated as a string
ADMIN_CREATION_PASSWORD = str(ADMIN_CREATION_PASSWORD)
# Log the admin creation password from config for debugging purposes
logger.debug(f"ADMIN_CREATION_PASSWORD from config: {ADMIN_CREATION_PASSWORD} (type: {type(ADMIN_CREATION_PASSWORD)})")

# Configure the database engine
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

def create_initial_admin(admin_password):
    if not ADMIN_CREATION_PASSWORD:
        logger.error("Admin creation password not set. Exiting.")
        return

    admin_password = str(admin_password)  # Ensure entered password is treated as a string
    logger.debug(f"Entered admin password: {admin_password} (type: {type(admin_password)})")

    # Strip any whitespace for a clean comparison
    if admin_password.strip() != ADMIN_CREATION_PASSWORD.strip():
        logger.error("Incorrect password. Exiting.")
        return

    session = Session()

    existing_admin = session.query(User).filter_by(user_level=UserLevel.ADMIN).first()
    if existing_admin:
        logger.info("Admin user already exists. Exiting.")
        return

    admin_user = User(
        employee_id='admin',
        first_name='Admin',
        last_name='User',
        current_shift='Day',
        primary_area='Administration',
        age=30,
        education_level='Masters',
        start_date=None,
        user_level=UserLevel.ADMIN
    )
    admin_user.set_password('admin123')  # Set a secure password

    session.add(admin_user)
    session.commit()

    logger.info("Initial admin user created successfully.")

    session.close()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        logger.error("Usage: python initial_admin.py <admin_creation_password>")
        sys.exit(1)
    
    admin_password = sys.argv[1]
    create_initial_admin(admin_password)
    # Optionally delete the script after execution
    # os.remove(__file__)
