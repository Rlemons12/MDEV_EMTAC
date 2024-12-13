import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker
from modules.emtacdb.emtacdb_fts import User, UserLevel  # Import your User model and UserLevel enum
from modules.configuration.config import DATABASE_URL, ADMIN_CREATION_PASSWORD

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure ADMIN_CREATION_PASSWORD is treated as a string
ADMIN_CREATION_PASSWORD = str(ADMIN_CREATION_PASSWORD)
# Log the admin creation password from config for debugging purposes
logger.debug(f"ADMIN_CREATION_PASSWORD from config: {ADMIN_CREATION_PASSWORD} (type: {type(ADMIN_CREATION_PASSWORD)})")

# Create SQLAlchemy engine for the main database
logger.info(f"Creating SQLAlchemy engine with DATABASE_URL: {DATABASE_URL}")
engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = scoped_session(sessionmaker(bind=engine))

def create_initial_admin(admin_password):
    logger.debug("Starting create_initial_admin function.")
    
    if not ADMIN_CREATION_PASSWORD:
        logger.error("Admin creation password not set. Exiting.")
        return

    admin_password = str(admin_password).strip()  # Ensure entered password is treated as a string and stripped of whitespace
    logger.debug(f"Entered admin password: '{admin_password}' (type: {type(admin_password)}, length: {len(admin_password)})")

    # Strip any whitespace for a clean comparison
    if admin_password != ADMIN_CREATION_PASSWORD:
        logger.error("Incorrect password. Exiting.")
        return

    # Use a new session for this function
    session = Session()
    logger.debug("Session started.")

    try:
        existing_admin = session.query(User).filter_by(user_level=UserLevel.ADMIN).first()
        if existing_admin:
            logger.info("Admin user already exists. Exiting.")
            return

        logger.info("Creating initial admin user.")
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
        
        # Use set_password to hash the password, just like in the user creation script
        admin_user.set_password('admin123')  # Set a secure password
        logger.debug(f"Password set for admin user.")

        session.add(admin_user)
        session.commit()

        logger.info("Initial admin user created successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        session.rollback()  # Rollback in case of error
    finally:
        Session.remove()  # Correctly close the session
        logger.debug("Session closed.")

if __name__ == '__main__':
    # Prompt the user for the admin password
    admin_password = input("Enter the admin creation password: ").strip()
    logger.info(f"Admin password received from input: {admin_password}")
    create_initial_admin(admin_password)
    # Optionally delete the script after execution
    # os.remove(__file__)
