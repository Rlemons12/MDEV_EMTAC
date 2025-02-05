from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker
from modules.emtacdb.emtacdb_fts import User, UserLevel  # Import your User model and UserLevel enum
from modules.configuration.config import DATABASE_URL, ADMIN_CREATION_PASSWORD
from modules.initial_setup.initializer_logger import (
    initializer_logger, close_initializer_logger,
    compress_logs_except_most_recent, LOG_DIRECTORY
)

# Ensure ADMIN_CREATION_PASSWORD is treated as a string
ADMIN_CREATION_PASSWORD = str(ADMIN_CREATION_PASSWORD)


def configure_engine_and_session():
    """
    Configure the SQLAlchemy engine and session.
    """
    initializer_logger.info(f"Creating SQLAlchemy engine with DATABASE_URL: {DATABASE_URL}")
    engine = create_engine(DATABASE_URL)
    Base = declarative_base()
    Session = scoped_session(sessionmaker(bind=engine))
    return Session


def create_initial_admin(admin_password, session):
    """
    Function to create the initial admin user.
    """
    initializer_logger.debug("Starting create_initial_admin function.")

    # Check if admin creation password is set
    if not ADMIN_CREATION_PASSWORD:
        initializer_logger.error("Admin creation password not set. Exiting.")
        return False

    # Validate the entered password
    admin_password = str(admin_password).strip()  # Ensure it's a string and trimmed
    initializer_logger.debug(
        f"Entered admin password: '{admin_password}' (type: {type(admin_password)}, length: {len(admin_password)})")

    if admin_password != ADMIN_CREATION_PASSWORD:
        initializer_logger.error("Incorrect password. Exiting.")
        return False

    try:
        # Check if an admin user already exists
        existing_admin = session.query(User).filter_by(user_level=UserLevel.ADMIN).first()
        if existing_admin:
            initializer_logger.info("Admin user already exists. Exiting.")
            return True

        # Create the admin user
        initializer_logger.info("Creating initial admin user.")
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

        # Hash and set the password
        admin_user.set_password('admin123')  # Set a secure password
        initializer_logger.debug("Password set for admin user.")

        session.add(admin_user)
        session.commit()

        initializer_logger.info("Initial admin user created successfully.")
        return True
    except Exception as e:
        initializer_logger.error(f"An error occurred: {e}")
        session.rollback()
        return False
    finally:
        initializer_logger.debug("Closing the session.")
        session.close()


def main():
    """
    Main function to handle the admin creation process.
    """
    # Configure the engine and session
    session = configure_engine_and_session()

    # Prompt the user for the admin password
    admin_password = input("Enter the admin creation password: ").strip()
    initializer_logger.info(f"Admin password received from input: {admin_password}")

    # Create the initial admin user
    success = create_initial_admin(admin_password, session)

    if success:
        initializer_logger.info("Admin creation process completed successfully.")
    else:
        initializer_logger.info("Admin creation process failed or was not necessary.")

    # Perform cleanup tasks
    close_initializer_logger()
    compress_logs_except_most_recent(LOG_DIRECTORY)


if __name__ == '__main__':
    main()
