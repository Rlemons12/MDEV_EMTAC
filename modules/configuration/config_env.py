from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from modules.configuration.config import DATABASE_URL, REVISION_CONTROL_DB_PATH

class DatabaseConfig:
    def __init__(self):
        # Main database configuration
        self.main_database_url = DATABASE_URL
        self.main_engine = create_engine(self.main_database_url)
        self.MainBase = declarative_base()
        self.MainSession = scoped_session(sessionmaker(bind=self.main_engine))

        # Revision control database configuration
        self.revision_control_db_path = REVISION_CONTROL_DB_PATH
        self.revision_control_engine = create_engine(f'sqlite:///{self.revision_control_db_path}')
        self.RevisionControlBase = declarative_base()
        self.RevisionControlSession = scoped_session(sessionmaker(bind=self.revision_control_engine))

        # Apply PRAGMA settings
        self._apply_sqlite_pragmas(self.main_engine)
        self._apply_sqlite_pragmas(self.revision_control_engine)

    def get_main_session(self):
        return self.MainSession()

    def get_revision_control_session(self):
        return self.RevisionControlSession()

    def get_main_base(self):
        return self.MainBase

    def get_revision_control_base(self):
        return self.RevisionControlBase

    def get_main_session_registry(self):
        """Return the scoped_session registry for the main database."""
        return self.MainSession

    def get_revision_control_session_registry(self):
        """Return the scoped_session registry for the revision control database."""
        return self.RevisionControlSession

    def _apply_sqlite_pragmas(self, engine):
        def set_sqlite_pragmas(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute('PRAGMA synchronous = OFF;')
            cursor.execute('PRAGMA journal_mode = MEMORY;')
            cursor.execute('PRAGMA temp_store = MEMORY;')
            cursor.execute('PRAGMA cache_size = -64000;')
            cursor.close()

        event.listen(engine, 'connect', set_sqlite_pragmas)
