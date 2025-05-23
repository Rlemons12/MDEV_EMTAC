# modules/configuration/config_env.py
import os
import threading
import time
from functools import wraps
from contextlib import contextmanager
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from modules.configuration.config import DATABASE_URL, REVISION_CONTROL_DB_PATH
from modules.configuration.log_config import logger

# Global environment variable that can be set to enable/disable connection limiting
CONNECTION_LIMITING_ENABLED = False

# Global maximum concurrent connections - can also be set via environment variable
MAX_CONCURRENT_CONNECTIONS = int(os.environ.get('MAX_DB_CONNECTIONS', '4'))  # Reduced to 4 for better stability

# Global timeout for acquiring a database connection (in seconds)
CONNECTION_TIMEOUT = int(os.environ.get('DB_CONNECTION_TIMEOUT', '60'))

# Global semaphores for database connection limiting
_main_db_semaphore = threading.Semaphore(MAX_CONCURRENT_CONNECTIONS)
_revision_db_semaphore = threading.Semaphore(MAX_CONCURRENT_CONNECTIONS)

# Global counter for active connections (for monitoring purposes)
_active_main_connections = 0
_active_revision_connections = 0
_connection_lock = threading.Lock()


def with_connection_limiting(func):
    """
    Decorator to apply connection limiting to a function that creates a database session.
    This allows us to maintain backward compatibility with existing code.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        global _active_main_connections, _active_revision_connections

        # Only apply limiting if enabled
        if not CONNECTION_LIMITING_ENABLED:
            return func(*args, **kwargs)

        # Determine which semaphore to use based on the function name
        if 'main' in func.__name__.lower():
            semaphore = _main_db_semaphore
            connection_type = "main"
        else:
            semaphore = _revision_db_semaphore
            connection_type = "revision"

        # Try to acquire semaphore with timeout
        start_time = time.time()
        acquired = False

        while not acquired and (time.time() - start_time < CONNECTION_TIMEOUT):
            acquired = semaphore.acquire(blocking=False)
            if acquired:
                break
            # Log waiting status every 5 seconds
            if int(time.time() - start_time) % 5 == 0:
                logger.debug(f"Waiting for {connection_type} database connection ({int(time.time() - start_time)}s)...")
            time.sleep(0.1)  # Short sleep to prevent CPU spinning

        if not acquired:
            logger.warning(f"Timeout waiting for {connection_type} database connection after {CONNECTION_TIMEOUT}s")
            # Force acquire the semaphore - might cause issues but better than deadlock
            semaphore.acquire(blocking=True)
            logger.info(f"Forced acquisition of {connection_type} database connection after timeout")

        # Update connection counter for monitoring
        with _connection_lock:
            if connection_type == "main":
                _active_main_connections += 1
            else:
                _active_revision_connections += 1

        # Log connection acquisition
        logger.debug(
            f"Acquired {connection_type} database connection. Active: {_active_main_connections} main, {_active_revision_connections} revision")

        # Create the session
        session = func(*args, **kwargs)

        # Patch the session's close method to release the semaphore
        original_close = session.close

        def patched_close():
            global _active_main_connections, _active_revision_connections
            # Call the original close method
            result = original_close()

            # Release the semaphore
            semaphore.release()

            # Update connection counter
            with _connection_lock:
                if connection_type == "main":
                    _active_main_connections = max(0, _active_main_connections - 1)  # Prevent negative counts
                else:
                    _active_revision_connections = max(0, _active_revision_connections - 1)

            # Log connection release
            logger.debug(
                f"Released {connection_type} database connection. Active: {_active_main_connections} main, {_active_revision_connections} revision")

            return result

        # Replace the session's close method with our patched version
        session.close = patched_close

        return session

    return wrapper


class DatabaseConfig:
    def __init__(self):
        # Main database configuration
        self.main_database_url = DATABASE_URL
        self.main_engine = create_engine(self.main_database_url)
        self.MainBase = declarative_base()
        self.MainSession = scoped_session(sessionmaker(bind=self.main_engine))
        self.MainSessionMaker = sessionmaker(bind=self.main_engine)

        # Revision control database configuration
        self.revision_control_db_path = REVISION_CONTROL_DB_PATH
        self.revision_control_engine = create_engine(f'sqlite:///{self.revision_control_db_path}')
        self.RevisionControlBase = declarative_base()
        self.RevisionControlSession = scoped_session(sessionmaker(bind=self.revision_control_engine))
        self.RevisionControlSessionMaker = sessionmaker(bind=self.revision_control_engine)

        # Apply PRAGMA settings
        self._apply_sqlite_pragmas(self.main_engine)
        self._apply_sqlite_pragmas(self.revision_control_engine)

        logger.info(
            f"DatabaseConfig initialized with connection limiting: {CONNECTION_LIMITING_ENABLED}, max connections: {MAX_CONCURRENT_CONNECTIONS}")

    @with_connection_limiting
    def get_main_session(self):
        """
        Return a session from the main database session factory.
        If connection limiting is enabled, this will automatically
        use a semaphore to limit concurrent connections.
        """
        return self.MainSession()

    @with_connection_limiting
    def get_revision_control_session(self):
        """
        Return a session from the revision control database session factory.
        If connection limiting is enabled, this will automatically
        use a semaphore to limit concurrent connections.
        """
        return self.RevisionControlSession()

    @contextmanager
    def main_session(self):
        """
        Context manager for main database sessions.
        Usage:
            with db_config.main_session() as session:
                # use session here
                session.query(...)
        """
        if CONNECTION_LIMITING_ENABLED:
            session = self.get_main_session()  # This will handle connection limiting
        else:
            session = self.MainSessionMaker()

        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    @contextmanager
    def revision_control_session(self):
        """
        Context manager for revision control database sessions.
        Usage:
            with db_config.revision_control_session() as session:
                # use session here
                session.query(...)
        """
        if CONNECTION_LIMITING_ENABLED:
            session = self.get_revision_control_session()  # This will handle connection limiting
        else:
            session = self.RevisionControlSessionMaker()

        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    # Alternative method names for clarity
    @contextmanager
    def get_main_session_context(self):
        """Alias for main_session() context manager."""
        with self.main_session() as session:
            yield session

    @contextmanager
    def get_revision_control_session_context(self):
        """Alias for revision_control_session() context manager."""
        with self.revision_control_session() as session:
            yield session

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

    def get_connection_stats(self):
        """
        Return statistics about database connections.
        Useful for monitoring and debugging connection issues.
        """
        return {
            'connection_limiting_enabled': CONNECTION_LIMITING_ENABLED,
            'max_concurrent_connections': MAX_CONCURRENT_CONNECTIONS,
            'active_main_connections': _active_main_connections,
            'active_revision_connections': _active_revision_connections,
            'connection_timeout': CONNECTION_TIMEOUT
        }

    def _apply_sqlite_pragmas(self, engine):
        def set_sqlite_pragmas(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute('PRAGMA journal_mode = WAL;')
            cursor.execute('PRAGMA synchronous = NORMAL;')
            cursor.execute('PRAGMA temp_store = MEMORY;')
            cursor.execute('PRAGMA cache_size = -64000;')
            cursor.execute('PRAGMA busy_timeout = 5000;')  # Reduced to 5 seconds
            cursor.close()

        event.listen(engine, 'connect', set_sqlite_pragmas)

    def create_documents_fts(self):
        """
        Create the FTS5 virtual table for documents if it doesn't already exist.
        This lets you run full-text queries against (title, content).
        """
        with self.main_session() as session:
            session.execute(
                text(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts "
                    "USING FTS5(title, content)"
                )
            )
            # session.commit() is called automatically by the context manager