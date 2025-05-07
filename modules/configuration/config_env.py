# modules/configuration/config_env.py
from sqlalchemy import create_engine, event, text
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

    def create_documents_fts(self):
        """
        Create the FTS5 virtual table for documents if it doesn't already exist.
        This lets you run full-text queries against (title, content).
        """
        session = self.get_main_session()
        try:
            session.execute(
                text(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts "
                    "USING FTS5(title, content)"
                )
            )
            session.commit()
        finally:
            session.close()

    def load_config_from_db(self):
        """
        Load AI model configuration from the database.

        Returns:
            Tuple of (current_ai_model, current_embedding_model)
        """
        session = self.get_main_session()
        try:
            from modules.emtacdb.emtacdb_fts import AIModelConfig

            ai_model_config = session.query(AIModelConfig).filter_by(key="CURRENT_AI_MODEL").first()
            embedding_model_config = session.query(AIModelConfig).filter_by(key="CURRENT_EMBEDDING_MODEL").first()

            current_ai_model = ai_model_config.value if ai_model_config else "NoAIModel"
            current_embedding_model = embedding_model_config.value if embedding_model_config else "NoEmbeddingModel"

            return current_ai_model, current_embedding_model
        finally:
            session.close()

    def load_image_model_config_from_db(self):
        """
        Load image model configuration from the database.

        Returns:
            String representing the current image model
        """
        session = self.get_main_session()
        try:
            from modules.emtacdb.emtacdb_fts import ImageModelConfig

            image_model_config = session.query(ImageModelConfig).filter_by(key="CURRENT_IMAGE_MODEL").first()
            current_image_model = image_model_config.value if image_model_config else "no_model"

            return current_image_model
        finally:
            session.close()