#!/usr/bin/env python3
"""
EMTAC Database Audit System Setup Script
Integrates with your existing setup process to add comprehensive auditing
"""

import os
import sys
import subprocess
from datetime import datetime

# Add the project root to the path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, project_root)

try:
    from modules.configuration.config_env import DatabaseConfig
    from modules.emtacdb.emtacdb_fts import Base as MainBase
    from modules.configuration.log_config import info_id, warning_id, error_id
    LOGGING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import EMTAC modules: {e}")
    LOGGING_AVAILABLE = False

    def info_id(msg, **kwargs):
        print(f"INFO: {msg}")

    def warning_id(msg, **kwargs):
        print(f"WARNING: {msg}")

    def error_id(msg, **kwargs):
        print(f"ERROR: {msg}")

# Basic audit system setup (simplified version)
def setup_basic_audit_system():
    """Set up a basic audit system"""
    try:
        info_id("Setting up basic audit system...")

        db_config = DatabaseConfig()

        # Create basic audit table
        with db_config.main_session() as session:
            from sqlalchemy import text

            # Create audit_log table
            create_audit_table_sql = """
            CREATE TABLE IF NOT EXISTS audit_log (
                id SERIAL PRIMARY KEY,
                table_name VARCHAR(100) NOT NULL,
                record_id VARCHAR(100) NOT NULL,
                operation VARCHAR(10) NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id VARCHAR(100),
                user_name VARCHAR(200),
                session_id VARCHAR(100),
                old_values TEXT,
                new_values TEXT,
                changed_fields TEXT,
                ip_address VARCHAR(50),
                user_agent TEXT,
                application VARCHAR(100) DEFAULT 'EMTAC',
                notes TEXT
            );
            """

            session.execute(text(create_audit_table_sql))

            # Create indexes
            index_statements = [
                "CREATE INDEX IF NOT EXISTS idx_audit_log_table_record ON audit_log(table_name, record_id);",
                "CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp DESC);",
                "CREATE INDEX IF NOT EXISTS idx_audit_log_user ON audit_log(user_id);",
                "CREATE INDEX IF NOT EXISTS idx_audit_log_operation ON audit_log(operation);",
            ]

            for idx_sql in index_statements:
                try:
                    session.execute(text(idx_sql))
                except Exception as e:
                    warning_id(f"Index creation skipped: {e}")

            session.commit()
            info_id("Basic audit system created successfully")

        return True

    except Exception as e:
        error_id(f"Failed to setup basic audit system: {e}")
        return False

def main():
    """Main setup function"""
    try:
        return setup_basic_audit_system()
    except Exception as e:
        error_id(f"Audit setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
