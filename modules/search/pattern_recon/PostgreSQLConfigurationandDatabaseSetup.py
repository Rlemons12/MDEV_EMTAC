#!/usr/bin/env python3
"""
PostgreSQL Configuration and Database Setup

This script helps configure PostgreSQL paths and provides utilities for
running the pattern generation scripts with your database.
"""

import os
import subprocess
import sys
from pathlib import Path

# PostgreSQL Configuration
BIN_DIR = r"C:\Users\10169062\Desktop\AU_IndusMaintdb\Database\postgreSQL\pgsql\bin"
DATA_DIR = r"C:\Users\10169062\PostgreSQL\data"
PATTERN_RECON_DIR = r"C:\Users\10169062\Desktop\AU_IndusMaintdb\modules\search\pattern_recon"

# Database connection parameters
DB_CONFIG = {
    'host': 'localhost',
    'port': '5432',
    'database': 'emtac',  # Update this
    'user': 'postgres',  # Update this
    'password': 'emtac123'  # Update this
}


class PostgreSQLManager:
    """Manage PostgreSQL operations for pattern generation"""

    def __init__(self):
        self.bin_dir = Path(BIN_DIR)
        self.data_dir = Path(DATA_DIR)
        self.pattern_recon_dir = Path(PATTERN_RECON_DIR)
        self.validate_paths()

    def validate_paths(self):
        """Validate PostgreSQL paths exist"""
        if not self.bin_dir.exists():
            raise FileNotFoundError(f"PostgreSQL bin directory not found: {self.bin_dir}")

        if not self.data_dir.exists():
            raise FileNotFoundError(f"PostgreSQL data directory not found: {self.data_dir}")

        if not self.pattern_recon_dir.exists():
            raise FileNotFoundError(f"Pattern recon directory not found: {self.pattern_recon_dir}")

        # Check for key executables
        required_exes = ['psql.exe', 'pg_dump.exe', 'createdb.exe']
        for exe in required_exes:
            exe_path = self.bin_dir / exe
            if not exe_path.exists():
                print(f"⚠️  Warning: {exe} not found in {self.bin_dir}")

        # Check for pattern scripts
        required_scripts = ['DatabasePatternExtractorandEnhancer.py', 'SearchPatternOptimizerandMerger.py']
        for script in required_scripts:
            script_path = self.pattern_recon_dir / script
            if not script_path.exists():
                print(f"⚠️  Warning: {script} not found in {self.pattern_recon_dir}")

    def get_db_url(self) -> str:
        """Generate database URL for pattern extraction scripts"""
        return (f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
                f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")

    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            psql_path = self.bin_dir / 'psql.exe'
            cmd = [
                str(psql_path),
                '-h', DB_CONFIG['host'],
                '-p', DB_CONFIG['port'],
                '-U', DB_CONFIG['user'],
                '-d', DB_CONFIG['database'],
                '-c', 'SELECT version();'
            ]

            env = os.environ.copy()
            env['PGPASSWORD'] = DB_CONFIG['password']

            result = subprocess.run(cmd, env=env, capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Database connection successful!")
                print(f"PostgreSQL version info:")
                print(result.stdout.strip())
                return True
            else:
                print("❌ Database connection failed!")
                print(f"Error: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False

    def run_sql_file(self, sql_file_path: str) -> bool:
        """Execute a SQL file"""
        try:
            psql_path = self.bin_dir / 'psql.exe'
            cmd = [
                str(psql_path),
                '-h', DB_CONFIG['host'],
                '-p', DB_CONFIG['port'],
                '-U', DB_CONFIG['user'],
                '-d', DB_CONFIG['database'],
                '-f', sql_file_path
            ]

            env = os.environ.copy()
            env['PGPASSWORD'] = DB_CONFIG['password']

            result = subprocess.run(cmd, env=env, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✅ Successfully executed {sql_file_path}")
                return True
            else:
                print(f"❌ Failed to execute {sql_file_path}")
                print(f"Error: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ SQL execution failed: {e}")
            return False

    def backup_database(self, backup_file: str) -> bool:
        """Create database backup before running pattern scripts"""
        try:
            pg_dump_path = self.bin_dir / 'pg_dump.exe'
            cmd = [
                str(pg_dump_path),
                '-h', DB_CONFIG['host'],
                '-p', DB_CONFIG['port'],
                '-U', DB_CONFIG['user'],
                '-d', DB_CONFIG['database'],
                '-f', backup_file,
                '--verbose'
            ]

            env = os.environ.copy()
            env['PGPASSWORD'] = DB_CONFIG['password']

            result = subprocess.run(cmd, env=env, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✅ Database backup created: {backup_file}")
                return True
            else:
                print(f"❌ Backup failed!")
                print(f"Error: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False

    def run_pattern_generation_workflow(self):
        """Run the complete pattern generation workflow"""
        print("🚀 Starting Pattern Generation Workflow")
        print("=" * 50)

        # 1. Test connection
        print("\n1. Testing database connection...")
        if not self.test_connection():
            print("❌ Cannot proceed without database connection")
            return False

        # 2. Create backup
        backup_file = f"backup_before_patterns_{int(time.time())}.sql"
        print(f"\n2. Creating backup: {backup_file}")
        if not self.backup_database(backup_file):
            print("⚠️  Proceeding without backup (risky!)")

        # 3. Run pattern extraction
        print("\n3. Running database pattern extraction...")
        db_url = self.get_db_url()

        # Use full path to the extraction script
        extraction_script = self.pattern_recon_dir / 'DatabasePatternExtractorandEnhancer.py'
        extraction_cmd = [
            sys.executable,
            str(extraction_script),
            '--db-url', db_url,
            '--table', 'part',
            '--sample-size', '5000',
            '--output-sql', 'extracted_patterns.sql',
            '--output-json', 'extracted_data.json'
        ]

        try:
            result = subprocess.run(extraction_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Pattern extraction completed successfully")
                print(result.stdout)
            else:
                print("❌ Pattern extraction failed")
                print(result.stderr)
                return False
        except Exception as e:
            print(f"❌ Failed to run extraction script: {e}")
            return False

        # 4. Run pattern optimization
        print("\n4. Running pattern optimization...")
        optimization_script = self.pattern_recon_dir / 'SearchPatternOptimizerandMerger.py'
        optimization_cmd = [
            sys.executable,
            str(optimization_script),
            '--input-patterns', 'extracted_data.json',
            '--db-url', db_url,
            '--merge-existing',
            '--output-sql', 'optimized_patterns.sql',
            '--output-report', 'optimization_report.txt'
        ]

        try:
            result = subprocess.run(optimization_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Pattern optimization completed successfully")
                print(result.stdout)
            else:
                print("❌ Pattern optimization failed")
                print(result.stderr)
                return False
        except Exception as e:
            print(f"❌ Failed to run optimization script: {e}")
            return False

        # 5. Deploy patterns
        print("\n5. Deploying optimized patterns...")
        if self.run_sql_file('optimized_patterns.sql'):
            print("✅ Patterns deployed successfully!")
        else:
            print("❌ Pattern deployment failed")
            return False

        print("\n🎉 Pattern generation workflow completed successfully!")
        print("\nGenerated files:")
        print("- extracted_patterns.sql")
        print("- extracted_data.json")
        print("- optimized_patterns.sql")
        print("- optimization_report.txt")
        print(f"- {backup_file} (backup)")

        return True


def setup_environment():
    """Set up environment variables for PostgreSQL"""
    os.environ['PATH'] = f"{BIN_DIR};{os.environ.get('PATH', '')}"
    os.environ['PGDATA'] = DATA_DIR
    print(f"✅ Added PostgreSQL bin directory to PATH: {BIN_DIR}")
    print(f"✅ Set PGDATA environment variable: {DATA_DIR}")


def check_python_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        'psycopg2',
        'sqlalchemy',
        'pandas',
        'numpy'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")

    if missing_packages:
        print(f"\n📦 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True


def main():
    """Main function to run the PostgreSQL setup and pattern generation"""
    print("PostgreSQL Pattern Generation Setup")
    print("=" * 40)

    # Update these with your actual database details
    print("\n⚠️  IMPORTANT: Update DB_CONFIG with your database details!")
    print("Current config:")
    for key, value in DB_CONFIG.items():
        if key == 'password':
            print(f"  {key}: {'*' * len(str(value))}")
        else:
            print(f"  {key}: {value}")

    # Setup environment
    print(f"\n🔧 Setting up environment...")
    setup_environment()

    # Check dependencies
    print(f"\n📦 Checking Python dependencies...")
    if not check_python_dependencies():
        print("❌ Please install missing dependencies first")
        return 1

    # Initialize PostgreSQL manager
    try:
        pg_manager = PostgreSQLManager()
        print("✅ PostgreSQL paths validated")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1

    # Ask user if they want to run the full workflow
    response = input("\n🚀 Run complete pattern generation workflow? (y/n): ")
    if response.lower() in ['y', 'yes']:
        if pg_manager.run_pattern_generation_workflow():
            print("\n🎉 All done! Your search patterns are ready!")
            return 0
        else:
            print("\n❌ Workflow failed. Check errors above.")
            return 1
    else:
        print("\n💡 You can run individual scripts manually:")
        print(f"Database URL: {pg_manager.get_db_url()}")
        print("\nExample commands:")
        extraction_script = PATTERN_RECON_DIR + r"\DatabasePatternExtractorandEnhancer.py"
        optimization_script = PATTERN_RECON_DIR + r"\SearchPatternOptimizerandMerger.py"
        print(f'python "{extraction_script}" --db-url "{pg_manager.get_db_url()}"')
        print(f'python "{optimization_script}" --input-patterns extracted_data.json')
        return 0


if __name__ == "__main__":
    import time

    exit(main())