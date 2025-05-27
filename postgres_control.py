import subprocess
import os
import sys
import time
from datetime import datetime

#cd "C:\Users\10169062\Desktop\AU_IndusMaintdb\Database\postgreSQL\pgsql\bin"
#.\pg_ctl.exe -D "C:\Users\10169062\PostgreSQL\data" start



# Set these to your actual paths
BIN_DIR = r"C:\Users\10169062\Desktop\AU_IndusMaintdb\Database\postgreSQL\pgsql\bin"
DATA_DIR = r"C:\Users\10169062\PostgreSQL\data"


def run_pg_ctl(args, timeout=30):
    """Helper to run pg_ctl with given arguments and timeout."""
    cmd = [os.path.join(BIN_DIR, "pg_ctl.exe"), "-D", DATA_DIR] + args

    try:
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=BIN_DIR,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result
    except subprocess.TimeoutExpired:
        print(f"⚠️  Command timed out after {timeout} seconds!")
        print("This might indicate PostgreSQL is taking longer than usual to start/stop.")
        return None
    except FileNotFoundError:
        print(f"❌ pg_ctl.exe not found at: {os.path.join(BIN_DIR, 'pg_ctl.exe')}")
        print("Please check your BIN_DIR path in the script.")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


def start_postgres():
    """Start PostgreSQL server with improved feedback."""
    print("🚀 Starting PostgreSQL server...")

    # Use a log file in the data directory
    log_file = os.path.join(DATA_DIR, "server.log")
    result = run_pg_ctl(["-l", log_file, "start"], timeout=45)

    if result is None:
        print("❌ Failed to start PostgreSQL (command timeout or error)")
        return

    print(f"📊 Return code: {result.returncode}")

    if result.stdout.strip():
        print(f"📤 Output: {result.stdout.strip()}")

    if result.stderr.strip():
        print(f"⚠️  Errors: {result.stderr.strip()}")

    if result.returncode == 0:
        print("✅ PostgreSQL server started successfully!")

        # Wait a moment and verify it's actually running
        print("🔍 Verifying server status...")
        time.sleep(2)
        verify_status()
    else:
        print("❌ Error starting PostgreSQL server")
        print(f"💡 Check the log file: {log_file}")

        # Try to show recent log entries
        show_recent_logs(log_file)

        # Run quick diagnosis automatically
        quick_diagnosis()

        print(f"\n🔧 Quick Manual Commands to Try:")
        print(f'   cd "{BIN_DIR}"')
        if os.name == 'nt':  # Windows
            print(f'   PowerShell: .\\pg_ctl.exe -D "{DATA_DIR}" start')
            print(f'   Command Prompt: pg_ctl.exe -D "{DATA_DIR}" start')
        print("💡 See menu option 7 for complete troubleshooting guide")


def stop_postgres():
    """Stop PostgreSQL server."""
    print("🛑 Stopping PostgreSQL server...")
    result = run_pg_ctl(["stop"])

    if result is None:
        print("❌ Failed to stop PostgreSQL (command timeout or error)")
        return

    if result.returncode == 0:
        print("✅ PostgreSQL server stopped successfully.")
    else:
        print("❌ Error stopping PostgreSQL server:")
        if result.stderr.strip():
            print(f"⚠️  Error details: {result.stderr.strip()}")


def status_postgres():
    """Check PostgreSQL server status with detailed info."""
    print("🔍 Checking PostgreSQL server status...")
    result = run_pg_ctl(["status"])

    if result is None:
        print("❌ Failed to check status (command timeout or error)")
        return

    if "server is running" in result.stdout:
        print("✅ PostgreSQL server is RUNNING")

        # Extract PID if available
        lines = result.stdout.split('\n')
        for line in lines:
            if "PID" in line:
                print(f"📋 {line.strip()}")
    elif "no server running" in result.stdout:
        print("❌ PostgreSQL server is NOT running")
    else:
        print("❓ Unable to determine server status")
        if result.stdout.strip():
            print(f"📤 Output: {result.stdout.strip()}")
        if result.stderr.strip():
            print(f"⚠️  Errors: {result.stderr.strip()}")


def verify_status():
    """Quick status verification without extra output."""
    result = run_pg_ctl(["status"])
    if result and "server is running" in result.stdout:
        print("✅ Status verified: Server is running")
        return True
    else:
        print("⚠️  Status check: Server may not be fully started")
        return False


def show_recent_logs(log_file, lines=10):
    """Show recent entries from PostgreSQL log file."""
    try:
        if os.path.exists(log_file):
            print(f"\n📄 Last {lines} lines from {log_file}:")
            print("-" * 50)
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                for line in recent_lines:
                    print(line.rstrip())
            print("-" * 50)
        else:
            print(f"📄 Log file not found: {log_file}")
    except Exception as e:
        print(f"❌ Could not read log file: {e}")


def check_paths():
    """Verify that required paths exist."""
    print("🔍 Checking configuration...")

    if not os.path.exists(BIN_DIR):
        print(f"❌ BIN_DIR not found: {BIN_DIR}")
        return False

    pg_ctl_path = os.path.join(BIN_DIR, "pg_ctl.exe")
    if not os.path.exists(pg_ctl_path):
        print(f"❌ pg_ctl.exe not found: {pg_ctl_path}")
        return False

    if not os.path.exists(DATA_DIR):
        print(f"❌ DATA_DIR not found: {DATA_DIR}")
        return False

    # Check for essential PostgreSQL files
    config_file = os.path.join(DATA_DIR, "postgresql.conf")
    if not os.path.exists(config_file):
        print(f"❌ PostgreSQL config not found: {config_file}")
        print("💡 You may need to initialize the database with initdb")
        return False

    print("✅ All paths and files look good!")
    return True


def show_connection_info():
    """Display connection information."""
    print("\n🔗 Connection Information:")
    print(f"   Host: localhost")
    print(f"   Port: 5432")
    print(f"   Database: pgsql_emtac")
    print(f"   User: postgres")
    print(f"   Connection string: psql -U postgres -d pgsql_emtac -h localhost")


def quick_diagnosis():
    """Run quick diagnostic checks and provide immediate help."""
    print("\n🔍 Running Quick Diagnosis...")
    print("-" * 30)

    issues_found = []

    # Check 1: Paths exist
    if not os.path.exists(BIN_DIR):
        issues_found.append(f"❌ BIN_DIR not found: {BIN_DIR}")
    else:
        print(f"✅ BIN_DIR exists: {BIN_DIR}")

    pg_ctl_path = os.path.join(BIN_DIR, "pg_ctl.exe")
    if not os.path.exists(pg_ctl_path):
        issues_found.append(f"❌ pg_ctl.exe not found: {pg_ctl_path}")
    else:
        print(f"✅ pg_ctl.exe found")

    if not os.path.exists(DATA_DIR):
        issues_found.append(f"❌ DATA_DIR not found: {DATA_DIR}")
    else:
        print(f"✅ DATA_DIR exists: {DATA_DIR}")

    # Check 2: PostgreSQL config
    config_file = os.path.join(DATA_DIR, "postgresql.conf")
    if not os.path.exists(config_file):
        issues_found.append("❌ Database not initialized (postgresql.conf missing)")
        issues_found.append(f"💡 Run: initdb.exe -D \"{DATA_DIR}\" -U postgres")
    else:
        print("✅ Database appears initialized")

    # Check 3: Port availability
    try:
        result = subprocess.run(["netstat", "-an"], capture_output=True, text=True, timeout=10)
        if ":5432" in result.stdout:
            print("⚠️  Port 5432 is in use (this might be normal if PostgreSQL is running)")
        else:
            print("✅ Port 5432 appears available")
    except:
        print("❓ Could not check port availability")

    # Check 4: Try to get PostgreSQL version
    try:
        version_file = os.path.join(DATA_DIR, "PG_VERSION")
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                version = f.read().strip()
            print(f"✅ PostgreSQL version: {version}")
        else:
            issues_found.append("❌ PG_VERSION file not found")
    except:
        print("❓ Could not read PostgreSQL version")

    # Summary
    if issues_found:
        print(f"\n🚨 Found {len(issues_found)} issue(s):")
        for issue in issues_found:
            print(f"   {issue}")
        print("\n💡 Suggested actions:")
        print("   1. Check menu option 6 for detailed troubleshooting")
        print("   2. Try the manual commands shown above")
        print("   3. Run as Administrator if permission issues")
    else:
        print("\n✅ No obvious issues detected!")
        print("💡 If you're still having problems:")
        print("   1. Check recent logs (menu option 4)")
        print("   2. Try manual commands (menu option 6)")


def troubleshooting_guide():
    """Display comprehensive troubleshooting instructions."""
    print("\n🔧 PostgreSQL Troubleshooting Guide")
    print("=" * 40)

    print("\n📋 STEP 1: Manual Command Testing")
    print("-" * 30)
    print("If the script isn't working, try these manual commands:")
    print()
    print("🖥️  Option A: Using PowerShell")
    print(f'   cd "{BIN_DIR}"')
    print(f'   .\\pg_ctl.exe -D "{DATA_DIR}" status')
    print(f'   .\\pg_ctl.exe -D "{DATA_DIR}" start')
    print("   ⚠️  Note: PowerShell requires .\\ prefix!")
    print()
    print("🖥️  Option B: Using Command Prompt (easier)")
    print("   Press Win+R, type 'cmd', press Enter, then:")
    print(f'   cd "{BIN_DIR}"')
    print(f'   pg_ctl.exe -D "{DATA_DIR}" status')
    print(f'   pg_ctl.exe -D "{DATA_DIR}" start')

    print("\n📋 STEP 2: Check File Existence")
    print("-" * 30)
    print("Verify these files exist:")
    print(f'   dir "{os.path.join(BIN_DIR, "pg_ctl.exe")}"')
    print(f'   dir "{DATA_DIR}"')
    print(f'   dir "{os.path.join(DATA_DIR, "postgresql.conf")}"')

    print("\n📋 STEP 3: Database Initialization")
    print("-" * 30)
    print("If DATA_DIR is empty or missing postgresql.conf:")
    print(f'   cd "{BIN_DIR}"')
    print(f'   initdb.exe -D "{DATA_DIR}" -U postgres')

    print("\n📋 STEP 4: Port Conflicts")
    print("-" * 30)
    print("Check if port 5432 is already in use:")
    print("   netstat -an | findstr :5432")
    print("If another service is using port 5432, either:")
    print("   - Stop that service")
    print("   - Change PostgreSQL port in postgresql.conf")

    print("\n📋 STEP 5: Log File Analysis")
    print("-" * 30)
    log_file = os.path.join(DATA_DIR, "server.log")
    print(f"Check the log file for errors:")
    print(f'   type "{log_file}"')
    print("Look for ERROR or FATAL messages")

    print("\n📋 STEP 6: Permissions")
    print("-" * 30)
    print("If you get permission errors:")
    print("   - Run Command Prompt as Administrator")
    print("   - Check folder permissions on DATA_DIR")
    print("   - Ensure user has read/write access to DATA_DIR")

    print("\n📋 STEP 7: Fresh Start")
    print("-" * 30)
    print("For a complete reset:")
    print("1. Stop any running PostgreSQL:")
    print(f'   pg_ctl.exe -D "{DATA_DIR}" stop')
    print("2. Backup your data directory (if needed)")
    print("3. Delete and recreate DATA_DIR:")
    print(f'   rmdir /s "{DATA_DIR}"')
    print(f'   mkdir "{DATA_DIR}"')
    print("4. Initialize fresh database:")
    print(f'   initdb.exe -D "{DATA_DIR}" -U postgres')
    print("5. Start server:")
    print(f'   pg_ctl.exe -D "{DATA_DIR}" start')

    print("\n📋 STEP 8: Connection Testing")
    print("-" * 30)
    print("Once server is running, test connection:")
    print(f'   cd "{BIN_DIR}"')
    print("   psql.exe -U postgres -h localhost")
    print("   # Should connect to PostgreSQL prompt")
    print("   # Type \\q to quit")

    print("\n📋 STEP 9: Common Error Solutions")
    print("-" * 30)
    print("🔸 'pg_ctl.exe not recognized':")
    print("   → Use full path or add .\\ prefix in PowerShell")
    print()
    print("🔸 'database system is starting up':")
    print("   → Wait 10-30 seconds, server is still initializing")
    print()
    print("🔸 'could not connect to server':")
    print("   → Check server is running: pg_ctl status")
    print("   → Check port: netstat -an | findstr :5432")
    print()
    print("🔸 'permission denied':")
    print("   → Run as Administrator")
    print("   → Check DATA_DIR permissions")
    print()
    print("🔸 'port 5432 already in use':")
    print("   → Another PostgreSQL is running")
    print("   → Or another service is using port 5432")

    print("\n📋 STEP 10: Get Help")
    print("-" * 30)
    print("If still having issues:")
    print("1. Run manual commands from STEP 1")
    print("2. Copy the exact error message")
    print("3. Check the log file from STEP 5")
    print("4. Note your Windows version and PostgreSQL version")

    print("\n💡 Quick Diagnosis Commands:")
    print("-" * 30)
    print("Run these to gather info:")
    print(f'echo "PostgreSQL Version:" && type "{os.path.join(DATA_DIR, "PG_VERSION")}"')
    print('echo "Windows Version:" && ver')
    print('echo "Port Check:" && netstat -an | findstr :5432')
    print(f'echo "Files Check:" && dir "{BIN_DIR}\\pg_ctl.exe" && dir "{DATA_DIR}\\postgresql.conf"')


def main():
    """Main menu loop with improved interface."""
    print("🐘 PostgreSQL Server Control Panel")
    print("=" * 35)

    # Check paths on startup
    if not check_paths():
        print("\n❌ Configuration issues detected. Please fix paths and try again.")
        input("Press Enter to exit...")
        return

    while True:
        print(f"\n📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🐘 PostgreSQL Server Control")
        print("=" * 28)
        print("1. 🚀 Start server")
        print("2. 🛑 Stop server")
        print("3. 🔍 Check status")
        print("4. 📄 Show recent logs")
        print("5. 🔗 Show connection info")
        print("6. 🔍 Quick diagnosis")
        print("7. 🔧 Troubleshooting guide")
        print("8. 🚪 Exit")

        choice = input("\nChoose an option [1-8]: ").strip()

        if choice == '1':
            start_postgres()
        elif choice == '2':
            stop_postgres()
        elif choice == '3':
            status_postgres()
        elif choice == '4':
            log_file = os.path.join(DATA_DIR, "server.log")
            show_recent_logs(log_file, 20)
        elif choice == '5':
            show_connection_info()
        elif choice == '6':
            quick_diagnosis()
        elif choice == '7':
            troubleshooting_guide()
        elif choice == '8':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid option. Please enter 1-8.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()