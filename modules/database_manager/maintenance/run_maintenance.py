# C:\Users\10169062\Desktop\AU_IndusMaintdb\modules\database_manager\maintenance\run_maintenance.py
# CLI script to run database maintenance functions

import sys
import os
import subprocess
import argparse


def main():
    """Main entry point for the maintenance CLI"""
    parser = argparse.ArgumentParser(description='Run database maintenance tasks')

    # Add arguments
    parser.add_argument('--task', choices=['associate-drawings'], default='associate-drawings',
                        help='Maintenance task to run (default: associate-drawings)')
    parser.add_argument('--report-dir', type=str, help='Directory to save reports', default=None)
    parser.add_argument('--no-report', action='store_true', help='Do not generate report files')

    # Parse arguments
    args = parser.parse_args()

    # Create the command to run
    # We need to run the db_maintenance.py script with the Click command
    python_exe = sys.executable
    db_maintenance_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'db_maintenance.py')

    if args.task == 'associate-drawings':
        print("Starting drawing-part association...")

        # Build the command to call the Click function
        cmd = [python_exe, db_maintenance_script, 'associate-drawings-with-parts']

        # Add options based on command line args
        if args.report_dir:
            cmd.extend(['--report-dir', args.report_dir])

        if args.no_report:
            cmd.append('--no-export-report')
        else:
            cmd.append('--export-report')

        # Execute the Click command
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"Error: Command failed with exit code {result.returncode}")
            return result.returncode

        print("\nMaintenance complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())