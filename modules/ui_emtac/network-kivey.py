import os
import sys
import socket
import argparse
from kivy.config import Config
from kivy.core.window import Window

# Get the current directory to help with imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Parent directory might be needed for imports
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def get_local_ip():
    """Get the local IP address of the machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Kivey app on the network')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    args = parser.parse_args()

    local_ip = get_local_ip()

    # Configure Kivy to listen on all network interfaces
    Config.set('network', 'port', str(args.port))
    Config.set('network', 'enable', '1')
    Config.set('modules', 'inspector', '1')  # Enable the inspector module
    Config.set('modules', 'webdebugger', '1')  # Enable the web debugger module
    Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
    Config.write()

    # Set environment variables for network access
    os.environ['KIVY_REMOTE'] = "1"
    os.environ['KIVY_REMOTE_PORT'] = str(args.port)
    os.environ['KIVY_AUDIO'] = 'sdl2'  # Use SDL2 audio
    os.environ['KIVY_WINDOW'] = 'sdl2'  # Use SDL2 window

    print(f"\n=== Starting Kivey Application on Network ===")
    print(f"Local IP Address: {local_ip}")
    print(f"Port: {args.port}")
    print(f"Others on your network can access this app at: http://{local_ip}:{args.port}")
    print("Press Ctrl+C to stop the server")
    print("=======================================\n")

    # Now import the app (after configurations are set)
    try:
        from main_app import MaintenanceTroubleshootingApp

        # Run the application
        app = MaintenanceTroubleshootingApp()
        app.run()
    except ImportError as e:
        print(f"Error importing MaintenanceTroubleshootingApp: {e}")
        print("Make sure this script is in the same directory as main_app.py")
        sys.exit(1)
    except Exception as e:
        print(f"Error running the Kivey application: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)