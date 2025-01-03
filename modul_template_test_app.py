from flask import Flask, render_template
import webbrowser
import threading
import time

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('module_template_html/test_template.html')

def open_browser(port):
    # Wait a bit to ensure the server starts before opening the browser
    time.sleep(1)
    webbrowser.open(f"http://127.0.0.1:5000")

if __name__ == '__main__':
# Run the app on port 5000
    app.run(debug=True)
