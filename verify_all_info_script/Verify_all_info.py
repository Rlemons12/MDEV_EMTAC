import time
from tkinter import Tk, messagebox
import os
from pywinauto import Desktop
import logging

# Set up logging
logging.basicConfig(filename='document_reminder.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def show_reminder():
    root = Tk()
    root.withdraw()
    messagebox.showinfo("Reminder", "Please verify all information before using this document.")
    root.destroy()

def monitor_documents():
    extensions = ['.doc', '.docx', '.pdf', '.odt', '.txt']

    logging.info("Document monitor started.")
    
    while True:
        windows = Desktop(backend="uia").windows()
        
        for win in windows:
            try:
                title = win.window_text().lower()
                if any(title.endswith(ext) for ext in extensions):
                    logging.info(f"Document detected: {title}")
                    show_reminder()
                    time.sleep(10)
            except Exception as e:
                logging.error(f"Error: {e}")
        
        time.sleep(5)

if __name__ == "__main__":
    monitor_documents()
