import subprocess


# List of required packages
packages = [
    'flask',
    'upgrade simplejson',
    'Levenshtein',
    'U spacy',
    'spacy download en_core_web_sm',
    'sqlalchemy',
    'openai==0.28',  # Specific version of OpenAI package
    'docx2pdf',
    'nltk',
    'requests',
    'simplejson',
    'pdfplumber',
    'fuzzywuzzy',
    'comtypes',
    'pywin32',
    'python-docx',
    'python-pptx',
    'spacy',
    'pandas',
    'pyarrow',
    'openpyxl',
    'flask-bcrypt',
    'PyMuPDF'

]

# Function to uninstall a package using pip
def uninstall_package(package):
    subprocess.call(['pip', 'uninstall', '-y', package])

# Uninstall all required packages
for package in packages:
    print(f"Uninstalling {package}...")
    uninstall_package(package)
    print(f"{package} uninstalled successfully!")

print("All required packages are uninstalled.")
