import os
from pathlib import Path

# Personal data
USER_NAME = "Your Name"
USER_EMAIL = "your.email@example.com"

# Base folders
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
CONFIGS_DIR = BASE_DIR / "configs"

IMPORT_PATH = r"C:\Users\amirl\Documents\Education\סמסטר ו'\0.אספקה\הרצאות"

# Paths
DATABASE_PATH = DATA_DIR / "database.db"
LOG_FILE_PATH = LOGS_DIR / "app.log"
SETTINGS_PATH = CONFIGS_DIR / "settings.json"

# Ensure folders exist
for folder in [DATA_DIR, LOGS_DIR, CONFIGS_DIR]:
    os.makedirs(folder, exist_ok=True)