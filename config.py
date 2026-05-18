import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def ensure_templates_dir():
    os.makedirs(TEMPLATES_DIR, exist_ok=True)