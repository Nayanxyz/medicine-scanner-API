import os
import cv2
import sqlite3
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI
from google import genai



# --- INITIALIZATION ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)
app = FastAPI()

# --- DATABASE ARCHITECTURE ---
DB_FILE = "medicines.db"


def init_db():
    print("Initializing SQLite Database...")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scanned_medicines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            medicine_name TEXT,
            expiry_date TEXT,
            manufacture_date TEXT,
            company TEXT,
            scan_timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()


# Run the DB init on startup
init_db()


def clean_and_prepare_image(file_bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    height, width = img.shape[:2]
    max_dim = 1024
    if width > max_dim or height > max_dim:
        scaling_factor = max_dim / float(max(width, height))
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


