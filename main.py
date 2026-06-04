import os

import sqlite3

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


