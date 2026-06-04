import os
import cv2
import json
import sqlite3
import numpy as np
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from google import genai
import io
import csv

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


# --- 1: THE SCANNER & SAVER ---
@app.post("/extract-medicine")
async def extract_medicine(front_image: UploadFile = File(...), back_image: UploadFile = File(...)):
    print("Received POST request. Extracting and saving data...")
    try:
        front_bytes = await front_image.read()
        back_bytes = await back_image.read()

        cleaned_images = [
            clean_and_prepare_image(front_bytes),
            clean_and_prepare_image(back_bytes)
        ]

        prompt = """
        You are an expert pharmaceutical data extraction AI. 
        Analyze the front and back images of this medicine.
        Extract the combined data and return ONLY a raw JSON object.
        {
            "medicine_name": "String",
            "expiry_date": "YYYY-MM",
            "manufacture_date": "YYYY-MM",
            "company": "String"
        }
        """

        payload = [prompt] + cleaned_images
        response = client.models.generate_content(model='gemini-3.5-flash', contents=payload)

        clean_json = response.text.replace('```json', '').replace('```', '').strip()
        parsed_data = json.loads(clean_json)

        # SAVE TO DATABASE
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO scanned_medicines (medicine_name, expiry_date, manufacture_date, company, scan_timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            parsed_data.get("medicine_name"),
            parsed_data.get("expiry_date"),
            parsed_data.get("manufacture_date"),
            parsed_data.get("company"),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()

        # Attach the new database ID to the response so the mobile app knows it saved
        parsed_data["db_id"] = cursor.lastrowid
        conn.close()

        return JSONResponse(content=parsed_data)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ---2: FETCH FOR MOBILE APP ---
@app.get("/medicines")
async def get_all_medicines():
    print("[*] Mobile app requested medicine history.")
    conn = sqlite3.connect(DB_FILE)
    # Configure SQLite to return dictionaries instead of raw tuples
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM scanned_medicines ORDER BY id DESC')
    rows = cursor.fetchall()
    conn.close()

    # Convert SQL rows to a JSON array for the mobile app
    return JSONResponse(content=[dict(row) for row in rows])


# ---3: EXPORT TO EXCEL/CSV ---
@app.get("/export")
async def export_medicines_csv():
    print("[*] User requested CSV export.")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM scanned_medicines ORDER BY id DESC')
    rows = cursor.fetchall()

    # Extract column names dynamically
    column_names = [description[0] for description in cursor.description]
    conn.close()

    # Build a virtual CSV file in memory
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(column_names)
    writer.writerows(rows)

    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=scanned_medicines.csv"
    return response