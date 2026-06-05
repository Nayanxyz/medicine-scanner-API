import os
import cv2
import json
import numpy as np
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from google import genai
import io
import csv
import psycopg2
import psycopg2.extras
from typing import Optional

# --- INITIALIZATION ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

client = genai.Client(api_key=API_KEY)
app = FastAPI()


# --- DATABASE ARCHITECTURE (POSTGRESQL) ---
def get_db_connection():
    # Establishes a fresh connection to Supabase
    return psycopg2.connect(DATABASE_URL)


def init_db():
    print("Verifying Supabase PostgreSQL Database...")
    conn = get_db_connection()
    cursor = conn.cursor()
    # Note: Postgres uses SERIAL for auto-incrementing IDs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scanned_medicines (
            id SERIAL PRIMARY KEY,
            medicine_name TEXT,
            expiry_date TEXT,
            manufacture_date TEXT,
            company TEXT,
            scan_timestamp TEXT
        )
    ''')
    conn.commit()
    cursor.close()
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

def save_medicine_to_db(parsed_data: dict):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO scanned_medicines (medicine_name, expiry_date, manufacture_date, company, scan_timestamp)
            VALUES (%s, %s, %s, %s, %s);
        ''', (
            parsed_data.get("medicine_name"),
            parsed_data.get("expiry_date"),
            parsed_data.get("manufacture_date"),
            parsed_data.get("company"),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        cursor.close()
        conn.close()
        print("[*] Background DB save complete.")
    except Exception as e:
        print(f" Background DB save failed: {e}")


# --- 1: THE SCANNER & SAVER ---
@app.post("/extract-medicine")
async def extract_medicine(
        front_image: UploadFile = File(...),
        back_image: Optional[UploadFile] = File(None)  # Now optional
):
    print("Received POST request. Extracting via Gemini...")
    try:
        # Always process the front image
        cleaned_images = [clean_and_prepare_image(await front_image.read())]

        # Only process the back image if the user actually sent one
        if back_image is not None:
            cleaned_images.append(clean_and_prepare_image(await back_image.read()))

        prompt = """
        You are an expert pharmaceutical data extraction AI. 
        Analyze the provided image(s) of this medicine.
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

        print("Saving to Supabase...")
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO scanned_medicines (medicine_name, expiry_date, manufacture_date, company, scan_timestamp)
            VALUES (%s, %s, %s, %s, %s) RETURNING id;
        ''', (
            parsed_data.get("medicine_name"),
            parsed_data.get("expiry_date"),
            parsed_data.get("manufacture_date"),
            parsed_data.get("company"),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        new_id = cursor.fetchone()[0]
        conn.commit()
        parsed_data["db_id"] = new_id

        cursor.close()
        conn.close()

        return JSONResponse(content=parsed_data)

    except Exception as e:
        print(f"[!] ERROR: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# --- 2: FETCH FOR MOBILE APP ---
@app.get("/medicines")
async def get_all_medicines():
    print("[*] Mobile app requested medicine history from Supabase.")
    conn = get_db_connection()
    # RealDictCursor formats the Postgres output as a clean JSON dictionary, matching SQLite's row_factory
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute('SELECT * FROM scanned_medicines ORDER BY id DESC')
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    return JSONResponse(content=rows)


# --- 3: EXPORT TO CSV ---
@app.get("/export")
async def export_medicines_csv():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM scanned_medicines ORDER BY id DESC')
    rows = cursor.fetchall()

    column_names = [desc[0] for desc in cursor.description]

    cursor.close()
    conn.close()

    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(column_names)
    writer.writerows(rows)

    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=supabase_medicines.csv"
    return response

# --- 4: DELETE A RECORD ---
@app.delete("/medicines/{medicine_id}")
async def delete_medicine(medicine_id: int):
    print(f"[*] Deleting DB ID {medicine_id} from Supabase...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM scanned_medicines WHERE id = %s', (medicine_id,))
        conn.commit()
        cursor.close()
        conn.close()
        return JSONResponse(content={"status": "success", "message": f"Deleted ID {medicine_id}"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)