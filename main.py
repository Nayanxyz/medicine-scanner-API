import os
import json
import csv
import io
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from google import genai
import psycopg2
import psycopg2.extras

# --- INITIALIZATION ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

client = genai.Client(api_key=API_KEY)
app = FastAPI()


# --- DATABASE LOGIC ---
def get_db_connection():
    return psycopg2.connect(DATABASE_URL)


def init_db():
    print("[*] Verifying Supabase PostgreSQL Database...")
    conn = get_db_connection()
    cursor = conn.cursor()
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


init_db()


# --- BACKGROUND TASK ---
def save_medicine_to_db(parsed_data: dict):
    """Saves data to DB without waiting."""
    print("[*] Background DB save initiated...")
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
        print(f"[!] Background DB save failed: {e}")


# --- API MODELS ---
class OCRTextPayload(BaseModel):
    raw_text: str


# --- 1: THE NEW TEXT-ONLY PIPELINE ---
@app.post("/structure-text")
async def structure_text(payload: OCRTextPayload, background_tasks: BackgroundTasks):
    print("[*] Received raw text. Structuring via Gemini...")
    try:
        prompt = f"""
        You are a strict data extraction AI. Extract medical details from the text below.
        Return ONLY a raw JSON object. Do not include markdown, code blocks, or conversational text.
        If a field is not found, use null.

        Format required:
        {{
            "medicine_name": "String",
            "expiry_date": "YYYY-MM",
            "manufacture_date": "YYYY-MM",
            "company": "String"
        }}

        Raw OCR Text:
        {payload.raw_text}
        """

        response = client.models.generate_content(model='gemini-3.5-flash', contents=prompt)

        # Clean the output just in case the LLM ignores strict JSON instructions
        clean_json = response.text.replace('```json', '').replace('```', '').strip()
        parsed_data = json.loads(clean_json)

        # Hand off the DB save to a background thread
        background_tasks.add_task(save_medicine_to_db, parsed_data)

        # Immediately return the data to the user
        return JSONResponse(content=parsed_data)

    except Exception as e:
        print(f"[!] ERROR: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


# --- 2: FETCH HISTORY ---
@app.get("/medicines")
async def get_all_medicines():
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute('SELECT * FROM scanned_medicines ORDER BY id DESC')
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return JSONResponse(content=rows)


# --- 3: EXPORT ---
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


# --- 4: DELETE ---
@app.delete("/medicines/{medicine_id}")
async def delete_medicine(medicine_id: int):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM scanned_medicines WHERE id = %s', (medicine_id,))
        conn.commit()
        cursor.close()
        conn.close()
        return JSONResponse(content={"status": "success"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)