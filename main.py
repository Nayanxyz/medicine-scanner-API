import os
import json
import csv
import io
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
from google import genai
from google.genai import types  # Imported for structured output configuration
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
    # Added mrp column to align with frontend requirements
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scanned_medicines (
            id SERIAL PRIMARY KEY,
            medicine_name TEXT,
            expiry_date TEXT,
            manufacture_date TEXT,
            mrp TEXT,
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
    """Saves data to DB without stalling the client response."""
    print("Background DB save initiated...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO scanned_medicines (medicine_name, expiry_date, manufacture_date, mrp, company, scan_timestamp)
            VALUES (%s, %s, %s, %s, %s, %s);
        ''', (
            parsed_data.get("medicine_name"),
            parsed_data.get("expiry_date"),
            parsed_data.get("manufacture_date"),
            parsed_data.get("mrp"),
            parsed_data.get("company"),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        cursor.close()
        conn.close()
        print("[*] Background DB save complete.")
    except Exception as e:
        print(f"[!] Background DB save failed: {e}")


# --- API & DATA SCHEMAS ---
class OCRTextPayload(BaseModel):
    raw_text: str


# Pydantic schema for strict Gemini structured extraction
class MedicineDataSchema(BaseModel):
    medicine_name: Optional[str] = Field(None, description="The commercial name of the medicine")
    expiry_date: Optional[str] = Field(None,
                                       description="Expiry date formatted as YYYY-MM or MM/YY if exact format unavailable")
    manufacture_date: Optional[str] = Field(None, description="Manufacturing date formatted as YYYY-MM or MM/YY")
    mrp: Optional[str] = Field(None, description="Maximum Retail Price, including currency symbol if present")
    company: Optional[str] = Field(None, description="The manufacturing or marketing pharmaceutical company")


# --- 1: THE TEXT-ONLY PIPELINE ---
@app.post("/scan")
async def structure_text(payload: OCRTextPayload, background_tasks: BackgroundTasks):
    print("[*] Received raw text. Structuring via Gemini...")
    try:
        prompt = f"""
        Extract all relevant medical inventory details from the following raw OCR text block.
        Ensure you look carefully for the product name, dates, pricing markings, and manufacturer branding.

        Raw OCR Text:
        {payload.raw_text}
        """

        # Enforce structured native JSON parsing via Google GenAI SDK using valid model
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MedicineDataSchema,
            ),
        )

        # The response text is guaranteed to parse straight into our dictionary structure
        parsed_data = json.loads(response.text)

        # Hand off the DB save to a background thread execution loop
        background_tasks.add_task(save_medicine_to_db, parsed_data)

        return JSONResponse(content=parsed_data)

    except Exception as e:
        print(f"ERROR: {e}")
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