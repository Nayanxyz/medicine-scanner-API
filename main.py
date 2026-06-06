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
import requests


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
    medicine_name: str = Field(description="The commercial name of the medicine. Return 'Unknown' if not found.")
    expiry_date: str = Field(description="Expiry date formatted as YYYY-MM. Return 'Unknown' if not found.")
    manufacture_date: str = Field(description="Manufacturing date formatted as YYYY-MM. Return 'Unknown' if not found.")
    mrp: str = Field(description="Maximum Retail Price. Return 'Unknown' if not found.")
    company: str = Field(description="The manufacturing company. Return 'Unknown' if not found.")


def ask_gemini_2_5(prompt: str) -> dict:
    """Primary Engine: High Intelligence, Google Infra"""
    print("[*] Attempting Gemini 2.5 Flash...")
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MedicineDataSchema,
        ),
    )
    return json.loads(response.text)


def ask_grok(prompt: str) -> dict:
    """Secondary Engine: Multi-Vendor Fallback (xAI)"""
    print("[*] Attempting Grok Fallback...")
    GROK_API_KEY = os.getenv("GROK_API_KEY")
    if not GROK_API_KEY:
        raise ValueError("Grok API key missing.")

    # Grok uses an OpenAI-compatible REST endpoint
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "grok-beta",  # Or grok-2-vision if doing direct images later
        "messages": [
            {"role": "system", "content": "You output strict JSON matching the requested schema. No markdown."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"}
    }

    response = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()

    raw_json_str = response.json()['choices'][0]['message']['content']
    return json.loads(raw_json_str)


def ask_gemini_1_5(prompt: str) -> dict:
    """Tertiary Engine: The Cheap, Reliable Backup"""
    print("[*] Attempting Gemini 1.5 Flash Backup...")
    response = client.models.generate_content(
        model='gemini-1.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MedicineDataSchema,
        ),
    )
    return json.loads(response.text)



# --- 1: THE TEXT-ONLY PIPELINE ---
@app.post("/scan")
async def structure_text(payload: OCRTextPayload):
    print("Received raw text. Structuring via Gemini...")
    try:
        prompt = f"""
        You are a highly precise medical data extraction AI. Extract the exact inventory details from the raw OCR text below.

        CRITICAL RULES FOR ACCURACY:
        1. Indian medicine packaging clusters text. You must carefully separate Batch Numbers (B.No) from Dates.
        2. Manufacture Date is often abbreviated as "MFD", "MFG", or "PKD" (Packed).
        3. Expiry Date is often abbreviated as "EXP" or "USE BY".
        4. MRP is often written as "Max. Retail Price" or "Inclusive of all taxes".
        5. DO NOT GUESS OR HALLUCINATE. If a date is smeared or partially missing, return ONLY the visible numbers. If it is completely unreadable, return "Unknown".

        Raw OCR Text:
        {payload.raw_text}
        """

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MedicineDataSchema,
            ),
        )

        parsed_data = json.loads(response.text)

        # SYNCHRONOUS LOCK: Do not return to the phone until Supabase confirms the save.
        save_medicine_to_db(parsed_data)

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