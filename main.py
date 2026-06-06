import os
import json
import csv
import io
import base64
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from google import genai
from google.genai import types
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
    print("Verifying Supabase PostgreSQL Database...")
    conn = get_db_connection()
    cursor = conn.cursor()
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


def save_medicine_to_db(parsed_data: dict):
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
class VisionPayload(BaseModel):
    images_base64: List[str]


class MedicineDataSchema(BaseModel):
    medicine_name: str = Field(description="The commercial name of the medicine. Return 'Unknown' if not found.")
    expiry_date: str = Field(description="Expiry date formatted as YYYY-MM. Return 'Unknown' if not found.")
    manufacture_date: str = Field(description="Manufacturing date formatted as YYYY-MM. Return 'Unknown' if not found.")
    mrp: str = Field(description="Maximum Retail Price. Return 'Unknown' if not found.")
    company: str = Field(description="The manufacturing company. Return 'Unknown' if not found.")


def ask_gemini_vision(model_name: str, prompt: str, base64_images: List[str]) -> dict:
    print(f" Attempting {model_name} Vision...")
    contents = [prompt]

    # Convert base64 strings back to bytes for Gemini Vision
    for b64_str in base64_images:
        image_bytes = base64.b64decode(b64_str)
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'))

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MedicineDataSchema,
        ),
    )
    return json.loads(response.text)


def ask_grok_vision(prompt: str, base64_images: List[str]) -> dict:
    print(" Attempting Grok Vision Fallback...")
    GROK_API_KEY = os.getenv("GROK_API_KEY")
    if not GROK_API_KEY:
        raise ValueError("Grok API key missing.")

    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}

    content_list = [{"type": "text", "text": prompt}]
    for b64_str in base64_images:
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}
        })

    payload = {
        "model": "grok-2-vision",
        "messages": [
            {"role": "system", "content": "You output strict JSON matching the requested schema. No markdown."},
            {"role": "user", "content": content_list}
        ],
        "response_format": {"type": "json_object"}
    }

    response = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return json.loads(response.json()['choices'][0]['message']['content'])


# --- 1: THE VISION PIPELINE ---
@app.post("/scan")
async def process_images(payload: VisionPayload):
    try:
        prompt = """
        You are a highly precise medical data extraction AI. Look at the provided images of medicine packaging and extract the exact inventory details.

        CRITICAL RULES FOR ACCURACY:
        1. Indian medicine packaging clusters text. You must carefully separate Batch Numbers (B.No) from Dates.
        2. Manufacture Date is abbreviated as "MFD", "MFG", or "PKD".
        3. Expiry Date is abbreviated as "EXP" or "USE BY".
        4. MRP is written as "Max. Retail Price" or "Inclusive of all taxes".
        5. DO NOT GUESS OR HALLUCINATE. Return "Unknown" if unreadable.

        Expected JSON format keys: medicine_name, expiry_date, manufacture_date, mrp, company.
        """

        parsed_data = None

        # THE WATERFALL EXECUTION
        try:
            parsed_data = ask_gemini_vision('gemini-2.5-flash', prompt, payload.images_base64)
        except Exception as e1:
            print(f"Gemini 2.5 Failed: {e1}")
            try:
                parsed_data = ask_grok_vision(prompt, payload.images_base64)
            except Exception as e2:
                print(f"Grok Failed: {e2}")
                try:
                    parsed_data = ask_gemini_vision('gemini-1.5-flash', prompt, payload.images_base64)
                except Exception as e3:
                    print(f" Gemini 1.5 Failed: {e3}")
                    return JSONResponse(content={"error": "All AI inference engines offline."}, status_code=503)

        if parsed_data:
            save_medicine_to_db(parsed_data)
            return JSONResponse(content=parsed_data)

    except Exception as fatal_error:
        print(f"FATAL ERROR: {fatal_error}")
        return JSONResponse(content={"error": str(fatal_error)}, status_code=500)


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