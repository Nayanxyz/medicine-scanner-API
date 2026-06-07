import os
import json
import csv
import io
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types
import psycopg2
import psycopg2.extras
from fastapi.middleware.cors import CORSMiddleware


# --- INITIALIZATION ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# LOAD MULTIPLE KEYS
# Ensure your Render environment variables are named exactly like this
GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2")
]
# Filter out any None/empty values in case you only use 1 key temporarily
GEMINI_KEYS = [k for k in GEMINI_KEYS if k]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your specific frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- DATABASE LOGIC ---
def get_db_connection():
    return psycopg2.connect(DATABASE_URL, connect_timeout=5)


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
    print("Synchronous DB save initiated...")
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
        print("[*] DB save complete.")
    except Exception as e:
        print(f"[!] DB save failed: {e}")


# --- DATA SCHEMAS ---
class MedicineDataSchema(BaseModel):
    medicine_name: str = Field(description="The commercial name of the medicine. Return 'Unknown' if not found.")
    expiry_date: str = Field(description="Expiry date formatted as YYYY-MM. Return 'Unknown' if not found.")
    manufacture_date: str = Field(description="Manufacturing date formatted as YYYY-MM. Return 'Unknown' if not found.")
    mrp: str = Field(description="Maximum Retail Price. Return 'Unknown' if not found.")
    company: str = Field(description="The manufacturing company. Return 'Unknown' if not found.")


# --- MULTI-KEY GEMINI AI ENGINE ---
def ask_gemini_vision_binary(model_name: str, prompt: str, image_bytes_list: List[bytes]) -> dict:
    if not GEMINI_KEYS:
        raise ValueError("No Gemini API keys found in environment. Check Render settings.")

    contents = [prompt]
    for img_bytes in image_bytes_list:
        contents.append(types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'))

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=MedicineDataSchema,
    )

    # THE ROTATION LOOP
    for index, key in enumerate(GEMINI_KEYS):
        print(f"[*] Attempting {model_name} Vision with Key {index + 1}...")
        try:
            # Initialize a fresh client for this specific key
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            return json.loads(response.text)

        except Exception as e:
            print(f"[!] Key {index + 1} failed: {e}")
            # If we are on the last key and it still fails, crash out and tell the user
            if index == len(GEMINI_KEYS) - 1:
                raise Exception(f"All available Gemini keys failed or hit quota. Last error: {e}")


# --- THE ENDPOINT ---
@app.post("/scan-binary")
async def process_binary_image(files: List[UploadFile] = File(...)):
    try:
        image_bytes_list = [await f.read() for f in files]

        # The highly optimized prompt
        prompt = """
        You are a specialized Data Extraction Engine for Indian pharmaceutical packaging.
        Analyze the provided images and extract the exact values for the requested fields.

        ### STRICT EXTRACTION RULES:
        * **medicine_name**: Extract the primary commercial brand name. Do NOT extract generic chemical salts unless no brand name is visible.
        * **company**: Extract the manufacturer or "marketed by" entity.
        * **mrp**: Locate "MRP", "Max Retail Price", or "Incl. of all taxes". Strip all commas and alphabetic characters. Include the ₹ symbol. Distinguish carefully between '0', '8', and '9'.
        * **manufacture_date**: Look for prefixes: MFD, MFG, PKD, or Mfg.Date. Format exactly as MM/YYYY or DD/MM/YYYY.
        * **expiry_date**: Look for prefixes: EXP, Expiry, or Use By. Format exactly as MM/YYYY or DD/MM/YYYY.

        ### NEGATIVE CONSTRAINTS (DO NOT DO THIS):
        1. NEVER confuse a Batch Number (B.No, B., L.No) with a Date or MRP.
        2. NEVER guess a number if it is entirely obscured by glare or damage. If a character is partially visible, infer it based on standard medical date/price formatting.
        3. DO NOT output 'Unknown' unless the section of the box containing that data is completely missing or pitch black.
        """

        # 1. PRIMARY AI WITH ROTATION
        try:
            # Note: I changed this back to gemini-2.0-flash because 2.5-flash often throws 404 errors on certain API versions.
            data = ask_gemini_vision_binary('gemini-2.0-flash', prompt, image_bytes_list)
        except Exception as ai_error:
            print(f"[!] AI Engine completely exhausted: {ai_error}")
            return JSONResponse(content={"error": f"AI Engine Failed: {ai_error}"}, status_code=503)

        # 2. VALIDATION
        if not data or not isinstance(data, dict):
            return JSONResponse(content={"error": "AI returned invalid format."}, status_code=500)

        # 3. DATABASE SAVE
        save_medicine_to_db(data)

        # 4. SUCCESS
        return JSONResponse(content=data)

    except Exception as fatal_error:
        print(f"[!] FATAL ERROR: {fatal_error}")
        return JSONResponse(content={"error": str(fatal_error)}, status_code=500)


# --- FETCH HISTORY ---
@app.get("/medicines")
async def get_all_medicines():
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute('SELECT * FROM scanned_medicines ORDER BY id DESC')
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return JSONResponse(content=rows)


# --- EXPORT ---
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


# --- DELETE ---
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