import os
import json
import csv
import io
import base64
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List
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
    # Adding connect_timeout=5 ensures it fails fast if DB is unreachable
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


# --- DATA SCHEMAS ---
class MedicineDataSchema(BaseModel):
    medicine_name: str = Field(description="The commercial name of the medicine. Return 'Unknown' if not found.")
    expiry_date: str = Field(description="Expiry date formatted as YYYY-MM. Return 'Unknown' if not found.")
    manufacture_date: str = Field(description="Manufacturing date formatted as YYYY-MM. Return 'Unknown' if not found.")
    mrp: str = Field(description="Maximum Retail Price. Return 'Unknown' if not found.")
    company: str = Field(description="The manufacturing company. Return 'Unknown' if not found.")



# --- RAW BINARY MULTI-IMAGE AI ENGINES ---
def ask_gemini_vision_binary(model_name: str, prompt: str, image_bytes_list: List[bytes]) -> dict:
    print(f" Attempting {model_name} Vision via Binary Stream...")
    contents = [prompt]

    for img_bytes in image_bytes_list:
        contents.append(types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'))

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MedicineDataSchema,
        ),
    )
    return json.loads(response.text)


def ask_grok_vision_binary(prompt: str, image_bytes_list: List[bytes]) -> dict:
    print("[*] Attempting Grok Vision Fallback...")
    GROK_API_KEY = os.getenv("GROK_API_KEY")
    if not GROK_API_KEY:
        raise ValueError("Grok API key missing.")

    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}

    content_list = [{"type": "text", "text": prompt}]

    for img_bytes in image_bytes_list:
        b64_str = base64.b64encode(img_bytes).decode('utf-8')
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}
        })

    payload = {
        "model": "grok-2-vision-1212",  # The precise versioned model string
        "messages": [
            {"role": "system",
             "content": "You are a precise data extractor. You must output ONLY raw, valid JSON matching the requested keys. Do not use markdown blocks like ```json."},
            {"role": "user", "content": content_list}
        ]
        # Stripped the response_format flag that triggered the 400 crash
    }

    response = requests.post("https://api.x.ai/v1/chat/completions",
                             headers=headers, json=payload)

    if not response.ok:
        print(f"[!] Grok API Error Details: {response.text}")
        response.raise_for_status()

    text_response = response.json()['choices'][0]['message']['content']

    # Sanitize any rogue markdown formatting Grok might return
    text_response = text_response.replace('```json', '').replace('```', '').strip()
    return json.loads(text_response)


def ask_openrouter_vision(prompt: str, image_bytes_list: List[bytes]) -> dict:
    print("[*] Attempting OpenRouter Fallback...")
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "HTTP-Referer": "https://medicine-scanner-api.onrender.com",  # Required by OpenRouter
        "Content-Type": "application/json"
    }

    content_list = [{"type": "text", "text": prompt}]
    for img_bytes in image_bytes_list:
        b64_str = base64.b64encode(img_bytes).decode('utf-8')
        content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}})

    payload = {
        "model": "anthropic/claude-3.5-sonnet",  # Highly accurate for medical OCR
        "messages": [{"role": "user", "content": content_list}]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()

    # Strip markdown and return JSON
    text = response.json()['choices'][0]['message']['content'].replace('```json', '').replace('```', '').strip()
    return json.loads(text)


# --- THE BULLETPROOF ENDPOINT ---
@app.post("/scan-binary")
async def process_binary_image(files: List[UploadFile] = File(...)):
    try:
        # 1. READ INPUTS
        image_bytes_list = [await f.read() for f in files]

        prompt = """
        You are a highly precise medical data extraction AI. Look at these raw, high-resolution images of different sides of a medicine box.

        CRITICAL RULES:
        1. Separate Batch Numbers (B.No) from Dates.
        2. MFD/MFG/PKD = Manufacture Date. EXP/USE BY = Expiry Date.
        3. PRICING ACCURACY: Distinguish clearly between 0, 8, and 9. Include currency symbols (₹ or Rs).
        4. Return "Unknown" if unreadable.
        
        STRICT RULES:
                1. If you see text, DO NOT return 'Unknown'. Make a logical prediction based on character shapes.
                2. If 'MFG' or 'EXP' is blurry, look at surrounding context to infer the date.
                3. For MRP, ignore formatting (like commas or spaces); just return the numbers.
                4. If a field is 80% likely to be correct, output the value. Precision is more important than perfect confidence.

                

        Expected JSON keys: medicine_name, expiry_date, manufacture_date, mrp, company.
        """

        data = None

        # Try Gemini 2.0
        # 1. Primary: Gemini 2.0
        try:
            data = ask_gemini_vision_binary('gemini-2.5-flash', prompt, image_bytes_list)
        except Exception as e1:
            print(f"[!] Gemini 2.5 failed: {str(e1)}")

            # 2. Secondary: OpenRouter
            try:
                data = ask_openrouter_vision(prompt, image_bytes_list)
            except Exception as e2:
                print(f"[!] OpenRouter failed: {str(e2)}")

                # 3. Last Resort: Grok
                try:
                    data = ask_grok_vision_binary(prompt, image_bytes_list)
                except Exception as e3:
                    print(f"[!] Grok failed: {str(e3)}")
                    # THIS IS WHERE YOU ARE HITTING 503
                    return JSONResponse(
                        content={"error": f"All engines failed. Grok error: {str(e3)}"},
                        status_code=503
                    )

        # 3. VALIDATION
        if not data or not isinstance(data, dict):
            return JSONResponse(content={"error": "AI returned invalid format."}, status_code=500)

        # 4. DATABASE SAVE
        try:
            save_medicine_to_db(data)
        except Exception as db_err:
            print(f"[!] DATABASE SAVE FAILED: {db_err}")
            return JSONResponse(content={"error": "Data extracted but failed to save."}, status_code=500)

        # 5. SUCCESS
        return JSONResponse(content=data)

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