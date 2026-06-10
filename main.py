import os
import json
import csv
import io
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
from contextlib import asynccontextmanager
from google import genai
from google.genai import types
import psycopg2
from psycopg2 import pool
import psycopg2.extras


# 1. INITIALIZATION & ENVIRONMENT

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Load multiplexed keys. Strict filtering removes empty strings or None values.
GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2")
]
GEMINI_KEYS = [k for k in GEMINI_KEYS if k and k.strip()]

if not GEMINI_KEYS:
    raise ValueError("CRITICAL: No Gemini API keys found in environment variables.")


# 2. LIFESPAN & CONNECTION POOLING (THE FIX)

# We use a global connection pool to prevent TCP handshake exhaustion (Database DDoS fix)
db_pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the connection pool and database schema globally ONCE
    global db_pool
    print("Initializing PostgreSQL Connection Pool...")
    try:
        # Min 1 connection, Max 10 connections per worker
        db_pool = psycopg2.pool.SimpleConnectionPool(1, 10, DATABASE_URL)
        if db_pool:
            print("[*] Connection pool created successfully.")

            # Initialize schema using a borrowed connection
            conn = db_pool.getconn()
            try:
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
                print("[*] Database Schema Verified.")
            except Exception as e:
                conn.rollback()
                print(f"[!] Database schema init failed: {e}")
            finally:
                db_pool.putconn(conn)
    except Exception as e:
        print(f"Failed to create connection pool: {e}")

    yield  # Application runs during this yield

    # Shutdown: Cleanly close all database connections
    if db_pool:
        db_pool.closeall()
        print("[*] PostgreSQL Connection Pool safely closed.")


app = FastAPI(lifespan=lifespan)

# Enable CORS for web-dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# 3. DATABASE LOGIC (SYNCHRONOUS GUARANTEE)

def save_medicine_to_db(parsed_data: dict):
    """
    Saves to DB synchronously. Raises an exception if it fails,
    ensuring we do not lie to the client about data integrity.
    """
    global db_pool
    conn = db_pool.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO scanned_medicines (medicine_name, expiry_date, manufacture_date, mrp, company, scan_timestamp)
            VALUES (%s, %s, %s, %s, %s, %s);
        ''', (
            parsed_data.get("medicine_name", "Unknown"),
            parsed_data.get("expiry_date", "Unknown"),
            parsed_data.get("manufacture_date", "Unknown"),
            parsed_data.get("mrp", "Unknown"),
            parsed_data.get("company", "Unknown"),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        cursor.close()
    except Exception as e:
        conn.rollback()  # Prevent poisoned connections in the pool
        raise e
    finally:
        db_pool.putconn(conn)



# 4. AI SCHEMAS & LOGIC

class MedicineDataSchema(BaseModel):
    medicine_name: str = Field(description="The commercial brand name. Return 'Unknown' if not found.")
    expiry_date: str = Field(description="Expiry date (MM/YYYY). Return 'Unknown' if not found.")
    manufacture_date: str = Field(description="Manufacturing date (MM/YYYY). Return 'Unknown' if not found.")
    mrp: str = Field(description="Maximum Retail Price (Include ₹). Return 'Unknown' if not found.")
    company: str = Field(description="Manufacturing company. Return 'Unknown' if not found.")


def ask_gemini_vision_binary(model_name: str, prompt: str, image_bytes_list: List[bytes]) -> dict:
    contents = [prompt]
    for img_bytes in image_bytes_list:
        contents.append(types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'))

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=MedicineDataSchema,
    )

    # Multiplexing Loop
    for index, key in enumerate(GEMINI_KEYS):
        print(f"[*] Attempting {model_name} with Key {index + 1}...")
        try:
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            return json.loads(response.text)

        except Exception as e:
            error_str = str(e)
            print(f"[!] Key {index + 1} failed: {error_str}")

            # Smart Rotation: Only burn the next key if this is a Google-side quota/server error.
            if "429" not in error_str and "503" not in error_str and "500" not in error_str:
                raise Exception(f"Client/Payload error. Aborting rotation. Details: {error_str}")

            if index == len(GEMINI_KEYS) - 1:
                raise Exception(f"All available Gemini keys failed or hit quota. Last error: {error_str}")



# 5. API ENDPOINTS (THREAD-POOLED SYNCHRONOUS)

@app.post("/scan-binary")
def process_binary_image(files: List[UploadFile] = File(...)):
    try:
        # Defense 1: The Empty Payload Trap
        if not files or len(files) == 0:
            return JSONResponse(content={"error": "No images provided in the payload."}, status_code=400)


        image_bytes_list = [f.file.read() for f in files]

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

        # 1. AI Parsing
        try:
            data = ask_gemini_vision_binary('gemini-2.5-flash', prompt, image_bytes_list)
        except Exception as ai_error:
            print(f"[!] AI Engine Exhausted/Failed: {ai_error}")
            return JSONResponse(content={"error": f"AI Parsing Error: {ai_error}"}, status_code=503)

        if not data or not isinstance(data, dict):
            return JSONResponse(content={"error": "AI returned invalid or malformed data."}, status_code=500)

        # 2. Database Save (Silent Data Loss FIX)
        # We wait for this to succeed BEFORE returning the JSON response.
        try:
            save_medicine_to_db(data)
        except Exception as db_error:
            print(f"[!] Critical Database Write Failure: {db_error}")
            return JSONResponse(content={"error": "Data extracted, but database failed to save."}, status_code=500)

        # 3. Success Guarantee
        return JSONResponse(content=data)

    except Exception as fatal_error:
        print(f"[!] FATAL ERROR in Endpoint: {fatal_error}")
        return JSONResponse(content={"error": str(fatal_error)}, status_code=500)


@app.get("/medicines")
def get_all_medicines():
    global db_pool
    conn = db_pool.getconn()
    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute('SELECT * FROM scanned_medicines ORDER BY id DESC')
        rows = cursor.fetchall()
        cursor.close()
        return JSONResponse(content=rows)
    except Exception as e:
        conn.rollback()
        print(f"[!] GET /medicines failed: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        db_pool.putconn(conn)


@app.get("/export")
def export_medicines_csv():
    global db_pool
    conn = db_pool.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM scanned_medicines ORDER BY id DESC')
        rows = cursor.fetchall()

        if not rows:
            cursor.close()
            return JSONResponse(content={"error": "Database is empty."}, status_code=404)

        column_names = [desc[0] for desc in cursor.description]
        cursor.close()

        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(column_names)
        writer.writerows(rows)

        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=supabase_medicines.csv"
        return response
    except Exception as e:
        conn.rollback()
        print(f"[!] GET /export failed: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        db_pool.putconn(conn)


@app.delete("/medicines/{medicine_id}")
def delete_medicine(medicine_id: int):
    global db_pool
    conn = db_pool.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM scanned_medicines WHERE id = %s', (medicine_id,))
        conn.commit()

        if cursor.rowcount == 0:
            cursor.close()
            return JSONResponse(content={"error": "Record not found."}, status_code=404)

        cursor.close()
        return JSONResponse(content={"status": "success", "deleted_id": medicine_id})
    except Exception as e:
        conn.rollback()
        print(f"[!] DELETE /medicines failed: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        db_pool.putconn(conn)