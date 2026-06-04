import os

from dotenv import load_dotenv
from fastapi import FastAPI
from google import genai


# --- INITIALIZATION ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)
app = FastAPI()

