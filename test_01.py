import os
import cv2
import json
from PIL import Image
from dotenv import load_dotenv
from google import genai

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)


def process_medicine_images(image_paths):
    cleaned_images = []

    for path in image_paths:
        print(f"Processing: {os.path.abspath(path)}")
        if not os.path.exists(path):
            return f"ERROR: File missing -> {path}"

