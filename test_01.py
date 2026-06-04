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

        # OpenCV Preprocessing
        img = cv2.imread(path)
        height, width = img.shape[:2]
        max_dim = 1024
        if width > max_dim or height > max_dim:
            scaling_factor = max_dim / float(max(width, height))
            img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cleaned_images.append(Image.fromarray(img_rgb))

