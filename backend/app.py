#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
from pathlib import Path
import os, io, base64, json, re

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv("my.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


PROMPT = """
You are a nutrition expert. Return ONLY a single JSON object with EXACTLY these keys:
"food_name", "serving_description", "calories", "fat_grams", "protein_grams", "confidence_level".
If you cannot estimate a value, use null. Do not include any other keys or text.
Analyze the image content to estimate typical values for a normal serving.
"""


def encode_image_to_base64(image_or_path):
    if isinstance(image_or_path, str):
        with open(image_or_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    elif isinstance(image_or_path, Image.Image):
        buf = io.BytesIO()
        image_or_path.save(buf, format=image_or_path.format or "JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    else:
        raise ValueError("Input must be a file path or PIL.Image.Image")

def query_json_only(client, pil_image, prompt, model="gpt-4o", max_tokens=200):

    img_b64 = encode_image_to_base64(pil_image)
    messages = [
        {"role": "system", "content": "You only ever respond with one valid JSON object. No prose."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ],
        },
    ]


    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
    except Exception:

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
        )

    return resp.choices[0].message.content

from flask_cors import CORS
# --- Flask app ---
app = Flask(__name__)
CORS(app, resources={r"/analyze": {"origins": "https://kaursimar9464.github.io"}})
BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()

@app.get("/")
def index():
    # Serve your HTML from the same folder
    return send_file(BASE_DIR / "index.html")

@app.post("/analyze")
def analyze():
    # Basic checks
    if "image" not in request.files:
        return jsonify({"error": "No file part 'image'"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400


    if client is None:
        return jsonify({"error": "Server missing OPENAI_API_KEY (check my.env)."}), 502


    try:
        pil_img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image"}), 400


    try:
        raw = query_json_only(client, pil_img, PROMPT)
    except Exception as e:

        return jsonify({"error": f"Model error: {e}"}), 502


    try:
        data = json.loads(raw)
    except Exception:
        return jsonify({"error": "Model did not return valid JSON", "raw": raw}), 500

    # Keep only expected keys (in case extras sneak in)
    allowed = {"food_name", "serving_description", "calories", "fat_grams", "protein_grams", "confidence_level"}
    cleaned = {k: data.get(k, None) for k in allowed}

    return jsonify(cleaned), 200

if __name__ == "__main__":
    print("Serving from:", BASE_DIR)
    print("Has index.html?", (BASE_DIR / "index.html").exists())
    print("Routes:", app.url_map)
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)


# In[ ]:




