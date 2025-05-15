# app.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from flask import Flask, request, jsonify, render_template
import easyocr
from utils import preprocessing
import math
from PIL import Image
from transformers import BartForConditionalGeneration, BartTokenizer
import numpy as np

app = Flask(__name__)

device = "cpu"
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# Инициализация EasyOCR reader один раз, с нужными языками, например, английский
reader = easyocr.Reader(['en'])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_image():
    image_file = request.files["image"]
    remove_grid_line = request.form.get("removeGridLine", "false") == "true"

    image = Image.open(image_file).convert("RGB")
    image = np.array(image)
    
    if remove_grid_line:
        image = preprocessing.preprocess(image_file)

    # OCR с easyocr
    results = reader.readtext(image)
    extracted_text = " ".join([res[1] for res in results])  # объединяем все распознанные тексты
    print('extracted text:', extracted_text)

    # Summarization
    summary = summarize(extracted_text)

    return jsonify({"summary": summary})

def summarize(text):
    word_count = len(text.split())
    min_length = int(10 * math.sqrt(word_count + 64) - 80)

    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=700,
        min_length=min_length,
        length_penalty=1.7,
        num_beams=4,
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    app.run(debug=True)
