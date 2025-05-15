# app.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from flask import Flask, request, jsonify, render_template
import pytesseract
from utils import preprocessing
import cv2 as cv
import math
from transformers import BartForConditionalGeneration, BartTokenizer


app = Flask(__name__)
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to('cpu')
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_image():
    image_file = request.files["image"]
    remove_grid_line = request.form.get("removeGridLine", "false") == "true"

    image = preprocessing.preprocess(image_file, remove_grid=remove_grid_line)
    cv.imwrite('result.png', image)

    # OCR
    extracted_text = pytesseract.image_to_string(image)
    print('extracted text:',extracted_text) 

    # Summarization
    summary = summarize(extracted_text)

    return jsonify({"summary": summary})

def summarize(text):
    global device, model, tokenizer
    word_count = len(text.split())
    min_length = int(10 * math.sqrt(word_count + 64) - 80)

    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=700, min_length=min_length, length_penalty=1.7, num_beams=4)

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    app.run(debug=True)
