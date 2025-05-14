# app.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import pytesseract
from PIL import Image
from utils import preprocessing, summarizing
import cv2 as cv


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_image():
    image_file = request.files["image"]
    image = preprocessing.remove_grid_lines(image_file)
    cv.imwrite('result.png', image)

    # OCR
    extracted_text = pytesseract.image_to_string(image)
    print('extracted text:',extracted_text) 

    # Summarization
    summary = summarizing.summarize(extracted_text)

    return jsonify({"summary": summary})

if __name__ == "__main__":
    app.run(debug=True)
