import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from flask import Flask, request, jsonify, render_template
import math
import numpy as np
from PIL import Image
from transformers import BartTokenizer, BartForConditionalGeneration
import gc

from utils import preprocessing

app = Flask(__name__)
device = "cpu"

model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name).to(device)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_image():
    image_file = request.files["image"]
    remove_grid_line = request.form.get("removeGridLine", "false") == "true"

    image = Image.open(image_file).convert("RGB")
    if remove_grid_line:
        image = preprocessing.preprocess(image_file)
    else:
        image = np.array(image)

    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)

    # OCR
    results = reader.readtext(image)
    extracted_text = " ".join([res[1] for res in results])
    del reader  # Free EasyOCR memory
    gc.collect()

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
        num_beams=2,  # Reduced beams for lower memory usage
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    app.run(debug=False)
