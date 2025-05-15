import cv2 as cv
from PIL import Image
import numpy as np
import os
from pathlib import Path

def preprocess(image_file):
    # Convert uploaded file to grayscale OpenCV image
    pil_image = Image.open(image_file).convert("RGB")
    image = np.array(pil_image)
    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    thresh = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                   cv.THRESH_BINARY_INV, 15, 10)
    # Remove horizontal lines
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 1))
    detected_lines = cv.morphologyEx(thresh, cv.MORPH_OPEN, horizontal_kernel, iterations=2)
    horizontal_removed = cv.subtract(thresh, detected_lines)

    # Remove vertical lines
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 24))
    detected_lines = cv.morphologyEx(horizontal_removed, cv.MORPH_OPEN, vertical_kernel, iterations=2)
    cleaned = cv.subtract(horizontal_removed, detected_lines)
    
    # Denoising
    denoised = cv.fastNlMeansDenoising(cleaned, h=100)

    # Resizing for tesseract
    resized = cv.resize(denoised, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

    sharpened = cv.filter2D(resized, -1, kernel)
    
    # Inverting image(Tesseract expect as default black text on white background)
    inverted = cv.bitwise_not(sharpened)

    return inverted

if __name__ == '__main__':
    images = 'data' # folder 'data' contains cropped images with one line of text to train tesseract to custom dataset
    if not os.path.isdir('image_preprocessed'):
        os.makedirs('image_preprocessed')

    images_dir = Path(images)
    output_dir = Path("image_preprocessed")

    for image_path in images_dir.iterdir():
        if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            # Preprocess
            preprocessed = preprocess(str(image_path), remove_grid=True)

            # Temporary save with OpenCV
            temp_path = output_dir / image_path.name
            cv.imwrite(str(temp_path), preprocessed)

            # Reopen with Pillow and save as PNG
            with Image.open(temp_path) as img:
                png_path = temp_path.with_suffix(".png")
                img.save(png_path)

            # Remove the temporary original format
            temp_path.unlink()
            # with open(os.path.join('image_preprocessed', os.path.splitext(image)[0] + '.gt.txt'), 'w') as f:
            #     pass         