import pytesseract
from PIL import Image
from preprocessing import remove_grid_lines
import cv2 as cv

try:
    # preprocessed = cv.imread(r'image_preprocessed/train_kaggle/TRAIN_00001.png')
    preprocessed = cv.imread(r'result.png')
except Exception as e:
    print('Exception:',e)


cv.imshow('w', preprocessed)
cv.waitKey(0)

text = pytesseract.image_to_string(preprocessed, lang='eng')
print(text)