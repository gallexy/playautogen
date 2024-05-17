# filename: ocr_to_json.py
import pytesseract
import json
import os

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the image
image_path = os.path.join(current_dir, '1.jpg')

# Extract text from image with Chinese language specification
text = pytesseract.image_to_string(image_path, lang='chi_sim')

# Split text into lines using splitlines()
lines = text.splitlines()

# Create a dictionary with lines as a list
data = {'lines': lines}

# Convert dictionary to JSON
json_data = json.dumps(data, ensure_ascii=False) 

# Save JSON data to a file named 'output.json'
with open('output.json', 'w', encoding='utf-8') as f:
    f.write(json_data)

print("OCR output with lines saved to output.json")