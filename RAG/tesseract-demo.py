
import pytesseract
from PIL import Image

image_path = "Z:\Python\media\microservices.jpg"
image = Image.open(image_path)

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
extracted_text = pytesseract.image_to_string(image)
print(extracted_text)

