import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# Ensure that Tesseract is correctly installed and its path is set
# Example: if Tesseract is in a custom location, set the path explicitly
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
poppler_path='E:\\Softwares\\poppler-24.08.0\\Library\\bin'

# Function to extract text from a PDF using Poppler (via pdf2image) and Tesseract OCR
def extract_text_from_pdf(pdf_path):
    # Convert PDF to images using Poppler (pdf2image)
    print("Converting PDF pages to images...")
    images = convert_from_path(pdf_path,poppler_path=poppler_path)

    text_output = ""
    
    # Process each image with Tesseract OCR
    for page_num, image in enumerate(images):
        print(f"Processing page {page_num + 1}...")
        # Perform OCR on the image (i.e., convert image to text)
        page_text = pytesseract.image_to_string(image)
        text_output += f"--- Page {page_num + 1} ---\n{page_text}\n\n"
    
    return text_output

# Path to your PDF file
pdf_path = 'Z:\\Python\\openapi\\RAG\\IF10244.pdf'

# Extract text from PDF
extracted_text = extract_text_from_pdf(pdf_path)

# Optionally, save the extracted text to a file
with open('Z:\\Python\\openapi\\RAG\\extracted_text.txt', 'w', encoding='utf-8') as f:
    f.write(extracted_text)

print("Text extraction complete!")
