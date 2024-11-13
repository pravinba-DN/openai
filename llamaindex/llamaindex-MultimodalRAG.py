# llamaindex_multimodal_rag.py

import pdfplumber
from PIL import Image
from io import BytesIO
from llama_index import GPTSimpleVectorIndex, Document
from llama_index import ServiceContext
from llama_index.llms import OpenAI
from llama_index.vector_stores import ChromaVectorStore

# Step 1: Parse PDF Document (Extract text, tables, and images)
def extract_pdf_data(pdf_path):
    text = ''
    images = []
    tables = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract text
            text += page.extract_text() or ''
            
            # Extract tables
            tables += page.extract_tables()  # List of tables on each page
            
            # Extract images
            for img in page.images:
                im = pdfplumber.utils.PillowImage.from_bytes(img['stream'].get_data())
                images.append(im)  # Append each image found
                
    return text, tables, images

# Step 2: Load PDF Data and Create Documents
pdf_path = 'your_pdf_file.pdf'
pdf_text, pdf_tables, pdf_images = extract_pdf_data(pdf_path)

# Create document objects for text, tables, and images
documents = []

# Add text data as documents
documents.append(Document(text=pdf_text, doc_id="text"))

# Add table data as documents (converting to plain text)
for idx, table in enumerate(pdf_tables):
    table_text = '\n'.join(['\t'.join(row) for row in table])  # Convert table rows to text
    documents.append(Document(text=table_text, doc_id=f"table_{idx}"))

# Add image data as documents (store image byte data)
for idx, img in enumerate(pdf_images):
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()  # Get image byte data
    documents.append(Document(text='', doc_id=f"image_{idx}", metadata={"image": img_byte_arr}))

# Step 3: Initialize the Chroma Vector Store
chroma_db = ChromaVectorStore("chroma_db")

# Step 4: Create a Service Context for LlamaIndex
service_context = ServiceContext.from_defaults(
    llm=OpenAI(),
    vector_store=chroma_db
)

# Step 5: Create and Index the Documents in LlamaIndex
index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

# Step 6: Perform Retrieval-Augmented Generation (RAG)
query = "What is the main content of the document?"
response = index.query(query)

print(f"Answer: {response}")
