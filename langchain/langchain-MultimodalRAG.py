# langchain_multimodal_rag.py

import pdfplumber
import os
from PIL import Image
from io import BytesIO
from langchain.document_loaders import PDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

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

# Step 2: Load PDF into LangChain's Document Loader
pdf_path = 'your_pdf_file.pdf'
pdf_text, pdf_tables, pdf_images = extract_pdf_data(pdf_path)

# Combine text, table, and image data
documents = []
for table in pdf_tables:
    table_text = '\n'.join(['\t'.join(row) for row in table])  # Convert table rows to text
    documents.append({'page_content': table_text, 'metadata': {'source': 'table'}})

for img in pdf_images:
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()  # Get image byte data
    documents.append({'page_content': '', 'metadata': {'source': 'image', 'image': img_byte_arr}})

# Add extracted text
documents.append({'page_content': pdf_text, 'metadata': {'source': 'text'}})

# Step 3: Convert Text into Embeddings using OpenAIEmbeddings
embedding_model = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embedding_model)

# Step 4: Perform Retrieval-Augmented Generation (RAG)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)

# Step 5: Query the model with a question
query = "What is the main content of the document?"
response = qa_chain.run(query)

print(f"Answer: {response}")

