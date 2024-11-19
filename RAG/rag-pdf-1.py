from langchain_community.document_loaders import UnstructuredPDFLoader
import htmltabletomd
from IPython.display import HTML, display, Markdown
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import matplotlib.pyplot as plt

doc = "Z:\\Python\\openapi\\RAG\\IF10244.pdf"
# takes 1 min on Colab
'''
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
#poppler_path=r'E:\Softwares\poppler-24.08.0\Library\bin'
#poppler_path='E:\\Softwares\\poppler-24.08.0\\Library\\bin\\'
#print(poppler_path)
# Load environment variables from the .env file
load_dotenv()

# Verify that POPPLER_PATH is correctly loaded
poppler_path = os.getenv('POPPLER_PATH')

if poppler_path:
    print(f'Poppler Path: {poppler_path}')
else:
    print('POPPLER_PATH environment variable not set')
'''
# Load environment variables from the .env file
load_dotenv()

# Verify that POPPLER_PATH is correctly loaded
openapi_key = os.getenv('OPENAPI_KEY')

loader = UnstructuredPDFLoader(file_path=doc, 
                                #convert_from_path=poppler_path,
                                #pdf2text_path=poppler_path,
                                #pdf2image_path=poppler_path,
                                strategy='hi_res',
                                extract_images_in_pdf=True,
                                infer_table_structure=True,
                                # section-based chunking
                                chunking_strategy="by_title",
                                max_characters=4000, # max size of chunks
                                new_after_n_chars=4000, # preferred size of chunks
                                # smaller chunks < 2000 chars will be combined into a larger chunk
                                combine_text_under_n_chars=2000,
                                mode='elements',
                                image_output_dir_path='./figures')
data = loader.load()
'''
print(len(data))
print([doc.metadata['category'] for doc in data])
print(data[0])
print(data[2])
print(data[2].page_content)
print(data[2].metadata['text_as_html'])
'''
md_table = htmltabletomd.convert_table(data[2].metadata['text_as_html'])
print(md_table)

#Load Tables and Documents into separate arrays, while images have already been filtered to ./figures directory

docs = []
tables = []

for doc in data:
    if doc.metadata['category'] == 'Table':
        tables.append(doc)
    elif doc.metadata['category'] == 'CompositeElement':
        docs.append(doc)

print(len(docs), len(tables))

for table in tables:
    table.page_content = htmltabletomd.convert_table(table.metadata['text_as_html'])

#Connect to OpenAI API
chatgpt = ChatOpenAI(model_name='gpt-4o', temperature=0, openai_api_key=openapi_key)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

# Prompt
prompt_text = """
You are an assistant tasked with summarizing tables and text particularly for semantic retrieval.
These summaries will be embedded and used to retrieve the raw text or table elements
Give a detailed summary of the table or text below that is well optimized for retrieval.
For any tables also add in a one line description of what the table is about besides the summary.
Do not add additional words like Summary: etc.
Table or text chunk:
{element}
"""
prompt = ChatPromptTemplate.from_template(prompt_text)

# Summary chain
summarize_chain = (
                    {"element": RunnablePassthrough()}
                      |
                    prompt
                      |
                    chatgpt
                      |
                    StrOutputParser() # extracts response as text
)

#Create text amd table summaries
text_summaries = []
table_summaries = []

text_docs = [doc.page_content for doc in docs]
table_docs = [table.page_content for table in tables]

text_summaries = summarize_chain.batch(text_docs, {"max_concurrency": 5})
table_summaries = summarize_chain.batch(table_docs, {"max_concurrency": 5})
print("Create text summaries >>>>")
#print(text_summaries[0])

#Create image summaries
import base64
import os
from langchain_core.messages import HumanMessage

# create a function to encode images
def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# create a function to summarize the image by passing a prompt to GPT-4o
def image_summarize(img_base64, prompt):
    """Make image summary"""
    chat = ChatOpenAI(model="gpt-4o", temperature=0)
    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": 
                                     f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content

def generate_img_summaries(path):
    """
    Generate summaries and base64 encoded strings for images
    path: Path to list of .jpg files extracted by Unstructured
    """
    # Store base64 encoded images
    img_base64_list = []
    # Store image summaries
    image_summaries = []
    
    # Prompt
    prompt = """You are an assistant tasked with summarizing images for retrieval.
                Remember these images could potentially contain graphs, charts or 
                tables also.
                These summaries will be embedded and used to retrieve the raw image 
                for question answering.
                Give a detailed summary of the image that is well optimized for 
                retrieval.
                Do not add additional words like Summary: etc.
             """
    
    # Apply to images
    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))
    return img_base64_list, image_summaries

# Image summaries
IMG_PATH = './figures'
imgs_base64, image_summaries = generate_img_summaries(IMG_PATH) 

# View the image summary generated by GPT-4o
print("Create image summaries >>>>")
#print(image_summaries[1])

#Embed the summaries with text-embedding-3-small model
from langchain_openai import OpenAIEmbeddings
openai_embed_model = OpenAIEmbeddings(model='text-embedding-3-small')

import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.storage import RedisStore
from langchain_community.utilities.redis import get_client
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

def create_multi_vector_retriever(
    docstore, vectorstore, text_summaries, texts, table_summaries, tables, 
    image_summaries, images
):
    """
    Create retriever that indexes summaries, but returns raw images or texts
    """
    id_key = "doc_id"
    
    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=id_key,
    )
    
    # Helper function to add documents to the vectorstore and docstore
    def add_documents(retriever, doc_summaries, doc_contents):
        print(">>> Add summaries and raw data into ChromaDB and Redis respectively")    
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))
    
    # Add texts, tables, and images
    # Check that text_summaries is not empty before adding
    print(">>> Splitting summaries and raw data ")    
    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    
    # Check that table_summaries is not empty before adding
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    
    # Check that image_summaries is not empty before adding
    if image_summaries:
        add_documents(retriever, image_summaries, images)
    return retriever

print(">>> Connecting to ChromaDB")
#Create vector database - Chroma
# The vectorstore to use to index the summaries and their embeddings
chroma_db = Chroma(
    collection_name="mm_rag",
    embedding_function=openai_embed_model,
    collection_metadata={"hnsw:space": "cosine"},
    #persist_directory="./chromadb", #If you need to persist the ChromaDB data to a storage device for debugging or future reference
)

print(">>> Connecting to Redis")
#Create docstore database - Redis
# Initialize the storage layer - to store raw images, text and tables
client = get_client('redis://localhost:6379')
redis_store = RedisStore(client=client) # you can use filestore, memorystore, any other DB store also

print(">>> Start loading to ChromaDB and Redis")
#Save the details in hand to both vector and docstore databases
# Create retriever
retriever_multi_vector = create_multi_vector_retriever(
    redis_store,  chroma_db,
    text_summaries, text_docs,
    table_summaries, table_docs,
    image_summaries, imgs_base64,
)
print(">>> Loaded to ChromaDB and Redis")

from IPython.display import HTML, display, Image
from PIL import Image
import base64
from io import BytesIO

def plt_img_base64(img_base64):
    """Disply base64 encoded string as image"""
    # Decode the base64 string
    img_data = base64.b64decode(img_base64)
    # Create a BytesIO object
    img_buffer = BytesIO(img_data)
    # Open the image using PIL
    img = Image.open(img_buffer)
    img.show()

import filetype
print(">>> Check Retrieval")
# Check retrieval
query = "Tell me about the annual wildfires trend with acres burned"
docs = retriever_multi_vector.invoke(query, limit=5)
# We get 3 relevant docs
print(len(docs))
print(docs)
for doc in docs:
    # Check if the document is bytes and try to determine the type
    if isinstance(doc, bytes):
        # Use filetype library to guess the file type from the first few bytes
        kind = filetype.guess(doc)
        if kind:
            print(f"Document is of type: {kind.mime} (extension: {kind.extension})")
        else:
            print("Could not determine the document type.")
    else:
        # Handle cases where the document is not in bytes, e.g., a string or dictionary
        print("Document is not in bytes.")

print(">>> Retrieval Output")
# view retrieved image
# plt_img_base64(docs[2])

import re
import base64

# helps in detecting base64 encoded strings
def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

# helps in checking if the base64 encoded image is actually an image
def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False

# returns a dictionary separating images and text (with table) elements
def split_image_text_types(docs):
    """
    Split base64-encoded images and texts (with tables)
    """
    b64_images = []
    texts = []
    for doc in docs:
        # Check if the document is of type Document and extract page_content if so
        if isinstance(doc, Document):
            doc = doc.page_content.decode('utf-8')
        else:
            doc = doc.decode('utf-8')
        if looks_like_base64(doc) and is_image_data(doc):
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}

'''
# Check retrieval and split docs according to their data type
print(">>> Retrieve again but with new query which should return image & text ")
query = "Tell me detailed statistics of the top 5 years with largest wildfire acres burned"
docs = retriever_multi_vector.invoke(query, limit=5)
r = split_image_text_types(docs)
#print(r)
print(">>> Retrieval Output")
'''

from operator import itemgetter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage

def multimodal_prompt_function(data_dict):
    """
    Create a multimodal prompt with both text and image context.
    This function formats the provided context from `data_dict`, which contains
    text, tables, and base64-encoded images. It joins the text (with table) portions
    and prepares the image(s) in a base64-encoded format to be included in a 
    message.
    The formatted text and images (context) along with the user question are used to
    construct a prompt for GPT-4o
    """
    print("Combines text and images >>>>")
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []
    
    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)
    
    # Adding the text for analysis
    text_message = {
        "type": "text",
        "text": (
            f"""You are an analyst tasked with understanding detailed information 
                and trends from text documents,
                data tables, and charts and graphs in images.
                You will be given context information below which will be a mix of 
                text, tables, and images usually of charts or graphs.
                Use this information to provide answers related to the user 
                question.
                Do not make up answers, use the provided context documents below and 
                answer the question to the best of your ability.
                
                User question:
                {data_dict['question']}
                
                Context documents:
                {formatted_texts}
                
                Answer:
            """
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]

# Create RAG chain
multimodal_rag = (
        {
            "context": itemgetter('context'),
            "question": itemgetter('input'),
        }
            |
        RunnableLambda(multimodal_prompt_function)
            |
        chatgpt
            |
        StrOutputParser()
)

# Pass input query to retriever and get context document elements
retrieve_docs = (itemgetter('input')
                    |
                retriever_multi_vector
                    |
                RunnableLambda(split_image_text_types))

# Below, we chain `.assign` calls. This takes a dict and successively
# adds keys-- "context" and "answer"-- where the value for each key
# is determined by a Runnable (function or chain executing at runtime).
# This helps in having the retrieved context along with the answer generated by GPT-4o
multimodal_rag_w_sources = (RunnablePassthrough.assign(context=retrieve_docs)
                                               .assign(answer=multimodal_rag)
)


# Run multimodal RAG chain

import sys

def check_interactive():
    if hasattr(sys, 'ps1'):
        print("Running in interactive mode")
        return True
    else:
        print("Not running in interactive mode")
        return False


print("Defining the QA logic into a function >>>>")
def multimodal_rag_qa(query):
    response = multimodal_rag_w_sources.invoke({'input': query})
    print('=='*50)
    print('Answer:')
    if(check_interactive()):
        print("<<In interactive mode>1>")
        display(Markdown(response['answer']))
    else:
        print("<<In non-interactive mode>1>")
        print(response['answer'])
    print('--'*50)
    print('Sources:')
    text_sources = response['context']['texts']
    img_sources = response['context']['images']
    for text in text_sources:
        if(check_interactive()):
            print("<<In interactive mode>2>")
            display(Markdown(response['answer']))
        else:
            print("<<In interactive mode>2>")
            print(response['answer'])
        print()
    for img in img_sources:
        plt_img_base64(img)
        print()
    print('=='*50)

#query = "Tell me detailed statistics of the top 5 years with largest wildfire acres burned"
query = "Which year had the worst wildfire ever?"
multimodal_rag_qa(query)