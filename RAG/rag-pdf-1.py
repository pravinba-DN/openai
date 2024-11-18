from langchain_community.document_loaders import UnstructuredPDFLoader
import htmltabletomd
from IPython.display import HTML, display, Markdown
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

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

print(text_summaries[0])

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
print(image_summaries[1])

