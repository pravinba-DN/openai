import os
from dotenv import load_dotenv
import warnings
from langchain_community.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator

# Suppress all warnings
warnings.filterwarnings("ignore")

class DataProcessor:
    def __init__(self):
        # Initialize instance variables
        self.openapi_key = ""
        self.model_name = ""
        self.file = ""
        self.loader = None
        self.docs = None
        self.embedding = None
        self.vectorstore = None
        self.index = None        

    def initialize(self):
        try:

            # Get the current working directory
            current_directory = os.getcwd()
            # Print the current directory
            print("Current Directory:", current_directory)

            # Load environment variables from .env file
            load_dotenv()
            # Access the OPENAPI_KEY
            self.openapi_key = os.getenv('OPENAPI_KEY')
            self.model_name = "gpt-3.5-turbo"
            self.file = './retail_sales_dataset_short.csv'  # Initialize file            
            print("File initialized to:", self.file)  # Debugging statement
            if os.path.isfile(self.file):
                print(f"The file '{self.file}' exists.")
                self.read_excel()  # Load the file
                self.load_vectordb()  # Load the vector database    
            else:
                print(f"The file '{self.file}' does not exist.")
                        
        except Exception as e:
            print(f"Error in initialization: {e}")

    def read_excel(self):
        try:
            print("Reading and loading >>>", self.file)
            self.loader = CSVLoader(file_path=self.file)
            self.docs = self.loader.load()  # Ensure this loads successfully
            print("<<Documents processed>>>")  # Debugging statement
            return self.docs        
        except FileNotFoundError:
            print(f"Error: The file '{self.file}' was not found.")
        except Exception as e:
            print(f"Error loading documents: {e}")
        return None

    def load_vectordb(self):
        try:
            # Initialize embedding
            self.embedding = OpenAIEmbeddings(api_key=self.openapi_key)  
            #Create the vector store
            #self.vectorstore = DocArrayInMemorySearch(embedding=self.embedding)  
            print("Getting ready to load docs >>>", len(self.docs))
            
            db = DocArrayInMemorySearch.from_documents(
                self.docs, 
                self.embedding
            )
            
            print("Completed loading docs >>>", len(self.docs))
            llm = ChatOpenAI(temperature = 0.0, model=self.model_name)
            
            '''
            print("Building Similarity query")
            query = "List all transactions that are made by customers whose gender is female"
            self.qdocs = db.similarity_search(query)
            print("Size of docs for female",len(self.qdocs))
            rdocs = "".join([self.qdocs[i].page_content for i in range(len(self.qdocs))])
            response = llm.call_as_llm(f"{rdocs} Question: What is the most popular product category purchased by total transaction amount?")
            print(response)
            '''

            retriever = db.as_retriever(search_kwargs={"k": len(self.docs)})
            qa_stuff = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever, 
            verbose=False
            )
            
            
            query =  "Can you give a detailed summary of the dataset? Also, who are the most valuable customers in each product category?"
            response = qa_stuff.run(query)
            print(response)


            '''
            response = index.query(query, llm=llm)
            index = VectorstoreIndexCreator(
            vectorstore_cls=DocArrayInMemorySearch,
            embedding=self.embedding,
            ).from_loaders([self.loader])
            '''
            print("Vector store loaded.")  # Debugging statement
        except Exception as e:
            print(f"Error loading vector database: {e}")

    def run_query(self, query):
        try:
            if self.docs is not None and self.embedding is not None:
                db = DocArrayInMemorySearch.from_documents(self.docs, self.embedding)
                results = db.similarity_search(query)
                print("Search results:", results)
            else:
                print("Error: Docs or embedding not initialized.")
        except Exception as e:
            print(f"Error running query: {e}")

def main():
    processor = DataProcessor()
    processor.initialize()
    #query = "Please suggest a shirt with sunblocking"
    #processor.run_query(query)

if __name__ == "__main__":
    main()
