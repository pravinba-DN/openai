import os
from dotenv import load_dotenv
import warnings
from langchain_community.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.evaluation.qa import QAGenerateChain
from langchain.evaluation.qa import QAEvalChain
import langchain


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
        self.llm = None    
        self.examples = []

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
            
            print("Getting ready to load docs >>>", len(self.docs))            
            index = VectorstoreIndexCreator(
                        vectorstore_cls=DocArrayInMemorySearch,
                        embedding=self.embedding
                    ).from_loaders([self.loader])
            
            print("Completed loading docs >>>", len(self.docs))
            self.llm = ChatOpenAI(temperature = 0.0, model=self.model_name)
            
            self.qa_stuff = RetrievalQA.from_chain_type(
            llm=self.llm, 
            chain_type="stuff", 
            retriever=index.vectorstore.as_retriever(search_kwargs={"k": len(self.docs)}), 
            verbose=True,
            chain_type_kwargs = {"document_separator": "<<<<>>>>>"}
            )
            
            
            query =  "Can you give a detailed summary of the dataset? Also, who are the most valuable customers in each product category?"
            response = self.qa_stuff.run(query)
            print(response)

            print("Vector store loaded.")  # Debugging statement
        except Exception as e:
            print(f"Error loading vector database: {e}")
        return None
    
    def build_eval_qa(self):
        self.examples = [
            {'qa_pairs':{
            "query": "How many product categories are there?",
            "answer": "There are 3 product types"
            }},
            {'qa_pairs':{
            "query": "What are the 3 product categories?",
            "answer": "Beauty, Clothing and Electronics"
            }}
        ]
        print("Generating QA Chain with llm:",self.model_name)
        example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(model=self.model_name))
        new_examples = example_gen_chain.apply_and_parse(
                    [{"doc": t} for t in self.docs[:5]]
                    )
        
        self.examples += new_examples
        '''
        print(f"Generated Q&A that we have >>>{len(self.examples)}\n")

        for i, eg in enumerate(self.examples):
            print(f"Example {i}:")
            print("Question: " + self.examples[i]["qa_pairs"]["query"])
            print("Real Answer: " + self.examples[i]["qa_pairs"]["answer"])
        '''

        return None
    
    def manual_evaluation(self):
        langchain.debug = True
        self.qa_stuff.run(self.examples[0]["qa_pairs"]["query"])
        langchain.debug = False
        return None
    
    def auto_evaluation(self):
        print(f"Iterating the Q&A that we have >>>{len(self.examples)}\n")
        
        formatted_examples = [{'query': example['qa_pairs']['query'], 'answer': example['qa_pairs']['answer']} for example in self.examples]

        '''
        for i, eg in enumerate(formatted_examples):
            print(f"Example {i}:")
            print("Question: " + formatted_examples[i]["query"])
            print("Real Answer: " + formatted_examples[i]["answer"])
        '''
      
        print("Starting Evaluation")
        predictions = self.qa_stuff.apply(formatted_examples)
        print("Created predictions based on sample queries",predictions)
        eval_chain = QAEvalChain.from_llm(self.llm)
        print("Created evaluation chain based on llm")
        graded_outputs = eval_chain.evaluate(formatted_examples, predictions)
        #print("Created graded outputs based on examples and predictions:", len(formatted_examples),len(predictions),len(graded_outputs))
        #print(graded_outputs)
        
        for i, eg in enumerate(formatted_examples):
            print(f"Example {i}:")
            print("Question: " + predictions[i]["query"])
            print("Real Answer: " + predictions[i]["answer"])
            print("Predicted Answer: " + predictions[i]["result"])
            print("Predicted Grade: " + graded_outputs[i]["results"])
            print()
        
        return None
    
def main():
    processor = DataProcessor()
    processor.initialize()
    processor.build_eval_qa()
    #processor.manual_evaluation()
    processor.auto_evaluation()
if __name__ == "__main__":
    main()
