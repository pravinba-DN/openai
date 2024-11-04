import os
from dotenv import load_dotenv
import warnings
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
#from langchain.tools import LLM_Math, Wikipedia
#from langchain.tools import PythonREPLTool
#from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI
import langchain

# Suppress all warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()
# Access the OPENAPI_KEY
openapi_key = os.getenv('OPENAPI_KEY')
model_name = "gpt-3.5-turbo"

# Initialize the large language model
llm = ChatOpenAI(
    temperature=0.9,
    openai_api_key=openapi_key,
    model_name=model_name    
)


tools = load_tools(["llm-math","wikipedia"], llm=llm)

agent= initialize_agent(
    tools, 
    llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)
langchain.debug=False
print(agent("What is the 25'%' of 300?"))
print(agent("Who is Andrej Karpathy?"))

