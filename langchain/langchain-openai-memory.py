import os
from dotenv import load_dotenv
import warnings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain_community.callbacks.manager import get_openai_callback
from langchain.chains.conversation.memory import (ConversationBufferMemory, 
                                                  ConversationSummaryMemory, 
                                                  ConversationBufferWindowMemory,
                                                  ConversationKGMemory)
import tiktoken

# Suppress all warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()
# Access the OPENAPI_KEY
openapi_key = os.getenv('OPENAPI_KEY')
model_name = "gpt-3.5-turbo"

# first initialize the large language model
llm = ChatOpenAI(
	temperature=0,
	openai_api_key=openapi_key,
	model_name=model_name
)
conversation_buf = ConversationChain(llm=llm,memory=ConversationBufferMemory())
conversation_sum = ConversationChain(llm=llm,memory=ConversationSummaryMemory(llm=llm))
conversation_bufw = ConversationChain(llm=llm,memory=ConversationBufferWindowMemory(k=1))
tokenizer = tiktoken.encoding_for_model(model_name)

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result

def conversation_buffer_memory():    
    print(conversation_buf.prompt.template)
    print(conversation_buf("Good morning AI!"))
    count_tokens(conversation_buf,"My interest here is to explore the potential of integrating Large Language Models with external knowledge")
    count_tokens(conversation_buf,"I just want to analyze the different possibilities. What can you think of?")
    count_tokens(conversation_buf,"Which data source types could be used to give context to the model?")
    count_tokens(conversation_buf,"What is my aim again?")
    #print(conversation_buf.memory.buffer)

    return ''

def conversation_summary_memory():    
    print(conversation_sum.prompt.template)
    print(conversation_sum("Good morning AI!"))
    count_tokens(conversation_sum,"My interest here is to explore the potential of integrating Large Language Models with external knowledge")
    count_tokens(conversation_sum,"I just want to analyze the different possibilities. What can you think of?")
    count_tokens(conversation_sum,"Which data source types could be used to give context to the model?")
    count_tokens(conversation_sum,"What is my aim again?")
    #print(conversation_sum.memory.buffer)

    return ''


def conversation_bufferwindow_memory():    
    print(conversation_bufw.prompt.template)
    print(conversation_bufw("Good morning AI!"))
    count_tokens(conversation_bufw,"My interest here is to explore the potential of integrating Large Language Models with external knowledge")
    count_tokens(conversation_bufw,"I just want to analyze the different possibilities. What can you think of?")
    count_tokens(conversation_bufw,"Which data source types could be used to give context to the model?")
    count_tokens(conversation_bufw,"What is my aim again?")
    #print(conversation_bufw.memory.buffer)

    return ''
def main():
    while True:
        print("\nSelect an operation:")
        print("1. Conversation Buffer Memory")
        print("2. Conversation Summary Memory")
        print("3. Conversation Buffer Window Memory")
        print("4. Conversation Window Memory")
        print("0. Exit")

        choice = input("Enter choice (1/2/3/4/0): ")

        if choice == '0':
            print("Exiting the program. Goodbye!")
            # show number of tokens for the memory used by each memory type
            print(
                    f'Buffer memory conversation length: {len(tokenizer.encode(conversation_buf.memory.buffer))}\n'
                    f'Summary memory conversation length: {len(tokenizer.encode(conversation_sum.memory.buffer))}\n'
                    f'Buffer Window memory conversation length: {len(tokenizer.encode(conversation_bufw.memory.buffer))}\n'
                )
            break
        
        if choice in ['1', '2', '3','4']:
            
            if choice == '1':
                conversation_buffer_memory()
                continue
            elif choice == '2':
                conversation_summary_memory()
                continue
            elif choice == '3':
                conversation_bufferwindow_memory()
                continue
            elif choice == '4':
                continue
        else:
            print("Invalid choice! Please select a valid option.")

    return ''

if __name__ == "__main__":
    main()