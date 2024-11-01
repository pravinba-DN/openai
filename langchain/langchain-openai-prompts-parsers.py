import openai
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

openapi_key = ""
model_name = "gpt-3.5-turbo"
prompt = ""
template_String = ""

customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""


def initialize():
    # Load environment variables from .env file
    load_dotenv()
    # Access the OPENAPI_KEY
    openapi_key = os.getenv('OPENAPI_KEY')
    model_name = "gpt-3.5-turbo"
    print(openai.__version__,"Program initialized successfully")
    return

def build_response_schema():
    gift_schema = ResponseSchema(name="gift",
                             description="Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")
    delivery_days_schema = ResponseSchema(name="delivery_days",
                                        description="How many days\
                                        did it take for the product\
                                        to arrive? If this \
                                        information is not found,\
                                        output -1.")
    price_value_schema = ResponseSchema(name="price_value",
                                        description="Extract any\
                                        sentences about the value or \
                                        price, and output them as a \
                                        comma separated Python list.")

    response_schema = [gift_schema, 
                        delivery_days_schema,
                        price_value_schema]
    return response_schema

def direct_openai_get_completion(prompt, style, text):
    
    messages = [{"role":"user","content":prompt}]
    #response = openai.chat.completions.create({"model": model_name,"messages":messages,"temperature":0})
    
    response = openai.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature = 0
    )

    return response.choices[0].message.content

def langchain_openai_chatlike(template_string,rqst_style,rqst_prompt):    
        print("Trying to call model",model_name)
        chat = ChatOpenAI(temperature=0.0, model=model_name)        
        prompt_template = ChatPromptTemplate.from_template(template_string)
        rqst_message = prompt_template.format_messages(
                    style=rqst_style,
                    text=rqst_prompt)
        print("Prompt:",prompt_template.messages[0].prompt)
        print("Input vars:",prompt_template.messages[0].input_variables)
        customer_response = chat.invoke(rqst_message)    
        return customer_response.content
    
def langchain_output_parser():
    response_schema = build_response_schema()
    output_parser = StructuredOutputParser.from_response_schemas(response_schema)
    return output_parser

def langchain_format_instructions(output_parser):
    format_instructions = output_parser.get_format_instructions()
    print("Format Instructions >>> ",format_instructions)
    return format_instructions

def langchain_openai_chat_parser(review_template,customer_review,format_instructions,output_parser):
    prompt = ChatPromptTemplate.from_template(template=review_template)
    chat = ChatOpenAI(temperature=0.0, model=model_name)
    messages = prompt.format_messages(text=customer_review, 
                                format_instructions=format_instructions)
    llm_response = chat.invoke(messages)
    output_dict = output_parser.parse(llm_response.content)
    return output_dict

def main():
    initialize()
    while True:
        print("\nSelect an operation:")
        print("1. Direct OpenAI")
        print("2. Langchain(OpenAI) Prompts, Templates")
        print("3. Langchain Parsers")
        print("4. Exit")

        choice = input("Enter choice (1/2/3/4): ")

        if choice == '4':
            print("Exiting the program. Goodbye!")
            break
        
        if choice in ['1', '2', '3']:
            
            if choice == '1':
                try:                
                    text = "These speakers be a treasure! Its sound be clearer than the sea, and it sails through tasks like a swift ship!"
                    style = "Robot slave english in a respectable and calm tone"
                    template_string_direct_openai = f"""Translate the text that is delimited by triple backticks into a style that is {style}. text: ```{text}``` """
                    prompt = template_string_direct_openai
                    print(prompt)
                    print(direct_openai_get_completion(prompt, style, text))

                except openai.OpenAIError as e:
                    # Catch general OpenAI API errors
                    print(f"OpenAI API error: {e}")

                except Exception as e:
                    # Catch other exceptions
                    print(f"An error occurred: {e}")

            elif choice == '2':
                try:    
                    text = "These speakers be a treasure! Its sound be clearer than the sea, and it sails through tasks like a swift ship!"
                    style = "American english in a funny tone"
                    template_string_lc_openai = f"""Translate the text that is delimited by triple backticks into a style that is {style}. text: ```{text}```"""
                    prompt = template_string_lc_openai
                    print(langchain_openai_chatlike(prompt,style,text))
                except openai.OpenAIError as e:
                    # Catch general OpenAI API errors
                    print(f"OpenAI API error: {e}")

                except Exception as e:
                    # Catch other exceptions
                    print(f"An error occurred: {e}")

            elif choice == '3':
                print(f"This is to parse the output of the model and transform it to a specific format, deriving key data in the required format")
                output_parser = langchain_output_parser()
                format_instructions = langchain_format_instructions(output_parser)
                print(langchain_openai_chat_parser(review_template,customer_review,format_instructions,output_parser))
           
        else:
            print("Invalid choice! Please select a valid option.")

if __name__ == "__main__":
    main()
