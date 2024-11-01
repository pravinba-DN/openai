import os
from dotenv import load_dotenv
import warnings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.prompts import PromptTemplate

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

def pipeline_chain():
    print("Executing Pipeline Chain...")
    prompt = ChatPromptTemplate.from_template("What is the best way to describe a company that makes {product}")
    chain = prompt | llm
    product = input("Enter a product for Pipeline Chain: ")
    response = chain.invoke(product)
    print("Response:", response.content)

def llm_chain():
    print("Executing LLM Chain...")
    product = input("Enter a product for LLM Chain: ")
    country = input("Enter a country for LLM Chain: ")
    
    params = {
        "product": product,
        "country": country
    }
    prompt = ChatPromptTemplate.from_template("Who are the competitors for selling {product} in {country}")
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    response = llm_chain.run(params)
    print("Response:", response)

def sequential_chain():
    print("Executing Sequential Chain...")
    # prompt template 1: translate to english
    first_prompt = ChatPromptTemplate.from_template("Translate the following review to english:\n\n{Review}")
    print("<<<< Chain 1  execution >>>>")
    # chain 1: input= Review and output= English_Review
    chain_one = LLMChain(llm=llm, prompt=first_prompt, 
                     output_key="English_Review"
    )
    # prompt template 2: Summarize review
    second_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following review in 1 sentence:"
    "\n\n{English_Review}"
    )

    print("<<<< Chain 2  execution >>>>")
    # chain 2: input= English_Review and output= summary
    chain_two = LLMChain(llm=llm, prompt=second_prompt, 
                     output_key="summary"
    )
    
    print("<<<< Chain 3  execution >>>>")
    # prompt template 3: Get the original language name
    third_prompt = ChatPromptTemplate.from_template(
        "What language is the following review:\n\n{Review}"
    )
    # chain 3: input= Review and output= language
    chain_three = LLMChain(llm=llm, prompt=third_prompt,
                        output_key="language"
    )
    print("<<<< Chain 4  execution >>>>")
    # prompt template 4: follow up message
    fourth_prompt = ChatPromptTemplate.from_template(
    "Write a follow up response to the following "
    "summary in the specified language:"
    "\n\nSummary: {summary}\n\nLanguage: {language}"
    )
    # chain 4: input= summary, language and output= followup_message
    chain_four = LLMChain(llm=llm, prompt=fourth_prompt,
                      output_key="followup_message"
                     )

    

    # overall_chain: input= Review and output= English_Review,summary, followup_message
    overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],
    output_variables=["English_Review", "summary","language","followup_message"],
    verbose=True
    )

    # Pass a long review in French
    review = """J'ai récemment acheté un purificateur d'air et je suis absolument ravi des résultats. Cet appareil fonctionne silencieusement tout en éliminant efficacement les allergènes et la poussière de l'air. La qualité de l'air dans ma maison s'est nettement améliorée, et je ressens moins de symptômes d'allergies. De plus, le design élégant s'intègre parfaitement à ma décoration intérieure. Les filtres sont faciles à remplacer, ce qui rend l'entretien simple. Je recommande vivement ce purificateur à tous ceux qui cherchent à améliorer leur environnement intérieur. C'est un investissement qui en vaut vraiment la peine !"""
    
    # Execute the overall chain with the review
    response = overall_chain(review)
    print("Response:")
    print("English Review:", response["English_Review"])
    print("Summary:", response["summary"])
    print("Detected Language:", response["language"])
    print("Follow-up Message:", response["followup_message"])


def router_chain():
    
    # Step 1 - Define the templates to be used in the router

    physics_template = """You are a very smart physics professor. \
    You are great at answering questions about physics in a concise\
    and easy to understand manner. \
    When you don't know the answer to a question you admit\
    that you don't know.

    Here is a question:
    {input}"""


    math_template = """You are a very good mathematician. \
    You are great at answering math questions. \
    You are so good because you are able to break down \
    hard problems into their component parts, 
    answer the component parts, and then put them together\
    to answer the broader question.

    Here is a question:
    {input}"""

    history_template = """You are a very good historian. \
    You have an excellent knowledge of and understanding of people,\
    events and contexts from a range of historical periods. \
    You have the ability to think, reflect, debate, discuss and \
    evaluate the past. You have a respect for historical evidence\
    and the ability to make use of it to support your explanations \
    and judgements.

    Here is a question:
    {input}"""


    computerscience_template = """ You are a successful computer scientist.\
    You have a passion for creativity, collaboration,\
    forward-thinking, confidence, strong problem-solving capabilities,\
    understanding of theories and algorithms, and excellent communication \
    skills. You are great at answering coding questions. \
    You are so good because you know how to solve a problem by \
    describing the solution in imperative steps \
    that a machine can easily interpret and you know how to \
    choose a solution that has a good balance between \
    time complexity and space complexity. 

    Here is a question:
    {input}"""

    # Step 2 -  Define the prompt info(name,description,prompt_template)

    prompt_infos = [
        {
            "name": "physics", 
            "description": "Good for answering questions about physics", 
            "prompt_template": physics_template
        },
        {
            "name": "math", 
            "description": "Good for answering math questions", 
            "prompt_template": math_template
        },
        {
            "name": "History", 
            "description": "Good for answering history questions", 
            "prompt_template": history_template
        },
        {
            "name": "computer science", 
            "description": "Good for answering computer science questions", 
            "prompt_template": computerscience_template
        }
    ]

    # Step 3 - Assign the prompt templates & prompt info to destination chains
    destination_chains = {}
    for p_info in prompt_infos:
        name = p_info["name"]
        prompt_template = p_info["prompt_template"]
        prompt = ChatPromptTemplate.from_template(template=prompt_template)
        chain = LLMChain(llm=llm, prompt=prompt)
        destination_chains[name] = chain  
    
    # Step 4 - List down destination strings with "\n"
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)

    # Step 5 - Define default chain when none of the destinations match
    default_prompt = ChatPromptTemplate.from_template("{input}")
    default_chain = LLMChain(llm=llm, prompt=default_prompt)

    # Step 6 - Define router template
    MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
    language model select the model prompt best suited for the input. \
    You will be given the names of the available prompts and a \
    description of what the prompt is best suited for. \
    You may also revise the original input if you think that revising\
    it will ultimately lead to a better response from the language model.

    << FORMATTING >>
    Return a markdown code snippet with a JSON object formatted to look like:
    ```json
    {{{{
        "destination": string \ name of the prompt to use or "DEFAULT"
        "next_inputs": string \ a potentially modified version of the original input
    }}}}
    ```

    REMEMBER: "destination" MUST be one of the candidate prompt \
    names specified below OR it can be "DEFAULT" if the input is not\
    well suited for any of the candidate prompts.
    REMEMBER: "next_inputs" can just be the original input \
    if you don't think any modifications are needed.

    << CANDIDATE PROMPTS >>
    {destinations}

    << INPUT >>
    {{input}}

    << OUTPUT (remember to include the ```json)>>"""

    # Step 7 - Define router template with destination strings
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
    )
    router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
    )

    router_chain = LLMRouterChain.from_llm(llm, router_prompt)

    # Step 8 - Put together the Multi Chain 
    chain = MultiPromptChain(router_chain=router_chain, 
                         destination_chains=destination_chains, 
                         default_chain=default_chain, verbose=True
                        ) 

    response = chain.run("What is a black hole")
    print(response)
    response = chain.run("Square root of 9")
    print(response)
    response = chain.run("Who is Vince Mac Mahon")
    print(response)
     


def main():
    while True:
        print("Choose the type of chain to execute:")
        print("1. Pipeline Chain")
        print("2. LLM Chain")
        print("3. Sequential Chain")
        print("4. Router Chain")        
        print("0 - Exit")
        
        choice = input("Enter your choice (1-4)/0: ")
        
        if choice == '0':
            print("Exiting the program. Goodbye!")
            break
        
        if choice == '1':
            pipeline_chain()
        elif choice == '2':
            llm_chain()
        elif choice == '3':
            sequential_chain()
        elif choice == '4':
            router_chain()
            
        else:
            print("Invalid choice. Please try again.")

    

if __name__ == "__main__":
    main()
