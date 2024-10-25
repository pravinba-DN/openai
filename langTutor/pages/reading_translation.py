import streamlit as st
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
# Access the OPENAPI_KEY
openapi_key = os.getenv('OPENAPI_KEY')

client = OpenAI(api_key=openapi_key)

def generate_random_sentence():
    # Using OpenAI to generate a random sentence (you can specify the language or context here)
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
      {"role": "system", "content": "You are a language teacher. your job is to generate a short sentence in Hindi"},
      {"role": "user", "content": " Please generate a short sentence in hindi. Do not give english translation as it defeats the purpose of learning"}
      ]
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content.strip()

def verify_translation(original, translation):
    # Using OpenAI to verify the translation and provide feedback
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
      {"role": "system", "content": "You are a language teacher. your job is to check the translation done by the user from hindi to english. You will be given an original sentence and a translation of it done by the user. You have to point out what is wrong in it, or what can be improved. if everything is fine appreciate the user"},
      {"role": "user", "content": f"Original sentence in hindi: {original},user translation: {translation}. based on the original sentance and the user generated translation guide, tell what is wrong and right. "}
      ]
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content.strip()


def app():
    st.header('Reading and Translation')
    st.write('Test your translation skills. Translate the following sentence into English.')

    # State management for sentence generation and user input
    if 'generated_sentence' not in st.session_state:
        st.session_state.generated_sentence = None
    if 'translation_input' not in st.session_state:
        st.session_state.translation_input = ''

    # Generate sentence button
    if st.button('Start'):
        st.session_state.generated_sentence = generate_random_sentence()
    
    if st.session_state.generated_sentence:
        st.subheader('Sentence to Translate:')
        st.write(st.session_state.generated_sentence)

        # User input for translation
        user_translation = st.text_input('Your translation:', key="translation")

        if st.button('Verify Translation'):
            if user_translation:
                st.session_state.translation_input = user_translation
                correction = verify_translation(st.session_state.generated_sentence, user_translation)
                st.subheader('Translation Feedback:')
                st.write(correction)
            else:
                st.error("Please enter a translation before verifying.")
