import streamlit as st
import os
import getpass
from dotenv import load_dotenv
from langchain_community.vectorstores import Vectara
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load the environment variables from .env file
load_dotenv()

# os.environ["VECTARA_CUSTOMER_ID"] = getpass.getpass("Vectara Customer ID:")
# os.environ["VECTARA_CORPUS_ID"] = getpass.getpass("Vectara Corpus ID:")
# os.environ["VECTARA_API_KEY"] = getpass.getpass("Vectara API Key:")
openai_api_key = os.getenv("OPENAI_API_KEY")

vectorstore = Vectara(
    vectara_customer_id=os.getenv("VECTARA_CUSTOMER_ID"),
    vectara_corpus_id=os.getenv("VECTARA_CORPUS_ID"),
    vectara_api_key=os.getenv("VECTARA_API_KEY"),
)

st.title("Interactive History Chatbot")

characters = ["Imam-Ahmad-bin-Hanbal", "Ibn Sina", "Ibn Battuta"]
selected_character = st.selectbox("Choose a historical figure:", characters)

user_age = st.number_input("Enter your age:", min_value=5, max_value=100)
