import streamlit as st
import os
import getpass
from dotenv import load_dotenv

# from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Vectara

# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain.chains.qa_with_sources import load_qa_with_sources_chain
# from langchain.prompts import PromptTemplate
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

history = StreamlitChatMessageHistory(key="chat_messages")


# Optionally, specify your own session_state key for storing messages
msgs = StreamlitChatMessageHistory(key="special_app_key")

if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")
# Load the environment variables from .env file
load_dotenv()

# Check if the environment variable is set
if "OPENAI_API_KEY" in os.environ:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    vectara_customer_id = os.getenv("VECTARA_CUSTOMER_ID")
    vectara_corpus_id = os.getenv("VECTARA_CORPUS_ID")
    vectara_api_key = os.getenv("VECTARA_API_KEY")
else:
    # Use st.secrets as fallback
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    vectara_customer_id = st.secrets["VECTARA_CUSTOMER_ID"]
    vectara_corpus_id = st.secrets["VECTARA_CORPUS_ID"]
    vectara_api_key = st.secrets["VECTARA_API_KEY"]

vectorstore = Vectara(
    vectara_customer_id=vectara_customer_id,
    vectara_corpus_id=vectara_corpus_id,
    vectara_api_key=vectara_api_key,
)


st.title("Interactive History Chatbot")
characters = ["Imam-Ahmad-bin-Hanbal", "Ibn Sina", "Ibn Battuta"]
selected_character = st.selectbox("Choose a historical figure:", characters)
user_age = st.number_input("Enter your age:", min_value=5, max_value=100)

# Create a prompt template with the initial context and the user's input
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"You are Historia, a master storyteller passionate about bringing the past to life. You don't just share facts and dates; you focus on the experiences of real people – their challenges, their victories, and the choices they made. Transport the listener to another era and help them understand the world through the eyes of {selected_character}. Consider the user's age ({user_age}); younger listeners will enjoy simpler language and a focus on exciting moments in the character's life. always reply to question based on user language.\n",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | ChatOpenAI()

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,  # Always return the instance created earlier
    input_messages_key="question",
    history_messages_key="history",
)


for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input():
    st.chat_message("human").write(prompt)

    # New messages added to StreamlitChatMessageHistory when the Chain is being call.
    config = {"configurable": {"session_id": "any"}}

    retriever = Vectara().as_retriever()

    # RAG prompt
    template = """Answer the question based only on the following context:
    {context}
    Question: {question},
    You are Historia, a master storyteller with a deep understanding of the past.  You believe history is more than just facts and dates – it's about the lives of real people, their struggles, and triumphs. You can weave narratives that transport the listener to different times and places, emphasizing the human element of history. reply to question based on user language.
    """
    prompt2 = ChatPromptTemplate.from_template(template)

    # RAG
    model = ChatOpenAI()
    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | prompt2
        | model
        | StrOutputParser()
    )

    class Question(BaseModel):
        __root__: str

    chain = chain.with_types(input_type=Question)

    # bugy, needs improvement
    prompt4 = prompt + "\nAdditional info: \n" + chain.invoke(prompt)

    response = chain_with_history.invoke({"question": prompt4}, config)
    st.chat_message("ai").write(response.content)

    # print(response.content)

    # print("\n\n\------------------------------/n/n/")
    # print(prompt)
    # print("\n\n\------------------------------/n/n/")
