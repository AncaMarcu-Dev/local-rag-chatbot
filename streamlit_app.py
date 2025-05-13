import streamlit as st
from main import RAGChatbot

MODEL_PATH = "./models/<add-model-name>"
PDF_PATH = "./data/<add-pdf-name>"

st.title("🧠 Local RAG Chatbot")

if "chatbot" not in st.session_state:
    st.session_state.chatbot = RAGChatbot(MODEL_PATH)
    docs = st.session_state.chatbot.load_and_split_docs(PDF_PATH)
    st.session_state.chatbot.embed_and_store(docs)

query = st.text_input("Ask a question:")
if query:
    answer = st.session_state.chatbot.ask(query)
    st.write(answer) 
    

