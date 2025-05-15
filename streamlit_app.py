import os
import streamlit as st
from main import RAGChatbot
import sys
import tempfile
import psutil

# Prevent Streamlit from trying to watch torch internals
sys.modules['torch.classes'].__path__ = []

MODEL_PATH = "./models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"

st.title("🧠 Local RAG Chatbot")

if "chatbot" not in st.session_state:
    st.session_state.chatbot = RAGChatbot(MODEL_PATH)

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return mem_bytes / 1024**2  # in MB

st.write(f"Model loaded in {st.session_state.chatbot.load_time:.2f} seconds.")
st.info(f"🧠 Current memory usage: {get_memory_usage():.2f} MB")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = tmp.name

    docs = st.session_state.chatbot.load_and_split_docs(temp_path)
    st.session_state.chatbot.embed_and_store(docs)

    os.remove(temp_path)  # Clean up after use
    st.success("PDF processed and ready for questions!")

query = st.text_input("Ask a question:")
if query:
    answer = st.session_state.chatbot.ask(query)
    st.write(f"Answer loaded in {st.session_state.chatbot.end_prompt_time:.2f} seconds.")
    st.write(answer)
