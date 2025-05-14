import streamlit as st
from main import RAGChatbot
import sys

# Prevent Streamlit from trying to watch torch internals
sys.modules['torch.classes'].__path__ = []

MODEL_PATH = "./models/llama-2-7b-chat.Q4_K_M.gguf"

st.title("🧠 Local RAG Chatbot")

if "chatbot" not in st.session_state:
    st.session_state.chatbot = RAGChatbot(MODEL_PATH)

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    docs = st.session_state.chatbot.load_and_split_docs("temp.pdf")
    st.session_state.chatbot.embed_and_store(docs)
    st.success("PDF processed and ready for questions!")

query = st.text_input("Ask a question:")
if query:
    answer = st.session_state.chatbot.ask(query)
    st.write(answer)