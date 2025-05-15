# 🧠 Local RAG Chatbot

A lightweight, fully **local** Retrieval-Augmented Generation (RAG) chatbot that lets you query your own PDFs using natural language.

Powered by:
- 🔍 `tinyllama` via `llama-cpp-python` for local language generation
- 📚 `sentence-transformers` for local document embeddings
- 📦 `FAISS` for fast vector similarity search
- 🌐 `Streamlit` for a simple web interface

All processing is done locally — your documents stay private.

---

## 🚀 Features
- Upload a PDF and ask questions about its contents
- Runs entirely offline — no internet or external API required
- Small and fast model (e.g. TinyLlama) 

---

## 🛠️ Setup

```bash
git clone <this-repo>
cd local_rag_chatbot
python -m venv venv
# Activate the virtual environment:
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
pip install -r requirements.txt


## 📄 Usage
1. Download a `.gguf` model and place it in the `models/` folder.
   Example from HuggingFace: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/blob/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf

2. Run the app:
```bash
streamlit run streamlit_app.py
```

📌 Notes
You can only upload one PDF at a time in the current version

Supports small models for testing, but larger GGUF models also work with enough memory

Tested with: tinyllama-1.1b-chat-v1.0.Q2_K.gguf on Windows
