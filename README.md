# 🧠 Local RAG Chatbot

A fully local Retrieval-Augmented Generation (RAG) chatbot using:
- `LLaMA2/Mistral` via `llama-cpp-python`
- Local document embedding with `sentence-transformers`
- Vector search using `FAISS`
- Frontend with `Streamlit`

## 🛠 Setup
```bash
git clone <this-repo>
cd local_rag_chatbot
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 📄 Usage
1. Download a `.gguf` quantized LLaMA2 or Mistral model and place it in the `models/` folder.

   Example from HuggingFace: [TheBloke/Mistral-7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)

2. Run the app:
```bash
streamlit run streamlit_app.py
```

3. Upload one or more PDFs. Their content will be indexed and merged into a single knowledge base.

4. Ask questions, and the chatbot will respond using information from all uploaded PDFs.
