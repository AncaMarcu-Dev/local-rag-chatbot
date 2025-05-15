from llama_cpp import Llama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import time
import os
os.environ["TORCH_USE_RTLD_GLOBAL"] = "YES"
os.environ["STREAMLIT_WATCH_DIRECTORIES"] = "false"


class RAGChatbot:
    def __init__(self, model_path):
        start_time= time.time()
        self.llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4,
        chat_format="chatml",  
        verbose=False
        )
        self.load_time = time.time() - start_time
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.texts = []

    def load_and_split_docs(self, path):
        loader = PyPDFLoader(path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        return splitter.split_documents(docs)

    def embed_and_store(self, docs):
        texts = [d.page_content for d in docs]
        embeddings = self.embedder.encode(texts)
        index = faiss.IndexFlatL2(embeddings[0].shape[0])
        index.add(embeddings)
        self.index = index
        self.texts = texts

    def ask(self, question):
        if self.index is None or not hasattr(self.index, 'ntotal') or self.index.ntotal == 0:
            return "Please upload a file first."
        self.start_prompt_time = time.time()
        q_embed = self.embedder.encode([question])
        D, I = self.index.search(q_embed, k=3)
        context = "\n".join([self.texts[i] for i in I[0]])
        prompt = f"Answer the question based on the context:\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        response = self.llm(prompt, max_tokens=200)
        self.end_prompt_time= time.time() - self.start_prompt_time
        return response["choices"][0]["text"] 