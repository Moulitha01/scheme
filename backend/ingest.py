# ingest.py
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
DATA_PATH = "data/schemes"
INDEX_PATH = "faiss_index"

def ingest():

    if not os.path.exists(DATA_PATH):
        print("❌ data/schemes folder not found.")
        return

    documents = []

    print("📂 Loading PDFs...")

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            file_path = os.path.join(DATA_PATH, file)
            print(f"   → Reading {file}")
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

    if not documents:
        print("❌ No PDF documents found.")
        return

    print("✂️ Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    docs = text_splitter.split_documents(documents)

    print("🧠 Loading embedding model (first time takes ~1 minute)...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("📦 Creating FAISS vector store...")

    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local(INDEX_PATH)

    print("✅ Index built successfully!")
    print(f"📁 Saved at: {INDEX_PATH}")

if __name__ == "__main__":
    ingest()