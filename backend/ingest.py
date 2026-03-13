from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Official scheme pages
urls = [
    "https://www.india.gov.in/spotlight/pradhan-mantri-awas-yojana",
    "https://www.india.gov.in/spotlight/ayushman-bharat-pradhan-mantri-jan-arogya-yojana",
    "https://www.india.gov.in/spotlight/pradhan-mantri-vaya-vandana-yojana"
]

print("Loading scheme webpages...")

loader = WebBaseLoader(urls)
documents = loader.load()

print("Splitting documents...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

docs = splitter.split_documents(documents)

print("Creating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(docs, embeddings)

db.save_local("faiss_index")

print("Ingestion complete")