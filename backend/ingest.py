from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from bs4 import BeautifulSoup

urls = [
    "https://www.india.gov.in/spotlight/pradhan-mantri-awas-yojana",
    "https://www.india.gov.in/spotlight/ayushman-bharat-pradhan-mantri-jan-arogya-yojana",
    "https://www.india.gov.in/spotlight/pradhan-mantri-vaya-vandana-yojana"
]

print("Loading scheme webpages...")

loader = WebBaseLoader(urls)
documents = loader.load()

cleaned_documents = []

print("Cleaning webpage content...")

for doc in documents:
    soup = BeautifulSoup(doc.page_content, "html.parser")

    # Remove scripts/styles
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)

    # Keep only meaningful content
    if len(text) > 200:
        doc.page_content = text
        cleaned_documents.append(doc)

print("Documents after cleaning:", len(cleaned_documents))

print("Splitting documents...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

docs = splitter.split_documents(cleaned_documents)

print("Total chunks:", len(docs))

if len(docs) == 0:
    print("ERROR: No text extracted from webpages.")
    exit()

print("Creating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(docs, embeddings)

db.save_local("faiss_index")

print("Ingestion complete")