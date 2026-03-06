import os
from PyPDF2 import PdfReader

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

pdf_folder = "./data/schemes"

documents = []

for filename in os.listdir(pdf_folder):

    if filename.endswith(".pdf"):

        filepath = os.path.join(pdf_folder, filename)

        reader = PdfReader(filepath)

        for page in reader.pages:

            text = page.extract_text()

            if text:

                chunk_size = 500
chunk_overlap = 100

for i in range(0, len(text), chunk_size - chunk_overlap):
    chunk = text[i:i + chunk_size]

    documents.append(
        Document(
            page_content=chunk,
            metadata={"source": filename}
        )
    )

print("Total documents loaded:", len(documents))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.from_documents(
    documents,
    embeddings
)

vector_db.save_local("faiss_index")

print("FAISS index built successfully with LangChain.")