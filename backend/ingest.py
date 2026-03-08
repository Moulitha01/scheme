import os
import fitz

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DATA_FOLDER = "./data/schemes"
INDEX_PATH = "./faiss_index"

documents = []

for filename in os.listdir(DATA_FOLDER):

    filepath = os.path.join(DATA_FOLDER, filename)

    # ---- HANDLE PDF FILES ----
    if filename.lower().endswith(".pdf"):

        pdf = fitz.open(filepath)

        for page_num, page in enumerate(pdf):

            text = page.get_text()

            if text.strip():

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": page_num + 1
                        }
                    )
                )

    # ---- HANDLE TXT FILES ----
    elif filename.lower().endswith(".txt"):

        with open(filepath, "r", encoding="utf-8") as f:

            text = f.read()

            if text.strip():

                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": filename}
                    )
                )

print("Documents loaded:", len(documents))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=80
)

chunks = splitter.split_documents(documents)

print("Total chunks created:", len(chunks))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.from_documents(chunks, embeddings)

vector_db.save_local(INDEX_PATH)

print("FAISS index built successfully.")