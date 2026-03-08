import os
import fitz  # PyMuPDF

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


PDF_FOLDER = "./data/schemes"
INDEX_PATH = "./faiss_index"


def load_pdfs(folder_path):
    documents = []

    for filename in os.listdir(folder_path):

        if not filename.lower().endswith(".pdf"):
            continue

        filepath = os.path.join(folder_path, filename)

        pdf = fitz.open(filepath)

        for page_num, page in enumerate(pdf):

            text = page.get_text("text")

            if text and text.strip():

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": page_num + 1
                        }
                    )
                )

    return documents


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    return chunks


def build_vector_store(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    vector_db.save_local(INDEX_PATH)

    return vector_db


def main():

    print("Loading PDFs...")
    docs = load_pdfs(PDF_FOLDER)

    print("Total pages extracted:", len(docs))

    print("Splitting into chunks...")
    chunks = split_documents(docs)

    print("Total chunks created:", len(chunks))

    print("Creating FAISS index...")
    build_vector_store(chunks)

    print("FAISS index built successfully.")


if __name__ == "__main__":
    main()