import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# -----------------------------
# 1️⃣ Folder containing scheme PDFs
# -----------------------------
PDF_FOLDER = "./data/schemes"

# -----------------------------
# 2️⃣ Function to split text into chunks
# -----------------------------
def split_text(text, chunk_size=400):

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# -----------------------------
# 3️⃣ Extract text from PDFs
# -----------------------------
documents = []

for file in os.listdir(PDF_FOLDER):

    if file.endswith(".pdf"):

        scheme_name = file.replace(".pdf", "").replace("_", " ")

        pdf_path = os.path.join(PDF_FOLDER, file)

        reader = PdfReader(pdf_path)

        full_text = ""

        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + " "

        chunks = split_text(full_text)

        for chunk in chunks:
            documents.append((scheme_name, chunk))


print(f"Loaded {len(documents)} chunks from scheme PDFs")

# -----------------------------
# 4️⃣ Load embedding model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [doc[1] for doc in documents]

# -----------------------------
# 5️⃣ Generate embeddings
# -----------------------------
embeddings = model.encode(texts, show_progress_bar=True)

# -----------------------------
# 6️⃣ Build FAISS index
# -----------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

# -----------------------------
# 7️⃣ Save FAISS index
# -----------------------------
os.makedirs("faiss_index", exist_ok=True)

faiss.write_index(index, "faiss_index/index.faiss")

# -----------------------------
# 8️⃣ Save mapping
# -----------------------------
with open("faiss_index/doc_mapping.pkl", "wb") as f:
    pickle.dump(documents, f)

print("FAISS index created successfully")
print(f"Total vectors stored: {len(documents)}")