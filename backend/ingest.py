import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# Folder containing PDFs
pdf_folder = "./data/schemes"  # adjust if needed

# 1️⃣ Load PDFs and split into text chunks
chunks = []
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        reader = PdfReader(os.path.join(pdf_folder, filename))
        for page in reader.pages:
            text = page.extract_text()
            if text:
                chunks.append(text)

print(f"Loaded {len(chunks)} text chunks from PDFs.")

# 2️⃣ Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3️⃣ Create embeddings for all chunks
embeddings = model.encode(chunks, show_progress_bar=True)

# 4️⃣ Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 5️⃣ Save FAISS index
if not os.path.exists("faiss_index"):
    os.mkdir("faiss_index")
faiss.write_index(index, "faiss_index/index.faiss")

# 6️⃣ Save mapping: FAISS vector id -> text chunk
# Using simple list where id = index in list
with open("faiss_index/doc_mapping.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("FAISS index and doc mapping saved successfully!")