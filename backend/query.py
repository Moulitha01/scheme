from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 1. Load embeddings and FAISS index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_index = FAISS.load_local("faiss_index", embeddings)

# 2. Query the index
query = input("Enter your question about schemes: ")
results = faiss_index.similarity_search(query, k=3)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---\n")
    print(doc.page_content)