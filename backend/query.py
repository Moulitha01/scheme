import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load your FAISS index
index = faiss.read_index("faiss_index/index.faiss")  # adjust path if needed

# Load embeddings model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the mapping of chunk IDs to text
with open("faiss_index/doc_mapping.pkl", "rb") as f:
    doc_mapping = pickle.load(f)

# Query function
def semantic_search(query, k=3):
    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, k)
    results = [doc_mapping[i] for i in indices[0]]
    return results

# Interactive query
while True:
    q = input("Enter your question (or 'exit' to quit): ")
    if q.lower() == "exit":
        break
    answers = semantic_search(q)
    for i, ans in enumerate(answers):
        print(f"\n--- Result {i+1} ---\n{ans}")