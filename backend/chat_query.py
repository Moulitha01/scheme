import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# 1️⃣ Load FAISS index and mapping
# -----------------------------
index = faiss.read_index("faiss_index/index.faiss")
with open("faiss_index/doc_mapping.pkl", "rb") as f:
    doc_mapping = pickle.load(f)

# Ensure each chunk has a scheme name: ("scheme_name", "chunk_text")
# Adjust this to match your PDFs
if isinstance(doc_mapping[0], str):
    scheme_names = ["PM-Kisan", "PMAY", "Ayushman Bharat"]
    doc_mapping = [(scheme_names[i], doc_mapping[i]) for i in range(len(doc_mapping))]

# -----------------------------
# 2️⃣ Load embeddings model
# -----------------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# 3️⃣ Load LLM for summarization
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")  # stronger than small
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# -----------------------------
# 4️⃣ Semantic search function
# -----------------------------
def semantic_search(query, k=3):
    query_vec = embed_model.encode([query])
    distances, indices = index.search(query_vec, k)
    results = [doc_mapping[i] for i in indices[0]]
    return results

# -----------------------------
# 5️⃣ Summarize a single chunk
# -----------------------------
def summarize_chunk(chunk_text, scheme_name, query):
    input_text = f"Scheme: {scheme_name}\nQuestion: {query}\nText: {chunk_text}\nAnswer concisely:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = llm_model.generate(**inputs, max_new_tokens=150)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return f"{scheme_name}: {summary}"

# -----------------------------
# 6️⃣ Generate final answer
# -----------------------------
def generate_answer(chunks, query):
    # Summarize each chunk individually
    summaries = [summarize_chunk(text, scheme, query) for scheme, text in chunks]
    # Combine summaries into final answer
    combined_text = " ".join(summaries)
    input_text = f"Answer the question based on the following summaries:\nQuestion: {query}\nText: {combined_text}\nAnswer concisely:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = llm_model.generate(**inputs, max_new_tokens=200)
    final_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return final_answer

# -----------------------------
# 7️⃣ Chat loop
# -----------------------------
print("=== Chat with your schemes index ===")
print("Type 'exit' to quit.\n")

