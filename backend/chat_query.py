import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load FAISS index
index = faiss.read_index("faiss_index/index.faiss")

# Load embeddings model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load mapping: FAISS vector id -> text chunk
with open("faiss_index/doc_mapping.pkl", "rb") as f:
    doc_mapping = pickle.load(f)

# Load small LLM for summarization
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# Semantic search
def semantic_search(query, k=3):
    query_vec = embed_model.encode([query])
    distances, indices = index.search(query_vec, k)
    results = [doc_mapping[i] for i in indices[0]]
    return results

# Summarize multiple chunks using LLM
# Improved summarize_text function
def summarize_text(chunks, query):
    # Prepend chunk numbers to distinguish each chunk
    combined_text = " ".join([f"Chunk {i+1}: {c}" for i, c in enumerate(chunks)])
    input_text = f"Answer the question based on the text below:\nQuestion: {query}\nText: {combined_text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
    outputs = llm_model.generate(**inputs, max_new_tokens=200)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# Chat loop
print("=== Chat with your schemes index ===")
print("Type 'exit' to quit.\n")

while True:
    user_query = input("You: ")
    if user_query.lower() == "exit":
        break
    relevant_chunks = semantic_search(user_query)
    answer = summarize_text(relevant_chunks, user_query)
    print(f"\nBot: {answer}\n")