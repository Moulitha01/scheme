import faiss
import pickle
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# Scheme metadata (for recommendation)
# -----------------------------
schemes = {
    "PM-Kisan": {
        "target": ["farmer"],
        "benefit": "₹6000 per year income support"
    },
    "PMAY": {
        "target": ["low income", "housing"],
        "benefit": "subsidy for house construction"
    },
    "Ayushman Bharat": {
        "target": ["poor", "low income"],
        "benefit": "₹5 lakh health insurance coverage"
    },
    "Skill India": {
        "target": ["student", "youth"],
        "benefit": "free skill development training"
    },
    "National Scholarship Portal": {
        "target": ["student"],
        "benefit": "government scholarships for education"
    }
}

# -----------------------------
# Load FAISS index
# -----------------------------
index = faiss.read_index("faiss_index/index.faiss")

with open("faiss_index/doc_mapping.pkl", "rb") as f:
    doc_mapping = pickle.load(f)

# -----------------------------
# Load embedding model
# -----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Load LLM
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# -----------------------------
# Detect user profile
# -----------------------------
def detect_profile(text):

    profile = {
        "age": None,
        "gender": None,
        "occupation": None
    }

    age_match = re.search(r"\b(\d{1,2})\b", text)
    if age_match:
        profile["age"] = int(age_match.group(1))

    if "girl" in text or "female" in text or "woman" in text:
        profile["gender"] = "female"

    if "student" in text or "college" in text:
        profile["occupation"] = "student"

    if "farmer" in text:
        profile["occupation"] = "farmer"

    return profile


# -----------------------------
# Recommend schemes
# -----------------------------
def recommend_schemes(profile):

    recommendations = []

    for scheme, info in schemes.items():

        if profile["occupation"]:

            for target in info["target"]:
                if profile["occupation"] in target or target in profile["occupation"]:
                    recommendations.append(f"{scheme} - {info['benefit']}")

    return recommendations


# -----------------------------
# Semantic search
# -----------------------------
def semantic_search(query, k=3):

    query_vec = embed_model.encode([query])
    distances, indices = index.search(query_vec, k)

    results = []

    for i in indices[0]:
        results.append(doc_mapping[i])

    return results


# -----------------------------
# Generate grounded answer
# -----------------------------
def generate_answer(context_chunks, question):

    context = "\n\n".join(
    f"Scheme: {scheme}\n{text}" for scheme, text in context_chunks
    )

    prompt = f"""
You are an assistant that explains Indian government schemes.

Use ONLY the information from the context below.
If the answer is not present, say "Information not found.

If multiple schemes appear, return only the scheme that best answers the question.

Answer in this format:

Scheme: <scheme name>

What it provides:
<short explanation>

Eligibility:
<who can apply>

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=120,
        num_beams=4,
        do_sample=False
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer


# -----------------------------
# Chat loop
# -----------------------------
print("\n=== Scheme Assistant ===")
print("Ask about government schemes.")
print("Type 'exit' to quit.\n")

while True:

    user_query = input("You: ")

    if user_query.lower() == "exit":
        break

    # Detect user profile
    profile = detect_profile(user_query)

    # Recommendation mode
    if profile["occupation"] or profile["age"]:

        recs = recommend_schemes(profile)

        if recs:
            print("\nBot:\n")

            for r in recs:
                print("•", r)

            print()
            continue

    # Normal RAG QA
    chunks = semantic_search(user_query, k=5)

    answer = generate_answer(chunks, user_query)

    print("\nBot:", answer, "\n")