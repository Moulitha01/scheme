import sys
import os
import re

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.eligibility_engine import check_eligibility
from agents.rag_engine import ask_scheme_sathi


# -----------------------------
# Scheme metadata (recommendation)
# -----------------------------
scheme_metadata = {
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

    # Age based recommendations
    if profile["age"] and profile["age"] <= 25:
        recommendations.append("Skill India - free skill development training")
        recommendations.append("National Scholarship Portal - government education scholarships")

    # Gender based
    if profile["gender"] == "female" and profile["age"] and profile["age"] <= 25:
        recommendations.append("Beti Bachao Beti Padhao - support for education of girls")

    # Occupation based
    if profile["occupation"] == "student":
        recommendations.append("National Scholarship Portal - scholarships for students")

    if profile["occupation"] == "farmer":
        recommendations.append("PM-Kisan - ₹6000 per year income support")

    if profile["age"] and profile["age"] >= 60:
        recommendations.append("Indira Gandhi National Old Age Pension Scheme - pension support for senior citizens")
        recommendations.append("Pradhan Mantri Vaya Vandana Yojana - pension scheme for elderly citizens")

    return recommendations
# -----------------------------
# Chat Loop
# -----------------------------
print("\n=== Scheme Assistant ===")
print("Ask about government schemes.")
print("Type 'exit' to quit.\n")

while True:

    user_query = input("You: ")

    if user_query.lower() == "exit":
        break

    # Detect profile
    profile = detect_profile(user_query)

    if profile["age"] or profile["occupation"] or profile["gender"]:

        recs = recommend_schemes(profile)

        if recs:
            print("\nBot:\n")

            for r in recs:
                print("•", r)

            print()
            continue


    # Rule based eligibility
    matched_schemes = check_eligibility(user_query)

    if matched_schemes:

        print("\nBot:\n")

        for scheme in matched_schemes:
            print(f"• {scheme['name']} - {scheme['description']}")

        print()
        continue


    # RAG answer
    result = ask_scheme_sathi(user_query)

    print("\nBot:", result["answer"], "\n")

    # Step 3: Detect profile
    profile = detect_profile(user_query)

    # Step 4: Recommend schemes
    if profile["occupation"] or profile["age"]:

        recs = recommend_schemes(profile)

        if recs:

            print("\nRecommended Schemes:\n")

            for r in recs:
                print("•", r)

            print()