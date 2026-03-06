from scheme.backend.agents.rag_engine import ask_scheme_sathi

while True:
    query = input("Ask about schemes: ")

    response = ask_scheme_sathi(query)

    print("\nAnswer:\n")
    print(response["answer"])
    print("\n")