import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


DB_PATH = "faiss_index"


# ----------------------------------
# Load RAG Pipeline
# ----------------------------------
def load_rag_pipeline():

    # 1. Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2. Load FAISS vector database
    vector_db = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # 3. Retriever configuration
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )

    # 4. Load local LLM (Ollama)
    llm = Ollama(
        model="tinyllama",
        temperature=0.2,
        num_predict=120
    )

    # 5. Prompt template
    prompt_template = """
You are a government schemes assistant.

Using the context, recommend relevant schemes.

Rules:
- Suggest schemes only if the eligibility roughly matches the user's profile.
- Do NOT recommend schemes if the eligibility contradicts the user's age.
- Answer in **maximum 3 bullet points**
- Each bullet should contain: Scheme name + short benefit
- Keep the answer under **80 words**
- Do not explain unnecessary details
- If the question asks about a specific scheme, explain it.
- Do not recommend unrelated schemes.
- Keep the answer short.

Question:
{question}

Context:
{context}

Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # 6. Format retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 7. Build RAG chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ----------------------------------
# Load pipeline once
# ----------------------------------
rag_chain = load_rag_pipeline()


# ----------------------------------
# Query function
# ----------------------------------
def ask_scheme_sathi(question):

    try:
        answer = rag_chain.invoke(question)

        return {
            "question": question,
            "answer": answer
        }

    except Exception as e:
        print("RAG ERROR:", e)
        return {"answer": "Error occurred. Check terminal."}