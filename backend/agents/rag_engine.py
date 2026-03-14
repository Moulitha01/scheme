import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

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
        search_kwargs={"k": 2}
    )

    # 4. Load local LLM
    llm = OllamaLLM(
        model="tinyllama",
        temperature=0.2,
        num_predict=120
    )

    # 5. Prompt template
    prompt_template = """
You are an AI assistant that explains Indian government schemes.

Use the information below to answer the user's question.

{context}

Question: {question}

Answer in maximum 3 bullet points under 80 words.
Do not repeat the text above.
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # 6. Clean retrieved documents
    def format_docs(docs):

        cleaned = []

        for doc in docs:
            text = doc.page_content

            if "National Informatics Centre" in text:
                continue
            if "Access to information" in text:
                continue
            if "India Portal" in text:
                continue

            cleaned.append(text)

        return "\n\n".join(cleaned)

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
        return {"answer": "Error occurred."}