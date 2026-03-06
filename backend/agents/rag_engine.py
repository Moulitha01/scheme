import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


DB_PATH = "faiss_index"


# -----------------------------
# Load RAG Pipeline
# -----------------------------
def load_rag_pipeline():

    # 1️⃣ Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2️⃣ Load FAISS vector database
    vector_db = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # 3️⃣ Retriever configuration
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    # 4️⃣ Load local LLM from Ollama
    llm = Ollama(
        model="llama3.1",
        temperature=0.2
    )

    # 5️⃣ Prompt template
    prompt = ChatPromptTemplate.from_template(
        """
You are Scheme-Sathi, an AI assistant that helps people understand Indian government schemes.

Use ONLY the information from the context.

If the answer is not present in the context, say:
"I could not find that information in the scheme documents."

Explain clearly and simply.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    # 6️⃣ Format retrieved docs
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 7️⃣ Build RAG chain
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


# -----------------------------
# Load pipeline once
# -----------------------------
rag_chain = load_rag_pipeline()


# -----------------------------
# Query function
# -----------------------------
def ask_scheme_sathi(question):

    try:
        answer = rag_chain.invoke(question)

        return {
            "question": question,
            "answer": answer
        }

    except Exception as e:

        return {
            "question": question,
            "answer": "Sorry, something went wrong while processing the request."
        }