import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


DB_PATH = "faiss_index"


def load_rag_pipeline():

    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load FAISS database
    vector_db = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # Load Ollama model
    llm = Ollama(model="llama3.1")

    prompt = ChatPromptTemplate.from_template(
        """
You are Scheme-Sathi, an AI assistant that helps rural citizens understand government schemes.

Use the context below to answer the question clearly.

Context:
{context}

Question:
{question}

Answer in simple language.
"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


rag_chain = load_rag_pipeline()


def ask_scheme_sathi(question):

    answer = rag_chain.invoke(question)

    return {"answer": answer}