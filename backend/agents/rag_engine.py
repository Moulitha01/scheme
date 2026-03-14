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
        search_kwargs={"k": 2}
    )

    # 4. Load local LLM (Ollama)
    llm = Ollama(
        model="tinyllama",
        temperature=0.2,
        num_predict=120
    )

    # 5. Prompt template
    prompt_template = """
    You are an AI assistant that helps citizens understand Indian government schemes.

    Use ONLY the provided context to answer.

    Rules:
    - Maximum 3 bullet points
    - Each bullet: Scheme name + benefit
    - Maximum 80 words total
    - Do NOT write paragraphs
    - Do NOT include unrelated schemes
    - If the user asks about a specific scheme, explain only that scheme

    Response format:

    • Bullet 1  
    • Bullet 2  
    • Bullet 3  

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