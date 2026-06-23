import os
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="Policy RAG Chatbot",
    page_icon="🤖"
)

st.title("🤖 Policy RAG Chatbot")

# -----------------------------
# Gemini API Key
# -----------------------------
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
except Exception:
    st.error(
        "GOOGLE_API_KEY not found. Add it in Streamlit Secrets."
    )
    st.stop()

# -----------------------------
# Initialize RAG
# -----------------------------
@st.cache_resource
def initialize_rag():

    pdf_folder = "Documents"

    if not os.path.exists(pdf_folder):
        raise Exception(
            f"Folder '{pdf_folder}' not found."
        )

    pdf_files = [
        f for f in os.listdir(pdf_folder)
        if f.lower().endswith(".pdf")
    ]

    if len(pdf_files) == 0:
        raise Exception(
            "No PDF files found in documents folder."
        )

    loader = PyPDFDirectoryLoader(pdf_folder)
    documents = loader.load()

    if len(documents) == 0:
        raise Exception(
            "No documents were loaded."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    if len(chunks) == 0:
        raise Exception(
            "No chunks were created."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Test embedding generation
    test_embedding = embeddings.embed_query(
        "hello world"
    )

    if len(test_embedding) == 0:
        raise Exception(
            "Embedding model failed."
        )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1
    )

    return retriever, llm, len(documents), len(chunks)

# -----------------------------
# Load Resources
# -----------------------------
try:
    retriever, llm, doc_count, chunk_count = initialize_rag()

except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# -----------------------------
# Debug Info
# -----------------------------
with st.expander("System Info"):
    st.write("Documents Loaded:", doc_count)
    st.write("Chunks Created:", chunk_count)

# -----------------------------
# Prompts
# -----------------------------
RAG_PROMPT = ChatPromptTemplate.from_template(
"""
You are a helpful assistant.

Answer ONLY using the provided context.

Context:
{context}

Question:
{question}

Answer:
"""
)

OOS_PROMPT = ChatPromptTemplate.from_template(
"""
You are a classifier.

Determine whether the question is related
to the uploaded policy documents.

Question:
{question}

Respond with only:

YES
or
NO
"""
)

REFUSAL_MESSAGE = (
    "I'm sorry, but I can only answer questions "
    "related to the policy documents."
)

# -----------------------------
# Helpers
# -----------------------------
def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )

def rag_chain(question):

    docs = retriever.invoke(question)

    context = format_docs(docs)

    prompt = RAG_PROMPT.invoke({
        "context": context,
        "question": question
    })

    response = llm.invoke(prompt)

    return response.content

def ask_bot(question):

    prompt = OOS_PROMPT.invoke({
        "question": question
    })

    decision = (
        llm.invoke(prompt)
        .content
        .strip()
        .upper()
    )

    if "NO" in decision:
        return REFUSAL_MESSAGE

    return rag_chain(question)

# -----------------------------
# UI
# -----------------------------
question = st.text_input(
    "Ask a question about the policy documents:"
)

if question:

    with st.spinner("Thinking..."):

        try:
            answer = ask_bot(question)
            st.write(answer)

        except Exception as e:
            st.error(str(e))