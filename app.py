import os
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="Zyro Dynamics HR Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Zyro Dynamics HR Assistant")
st.write("Ask questions about company policies and HR documents.")

# -----------------------------
# GROQ API Key
# -----------------------------
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
except Exception:
    st.error(
        "GROQ_API_KEY not found. Add it in Streamlit Secrets."
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
            "No PDF files found in Documents folder."
        )

    loader = PyPDFDirectoryLoader(pdf_folder)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=512
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
# Debug Information
# -----------------------------
with st.expander("System Information"):
    st.write("Documents Loaded:", doc_count)
    st.write("Chunks Created:", chunk_count)

# -----------------------------
# Prompts
# -----------------------------
RAG_PROMPT = ChatPromptTemplate.from_template(
"""
You are an HR assistant for Zyro Dynamics.

Use ONLY the provided context to answer.

If the answer is not present in the context,
reply exactly:

I could not find that information in the company policies.

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

Determine whether the question is related to:

- Employee Handbook
- Leave Policy
- Code of Conduct
- Compensation & Benefits
- Travel Policy
- Security Policy
- Onboarding Policy
- Company Profile

Question:
{question}

Respond with ONLY:

YES

or

NO
"""
)

REFUSAL_MESSAGE = (
    "I'm sorry, but I can only answer questions "
    "related to Zyro Dynamics HR policies and company documents."
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

    prompt = RAG_PROMPT.invoke(
        {
            "context": context,
            "question": question
        }
    )

    response = llm.invoke(prompt)

    return response.content

def ask_bot(question):

    prompt = OOS_PROMPT.invoke(
        {"question": question}
    )

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
# Chat UI
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input(
    "Ask a question about the policy documents:"
)

if st.button("Ask"):

    if question.strip():

        with st.spinner("Searching documents..."):

            try:
                answer = ask_bot(question)

                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": answer
                    }
                )

            except Exception as e:
                st.error(str(e))

# -----------------------------
# Chat History
# -----------------------------
for chat in reversed(st.session_state.chat_history):

    st.markdown(
        f"### 🙋 Question\n{chat['question']}"
    )

    st.markdown(
        f"### 🤖 Answer\n{chat['answer']}"
    )

    st.divider()

