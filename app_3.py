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
    page_title="Policy RAG Chatbot",
    page_icon="🤖"
)

st.title("🤖 Policy RAG Chatbot")

# -----------------------------
# Groq API Key
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
You are an HR assistant for Zyro Dynamics. Answer the question using ONLY the
information in the context below.

Write the answer directly and confidently, as a final, complete statement of
company policy. Follow these rules strictly:

- Do NOT mention "the context", "the provided information", "the document",
  or that you are answering "based on" anything. Just state the policy.
- Do NOT add disclaimers, hedges, or notes about missing details (e.g. "this
  is not fully specified", "for more details see..."). Include every relevant
  fact you do have, then stop.
- Be complete: include all relevant numbers, timeframes, steps, roles, and
  conditions found in the context, in clear prose or a short list.
- If, and only if, the context contains NO information relevant to the
  question, reply with exactly:
  "I could not find that information in the company policies."
  Do not use this reply if the context contains even partial relevant
  information — answer with what is available instead.

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

Determine whether the user's question is related to Zyro Dynamics HR policies,
employee handbook, leave policy, code of conduct, compensation, onboarding,
security policies, travel policy, or other company documents.

Question:
{question}

Respond with ONLY:
YES
or
NO
"""
)

REFUSAL_MESSAGE = (
    "I'm sorry, but I can only answer questions related to "
    "Zyro Dynamics HR policies and company documents."
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
        return {"answer": REFUSAL_MESSAGE, "blocked": True}

    answer = rag_chain(question)
    return {"answer": answer, "blocked": False}

# -----------------------------
# UI
# -----------------------------
question = st.text_input(
    "Ask a question about the policy documents:"
)

if question:

    with st.spinner("Thinking..."):

        try:
            result = ask_bot(question)
            st.write(result["answer"])

        except Exception as e:
            st.error(str(e))
