from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
import streamlit as st
import torch
import os
from fpdf import FPDF  # <-- Added import

# --- Streamlit Config ---
st.set_page_config(page_title="AI PDF Chatbot", layout="centered")

st.markdown(
    """
    <style>
        body {
            background-color: #fff0f0;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.1);
        }
        .stTextInput > div > div > input {
            padding: 0.75rem;
            font-size: 16px;
            border-radius: 10px;
            border: 1px solid #ff9999;
        }
        .stButton > button {
            font-size: 16px;
            padding: 0.6rem 1.2rem;
            border-radius: 10px;
            margin: 0.3rem 0;
            background-color: #ff6666 !important;
            color: white !important;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #ff4d4d !important;
        }
        .answer-box {
            background-color: #fff5f5;
            border-left: 6px solid #ff4d4d;
            padding: 1rem;
            border-radius: 10px;
            margin-top: 1rem;
            font-size: 16px;
        }
        .intro-text {
            font-size: 18px;
            color: #333;
            padding: 0.75rem;
            margin-bottom: 1.5rem;
            border-radius: 10px;
            background-color: #ffdada;
            border: 1px solid #ff9999;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Function to Save Chunks to PDF ---
def save_chunks_to_pdf(chunks, filename="pdf_chunks_output.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for i, doc in enumerate(chunks):
        text = f"Chunk {i+1}:\n{doc.page_content}\n\n"
        pdf.multi_cell(0, 10, text)
    pdf.output(filename)
    return filename

# --- Cached Model Loader ---
@st.cache_resource
def load_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        tokenizer="google/flan-t5-large",
        max_length=512,
        device=device
    )

# --- Cached Embedding Generator ---
@st.cache_data
def embed_pdf(pdf_bytes):
    with open("temp.pdf", "wb") as f:
        f.write(pdf_bytes)
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore, splits

# --- UI Header ---
st.markdown('<div class="intro-text">👋 Hello! How can I help you today?</div>', unsafe_allow_html=True)
st.title("📄💬 Smart PDF Q&A Assistant")
st.markdown("Upload a PDF and ask natural-language questions. This chatbot uses Hugging Face models with Retrieval-Augmented Generation (RAG) to give accurate, context-based answers.")

# --- Upload and Buttons ---
pdf_file = st.file_uploader("📎 **Upload your PDF**", type="pdf")

col1, col2, col3 = st.columns(3)
with col1:
    example_q = st.button("💡Key findings*\n")
with col2:
    clear = st.button("🧹 Clear Chat")
with col3:
    about = st.button("ℹ️ About Bot")

if example_q:
    st.info(
        "Try asking:\n"
        "- *What is the main topic of this document?*\n"
        "- *Summarize the key findings in this PDF.*\n"
        "- *List the main sections of this document.*"
    )

if clear:
    st.rerun()

if about:
    st.info("This bot uses Hugging Face's FLAN-T5 model and BGE embeddings with RAG (Retrieval-Augmented Generation) to answer questions based on your uploaded PDF content.")

# --- Main Logic ---
if pdf_file is not None:
    with st.spinner("📚 Processing your PDF..."):
        vectorstore, splits = embed_pdf(pdf_file.read())
        llm = HuggingFacePipeline(pipeline=load_model())

    # --- Show and Download Chunks ---
    with st.expander("Show embedded chunks"):
        for i, doc in enumerate(splits):
            st.write(f"**Chunk {i+1}:** {doc.page_content[:300]}{'...' if len(doc.page_content) > 300 else ''}")

    if st.button("📥 Download Chunks as PDF"):
        output_pdf = save_chunks_to_pdf(splits)
        with open(output_pdf, "rb") as f:
            st.download_button(
                label="Download Chunks PDF",
                data=f,
                file_name="pdf_chunks_output.pdf",
                mime="application/pdf"
            )

    user_question = st.text_input("**❓ Ask something about your PDF:**")

    if st.button("🔍 Execute"):
        if user_question.strip() == "":
            st.warning("Please enter a valid question.")
        else:
            retrieved_docs = vectorstore.similarity_search(user_question, k=3)
            context = "\n".join([doc.page_content for doc in retrieved_docs])

            if not context.strip():
                st.warning("No relevant information found in the document.")
            else:
                prompt = (
                    "Answer the question ONLY using the context below. "
                    "If the answer is not in the context, say 'I don't know.'\n\n"
                    f"Answer the question based only on the context below.\n\n"
                    f"Context: {context}\n\n"
                    f"Question: {user_question}\nAnswer:"
                )
                answer = llm.invoke(prompt)

                if (
                    "i don't know" in answer.lower()
                    or answer.strip() == ""
                    or answer.strip().lower().startswith("powered by")
                ):
                    st.warning("Sorry, the answer is not found in your PDF.")
                else:
                    st.markdown(
                        f'<div class="answer-box"><strong>📘 Answer:</strong><br>{answer}</div>',
                        unsafe_allow_html=True
                    )

    # Clean up
    if os.path.exists("temp.pdf"):
        os.remove("temp.pdf")

else:
    st.info("⬆️ Please upload a PDF to get started.")