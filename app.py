import os
import tempfile
from typing import List

import streamlit as st
from dotenv import load_dotenv

from src.chatbot import answer_question, create_memory
from src.document_loader import load_and_chunk_pdf
from src.embeddings import get_embeddings
from src.vector_store import create_faiss_store, get_retriever

load_dotenv()

# Set page configs
st.set_page_config(page_title="SmartQuery", page_icon="🤖", layout="wide")

# Inject premium CSS design styles
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Main Title and Subtitle */
    .app-header {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.1rem;
        letter-spacing: -1.5px;
    }
    
    .app-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Sidebar Branding */
    .sidebar-logo {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a855f7 0%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        margin-bottom: -10px;
    }

    /* Custom Glassmorphism Panels */
    div.stExpander {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        background-color: rgba(255, 255, 255, 0.02) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
        transition: all 0.3s ease !important;
    }
    
    div.stExpander:hover {
        border-color: rgba(168, 85, 247, 0.4) !important;
        box-shadow: 0 8px 30px rgba(168, 85, 247, 0.1) !important;
    }

    /* Primary and Secondary Action Buttons */
    button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4) !important;
    }
    
    button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6) !important;
    }

    button[kind="secondary"] {
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        transition: all 0.2s ease !important;
    }

    button[kind="secondary"]:hover {
        border-color: rgba(168, 85, 247, 0.5) !important;
        color: #a855f7 !important;
        background-color: rgba(168, 85, 247, 0.05) !important;
    }

    /* Notification Alerts Styling */
    div[data-testid="stNotification"] {
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state() -> None:
    if "memory" not in st.session_state:
        st.session_state.memory = create_memory()
    if "latest_sources" not in st.session_state:
        st.session_state.latest_sources = []


def reset_document_state() -> None:
    for key in ("chunks", "vector_store", "retriever", "document_names"):
        st.session_state.pop(key, None)
    st.session_state.memory = create_memory()
    st.session_state.latest_sources = []


def process_pdfs(uploaded_files, api_key: str = None) -> None:
    all_chunks = []
    processed_names = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name

        try:
            chunks = load_and_chunk_pdf(temp_path, original_filename=uploaded_file.name)
            all_chunks.extend(chunks)
            processed_names.append(uploaded_file.name)
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    if not all_chunks:
        raise ValueError("No text content could be extracted from any of the uploaded PDFs.")

    embeddings = get_embeddings(api_key=api_key)
    vector_store = create_faiss_store(all_chunks, embeddings)
    retriever = get_retriever(vector_store, k=4)

    st.session_state.chunks = all_chunks
    st.session_state.vector_store = vector_store
    st.session_state.retriever = retriever
    st.session_state.document_names = processed_names


def document_is_ready() -> bool:
    return "retriever" in st.session_state and "chunks" in st.session_state


def render_chat_history() -> None:
    messages = st.session_state.memory.chat_memory.messages
    if not messages:
        st.info("Ask your first question about the uploaded document(s).")
        return

    for message in messages:
        role = "user" if message.type == "human" else "assistant"
        with st.chat_message(role):
            st.write(message.content)


init_session_state()

# Page title
st.markdown('<h1 class="app-header">SmartQuery</h1>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Upload multiple PDFs, index with OpenAI + FAISS, and query your corpus in real-time.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-logo">🤖 SmartQuery</div>', unsafe_allow_html=True)
    st.divider()

    # Dynamic API Key Setup
    st.subheader("🔑 OpenAI API Config")
    env_key = os.getenv("OPENAI_API_KEY")
    is_key_placeholder = (not env_key or env_key == "your_openai_api_key_here")

    if is_key_placeholder:
        user_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="No API key found in env. Please input your key to use the app.",
        )
    else:
        user_api_key = st.text_input(
            "OpenAI API Key (Optional)",
            type="password",
            placeholder="Using key from .env...",
            help="Leave blank to use the .env key, or input a key to override it.",
        )

    # Determine which key to use
    active_api_key = user_api_key.strip() if user_api_key else (None if is_key_placeholder else env_key)

    st.divider()

    # Multiple PDF Uploader
    st.subheader("📄 Upload Documents")
    uploaded_pdfs = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select multiple PDF documents to query them collectively.",
    )

    process_disabled = not uploaded_pdfs or not active_api_key
    if not active_api_key:
        st.warning("Please configure an OpenAI API key to continue.")

    if st.button("Process PDFs", type="primary", disabled=process_disabled, use_container_width=True):
        reset_document_state()
        with st.spinner("Reading PDFs, chunking text, generating embeddings, and building FAISS index..."):
            try:
                process_pdfs(uploaded_pdfs, api_key=active_api_key)
                st.success("Documents processed successfully.")
            except Exception as exc:
                st.error(f"Processing failed: {exc}")

    if document_is_ready():
        st.divider()
        st.subheader("📊 Indexed Corpus")
        st.write(f"**Loaded Files:**")
        for name in st.session_state.document_names:
            st.markdown(f"- `{name}`")
        st.write(f"**Total Chunks:** `{len(st.session_state.chunks)}`")

        if st.button("Clear documents", use_container_width=True):
            reset_document_state()
            st.rerun()

        if st.button("Clear chat history", use_container_width=True):
            st.session_state.memory = create_memory()
            st.session_state.latest_sources = []
            st.rerun()


if not document_is_ready():
    st.info("Upload PDF(s) from the sidebar and click 'Process PDFs' to start Q&A.")
else:
    render_chat_history()

    if st.session_state.latest_sources:
        with st.expander("🔍 Retracted Source References from Latest Query"):
            for index, source in enumerate(st.session_state.latest_sources, start=1):
                doc_name = source.metadata.get("source", "Unknown Document")
                page = source.metadata.get("page")
                page_label = f"page {page + 1}" if isinstance(page, int) else "unknown page"
                st.markdown(f"**Source {index} — `{doc_name}` ({page_label})**")
                st.write(source.page_content)
                if index < len(st.session_state.latest_sources):
                    st.divider()

    question = st.chat_input("Ask a question about the uploaded document(s)...")

    if question:
        with st.spinner("Retrieving relevant chunks and generating an answer..."):
            try:
                _, sources = answer_question(
                    question,
                    st.session_state.retriever,
                    st.session_state.memory,
                    api_key=active_api_key,
                )
                st.session_state.latest_sources = sources
                st.rerun()
            except Exception as exc:
                st.error(f"Question answering failed: {exc}")

