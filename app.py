import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from src.chatbot import answer_question, create_memory
from src.document_loader import load_and_chunk_pdf
from src.embeddings import get_embeddings
from src.vector_store import create_faiss_store, get_retriever


load_dotenv()

st.set_page_config(page_title="SmartQuery", page_icon="SQ", layout="wide")


def init_session_state() -> None:
    if "memory" not in st.session_state:
        st.session_state.memory = create_memory()
    if "latest_sources" not in st.session_state:
        st.session_state.latest_sources = []


def reset_document_state() -> None:
    for key in ("chunks", "vector_store", "retriever", "document_name"):
        st.session_state.pop(key, None)
    st.session_state.memory = create_memory()
    st.session_state.latest_sources = []


def process_pdf(uploaded_file) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name

    try:
        chunks = load_and_chunk_pdf(temp_path)
        embeddings = get_embeddings()
        vector_store = create_faiss_store(chunks, embeddings)
        retriever = get_retriever(vector_store, k=4)

        st.session_state.chunks = chunks
        st.session_state.vector_store = vector_store
        st.session_state.retriever = retriever
        st.session_state.document_name = uploaded_file.name
    finally:
        os.unlink(temp_path)


def document_is_ready() -> bool:
    return "retriever" in st.session_state and "chunks" in st.session_state


def render_chat_history() -> None:
    messages = st.session_state.memory.chat_memory.messages
    if not messages:
        st.info("Ask your first question about the uploaded PDF.")
        return

    for message in messages:
        role = "user" if message.type == "human" else "assistant"
        with st.chat_message(role):
            st.write(message.content)


init_session_state()

st.title("SmartQuery")
st.caption("Upload a PDF, index it with OpenAI embeddings + FAISS, then ask questions.")

with st.sidebar:
    st.header("PDF")
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

    process_disabled = uploaded_pdf is None
    if st.button("Process PDF", type="primary", disabled=process_disabled, use_container_width=True):
        reset_document_state()
        with st.spinner("Reading PDF, chunking text, creating embeddings, and building FAISS index..."):
            try:
                process_pdf(uploaded_pdf)
                st.success("PDF processed successfully.")
            except Exception as exc:
                st.error(f"PDF processing failed: {exc}")

    if document_is_ready():
        st.divider()
        st.write(f"Document: {st.session_state.document_name}")
        st.write(f"Chunks: {len(st.session_state.chunks)}")

        if st.button("Clear document", use_container_width=True):
            reset_document_state()
            st.rerun()

        if st.button("Clear chat", use_container_width=True):
            st.session_state.memory = create_memory()
            st.session_state.latest_sources = []
            st.rerun()


if not document_is_ready():
    st.info("Upload a PDF from the sidebar and click Process PDF to enable Q&A.")
else:
    render_chat_history()

    if st.session_state.latest_sources:
        with st.expander("Sources from latest answer"):
            for index, source in enumerate(st.session_state.latest_sources, start=1):
                page = source.metadata.get("page")
                page_label = f"page {page + 1}" if isinstance(page, int) else "unknown page"
                st.markdown(f"**Source {index} ({page_label})**")
                st.write(source.page_content)

    question = st.chat_input("Ask a question about the uploaded PDF")

    if question:
        with st.spinner("Retrieving relevant chunks and generating an answer..."):
            try:
                _, sources = answer_question(
                    question,
                    st.session_state.retriever,
                    st.session_state.memory,
                )
                st.session_state.latest_sources = sources
                st.rerun()
            except Exception as exc:
                st.error(f"Question answering failed: {exc}")
