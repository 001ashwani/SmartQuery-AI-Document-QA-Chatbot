import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from chatbot import answer_question
from document_loader import load_pdf, split_text
from embeddings import get_embeddings
from vector_store import build_faiss_index, get_relevant_chunks


load_dotenv()

st.set_page_config(page_title="SmartQuery MVP", page_icon="SQ", layout="wide")


def reset_document_state() -> None:
    st.session_state.pop("chunks", None)
    st.session_state.pop("vector_store", None)
    st.session_state.pop("document_name", None)


def process_uploaded_pdf(uploaded_file) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name

    try:
        text = load_pdf(temp_path)
        chunks = split_text(text)

        if not chunks:
            raise ValueError("No readable text found in this PDF.")

        embeddings = get_embeddings()
        vector_store = build_faiss_index(chunks, embeddings)

        st.session_state.chunks = chunks
        st.session_state.vector_store = vector_store
        st.session_state.document_name = uploaded_file.name
    finally:
        os.unlink(temp_path)


def is_document_ready() -> bool:
    return "vector_store" in st.session_state and "chunks" in st.session_state


st.title("SmartQuery MVP")
st.caption("PDF upload -> read PDF -> chunk text -> FAISS -> ask question -> answer")

with st.sidebar:
    st.header("Document")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_pdf is not None:
        if st.button("Process PDF", type="primary", use_container_width=True):
            reset_document_state()
            with st.spinner("Reading, chunking, and indexing PDF..."):
                try:
                    process_uploaded_pdf(uploaded_pdf)
                    st.success("PDF indexed successfully.")
                except Exception as exc:
                    st.error(f"Could not process PDF: {exc}")

    if is_document_ready():
        st.divider()
        st.write(f"Ready: {st.session_state.document_name}")
        st.write(f"Chunks: {len(st.session_state.chunks)}")

        if st.button("Clear PDF", use_container_width=True):
            reset_document_state()
            st.rerun()


if not is_document_ready():
    st.info("Upload a PDF and click Process PDF to start asking questions.")
else:
    question = st.text_input("Ask a question about the PDF")

    if st.button("Get Answer", disabled=not question.strip()):
        with st.spinner("Searching the PDF and drafting an answer..."):
            try:
                relevant_chunks = get_relevant_chunks(
                    st.session_state.vector_store,
                    question,
                    k=4,
                )
                answer = answer_question(question, relevant_chunks)

                st.subheader("Answer")
                st.write(answer)

                with st.expander("Sources"):
                    for index, chunk in enumerate(relevant_chunks, start=1):
                        st.markdown(f"**Source {index}**")
                        st.write(chunk.page_content)
            except Exception as exc:
                st.error(f"Could not answer the question: {exc}")
