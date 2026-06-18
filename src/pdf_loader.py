"""
pdf_loader.py - PDF Loading & Text Extraction
Handles loading PDF files and splitting them into chunks.
"""

import os
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF file and return a list of LangChain Document objects.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of Document objects with page content and metadata.
    """
    try:
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    except Exception as e:
        raise RuntimeError(f"Failed to load PDF '{file_path}': {e}")


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.

    Args:
        documents:     List of Document objects.
        chunk_size:    Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def load_and_split_pdf(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Convenience function: load a PDF and return split chunks.

    Args:
        file_path:     Path to the PDF file.
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of chunked Document objects ready for embedding.
    """
    documents = load_pdf(file_path)
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    return chunks
