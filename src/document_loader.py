from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document




def load_pdf(file_path: str, original_filename: str = None) -> List[Document]:
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    readable_pages = []
    for doc in documents:
        if doc.page_content.strip():
            if original_filename:
                doc.metadata["source"] = original_filename
            readable_pages.append(doc)

    if not readable_pages:
        filename_display = original_filename if original_filename else file_path
        raise ValueError(f"No readable text found in PDF: {filename_display}")

    return readable_pages


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def load_and_chunk_pdf(
    file_path: str,
    original_filename: str = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    documents = load_pdf(file_path, original_filename=original_filename)
    return chunk_documents(documents, chunk_size, chunk_overlap)

