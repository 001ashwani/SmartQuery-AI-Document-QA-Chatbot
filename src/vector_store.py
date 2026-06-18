"""
vector_store.py - Vector Store CRUD Operations
Handles creating, saving, loading, and querying the vector store.
"""

import os
from typing import List, Optional

from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_store")


def create_faiss_store(documents: List[Document], embeddings):
    """
    Create a FAISS vector store from documents.

    Args:
        documents:  List of Document chunks.
        embeddings: Embeddings model instance.

    Returns:
        FAISS vector store instance.
    """
    from langchain_community.vectorstores import FAISS

    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


def save_vector_store(vector_store, path: str = VECTOR_STORE_PATH):
    """
    Persist a FAISS vector store to disk.

    Args:
        vector_store: FAISS vector store instance.
        path:         Directory path to save the index.
    """
    os.makedirs(path, exist_ok=True)
    vector_store.save_local(path)
    print(f"✅ Vector store saved to '{path}'")


def load_vector_store(embeddings, path: str = VECTOR_STORE_PATH):
    """
    Load a FAISS vector store from disk.

    Args:
        embeddings: Embeddings model instance (must match the saved index).
        path:       Directory path of the saved index.

    Returns:
        FAISS vector store instance.
    """
    from langchain_community.vectorstores import FAISS

    if not os.path.exists(path):
        raise FileNotFoundError(f"No vector store found at '{path}'. Please index documents first.")

    vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    print(f"✅ Vector store loaded from '{path}'")
    return vector_store


def similarity_search(vector_store, query: str, k: int = 4) -> List[Document]:
    """
    Retrieve the top-k most relevant documents for a query.

    Args:
        vector_store: FAISS vector store instance.
        query:        User query string.
        k:            Number of documents to retrieve.

    Returns:
        List of the most relevant Document objects.
    """
    results = vector_store.similarity_search(query, k=k)
    return results


def get_retriever(vector_store, k: int = 4):
    """
    Return a LangChain retriever from the vector store.

    Args:
        vector_store: FAISS vector store instance.
        k:            Number of documents to retrieve per query.

    Returns:
        VectorStoreRetriever instance.
    """
    return vector_store.as_retriever(search_kwargs={"k": k})
