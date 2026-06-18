"""
embeddings.py - Embedding Model Setup
Initialises and returns the embedding model used for vectorisation.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def get_openai_embeddings():
    """
    Return an OpenAI embeddings instance.

    Reads OPENAI_API_KEY and EMBEDDING_MODEL from environment variables.

    Returns:
        OpenAIEmbeddings instance.
    """
    from langchain_openai import OpenAIEmbeddings

    model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in environment variables.")

    return OpenAIEmbeddings(model=model, openai_api_key=api_key)


def get_huggingface_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Return a HuggingFace embeddings instance (free, runs locally).

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        HuggingFaceEmbeddings instance.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=model_name)


def get_embeddings(provider: str = "openai"):
    """
    Factory function — return the appropriate embeddings model.

    Args:
        provider: "openai" or "huggingface".

    Returns:
        Embeddings instance.
    """
    if provider == "openai":
        return get_openai_embeddings()
    elif provider == "huggingface":
        return get_huggingface_embeddings()
    else:
        raise ValueError(f"Unsupported embedding provider: '{provider}'. Choose 'openai' or 'huggingface'.")
