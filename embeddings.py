import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


load_dotenv()


def get_embeddings() -> OpenAIEmbeddings:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it to .env before processing a PDF.")

    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model, api_key=api_key)
