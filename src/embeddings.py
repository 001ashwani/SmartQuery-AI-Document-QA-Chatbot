import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


load_dotenv()


def get_embeddings(api_key: str = None) -> OpenAIEmbeddings:
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if api_key == "your_openai_api_key_here":
        api_key = None

    if api_key and api_key.startswith("mock"):
        from langchain_core.embeddings.fake import FakeEmbeddings
        # Return a type-compatible mock embeddings instance
        return FakeEmbeddings(size=1536)

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Please configure it in the sidebar or .env file.")

    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model, openai_api_key=api_key)


