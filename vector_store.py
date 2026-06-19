from typing import List

from langchain.schema import Document
from langchain_community.vectorstores import FAISS


def build_faiss_index(chunks: List[Document], embeddings) -> FAISS:
    return FAISS.from_documents(chunks, embeddings)


def get_relevant_chunks(vector_store: FAISS, question: str, k: int = 4) -> List[Document]:
    return vector_store.similarity_search(question, k=k)
