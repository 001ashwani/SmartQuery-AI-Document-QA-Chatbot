from typing import List

from langchain.schema import Document
from langchain_community.vectorstores import FAISS


def create_faiss_store(chunks: List[Document], embeddings) -> FAISS:
    if not chunks:
        raise ValueError("Cannot build a FAISS index without text chunks.")

    return FAISS.from_documents(chunks, embeddings)


def get_retriever(vector_store: FAISS, k: int = 4):
    return vector_store.as_retriever(search_kwargs={"k": k})


def retrieve_documents(retriever, question: str) -> List[Document]:
    if hasattr(retriever, "invoke"):
        return retriever.invoke(question)

    return retriever.get_relevant_documents(question)
