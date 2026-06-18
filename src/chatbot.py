"""
chatbot.py - QA Chain & Conversation Logic
Builds the conversational retrieval chain and manages chat history.
"""

import os
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv()


def get_llm():
    """
    Initialise and return the LLM (ChatOpenAI by default).

    Returns:
        ChatOpenAI instance.
    """
    from langchain_openai import ChatOpenAI

    model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in environment variables.")

    return ChatOpenAI(model_name=model, openai_api_key=api_key, temperature=0.2)


def build_qa_chain(retriever):
    """
    Build a ConversationalRetrievalChain using the given retriever.

    Args:
        retriever: VectorStoreRetriever instance.

    Returns:
        ConversationalRetrievalChain instance.
    """
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory

    llm = get_llm()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return qa_chain


def ask_question(qa_chain, question: str) -> Tuple[str, List]:
    """
    Send a question to the QA chain and return the answer with sources.

    Args:
        qa_chain: ConversationalRetrievalChain instance.
        question: The user's question string.

    Returns:
        Tuple of (answer_text, source_documents).
    """
    result = qa_chain({"question": question})
    answer = result.get("answer", "I couldn't find an answer.")
    sources = result.get("source_documents", [])
    return answer, sources


def format_sources(source_documents: List) -> str:
    """
    Format source documents into a human-readable string.

    Args:
        source_documents: List of source Document objects.

    Returns:
        Formatted string of sources.
    """
    if not source_documents:
        return "No sources found."

    formatted = []
    for i, doc in enumerate(source_documents, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        snippet = doc.page_content[:200].strip().replace("\n", " ")
        formatted.append(f"**[{i}] {source}** (Page {page})\n> {snippet}...")

    return "\n\n".join(formatted)
