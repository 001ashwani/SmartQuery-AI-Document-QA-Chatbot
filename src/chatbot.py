import os
from typing import List, Tuple

from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage
from langchain.schema import Document
from langchain_openai import ChatOpenAI

from .vector_store import retrieve_documents


load_dotenv()


def get_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")

    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, openai_api_key=api_key, temperature=0)


def create_memory() -> ConversationBufferMemory:
    return ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
    )


def build_context(documents: List[Document]) -> str:
    return "\n\n".join(
        f"Source {index}:\n{document.page_content}"
        for index, document in enumerate(documents, start=1)
    )


def format_chat_history(messages: List[BaseMessage]) -> str:
    if not messages:
        return "No previous conversation."

    formatted_messages = []
    for message in messages:
        role = "User" if message.type == "human" else "Assistant"
        formatted_messages.append(f"{role}: {message.content}")

    return "\n".join(formatted_messages)


def answer_question(question: str, retriever, memory: ConversationBufferMemory) -> Tuple[str, List[Document]]:
    source_documents = retrieve_documents(retriever, question)
    context = build_context(source_documents)
    chat_history = memory.load_memory_variables({}).get("chat_history", [])
    formatted_history = format_chat_history(chat_history)

    prompt = (
        "You are SmartQuery, a PDF question-answering assistant. "
        "Use the previous conversation to understand follow-up questions, "
        "but answer using only the PDF context below. If the answer is not present, "
        "say: I could not find that in the uploaded PDF.\n\n"
        f"Previous conversation:\n{formatted_history}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    response = get_llm().invoke(prompt)
    answer = response.content
    memory.save_context({"question": question}, {"answer": answer})

    return answer, source_documents
