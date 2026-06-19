import os
from typing import List

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import ChatOpenAI


load_dotenv()


def get_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it to .env before asking questions.")

    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, api_key=api_key, temperature=0)


def answer_question(question: str, chunks: List[Document]) -> str:
    context = "\n\n".join(chunk.page_content for chunk in chunks)
    prompt = (
        "Answer the question using only the PDF context below. "
        "If the answer is not in the context, say you could not find it in the PDF.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    response = get_llm().invoke(prompt)
    return response.content
