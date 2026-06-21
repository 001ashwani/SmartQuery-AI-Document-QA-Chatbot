import os
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_classic.memory import ConversationBufferMemory

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from .vector_store import retrieve_documents



load_dotenv()


from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class MockChatOpenAI(BaseChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # We can extract text from messages to see if there is context
        prompt_text = ""
        for m in messages:
            if hasattr(m, "content"):
                prompt_text += str(m.content) + "\n"

        # Construct a nice mock answer that mentions the source files
        sources_found = []
        if "doc_a.pdf" in prompt_text:
            sources_found.append("doc_a.pdf")
        if "doc_b.pdf" in prompt_text:
            sources_found.append("doc_b.pdf")

        source_str = " and ".join([f"`{s}`" for s in sources_found]) if sources_found else "the documents"
        response_text = (
            f"This is a mock answer generated from {source_str} since a mock API key was used. "
            "Based on the retrieved context: "
        )
        if "doc_a.pdf" in prompt_text:
            response_text += "SmartQuery is a multiple PDF QA system using a FAISS vector store. "
        if "doc_b.pdf" in prompt_text:
            response_text += "OpenAI embeddings are used for semantic search, and the retriever gets top k chunks."

        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "mock-chat-openai"


def get_llm(api_key: str = None) -> ChatOpenAI:
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if api_key == "your_openai_api_key_here":
        api_key = None

    if api_key and api_key.startswith("mock"):
        return MockChatOpenAI()

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Please configure it in the sidebar or .env file.")

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
        f"Source {index} (From {document.metadata.get('source', 'Unknown Doc')}, page {document.metadata.get('page', 0) + 1}):\n{document.page_content}"
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


def answer_question(
    question: str,
    retriever,
    memory: ConversationBufferMemory,
    api_key: str = None,
) -> Tuple[str, List[Document]]:
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

    response = get_llm(api_key=api_key).invoke(prompt)
    answer = response.content
    memory.save_context({"question": question}, {"answer": answer})

    return answer, source_documents

