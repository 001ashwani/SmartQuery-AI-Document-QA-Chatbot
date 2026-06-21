import os
import sys
from unittest.mock import MagicMock, patch

# Ensure the root project directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.chatbot import answer_question, create_memory
from src.document_loader import chunk_documents
from src.vector_store import create_faiss_store, get_retriever


# Create a Mock Embeddings model
class MockEmbeddings(Embeddings):
    def embed_documents(self, texts):
        # Return a simple mock vector for each text
        return [[0.1] * 1536 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 1536


# Create a Mock Chat OpenAI Model
class MockChatModel(BaseChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # We can inspect prompt or return a fixed mock answer
        message = AIMessage(content="This is a mock answer based on the retrieved context.")
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "mock-chat"


def test_pipeline():
    print("Starting pipeline test...")

    # 1. Mocking PDF documents loading
    # We simulate loading 2 different PDFs: doc_a.pdf and doc_b.pdf
    doc_a_pages = [
        Document(page_content="SmartQuery is a multiple PDF Q&A system.", metadata={"source": "doc_a.pdf", "page": 0}),
        Document(page_content="It uses FAISS vector stores for retrieval.", metadata={"source": "doc_a.pdf", "page": 1}),
    ]
    doc_b_pages = [
        Document(page_content="OpenAI embeddings are used for semantic search.", metadata={"source": "doc_b.pdf", "page": 0}),
        Document(page_content="It retrieves the top k similar chunks.", metadata={"source": "doc_b.pdf", "page": 1}),
    ]
    
    all_pages = doc_a_pages + doc_b_pages

    # 2. Chunking Documents
    print("Testing document chunking...")
    chunks = chunk_documents(all_pages, chunk_size=100, chunk_overlap=10)
    assert len(chunks) >= 4, f"Expected at least 4 chunks, got {len(chunks)}"
    print(f"Successfully chunked into {len(chunks)} chunks.")

    # Validate that source and page metadata is preserved
    for chunk in chunks:
        assert "source" in chunk.metadata, "Metadata 'source' field was lost during chunking!"
        assert "page" in chunk.metadata, "Metadata 'page' field was lost during chunking!"
    print("Metadata tracking check passed.")

    # 3. Create FAISS Vector Store using mock embeddings
    print("Testing FAISS vector store creation...")
    embeddings = MockEmbeddings()
    vector_store = create_faiss_store(chunks, embeddings)
    retriever = get_retriever(vector_store, k=2)
    print("FAISS vector store and retriever successfully created.")

    # 4. Test QA Chain with Mock LLM
    print("Testing QA prompt formatting and LLM response generation...")
    memory = create_memory()

    # Mock get_llm inside src/chatbot
    with patch("src.chatbot.get_llm") as mock_get_llm:
        mock_get_llm.return_value = MockChatModel()

        question = "How does SmartQuery work?"
        answer, sources = answer_question(question, retriever, memory)

        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Retrieved {len(sources)} sources:")
        for idx, src in enumerate(sources, start=1):
            print(f"  Source {idx}: Document: {src.metadata['source']}, Page: {src.metadata['page'] + 1}")
            print(f"  Content: {src.page_content[:60]}...")

        # Assertions
        assert len(sources) == 2, "Expected 2 retrieved sources"
        assert answer == "This is a mock answer based on the retrieved context."
        assert len(memory.chat_memory.messages) == 2, "Memory should contain the question and answer"

    print("\nAll pipeline tests passed successfully!")



if __name__ == "__main__":
    test_pipeline()
