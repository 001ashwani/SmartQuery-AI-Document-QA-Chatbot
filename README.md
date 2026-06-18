# SmartQuery — AI Document QA Chatbot 🤖

SmartQuery is an AI-powered document Q&A chatbot that lets you upload PDF documents and ask questions in natural language. It uses LangChain, OpenAI embeddings, and a FAISS/Chroma vector store to deliver accurate, context-aware answers.

---

## 🚀 Features

- 📄 Upload and parse PDF documents
- 🔍 Semantic search using vector embeddings
- 💬 Conversational Q&A with memory
- ⚡ Fast retrieval with FAISS / ChromaDB
- 🖥️ Clean Streamlit UI

---

## 🗂️ Project Structure

```
SmartQuery-AI-Document-QA-Chatbot/
│
├── app.py               # Streamlit app entry point
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (not committed)
├── .env.example         # Example env file
├── README.md
│
├── src/
│   ├── pdf_loader.py    # PDF loading & text extraction
│   ├── embeddings.py    # Embedding model setup
│   ├── vector_store.py  # Vector store CRUD operations
│   └── chatbot.py       # QA chain & conversation logic
│
├── assets/              # Images, logos, static files
├── docs/                # Project documentation
└── sample_docs/         # Sample PDFs for testing
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/SmartQuery-AI-Document-QA-Chatbot.git
cd SmartQuery-AI-Document-QA-Chatbot
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `HUGGINGFACE_API_TOKEN` | HuggingFace token (optional) |
| `VECTOR_STORE_PATH` | Path to persist vector store |
| `EMBEDDING_MODEL` | Embedding model name |
| `LLM_MODEL` | LLM model name |
| `CHUNK_SIZE` | Document chunk size |
| `CHUNK_OVERLAP` | Chunk overlap size |

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI GPT
- **Embeddings**: OpenAI / HuggingFace
- **Vector Store**: FAISS / ChromaDB
- **Framework**: LangChain
- **PDF Parsing**: PyPDF2 / pdfplumber

---

## 📄 License

MIT License — feel free to use and modify.
