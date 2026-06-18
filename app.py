"""
SmartQuery — AI Document QA Chatbot
app.py  |  Main Streamlit application
"""

import os
import sys
import tempfile

import streamlit as st
from dotenv import load_dotenv

# ── Make src/ importable ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pdf_loader import load_and_split_pdf
from embeddings import get_embeddings
from vector_store import create_faiss_store, get_retriever
from chatbot import build_qa_chain, ask_question

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
#  Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartQuery — AI Document QA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Base ── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(16px);
    }
    [data-testid="stSidebar"] * { color: #e0e0f0 !important; }

    /* ── Hero header ── */
    .hero {
        text-align: center;
        padding: 2rem 1rem 1.2rem;
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
    }

    /* ── Status pills ── */
    .pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .pill-ready  { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid #34d399; }
    .pill-idle   { background: rgba(148,163,184,0.12); color: #94a3b8; border: 1px solid #475569; }
    .pill-error  { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid #f87171; }

    /* ── Chat bubbles ── */
    .bubble-user {
        display: flex;
        justify-content: flex-end;
        margin: 0.6rem 0;
    }
    .bubble-user .msg {
        background: linear-gradient(135deg, #6d28d9, #4f46e5);
        color: #fff;
        padding: 0.75rem 1.1rem;
        border-radius: 18px 18px 4px 18px;
        max-width: 72%;
        font-size: 0.95rem;
        line-height: 1.55;
        box-shadow: 0 4px 15px rgba(109,40,217,0.35);
    }
    .bubble-bot {
        display: flex;
        justify-content: flex-start;
        margin: 0.6rem 0;
    }
    .bubble-bot .msg {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        color: #e2e8f0;
        padding: 0.75rem 1.1rem;
        border-radius: 18px 18px 18px 4px;
        max-width: 72%;
        font-size: 0.95rem;
        line-height: 1.6;
        backdrop-filter: blur(10px);
    }
    .avatar {
        width: 32px; height: 32px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem;
        flex-shrink: 0;
    }
    .avatar-user { background: linear-gradient(135deg,#6d28d9,#4f46e5); margin-left:8px; }
    .avatar-bot  { background: linear-gradient(135deg,#0ea5e9,#6366f1); margin-right:8px; }

    /* ── Source citation card ── */
    .source-card {
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 10px;
        padding: 0.65rem 0.9rem;
        margin-top: 0.4rem;
        font-size: 0.82rem;
        color: #a5b4fc;
    }
    .source-card strong { color: #c4b5fd; }
    .source-snippet { color: #94a3b8; margin-top: 4px; font-style: italic; }

    /* ── Divider ── */
    .chat-divider { border: none; border-top: 1px solid rgba(255,255,255,0.07); margin: 1rem 0; }

    /* ── Input area ── */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
        padding: 0.65rem 1rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #6d28d9 !important;
        box-shadow: 0 0 0 3px rgba(109,40,217,0.25) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6d28d9, #4f46e5) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(109,40,217,0.45) !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03) !important;
        border: 2px dashed rgba(99,102,241,0.4) !important;
        border-radius: 14px !important;
        padding: 1rem !important;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }
    [data-testid="stMetricValue"] { color: #a78bfa !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { color: #64748b !important; }

    /* ── Section headings ── */
    .section-title {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #475569;
        margin: 1.2rem 0 0.5rem;
    }

    /* ── Spinner override ── */
    .stSpinner > div { border-top-color: #6d28d9 !important; }

    /* ── Scrollable chat area ── */
    .chat-container {
        max-height: 58vh;
        overflow-y: auto;
        padding-right: 4px;
    }
    .chat-container::-webkit-scrollbar { width: 4px; }
    .chat-container::-webkit-scrollbar-thumb {
        background: rgba(99,102,241,0.4);
        border-radius: 4px;
    }

    /* ── Welcome card ── */
    .welcome-card {
        text-align: center;
        padding: 3rem 1rem;
        color: #475569;
    }
    .welcome-card .icon { font-size: 3.5rem; margin-bottom: 0.8rem; }
    .welcome-card p { font-size: 1rem; max-width: 380px; margin: 0 auto; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "chat_history": [],        # list of {"role": "user"|"bot", "content": str, "sources": list}
        "qa_chain": None,
        "vector_store": None,
        "doc_loaded": False,
        "doc_name": None,
        "doc_chunks": 0,
        "embedding_provider": "openai",
        "processing": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def reset_chat():
    st.session_state.chat_history = []
    st.session_state.qa_chain = None
    st.session_state.vector_store = None
    st.session_state.doc_loaded = False
    st.session_state.doc_name = None
    st.session_state.doc_chunks = 0

@st.cache_resource(show_spinner=False)
def get_cached_embeddings(provider: str):
    return get_embeddings(provider)

def process_pdf(uploaded_file, provider: str):
    """Save PDF → chunk → embed → FAISS → retriever → QA chain."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        chunks = load_and_split_pdf(tmp_path)
        embeddings = get_cached_embeddings(provider)
        vs = create_faiss_store(chunks, embeddings)
        retriever = get_retriever(vs, k=4)
        chain = build_qa_chain(retriever)
        return chain, vs, len(chunks)
    finally:
        os.unlink(tmp_path)

def render_bubble(role: str, content: str, sources: list = None):
    if role == "user":
        st.markdown(
            f"""
            <div class="bubble-user">
                <div class="msg">{content}</div>
                <div class="avatar avatar-user">🧑</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="bubble-bot">
                <div class="avatar avatar-bot">🤖</div>
                <div class="msg">{content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if sources:
            for i, doc in enumerate(sources, 1):
                src  = os.path.basename(doc.metadata.get("source", "Unknown"))
                page = doc.metadata.get("page", "N/A")
                snippet = doc.page_content[:180].strip().replace("\n", " ")
                st.markdown(
                    f"""
                    <div class="source-card">
                        <strong>📄 Source {i} — {src}</strong>&nbsp;&nbsp;
                        <span style="color:#64748b">Page {page}</span>
                        <div class="source-snippet">"{snippet}…"</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 SmartQuery")
    st.markdown("<div class='section-title'>Document</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Upload a PDF to ask questions about it.",
        label_visibility="collapsed",
    )

    st.markdown("<div class='section-title'>Embedding Model</div>", unsafe_allow_html=True)
    provider = st.radio(
        "provider",
        options=["openai", "huggingface"],
        format_func=lambda x: "🌐 OpenAI (API key required)" if x == "openai" else "🤗 HuggingFace (free, local)",
        index=0,
        label_visibility="collapsed",
    )
    st.session_state.embedding_provider = provider

    if provider == "openai":
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            value=os.getenv("OPENAI_API_KEY", ""),
        )
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input

    process_btn = st.button("⚡ Process Document", use_container_width=True, disabled=uploaded_file is None)

    # ── Status ────────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Status</div>", unsafe_allow_html=True)
    if st.session_state.doc_loaded:
        st.markdown(
            f"<span class='pill pill-ready'>✓ Ready — {st.session_state.doc_name}</span>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        col1.metric("Chunks", st.session_state.doc_chunks)
        col2.metric("Model", provider.upper()[:4])
    else:
        st.markdown("<span class='pill pill-idle'>○ No document loaded</span>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        reset_chat()
        st.rerun()

    st.markdown(
        "<div style='color:#334155;font-size:0.75rem;margin-top:1.5rem;text-align:center'>"
        "SmartQuery v1.0 · Powered by LangChain & FAISS"
        "</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
#  Process document when button clicked
# ─────────────────────────────────────────────────────────────────────────────
if process_btn and uploaded_file:
    with st.spinner(f"⚙️ Processing **{uploaded_file.name}** — chunking, embedding & indexing…"):
        try:
            chain, vs, n_chunks = process_pdf(uploaded_file, provider)
            st.session_state.qa_chain      = chain
            st.session_state.vector_store  = vs
            st.session_state.doc_loaded    = True
            st.session_state.doc_name      = uploaded_file.name
            st.session_state.doc_chunks    = n_chunks
            st.session_state.chat_history  = []   # fresh chat for new doc
            st.success(f"✅ **{uploaded_file.name}** indexed — {n_chunks} chunks ready!")
        except ValueError as e:
            st.error(f"⚠️ Configuration error: {e}")
            st.info("💡 Make sure your OpenAI API key is set in the sidebar or `.env` file.")
        except Exception as e:
            st.error(f"❌ Failed to process document: {e}")

# ─────────────────────────────────────────────────────────────────────────────
#  Main UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>SmartQuery</h1>
        <p>Ask anything about your documents — powered by AI</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Chat area ─────────────────────────────────────────────────────────────────
chat_area = st.container()

with chat_area:
    if not st.session_state.chat_history:
        st.markdown(
            """
            <div class="welcome-card">
                <div class="icon">📄</div>
                <p>Upload a PDF in the sidebar, click <strong>Process Document</strong>,
                then start asking questions below.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            render_bubble(msg["role"], msg["content"], msg.get("sources"))
        st.markdown("</div>", unsafe_allow_html=True)

# ── Input row ─────────────────────────────────────────────────────────────────
st.markdown("<hr class='chat-divider'>", unsafe_allow_html=True)

col_input, col_send = st.columns([9, 1])

with col_input:
    user_input = st.text_input(
        "question",
        placeholder="Ask a question about your document…",
        label_visibility="collapsed",
        key="question_input",
        disabled=not st.session_state.doc_loaded,
    )

with col_send:
    send_btn = st.button(
        "➤",
        use_container_width=True,
        disabled=not st.session_state.doc_loaded or not user_input.strip(),
    )

# Hint when no document loaded
if not st.session_state.doc_loaded:
    st.caption("⬆️ Upload and process a PDF in the sidebar to enable the chat.")

# ─────────────────────────────────────────────────────────────────────────────
#  Handle question submission  (button OR Enter key)
# ─────────────────────────────────────────────────────────────────────────────
def submit_question(question: str):
    if not question.strip():
        return
    if not st.session_state.doc_loaded or st.session_state.qa_chain is None:
        st.warning("Please upload and process a document first.")
        return

    # Append user message
    st.session_state.chat_history.append(
        {"role": "user", "content": question, "sources": []}
    )

    with st.spinner("🤔 Thinking…"):
        try:
            answer, sources = ask_question(st.session_state.qa_chain, question)
            st.session_state.chat_history.append(
                {"role": "bot", "content": answer, "sources": sources}
            )
        except ValueError as e:
            err = f"Configuration error: {e}"
            st.session_state.chat_history.append(
                {"role": "bot", "content": f"⚠️ {err}", "sources": []}
            )
        except Exception as e:
            err = f"Something went wrong: {e}"
            st.session_state.chat_history.append(
                {"role": "bot", "content": f"❌ {err}", "sources": []}
            )

    st.rerun()

if send_btn and user_input.strip():
    submit_question(user_input)
elif user_input.strip() and st.session_state.get("question_input"):
    # Allow submitting with Enter by checking last keystroke state
    pass
