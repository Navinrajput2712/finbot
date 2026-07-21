"""
FinBot — frontend/app.py
=========================
Claude-style chat UI for the FinBot financial advisory chatbot.
Connects to FastAPI backend for RAG-powered responses.

Usage:
    streamlit run frontend/app.py
"""

import os
import uuid
import time
import requests
import streamlit as st
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
APP_TITLE = "FinBot"
APP_TAGLINE = "AI-Powered Financial Advisor for India"
MAX_FILE_SIZE = 20 * 1024 * 1024
ALLOWED_TYPES = ["pdf", "docx", "txt", "csv"]

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="FinBot",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — Claude-inspired clean design
# ============================================================
st.markdown("""
<style>
    /* ── Global ─────────────────────────────────────── */
    .stApp { background: #faf9f7; }
    [data-theme="dark"] .stApp { background: #1a1a1a; }

    /* ── Sidebar ────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #f5f3ef;
        border-right: 1px solid #e5e2db;
        padding-top: 1rem;
    }
    [data-theme="dark"] section[data-testid="stSidebar"] {
        background: #202020;
        border-right: 1px solid #333;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background: #1a1a1a;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 16px;
        font-weight: 600;
        width: 100%;
        transition: background 0.2s;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #333;
    }
    [data-theme="dark"] section[data-testid="stSidebar"] .stButton > button {
        background: #e8e8e8;
        color: #1a1a1a;
    }
    [data-theme="dark"] section[data-testid="stSidebar"] .stButton > button:hover {
        background: #ccc;
    }

    /* ── Session list items ─────────────────────────── */
    .session-item {
        padding: 8px 12px;
        border-radius: 6px;
        cursor: pointer;
        margin-bottom: 2px;
        font-size: 0.88rem;
        color: #374151;
        transition: background 0.15s;
    }
    .session-item:hover { background: #e8e5df; }
    .session-item.active { background: #e0ddd7; font-weight: 500; }
    [data-theme="dark"] .session-item { color: #d1d5db; }
    [data-theme="dark"] .session-item:hover { background: #333; }
    [data-theme="dark"] .session-item.active { background: #444; }

    .session-title { font-size: 0.88rem; line-height: 1.3; }
    .session-time { font-size: 0.72rem; color: #9ca3af; margin-top: 1px; }

    /* ── Main chat area ─────────────────────────────── */
    .block-container {
        max-width: 780px;
        margin: 0 auto;
        padding-top: 2rem;
    }

    /* ── Message styling ────────────────────────────── */
    .msg-row { margin-bottom: 1.2rem; }
    .msg-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #6b7280;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .msg-label .avatar {
        width: 22px; height: 22px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
    }
    .avatar-user { background: #dbeafe; color: #2563eb; }
    .avatar-bot { background: #fef3c7; color: #d97706; }
    .msg-content {
        font-size: 0.95rem;
        line-height: 1.65;
        color: #1f2937;
        padding-left: 28px;
    }
    [data-theme="dark"] .msg-label { color: #9ca3af; }
    [data-theme="dark"] .msg-content { color: #e5e7eb; }

    /* ── Sources ─────────────────────────────────────── */
    .sources-toggle {
        font-size: 0.78rem;
        color: #6b7280;
        cursor: pointer;
        padding-left: 28px;
        margin-top: 4px;
    }
    .source-chip {
        display: inline-block;
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.72rem;
        margin: 2px 4px 2px 0;
        color: #4b5563;
    }
    [data-theme="dark"] .source-chip {
        background: #2d2d2d;
        border-color: #444;
        color: #9ca3af;
    }

    /* ── Empty state ─────────────────────────────────── */
    .empty-greeting {
        text-align: center;
        padding: 3rem 1rem 1rem;
    }
    .empty-greeting h2 {
        font-size: 1.8rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.3rem;
    }
    .empty-greeting p {
        color: #6b7280;
        font-size: 0.95rem;
    }
    [data-theme="dark"] .empty-greeting h2 { color: #f3f4f6; }
    [data-theme="dark"] .empty-greeting p { color: #9ca3af; }

    /* ── Suggestion cards ────────────────────────────── */
    .suggestion-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        max-width: 560px;
        margin: 1.5rem auto 0;
    }
    .suggestion-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 14px 16px;
        cursor: pointer;
        font-size: 0.88rem;
        color: #374151;
        transition: border-color 0.2s, box-shadow 0.2s;
        text-align: left;
    }
    .suggestion-card:hover {
        border-color: #a78bfa;
        box-shadow: 0 2px 8px rgba(167,139,250,0.15);
    }
    [data-theme="dark"] .suggestion-card {
        background: #262626;
        border-color: #444;
        color: #d1d5db;
    }

    /* ── File chip ────────────────────────────────────── */
    .file-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.8rem;
        color: #1d4ed8;
        margin-bottom: 8px;
    }
    .file-chip button {
        background: none;
        border: none;
        color: #6b7280;
        cursor: pointer;
        font-size: 1rem;
        padding: 0;
        line-height: 1;
    }

    /* ── System message ──────────────────────────────── */
    .system-msg {
        text-align: center;
        font-size: 0.82rem;
        color: #6b7280;
        padding: 6px 0;
    }

    /* ── Hide Streamlit branding ────────────────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* ── Fix suggestion cards in Streamlit buttons ──── */
    div[data-testid="stHorizontalBlock"] > div > button {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 14px 16px;
        text-align: left;
        font-size: 0.88rem;
        color: #374151;
        height: auto;
        transition: border-color 0.2s;
    }
    div[data-testid="stHorizontalBlock"] > div > button:hover {
        border-color: #a78bfa;
        box-shadow: 0 2px 8px rgba(167,139,250,0.15);
    }
    [data-theme="dark"] div[data-testid="stHorizontalBlock"] > div > button {
        background: #262626;
        border-color: #444;
        color: #d1d5db;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# API HELPERS
# ============================================================

def api_get(path: str, timeout: int = 10):
    try:
        r = requests.get(f"{BACKEND_URL}{path}", timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def api_post(path: str, json_data: dict = None, timeout: int = 60):
    try:
        r = requests.post(f"{BACKEND_URL}{path}", json=json_data, timeout=timeout)
        return r.status_code, r.json() if r.status_code == 200 else {"error": r.text}
    except requests.exceptions.ConnectionError:
        return 0, {"error": f"Cannot connect to {BACKEND_URL}"}
    except requests.exceptions.Timeout:
        return 0, {"error": "Request timed out"}
    except Exception as e:
        return 0, {"error": str(e)}


def api_patch(path: str, json_data: dict):
    try:
        r = requests.patch(f"{BACKEND_URL}{path}", json=json_data, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def api_delete(path: str):
    try:
        r = requests.delete(f"{BACKEND_URL}{path}", timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def api_upload(file_bytes, filename: str, session_id: str):
    try:
        r = requests.post(
            f"{BACKEND_URL}/upload",
            files={"file": (filename, file_bytes)},
            data={"session_id": session_id},
            timeout=120,
        )
        return r.status_code, r.json() if r.status_code == 200 else {"error": r.text}
    except requests.exceptions.ConnectionError:
        return 0, {"error": f"Cannot connect to {BACKEND_URL}"}
    except requests.exceptions.Timeout:
        return 0, {"error": "Upload timed out"}
    except Exception as e:
        return 0, {"error": str(e)}


# ============================================================
# STATE MANAGEMENT
# ============================================================

def init_state():
    if "current_session" not in st.session_state:
        st.session_state.current_session = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sessions" not in st.session_state:
        st.session_state.sessions = []
    if "pending_file" not in st.session_state:
        st.session_state.pending_file = None
    if "pending_filename" not in st.session_state:
        st.session_state.pending_filename = None
    if "rename_target" not in st.session_state:
        st.session_state.rename_target = None


def refresh_sessions():
    data = api_get("/sessions")
    if data:
        st.session_state.sessions = data.get("sessions", [])


def load_session_messages(session_id: str):
    data = api_get(f"/sessions/{session_id}/messages")
    if data:
        st.session_state.messages = data.get("messages", [])
        st.session_state.current_session = session_id
    else:
        st.session_state.messages = []
        st.session_state.current_session = session_id


def new_chat():
    sid = str(uuid.uuid4())
    st.session_state.current_session = sid
    st.session_state.messages = []
    st.session_state.pending_file = None
    st.session_state.pending_filename = None
    refresh_sessions()


def relative_time(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff = now - dt
        if diff.days > 30:
            return dt.strftime("%b %d")
        elif diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds > 3600:
            return f"{diff.seconds // 3600}h ago"
        elif diff.seconds > 60:
            return f"{diff.seconds // 60}m ago"
        else:
            return "just now"
    except Exception:
        return ""


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("### 💰 FinBot")
        st.markdown("")

        # New chat button
        if st.button("+ New chat", use_container_width=True):
            new_chat()
            st.rerun()

        st.markdown("")

        # Session list
        if st.session_state.sessions:
            st.markdown("**Chats**")
            for s in st.session_state.sessions:
                sid = s["session_id"]
                title = s.get("title", "New chat") or "New chat"
                updated = relative_time(s.get("updated_at", ""))
                is_active = sid == st.session_state.current_session

                col1, col2 = st.columns([5, 1])
                with col1:
                    btn_label = f"{'**' if is_active else ''}{title}{'**' if is_active else ''}"
                    if st.button(
                        btn_label,
                        key=f"sess_{sid}",
                        use_container_width=True,
                    ):
                        load_session_messages(sid)
                        st.rerun()
                with col2:
                    if st.button("...", key=f"menu_{sid}"):
                        st.session_state.rename_target = sid

                # Rename dialog
                if st.session_state.rename_target == sid:
                    new_title = st.text_input(
                        "Rename",
                        value=title,
                        key=f"rename_{sid}",
                        label_visibility="collapsed",
                    )
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        if st.button("Save", key=f"rsave_{sid}"):
                            api_patch(f"/sessions/{sid}", {"title": new_title})
                            st.session_state.rename_target = None
                            refresh_sessions()
                            st.rerun()
                    with rc2:
                        if st.button("Delete", key=f"rdel_{sid}"):
                            api_delete(f"/sessions/{sid}")
                            if st.session_state.current_session == sid:
                                st.session_state.current_session = None
                                st.session_state.messages = []
                            st.session_state.rename_target = None
                            refresh_sessions()
                            st.rerun()
        else:
            st.caption("No conversations yet")

        # Bottom: Market data + status
        st.markdown("---")
        st.markdown("**System Status**")
        health = api_get("/health", timeout=5)
        if health and health.get("status") == "healthy":
            st.success("API Online")
            st.caption(f"Docs: {health.get('document_count', 0)} chunks")
        else:
            st.error("API Offline")

        with st.expander("Market Snapshot"):
            nifty = api_get("/market/%5ENSEI", timeout=5)
            sensex = api_get("/market/%5EBSESN", timeout=5)
            if nifty:
                chg = nifty.get("change_percent", 0)
                sign = "+" if chg >= 0 else ""
                st.metric("Nifty 50", f"{nifty.get('current_price', 0):,.0f}", f"{sign}{chg:.2f}%")
            if sensex:
                chg = sensex.get("change_percent", 0)
                sign = "+" if chg >= 0 else ""
                st.metric("Sensex", f"{sensex.get('current_price', 0):,.0f}", f"{sign}{chg:.2f}%")


# ============================================================
# MAIN PANEL
# ============================================================

def render_empty_state():
    st.markdown("""
    <div class="empty-greeting">
        <h2>Good {}</h2>
        <p>I'm FinBot, your AI financial advisor. How can I help?</p>
    </div>
    """.format(
        "morning" if datetime.now().hour < 12 else
        "afternoon" if datetime.now().hour < 17 else "evening"
    ), unsafe_allow_html=True)

    suggestions = [
        ("💰", "How to save tax under Section 80C?"),
        ("📈", "How much SIP to build Rs 1 crore corpus?"),
        ("🏠", "How is home loan EMI calculated?"),
        ("🛡️", "Difference between term and ULIP insurance?"),
        ("📊", "What is the 50/30/20 budgeting rule?"),
        ("💳", "How does CIBIL score affect loan approval?"),
    ]

    cols = st.columns(2)
    for i, (emoji, question) in enumerate(suggestions):
        col = cols[i % 2]
        with col:
            if st.button(
                f"{emoji}  {question}",
                key=f"sug_{i}",
                use_container_width=True,
            ):
                st.session_state.pending_query = question
                st.rerun()


def render_messages():
    for msg in st.session_state.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        sources = msg.get("sources", [])

        if role == "user":
            st.markdown(f"""
            <div class="msg-row">
                <div class="msg-label">
                    <span class="avatar avatar-user">U</span> You
                </div>
                <div class="msg-content">{content}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-row">
                <div class="msg-label">
                    <span class="avatar avatar-bot">F</span> FinBot
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(content)

            # Sources
            if sources:
                with st.expander("Sources", expanded=False):
                    for src in sources:
                        fname = src.get("file_name", "unknown")
                        page = src.get("page_number", "?")
                        score = src.get("relevance_score", 0)
                        st.markdown(
                            f'<span class="source-chip">{fname} p.{page} ({score:.2f})</span>',
                            unsafe_allow_html=True,
                        )

            # Copy button
            st.markdown(
                f'<div class="sources-toggle">📋 Copy</div>',
                unsafe_allow_html=True,
            )


def handle_send(user_input: str):
    if not user_input or not user_input.strip():
        return

    # Ensure we have a session
    if not st.session_state.current_session:
        st.session_state.current_session = str(uuid.uuid4())

    session_id = st.session_state.current_session

    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Handle file upload if pending
    if st.session_state.pending_file:
        file_bytes = st.session_state.pending_file
        filename = st.session_state.pending_filename
        st.session_state.pending_file = None
        st.session_state.pending_filename = None

        status, resp = api_upload(file_bytes, filename, session_id)
        if status == 200:
            chunk_count = resp.get("chunk_count", 0)
            page_count = resp.get("page_count", 0)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"📄 Added `{filename}` — {page_count} pages, {chunk_count} chunks. I can now answer questions about it.",
            })
        else:
            error = resp.get("error", "Upload failed")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Upload failed: {error}",
            })

    # Show spinner and call API
    with st.spinner("Thinking..."):
        status, result = api_post("/chat", {
            "message": user_input,
            "session_id": session_id,
            "include_sources": True,
        })

    if status == 200:
        answer = result.get("answer", "Sorry, I could not process that.")
        sources = result.get("sources", [])
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
    else:
        error = result.get("error", "Unknown error")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ Error: {error}",
        })

    # Refresh session list
    refresh_sessions()


# ============================================================
# MAIN APP
# ============================================================

def main():
    init_state()
    render_sidebar()

    # Current session indicator
    if st.session_state.current_session:
        st.caption(f"Session: {st.session_state.current_session[:8]}...")

    # Render content
    if not st.session_state.messages:
        render_empty_state()
    else:
        render_messages()

    # Handle suggestion chip clicks
    if hasattr(st.session_state, "pending_query") and st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None
        handle_send(query)
        st.rerun()

    # File upload area
    uploaded_file = st.file_uploader(
        "Attach file",
        type=ALLOWED_TYPES,
        label_visibility="collapsed",
        key="file_uploader",
    )
    if uploaded_file:
        if uploaded_file.name != st.session_state.pending_filename:
            st.session_state.pending_file = uploaded_file.getvalue()
            st.session_state.pending_filename = uploaded_file.name
            st.markdown(
                f'<div class="file-chip">📎 {uploaded_file.name} <button onclick="this.parentElement.remove()">×</button></div>',
                unsafe_allow_html=True,
            )

    # Chat input
    user_input = st.chat_input("Ask FinBot about budgeting, investing, taxes, insurance, or loans...")
    if user_input:
        handle_send(user_input)
        st.rerun()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
