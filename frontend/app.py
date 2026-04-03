"""
FinBot — frontend/app.py
=========================
Streamlit multi-turn chat UI for FinBot financial advisory chatbot.
Connects to FastAPI backend at localhost:8000.

Usage:
    streamlit run frontend/app.py
"""

import uuid
import requests
import streamlit as st
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
BACKEND_URL  = "http://localhost:8000"
APP_TITLE    = "FinBot 💰"
APP_TAGLINE  = "AI-Powered Financial Advisor for India"


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="FinBot — AI Financial Advisor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0f1117; }

    /* Header */
    .finbot-header {
        background: linear-gradient(135deg, #1a1f2e, #2d3748);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid #2d3748;
    }
    .finbot-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f0b429;
        margin: 0;
    }
    .finbot-tagline {
        font-size: 1rem;
        color: #a0aec0;
        margin: 4px 0 0 0;
    }

    /* Chip buttons */
    .stButton > button {
        background: #1a2744;
        color: #90cdf4;
        border: 1px solid #2b4c7e;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.82rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #2b4c7e;
        color: white;
        border-color: #63b3ed;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #1a1f2e;
        border-right: 1px solid #2d3748;
    }

    /* Chat messages */
    .stChatMessage { border-radius: 10px; margin-bottom: 8px; }

    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
def init_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    if "total_latency_ms" not in st.session_state:
        st.session_state.total_latency_ms = 0
    if "total_confidence" not in st.session_state:
        st.session_state.total_confidence = 0.0
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []
    if "chip_query" not in st.session_state:
        st.session_state.chip_query = None

init_session_state()


# ============================================================
# API HELPERS
# ============================================================
def check_api_health() -> dict:
    """Check if FastAPI backend is healthy."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"status": "degraded"}
    except Exception:
        return {"status": "offline"}


def send_chat_message(message: str) -> dict:
    """Send message to FastAPI /chat endpoint."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={
                "message": message,
                "session_id": st.session_state.session_id,
                "include_sources": True,
            },
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()
        return {
            "answer": f"❌ API Error {response.status_code}: {response.text}",
            "sources": [],
            "confidence": 0.0,
            "latency_ms": 0,
        }
    except requests.exceptions.Timeout:
        return {
            "answer": "⏱️ Request timed out. Please try again in a few seconds.",
            "sources": [],
            "confidence": 0.0,
            "latency_ms": 0,
        }
    except requests.exceptions.ConnectionError:
        return {
            "answer": "❌ Cannot connect to FinBot API. Make sure the backend is running:\n`uvicorn backend.main:app --reload --port 8000`",
            "sources": [],
            "confidence": 0.0,
            "latency_ms": 0,
        }
    except Exception as e:
        return {
            "answer": f"❌ Unexpected error: {str(e)}",
            "sources": [],
            "confidence": 0.0,
            "latency_ms": 0,
        }


def get_index_data() -> dict:
    """Fetch Nifty and Sensex data from backend."""
    try:
        nifty  = requests.get(f"{BACKEND_URL}/market/%5ENSEI",  timeout=5).json()
        sensex = requests.get(f"{BACKEND_URL}/market/%5EBSESN", timeout=5).json()
        return {"nifty": nifty, "sensex": sensex}
    except Exception:
        return {}


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar():
    """Render the sidebar with stats and market data."""
    with st.sidebar:

        # ── Logo & Title ──────────────────────────────────────
        st.markdown("## 💰 FinBot")
        st.markdown("*AI Financial Advisor for India*")
        st.markdown("---")

        # ── API Health ────────────────────────────────────────
        st.markdown("### 🔌 System Status")
        health = check_api_health()
        status = health.get("status", "offline")

        if status == "healthy":
            st.success("✅ API Online")
            st.caption(f"Model: `{health.get('model','N/A')}`")
            st.caption(f"Docs: {health.get('document_count', 0)} chunks")
        elif status == "degraded":
            st.warning("⚠️ API Degraded")
        else:
            st.error("❌ API Offline")
            st.caption("Run: `uvicorn backend.main:app --port 8000`")

        st.markdown("---")

        # ── Session Stats ─────────────────────────────────────
        st.markdown("### 📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", st.session_state.total_queries)
        with col2:
            avg_latency = (
                st.session_state.total_latency_ms //
                max(st.session_state.total_queries, 1)
            )
            st.metric("Avg Speed", f"{avg_latency}ms")

        st.markdown("---")

        # ── Market Snapshot ───────────────────────────────────
        st.markdown("### 📈 Market Snapshot")
        with st.spinner("Loading..."):
            indices = get_index_data()

        if indices:
            nifty  = indices.get("nifty",  {})
            sensex = indices.get("sensex", {})

            if nifty.get("current_price"):
                chg  = nifty.get("change_percent", 0)
                sign = "▲" if chg >= 0 else "▼"
                st.metric(
                    "Nifty 50",
                    f"{nifty['current_price']:,.0f}",
                    f"{sign} {abs(chg):.2f}%"
                )

            if sensex.get("current_price"):
                chg  = sensex.get("change_percent", 0)
                sign = "▲" if chg >= 0 else "▼"
                st.metric(
                    "Sensex",
                    f"{sensex['current_price']:,.0f}",
                    f"{sign} {abs(chg):.2f}%"
                )
        else:
            st.caption("Market data unavailable.")

        st.markdown("---")

        # ── Action Buttons ────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages         = []
                st.session_state.total_queries    = 0
                st.session_state.total_latency_ms = 0
                st.session_state.total_confidence = 0.0
                st.session_state.last_sources     = []
                st.session_state.session_id       = str(uuid.uuid4())
                st.rerun()

        with col2:
            if st.session_state.messages:
                chat_export  = f"FinBot Chat Export\n{'='*40}\n"
                chat_export += f"Session: {st.session_state.session_id}\n"
                chat_export += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                chat_export += "="*40 + "\n\n"
                for msg in st.session_state.messages:
                    role = "You" if msg["role"] == "user" else "FinBot"
                    chat_export += f"{role}:\n{msg['content']}\n\n"
                    chat_export += "-"*40 + "\n\n"

                st.download_button(
                    "💾 Export",
                    data=chat_export,
                    file_name=f"finbot_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        st.markdown("---")
        st.caption("v1.0.0 • Powered by LLaMA 3.1 8B")
        st.caption("NVIDIA NIM API • ChromaDB RAG")


# ============================================================
# MAIN CHAT AREA
# ============================================================
def render_header():
    """Render the main page header — no disclaimer."""
    st.markdown(
        "<div class='finbot-header'>"
        "<p class='finbot-title'>💰 FinBot</p>"
        "<p class='finbot-tagline'>AI-Powered Financial Advisor for India • "
        "Budgeting • Investing • Taxation • Insurance • Loans</p>"
        "</div>",
        unsafe_allow_html=True
    )


def render_starter_chips():
    """Render clickable starter question buttons."""
    st.markdown("**💡 Quick Questions:**")
    chips = [
        ("💰", "How to save tax under Section 80C?"),
        ("📈", "How much SIP to build Rs 1 crore corpus?"),
        ("🏠", "How is home loan EMI calculated?"),
        ("🛡️", "What is difference between term and ULIP insurance?"),
        ("📊", "What is the 50/30/20 budgeting rule?"),
    ]

    cols = st.columns(len(chips))
    for i, (emoji, question) in enumerate(chips):
        with cols[i]:
            if st.button(f"{emoji} {question[:22]}...", key=f"chip_{i}"):
                st.session_state.chip_query = question


def render_chat_history():
    """Render all chat messages — answer only, no sources/confidence/latency."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def process_message(user_input: str):
    """
    Process a user message through the RAG pipeline.
    Shows only the answer — no sources, confidence, or latency.
    """
    if not user_input or not user_input.strip():
        return

    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call API and show response
    with st.chat_message("assistant"):
        with st.spinner("💭 FinBot is thinking..."):
            result = send_chat_message(user_input)

        answer     = result.get("answer", "Sorry, I could not process your request.")
        sources    = result.get("sources", [])
        confidence = result.get("confidence", 0.0)
        latency_ms = result.get("latency_ms", 0)

        # ✅ Show answer ONLY — sources, confidence, latency all hidden
        st.markdown(answer)

    # Save to session state (stored internally, never shown)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "confidence": confidence,
        "latency_ms": latency_ms,
    })

    # Update internal stats only
    st.session_state.total_queries    += 1
    st.session_state.total_latency_ms += latency_ms
    st.session_state.total_confidence += confidence
    st.session_state.last_sources      = sources


# ============================================================
# MAIN APP
# ============================================================
def main():
    """Main Streamlit app entry point."""

    # Render sidebar
    render_sidebar()

    # Render header — disclaimer removed
    render_header()

    # Show starter chips only if no messages yet
    if not st.session_state.messages:
        render_starter_chips()
        st.markdown("---")

    # Render chat history — answer only
    render_chat_history()

    # Handle chip query
    if st.session_state.chip_query:
        query = st.session_state.chip_query
        st.session_state.chip_query = None
        process_message(query)
        st.rerun()

    # Chat input
    user_input = st.chat_input(
        "Ask FinBot about budgeting, investing, taxes, insurance, or loans..."
    )

    if user_input:
        process_message(user_input)
        st.rerun()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()