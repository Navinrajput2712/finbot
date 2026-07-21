"""
FinBot — backend/main.py
=========================
FastAPI application entry point.
Loads ChromaDB vectorstore on startup and serves all API endpoints.

Usage:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Request, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# LIFESPAN — Startup & Shutdown
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.
    Loads ChromaDB vectorstore ONCE at startup — reused for all requests.
    """
    logger.info("="*50)
    logger.info("  FinBot API — Starting up...")
    logger.info("="*50)

    # Initialize SQLite database
    try:
        from backend.db import init_db
        init_db()
        logger.info("✅ SQLite database initialized")
    except Exception as e:
        logger.error(f"❌ Failed to init database: {str(e)}")

    # Load ChromaDB vectorstore
    try:
        from rag.retriever import load_vectorstore
        logger.info("Loading ChromaDB vectorstore...")
        app.state.vectorstore = load_vectorstore()
        logger.info("✅ ChromaDB vectorstore loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load ChromaDB: {str(e)}")
        logger.error("   Run: python -m rag.ingest first!")
        app.state.vectorstore = None

    # Warm up cross-encoder reranker (cached for all subsequent requests)
    try:
        from rag.retriever import get_reranker
        get_reranker()
        logger.info("✅ Cross-encoder reranker warmed up")
    except Exception as e:
        logger.error(f"❌ Failed to load reranker: {str(e)}")

    # Verify NVIDIA NIM connection
    try:
        from backend.llm_loader import test_nim_connection
        logger.info("Testing NVIDIA NIM API connection...")
        nim_ok = test_nim_connection()
        if nim_ok:
            logger.info("✅ NVIDIA NIM API connected")
        else:
            logger.warning("⚠️  NVIDIA NIM API connection failed — check API key")
    except Exception as e:
        logger.error(f"❌ NVIDIA NIM test failed: {str(e)}")

    logger.info("✅ FinBot API ready!")
    backend_host = os.getenv("BACKEND_HOST", "0.0.0.0")
    backend_port = os.getenv("PORT", os.getenv("BACKEND_PORT", "8000"))
    logger.info(f"   Listening on: http://{backend_host}:{backend_port}")
    logger.info("="*50)

    yield  # App runs here

    # Shutdown
    logger.info("FinBot API shutting down...")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="FinBot API",
    description=(
        "AI-powered financial advisory chatbot API. "
        "Powered by LLaMA 3.1 8B via NVIDIA NIM + RAG pipeline."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── Logging Middleware ────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"[BACKEND LOG] Incoming Request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"[BACKEND LOG] Outgoing Response: {request.method} {request.url.path} | Status: {response.status_code}")
    return response

# ── CORS Middleware ──────────────────────────────────────────
origins = [
    "https://finbot-8.onrender.com",
    "http://localhost:8501",  # Streamlit default
    "http://localhost:8000",
]

# Allow additional origins from environment variables if specified
cors_origins_env = os.getenv("CORS_ORIGINS")
if cors_origins_env:
    origins.extend([o.strip() for o in cors_origins_env.split(",")])

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include Routers ──────────────────────────────────────────
from backend.routes.chat      import router as chat_router
from backend.routes.health    import router as health_router
from backend.routes.sessions  import router as sessions_router
from backend.routes.upload    import router as upload_router

app.include_router(chat_router,      tags=["Chat"])
app.include_router(health_router,    tags=["Health"])
app.include_router(sessions_router,  tags=["Sessions"])
app.include_router(upload_router,    tags=["Upload"])


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
async def root() -> dict:
    """Root endpoint — confirms API is running."""
    return {
        "message": "FinBot API is running! 💰",
        "docs": "http://localhost:8000/docs",
        "health": "http://localhost:8000/health",
        "version": "1.0.0",
        "model": os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct"),
    }


# ============================================================
# MARKET ENDPOINT
# ============================================================

@app.get("/market/{ticker}")
async def get_market_data(ticker: str) -> dict:
    """
    Fetch live stock data for a given ticker symbol.

    Args:
        ticker: Stock symbol (e.g. RELIANCE, TCS, AAPL)

    Returns:
        Current price, change%, market cap
    """
    from backend.market_data import get_stock_data
    try:
        data = get_stock_data(ticker)
        return data.model_dump()
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================
# INGEST ENDPOINT (Admin)
# ============================================================

async def verify_admin_key(x_admin_key: str = Header(...)) -> None:
    """
    Dependency that verifies the X-Admin-Key header against the
    ADMIN_API_KEY environment variable. Returns 401 if missing or mismatched.
    """
    expected = os.getenv("ADMIN_API_KEY")
    if not expected:
        raise HTTPException(
            status_code=500,
            detail="ADMIN_API_KEY is not configured on the server"
        )
    if not x_admin_key or not secrets.compare_digest(x_admin_key, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing admin API key"
        )


@app.post("/ingest")
async def ingest_pdf(
    file: UploadFile = File(...),
    _admin: None = Depends(verify_admin_key),
) -> dict:
    """
    Admin endpoint — upload a PDF and add it to ChromaDB.

    Requires the ``X-Admin-Key`` header to match the ``ADMIN_API_KEY``
    environment variable.  If ``ADMIN_API_KEY`` is not set the endpoint
    refuses all requests.

    Args:
        file: PDF file to ingest

    Returns:
        Ingestion status and chunk count
    """
    import shutil
    from pathlib import Path
    from rag.ingest import load_documents, split_documents, create_vectorstore

    if not file.filename.endswith(".pdf"):
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    # Save uploaded file
    kb_path = Path(os.getenv("KNOWLEDGE_BASE_PATH", "./data/knowledge_base"))
    kb_path.mkdir(parents=True, exist_ok=True)
    save_path = kb_path / file.filename

    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"PDF saved: {save_path}")

        # Re-ingest everything
        documents = load_documents(str(kb_path))
        chunks    = split_documents(documents)
        create_vectorstore(chunks, reset=True)

        return {
            "status": "success",
            "message": f"PDF '{file.filename}' ingested successfully",
            "chunks_added": len(chunks),
            "collection_name": os.getenv(
                "CHROMA_COLLECTION_NAME", "finbot_knowledge"
            ),
        }

    except Exception as e:
        logger.error(f"Ingest failed: {str(e)}")
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn
    # Render injects $PORT — always prefer it over hardcoded BACKEND_PORT
    port = int(os.getenv("PORT", os.getenv("BACKEND_PORT", "8000")))
    uvicorn.run(
        "backend.main:app",
        host=os.getenv("BACKEND_HOST", "0.0.0.0"),
        port=port,
        reload=False,
    )
