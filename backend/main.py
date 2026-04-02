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
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File
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
    logger.info("   Docs: http://localhost:8000/docs")
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

# ── CORS Middleware ──────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include Routers ──────────────────────────────────────────
from backend.routes.chat   import router as chat_router
from backend.routes.health import router as health_router

app.include_router(chat_router,   tags=["Chat"])
app.include_router(health_router, tags=["Health"])


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

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)) -> dict:
    """
    Admin endpoint — upload a PDF and add it to ChromaDB.

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
    uvicorn.run(
        "backend.main:app",
        host=os.getenv("BACKEND_HOST", "0.0.0.0"),
        port=int(os.getenv("BACKEND_PORT", 8000)),
        reload=True,
    )
