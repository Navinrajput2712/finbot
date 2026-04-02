"""
FinBot — backend/routes/health.py
===================================
Health check endpoint — verifies all systems are running.

Endpoint:
    GET /health
"""

import time
import logging
from fastapi import APIRouter, Request
from backend.schemas import HealthResponse
from backend.llm_loader import test_nim_connection

logger    = logging.getLogger(__name__)
router    = APIRouter()
START_TIME = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """
    Health check endpoint.
    Verifies: NVIDIA NIM API, ChromaDB connection, document count.

    Returns:
        HealthResponse with status of all systems
    """
    import os

    # Check ChromaDB
    chromadb_status = "error"
    document_count  = 0

    try:
        vectorstore = getattr(request.app.state, "vectorstore", None)
        if vectorstore:
            document_count  = vectorstore._collection.count()
            chromadb_status = "connected"
        else:
            chromadb_status = "not loaded"
    except Exception as e:
        logger.error(f"ChromaDB health check failed: {e}")
        chromadb_status = "error"

    # Check NVIDIA NIM
    nim_ok = test_nim_connection()

    # Overall status
    status = "healthy" if (nim_ok and chromadb_status == "connected") else "degraded"

    uptime = round(time.time() - START_TIME, 1)

    logger.info(
        f"Health check — status={status} | "
        f"chromadb={chromadb_status} | "
        f"docs={document_count} | "
        f"nim={nim_ok}"
    )

    return HealthResponse(
        status=status,
        model=os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct"),
        chromadb_status=chromadb_status,
        document_count=document_count,
        uptime_seconds=uptime,
    )
