"""
FinBot — backend/schemas.py
============================
Pydantic v2 request/response models for all FastAPI endpoints.

All API input validation and output formatting is handled here.
"""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, field_validator


# ============================================================
# CHAT SCHEMAS
# ============================================================

class ChatRequest(BaseModel):
    """Request model for POST /chat endpoint."""
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str
    session_id: str
    include_sources: bool = True

    @field_validator("message")
    @classmethod
    def message_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message cannot be empty")
        if len(v) > 2000:
            raise ValueError("Message too long — max 2000 characters")
        return v

    @field_validator("session_id")
    @classmethod
    def session_id_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("session_id cannot be empty")
        return v


class Source(BaseModel):
    """A single source document used to generate the response."""
    file_name: str
    page_number: int
    relevance_score: float


class ChatResponse(BaseModel):
    """Response model for POST /chat endpoint."""
    answer: str
    sources: List[Source]
    confidence: float
    latency_ms: int
    session_id: str
    model: str
    disclaimer: str = (
        "⚠️ This is AI-generated information for educational purposes only. "
        "Please consult a SEBI-registered investment advisor before making "
        "any financial decisions."
    )


# ============================================================
# MARKET DATA SCHEMAS
# ============================================================

class MarketData(BaseModel):
    """Response model for GET /market/{ticker} endpoint."""
    ticker: str
    current_price: float
    change_percent: float
    currency: str = "INR"
    market_cap: Optional[str] = None
    last_updated: str
    exchange: Optional[str] = None


class IndexData(BaseModel):
    """Response model for market indices."""
    nifty: Optional[float] = None
    sensex: Optional[float] = None
    nifty_change_percent: Optional[float] = None
    sensex_change_percent: Optional[float] = None
    last_updated: str


# ============================================================
# HEALTH SCHEMAS
# ============================================================

class HealthResponse(BaseModel):
    """Response model for GET /health endpoint."""
    status: str                  # "healthy" or "degraded"
    model: str
    chromadb_status: str         # "connected" or "error"
    document_count: int
    uptime_seconds: float
    version: str = "1.0.0"


# ============================================================
# INGEST SCHEMAS
# ============================================================

class IngestResponse(BaseModel):
    """Response model for POST /ingest endpoint."""
    status: str
    message: str
    chunks_added: int
    collection_name: str
