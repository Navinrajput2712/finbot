"""
FinBot — backend/routes/chat.py
================================
Main chat endpoint — processes user financial queries
through the RAG pipeline and returns grounded responses.

Endpoint:
    POST /chat
"""

import json
import logging
from fastapi import APIRouter, Request, HTTPException

from backend.schemas import ChatRequest, ChatResponse, Source
from backend.db import db
from rag.pipeline import ask_finbot, is_market_query
from backend.market_data import get_stock_data, extract_ticker_from_query

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_HISTORY = 10


def _get_chat_history(session_id: str) -> list:
    """Get last N messages from SQLite for RAG context."""
    messages = db.get_messages(session_id)
    recent = messages[-MAX_HISTORY:]
    return [{"role": m["role"], "content": m["content"]} for m in recent]


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: Request,
    body: ChatRequest
) -> ChatResponse:
    """
    Main FinBot chat endpoint.

    Steps:
    1. Ensure session exists in SQLite
    2. Get session history from DB
    3. Check if live market query -> fetch market data
    4. Run RAG pipeline with NVIDIA NIM LLM
    5. Persist messages to DB
    6. Return response with sources and confidence
    """
    logger.info(
        "Chat request — session=%s | message='%s'",
        body.session_id, body.message[:80],
    )

    # Ensure session exists
    if not db.get_session(session_id=body.session_id):
        title = body.message[:40].strip()
        db.create_session(session_id=body.session_id, title=title)

    # Get vectorstore from app state
    vectorstore = getattr(request.app.state, "vectorstore", None)
    if vectorstore is None:
        raise HTTPException(
            status_code=503,
            detail="ChromaDB vectorstore not loaded. Check server logs."
        )

    # Get session history from SQLite
    chat_history = _get_chat_history(body.session_id)

    # Check for live market query
    market_context = None
    if is_market_query(body.message):
        ticker = extract_ticker_from_query(body.message)
        if ticker:
            try:
                market_data = get_stock_data(ticker)
                market_context = (
                    f"LIVE STOCK DATA for {market_data.ticker}:\n"
                    f"Current Price : Rs{market_data.current_price}\n"
                    f"Change        : {market_data.change_percent:+.2f}%\n"
                    f"Market Cap    : {market_data.market_cap}\n"
                    f"Exchange      : {market_data.exchange}\n"
                    f"Last Updated  : {market_data.last_updated}"
                )
                logger.info("Market data fetched for ticker: %s", ticker)
            except Exception as e:
                logger.warning("Market data fetch failed: %s", e)

    # Run RAG pipeline — pass session_id so it can merge session collection
    result = ask_finbot(
        query=body.message,
        chat_history=chat_history,
        vectorstore=vectorstore,
        market_context=market_context,
        session_id=body.session_id,
    )

    # Persist messages to SQLite
    db.add_message(body.session_id, "user", body.message)
    db.add_message(
        body.session_id,
        "assistant",
        result["answer"],
        sources_json=json.dumps(result.get("sources", [])),
    )

    # Build sources list
    sources = []
    if body.include_sources:
        for src in result.get("sources", []):
            sources.append(Source(
                file_name=src.get("file_name", "unknown"),
                page_number=src.get("page_number", 0),
                relevance_score=src.get("relevance_score", 0.0),
            ))

    logger.info(
        "Response sent — session=%s | confidence=%.2f | latency=%dms",
        body.session_id, result["confidence"], result["latency_ms"],
    )

    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        confidence=result["confidence"],
        latency_ms=result["latency_ms"],
        session_id=body.session_id,
        model=result.get("model", "meta/llama-3.1-8b-instruct"),
    )
