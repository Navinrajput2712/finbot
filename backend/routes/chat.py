"""
FinBot — backend/routes/chat.py
=================================
Main chat endpoint — processes user financial queries
through the RAG pipeline and returns grounded responses.

Endpoint:
    POST /chat
"""

import logging
from typing import Dict, List
from fastapi import APIRouter, Request, HTTPException

from backend.schemas import ChatRequest, ChatResponse, Source
from rag.pipeline import ask_finbot, is_market_query
from backend.market_data import get_stock_data, extract_ticker_from_query

logger = logging.getLogger(__name__)
router = APIRouter()

# ── In-memory session store ──────────────────────────────────
# Stores conversation history per session_id
# Format: { session_id: [ {"role": ..., "content": ...}, ... ] }
session_store: Dict[str, List[dict]] = {}

MAX_HISTORY = 10  # Last 5 turns = 10 messages


def get_session_history(session_id: str) -> List[dict]:
    """
    Retrieve conversation history for a session.
    Returns last 5 turns (10 messages) maximum.

    Args:
        session_id: Unique session identifier

    Returns:
        List of message dicts with role and content
    """
    if session_id not in session_store:
        session_store[session_id] = []
    return session_store[session_id][-MAX_HISTORY:]


def update_session(
    session_id: str,
    user_message: str,
    bot_message: str
) -> None:
    """
    Append user and assistant messages to session history.

    Args:
        session_id: Unique session identifier
        user_message: User's question
        bot_message: FinBot's response
    """
    if session_id not in session_store:
        session_store[session_id] = []

    session_store[session_id].append(
        {"role": "user", "content": user_message}
    )
    session_store[session_id].append(
        {"role": "assistant", "content": bot_message}
    )

    # Keep only last MAX_HISTORY messages
    if len(session_store[session_id]) > MAX_HISTORY:
        session_store[session_id] = session_store[session_id][-MAX_HISTORY:]


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: Request,
    body: ChatRequest
) -> ChatResponse:
    """
    Main FinBot chat endpoint.

    Steps:
    1. Get session history
    2. Check if live market query → fetch market data
    3. Run RAG pipeline with NVIDIA NIM LLM
    4. Update session history
    5. Return response with sources and confidence

    Args:
        request: FastAPI request (used to access app.state.vectorstore)
        body: ChatRequest with message and session_id

    Returns:
        ChatResponse with answer, sources, confidence, latency
    """
    logger.info(
        f"Chat request — session={body.session_id} | "
        f"message='{body.message[:80]}'"
    )

    # Get vectorstore from app state
    vectorstore = getattr(request.app.state, "vectorstore", None)
    if vectorstore is None:
        raise HTTPException(
            status_code=503,
            detail="ChromaDB vectorstore not loaded. Check server logs."
        )

    # Get session history
    chat_history = get_session_history(body.session_id)

    # Check for live market query
    market_context = None
    if is_market_query(body.message):
        ticker = extract_ticker_from_query(body.message)
        if ticker:
            try:
                market_data = get_stock_data(ticker)
                market_context = (
                    f"LIVE STOCK DATA for {market_data.ticker}:\n"
                    f"Current Price : ₹{market_data.current_price}\n"
                    f"Change        : {market_data.change_percent:+.2f}%\n"
                    f"Market Cap    : {market_data.market_cap}\n"
                    f"Exchange      : {market_data.exchange}\n"
                    f"Last Updated  : {market_data.last_updated}"
                )
                logger.info(f"Market data fetched for ticker: {ticker}")
            except Exception as e:
                logger.warning(f"Market data fetch failed: {e}")

    # Run RAG pipeline
    result = ask_finbot(
        query=body.message,
        chat_history=chat_history,
        vectorstore=vectorstore,
        market_context=market_context,
    )

    # Update session history
    update_session(
        session_id=body.session_id,
        user_message=body.message,
        bot_message=result["answer"],
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
        f"Response sent — session={body.session_id} | "
        f"confidence={result['confidence']} | "
        f"latency={result['latency_ms']}ms"
    )

    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        confidence=result["confidence"],
        latency_ms=result["latency_ms"],
        session_id=body.session_id,
        model=result.get("model", "meta/llama-3.1-8b-instruct"),
    )


@router.delete("/chat/{session_id}")
async def clear_session(session_id: str) -> dict:
    """
    Clear conversation history for a session.

    Args:
        session_id: Session to clear

    Returns:
        Confirmation message
    """
    if session_id in session_store:
        del session_store[session_id]
        logger.info(f"Session cleared: {session_id}")
        return {"message": f"Session {session_id} cleared successfully"}

    return {"message": f"Session {session_id} not found"}
