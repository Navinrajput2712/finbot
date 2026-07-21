"""
FinBot — backend/routes/sessions.py
====================================
Session management endpoints for the Claude-style sidebar.

Endpoints:
    GET    /sessions                          — list all sessions
    GET    /sessions/{session_id}/messages    — full message history
    PATCH  /sessions/{session_id}             — rename session
    DELETE /sessions/{session_id}             — delete session + uploaded docs
"""

import logging
from fastapi import APIRouter, HTTPException

from backend.db import db
from backend.schemas import (
    SessionSummary,
    SessionListResponse,
    SessionMessagesResponse,
    SessionRenameRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions() -> SessionListResponse:
    """Return all sessions sorted by most recently updated."""
    sessions = db.list_sessions()
    return SessionListResponse(sessions=[
        SessionSummary(**s) for s in sessions
    ])


@router.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
async def get_session_messages(session_id: str) -> SessionMessagesResponse:
    """Return the full message history for one session."""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = db.get_messages(session_id)
    return SessionMessagesResponse(
        session_id=session_id,
        title=session["title"],
        messages=messages,
    )


@router.patch("/sessions/{session_id}")
async def rename_session(session_id: str, body: SessionRenameRequest) -> dict:
    """Rename a session's title."""
    updated = db.update_session_title(session_id, body.title)
    if not updated:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "ok", "session_id": session_id, "title": body.title}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    """Delete a session, its messages, and its uploaded-doc collection."""
    import chromadb

    # Delete session-scoped Chroma collection if it exists
    try:
        import os
        chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        client = chromadb.PersistentClient(path=chroma_path)
        client.delete_collection(f"session_{session_id}")
        logger.info("Deleted session collection: session_%s", session_id)
    except Exception as e:
        logger.info("No session collection to delete: %s", e)

    deleted = db.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "ok", "session_id": session_id}
