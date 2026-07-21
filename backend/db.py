"""
FinBot — backend/db.py
======================
Lightweight SQLite persistence for sessions and messages.
Replaces the in-memory session_store dict so history survives restarts.

Usage:
    from backend.db import init_db, db
    init_db()
    db.create_session("my-session-id", "How to save tax?")
    db.add_message("my-session-id", "user", "How to save tax?")
"""

import os
import json
import sqlite3
import logging
from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("FINBOT_DB_PATH", "./finbot.db")


def _get_conn(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        return _get_conn(self.db_path)

    # ── Sessions ────────────────────────────────────────────
    def create_session(self, session_id: str, title: str = "") -> dict:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, title[:100], now, now),
            )
        return {"session_id": session_id, "title": title[:100], "created_at": now, "updated_at": now}

    def list_sessions(self) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT session_id, title, created_at, updated_at "
                "FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_session(self, session_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT session_id, title, created_at, updated_at "
                "FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def update_session_title(self, session_id: str, title: str) -> bool:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                (title[:100], now, session_id),
            )
        return cur.rowcount > 0

    def delete_session(self, session_id: str) -> bool:
        with self._conn() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cur = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        return cur.rowcount > 0

    def touch_session(self, session_id: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )

    # ── Messages ────────────────────────────────────────────
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources_json: str = "[]",
    ) -> dict:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO messages (session_id, role, content, sources_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, sources_json, now),
            )
            self.touch_session(session_id)
        return {"id": cur.lastrowid, "role": role, "content": content, "created_at": now}

    def get_messages(self, session_id: str) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, role, content, sources_json, created_at "
                "FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            try:
                d["sources"] = json.loads(d.pop("sources_json"))
            except (json.JSONDecodeError, TypeError):
                d["sources"] = []
            results.append(d)
        return results

    def get_last_user_message(self, session_id: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT content FROM messages "
                "WHERE session_id = ? AND role = 'user' ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        return row["content"] if row else None


def init_db(db_path: str = DB_PATH) -> None:
    """Create tables if they don't exist."""
    conn = _get_conn(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources_json TEXT DEFAULT '[]',
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
        CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);
    """)
    conn.close()
    logger.info("SQLite database initialized at %s", db_path)


# Module-level singleton
db = Database()
