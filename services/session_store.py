"""
services/session_store.py
Thread-safe in-memory session store with TTL.
"""

import time
from threading import Lock
from config import settings

_store: dict = {}
_lock  = Lock()
TTL    = settings.SESSION_EXPIRY * 60


def save(session_id: str, data: dict) -> None:
    with _lock:
        _store[session_id] = {"ts": time.time(), "data": data}


def get(session_id: str) -> dict | None:
    with _lock:
        entry = _store.get(session_id)
        if not entry:
            return None
        if time.time() - entry["ts"] > TTL:
            del _store[session_id]
            return None
        return entry["data"]


def delete(session_id: str) -> None:
    with _lock:
        _store.pop(session_id, None)


def list_all() -> list[str]:
    with _lock:
        now = time.time()
        return [k for k, v in _store.items() if now - v["ts"] <= TTL]


def count() -> int:
    return len(list_all())
