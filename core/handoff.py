"""One-time, in-memory hand-off of a session's analysis across a page reload.

Switching between light and dark mode requires a browser reload, and a reload
starts a new Streamlit session. Just before reloading, the current analysis is
parked here under an unguessable random token; the new session claims it once
and it is deleted. Entries expire after ``TTL_SECONDS`` and are never written
to disk, so this is not persistent storage and nothing outlives the switch.
"""
from __future__ import annotations

import secrets
import threading
import time
from typing import Any

TTL_SECONDS = 120
MAX_ENTRIES = 64
_LOCK = threading.Lock()
_STORE: dict[str, tuple[float, dict[str, Any]]] = {}


def _purge(now: float) -> None:
    for k in [k for k, (t, _) in _STORE.items() if now - t > TTL_SECONDS]:
        del _STORE[k]
    while len(_STORE) > MAX_ENTRIES:
        del _STORE[min(_STORE, key=lambda k: _STORE[k][0])]


def park(payload: dict[str, Any]) -> str:
    token = secrets.token_urlsafe(24)
    now = time.time()
    with _LOCK:
        _purge(now)
        _STORE[token] = (now, payload)
    return token


def claim(token: str) -> dict[str, Any] | None:
    """Return and delete the payload; None if unknown or expired."""
    if not token or len(token) > 64:
        return None
    now = time.time()
    with _LOCK:
        _purge(now)
        item = _STORE.pop(token, None)
    return item[1] if item else None
