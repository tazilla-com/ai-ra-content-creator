# workshops_ai/graph/assets/modules/ui_helpers.py
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def now_iso() -> str:
    """
    Return current UTC time as ISO-8601 string with 'Z' suffix and no microseconds.
    Example: '2025-10-21T05:57:51Z'
    """
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _mk(kind: str, text: str | None = None, **extra: Any) -> dict:
    """
    Build a single UI message dict with a timestamp.

    Parameters
    ----------
    kind : str
        UI message kind (e.g., 'text', 'title', 'error', ...).
    text : Optional[str]
        Human-readable text for the message. If None, only metadata/extra are used.
    extra : Any
        Any additional fields to include (e.g., structured payload for the UI).

    Returns
    -------
    dict
        UI message with at least {'kind': kind, 'ts': <ISO time>} and optionally 'text' and extra fields.
    """
    msg: dict = {"kind": kind, "ts": now_iso()}
    if text is not None:
        msg["text"] = str(text)
    if extra:
        msg.update(extra)
    return msg


# --- Convenience message builders (kept tiny & generic on purpose) -------------------------------

def title(text: str, **extra: Any) -> dict:
    """
    Create a 'title' message. Use at the beginning of a response group.
    """
    return _mk("title", text, **extra)


def text_msg(text: str, **extra: Any) -> dict:
    """
    Create a simple 'text' message.
    """
    return _mk("text", text, **extra)


def error_msg(text: str, **extra: Any) -> dict:
    """
    Create an 'error' message.
    """
    return _mk("error", text, **extra)


def info_msg(text: str, **extra: Any) -> dict:
    """
    Create an 'info' message.
    """
    return _mk("info", text, **extra)


def success_msg(text: str, **extra: Any) -> dict:
    """
    Create a 'success' message.
    """
    return _mk("success", text, **extra)


# --- Envelope used by the chat/agent layer --------------------------------------------------------

def ui_response(message: str | dict, *, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build a front-end-friendly response envelope.

    Always returns:
      {
        "messages": [ ... ],
        "staging": { "needs_commit": False },
        ...extra   # merged at top level
      }

    Behavior:
    - If `message` is a string -> one 'text' message.
    - If `message` is a dict with key 'error' -> one 'error' message.
    - If `message` is a dict with key 'kind' -> use it directly as a message.
    """
    if isinstance(message, dict) and "kind" in message:
        messages = [message]
    elif isinstance(message, dict) and "error" in message:
        messages = [error_msg(str(message.get("error")))]
    else:
        messages = [text_msg(str(message))]

    envelope: Dict[str, Any] = {
        "messages": messages,
        "staging": {"needs_commit": False},
    }

    if extra:
        # Merge extra to top-level, avoiding accidental overwrite of 'messages'
        for k, v in extra.items():
            if k == "messages":
                continue
            envelope[k] = v

    return envelope


# --- Backwards/forwards-compat aliases ------------------------------------------------------------
build_ui_response = ui_response
ui = ui_response

__all__ = [
    "now_iso",
    "ui_response",
    "build_ui_response",
    "ui",
    "title",
    "text_msg",
    "error_msg",
    "info_msg",
    "success_msg",
]
