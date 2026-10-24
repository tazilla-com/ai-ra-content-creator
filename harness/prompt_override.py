"""Serve hosted prompts from files instead of Langfuse — to compare prompt versions.

``--prompts-dir DIR`` makes every prompt the application asks Langfuse for come from
``DIR/<name>.json`` when that file exists (``DIR/wizard/assets.json`` for ``wizard/assets``).
A file has the shape of one version in a Langfuse export: ``type`` (``chat`` or ``text``) and
``prompt`` (a list of ``{role, content}`` for chat, a string for text); ``version`` and
``labels`` are optional. The application then converts it exactly as it converts a fetched
prompt, through the Langfuse SDK's own prompt classes. A prompt without a file is fetched
from Langfuse as usual.

Only the client's ``get_prompt`` is replaced; tracing still goes through the real client,
so the application's Langfuse client must exist (``LANGFUSE_ENABLED`` and both keys).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_served: dict[str, str] = {}


def served() -> dict[str, str]:
    """Prompt name -> the first 12 hex digits of its file's sha256, for every prompt served from a file."""
    return dict(_served)


def _prompt_client(name: str, data: dict) -> Any:
    from langfuse.api.resources.prompts.types import Prompt_Chat, Prompt_Text
    from langfuse.model import ChatPromptClient, TextPromptClient

    common = {
        "name": name,
        "version": int(data.get("version") or 0),
        "labels": list(data.get("labels") or []),
        "tags": [],
        "config": data.get("config") or {},
    }
    if data.get("type") == "chat":
        return ChatPromptClient(Prompt_Chat.parse_obj({**common, "type": "chat", "prompt": data["prompt"]}))
    if data.get("type") == "text":
        return TextPromptClient(Prompt_Text.parse_obj({**common, "type": "text", "prompt": data["prompt"]}))
    raise ValueError(f"prompt file for {name!r}: type must be 'chat' or 'text', not {data.get('type')!r}")


class _FilePromptClient:
    """The application's Langfuse client, with get_prompt reading a file when one exists."""

    def __init__(self, inner: Any, prompts_dir: Path):
        self._inner = inner
        self._dir = prompts_dir

    def get_prompt(self, name: str, **kwargs: Any) -> Any:
        path = self._dir / f"{name}.json"
        if not path.is_file():
            return self._inner.get_prompt(name, **kwargs)
        return _prompt_client(name, json.loads(path.read_text(encoding="utf-8")))

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._inner, attr)


def install(prompts_dir: str | Path) -> dict[str, str]:
    """Wrap the application's Langfuse client so the prompts under ``prompts_dir`` come from their files."""
    from workshops_ai.llm import langfuse_helper

    root = Path(prompts_dir)
    files = sorted(root.rglob("*.json"))
    if not files:
        raise SystemExit(f"--prompts-dir {root}: no prompt files (<name>.json)")
    for path in files:
        name = path.relative_to(root).with_suffix("").as_posix()
        _prompt_client(name, json.loads(path.read_text(encoding="utf-8")))  # fail now, not mid-run
        _served[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:12]

    inner = langfuse_helper.get_langfuse_client()
    if inner is None:
        raise SystemExit("--prompts-dir needs the application's Langfuse client: LANGFUSE_ENABLED and both keys")
    langfuse_helper._langfuse_client = _FilePromptClient(inner, root)
    return served()
