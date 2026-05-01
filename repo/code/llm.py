"""Ollama wrapper. Local-only. Seeded for determinism."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Load .env from repo root (one dir up from code/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

# Force a valid client URL. Ollama's daemon-side OLLAMA_HOST is sometimes
# set to "0.0.0.0" (bind-all), which is not a valid client target. We always
# point the client at the loopback interface unless the caller explicitly
# overrides via a URL with an http(s) scheme.
_host_env = os.environ.get("OLLAMA_HOST", "").strip()
if not _host_env.startswith(("http://", "https://")):
    os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

from ollama import Client  # noqa: E402

REASON_MODEL = os.getenv("OLLAMA_REASON", "qwen2.5:7b-instruct-q4_K_M")
FAST_MODEL = os.getenv("OLLAMA_FAST", "qwen2.5:3b-instruct-q4_K_M")
EMBED_MODEL = os.getenv("OLLAMA_EMBED", "nomic-embed-text")
SEED = 42

_client = Client(host=os.environ["OLLAMA_HOST"], timeout=600)


def chat(
    messages: list[dict],
    *,
    model: Optional[str] = None,
    fast: bool = False,
    json_mode: bool = False,
    temperature: float = 0.0,
) -> str:
    """Single-turn chat call. Returns text content."""
    m = model or (FAST_MODEL if fast else REASON_MODEL)
    options: dict[str, Any] = {"temperature": temperature, "seed": SEED, "num_ctx": 8192}
    kwargs: dict[str, Any] = {"model": m, "messages": messages, "options": options}
    if json_mode:
        kwargs["format"] = "json"
    resp = _client.chat(**kwargs)
    return resp["message"]["content"]


def chat_json(messages: list[dict], *, fast: bool = False) -> dict:
    raw = chat(messages, fast=fast, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # one retry with stricter instruction
        retry = messages + [
            {"role": "user", "content": "Output STRICT JSON only. No prose."}
        ]
        return json.loads(chat(retry, fast=fast, json_mode=True))


def embed(text: str | list[str]) -> list[list[float]]:
    """Return embedding vector(s)."""
    if isinstance(text, str):
        text = [text]
    out: list[list[float]] = []
    for t in text:
        r = _client.embeddings(model=EMBED_MODEL, prompt=t)
        out.append(r["embedding"])
    return out
