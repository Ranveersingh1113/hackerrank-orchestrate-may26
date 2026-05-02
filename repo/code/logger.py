"""AGENTS.md §5.2 turn logger. Append-only to %USERPROFILE%/hackerrank_orchestrate/log.txt."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


def log_path() -> Path:
    home = Path(os.environ.get("USERPROFILE") or Path.home())
    p = home / "hackerrank_orchestrate" / "log.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def append(title: str, *, user_prompt: str, summary: str, actions: list[str], context: dict) -> None:
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    block = [
        f"## [{ts}] {title[:80]}",
        "",
        "User Prompt (verbatim, secrets redacted):",
        user_prompt,
        "",
        "Agent Response Summary:",
        summary,
        "",
        "Actions:",
        *[f"* {a}" for a in actions],
        "",
        "Context:",
        *[f"{k}={v}" for k, v in context.items()],
        "",
    ]
    with log_path().open("a", encoding="utf-8") as f:
        f.write("\n".join(block) + "\n")
