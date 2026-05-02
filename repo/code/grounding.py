"""Generate grounded response from retrieved wiki pages.

Hard constraint: every claim in the response must come from the page contents.
Tail justification cites the wiki page paths used.
"""
from __future__ import annotations

import re

from llm import chat
from schemas import WikiHit

REPLY_PROMPT = """You are a support agent. Use ONLY the WIKI EXCERPTS to answer the TICKET.
If the excerpts do not directly support an answer, reply: "Escalated to a human."

Rules:
- Be concise, polite, and direct. No filler.
- Do NOT invent steps, policies, phone numbers, prices, or timelines.
- Quote specific values (numbers, URLs) only if present in excerpts.
- Do not echo PII or guess at user identity.
- Address the user's actual question, not the subject line.
- 4-12 sentences. Plain text. No markdown headers.

WIKI EXCERPTS (each starts with [path]):
{excerpts}

TICKET:
SUBJECT: {subject}
COMPANY: {company}
ISSUE: {issue}

Reply:"""

ESCALATE_REPLY = "Escalated to a human."

JUSTIFY_PROMPT = """In 1-2 sentences, explain why the agent {decision} this ticket.
Cite specific signals: risk tags, retrieved wiki paths, retrieval confidence, or PII detection.
No fluff. Plain text only.

DECISION: {decision}
REASONS: {reasons}
RETRIEVED: {paths}
TICKET (truncated): {ticket}"""


def render_excerpts(hits: list[WikiHit], page_contents: list[str], max_chars: int = 1800) -> str:
    out = []
    for h, body in zip(hits, page_contents):
        snippet = body[:max_chars].strip()
        out.append(f"[{h.path}]\n{snippet}\n")
    return "\n---\n".join(out) if out else "(none)"


def generate_response(
    issue: str,
    subject: str | None,
    company: str,
    hits: list[WikiHit],
    page_contents: list[str],
) -> tuple[str, bool]:
    """Return (response_text, grounded_flag).

    grounded_flag=False signals the caller that the response could not be
    grounded in the retrieved excerpts and the ticket should be escalated.
    """
    if not hits:
        return ESCALATE_REPLY, False
    excerpts = render_excerpts(hits, page_contents)
    prompt = REPLY_PROMPT.format(
        excerpts=excerpts,
        subject=subject or "(blank)",
        company=company,
        issue=issue[:2000],
    )
    try:
        text = chat([{"role": "user", "content": prompt}], fast=False, temperature=0.0).strip()
    except Exception:
        text = _extractive_reply(company, hits, page_contents)
    if not text:
        return ESCALATE_REPLY, False
    if _is_escalate_signal(text):
        return ESCALATE_REPLY, False
    if not _is_grounded(text, page_contents):
        # Model fell back to parametric knowledge. Try the extractive path.
        fallback = _extractive_reply(company, hits, page_contents)
        if fallback != ESCALATE_REPLY and _is_grounded(fallback, page_contents):
            return fallback, True
        return ESCALATE_REPLY, False
    return text, True


_ESCALATE_RE = re.compile(r"\bescalat(e|ed|ing)\s+to\s+a?\s*human\b", re.I)


def _is_escalate_signal(text: str) -> bool:
    """The reply prompt instructs the model to emit this exact phrase when the
    excerpts do not support an answer. We treat any 'escalate to a human' phrasing
    in a short reply as an escalation marker.
    """
    if _ESCALATE_RE.search(text) and len(text) < 120:
        return True
    return text.strip() == ESCALATE_REPLY


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{3,}")


def _is_grounded(reply: str, page_contents: list[str], min_overlap: float = 0.35) -> bool:
    """Cheap groundedness heuristic: at least `min_overlap` of the content
    tokens in the reply must also appear somewhere in the retrieved excerpts.

    This is a guardrail against the model inventing steps when retrieval was
    weak; it is not a substitute for the no-fabrication prompt rule.
    """
    if not page_contents:
        return False
    reply_tokens = {t.lower() for t in _TOKEN_RE.findall(reply)}
    if not reply_tokens:
        return True
    corpus = " ".join(page_contents).lower()
    corpus_tokens = set(_TOKEN_RE.findall(corpus))
    if not corpus_tokens:
        return False
    common = reply_tokens & corpus_tokens
    return (len(common) / max(len(reply_tokens), 1)) >= min_overlap


def generate_justification(
    decision: str,
    reasons: list[str],
    hits: list[WikiHit],
    issue: str,
) -> str:
    paths = [h.path for h in hits[:3]] or ["(none)"]
    prompt = JUSTIFY_PROMPT.format(
        decision=decision,
        reasons="; ".join(reasons) or "n/a",
        paths=", ".join(paths),
        ticket=issue[:600],
    )
    try:
        text = chat([{"role": "user", "content": prompt}], fast=True, temperature=0.0).strip()
        return _clean_justification(text) or _fallback_justification(decision, reasons, hits)
    except Exception:
        return _fallback_justification(decision, reasons, hits)


def _extractive_reply(company: str, hits: list[WikiHit], page_contents: list[str]) -> str:
    if not hits or not page_contents:
        return ESCALATE_REPLY
    lines = []
    for line in page_contents[0].splitlines():
        clean = line.strip(" -*\t")
        if 40 <= len(clean) <= 220 and not clean.lower().startswith(("#", "source:")):
            lines.append(clean)
        if len(lines) == 3:
            break
    if not lines:
        return ESCALATE_REPLY
    return (
        f"Based on the {company} support article '{hits[0].title}', "
        + " ".join(lines)
    )


def _fallback_justification(decision: str, reasons: list[str], hits: list[WikiHit]) -> str:
    paths = ", ".join(h.path for h in hits[:3]) or "no retrieved page"
    reason = "; ".join(reasons) or "no escalation signals fired"
    return f"Decision {decision} because {reason}. Retrieved: {paths}."


def _clean_justification(text: str) -> str:
    cleaned = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if not cleaned:
        return ""
    label_markers = ("DECISION:", "REASONS:", "RETRIEVED:", "TICKET")
    if any(marker in cleaned.upper() for marker in label_markers):
        return ""
    return cleaned[:400]
