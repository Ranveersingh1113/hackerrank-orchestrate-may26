"""Generate grounded response from retrieved wiki pages.

Hard constraint: every claim in the response must come from the page contents.
Tail justification cites the wiki page paths used.
"""
from __future__ import annotations

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

ESCALATE_REPLY = "Escalate to a human"

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
) -> str:
    if not hits:
        return ESCALATE_REPLY
    excerpts = render_excerpts(hits, page_contents)
    prompt = REPLY_PROMPT.format(
        excerpts=excerpts,
        subject=subject or "(blank)",
        company=company,
        issue=issue[:2000],
    )
    text = chat([{"role": "user", "content": prompt}], fast=False, temperature=0.0).strip()
    return text or ESCALATE_REPLY


def generate_justification(
    decision: str,
    reasons: list[str],
    hits: list[WikiHit],
    issue: str,
) -> str:
    paths = [h.path for h in hits] or ["(none)"]
    prompt = JUSTIFY_PROMPT.format(
        decision=decision.lower(),
        reasons="; ".join(reasons) or "n/a",
        paths=", ".join(paths),
        ticket=issue[:600],
    )
    return chat([{"role": "user", "content": prompt}], fast=True, temperature=0.0).strip()
