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
    try:
        text = chat([{"role": "user", "content": prompt}], fast=False, temperature=0.0).strip()
    except Exception:
        text = _extractive_reply(company, hits, page_contents)
    return text or ESCALATE_REPLY


def generate_justification(
    decision: str,
    reasons: list[str],
    hits: list[WikiHit],
    issue: str,
) -> str:
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
