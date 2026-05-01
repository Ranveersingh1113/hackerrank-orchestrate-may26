"""Triage agent orchestrator. Hand-rolled state machine (no LangGraph dep needed at runtime;
keeps it portable). Each node mutates an AgentState.

Pipeline:
  sanitize -> classify -> route_domain -> retrieve -> escalate? -> respond -> finalize
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from classifier import classify, infer_domain
from escalation import decide
from grounding import ESCALATE_REPLY, generate_justification, generate_response
from retriever import WikiRetriever
from safety import sanitize
from schemas import AgentState, TicketIn, TicketOut

_RETRIEVER: Optional[WikiRetriever] = None


def get_retriever(wiki_root: Path = Path("wiki")) -> WikiRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = WikiRetriever(wiki_root)
    return _RETRIEVER


def process_ticket(ticket: TicketIn, wiki_root: Path = Path("wiki")) -> TicketOut:
    state = AgentState(ticket=ticket)
    retr = get_retriever(wiki_root)

    # 1. Sanitize input (PII, injection)
    cleaned, flags = sanitize(ticket.issue)
    state.sanitized_issue = cleaned

    # 2. Classify
    state.classification = classify(cleaned, ticket.subject, ticket.company)

    # 3. Route domain (if company=None, infer)
    company = ticket.company
    if company == "None":
        company = infer_domain(cleaned, ticket.subject)  # type: ignore[assignment]

    # 4. Retrieve from wiki
    hits = []
    if company in ("HackerRank", "Claude", "Visa"):
        hits = retr.lookup_via_index(cleaned, company.lower())
    if not hits:
        bm = retr.bm25_search(cleaned, domain=(company.lower() if company != "None" else None), k=8)
        hits = retr.embed_rerank(cleaned, bm, k=3) if bm else []
    state.hits = hits
    state.page_contents = [retr.page_content(h) for h in hits]

    # 5. Escalation decision
    decision = decide(
        classification=state.classification,
        hits=hits,
        raw_issue=ticket.issue,
        sanitized_issue=cleaned,
        pii_flag=flags["pii"],
        injection_flag=flags["injection"],
    )
    state.escalation = decision

    # 6. Generate response
    if decision.status == "Escalated":
        state.response = ESCALATE_REPLY
    elif state.classification.request_type == "invalid":
        state.response = (
            "This appears to be outside the scope of our supported product areas. "
            "Please rephrase with a specific issue or question, or contact us through "
            "an official channel for further help."
        )
    else:
        state.response = generate_response(
            cleaned, ticket.subject, company, hits, state.page_contents
        )

    # 7. Justification
    state.justification = generate_justification(
        decision.status, decision.reasons, hits, ticket.issue
    )

    state.final = TicketOut(
        issue=ticket.issue,
        subject=ticket.subject,
        company=company,  # type: ignore[arg-type]
        response=state.response,
        product_area=state.classification.product_area or _fallback_area(hits),
        status=decision.status,
        request_type=state.classification.request_type,
    )
    return state.final


def _fallback_area(hits: list) -> str:
    if hits and hits[0].title:
        return hits[0].title
    return "General"
