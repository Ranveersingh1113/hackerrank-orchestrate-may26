"""Triage agent orchestrator. Hand-rolled state machine (no LangGraph dep needed at runtime;
keeps it portable). Each node mutates an AgentState.

Pipeline:
  sanitize -> classify -> route_domain -> retrieve -> escalate? -> respond -> finalize
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from classifier import classify, infer_domain
from escalation import decide
from grounding import ESCALATE_REPLY, generate_justification, generate_response
from retriever import WikiRetriever
from safety import sanitize
from schemas import AgentState, TicketIn, TicketOut

_RETRIEVER: Optional[WikiRetriever] = None
DOMAINS = ("HackerRank", "Claude", "Visa")


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

    # 3. Route domain
    company = ticket.company

    # 4. Retrieve from wiki
    if company == "None":
        hits = _retrieve_cross_domain(retr, cleaned)
        if hits:
            company = _company_from_domain(hits[0].domain)  # type: ignore[assignment]
        else:
            company = infer_domain(cleaned, ticket.subject)  # type: ignore[assignment]
            hits = _retrieve_domain(retr, cleaned, company) if company != "None" else []
    else:
        hits = _retrieve_domain(retr, cleaned, company)
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
    if decision.status == "escalated":
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
        justification=state.justification,
    )
    return state.final


def _fallback_area(hits: list) -> str:
    if hits and hits[0].title:
        return hits[0].title
    return "General"


def _retrieve_domain(retr: WikiRetriever, query: str, company: str):
    if company not in DOMAINS:
        return []
    bm = retr.bm25_search(query, domain=company.lower(), k=12)
    hits = retr.embed_rerank(query, bm, k=3) if bm else []
    return hits or retr.lookup_via_index(query, company.lower())


def _retrieve_cross_domain(retr: WikiRetriever, query: str):
    candidates = retr.bm25_search(query, domain=None, k=24)
    if not candidates:
        for company in DOMAINS:
            candidates.extend(retr.lookup_via_index(query, company.lower(), k=2))

    dedup = {}
    for hit in candidates:
        existing = dedup.get(hit.path)
        if existing is None or hit.score > existing.score:
            dedup[hit.path] = hit
    pooled = list(dedup.values())
    pooled.sort(key=lambda h: _domain_hint(query, h.domain) + h.score * 0.01, reverse=True)
    ranked = retr.embed_rerank(query, pooled, k=5) if pooled else []
    ranked.sort(key=lambda h: _domain_hint(query, h.domain) + h.score * 0.01, reverse=True)
    return ranked[:3]


def _company_from_domain(domain: str) -> str:
    d = domain.lower()
    if d == "hackerrank":
        return "HackerRank"
    if d == "claude":
        return "Claude"
    if d == "visa":
        return "Visa"
    return "None"


def _domain_hint(query: str, domain: str) -> float:
    text = query.lower()
    d = domain.lower()
    patterns = {
        "visa": r"\b(visa|card|merchant|atm|fraud|dispute|chargeback|transaction)\b",
        "claude": r"\b(claude|anthropic|workspace|seat|mcp|project|opus|sonnet|api)\b",
        "hackerrank": r"\b(hackerrank|test|candidate|assessment|proctor|recruiter|interview)\b",
    }
    return 2.0 if re.search(patterns.get(d, r"$^"), text) else 0.0
