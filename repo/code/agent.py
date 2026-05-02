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
        # Two-pass: try LLM-based domain inference first (cheap, accurate when
        # ticket has clear signals), then cross-domain BM25+rerank as a backstop.
        inferred = infer_domain(cleaned, ticket.subject)
        if inferred in DOMAINS:
            company = inferred  # type: ignore[assignment]
            hits = _retrieve_domain(retr, cleaned, company)
            if not hits:
                # Inference was confident but corpus didn't back it; fall back.
                hits = _retrieve_cross_domain(retr, cleaned)
                if hits:
                    company = _company_from_domain(hits[0].domain)  # type: ignore[assignment]
        else:
            hits = _retrieve_cross_domain(retr, cleaned)
            if hits:
                company = _company_from_domain(hits[0].domain)  # type: ignore[assignment]
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
        text, grounded = generate_response(
            cleaned, ticket.subject, company, hits, state.page_contents
        )
        if not grounded:
            # Generation could not stay grounded in the retrieved excerpts.
            # Flip to escalate rather than ship a hallucinated reply.
            decision = decision.model_copy(
                update={
                    "status": "escalated",
                    "reasons": decision.reasons + ["Response failed groundedness check"],
                }
            )
            state.escalation = decision
            state.response = ESCALATE_REPLY
        else:
            state.response = text

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

    dedup: dict[str, object] = {}
    for hit in candidates:
        existing = dedup.get(hit.path)
        if existing is None or hit.score > existing.score:  # type: ignore[union-attr]
            dedup[hit.path] = hit
    pooled = list(dedup.values())
    # Pre-rerank pool sort: domain hint as a small nudge (0.15 max) on top of
    # whatever score we have. This mainly tie-breaks BM25 hits where the same
    # query maps to multiple domains (e.g. "billing").
    pooled.sort(key=_pool_key(query), reverse=True)  # type: ignore[arg-type]
    ranked = retr.embed_rerank(query, pooled, k=5) if pooled else []  # type: ignore[arg-type]
    # Post-rerank: scores are cosine sims (0..1). Apply a small domain-hint
    # nudge (0.10 max) so we don't override a clearly stronger semantic match.
    ranked.sort(key=lambda h: h.score + _domain_hint(query, h.domain, weight=0.10), reverse=True)
    return ranked[:3]


def _pool_key(query: str):
    def key(h):
        # BM25 raw score can be 0..30+; normalize roughly by /10 then add hint.
        return min(h.score, 10.0) / 10.0 + _domain_hint(query, h.domain, weight=0.15)
    return key


def _company_from_domain(domain: str) -> str:
    d = domain.lower()
    if d == "hackerrank":
        return "HackerRank"
    if d == "claude":
        return "Claude"
    if d == "visa":
        return "Visa"
    return "None"


def _domain_hint(query: str, domain: str, weight: float = 0.10) -> float:
    text = query.lower()
    d = domain.lower()
    patterns = {
        "visa": r"\b(visa|card|merchant|atm|fraud|dispute|chargeback|transaction)\b",
        "claude": r"\b(claude|anthropic|workspace|seat|mcp|project|opus|sonnet|api)\b",
        "hackerrank": r"\b(hackerrank|test|candidate|assessment|proctor|recruiter|interview)\b",
    }
    return weight if re.search(patterns.get(d, r"$^"), text) else 0.0
