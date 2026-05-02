"""Escalation policy for support-ticket triage."""
from __future__ import annotations

import re

from schemas import Classification, EscalationDecision, WikiHit

HARD_ESCALATION_TAGS = {
    "account_compromise",
    "legal",
    "exam_integrity",
    "payment_failure",
    "fraud",
}

THREAT_RE = re.compile(
    r"\b(legal\s+action|sue|lawsuit|regulator|attorney|press|news|"
    r"defamation|escalat\w+\s+to\s+(legal|management)|threaten)\b",
    re.I,
)
URGENT_RE = re.compile(
    r"\b(asap|immediately|urgent|right\s+now|service\s+down|"
    r"site\s+is\s+down|cannot\s+access|locked\s+out|production\s+down)\b",
    re.I,
)
# Threshold against post-rerank cosine similarity (0..1). BM25 raw scores are
# unbounded and not directly comparable, so we only enforce this floor when the
# scores look normalized (max <= 1.0), which is the case after embed_rerank().
RERANK_FLOOR = 0.30
# BM25-only fallback: very low absolute scores still suggest weak retrieval.
BM25_WEAK_FLOOR = 1.5


def _retrieval_signal(hits: list[WikiHit]) -> tuple[float, str | None]:
    """Return (escalation_increment, reason_text_or_None) based on hit quality."""
    if not hits:
        return 0.6, "No relevant wiki page found"
    top = max(h.score for h in hits)
    looks_normalized = all(h.score <= 1.0 for h in hits)
    if looks_normalized:
        if top < RERANK_FLOOR:
            return 0.4, f"Low rerank similarity ({top:.2f})"
    else:
        if top < BM25_WEAK_FLOOR:
            return 0.3, f"Weak BM25 match ({top:.2f})"
    return 0.0, None


def decide(
    classification: Classification,
    hits: list[WikiHit],
    raw_issue: str,
    sanitized_issue: str,
    pii_flag: bool,
    injection_flag: bool,
) -> EscalationDecision:
    reasons: list[str] = []
    score = 0.0

    if classification.request_type == "invalid":
        return EscalationDecision(
            status="replied",
            score=0.0,
            reasons=["Invalid/out-of-scope reply"],
        )

    if pii_flag:
        return EscalationDecision(
            status="escalated",
            score=1.0,
            reasons=["PII detected - never echo and never auto-resolve"],
        )

    risk_hit = [t for t in classification.risk_tags if t in HARD_ESCALATION_TAGS]
    if risk_hit:
        return EscalationDecision(
            status="escalated",
            score=1.0,
            reasons=[f"High-risk tag(s): {', '.join(risk_hit)}"],
        )

    high_pages = [h for h in hits if h.escalation_risk == "high"]
    if high_pages:
        reasons.append(f"High-risk source consulted: {high_pages[0].path}")
        score += 0.35

    delta, reason = _retrieval_signal(hits)
    score += delta
    if reason:
        reasons.append(reason)

    if THREAT_RE.search(raw_issue):
        score += 0.4
        reasons.append("Threat / legal language present")

    if URGENT_RE.search(raw_issue):
        score += 0.6 if classification.request_type == "bug" else 0.2
        reasons.append("High urgency language")

    if injection_flag:
        reasons.append("Prompt injection pattern detected")

    if score >= 0.5:
        return EscalationDecision(status="escalated", score=score, reasons=reasons)

    return EscalationDecision(
        status="replied",
        score=score,
        reasons=reasons or ["No escalation signals fired"],
    )
