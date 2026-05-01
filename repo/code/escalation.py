"""Escalation policy for support-ticket triage."""
from __future__ import annotations

import re

from schemas import Classification, EscalationDecision, WikiHit

HARD_ESCALATION_TAGS = {
    "account_compromise",
    "legal",
    "exam_integrity",
    "payment_failure",
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
RETRIEVAL_THRESHOLD = 0.25


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

    if hits:
        top_score = max(h.score for h in hits)
        if top_score < RETRIEVAL_THRESHOLD:
            score += 0.4
            reasons.append(f"Low retrieval confidence ({top_score:.2f})")
    else:
        score += 0.6
        reasons.append("No relevant wiki page found")

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
