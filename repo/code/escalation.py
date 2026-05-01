"""5-signal escalation policy.

Signals:
  - Retrieval confidence (max hit score below threshold -> escalate)
  - Topic risk (any high-risk tag fires)
  - PII present
  - Sentiment / threat language (legal, regulator, sue, lawsuit, refund now)
  - Urgency (P0/P1)

Hard rules: any high-risk topic OR PII -> escalate.
Otherwise weighted vote.
"""
from __future__ import annotations

import re

from schemas import AgentState, Classification, EscalationDecision, WikiHit

HIGH_RISK_TAGS = {
    "fraud",
    "lost_card",
    "dispute",
    "chargeback",
    "account_compromise",
    "legal",
    "exam_integrity",
    "payment_failure",
    "pii",
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
RETRIEVAL_THRESHOLD = 0.25  # below this, escalate (no grounded answer)


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

    # Hard rule 1: PII present
    if pii_flag or classification.is_pii:
        reasons.append("PII detected — never echo and never auto-resolve")
        return EscalationDecision(status="Escalated", score=1.0, reasons=reasons)

    # Hard rule 2: high-risk topic
    risk_hit = [t for t in classification.risk_tags if t in HIGH_RISK_TAGS]
    if risk_hit:
        reasons.append(f"High-risk tag(s): {', '.join(risk_hit)}")
        return EscalationDecision(status="Escalated", score=1.0, reasons=reasons)

    # Hard rule 3: any retrieved page tagged high risk
    high_pages = [h for h in hits if h.escalation_risk == "high"]
    if high_pages:
        reasons.append(f"Retrieved page tagged high-risk: {high_pages[0].path}")
        return EscalationDecision(status="Escalated", score=1.0, reasons=reasons)

    # Soft signals
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
        score += 0.2
        reasons.append("High urgency language")

    if injection_flag:
        # injection alone doesn't trigger escalate; we reply with refusal in invalid path
        reasons.append("Prompt injection pattern detected (handled as invalid)")

    if classification.request_type == "invalid":
        # invalid does NOT escalate — replies with polite refusal
        return EscalationDecision(status="Replied", score=score, reasons=reasons or ["Invalid/out-of-scope reply"])

    if score >= 0.5:
        return EscalationDecision(status="Escalated", score=score, reasons=reasons)

    return EscalationDecision(status="Replied", score=score, reasons=reasons or ["No escalation signals fired"])
