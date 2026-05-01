"""PII redaction + prompt injection detection. Pre-LLM filter."""
from __future__ import annotations

import re

# Card numbers (13-19 digits, optional separators)
CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
# OTP / 6-digit codes near keywords
OTP_RE = re.compile(r"\b(otp|one[- ]?time[- ]?password|verification code)\b[:\s]*\d{4,8}", re.I)
# Indian PAN
PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
# Aadhaar 12-digit
AADHAAR_RE = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
# US SSN
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# Email (kept; only mask if asked)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?prior",
    r"system\s*[:>].*you\s+are",
    r"reveal\s+your\s+prompt",
    r"</?\s*system\s*>",
    r"jailbreak",
    r"DAN\s+mode",
]
INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.I)


def has_pii(text: str) -> bool:
    return bool(
        CARD_RE.search(text)
        or OTP_RE.search(text)
        or PAN_RE.search(text)
        or AADHAAR_RE.search(text)
        or SSN_RE.search(text)
    )


def redact_pii(text: str) -> str:
    text = CARD_RE.sub("[REDACTED_CARD]", text)
    text = OTP_RE.sub("[REDACTED_OTP]", text)
    text = PAN_RE.sub("[REDACTED_PAN]", text)
    text = AADHAAR_RE.sub("[REDACTED_AADHAAR]", text)
    text = SSN_RE.sub("[REDACTED_SSN]", text)
    return text


def is_injection(text: str) -> bool:
    return bool(INJECTION_RE.search(text))


def sanitize(text: str) -> tuple[str, dict]:
    flags = {
        "pii": has_pii(text),
        "injection": is_injection(text),
    }
    cleaned = redact_pii(text)
    return cleaned, flags
