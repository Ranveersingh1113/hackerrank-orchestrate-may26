"""Pydantic schemas for triage agent I/O and internal state."""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field

Domain = Literal["HackerRank", "Claude", "Visa", "None"]
Status = Literal["Replied", "Escalated"]
RequestType = Literal["product_issue", "feature_request", "bug", "invalid"]


class TicketIn(BaseModel):
    issue: str = Field(..., description="Ticket body")
    subject: Optional[str] = Field(default=None, description="May be blank/noisy")
    company: Domain = Field(default="None")


class TicketOut(BaseModel):
    issue: str
    subject: Optional[str]
    company: Domain
    response: str
    product_area: str
    status: Status
    request_type: RequestType


class WikiHit(BaseModel):
    domain: str
    path: str
    title: str
    score: float
    escalation_risk: Literal["low", "medium", "high"] = "low"
    source_url: Optional[str] = None


class Classification(BaseModel):
    request_type: RequestType
    product_area: str
    risk_tags: list[str] = Field(default_factory=list)
    sub_issues: list[str] = Field(default_factory=list)
    is_injection: bool = False
    is_pii: bool = False


class EscalationDecision(BaseModel):
    status: Status
    score: float
    reasons: list[str]


class AgentState(BaseModel):
    ticket: TicketIn
    sanitized_issue: str = ""
    classification: Optional[Classification] = None
    hits: list[WikiHit] = Field(default_factory=list)
    page_contents: list[str] = Field(default_factory=list)
    escalation: Optional[EscalationDecision] = None
    response: str = ""
    justification: str = ""
    final: Optional[TicketOut] = None
