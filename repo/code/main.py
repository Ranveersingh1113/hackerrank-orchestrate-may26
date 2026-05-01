"""Entry point. Typer CLI for triage agent."""
from __future__ import annotations

import csv
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

from schemas import TicketIn, TicketOut

app = typer.Typer(add_completion=False, help="HackerRank Orchestrate triage agent")
console = Console()


@app.command()
def ingest(
    raw: Path = typer.Option(Path("data"), help="Corpus root"),
    out: Path = typer.Option(Path("wiki"), help="Output wiki dir"),
    workers: int = typer.Option(4, "--workers", min=1, help="Concurrent article extraction workers"),
):
    """Build LLM Wiki from corpus."""
    from wiki import build_wiki

    build_wiki(raw, out, workers=workers)


@app.command()
def run(
    csv_in: Path = typer.Option(Path("support_tickets/support_tickets.csv"), "--csv"),
    csv_out: Path = typer.Option(Path("support_tickets/output.csv"), "--out"),
):
    """Batch-process tickets CSV."""
    from agent import process_ticket

    rows = list(_read_tickets(csv_in))
    out_rows: list[TicketOut] = []
    for row in track(rows, description="Triaging tickets"):
        result = process_ticket(row)
        out_rows.append(result)
    _write_outputs(csv_out, out_rows)
    _summary(out_rows)


@app.command()
def ticket(
    issue: str = typer.Argument(..., help="Ticket body"),
    subject: str = typer.Option("", "--subject"),
    company: str = typer.Option("None", "--company"),
):
    """Single ticket interactive."""
    from agent import process_ticket

    t = TicketIn(issue=issue, subject=subject or None, company=company)  # type: ignore[arg-type]
    out = process_ticket(t)
    console.print_json(data=out.model_dump())


def _read_tickets(p: Path):
    with p.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield TicketIn(
                issue=(row.get("Issue") or row.get("issue") or "").strip(),
                subject=(row.get("Subject") or row.get("subject") or "").strip() or None,
                company=_normalize_company(row.get("Company") or row.get("company") or "None"),  # type: ignore[arg-type]
            )


def _write_outputs(p: Path, rows: list[TicketOut]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "issue",
        "subject",
        "company",
        "response",
        "product_area",
        "status",
        "request_type",
        "justification",
    ]
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(fields)
        for r in rows:
            w.writerow([
                r.issue,
                r.subject or "",
                r.company,
                r.response,
                r.product_area,
                r.status,
                r.request_type,
                r.justification,
            ])


def _summary(rows: list[TicketOut]) -> None:
    t = Table(title="Triage Summary")
    t.add_column("Metric"); t.add_column("Count", justify="right")
    n = len(rows)
    replied = sum(1 for r in rows if r.status == "replied")
    escalated = n - replied
    t.add_row("Total", str(n))
    t.add_row("Replied", f"[green]{replied}[/green]")
    t.add_row("Escalated", f"[yellow]{escalated}[/yellow]")
    for rt in ("product_issue", "feature_request", "bug", "invalid"):
        t.add_row(rt, str(sum(1 for r in rows if r.request_type == rt)))
    console.print(t)


def _normalize_company(value: str) -> str:
    cleaned = value.strip().lower()
    if cleaned == "hackerrank":
        return "HackerRank"
    if cleaned == "claude":
        return "Claude"
    if cleaned == "visa":
        return "Visa"
    return "None"


if __name__ == "__main__":
    app()
