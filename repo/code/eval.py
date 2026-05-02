"""Quick accuracy harness against support_tickets/sample_support_tickets.csv.

Run from repo root:
    python code/eval.py

Reports per-row hit rates on the two enum-constrained columns we can score
deterministically (`status`, `request_type`) plus a confusion matrix and a
list of disagreements for manual review. `product_area` and `response` are
free-text and are not auto-scored — eyeball them in the printed dump.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Allow `python code/eval.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import process_ticket  # noqa: E402
from schemas import TicketIn  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = REPO_ROOT / "support_tickets" / "sample_support_tickets.csv"


def _norm(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "_")


def run_eval(csv_path: Path = SAMPLE_CSV, limit: int | None = None) -> None:
    if not csv_path.exists():
        print(f"[eval] sample CSV not found: {csv_path}")
        return

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if limit:
        rows = rows[:limit]

    correct = Counter()
    total = len(rows)
    confusion: dict[str, Counter] = defaultdict(Counter)  # column -> Counter[(expected, got)]
    disagreements: list[dict] = []

    for i, row in enumerate(rows, 1):
        ticket = TicketIn(
            issue=(row.get("Issue") or "").strip(),
            subject=(row.get("Subject") or "").strip() or None,
            company=_normalize_company(row.get("Company") or "None"),  # type: ignore[arg-type]
        )
        try:
            out = process_ticket(ticket)
        except Exception as e:  # noqa: BLE001
            print(f"[eval] row {i} FAILED: {type(e).__name__}: {e}")
            continue

        expected_status = _norm(row.get("Status", ""))
        expected_rt = _norm(row.get("Request Type", ""))
        got_status = _norm(out.status)
        got_rt = _norm(out.request_type)

        if got_status == expected_status:
            correct["status"] += 1
        else:
            confusion["status"][(expected_status, got_status)] += 1

        if got_rt == expected_rt:
            correct["request_type"] += 1
        else:
            confusion["request_type"][(expected_rt, got_rt)] += 1

        if got_status != expected_status or got_rt != expected_rt:
            disagreements.append(
                {
                    "row": i,
                    "issue": ticket.issue[:120],
                    "expected_status": expected_status,
                    "got_status": got_status,
                    "expected_rt": expected_rt,
                    "got_rt": got_rt,
                    "expected_area": row.get("Product Area", ""),
                    "got_area": out.product_area,
                }
            )
        print(f"[{i:3d}/{total}] status={got_status:9} rt={got_rt:14} expected status={expected_status:9} rt={expected_rt}")

    print()
    print("=" * 60)
    print(f"status         accuracy: {correct['status']}/{total} = {correct['status']/total:.0%}")
    print(f"request_type   accuracy: {correct['request_type']}/{total} = {correct['request_type']/total:.0%}")
    print("=" * 60)

    for col, c in confusion.items():
        if not c:
            continue
        print(f"\n{col} confusion (expected -> got: count):")
        for (exp, got), n in c.most_common():
            print(f"  {exp:15} -> {got:15}  {n}")

    if disagreements:
        print(f"\n{len(disagreements)} disagreements (showing first 10):")
        for d in disagreements[:10]:
            print(
                f"  row {d['row']:3d} | exp st={d['expected_status']:9} rt={d['expected_rt']:14} "
                f"got st={d['got_status']:9} rt={d['got_rt']:14}"
            )
            print(f"           area exp={d['expected_area']!r} got={d['got_area']!r}")
            print(f"           issue: {d['issue']}")


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
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_eval(limit=lim)
