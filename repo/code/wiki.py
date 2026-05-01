"""LLM Wiki ingest. raw markdown corpus -> wiki/{domain}/<slug>.md + index.md.

Each wiki page has frontmatter:
  ---
  title: ...
  source_path: data/visa/support/consumer/lost-stolen-card.md
  source_url: https://...
  domain: visa
  escalation_risk: low|medium|high
  topic_tags: [fraud, lost_card, ...]
  ---
  ## Summary
  ...
  ## Key Claims
  - <claim> (<source_path>#section)
"""
from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from llm import chat_json

DOMAINS = ("hackerrank", "claude", "visa")

# Heuristic risk tagger — runs first; LLM confirms/upgrades only.
HIGH_RISK_PATTERNS = re.compile(
    r"\b(fraud|scam|phishing|stolen|lost\s+card|chargeback|dispute|"
    r"refund|unauthori[sz]ed|identity\s+theft|account\s+(compromis|hack|takeover)|"
    r"billing|payment\s+failure|legal|regulator|exam\s+integrity|"
    r"proctor|password\s+reset|2fa|verify\s+identity|sso|seat\s+remov)",
    re.I,
)
MEDIUM_RISK_PATTERNS = re.compile(
    r"\b(invoice|subscription|cancel|delete\s+account|data\s+export|"
    r"permission|admin|owner|workspace|usage\s+limit|rate\s+limit|"
    r"assessment|test|candidate|recruiter|integration|api\s+key)",
    re.I,
)


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower())
    text = re.sub(r"[-\s]+", "-", text).strip("-")
    return text[:80] or "page"


def parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    fm_block = text[3:end].strip()
    body = text[end + 4 :].lstrip("\n")
    fm: dict = {}
    for line in fm_block.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip().strip('"').strip("'")
    return fm, body


def _as_text(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(x) for x in value)
    return str(value or "")


def _as_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if not value:
        return []
    return [s.strip().strip('"') for s in str(value).strip("[]").split(",") if s.strip()]


def _normalize_page(page: dict) -> dict:
    page = dict(page)
    page["title"] = _as_text(page.get("title")) or "Untitled"
    page["product_area"] = _as_text(page.get("product_area")) or "General"
    risk = _as_text(page.get("escalation_risk")).lower()
    page["escalation_risk"] = risk if risk in ("low", "medium", "high") else "low"
    page["topic_tags"] = _as_list(page.get("topic_tags"))
    return page


def heuristic_risk(text: str, source_path: str) -> str:
    blob = f"{source_path}\n{text[:4000]}"
    if HIGH_RISK_PATTERNS.search(blob):
        return "high"
    if MEDIUM_RISK_PATTERNS.search(blob):
        return "medium"
    return "low"


def file_hash(p: Path) -> str:
    h = hashlib.sha1()
    h.update(p.read_bytes())
    return h.hexdigest()[:12]


def iter_corpus(raw_root: Path) -> Iterable[tuple[str, Path]]:
    for domain in DOMAINS:
        droot = raw_root / domain
        if not droot.exists():
            continue
        for p in droot.rglob("*.md"):
            # skip the domain index itself
            if p.name == "index.md" and p.parent == droot:
                continue
            yield domain, p


def build_wiki_page(domain: str, src_path: Path, raw_root: Path) -> dict:
    text = src_path.read_text(encoding="utf-8", errors="ignore")
    fm, body = parse_frontmatter(text)
    title = fm.get("title") or src_path.stem.replace("-", " ").title()
    source_url = fm.get("source_url") or fm.get("final_url") or ""
    rel_src = src_path.relative_to(raw_root.parent).as_posix()

    # Truncate body for LLM (cheap article = full; long article = first 6k chars)
    excerpt = body[:6000]

    prompt = f"""You are an extraction assistant building a structured help-center wiki.

DOMAIN: {domain}
ARTICLE TITLE: {title}
SOURCE PATH: {rel_src}

ARTICLE CONTENT:
\"\"\"
{excerpt}
\"\"\"

Extract a structured summary as STRICT JSON with these keys:
- "summary": 2-4 sentence neutral summary of what this article covers (no fluff).
- "topic_tags": 3-7 lowercase snake_case tags (e.g. ["lost_card","fraud","emergency_replacement"]).
- "escalation_risk": one of "low" | "medium" | "high".
   high = fraud, account compromise, lost/stolen card, dispute, billing problems, legal, exam integrity.
   medium = subscriptions, permissions, integrations, account settings, API keys.
   low = how-tos, general FAQs, informational.
- "key_claims": list of 3-7 short factual claims grounded in the article. Each claim is one sentence,
   and must be directly supported by the article content. No outside knowledge.
- "product_area": short canonical category (e.g. "Lost/Stolen Card", "Disputes", "Billing",
   "Usage Limits", "Claude Code", "Assessments", "Proctoring", "Integrations").

Output JSON only. No prose. No markdown fences.
"""
    data = chat_json([{"role": "user", "content": prompt}], fast=True)

    # Defensive fill
    risk = data.get("escalation_risk", "").lower()
    if risk not in ("low", "medium", "high"):
        risk = heuristic_risk(text, rel_src)
    # If heuristic says high but LLM said low, upgrade.
    h = heuristic_risk(text, rel_src)
    if h == "high" and risk != "high":
        risk = "high"

    return {
        "title": title,
        "source_path": rel_src,
        "source_url": source_url,
        "domain": domain,
        "escalation_risk": risk,
        "topic_tags": data.get("topic_tags", []) or [],
        "product_area": data.get("product_area", "") or "",
        "summary": data.get("summary", "") or "",
        "key_claims": data.get("key_claims", []) or [],
    }


def render_wiki_page(page: dict) -> str:
    page = _normalize_page(page)
    fm_lines = [
        "---",
        f'title: "{page["title"]}"',
        f'source_path: "{page["source_path"]}"',
        f'source_url: "{page["source_url"]}"',
        f'domain: "{page["domain"]}"',
        f'escalation_risk: "{page["escalation_risk"]}"',
        f'product_area: "{page["product_area"]}"',
        f'topic_tags: {json.dumps(page["topic_tags"])}',
        "---",
        "",
    ]
    body = [
        f"# {page['title']}",
        "",
        "## Summary",
        page["summary"],
        "",
        "## Key Claims",
    ]
    for c in page["key_claims"]:
        body.append(f"- {c}")
    body += ["", f"_Source: [{page['source_path']}]({page['source_url']})_", ""]
    return "\n".join(fm_lines + body)


def render_index(domain: str, pages: list[dict]) -> str:
    pages = [_normalize_page(p) for p in pages]
    lines = [
        f"# {domain.capitalize()} Support — Wiki Index",
        "",
        f"_{len(pages)} articles ingested. Use this index to find relevant pages by topic, area, or risk._",
        "",
        "| Page | Product Area | Risk | Tags |",
        "|------|--------------|------|------|",
    ]
    for p in sorted(pages, key=lambda x: (x["escalation_risk"] != "high", x["product_area"], x["title"])):
        slug = p.get("_slug") or slugify(p["title"])
        risk = p["escalation_risk"]
        risk_badge = {"high": "🔴 high", "medium": "🟡 med", "low": "🟢 low"}[risk]
        tags = ", ".join(p["topic_tags"][:5])
        lines.append(f"| [{p['title']}]({slug}.md) | {p['product_area']} | {risk_badge} | {tags} |")
    lines.append("")
    return "\n".join(lines)


def _cache_file(out_dir: Path, cached: dict) -> Path:
    return out_dir / f"{cached.get('slug', 'x')}.md"


def _build_pending(domain: str, src_path: Path, raw_root: Path) -> tuple[str, str, str, dict]:
    h = file_hash(src_path)
    rel = src_path.relative_to(raw_root.parent).as_posix()
    page = build_wiki_page(domain, src_path, raw_root)
    return domain, rel, h, page


def _write_page(out_root: Path, cache: dict, domain: str, rel: str, h: str, page: dict) -> dict:
    out_dir = out_root / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    page = _normalize_page(page)
    slug = cache.get(rel, {}).get("slug") or slugify(page["title"])
    candidate = slug
    i = 1
    while (out_dir / f"{candidate}.md").exists() and cache.get(rel, {}).get("slug") != candidate:
        i += 1
        candidate = f"{slug}-{i}"
    slug = candidate

    page_with_slug = dict(page)
    page_with_slug["_slug"] = slug
    (out_dir / f"{slug}.md").write_text(render_wiki_page(page), encoding="utf-8")
    cache[rel] = {"hash": h, "slug": slug, "page": page}
    return page_with_slug


def build_wiki(raw_root: Path, out_root: Path, force: bool = False, workers: int = 4) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    cache_path = out_root / ".cache.json"
    cache: dict = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    pages_by_domain: dict[str, list[dict]] = {d: [] for d in DOMAINS}
    items = sorted(
        iter_corpus(raw_root),
        key=lambda item: item[1].relative_to(raw_root.parent).as_posix(),
    )
    print(f"[wiki] {len(items)} corpus articles found")

    pending: list[tuple[str, Path]] = []
    for domain, src_path in tqdm(items, desc="Checking cache"):
        h = file_hash(src_path)
        rel = src_path.relative_to(raw_root.parent).as_posix()
        cached = cache.get(rel)
        out_dir = out_root / domain
        out_dir.mkdir(parents=True, exist_ok=True)

        if cached and cached.get("hash") == h and not force and _cache_file(out_dir, cached).exists():
            page = _normalize_page(cached["page"])
            page["_slug"] = cached.get("slug") or slugify(page["title"])
            pages_by_domain[domain].append(page)
            continue

        pending.append((domain, src_path))

    print(f"[wiki] {len(pending)} articles need extraction; workers={workers}")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_build_pending, domain, src_path, raw_root): (
                domain,
                src_path.relative_to(raw_root.parent).as_posix(),
            )
            for domain, src_path in pending
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Ingesting"):
            domain, rel = futures[future]
            try:
                domain, rel, h, page = future.result()
                page_with_slug = _write_page(out_root, cache, domain, rel, h, page)
                pages_by_domain[domain].append(page_with_slug)
                cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
            except Exception as e:
                import traceback as _tb

                print(f"[wiki] ERROR {rel}: {type(e).__name__}: {e}")
                _tb.print_exc()

    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    for domain, pages in pages_by_domain.items():
        if not pages:
            continue
        (out_root / domain / "index.md").write_text(render_index(domain, pages), encoding="utf-8")

    total = sum(len(v) for v in pages_by_domain.values())
    print(f"[wiki] done. {total} pages across {sum(1 for v in pages_by_domain.values() if v)} domains")


if __name__ == "__main__":
    import sys

    raw = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "wiki")
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    build_wiki(raw, out, workers=workers)
