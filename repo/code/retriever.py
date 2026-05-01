"""Retrieval over the LLM Wiki.

Two layers:
  1) Index lookup (fast first pass): read wiki/{domain}/index.md, ask LLM to
     pick 1-3 candidate page slugs. No embeddings needed.
  2) BM25 + embedding rerank fallback for cross-domain (company=None) or
     when index lookup returns nothing.

Wiki page frontmatter is parsed for risk/tags/product_area metadata.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from llm import chat_json, embed
from schemas import WikiHit
from wiki import parse_frontmatter

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")
_WIKI_ROOT_DEFAULT = Path("wiki")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class WikiRetriever:
    def __init__(self, wiki_root: Path = _WIKI_ROOT_DEFAULT):
        self.root = wiki_root
        self.pages: list[dict] = []  # frontmatter + body + path
        self.bm25: Optional[BM25Okapi] = None
        self.embeddings: Optional[np.ndarray] = None
        self._load()

    def _load(self) -> None:
        if not self.root.exists():
            return
        for domain_dir in sorted(self.root.iterdir()):
            if not domain_dir.is_dir():
                continue
            for md in domain_dir.glob("*.md"):
                if md.name == "index.md":
                    continue
                text = md.read_text(encoding="utf-8", errors="ignore")
                fm, body = parse_frontmatter(text)
                self.pages.append(
                    {
                        "path": md.relative_to(self.root).as_posix(),
                        "domain": fm.get("domain", domain_dir.name),
                        "title": fm.get("title", md.stem),
                        "source_path": fm.get("source_path", ""),
                        "source_url": fm.get("source_url", ""),
                        "escalation_risk": fm.get("escalation_risk", "low"),
                        "product_area": fm.get("product_area", ""),
                        "topic_tags": _parse_list(fm.get("topic_tags", "[]")),
                        "body": body,
                        "abs_path": md,
                    }
                )
        if self.pages:
            corpus = [_tokenize(p["title"] + " " + p["body"]) for p in self.pages]
            self.bm25 = BM25Okapi(corpus)

    def index_text(self, domain: str) -> str:
        idx = self.root / domain / "index.md"
        return idx.read_text(encoding="utf-8") if idx.exists() else ""

    def lookup_via_index(self, query: str, domain: str, k: int = 3) -> list[WikiHit]:
        index_md = self.index_text(domain)
        if not index_md:
            return []
        prompt = (
            f"You are a help-desk routing agent. Given the user TICKET and the WIKI INDEX, "
            f"return the {k} most relevant page slugs.\n\n"
            f"TICKET:\n{query[:1500]}\n\n"
            f"WIKI INDEX ({domain}):\n{index_md[:6000]}\n\n"
            f'Output STRICT JSON: {{"slugs": ["slug-1", ...]}}. Slugs are filenames without ".md". '
            "If nothing relevant, return empty list."
        )
        try:
            out = chat_json([{"role": "user", "content": prompt}], fast=True)
        except Exception:
            return []
        slugs = out.get("slugs", []) or []
        hits: list[WikiHit] = []
        for s in slugs[:k]:
            page = next(
                (p for p in self.pages if p["domain"] == domain and Path(p["path"]).stem == s),
                None,
            )
            if page:
                hits.append(self._to_hit(page, score=1.0))
        return hits

    def bm25_search(self, query: str, domain: Optional[str] = None, k: int = 5) -> list[WikiHit]:
        if not self.bm25:
            return []
        toks = _tokenize(query)
        scores = self.bm25.get_scores(toks)
        ranked = np.argsort(scores)[::-1]
        out: list[WikiHit] = []
        for i in ranked:
            p = self.pages[i]
            if domain and p["domain"] != domain.lower():
                continue
            out.append(self._to_hit(p, score=float(scores[i])))
            if len(out) >= k:
                break
        return out

    def embed_rerank(self, query: str, candidates: list[WikiHit], k: int = 3) -> list[WikiHit]:
        if not candidates:
            return []
        q_emb = np.array(embed(query)[0])
        hit_texts = []
        for h in candidates:
            page = next((p for p in self.pages if p["path"] == h.path), None)
            hit_texts.append((page["title"] + "\n" + page["body"][:1500]) if page else h.title)
        page_embs = np.array(embed(hit_texts))
        sims = page_embs @ q_emb / (
            np.linalg.norm(page_embs, axis=1) * np.linalg.norm(q_emb) + 1e-9
        )
        order = np.argsort(sims)[::-1][:k]
        return [
            candidates[i].model_copy(update={"score": float(sims[i])}) for i in order
        ]

    def page_content(self, hit: WikiHit) -> str:
        page = next((p for p in self.pages if p["path"] == hit.path), None)
        return page["body"] if page else ""

    def _to_hit(self, page: dict, score: float) -> WikiHit:
        return WikiHit(
            domain=page["domain"],
            path=page["path"],
            title=page["title"],
            score=score,
            escalation_risk=page["escalation_risk"],  # type: ignore[arg-type]
            source_url=page["source_url"] or None,
        )


def _parse_list(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        return []
    try:
        v = json.loads(raw)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        pass
    return [s.strip().strip('"') for s in raw.strip("[]").split(",") if s.strip()]
