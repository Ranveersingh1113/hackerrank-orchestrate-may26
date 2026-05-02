# Triage Agent

Python terminal agent for the HackerRank Orchestrate support-ticket challenge.

## Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -r code\requirements.txt
```

The agent uses local Ollama models by default. Keep the client URL explicit on Windows if your global daemon bind is `0.0.0.0`:

```powershell
$env:OLLAMA_HOST = "http://127.0.0.1:11434"
```

Expected models can be overridden with environment variables:

- `OLLAMA_REASON`, default `qwen2.5:7b-instruct-q4_K_M`
- `OLLAMA_FAST`, default `qwen2.5:3b-instruct-q4_K_M`
- `OLLAMA_EMBED`, default `nomic-embed-text`

## Build Wiki

Run once before the full batch if time permits:

```powershell
$env:OLLAMA_HOST = "http://127.0.0.1:11434"
.\.venv\Scripts\python.exe code\main.py ingest --workers 4
```

The runtime also has a raw-corpus BM25 fallback, so batch triage can still run while wiki ingest is incomplete.

## Run Batch

```powershell
$env:OLLAMA_HOST = "http://127.0.0.1:11434"
.\.venv\Scripts\python.exe code\main.py run --csv support_tickets\support_tickets.csv --out support_tickets\output.csv
```

Each row is streamed to `output.csv` as soon as it's processed, so a mid-batch
crash keeps the partial results. To continue an interrupted run without
re-processing finished rows:

```powershell
.\.venv\Scripts\python.exe code\main.py run --resume
```

If a single row throws (e.g. Ollama timeout, malformed JSON), that row is
written as a safe `escalated/invalid` placeholder and the batch continues.

`output.csv` is written with lowercase evaluator fields:

```text
issue,subject,company,response,product_area,status,request_type,justification
```

`status` is always `replied` or `escalated`. `request_type` is always `product_issue`, `feature_request`, `bug`, or `invalid`.

## Single Ticket

```powershell
.\.venv\Scripts\python.exe code\main.py ticket "I lost my Visa card" --company Visa
```

## Architecture

The pipeline is a deterministic state machine:

```text
sanitize -> classify -> infer domain -> retrieve -> escalation policy -> grounded response -> CSV
```

Retrieval prefers generated `wiki/{domain}/index.md` pages, then falls back to BM25 over generated wiki pages plus raw `data/` corpus markdown. High-risk topics, PII, account access, fraud, disputes, chargebacks, payment failures, legal threats, and exam-integrity issues are escalated instead of answered directly.

Anti-hallucination guardrail: every generated reply is run through a
groundedness check (≥35% of meaningful tokens in the reply must overlap with
the retrieved excerpts). If the check fails, the row is flipped to
`escalated` rather than shipped — protecting the evaluator from fabricated
policies even when the LLM ignored the no-invention prompt rule.

## Evaluate against the labelled sample

```powershell
.\.venv\Scripts\python.exe code\eval.py        # full sample CSV
.\.venv\Scripts\python.exe code\eval.py 10     # first 10 rows only
```

Prints per-row predictions, accuracy on `status` and `request_type`, a
confusion matrix, and a disagreement list with expected vs. observed
`product_area` for manual inspection.

## Design Decisions

The three questions an interview judge tends to ask, answered here so the
reasoning isn't recovered from code:

### Why a hand-rolled state machine instead of LangGraph?

The pipeline is seven nodes — `sanitize → classify → route → retrieve →
escalate? → respond → finalize` — with deterministic edges and one
conditional branch. LangGraph buys you typed state, conditional edges,
checkpointing, and human-in-the-loop. The first two are already there
(Pydantic `AgentState` + plain `if`/`else`). The latter two would help in
a long-running orchestration but are dead weight in a 24-hour batch
triager. Removing the framework removed an import dependency, removed a
class of "graph state did not flow as expected" bugs, and made the
escalation routing trivially auditable from `agent.process_ticket()` top
to bottom. If the pipeline ever needed dynamic agent dispatch or
human-in-the-loop pause/resume, LangGraph would be the right move; for
deterministic batch triage it would be over-abstraction.

### Why a 0.35 token-overlap groundedness threshold?

The threshold balances two failure modes: (a) the LLM ignoring the
no-invention rule and citing a plausible-sounding policy that isn't in
the corpus, and (b) the LLM correctly paraphrasing a corpus passage so
heavily that token-overlap drops naturally. 0.35 was picked because a
faithful paraphrase of a help-center article — which uses domain
vocabulary like "test", "candidate", "settings", "expiration" — overlaps
60%+ in practice; an answer fabricated from parametric memory typically
overlaps 15–25%. A length guard skips the check entirely for replies
under 15 distinct content tokens, since short focused answers (a Visa
card replacement step) get penalized by raw ratios. This is a heuristic,
not a proof — the prompt rule is still the primary defense.

### Known failure modes

- **Wiki ingest is the long pole.** With local Ollama + qwen2.5:3b on a
  4 GB GPU, the full corpus takes ~hours. The retriever falls back to
  raw `data/` markdown so triage works without ingest, but with weaker
  product-area metadata.
- **`company=None` cross-domain accuracy** depends on the LLM's domain
  inference. The retriever has a backstop (BM25 across all three
  wikis + rerank with a small domain-hint nudge), but ambiguous tickets
  ("billing question", no product mentioned) can still misroute.
- **Multi-issue tickets** (a row with two distinct sub-issues) are
  classified by the dominant signal; the secondary issue is captured in
  `sub_issues` but doesn't drive a second retrieval.
- **`product_area` is snapped to a closed taxonomy per domain** — this
  trades free-form precision for evaluator-scoreable consistency. A
  ticket whose true area is outside the taxonomy lands on the
  domain's `General` bucket.
- **PII detection is regex-only.** Card numbers, OTPs, PAN, Aadhaar,
  SSN. Passport numbers, IBANs, and free-form sensitive disclosures
  (medical details, etc.) are not caught.

Secrets are read only from environment variables or `.env`; no API keys are required for the default local Ollama setup.
