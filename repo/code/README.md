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

Secrets are read only from environment variables or `.env`; no API keys are required for the default local Ollama setup.
