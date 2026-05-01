Multi-Domain Support Triage Challenge
Challenge Overview
Build a terminal-based support triage agent that can handle support tickets across three ecosystems:

HackerRank Support: https://support.hackerrank.com/
Claude Help Center: https://support.claude.com/en/
Visa Support: https://www.visa.co.in/support.html
Your agent must use only the provided support corpus to understand the issue, decide whether it can be answered safely, and determine when it should be escalated to a human.

For each issue, the agent should:

identify the request type
classify the issue into a product area
decide whether to reply or escalate
retrieve the most relevant support documentation
generate a safe, grounded response
Some cases will be simple FAQs. Others may involve billing, bugs, fraud, permissions, account access, assessments, or other sensitive situations that require careful routing.

Read the detailed problem statement here - Problem Statement
Support Tickets
A set of support tickets is provided. Some include expected output for validation; others do not. Run all support tickets with your agent and submit the results as a CSV.

support_tickets.zip
Requirements
Your solution must:

be terminal-based
use only the provided support corpus
avoid unsupported claims or hallucinated policies
escalate high-risk, sensitive, or unsupported cases when appropriate
These are the must-haves. Beyond that, participants are encouraged to add improvements or features of their own, such as better retrieval, stronger safety checks, clearer reasoning etc.
Submissions
Clone the repo from Github to get started
You will upload three files on HackerRank Platform:

Code zip: zip your code/ directory and upload it. Exclude virtualenvs, node_modules, build artifacts, the data/ corpus, and the support_issues/ CSVs.
Predictions CSV: your agent's output for support_issues/support_issues.csv (i.e. the populated output.csv).
Chat transcript: the log.txt from the path in Chat transcript logging.