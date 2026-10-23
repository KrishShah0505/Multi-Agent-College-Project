## Pipeline vs single-prompt baseline

| Metric | 3-agent pipeline | Single prompt |
|---|---|---|
| Runs completed (errors) | 3 (0) | 3 (0) |
| First draft: unusable JSON/schema | 0/3 (0%, CI 0-56%) | 0/3 (0%, CI 0-56%) |
| First draft: has a wrong number | 1/3 (33%, CI 6-79%) | 0/3 (0%, CI 0-56%) |
| First draft: other rule break (bad key/id, advice phrase) | 0/3 (0%, CI 0-56%) | 1/3 (33%, CI 6-79%) |
| First draft: wording flagged by reviewer | 1/2 (50%, CI 9-91%) | 3/3 (100%, CI 44-100%) |
| DELIVERED to user: has a wrong number | 0/3 (0%, CI 0-56%) | 0/3 (0%, CI 0-56%) |
| DELIVERED to user: other rule break | 0/3 (0%, CI 0-56%) | 1/3 (33%, CI 6-79%) |
| Needed more than one attempt | 2/3 (67%, CI 21-94%) | n/a |
| Mean attempts | 2.33 | 1.00 |
| Mean model calls per query | 11.3 | 1.7 |
| Mean tokens per query | 12748 | 2791 |
| Seconds per query (mean / median) | 79.0 / 84.2 | 13.4 / 12.9 |

Pipeline final status: {'fallback': 2, 'verified': 1}

Runs where the first draft had a problem and a retry fixed it: 0/3
Runs that ended in the code-written fallback: 2/3

Notes: the wrong-number grader is the same code check the pipeline enforces, so 'delivered' numbers for the pipeline are low by construction. The fair comparison is the first-draft row. Wording is only graded by an LLM reviewer; use the spot-check sheet for human ratings.