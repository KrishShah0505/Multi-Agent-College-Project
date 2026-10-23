# CVA agents

> **Backend:** the three agents run as a **CrewAI** crew inside a **CrewAI Flow** (`AGENT_BACKEND=crewai`, the default).
> `crew_team.py` = Agents, Tasks, sequential Crew and Task guardrails. `flow.py` = the Flow (start, router, listeners).
> The earlier framework-free code (`orchestrator.py`, `llm.py`) is kept as the single-prompt baseline and as `AGENT_BACKEND=direct`.

Three LLM agents (Groq) around the unchanged CVA engine. Code runs the flow.

```
POST /cva (engine) -> facts (code) -> Summariser + Graph agent -> Verifier
                                          ^                          |
                                          +---- issues, max 2 retries +--> fallback if still failing
```

- **Summariser**: plain-language explanation. Copies numbers from code-made facts; key-number values are filled in by code.
- **Graph agent**: picks 1-3 charts from a fixed catalog (`exposure_profile`, `default_probability`, `cva_contribution`) and writes captions. Chart data is built by code.
- **Verifier**: code checks (engine reproducibility, every number traced to a fact, valid ids, no advice phrases) then an LLM wording review.

Statuses: `verified`, `partially_verified` (LLM review unavailable), `fallback` (code-written text after retries), `engine_check_failed` (agents not run).

## Run (from repo root)
```
pip install -r agents/requirements.txt
export GROQ_API_KEY=...            # optional: GROQ_MODEL, LLM_MIN_INTERVAL_S=2
uvicorn api.main:app --app-dir backend --port 8000     # engine
uvicorn agents.app:app --port 8001                      # agents
curl -X POST localhost:8001/explain -H 'content-type: application/json' -d '{"params":{"notional":1000000,"tenor_years":5,"payment_freq_months":6,"kappa":0.3,"theta":0.06,"sigma":0.01,"r0":0.05,"cds_spread":0.015,"recovery_rate":0.4,"n_paths":5000,"seed":42},"question":"Why is CVA this size?"}'
pytest agents/tests -q
```

## Known limits
- Number check is rounding-based, so a coarse number (e.g. "5") can match an unrelated fact by coincidence. Number words ("five") are not checked.
- The LLM half of the verifier only judges wording and can raise false alarms.
- Not yet tested against live Groq.

## Evaluation (agents/evaluation/)
```
python -m agents.evaluation.run_eval faults               # offline, no API key; code-layer fault injection
python -m agents.evaluation.run_eval faults --with-llm    # + LLM wording review on number-free faults (~36 calls)
python -m agents.evaluation.run_eval compare --n 20 --repeats 1 --judge   # pipeline vs single prompt (needs GROQ key)
python -m agents.evaluation.run_eval report               # rebuild tables + blind spot-check sheet
```
Results land in `agents/evaluation/results/` (`faults.md`, `compare.jsonl`, `compare.md`, `spot_check.md`).
`compare` resumes where it stopped (free-tier limits), `--fresh` starts over, `--repeats 3` measures consistency.
Start with `--n 3` to check the setup before the full run. Add `agents/evaluation/results/` to `.gitignore` if you do not want raw runs committed.


## CrewAI architecture
```
POST /explain -> CvaExplainFlow
   @start  fetch_engine      call the CVA engine, build facts, run engine checks
   @router engine_check      engine_ok | engine_failed
   @listen run_agents        one CrewAI Crew (sequential, 3 agents):
        Plain-Language CVA Explainer  -> Task + guardrail (code verifier)
        Chart Selection Specialist    -> Task + guardrail (code verifier)
        Fact-Checking Reviewer        -> Task, reads the other two via context
        reviewer objections -> next attempt (only the agent at fault re-runs), max 2 retries
        still failing -> code-written fallback
```
Settings: `AGENT_BACKEND` (crewai|direct), `CREW_VERBOSE=true` prints each agent's work (use it for screenshots),
`GUARDRAIL_RETRIES` (default 2). Groq is reached through CrewAI's OpenAI-compatible provider
(`openai/<GROQ_MODEL>` + Groq base URL), so LiteLLM is not needed. CrewAI telemetry and tracing are switched off in `config.py`.

Install: `pip install -r agents/requirements.txt` (CrewAI is a large install; use Python 3.10 to 3.13).
Evaluation through the crew: `python -m agents.evaluation.run_eval compare --n 3 --judge --backend crewai`.
