# CVA agents

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
