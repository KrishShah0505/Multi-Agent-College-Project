"""
Usage (from the repo root):
  python -m agents.evaluation.run_eval faults                 # no API key needed
  python -m agents.evaluation.run_eval faults --with-llm      # also tests the LLM wording review
  python -m agents.evaluation.run_eval compare --n 20 --repeats 1 --judge
  python -m agents.evaluation.run_eval report
Raw results go to agents/evaluation/results/. 'compare' resumes where it stopped.
"""
import argparse
import json
from pathlib import Path

from agents.evaluation import report
from agents.evaluation.faults import run_code_faults, run_llm_faults
from agents.evaluation.local_engine import engine_local
from agents.evaluation.scenarios import SCENARIOS

RESULTS = Path(__file__).resolve().parent / "results"


def _engine(kind):
    if kind == "http":
        from agents.orchestrator import call_engine
        return call_engine
    return engine_local


def cmd_faults(a):
    RESULTS.mkdir(exist_ok=True)
    engine = _engine(a.engine)
    data = {"code": run_code_faults(SCENARIOS, engine)}
    if a.with_llm:
        from agents.llm import GroqClient
        data["llm"] = run_llm_faults(SCENARIOS, engine, GroqClient(), k=a.llm_k)
    (RESULTS / "faults.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    text = report.faults_report(data)
    (RESULTS / "faults.md").write_text(text, encoding="utf-8")
    print(text)


def cmd_compare(a):
    from agents.evaluation.compare import run_compare
    from agents.llm import GroqClient
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "compare.jsonl"
    if a.fresh and path.exists():
        path.unlink()
    run_compare(SCENARIOS[:a.n], GroqClient(), _engine(a.engine), path,
                repeats=a.repeats, judge=a.judge)
    cmd_report(a)


def cmd_report(a):
    from agents.evaluation.compare import load_records
    recs = load_records(RESULTS / "compare.jsonl")
    if not recs:
        print("No compare results yet. Run: python -m agents.evaluation.run_eval compare")
        return
    (RESULTS / "compare.md").write_text(report.compare_report(recs), encoding="utf-8")
    (RESULTS / "spot_check.md").write_text(report.spot_check(recs), encoding="utf-8")
    print(report.compare_report(recs))
    print(f"\nWrote {RESULTS / 'compare.md'} and {RESULTS / 'spot_check.md'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["local", "http"], default="local")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("faults"); f.add_argument("--with-llm", action="store_true"); f.add_argument("--llm-k", type=int, default=6)
    c = sub.add_parser("compare")
    c.add_argument("--n", type=int, default=20); c.add_argument("--repeats", type=int, default=1)
    c.add_argument("--judge", action="store_true"); c.add_argument("--fresh", action="store_true")
    sub.add_parser("report")
    a = p.parse_args()
    {"faults": cmd_faults, "compare": cmd_compare, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
