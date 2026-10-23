"""Run the 3-agent pipeline and the single-prompt baseline on the same inputs."""
import copy
import json
import time
from pathlib import Path

from pydantic import ValidationError

from agents import llm as llm_module
from agents import verifier
from agents.evaluation import baseline
from agents.facts import allowed_numbers, compute_facts
from agents.orchestrator import run_pipeline
from agents.schemas import ChartsOut, SummaryOut

FATAL = getattr(llm_module, "LLMFatalError", ())  # exists once the fail-fast patch is applied


def _grade(summary, charts, allowed):
    """Code-layer problems in a draft: {'numeric': n, 'other': n}."""
    if summary is None or charts is None:
        return {"numeric": 0, "other": 0}
    issues = verifier.verify_summary(summary, allowed) + verifier.verify_charts(charts, allowed)
    numeric = sum(i.code == "unsupported_number" for i in issues)
    return {"numeric": numeric, "other": len(issues) - numeric}


def _text(summary, charts):
    return {"headline": summary.headline, "explanation": summary.explanation,
            "caveats": summary.caveats, "chart_ids": [c.id for c in charts.charts],
            "captions": [c.caption for c in charts.charts]}


def _usage(llm):
    return getattr(llm, "calls", 0), getattr(llm, "tokens", 0)


def run_pipeline_record(sc, repeat, result, llm):
    params, q = sc["params"], sc["question"]
    facts = compute_facts(params, result)
    allowed = allowed_numbers(facts, result, params)
    c0, t0 = _usage(llm)
    start = time.time()
    out = run_pipeline(params, llm, q, engine_fn=lambda p: copy.deepcopy(result))
    seconds = time.time() - start
    c1, t1 = _usage(llm)

    first = out["trace"]["attempts"][0] if out["trace"]["attempts"] else {
        "summary_issues": [], "chart_issues": [], "review": "skipped"}
    codes = [i["code"] for i in first["summary_issues"] + first["chart_issues"]]
    rec = {
        "key": f"{sc['id']}|r{repeat}|pipeline", "scenario": sc["id"], "repeat": repeat,
        "system": "pipeline", "question": q, "status": out["status"],
        "attempts": len(out["trace"]["attempts"]),
        "first_draft": {
            "invalid_output": "agent_failed" in codes,
            "numeric": codes.count("unsupported_number"),
            "other": sum(c not in ("unsupported_number", "agent_failed", "llm_review") for c in codes),
            "review": first["review"],
        },
        "llm_calls": c1 - c0, "tokens": t1 - t0, "seconds": round(seconds, 2), "error": None,
    }
    if out["summary"]:
        try:
            s = SummaryOut.model_validate(out["summary"])
            c = ChartsOut.model_validate({"charts": [{"id": x["id"], "caption": x["caption"]} for x in out["charts"]]})
            rec["delivered"] = _grade(s, c, allowed)
            rec["text"] = _text(s, c)
        except ValidationError:
            rec["delivered"] = {"numeric": 0, "other": 1}
    else:
        rec["delivered"] = {"numeric": 0, "other": 0}
    return rec


def run_baseline_record(sc, repeat, result, llm, judge):
    params, q = sc["params"], sc["question"]
    facts = compute_facts(params, result)
    allowed = allowed_numbers(facts, result, params)
    c0, t0 = _usage(llm)
    start = time.time()
    summary = charts = None
    invalid = False
    try:
        summary, charts = baseline.run(llm, facts, q)
    except FATAL:
        raise
    except Exception:
        invalid = True
    seconds = time.time() - start
    c1, t1 = _usage(llm)

    review, judge_calls = "not_run", 0
    if judge and summary is not None:
        j0 = _usage(llm)[0]
        review = verifier.llm_review(llm, summary, charts, facts)[1]
        judge_calls = _usage(llm)[0] - j0

    g = _grade(summary, charts, allowed)
    rec = {
        "key": f"{sc['id']}|r{repeat}|baseline", "scenario": sc["id"], "repeat": repeat,
        "system": "baseline", "question": q, "status": "invalid_output" if invalid else "delivered",
        "attempts": 1,
        "first_draft": {"invalid_output": invalid, **g, "review": review},
        "delivered": g, "llm_calls": c1 - c0, "tokens": t1 - t0, "seconds": round(seconds, 2),
        "judge_calls": judge_calls, "error": None,
    }
    if summary is not None:
        rec["text"] = _text(summary, charts)
    return rec


def load_records(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_compare(scenarios, llm, engine_fn, path: Path, repeats=1, judge=False, log=print):
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["key"] for r in load_records(path)}
    engine_cache = {}
    for sc in scenarios:
        for rep in range(repeats):
            for system in ("pipeline", "baseline"):
                key = f"{sc['id']}|r{rep}|{system}"
                if key in done:
                    continue
                if sc["id"] not in engine_cache:
                    engine_cache[sc["id"]] = engine_fn(sc["params"])
                result = engine_cache[sc["id"]]
                try:
                    if system == "pipeline":
                        rec = run_pipeline_record(sc, rep, result, llm)
                    else:
                        rec = run_baseline_record(sc, rep, result, llm, judge)
                except FATAL:
                    raise
                except Exception as exc:  # keep the run going; the error is recorded
                    rec = {"key": key, "scenario": sc["id"], "repeat": rep, "system": system,
                           "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                log(f"{key}: {rec.get('status', 'ERROR')} "
                    f"({rec.get('llm_calls', '-')} calls, {rec.get('seconds', '-')}s)")
