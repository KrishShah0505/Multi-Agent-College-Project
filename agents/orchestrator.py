"""
Plain-Python orchestrator (no LLM): fixed flow, retry loop, fallback.

engine -> summariser + graph agent -> verifier -> (issues back to the agent that
caused them, up to MAX_RETRIES) -> verified output, or a code-written fallback.
"""
import time

import httpx

from agents import config, grapher, summariser, verifier
from agents.facts import allowed_numbers, compute_facts
from agents.llm import LLMError, LLMFatalError
from agents.schemas import AgentOutputError


class EngineError(Exception):
    pass


def call_engine(params: dict) -> dict:
    payload = {k: v for k, v in params.items() if v is not None}
    try:
        r = httpx.post(f"{config.CVA_API_URL}/cva", json=payload, timeout=60)
    except httpx.HTTPError as exc:
        raise EngineError(f"Could not reach the CVA engine at {config.CVA_API_URL}: {exc}") from exc
    if r.status_code != 200:
        raise EngineError(f"CVA engine returned {r.status_code}: {r.text[:300]}")
    return r.json()


def _run_agent(fn, label, *args, **kwargs):
    """Run an agent; turn failures into issues instead of crashing the loop."""
    try:
        return fn(*args, **kwargs), []
    except LLMFatalError:
        raise
    except (AgentOutputError, LLMError) as exc:
        return None, [verifier.Issue(label, "agent_failed", str(exc))]


def run_pipeline(params: dict, llm, question: str | None = None,
                 engine_fn=call_engine, max_retries: int | None = None) -> dict:
    t0 = time.time()
    max_retries = config.MAX_RETRIES if max_retries is None else max_retries

    result = engine_fn(params)
    facts = compute_facts(params, result)
    allowed = allowed_numbers(facts, result, params)

    base = {"engine_result": result, "facts": facts}
    engine_issues = verifier.verify_engine(params, result)
    if engine_issues:  # agents must not explain a result that fails its own checks
        return {**base, "status": "engine_check_failed", "summary": None, "charts": [],
                "verification": {"status": "failed", "issues": [i.as_dict() for i in engine_issues]},
                "trace": {"attempts": [], "llm_calls": 0, "tokens": 0,
                          "seconds": round(time.time() - t0, 3)}}

    summary = charts = None
    s_problems: list[str] = []
    c_problems: list[str] = []
    attempts, review_status, final_issues = [], "skipped", []

    for n in range(max_retries + 1):
        s_issues, c_issues = [], []
        if summary is None or s_problems:
            summary_new, s_issues = _run_agent(summariser.run, "summary", llm, facts, question,
                                               summary, s_problems or None)
            summary = summary_new or summary
            if summary_new is None and n == 0:
                summary = None
        if charts is None or c_problems:
            charts_new, c_issues = _run_agent(grapher.run, "charts", llm, facts, question,
                                              charts, c_problems or None)
            charts = charts_new or charts
            if charts_new is None and n == 0:
                charts = None

        # Layer 1: code checks (only on drafts that exist and were just produced)
        if summary is not None and not s_issues:
            s_issues += verifier.verify_summary(summary, allowed)
        if charts is not None and not c_issues:
            c_issues += verifier.verify_charts(charts, allowed)

        # Layer 2: LLM wording review, only when the code checks are clean
        review_status, r_issues = "skipped", []
        if not s_issues and not c_issues and summary and charts:
            r_issues, review_status = verifier.llm_review(llm, summary, charts, facts)
            s_issues += [i for i in r_issues if i.source == "summary"]
            c_issues += [i for i in r_issues if i.source == "charts"]

        attempts.append({"attempt": n + 1, "review": review_status,
                         "summary_issues": [i.as_dict() for i in s_issues],
                         "chart_issues": [i.as_dict() for i in c_issues]})
        final_issues = s_issues + c_issues
        if not final_issues:
            break
        s_problems = [i.message for i in s_issues]
        c_problems = [i.message for i in c_issues]

    trace = {"attempts": attempts, "llm_calls": getattr(llm, "calls", None),
             "tokens": getattr(llm, "tokens", None), "seconds": round(time.time() - t0, 3)}

    if final_issues:  # retries exhausted: show code-written content, clearly flagged
        return {**base, "status": "fallback",
                "summary": summariser.fallback(facts),
                "charts": grapher.fallback(params, result),
                "verification": {"status": "failed", "issues": [i.as_dict() for i in final_issues]},
                "trace": trace}

    status = "verified" if review_status == "pass" else "partially_verified"
    return {**base, "status": status,
            "summary": summariser.render(summary, facts),
            "charts": grapher.render(charts, params, result),
            "verification": {"status": status, "issues": []},
            "trace": trace}
