"""
CrewAI Flow that runs one explanation request end to end.

    fetch_engine (start) -> engine_check (router) -> run_agents  | stop_on_engine_error

The Flow gives the workflow explicit, inspectable steps and a router; the agent work
itself is done by the crew in crew_team.py.
"""
import time
from typing import Any, Optional

from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel, ConfigDict

from agents import config, grapher, summariser, verifier
from agents.crew_team import CvaCrewTeam
from agents.facts import allowed_numbers, compute_facts
from agents.orchestrator import call_engine


class ExplainState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    params: dict = {}
    question: Optional[str] = None
    max_retries: int = 2
    engine_fn: Any = None
    team: Any = None
    result: Any = None
    facts: Any = None
    allowed: Any = None
    engine_issues: Any = None
    t0: float = 0.0
    output: Any = None


class CvaExplainFlow(Flow[ExplainState]):

    @start()
    def fetch_engine(self):
        s = self.state
        s.t0 = time.time()
        s.result = s.engine_fn(s.params)
        s.facts = compute_facts(s.params, s.result)
        s.allowed = allowed_numbers(s.facts, s.result, s.params)
        s.engine_issues = verifier.verify_engine(s.params, s.result)

    @router(fetch_engine)
    def engine_check(self):
        return "engine_failed" if self.state.engine_issues else "engine_ok"

    @listen("engine_failed")
    def stop_on_engine_error(self):
        s = self.state
        s.output = {"engine_result": s.result, "facts": s.facts, "status": "engine_check_failed",
                    "summary": None, "charts": [],
                    "verification": {"status": "failed", "issues": [i.as_dict() for i in s.engine_issues]},
                    "trace": {"attempts": [], "llm_calls": 0, "tokens": 0,
                              "seconds": round(time.time() - s.t0, 3), "crew": s.team.description}}
        return s.output

    @listen("engine_ok")
    def run_agents(self):
        s, team = self.state, self.state.team
        summary = charts = None
        s_problems, c_problems, attempts = [], [], []
        calls = tokens = 0
        final_issues, review_status = [], "skipped"

        for n in range(s.max_retries + 1):
            att = team.run_attempt(s.facts, s.allowed, s.question, summary, charts,
                                   s_problems or None, c_problems or None)
            summary, charts = att["summary"], att["charts"]
            calls, tokens = calls + att["calls"], tokens + att["tokens"]

            # problems the guardrails caught and the agents then fixed (or failed to fix)
            s_issues = [e for e in att["events"] if e["task"] == "summary"]
            c_issues = [e for e in att["events"] if e["task"] == "charts"]
            review = att["review"]
            review_status = "skipped"
            if att["error"] and not att["guardrail_exhausted"]:
                s_issues.append({"code": "agent_failed", "message": att["error"]})
                review_status = "unavailable"
            elif review is not None:
                review_status = "pass" if (review.pass_ and not review.issues) else "fail"
                for i in review.issues:
                    (c_issues if i.target == "charts" else s_issues).append(
                        {"code": "llm_review", "message": i.message})
            elif summary is not None and charts is not None and not att["error"]:
                review_status = "unavailable"

            attempts.append({"attempt": n + 1, "review": review_status,
                             "summary_issues": [{"source": "summary", **{k: e[k] for k in ("code", "message")}} for e in s_issues],
                             "chart_issues": [{"source": "charts", **{k: e[k] for k in ("code", "message")}} for e in c_issues]})

            if att["guardrail_exhausted"] or summary is None or charts is None:
                final_issues = (s_issues + c_issues) or [{"code": "agent_failed", "message": att["error"] or "no output"}]
                break
            # open problems = what the reviewer still objects to (guardrail problems were fixed in-task)
            open_s = [e["message"] for e in s_issues if e["code"] == "llm_review"]
            open_c = [e["message"] for e in c_issues if e["code"] == "llm_review"]
            if not open_s and not open_c:
                final_issues = []
                break
            s_problems, c_problems = open_s, open_c
            final_issues = open_s + open_c

        trace = {"attempts": attempts, "llm_calls": calls, "tokens": tokens,
                 "seconds": round(time.time() - s.t0, 3), "crew": team.description}
        base = {"engine_result": s.result, "facts": s.facts}
        if final_issues:
            s.output = {**base, "status": "fallback", "summary": summariser.fallback(s.facts),
                        "charts": grapher.fallback(s.params, s.result),
                        "verification": {"status": "failed", "issues": [
                            {"source": "summary", "code": i.get("code", "issue"), "message": i["message"]}
                            if isinstance(i, dict) else {"source": "summary", "code": "issue", "message": i}
                            for i in final_issues]},
                        "trace": trace}
        else:
            status = "verified" if review_status == "pass" else "partially_verified"
            s.output = {**base, "status": status, "summary": summariser.render(summary, s.facts),
                        "charts": grapher.render(charts, s.params, s.result),
                        "verification": {"status": status, "issues": []}, "trace": trace}
        return s.output


def run_crew_pipeline(params: dict, question: Optional[str] = None, team: Optional[CvaCrewTeam] = None,
                      engine_fn=call_engine, max_retries: Optional[int] = None) -> dict:
    flow = CvaExplainFlow(suppress_flow_events=not config.CREW_VERBOSE)
    flow.state.params = params
    flow.state.question = question
    flow.state.max_retries = config.MAX_RETRIES if max_retries is None else max_retries
    flow.state.engine_fn = engine_fn
    flow.state.team = team or CvaCrewTeam()
    flow.kickoff()
    return flow.state.output
