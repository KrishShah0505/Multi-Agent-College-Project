"""CrewAI path, tested offline with a scripted model (no network, no API key)."""
import copy
import json

import pytest
from crewai import BaseLLM

from agents.crew_team import CvaCrewTeam
from agents.facts import compute_facts, display
from agents.flow import run_crew_pipeline
from agents.llm import LLMFatalError
from agents.tests.conftest import PARAMS, engine


class ScriptedCrewLLM(BaseLLM):
    """Answers as each CrewAI agent would, building valid JSON from the facts in the prompt."""

    def __init__(self, bad_first_summary=False, always_bad_summary=False, reviews=None, raise_exc=None):
        super().__init__(model="scripted")
        self.bad_first, self.always_bad, self.raise_exc = bad_first_summary, always_bad_summary, raise_exc
        self.reviews = list(reviews or [])
        self.n = {"summariser": 0, "grapher": 0, "reviewer": 0}

    def supports_function_calling(self):
        return False

    def get_context_window_size(self):
        return 8000

    @staticmethod
    def _find(text, marker):
        i = text.find(marker)
        obj, _ = json.JSONDecoder().raw_decode(text[i + len(marker):].lstrip())
        return obj

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, **kw) -> str:
        if self.raise_exc:
            raise self.raise_exc
        text = "\n".join(m["content"] for m in messages if isinstance(m.get("content"), str))
        if "Fact-Checking Reviewer" in text:
            self.n["reviewer"] += 1
            return json.dumps(self.reviews.pop(0) if self.reviews else {"pass": True, "issues": []})
        if "Chart Selection Specialist" in text:
            self.n["grapher"] += 1
            return json.dumps({"charts": [{"id": "exposure_profile", "caption": "Look at when the lines peak."},
                                          {"id": "cva_contribution", "caption": "Each bar is one period's share of CVA."}]})
        self.n["summariser"] += 1
        d = {f["key"]: f["display"] for f in self._find(text, "INPUT (JSON):")["facts"]}
        cva = "$9,999" if (self.always_bad or (self.bad_first and self.n["summariser"] == 1)) else d["cva"]
        return json.dumps({
            "headline": f"The counterparty risk costs about {d['cva']} today",
            "explanation": ("CVA is the estimated cost today of the chance that the other side defaults before the swap ends. "
                            f"Here it is {cva}, or {d['cva_pct_notional']} of the {d['notional']} notional.\n\n"
                            f"The average amount owed peaks at {d['peak_epe']}."),
            "key_numbers": [{"fact_key": k, "label": k} for k in ("cva", "cva_pct_notional", "peak_epe")],
            "caveats": ["Default risk is assumed constant over time."]})


class HttpErr(Exception):
    status_code = 404


@pytest.fixture(scope="module")
def result():
    return engine(dict(PARAMS))


def run(result, llm, **kw):
    team = CvaCrewTeam(llm=llm, verbose=False)
    out = run_crew_pipeline(dict(PARAMS), "Why is the CVA this size?", team=team,
                            engine_fn=lambda p: copy.deepcopy(result), **kw)
    return out, llm


def test_three_agents_run_in_one_crew_and_output_is_verified(result):
    out, llm = run(result, ScriptedCrewLLM())
    assert out["status"] == "verified" and len(out["trace"]["attempts"]) == 1
    assert llm.n == {"summariser": 1, "grapher": 1, "reviewer": 1}
    assert out["trace"]["crew"]["framework"] == "CrewAI"
    assert len(out["trace"]["crew"]["agents"]) == 3
    facts = compute_facts(dict(PARAMS), result)
    assert out["summary"]["key_numbers"][0]["value"] == display("cva", facts)
    assert [c["id"] for c in out["charts"]] == ["exposure_profile", "cva_contribution"]


def test_guardrail_sends_a_wrong_number_back_to_the_agent(result):
    out, llm = run(result, ScriptedCrewLLM(bad_first_summary=True))
    assert out["status"] == "verified"
    assert llm.n["summariser"] == 2                       # first draft rejected, second accepted
    issues = out["trace"]["attempts"][0]["summary_issues"]
    assert any(i["code"] == "unsupported_number" for i in issues)


def test_reviewer_objection_triggers_a_second_attempt_for_that_agent_only(result):
    review = [{"pass": False, "issues": [{"target": "summary", "message": "Explain what PFE means."}]},
              {"pass": True, "issues": []}]
    out, llm = run(result, ScriptedCrewLLM(reviews=review))
    assert out["status"] == "verified" and len(out["trace"]["attempts"]) == 2
    assert llm.n == {"summariser": 2, "grapher": 1, "reviewer": 2}
    assert out["trace"]["attempts"][0]["summary_issues"][0]["code"] == "llm_review"


def test_persistent_wrong_numbers_end_in_the_fallback(result):
    out, llm = run(result, ScriptedCrewLLM(always_bad_summary=True))
    assert out["status"] == "fallback" and out["verification"]["status"] == "failed"
    assert llm.n["summariser"] == 3                       # first try plus two guardrail retries
    assert out["summary"]["key_numbers"]                  # code-written text is still shown


def test_broken_engine_output_stops_before_any_agent_runs(result):
    broken = copy.deepcopy(result)
    broken["cva"] *= 2
    llm = ScriptedCrewLLM()
    out = run_crew_pipeline(dict(PARAMS), None, team=CvaCrewTeam(llm=llm, verbose=False),
                            engine_fn=lambda p: broken)
    assert out["status"] == "engine_check_failed" and sum(llm.n.values()) == 0


def test_permanent_llm_error_fails_fast_with_a_clear_error(result):
    with pytest.raises(LLMFatalError):
        run(result, ScriptedCrewLLM(raise_exc=HttpErr("model not found")))
