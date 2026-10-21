import copy
import json

import pytest

from agents import verifier
from agents.charts import CATALOG, build_chart
from agents.facts import allowed_numbers, compute_facts, cva_contributions, display
from agents.orchestrator import run_pipeline
from agents.schemas import ChartsOut, SummaryOut
from agents.tests.conftest import FakeLLM


# ---------------------------------------------------------------- helpers ---
def good_summary(facts, **over):
    d = lambda k: display(k, facts)  # noqa: E731
    s = {"headline": f"The counterparty risk on this swap costs about {d('cva')} today",
         "explanation": (f"CVA is the estimated cost today of the chance that the other side defaults before the swap ends. "
                         f"Here it is {d('cva')}, or {d('cva_pct_notional')} of the {d('notional')} notional over {d('tenor_years')}.\n\n"
                         f"The average amount owed to us peaks at {d('peak_epe')}."),
         "key_numbers": [{"fact_key": "cva", "label": "Estimated CVA"},
                         {"fact_key": "cva_pct_notional", "label": "Share of notional"},
                         {"fact_key": "peak_epe", "label": "Peak average owed"}],
         "caveats": ["Default risk is assumed constant over time."]}
    s.update(over)
    return s


GOOD_CHARTS = {"charts": [{"id": "exposure_profile", "caption": "Look at when the lines peak."},
                          {"id": "cva_contribution", "caption": "Each bar is one period's share of CVA."}]}


def make_handler(facts, summaries, charts=None, review=None):
    """Return summaries[i] on the i-th summariser call, etc."""
    state = {"s": 0, "c": 0, "r": 0}

    def h(system, user):
        if system.startswith("You explain"):
            out = summaries[min(state["s"], len(summaries) - 1)]
            state["s"] += 1
            return out
        if system.startswith("You choose"):
            out = (charts or [GOOD_CHARTS])[min(state["c"], len(charts or [1]) - 1)]
            state["c"] += 1
            return out
        out = (review or [{"pass": True, "issues": []}])[min(state["r"], len(review or [1]) - 1)]
        state["r"] += 1
        return out
    return h


# ------------------------------------------------------------ facts/charts ---
def test_contributions_sum_to_engine_cva(params, result):
    c = cva_contributions(result["time_grid"], result["epe"], result["hazard"], params["recovery_rate"])
    assert sum(c) == pytest.approx(result["cva"], rel=1e-9)


def test_charts_build_and_have_consistent_rows(params, result):
    for cid in CATALOG:
        ch = build_chart(cid, params, result)
        assert ch["id"] == cid and ch["data"]
        keys = {s["key"] for s in ch["series"]} | {"t"}
        assert all(keys <= set(row) for row in ch["data"])
    cc = build_chart("cva_contribution", params, result)
    assert cc["data"][-1]["cumulative"] == pytest.approx(result["cva"], rel=1e-9)
    assert sum(r["share"] for r in cc["data"]) == pytest.approx(1.0)
    dp = build_chart("default_probability", params, result)
    assert all(0 <= r["marginal_pd"] <= r["cumulative_pd"] <= 1 for r in dp["data"])


# -------------------------------------------------------- verifier: engine ---
def test_engine_passes_on_real_output(params, result):
    assert verifier.verify_engine(params, result) == []


def test_engine_catches_tampered_cva(params, fresh_result):
    fresh_result["cva"] *= 1.10
    assert any(i.code == "cva_not_reproducible" for i in verifier.verify_engine(params, fresh_result))


def test_engine_catches_negative_epe(params, fresh_result):
    fresh_result["epe"][2] = -5.0
    assert any(i.code == "negative_exposure" for i in verifier.verify_engine(params, fresh_result))


# ------------------------------------------------- verifier: fault injection ---
def test_good_summary_passes_numbers(params, result):
    facts = compute_facts(params, result)
    s = SummaryOut.model_validate(good_summary(facts))
    assert verifier.verify_summary(s, allowed_numbers(facts, result, params)) == []


@pytest.mark.parametrize("old_fn,new", [
    (lambda f: display("cva", f), "$9,999"),                 # wrong dollar figure
    (lambda f: display("cva_pct_notional", f), "4.2%"),      # wrong percentage
    (lambda f: display("tenor_years", f), "7 years"),        # wrong tenor
    (lambda f: display("notional", f), "$2,000,000"),        # wrong notional
])
def test_injected_wrong_numbers_are_caught(params, result, old_fn, new):
    facts = compute_facts(params, result)
    raw = good_summary(facts)
    raw["explanation"] = raw["explanation"].replace(old_fn(facts), new, 1)
    s = SummaryOut.model_validate(raw)
    issues = verifier.verify_summary(s, allowed_numbers(facts, result, params))
    assert any(i.code == "unsupported_number" for i in issues)


def test_unit_mismatch_is_caught(params, result):
    facts = compute_facts(params, result)
    raw = good_summary(facts)
    raw["headline"] = "The risk is about 8% today"   # no percentage fact is near 8
    s = SummaryOut.model_validate(raw)
    assert any(i.code == "unsupported_number"
               for i in verifier.verify_summary(s, allowed_numbers(facts, result, params)))


def test_advice_and_bad_keys_are_caught(params, result):
    facts = compute_facts(params, result)
    raw = good_summary(facts)
    raw["caveats"] = ["You should hedge this exposure."]
    raw["key_numbers"][0]["fact_key"] = "made_up"
    issues = verifier.verify_summary(SummaryOut.model_validate(raw), allowed_numbers(facts, result, params))
    codes = {i.code for i in issues}
    assert {"banned_phrase", "unknown_fact_key"} <= codes


def test_bad_chart_id_is_caught(params, result):
    facts = compute_facts(params, result)
    c = ChartsOut.model_validate({"charts": [{"id": "pie_of_doom", "caption": "x"}]})
    assert any(i.code == "unknown_chart" for i in verifier.verify_charts(c, allowed_numbers(facts, result, params)))


# --------------------------------------------------------- orchestrator ------
def run(params, result, handler, **kw):
    llm = FakeLLM(handler)
    out = run_pipeline(params, llm, engine_fn=lambda p: copy.deepcopy(result), **kw)
    return out, llm


def test_happy_path_is_verified(params, result):
    facts = compute_facts(params, result)
    out, llm = run(params, result, make_handler(facts, [good_summary(facts)]))
    assert out["status"] == "verified" and len(out["trace"]["attempts"]) == 1
    assert llm.calls == 3                       # summariser, grapher, review
    assert out["summary"]["key_numbers"][0]["value"] == display("cva", facts)
    assert [c["id"] for c in out["charts"]] == ["exposure_profile", "cva_contribution"]


def test_retry_fixes_a_wrong_number(params, result):
    facts = compute_facts(params, result)
    bad = good_summary(facts)
    bad["explanation"] = bad["explanation"].replace(display("cva", facts), "$9,999", 1)
    out, llm = run(params, result, make_handler(facts, [bad, good_summary(facts)]))
    assert out["status"] == "verified"
    assert len(out["trace"]["attempts"]) == 2
    assert out["trace"]["attempts"][0]["summary_issues"][0]["code"] == "unsupported_number"


def test_persistent_errors_fall_back_to_code_summary(params, result):
    facts = compute_facts(params, result)
    bad = good_summary(facts)
    bad["explanation"] = bad["explanation"].replace(display("cva", facts), "$9,999", 1)
    out, _ = run(params, result, make_handler(facts, [bad]), max_retries=2)
    assert out["status"] == "fallback" and out["verification"]["status"] == "failed"
    # the fallback itself must satisfy the same numeric checks
    allowed = allowed_numbers(facts, result, params)
    text = out["summary"]["headline"] + " " + out["summary"]["explanation"]
    assert verifier.check_numbers(text, allowed, "summary", "fallback") == []


def test_llm_review_failure_triggers_rewrite(params, result):
    facts = compute_facts(params, result)
    review = [{"pass": False, "issues": [{"target": "summary", "message": "Explain what PFE means."}]},
              {"pass": True, "issues": []}]
    out, _ = run(params, result, make_handler(facts, [good_summary(facts)], review=review))
    assert out["status"] == "verified" and len(out["trace"]["attempts"]) == 2


def test_unusable_review_gives_partially_verified(params, result):
    facts = compute_facts(params, result)
    out, _ = run(params, result, make_handler(facts, [good_summary(facts)], review=[{"nonsense": 1}]))
    assert out["status"] == "partially_verified"


def test_malformed_summary_json_is_retried(params, result):
    facts = compute_facts(params, result)
    out, _ = run(params, result, make_handler(facts, [{"headline": "x"}, good_summary(facts)]))
    assert out["status"] == "verified"
    assert out["trace"]["attempts"][0]["summary_issues"][0]["code"] == "agent_failed"


def test_broken_engine_output_blocks_agents(params, result):
    facts = compute_facts(params, result)
    broken = copy.deepcopy(result)
    broken["cva"] *= 2
    llm = FakeLLM(make_handler(facts, [good_summary(facts)]))
    out = run_pipeline(params, llm, engine_fn=lambda p: broken)
    assert out["status"] == "engine_check_failed" and llm.calls == 0
