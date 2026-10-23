import json

import pytest

from agents import verifier
from agents.evaluation import faults, report
from agents.evaluation.compare import load_records, run_compare
from agents.evaluation.local_engine import engine_local
from agents.evaluation.scenarios import SCENARIOS
from agents.evaluation.stats import wilson
from agents.facts import allowed_numbers, compute_facts
from agents.schemas import ChartsOut, SummaryOut
from agents.tests.conftest import FakeLLM


def test_all_scenarios_give_consistent_engine_output():
    assert len(SCENARIOS) == 20 and len({s["id"] for s in SCENARIOS}) == 20
    for sc in SCENARIOS:
        assert verifier.verify_engine(sc["params"], engine_local(sc["params"])) == [], sc["id"]


@pytest.fixture(scope="module")
def code_faults():
    return faults.run_code_faults(SCENARIOS[:8], engine_local)


def test_clean_output_is_never_flagged(code_faults):
    assert code_faults["clean"]["flagged"] == 0


def test_structural_and_engine_faults_are_all_caught(code_faults):
    for t, r in code_faults["by_type"].items():
        if t.startswith("engine") or t in ("unknown_fact_key", "duplicate_fact_key", "unknown_chart",
                                           "duplicate_chart", "advice_phrase"):
            assert r["n"] > 0 and r["caught"] == r["n"], t


def test_known_blind_spots_are_measured_as_missed(code_faults):
    for t in ("number_in_words (known blind spot)", "right_number_wrong_place (known blind spot)"):
        r = code_faults["by_type"][t]
        assert r["n"] > 0 and r["caught"] == 0, t


def test_large_number_errors_mostly_caught(code_faults):
    r = code_faults["by_type"]["numeric_inplace x10"]
    assert r["caught"] / r["n"] > 0.9


def test_llm_fault_sentences_are_invisible_to_the_code_layer():
    sc = SCENARIOS[0]
    res = engine_local(sc["params"])
    facts = compute_facts(sc["params"], res)
    allowed = allowed_numbers(facts, res, sc["params"])
    for sentences in faults.LLM_FAULTS.values():
        for sent in sentences:
            s = faults._clean(facts)
            s["explanation"] += " " + sent
            assert faults._check(s, faults.CLEAN_CHARTS, allowed) == [], sent


def test_wilson_interval_sane():
    lo, hi = wilson(0, 20)
    assert lo == 0 and 0.1 < hi < 0.2
    assert wilson(0, 0) == (0.0, 0.0)


class Scripted:
    """Builds valid agent answers from the facts table inside the prompt itself."""
    def __init__(self, corrupt_first_summary=False):
        self.calls = self.tokens = 0
        self.corrupt = corrupt_first_summary
        self.first = True

    def complete_json(self, system, user):
        self.calls += 1
        self.tokens += 100
        u = json.loads(user)
        if system.startswith("You are a strict"):
            return {"pass": True, "issues": []}
        d = {f["key"]: f["display"] for f in u["facts"]}
        expl_cva = "$9,999" if (self.corrupt and self.first and "catalog" not in u) else d["cva"]
        summary = {
            "headline": f"The counterparty risk costs about {d['cva']} today",
            "explanation": (f"CVA is the estimated cost today of the chance that the other side defaults before the swap ends. "
                            f"Here it is {expl_cva}, or {d['cva_pct_notional']} of the {d['notional']} notional.\n\n"
                            f"The average amount owed peaks at {d['peak_epe']}."),
            "key_numbers": [{"fact_key": k, "label": k} for k in ("cva", "cva_pct_notional", "peak_epe")],
            "caveats": ["Default risk is assumed constant over time."]}
        charts = [{"id": "exposure_profile", "caption": "Look at when the lines peak."}]
        if "catalog" in u and "model_limitations" in u:      # baseline: one call does everything
            self.first = False
            return {**summary, "charts": charts}
        if "catalog" in u:                                    # graph agent
            return {"charts": charts}
        self.first = False                                    # summariser
        return summary


def test_compare_runner_metrics_and_resume(tmp_path):
    path = tmp_path / "compare.jsonl"
    run_compare(SCENARIOS[:2], Scripted(corrupt_first_summary=True), engine_local, path, log=lambda *_: None)
    recs = load_records(path)
    assert len(recs) == 4
    pipe = [r for r in recs if r["system"] == "pipeline"]
    base = [r for r in recs if r["system"] == "baseline"]
    # first scenario: summariser lies once -> caught, retried, delivered clean
    assert pipe[0]["first_draft"]["numeric"] == 1 and pipe[0]["attempts"] == 2
    assert pipe[0]["delivered"]["numeric"] == 0 and pipe[0]["status"] == "verified"
    assert all(b["llm_calls"] == 1 for b in base)
    # re-running does not duplicate completed work
    run_compare(SCENARIOS[:2], Scripted(), engine_local, path, log=lambda *_: None)
    assert len(load_records(path)) == 4

    text = report.compare_report(recs)
    assert "3-agent pipeline" in text and "DELIVERED to user" in text
    assert "Output A" in report.spot_check(recs, n=2)
