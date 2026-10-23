import json

from agents import summariser, verifier
from agents.facts import allowed_numbers, compute_facts, display
from agents.schemas import SummaryOut
from agents.tests.conftest import FakeLLM
from agents.tests.test_agents import good_summary


def test_time_and_small_amount_display(params, result):
    f = compute_facts(params, result)
    assert display("tenor_years", {**f, "tenor_years": 2.5}) == "2.5 years"
    assert display("peak_epe_time", {**f, "peak_epe_time": 3.25}) == "3.25 years"
    assert display("tenor_years", {**f, "tenor_years": 5}) == "5 years"
    assert display("tenor_years", {**f, "tenor_years": 1}) == "1 year"
    assert display("cva", {**f, "cva": 1.62}) == "$1.62"
    assert display("cva", {**f, "cva": 525.9}) == "$526"
    assert display("cva_pct_notional", {**f, "cva_pct_notional": 0.0000032}) == "0.0003%"


def _issues(params, result, **over):
    f = compute_facts(params, result)
    s = SummaryOut.model_validate(good_summary(f, **over))
    return {i.code for i in verifier.verify_summary(s, allowed_numbers(f, result, params))}


def test_good_summary_still_passes(params, result):
    assert _issues(params, result) == set()


def test_length_number_dump_and_jargon_are_flagged(params, result):
    f = compute_facts(params, result)
    base = good_summary(f)["explanation"]
    assert "too_long" in _issues(params, result, explanation=base + " Filler words here." * 60)
    dump = base + " " + " ".join(display(k, f) + "." for k in ("peak_pfe", "hazard", "recovery_rate", "cds_spread", "par_rate"))
    assert "too_many_numbers" in _issues(params, result, explanation=dump)
    assert "technical_term" in _issues(params, result, explanation=base + " A Vasicek model is used.")


def test_will_default_is_left_to_the_llm_review(params, result):
    f = compute_facts(params, result)
    ex = good_summary(f)["explanation"] + " If the counterparty will default, this is the cost."
    assert "banned_phrase" not in _issues(params, result, explanation=ex)


def test_summariser_never_sees_model_settings(params, result):
    f = compute_facts(params, result)
    seen = {}

    def handler(system, user):
        seen.update(user)
        return good_summary(f)
    summariser.run(FakeLLM(handler), f)
    keys = {x["key"] for x in seen["facts"]}
    assert not keys & summariser.HIDDEN and "cva" in keys
