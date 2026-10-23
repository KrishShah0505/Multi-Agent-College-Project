"""
Fault injection: deliberately break known-good output and measure how often the
verifier notices. The code layer needs no LLM. The LLM-review layer is tested
separately with number-free faults that only a reader could catch.
"""
import copy
from collections import defaultdict

from pydantic import ValidationError

from agents import summariser, verifier
from agents.facts import FACT_META, allowed_numbers, compute_facts, display
from agents.schemas import ChartsOut, SummaryOut

LEVELS = {"+1%": 0.01, "+5%": 0.05, "+10%": 0.10, "+50%": 0.50, "x10": 9.0, "/10": -0.9}
INSERT_KEYS = ["cumulative_default_prob", "peak_pfe", "hazard", "recovery_rate",
               "cds_spread", "par_rate", "peak_pfe_time"]
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
         8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
ADVICE = ["You should hedge this exposure.", "We recommend reducing the position.",
          "The counterparty will lose us money.", "This is a good deal."]

CLEAN_CHARTS = {"charts": [{"id": "exposure_profile", "caption": "Look at when the lines peak."},
                           {"id": "cva_contribution", "caption": "Each bar is one period's share of CVA."}]}

# number-free sentences: the code layer cannot see these, only a reader can
LLM_FAULTS = {
    "unsupported_claim": [
        "The counterparty is in poor financial health, which is why the risk is high.",
        "Interest rates are expected to rise next year, which pushes the exposure up.",
        "The other side has missed payments before, so a default is likely.",
        "The counterparty will default before the swap ends."],
    "soft_advice": [
        "It would be wise to reduce this position soon.",
        "Anyone holding this swap ought to look for a different counterparty."],
    "wrong_definition": [
        "CVA is the profit the bank earns from entering the swap.",
        "CVA is the amount the bank will pay to the counterparty at the end of the swap."],
    "unexplained_jargon": [
        "Exposure here is the positive part of the mark-to-market value, netted per counterparty.",
        "The hazard rate and the PFE quantile are calibrated to the CDS curve."],
}


def _clean(facts):
    """Known-good, code-written summary (the pipeline's own fallback text)."""
    return summariser.fallback(facts)


def _check(summary: dict, charts: dict, allowed: dict):
    try:
        s, c = SummaryOut.model_validate(summary), ChartsOut.model_validate(charts)
    except ValidationError:
        return [verifier.Issue("summary", "schema", "invalid")]
    return verifier.verify_summary(s, allowed) + verifier.verify_charts(c, allowed)


def _perturbed(key, facts, eps):
    f2 = dict(facts)
    f2[key] = facts[key] * (1 + eps)
    return display(key, f2)


def _swap_digits(s: str):
    idx = [i for i, ch in enumerate(s) if ch.isdigit()]
    for a, b in zip(idx, idx[1:]):
        if s[a] != s[b]:
            return s[:a] + s[b] + s[a + 1:b] + s[a] + s[b + 1:]
    return s


def numeric_and_structural(facts):
    """Yield (fault_type, summary, charts, detail) for one scenario. summary is None for a no-op."""
    base = _clean(facts)
    text = base["explanation"]

    def with_text(new):
        s = copy.deepcopy(base)
        s["explanation"] = new
        return s

    for key in FACT_META:
        old = display(key, facts)
        if old not in text:
            continue
        for name, eps in LEVELS.items():
            new = _perturbed(key, facts, eps)
            yield (f"numeric_inplace {name}", with_text(text.replace(old, new, 1)) if new != old else None,
                   CLEAN_CHARTS, f"{key}: {old} -> {new}")
        swapped = _swap_digits(old)
        yield ("digit_swap", with_text(text.replace(old, swapped, 1)) if swapped != old else None,
               CLEAN_CHARTS, f"{key}: {old} -> {swapped}")

    for key in INSERT_KEYS:
        old = display(key, facts)
        for name, eps in LEVELS.items():
            new = _perturbed(key, facts, eps)
            yield (f"numeric_inserted {name}",
                   with_text(text + f" The {key.replace('_', ' ')} is {new}.") if new != old else None,
                   CLEAN_CHARTS, f"{key}: {old} -> {new}")

    tenor = display("tenor_years", facts)
    wrong = facts["tenor_years"] + 2
    yield ("number_in_words (known blind spot)",
           with_text(text.replace(tenor, f"{WORDS[int(wrong)]} years", 1)) if tenor in text and wrong in WORDS else None,
           CLEAN_CHARTS, f"tenor_years: {tenor} -> {WORDS.get(int(wrong), '?')} years")

    # a real number attached to the wrong thing: the check only asks "is this a known number?"
    cva_d, epe_d = display("cva", facts), display("peak_epe", facts)
    yield ("right_number_wrong_place (known blind spot)",
           with_text(text.replace(cva_d, epe_d, 1)) if cva_d in text and cva_d != epe_d else None,
           CLEAN_CHARTS, f"cva shown as peak_epe: {cva_d} -> {epe_d}")

    yield ("model_jargon", with_text(text + " The Vasicek model drives this."), CLEAN_CHARTS, "Vasicek")
    yield ("too_long", with_text(text + " " + "Extra words that say very little. " * 40), CLEAN_CHARTS, "")

    s = copy.deepcopy(base); s["key_numbers"][0]["fact_key"] = "made_up"
    yield ("unknown_fact_key", s, CLEAN_CHARTS, "")
    s = copy.deepcopy(base); s["key_numbers"][1]["fact_key"] = s["key_numbers"][0]["fact_key"]
    yield ("duplicate_fact_key", s, CLEAN_CHARTS, "")
    yield ("unknown_chart", base, {"charts": [{"id": "pie_of_doom", "caption": "x"}]}, "")
    yield ("duplicate_chart", base, {"charts": [CLEAN_CHARTS["charts"][0]] * 2}, "")
    for phrase in ADVICE:
        yield ("advice_phrase", with_text(text + " " + phrase), CLEAN_CHARTS, phrase)


def engine_faults(result):
    def mut(fn):
        r = copy.deepcopy(result)
        fn(r)
        return r
    yield "engine cva x1.01", mut(lambda r: r.__setitem__("cva", r["cva"] * 1.01))
    yield "engine cva x1.10", mut(lambda r: r.__setitem__("cva", r["cva"] * 1.10))
    yield "engine epe tampered (cva unchanged)", mut(lambda r: r["epe"].__setitem__(2, r["epe"][2] * 1.5))
    yield "engine negative epe", mut(lambda r: r["epe"].__setitem__(2, -5.0))
    yield "engine hazard x1.05", mut(lambda r: r.__setitem__("hazard", r["hazard"] * 1.05))
    yield "engine time grid out of order", mut(lambda r: r["time_grid"].__setitem__(slice(1, 3), [r["time_grid"][2], r["time_grid"][1]]))
    yield "engine truncated epe", mut(lambda r: r.__setitem__("epe", r["epe"][:-1]))


def run_code_faults(scenarios, engine_fn):
    by = defaultdict(lambda: {"n": 0, "noop": 0, "caught": 0, "missed_examples": []})
    clean = {"n": 0, "flagged": 0}
    for sc in scenarios:
        params, result = sc["params"], engine_fn(sc["params"])
        facts = compute_facts(params, result)
        allowed = allowed_numbers(facts, result, params)

        clean["n"] += 1
        clean["flagged"] += bool(_check(_clean(facts), CLEAN_CHARTS, allowed)
                                 or verifier.verify_engine(params, result))

        for ftype, summary, charts, detail in numeric_and_structural(facts):
            row = by[ftype]
            if summary is None:
                row["noop"] += 1  # perturbation rounded to the same display string
                continue
            row["n"] += 1
            if _check(summary, charts, allowed):
                row["caught"] += 1
            elif len(row["missed_examples"]) < 8:
                row["missed_examples"].append({"scenario": sc["id"], "detail": detail})

        for ftype, broken in engine_faults(result):
            row = by[ftype]
            if broken == result:  # mutation changed nothing (e.g. EPE is 0 at maturity)
                row["noop"] += 1
                continue
            row["n"] += 1
            try:
                row["caught"] += bool(verifier.verify_engine(params, broken))
            except Exception:  # a crash on malformed output also counts as "not accepted"
                row["caught"] += 1
    return {"by_type": dict(by), "clean": clean}


def run_llm_faults(scenarios, engine_fn, llm, k=6):
    by = defaultdict(lambda: {"n": 0, "caught": 0, "unavailable": 0})
    control = {"n": 0, "flagged": 0, "unavailable": 0}
    for i, sc in enumerate(scenarios[:k]):
        params, result = sc["params"], engine_fn(sc["params"])
        facts = compute_facts(params, result)
        charts = ChartsOut.model_validate(CLEAN_CHARTS)

        def review(summary_dict):
            return verifier.llm_review(llm, SummaryOut.model_validate(summary_dict), charts, facts)[1]

        st = review(_clean(facts))
        control["n"] += 1
        control["flagged"] += st == "fail"
        control["unavailable"] += st == "unavailable"

        for ftype, sentences in LLM_FAULTS.items():
            s = _clean(facts)
            s["explanation"] += " " + sentences[i % len(sentences)]
            st = review(s)
            by[ftype]["n"] += 1
            by[ftype]["caught"] += st == "fail"
            by[ftype]["unavailable"] += st == "unavailable"
    return {"by_type": dict(by), "control": control}
