"""Turn raw results into markdown tables, plus a blind spot-check sheet for human rating."""
import random
from collections import defaultdict

from agents.evaluation.stats import mean, median, rate


def _fmt(x, d=1):
    return "n/a" if x is None else f"{x:.{d}f}"


def faults_report(data: dict) -> str:
    out = ["## Fault injection: does the verifier notice?", ""]
    c = data["code"]
    out += [f"False alarms on known-good output (code layer): {rate(c['clean']['flagged'], c['clean']['n'])}", "",
            "| Fault | Injected | No-op (rounded to same text) | Caught |", "|---|---|---|---|"]
    for ftype in sorted(c["by_type"]):
        r = c["by_type"][ftype]
        out.append(f"| {ftype} | {r['n']} | {r['noop']} | {rate(r['caught'], r['n'])} |")
    misses = [(t, m) for t, r in sorted(c["by_type"].items()) for m in r["missed_examples"]]
    if misses:
        out += ["", "Examples of faults that were NOT caught:", ""]
        out += [f"- {t} ({m['scenario']}): {m['detail']}" for t, m in misses[:24]]
    if data.get("llm"):
        l = data["llm"]
        out += ["", "### LLM wording review (number-free faults only a reader can catch)", "",
                f"False alarms on clean text: {rate(l['control']['flagged'], l['control']['n'])}"
                f" (review unavailable: {l['control']['unavailable']})", "",
                "| Fault | Injected | Caught | Review unavailable |", "|---|---|---|---|"]
        for ftype, r in sorted(l["by_type"].items()):
            out.append(f"| {ftype} | {r['n']} | {rate(r['caught'], r['n'])} | {r['unavailable']} |")
    return "\n".join(out)


def _agg(recs):
    ok = [r for r in recs if not r.get("error")]
    n = len(ok)
    valid = [r for r in ok if not r["first_draft"]["invalid_output"]]
    judged = [r for r in valid if r["first_draft"]["review"] in ("pass", "fail")]
    return {
        "n": n, "errors": len(recs) - n,
        "invalid": rate(sum(r["first_draft"]["invalid_output"] for r in ok), n),
        "num1": rate(sum(r["first_draft"]["numeric"] > 0 for r in valid), len(valid)),
        "oth1": rate(sum(r["first_draft"]["other"] > 0 for r in valid), len(valid)),
        "rev1": rate(sum(r["first_draft"]["review"] == "fail" for r in judged), len(judged)) if judged else "not measured",
        "numD": rate(sum(r["delivered"]["numeric"] > 0 for r in ok), n),
        "othD": rate(sum(r["delivered"]["other"] > 0 for r in ok), n),
        "retry": rate(sum(r["attempts"] >= 2 for r in ok), n),
        "attempts": mean([r["attempts"] for r in ok]),
        "calls": mean([r["llm_calls"] for r in ok]), "tokens": mean([r["tokens"] for r in ok]),
        "sec_mean": mean([r["seconds"] for r in ok]), "sec_med": median([r["seconds"] for r in ok]),
        "status": {s: sum(r["status"] == s for r in ok) for s in sorted({r["status"] for r in ok})},
    }


def compare_report(records) -> str:
    P = _agg([r for r in records if r.get("system") == "pipeline"])
    B = _agg([r for r in records if r.get("system") == "baseline"])
    rows = [
        ("Runs completed (errors)", f"{P['n']} ({P['errors']})", f"{B['n']} ({B['errors']})"),
        ("First draft: unusable JSON/schema", P["invalid"], B["invalid"]),
        ("First draft: has a wrong number", P["num1"], B["num1"]),
        ("First draft: other rule break (bad key/id, advice phrase)", P["oth1"], B["oth1"]),
        ("First draft: wording flagged by reviewer", P["rev1"], B["rev1"]),
        ("DELIVERED to user: has a wrong number", P["numD"], B["numD"]),
        ("DELIVERED to user: other rule break", P["othD"], B["othD"]),
        ("Needed more than one attempt", P["retry"], "n/a"),
        ("Mean attempts", _fmt(P["attempts"], 2), "1.00"),
        ("Mean model calls per query", _fmt(P["calls"]), _fmt(B["calls"])),
        ("Mean tokens per query", _fmt(P["tokens"], 0), _fmt(B["tokens"], 0)),
        ("Seconds per query (mean / median)", f"{_fmt(P['sec_mean'])} / {_fmt(P['sec_med'])}",
         f"{_fmt(B['sec_mean'])} / {_fmt(B['sec_med'])}"),
    ]
    out = ["## Pipeline vs single-prompt baseline", "",
           "| Metric | 3-agent pipeline | Single prompt |", "|---|---|---|"]
    out += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    out += ["", f"Pipeline final status: {P['status']}", ""]

    # what the verifier bought, and consistency across repeats
    pipe = [r for r in records if r.get("system") == "pipeline" and not r.get("error")]
    fixed = sum(1 for r in pipe if (r["first_draft"]["numeric"] or r["first_draft"]["other"]
                                    or r["first_draft"]["review"] == "fail" or r["first_draft"]["invalid_output"])
                and r["status"] in ("verified", "partially_verified"))
    fell = sum(1 for r in pipe if r["status"] == "fallback")
    out += [f"Runs where the first draft had a problem and a retry fixed it: {fixed}/{len(pipe)}",
            f"Runs that ended in the code-written fallback: {fell}/{len(pipe)}", ""]

    by_sc = defaultdict(list)
    for r in pipe:
        by_sc[r["scenario"]].append(r)
    multi = {k: v for k, v in by_sc.items() if len(v) >= 2}
    if multi:
        same = sum(len({r["status"] for r in v}) == 1 for v in multi.values())
        flip = sum(len({(r["first_draft"]["review"] == "fail" or r["first_draft"]["numeric"] > 0) for r in v}) > 1
                   for v in multi.values())
        out += ["### Consistency (same input, repeated)", "",
                f"Scenarios with the same final status every time: {rate(same, len(multi))}",
                f"Scenarios where the first draft passed in some runs and failed in others: {rate(flip, len(multi))}", ""]

    out += ["Notes: the wrong-number grader is the same code check the pipeline enforces, so 'delivered' "
            "numbers for the pipeline are low by construction. The fair comparison is the first-draft row. "
            "Wording is only graded by an LLM reviewer; use the spot-check sheet for human ratings."]
    return "\n".join(out)


def spot_check(records, n=5, seed=7) -> str:
    rng = random.Random(seed)
    by = defaultdict(dict)
    for r in records:
        if r.get("text") and r["repeat"] == 0:
            by[r["scenario"]][r["system"]] = r
    pairs = [(k, v) for k, v in sorted(by.items()) if len(v) == 2]
    rng.shuffle(pairs)
    out = ["# Blind spot-check", "",
           "Rate each output 1-5 for: plain language, accuracy against the numbers, usefulness of charts. "
           "Which system wrote which is revealed at the bottom.", ""]
    key = []
    for sid, v in pairs[:n]:
        order = ["pipeline", "baseline"]
        rng.shuffle(order)
        out += [f"## {sid} - question: {v['pipeline']['question'] or '(none)'}", ""]
        for label, system in zip("AB", order):
            t = v[system]["text"]
            out += [f"### Output {label}", f"**{t['headline']}**", "", t["explanation"], "",
                    "Caveats: " + " | ".join(t["caveats"]),
                    "Charts: " + ", ".join(f"{i} ({c})" for i, c in zip(t["chart_ids"], t["captions"])),
                    "", "Plain language __/5   Accuracy __/5   Charts __/5", ""]
            key.append(f"{sid} {label} = {system}")
    out += ["---", "Key (do not read before rating): " + "; ".join(key)]
    return "\n".join(out)
