"""
Verifier agent.

Layer 1 (code, deterministic): engine consistency, every number in the text
traced back to a fact, fact keys and chart ids valid, banned phrasing.
Layer 2 (LLM): wording review for unsupported claims, advice, unexplained jargon.
Numbers are never checked by the LLM.
"""
import json
import math
import re
from dataclasses import dataclass

from agents.charts import CATALOG
from agents.facts import FACT_META, cva_contributions, fact_table
from agents.llm import LLMError, LLMFatalError
from agents.schemas import AgentOutputError, ReviewOut, SummaryOut, ChartsOut, parse


@dataclass
class Issue:
    source: str   # "engine" | "summary" | "charts"
    code: str
    message: str

    def as_dict(self):
        return {"source": self.source, "code": self.code, "message": self.message}


# ---------------------------------------------------------------- engine ----
def verify_engine(params: dict, result: dict) -> list[Issue]:
    """Sanity-check the engine output; agents cannot fix these."""
    iss = []
    tg, epe, pfe = result["time_grid"], result["epe"], result["pfe"]
    if not (len(tg) == len(epe) == len(pfe)):
        return [Issue("engine", "length_mismatch", "time_grid, epe and pfe differ in length.")]
    if tg[0] != 0 or any(b <= a for a, b in zip(tg, tg[1:])):
        iss.append(Issue("engine", "bad_time_grid", "time_grid must start at 0 and increase."))
    if any(x < 0 or not math.isfinite(x) for x in list(epe) + list(pfe)):
        iss.append(Issue("engine", "negative_exposure", "EPE/PFE contain negative or non-finite values."))
    if not math.isfinite(result["cva"]) or result["cva"] < 0:
        iss.append(Issue("engine", "bad_cva", "CVA is negative or non-finite."))
    R = params.get("recovery_rate", 0.40)
    if abs(result["hazard"] - params["cds_spread"] / (1 - R)) > 1e-9:
        iss.append(Issue("engine", "hazard_mismatch", "hazard != cds_spread / (1 - recovery)."))
    recomputed = sum(cva_contributions(tg, epe, result["hazard"], R))
    if abs(recomputed - result["cva"]) > 1e-6 * max(1.0, abs(result["cva"])):
        iss.append(Issue("engine", "cva_not_reproducible",
                         f"CVA recomputed from EPE and hazard is {recomputed:.6f}, engine says {result['cva']:.6f}."))
    return iss


# --------------------------------------------------------------- numbers ----
_NUM = re.compile(
    r"(?P<cur>\$)?\s?(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?P<dec>\.\d+)?"
    r"\s?(?P<suf>%|bps\b|basis points?|billion|million|thousand|years?\b|yrs?\b)?",
    re.I)
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9}


def extract_numbers(text: str):
    """Yield (token, value, decimals, unit, scale) for each number in text."""
    for m in _NUM.finditer(text):
        value = float(m.group("int").replace(",", "") + (m.group("dec") or ""))
        decimals = len(m.group("dec")) - 1 if m.group("dec") else 0
        suf = (m.group("suf") or "").lower()
        scale = _SCALE.get(suf, 1.0)
        if m.group("cur"):
            unit = "usd"
        elif suf == "%":
            unit = "pct"
        elif suf.startswith("bp") or suf.startswith("basis"):
            unit = "bps"
        elif suf.startswith("y"):
            unit = "years"
        else:
            unit = "bare"
        yield m.group(0).strip(), value, decimals, unit, scale


def number_supported(value, decimals, unit, scale, allowed) -> bool:
    if unit == "usd":
        cands = [c / scale for c in allowed["usd"]]
    elif unit == "pct":
        cands = [c * 100 for c in allowed["pct"]]
    elif unit == "bps":
        cands = [c * 1e4 for c in allowed["pct"]]
    elif unit == "years":
        cands = list(allowed["years"])
    else:  # a bare number may legitimately belong to any group
        cands = (allowed["usd"] + [c * 100 for c in allowed["pct"]] + allowed["pct"]
                 + allowed["years"] + allowed["num"])
    tol = 0.5 * 10 ** (-decimals) + 1e-9   # "0.05" is a fair rounding of 0.0523
    return any(abs(c - value) <= tol for c in cands)


def check_numbers(text: str, allowed: dict, source: str, where: str) -> list[Issue]:
    return [Issue(source, "unsupported_number",
                  f"'{tok}' in {where} does not match any provided fact (copy numbers exactly from the facts).")
            for tok, v, d, u, s in extract_numbers(text)
            if not number_supported(v, d, u, s, allowed)]


_BANNED = re.compile(
    r"\b(you should|we recommend|i recommend|should (buy|sell|hedge|avoid)|will lose|guaranteed|"
    r"good deal|bad deal|safe investment)\b", re.I)
_TECH = re.compile(r"\b(vasicek|monte[- ]carlo|risk[- ]neutral|stochastic|mean[- ]revers\w*)\b", re.I)
MAX_WORDS, MAX_NUMBERS = 200, 8


# ----------------------------------------------------------- code checks ----
def verify_summary(summary: SummaryOut, allowed: dict) -> list[Issue]:
    iss = []
    keys = [k.fact_key for k in summary.key_numbers]
    for k in keys:
        if k not in FACT_META:
            iss.append(Issue("summary", "unknown_fact_key", f"key_numbers uses unknown fact_key '{k}'."))
    if len(set(keys)) != len(keys):
        iss.append(Issue("summary", "duplicate_fact_key", "key_numbers repeats a fact_key."))
    parts = [("headline", summary.headline), ("explanation", summary.explanation)]
    parts += [(f"caveat {i + 1}", c) for i, c in enumerate(summary.caveats)]
    parts += [(f"label of {k.fact_key}", k.label) for k in summary.key_numbers]
    words = len(summary.explanation.split())
    if words > MAX_WORDS:
        iss.append(Issue("summary", "too_long", f"explanation is {words} words; keep it under 150."))
    n_nums = sum(1 for _ in extract_numbers(summary.explanation))
    if n_nums > MAX_NUMBERS:
        iss.append(Issue("summary", "too_many_numbers", f"explanation contains {n_nums} numbers; use at most 6 and leave detail to the key numbers."))
    for where, text in parts:
        m = _TECH.search(text)
        if m:
            iss.append(Issue("summary", "technical_term", f"'{m.group(0)}' in {where}: do not mention model names; describe assumptions in plain words."))
        iss += check_numbers(text, allowed, "summary", where)
        m = _BANNED.search(text)
        if m:
            iss.append(Issue("summary", "banned_phrase", f"'{m.group(0)}' in {where}: no advice or loss predictions."))
    return iss


def verify_charts(charts: ChartsOut, allowed: dict) -> list[Issue]:
    iss, ids = [], [c.id for c in charts.charts]
    for cid in ids:
        if cid not in CATALOG:
            iss.append(Issue("charts", "unknown_chart", f"'{cid}' is not in the catalog."))
    if len(set(ids)) != len(ids):
        iss.append(Issue("charts", "duplicate_chart", "Same chart chosen twice."))
    for i, c in enumerate(charts.charts):
        iss += check_numbers(c.caption, allowed, "charts", f"caption {i + 1}")
        m = _BANNED.search(c.caption)
        if m:
            iss.append(Issue("charts", "banned_phrase", f"'{m.group(0)}' in caption {i + 1}."))
    return iss


# ------------------------------------------------------------ LLM review ----
REVIEW_SYSTEM = """You are a strict fact-checker for plain-language explanations of a CVA (credit valuation adjustment) result for non-finance readers.
You get FACTS and a DRAFT (summary and chart captions). Report problems of these kinds only:
- a claim that the facts do not support
- advice, an opinion on the deal, or a prediction of actual losses
- a wrong or misleading description of what CVA, EPE, PFE or the hazard rate mean
- financial jargon used without a plain explanation
Do NOT check numbers; another system does that. Do not nitpick style. If there are no real problems, pass.
Return ONLY JSON: {"pass": boolean, "issues": [{"target": "summary" or "charts", "message": string}]}"""


def build_review_user(summary: SummaryOut, charts: ChartsOut, facts: dict) -> dict:
    return {"facts": fact_table(facts),
            "draft": {"summary": summary.model_dump(), "charts": charts.model_dump()}}


def llm_review(llm, summary: SummaryOut, charts: ChartsOut, facts: dict):
    """Returns (issues, status) where status is 'pass', 'fail' or 'unavailable'."""
    user = json.dumps(build_review_user(summary, charts, facts))
    try:
        out = parse(ReviewOut, llm.complete_json(REVIEW_SYSTEM, user))
    except LLMFatalError:
        raise
    except (LLMError, AgentOutputError):
        return [], "unavailable"
    if out.pass_ and not out.issues:
        return [], "pass"
    issues = [Issue(i.target if i.target in ("summary", "charts") else "summary",
                    "llm_review", i.message) for i in out.issues]
    if not issues:  # model said fail but gave no reasons: do not loop on nothing
        return [], "pass"
    return issues, "fail"
