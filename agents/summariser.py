"""Summariser agent: explains the CVA result to a non-finance reader."""
import json

from agents.facts import MODEL_LIMITATIONS, display, fact_table
from agents.schemas import SummaryOut, parse

SYSTEM = """You explain the result of a credit valuation adjustment (CVA) calculation to a reader with no finance background.
You are given FACTS. Each fact has a key, a display string and a meaning.

Rules:
1. Every number you write must be copied exactly from a fact's display string. Never calculate, round, convert or estimate a number yourself.
2. Give no advice, no prediction of real losses, and no opinion on whether the deal is good or bad.
3. Use plain words. If you use a term such as EPE, PFE or hazard rate, explain it in the same sentence.
4. Explain what CVA means in one sentence early in the explanation.
5. Choose 3 to 5 fact_keys for key_numbers and give each a short plain label. Do not include the value; the system adds it.
6. caveats: 2 or 3 short sentences based on the MODEL_LIMITATIONS provided.
7. If problems_to_fix is given, fix every one of them and keep the rest of the draft as it was.

Return ONLY a JSON object:
{"headline": string (max 20 words),
 "explanation": string (120 to 200 words, 2 or 3 short paragraphs separated by \\n\\n),
 "key_numbers": [{"fact_key": string, "label": string}],
 "caveats": [string]}"""


def run(llm, facts, question=None, previous=None, problems=None) -> SummaryOut:
    user = {
        "audience_question": question or "What does this result mean?",
        "facts": fact_table(facts),
        "model_limitations": MODEL_LIMITATIONS,
    }
    if problems:
        user["previous_draft"] = previous.model_dump() if previous else None
        user["problems_to_fix"] = problems
    return parse(SummaryOut, llm.complete_json(SYSTEM, json.dumps(user)))


def render(summary: SummaryOut, facts: dict) -> dict:
    """Attach code-formatted values to the chosen key numbers."""
    return {
        "headline": summary.headline,
        "explanation": summary.explanation,
        "key_numbers": [{"fact_key": k.fact_key, "label": k.label,
                         "value": display(k.fact_key, facts)}
                        for k in summary.key_numbers],
        "caveats": summary.caveats,
    }


def fallback(facts: dict) -> dict:
    """Code-written summary used when the agent cannot pass verification."""
    d = lambda k: display(k, facts)  # noqa: E731
    return {
        "headline": f"Estimated cost of counterparty default risk: {d('cva')}",
        "explanation": (
            f"CVA is the estimated cost today of the risk that the other side of the swap "
            f"fails to pay before the swap ends. For this {d('tenor_years')} swap on "
            f"{d('notional')}, it is {d('cva')}, which is {d('cva_pct_notional')} of the notional.\n\n"
            f"The amount the counterparty could owe us peaks at {d('peak_epe')} on average, "
            f"around {d('peak_epe_time')} in."
        ),
        "key_numbers": [{"fact_key": k, "label": lbl, "value": d(k)} for k, lbl in [
            ("cva", "Estimated CVA"), ("cva_pct_notional", "CVA as share of notional"),
            ("peak_epe", "Peak average amount owed"), ("cumulative_default_prob", "Chance of default by the end")]],
        "caveats": MODEL_LIMITATIONS[:3],
    }
