"""
Single-prompt baseline: one LLM call does the whole job (explain + choose charts),
with the same facts, the same rules, no verifier and no retries.
"""
import json

from agents.charts import CATALOG
from agents.facts import MODEL_LIMITATIONS, fact_table
from agents.schemas import ChartsOut, SummaryOut, parse
from agents.summariser import HIDDEN

SYSTEM = """You explain the result of a credit valuation adjustment (CVA) calculation to a reader with no finance background, and you choose which charts to show them.
You are given FACTS. Each fact has a key, a display string and a meaning.

Rules:
1. Every number you write must be copied exactly from a fact's display string. Never calculate, round, convert or estimate a number yourself.
2. Give no advice, no prediction of real losses, and no opinion on whether the deal is good or bad.
3. Use plain words. If you use a term such as EPE, PFE or hazard rate, explain it in the same sentence.
4. Explain what CVA means in one sentence early in the explanation.
5. Choose 3 to 5 fact_keys for key_numbers and give each a short plain label. Do not include the value; the system adds it.
6. caveats: 2 or 3 short sentences based on the MODEL_LIMITATIONS provided.
7. Maximum 150 words in explanation, in 2 short paragraphs. Use at most 6 numbers in total.
8. Do not mention model names (such as Vasicek or Monte Carlo) or technical settings. Write numbers as digits, never as words.
9. Choose 1 to 3 chart ids from the catalog only, in reading order, preferring charts that answer audience_question. Each caption is one plain sentence; any number in it must be copied exactly from a fact's display string.

Return ONLY a JSON object:
{"headline": string (max 20 words),
 "explanation": string (80 to 150 words, 2 short paragraphs separated by \\n\\n),
 "key_numbers": [{"fact_key": string, "label": string}],
 "caveats": [string],
 "charts": [{"id": string, "caption": string}]}"""


def run(llm, facts, question=None):
    user = {
        "audience_question": question or "What does this result mean?",
        "facts": [f for f in fact_table(facts) if f["key"] not in HIDDEN],
        "model_limitations": MODEL_LIMITATIONS,
        "catalog": [{"id": k, "use_when": v["use_when"]} for k, v in CATALOG.items()],
    }
    raw = llm.complete_json(SYSTEM, json.dumps(user))
    summary = parse(SummaryOut, raw)
    charts = parse(ChartsOut, {"charts": raw.get("charts") if isinstance(raw, dict) else None})
    return summary, charts
