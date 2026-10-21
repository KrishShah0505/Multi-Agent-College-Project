"""Graph agent: chooses which charts to show and writes a caption for each."""
import json

from agents.charts import CATALOG, build_chart
from agents.facts import fact_table
from agents.schemas import ChartsOut, parse

SYSTEM = """You choose which charts to show a reader with no finance background, from a fixed catalog, and write a one-sentence caption for each.

Rules:
1. Pick 1 to 3 chart ids from the catalog only, in the order they should be read. Prefer charts that answer audience_question.
2. Each caption is one plain sentence saying what to look at in the chart.
3. Any number in a caption must be copied exactly from a fact's display string. Never calculate or estimate a number.
4. No advice and no predictions of real losses.
5. If problems_to_fix is given, fix every one of them.

Return ONLY a JSON object: {"charts": [{"id": string, "caption": string}]}"""


def run(llm, facts, question=None, previous=None, problems=None) -> ChartsOut:
    user = {
        "audience_question": question or "What does this result mean?",
        "catalog": [{"id": k, "use_when": v["use_when"]} for k, v in CATALOG.items()],
        "facts": fact_table(facts),
    }
    if problems:
        user["previous_choice"] = previous.model_dump() if previous else None
        user["problems_to_fix"] = problems
    return parse(ChartsOut, llm.complete_json(SYSTEM, json.dumps(user)))


def render(picks: ChartsOut, params: dict, result: dict) -> list[dict]:
    out = []
    for p in picks.charts:
        chart = build_chart(p.id, params, result)
        chart["caption"] = p.caption
        out.append(chart)
    return out


def fallback(params: dict, result: dict) -> list[dict]:
    caps = {"exposure_profile": "The blue line is the average amount the counterparty could owe us; the dashed line is a bad case.",
            "cva_contribution": "Each bar shows how much a period adds to the final CVA figure."}
    out = []
    for cid, cap in caps.items():
        chart = build_chart(cid, params, result)
        chart["caption"] = cap
        out.append(chart)
    return out
