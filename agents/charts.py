"""
Chart catalog. Each chart is built by code from the engine output; the graph
agent only chooses which ids to show and writes captions.
"""
import math

from agents.facts import cva_contributions, marginal_pds


def exposure_profile(params, result):
    tg = result["time_grid"]
    return {
        "id": "exposure_profile",
        "title": "How much the counterparty could owe us over time",
        "x_label": "Years from today",
        "y_label": "Dollars",
        "series": [{"key": "epe", "label": "Average (EPE)"},
                   {"key": "pfe", "label": "Bad case (PFE)"}],
        "data": [{"t": tg[i], "epe": result["epe"][i], "pfe": result["pfe"][i]}
                 for i in range(len(tg))],
    }


def default_probability(params, result):
    tg, lam = result["time_grid"], result["hazard"]
    pds = marginal_pds(tg, lam)
    return {
        "id": "default_probability",
        "title": "Chance the counterparty defaults",
        "x_label": "Years from today",
        "y_label": "Probability",
        "series": [{"key": "marginal_pd", "label": "Within this period"},
                   {"key": "cumulative_pd", "label": "By this date (cumulative)"}],
        "data": [{"t": tg[i + 1], "marginal_pd": pds[i],
                  "cumulative_pd": 1.0 - math.exp(-lam * tg[i + 1])}
                 for i in range(len(pds))],
    }


def cva_contribution(params, result):
    tg = result["time_grid"]
    c = cva_contributions(tg, result["epe"], result["hazard"],
                          params.get("recovery_rate", 0.40))
    total = sum(c)
    running, rows = 0.0, []
    for i, v in enumerate(c):
        running += v
        rows.append({"t": tg[i + 1], "contribution": v, "cumulative": running,
                     "share": (v / total) if total > 0 else 0.0})
    return {
        "id": "cva_contribution",
        "title": "Where the CVA comes from",
        "x_label": "Period ending (years from today)",
        "y_label": "Dollars of CVA",
        "series": [{"key": "contribution", "label": "CVA added in this period"},
                   {"key": "cumulative", "label": "Running total"}],
        "data": rows,
    }


CATALOG = {
    "exposure_profile": {
        "builder": exposure_profile,
        "use_when": "Show how much could be owed over the life of the swap, average vs bad case. Best for questions about risk size or timing.",
    },
    "default_probability": {
        "builder": default_probability,
        "use_when": "Show how likely the counterparty is to default, per period and cumulatively. Best for questions about counterparty credit quality.",
    },
    "cva_contribution": {
        "builder": cva_contribution,
        "use_when": "Show which periods add most to the final CVA number. Best for 'why is CVA this size' questions.",
    },
}


def build_chart(chart_id, params, result):
    return CATALOG[chart_id]["builder"](params, result)
