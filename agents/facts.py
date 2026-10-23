"""
Deterministic facts derived from the CVA engine output.

Everything numeric the agents may talk about comes from here, formatted by
code. The LLMs copy display strings; they never calculate.
"""
import math

# key -> (unit, decimals, plain-English meaning)
FACT_META = {
    "cva": ("usd", 0, "CVA: estimated cost today of the risk that the counterparty defaults before the swap ends"),
    "cva_pct_notional": ("pct", 4, "CVA as a share of the swap's notional (face amount)"),
    "notional": ("usd", 0, "Notional: the face amount the swap's payments are calculated on"),
    "tenor_years": ("years", 2, "Time until the swap ends"),
    "fixed_rate": ("pct", 3, "Fixed interest rate on the swap"),
    "par_rate": ("pct", 3, "Fixed rate that would make the swap worth exactly zero at the start"),
    "cds_spread": ("pct", 2, "Counterparty's CDS spread: the yearly market price of insuring against its default"),
    "recovery_rate": ("pct", 0, "Share of the amount owed assumed to be recovered if the counterparty defaults"),
    "hazard": ("pct", 3, "Implied yearly default intensity of the counterparty (hazard rate)"),
    "cumulative_default_prob": ("pct", 2, "Model chance that the counterparty defaults at some point before the swap ends"),
    "peak_epe": ("usd", 0, "Highest average amount the counterparty would owe us at any date (peak EPE)"),
    "peak_epe_time": ("years", 2, "When the average amount owed peaks"),
    "peak_pfe": ("usd", 0, "Highest bad-case amount the counterparty could owe us at any date (peak PFE)"),
    "peak_pfe_time": ("years", 2, "When the bad-case amount owed peaks"),
    "pfe_quantile": ("pct", 0, "Confidence level used for the bad-case (PFE) estimate"),
    "peak_contribution_time": ("years", 2, "End of the period that adds the most to CVA"),
    "peak_contribution_share": ("pct", 0, "Share of total CVA coming from that single biggest period"),
    "n_paths": ("num", 0, "Number of simulated interest-rate scenarios"),
    "sigma": ("pct", 2, "Interest-rate volatility assumed in the model"),
    "kappa": ("num", 2, "Speed at which rates are assumed to drift back to their long-run level"),
    "theta": ("pct", 2, "Long-run interest rate assumed in the model"),
    "r0": ("pct", 2, "Starting short-term interest rate"),
}

MODEL_LIMITATIONS = [
    "Counterparty default risk is assumed constant over time, based on a single CDS spread.",
    "Interest rates are simulated with a simple one-factor model, which can produce negative rates.",
    "Only CVA is measured; other valuation adjustments (own default, funding, capital) are not.",
]


def marginal_pds(time_grid, hazard):
    """P(default in (t_{i-1}, t_i]) = exp(-L t_{i-1}) - exp(-L t_i)."""
    return [math.exp(-hazard * time_grid[i - 1]) - math.exp(-hazard * time_grid[i])
            for i in range(1, len(time_grid))]


def cva_contributions(time_grid, epe, hazard, recovery):
    pds = marginal_pds(time_grid, hazard)
    return [(1.0 - recovery) * epe[i + 1] * pds[i] for i in range(len(pds))]


def compute_facts(params: dict, result: dict) -> dict:
    tg, epe, pfe = result["time_grid"], result["epe"], result["pfe"]
    recovery = params.get("recovery_rate", 0.40)
    notional = params["notional"]
    contribs = cva_contributions(tg, epe, result["hazard"], recovery)
    total = sum(contribs)
    i_epe = max(range(len(epe)), key=lambda i: epe[i])
    i_pfe = max(range(len(pfe)), key=lambda i: pfe[i])
    i_con = max(range(len(contribs)), key=lambda i: contribs[i])
    return {
        "cva": result["cva"],
        "cva_pct_notional": result["cva"] / notional,
        "notional": notional,
        "tenor_years": params["tenor_years"],
        "fixed_rate": result["fixed_rate_used"],
        "par_rate": result["par_rate"],
        "cds_spread": params["cds_spread"],
        "recovery_rate": recovery,
        "hazard": result["hazard"],
        "cumulative_default_prob": 1.0 - math.exp(-result["hazard"] * tg[-1]),
        "peak_epe": epe[i_epe],
        "peak_epe_time": tg[i_epe],
        "peak_pfe": pfe[i_pfe],
        "peak_pfe_time": tg[i_pfe],
        "pfe_quantile": params.get("pfe_quantile", 0.95),
        "peak_contribution_time": tg[i_con + 1],
        "peak_contribution_share": (contribs[i_con] / total) if total > 0 else 0.0,
        "n_paths": params.get("n_paths", 1000),
        "sigma": params["sigma"],
        "kappa": params["kappa"],
        "theta": params["theta"],
        "r0": params["r0"],
    }


def display(key: str, facts: dict) -> str:
    unit, d, _ = FACT_META[key]
    v = facts[key]
    if unit == "usd":
        d = 2 if abs(v) < 100 else d   # keep resolution for very small amounts
        return f"${v:,.{d}f}"
    if unit == "pct":
        return f"{v * 100:.{d}f}%"
    if unit == "years":
        s = f"{v:.{d}f}".rstrip("0").rstrip(".") if d else f"{v:.0f}"
        return f"{s} year" if s == "1" else f"{s} years"
    return f"{v:.{d}f}"


def fact_table(facts: dict) -> list[dict]:
    """What the LLM sees: key, display string, meaning."""
    return [{"key": k, "display": display(k, facts), "meaning": FACT_META[k][2]}
            for k in FACT_META]


def allowed_numbers(facts: dict, result: dict, params: dict) -> dict:
    """Every number an explanation may legitimately mention, grouped by unit."""
    recovery = params.get("recovery_rate", 0.40)
    contribs = cva_contributions(result["time_grid"], result["epe"],
                                 result["hazard"], recovery)
    by_unit = {"usd": [], "pct": [], "years": [], "num": []}
    for k, (unit, _, _) in FACT_META.items():
        by_unit[unit].append(facts[k])
    by_unit["usd"] += list(result["epe"]) + list(result["pfe"]) + contribs
    by_unit["years"] += list(result["time_grid"])
    return by_unit
