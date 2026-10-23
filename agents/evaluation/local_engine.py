"""Run the real CVA engine in-process (no server), same output shape as POST /cva."""
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[2] / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.cva import run_cva                                       # noqa: E402
from models.swap import generate_payment_schedule, par_swap_rate   # noqa: E402


def engine_local(params: dict) -> dict:
    pt = generate_payment_schedule(params["tenor_years"], params["payment_freq_months"])
    par = par_swap_rate(pt, params["kappa"], params["theta"], params["sigma"], params["r0"], t=0.0)
    fixed = params.get("fixed_rate")
    fixed = par if fixed is None else fixed
    r = run_cva(notional=params["notional"], fixed_rate=fixed, payment_times=pt,
                kappa=params["kappa"], theta=params["theta"], sigma=params["sigma"],
                r0=params["r0"], cds_spread=params["cds_spread"],
                recovery_rate=params.get("recovery_rate", 0.40), n_paths=params["n_paths"],
                seed=params.get("seed"), pfe_quantile=params.get("pfe_quantile", 0.95))
    return {"time_grid": r["time_grid"].tolist(), "epe": r["epe"].tolist(),
            "pfe": r["pfe"].tolist(), "cva": r["cva"], "hazard": r["hazard"],
            "fixed_rate_used": fixed, "par_rate": par}
