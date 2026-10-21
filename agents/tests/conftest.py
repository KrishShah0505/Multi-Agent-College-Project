"""Real engine output for tests: imports the backend directly (no server needed)."""
import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from core.cva import run_cva                                   # noqa: E402
from models.swap import generate_payment_schedule, par_swap_rate  # noqa: E402

PARAMS = {"notional": 1_000_000, "tenor_years": 5, "payment_freq_months": 6,
          "kappa": 0.3, "theta": 0.06, "sigma": 0.01, "r0": 0.05,
          "cds_spread": 0.015, "recovery_rate": 0.40, "n_paths": 5000,
          "seed": 42, "pfe_quantile": 0.95}


def engine(params):
    """Same shape as the API's JSON response."""
    pt = generate_payment_schedule(params["tenor_years"], params["payment_freq_months"])
    par = par_swap_rate(pt, params["kappa"], params["theta"], params["sigma"], params["r0"], t=0.0)
    fixed = params.get("fixed_rate") if params.get("fixed_rate") is not None else par
    r = run_cva(notional=params["notional"], fixed_rate=fixed, payment_times=pt,
                kappa=params["kappa"], theta=params["theta"], sigma=params["sigma"],
                r0=params["r0"], cds_spread=params["cds_spread"],
                recovery_rate=params["recovery_rate"], n_paths=params["n_paths"],
                seed=params["seed"], pfe_quantile=params["pfe_quantile"])
    return {"time_grid": r["time_grid"].tolist(), "epe": r["epe"].tolist(),
            "pfe": r["pfe"].tolist(), "cva": r["cva"], "hazard": r["hazard"],
            "fixed_rate_used": fixed, "par_rate": par}


@pytest.fixture(scope="session")
def params():
    return dict(PARAMS)


@pytest.fixture(scope="session")
def result(params):
    return engine(params)


@pytest.fixture()
def fresh_result(result):
    return copy.deepcopy(result)


class FakeLLM:
    """Scripted LLM: handler(system, user_json) -> dict. Counts calls."""
    def __init__(self, handler):
        self.handler, self.calls, self.tokens = handler, 0, 0

    def complete_json(self, system, user):
        import json
        self.calls += 1
        return self.handler(system, json.loads(user))
