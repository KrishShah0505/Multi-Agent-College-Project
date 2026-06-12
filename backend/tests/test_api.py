"""
Tests for the /cva API endpoint.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from models.swap import generate_payment_schedule, par_swap_rate

client = TestClient(app)


BASE_PAYLOAD = {
    "notional": 1_000_000,
    "tenor_years": 5,
    "payment_freq_months": 6,
    "kappa": 0.3,
    "theta": 0.06,
    "sigma": 0.01,
    "r0": 0.05,
    "cds_spread": 0.015,
    "recovery_rate": 0.40,
    "n_paths": 500,
    "seed": 42,
}


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_cva_endpoint_basic_response_shape():
    resp = client.post("/cva", json=BASE_PAYLOAD)
    assert resp.status_code == 200

    data = resp.json()
    n_steps = 10  # 5 years / 6 months

    assert len(data["time_grid"]) == n_steps + 1
    assert len(data["epe"]) == n_steps + 1
    assert len(data["pfe"]) == n_steps + 1
    assert data["cva"] >= 0.0
    assert data["hazard"] > 0.0


def test_cva_endpoint_defaults_to_par_rate_when_fixed_rate_omitted():
    resp = client.post("/cva", json=BASE_PAYLOAD)
    data = resp.json()

    expected_par_rate = par_swap_rate(
        generate_payment_schedule(5, 6), 0.3, 0.06, 0.01, 0.05, t=0.0
    )

    assert data["par_rate"] == pytest.approx(expected_par_rate)
    assert data["fixed_rate_used"] == pytest.approx(expected_par_rate)


def test_cva_endpoint_respects_explicit_fixed_rate():
    payload = dict(BASE_PAYLOAD, fixed_rate=0.07)
    resp = client.post("/cva", json=payload)
    data = resp.json()

    assert data["fixed_rate_used"] == pytest.approx(0.07)
    # par_rate is still reported, and should differ from the explicit rate
    assert data["par_rate"] != pytest.approx(0.07)


def test_cva_endpoint_at_par_rate_gives_epe_zero_at_inception():
    resp = client.post("/cva", json=BASE_PAYLOAD)
    data = resp.json()
    assert data["epe"][0] == pytest.approx(0.0, abs=1e-3 * BASE_PAYLOAD["notional"])


def test_cva_endpoint_zero_spread_gives_zero_cva():
    payload = dict(BASE_PAYLOAD, cds_spread=0.0)
    resp = client.post("/cva", json=payload)
    data = resp.json()
    assert data["cva"] == pytest.approx(0.0)


def test_cva_endpoint_higher_spread_gives_higher_cva():
    payload_low = dict(BASE_PAYLOAD, cds_spread=0.010)
    payload_high = dict(BASE_PAYLOAD, cds_spread=0.030)

    cva_low = client.post("/cva", json=payload_low).json()["cva"]
    cva_high = client.post("/cva", json=payload_high).json()["cva"]

    assert cva_high > cva_low


def test_cva_endpoint_rejects_non_integer_schedule():
    payload = dict(BASE_PAYLOAD, tenor_years=5, payment_freq_months=7)
    resp = client.post("/cva", json=payload)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Pydantic-level validation (rejected before reaching business logic)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field, bad_value", [
    ("notional", -1000),
    ("tenor_years", 0),
    ("payment_freq_months", 0),
    ("kappa", 0),
    ("kappa", -0.1),
    ("sigma", -0.01),
    ("cds_spread", -0.01),
    ("recovery_rate", 1.0),
    ("recovery_rate", -0.1),
    ("n_paths", 0),
    ("pfe_quantile", 0.0),
    ("pfe_quantile", 1.0),
])
def test_invalid_inputs_return_422(field, bad_value):
    payload = dict(BASE_PAYLOAD)
    payload[field] = bad_value
    resp = client.post("/cva", json=payload)
    assert resp.status_code == 422


def test_reproducibility_with_seed():
    """Same seed -> identical CVA (and EPE/PFE curves)."""
    resp_a = client.post("/cva", json=BASE_PAYLOAD).json()
    resp_b = client.post("/cva", json=BASE_PAYLOAD).json()

    assert resp_a["cva"] == resp_b["cva"]
    assert resp_a["epe"] == resp_b["epe"]