"""
Verification tests for core/cva.py.

Strategy:
1. Consistency: price_swap_vectorized must match models.swap.price_swap
   for scalar inputs -- this catches bugs introduced by vectorization.
2. Structural sanity: exposure >= 0 everywhere, exposure == 0 at maturity.
3. EPE(0) ~ 0 when fixed_rate = par rate (Phase 1 step 4 sanity check).
4. Deterministic case (sigma=0, r0=theta): EPE matches the closed-form
   swap value exactly, since every path is identical (Phase 4 check,
   done early here because it's the strongest correctness signal).
5. compute_cva limiting behavior: hazard=0 -> CVA=0; hazard->inf -> CVA
   concentrates on the first interval's exposure.
6. run_cva integration test + Monte Carlo stability across seeds.
"""

import numpy as np
import pytest

from core.cva import (
    compute_cva,
    compute_epe_pfe,
    exposure_matrix,
    price_swap_vectorized,
    run_cva,
)
from models.credit import hazard_rate
from models.swap import par_swap_rate, price_swap
from models.vasicek import simulate_paths


NOTIONAL = 1_000_000.0
PAYMENT_TIMES = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
TIME_GRID = np.concatenate(([0.0], PAYMENT_TIMES))

KAPPA, THETA, SIGMA, R0 = 0.3, 0.06, 0.01, 0.05


# ---------------------------------------------------------------------------
# 1. price_swap_vectorized vs price_swap consistency
# ---------------------------------------------------------------------------

def test_vectorized_price_matches_scalar_price():
    fixed_rate = par_swap_rate(PAYMENT_TIMES, KAPPA, THETA, SIGMA, R0, t=0.0)

    r_values = np.array([0.02, 0.05, 0.06, 0.09, 0.12])

    vectorized = price_swap_vectorized(
        r_values, t=0.0, fixed_rate=fixed_rate, payment_times=PAYMENT_TIMES,
        notional=NOTIONAL, kappa=KAPPA, theta=THETA, sigma=SIGMA,
    )

    scalar = np.array([
        price_swap(r, 0.0, fixed_rate, PAYMENT_TIMES, NOTIONAL, KAPPA, THETA, SIGMA)
        for r in r_values
    ])

    assert np.allclose(vectorized, scalar)


def test_vectorized_price_matches_scalar_price_mid_schedule():
    """Same check, but at t > 0 where the residual schedule is shorter."""
    fixed_rate = par_swap_rate(PAYMENT_TIMES, KAPPA, THETA, SIGMA, R0, t=0.0)
    t = 2.0  # 4 payments remain: 2.5, 3.0, 3.5, 4.0, 4.5, 5.0 -> wait, >2.0

    r_values = np.array([0.03, 0.05, 0.07])

    vectorized = price_swap_vectorized(
        r_values, t=t, fixed_rate=fixed_rate, payment_times=PAYMENT_TIMES,
        notional=NOTIONAL, kappa=KAPPA, theta=THETA, sigma=SIGMA,
    )
    scalar = np.array([
        price_swap(r, t, fixed_rate, PAYMENT_TIMES, NOTIONAL, KAPPA, THETA, SIGMA)
        for r in r_values
    ])

    assert np.allclose(vectorized, scalar)


def test_vectorized_price_after_maturity_is_zero_array():
    r_values = np.array([0.02, 0.05, 0.09])
    V = price_swap_vectorized(
        r_values, t=5.0, fixed_rate=0.06, payment_times=PAYMENT_TIMES,
        notional=NOTIONAL, kappa=KAPPA, theta=THETA, sigma=SIGMA,
    )
    assert np.array_equal(V, np.zeros_like(r_values))


# ---------------------------------------------------------------------------
# 2. exposure_matrix structural sanity
# ---------------------------------------------------------------------------

def test_exposure_matrix_non_negative():
    fixed_rate = par_swap_rate(PAYMENT_TIMES, KAPPA, THETA, SIGMA, R0, t=0.0)
    rate_paths = simulate_paths(500, len(PAYMENT_TIMES), 0.5, R0, KAPPA, THETA, SIGMA, seed=1)

    exposure = exposure_matrix(
        rate_paths, TIME_GRID, fixed_rate, PAYMENT_TIMES, NOTIONAL, KAPPA, THETA, SIGMA
    )

    assert np.all(exposure >= 0.0)


def test_exposure_zero_at_maturity():
    fixed_rate = par_swap_rate(PAYMENT_TIMES, KAPPA, THETA, SIGMA, R0, t=0.0)
    rate_paths = simulate_paths(500, len(PAYMENT_TIMES), 0.5, R0, KAPPA, THETA, SIGMA, seed=1)

    exposure = exposure_matrix(
        rate_paths, TIME_GRID, fixed_rate, PAYMENT_TIMES, NOTIONAL, KAPPA, THETA, SIGMA
    )

    # last column corresponds to t=5.0, the final payment date -> no remaining cashflows
    assert np.allclose(exposure[:, -1], 0.0)


# ---------------------------------------------------------------------------
# 3. EPE(0) ~ 0 when fixed_rate is the par rate
# ---------------------------------------------------------------------------

def test_epe_at_inception_is_near_zero_at_par_rate():
    """
    At t=0, every path starts at r0, so V_swap(0) is identical across all
    paths. If fixed_rate = par_rate(r0), V_swap(0) ~ 0 -> EPE(0) ~ 0.
    """
    fixed_rate = par_swap_rate(PAYMENT_TIMES, KAPPA, THETA, SIGMA, R0, t=0.0)
    rate_paths = simulate_paths(200, len(PAYMENT_TIMES), 0.5, R0, KAPPA, THETA, SIGMA, seed=2)

    exposure = exposure_matrix(
        rate_paths, TIME_GRID, fixed_rate, PAYMENT_TIMES, NOTIONAL, KAPPA, THETA, SIGMA
    )
    epe, _ = compute_epe_pfe(exposure)

    assert epe[0] == pytest.approx(0.0, abs=1e-6 * NOTIONAL)


# ---------------------------------------------------------------------------
# 4. Deterministic closed-form check (sigma=0, r0=theta)
# ---------------------------------------------------------------------------

def test_deterministic_case_matches_closed_form_swap_value():
    """
    If sigma=0 and r0=theta, the Vasicek SDE has zero drift and zero noise
    at r=theta -- every path stays at r=theta forever, deterministically.

    In that case EPE(t_j) must exactly equal max(price_swap(theta, t_j,
    ...), 0), computed directly via the scalar pricer. This is the
    strongest end-to-end correctness check available without external
    data: it ties simulate_paths -> exposure_matrix -> compute_epe_pfe
    back to the already-verified scalar price_swap.
    """
    r0_det = THETA  # = 0.06
    sigma_det = 0.0
    fixed_rate = par_swap_rate(PAYMENT_TIMES, KAPPA, THETA, sigma_det, r0_det, t=0.0)

    rate_paths = simulate_paths(
        10, len(PAYMENT_TIMES), 0.5, r0_det, KAPPA, THETA, sigma_det, seed=3
    )

    # sanity: all paths identical and equal to theta
    assert np.allclose(rate_paths, THETA)

    exposure = exposure_matrix(
        rate_paths, TIME_GRID, fixed_rate, PAYMENT_TIMES, NOTIONAL, KAPPA, THETA, sigma_det
    )
    epe, pfe = compute_epe_pfe(exposure)

    expected = np.array([
        max(price_swap(THETA, t, fixed_rate, PAYMENT_TIMES, NOTIONAL, KAPPA, THETA, sigma_det), 0.0)
        for t in TIME_GRID
    ])

    assert np.allclose(epe, expected)
    # with zero variance, EPE and PFE must coincide exactly
    assert np.allclose(epe, pfe)


# ---------------------------------------------------------------------------
# 5. compute_cva limiting behavior
# ---------------------------------------------------------------------------

def test_cva_zero_when_hazard_zero():
    epe = np.array([0.0, 100.0, 200.0, 150.0, 50.0, 0.0])
    time_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    cva = compute_cva(epe, time_grid, hazard=0.0, recovery_rate=0.40)
    assert cva == pytest.approx(0.0)


def test_cva_concentrates_on_first_interval_for_huge_hazard():
    epe = np.array([0.0, 100.0, 200.0, 150.0, 50.0, 0.0])
    time_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    recovery = 0.40

    cva = compute_cva(epe, time_grid, hazard=1e6, recovery_rate=recovery)

    expected = (1 - recovery) * epe[1]  # all default probability mass on (0,1]
    assert cva == pytest.approx(expected, rel=1e-6)


def test_cva_increases_with_hazard():
    epe = np.array([0.0, 100.0, 200.0, 150.0, 50.0, 0.0])
    time_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    cva_low = compute_cva(epe, time_grid, hazard=0.01, recovery_rate=0.40)
    cva_high = compute_cva(epe, time_grid, hazard=0.05, recovery_rate=0.40)

    assert cva_high > cva_low


def test_cva_increases_as_recovery_decreases():
    """Lower recovery -> higher loss-given-default -> higher CVA, for a
    FIXED hazard rate."""
    epe = np.array([0.0, 100.0, 200.0, 150.0, 50.0, 0.0])
    time_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    cva_high_recovery = compute_cva(epe, time_grid, hazard=0.025, recovery_rate=0.60)
    cva_low_recovery = compute_cva(epe, time_grid, hazard=0.025, recovery_rate=0.20)

    assert cva_low_recovery > cva_high_recovery


# ---------------------------------------------------------------------------
# 6. run_cva integration tests
# ---------------------------------------------------------------------------

def test_run_cva_output_shapes_and_types():
    fixed_rate = par_swap_rate(PAYMENT_TIMES, KAPPA, THETA, SIGMA, R0, t=0.0)

    result = run_cva(
        notional=NOTIONAL,
        fixed_rate=fixed_rate,
        payment_times=PAYMENT_TIMES,
        kappa=KAPPA, theta=THETA, sigma=SIGMA, r0=R0,
        cds_spread=0.015, recovery_rate=0.40,
        n_paths=500, seed=42,
    )

    n_steps = len(PAYMENT_TIMES)
    assert result["time_grid"].shape == (n_steps + 1,)
    assert result["rate_paths"].shape == (500, n_steps + 1)
    assert result["exposure"].shape == (500, n_steps + 1)
    assert result["epe"].shape == (n_steps + 1,)
    assert result["pfe"].shape == (n_steps + 1,)
    assert isinstance(result["cva"], float)
    assert result["cva"] >= 0.0
    assert result["hazard"] == pytest.approx(hazard_rate(0.015, 0.40))


def test_run_cva_rejects_non_uniform_schedule():
    fixed_rate = 0.06
    bad_schedule = np.array([0.5, 1.0, 2.5, 5.0])  # uneven gaps

    with pytest.raises(ValueError):
        run_cva(
            notional=NOTIONAL, fixed_rate=fixed_rate, payment_times=bad_schedule,
            kappa=KAPPA, theta=THETA, sigma=SIGMA, r0=R0,
            cds_spread=0.015, recovery_rate=0.40, n_paths=100, seed=1,
        )


def test_run_cva_positive_cds_spread_gives_positive_cva():
    """With non-zero volatility, EPE > 0 at some future timestep even when
    fixed_rate = par rate, so a positive CDS spread should give CVA > 0."""
    fixed_rate = par_swap_rate(PAYMENT_TIMES, KAPPA, THETA, SIGMA, R0, t=0.0)

    result = run_cva(
        notional=NOTIONAL, fixed_rate=fixed_rate, payment_times=PAYMENT_TIMES,
        kappa=KAPPA, theta=THETA, sigma=SIGMA, r0=R0,
        cds_spread=0.015, recovery_rate=0.40, n_paths=2000, seed=10,
    )

    assert result["cva"] > 0.0


def test_run_cva_zero_spread_gives_zero_cva():
    fixed_rate = par_swap_rate(PAYMENT_TIMES, KAPPA, THETA, SIGMA, R0, t=0.0)

    result = run_cva(
        notional=NOTIONAL, fixed_rate=fixed_rate, payment_times=PAYMENT_TIMES,
        kappa=KAPPA, theta=THETA, sigma=SIGMA, r0=R0,
        cds_spread=0.0, recovery_rate=0.40, n_paths=500, seed=11,
    )

    assert result["cva"] == pytest.approx(0.0)


@pytest.mark.parametrize("n_paths", [5000])
def test_cva_is_stable_across_seeds_at_high_path_count(n_paths):
    """
    Monte Carlo stability check: with enough paths, CVA computed with two
    different seeds should be close (relative difference small). This is
    a lightweight stand-in for the full convergence plot in Phase 4 --
    it doesn't prove convergence, but it would catch a simulation that's
    wildly unstable even at 5000 paths.
    """
    fixed_rate = par_swap_rate(PAYMENT_TIMES, KAPPA, THETA, SIGMA, R0, t=0.0)

    result_a = run_cva(
        notional=NOTIONAL, fixed_rate=fixed_rate, payment_times=PAYMENT_TIMES,
        kappa=KAPPA, theta=THETA, sigma=SIGMA, r0=R0,
        cds_spread=0.015, recovery_rate=0.40, n_paths=n_paths, seed=100,
    )
    result_b = run_cva(
        notional=NOTIONAL, fixed_rate=fixed_rate, payment_times=PAYMENT_TIMES,
        kappa=KAPPA, theta=THETA, sigma=SIGMA, r0=R0,
        cds_spread=0.015, recovery_rate=0.40, n_paths=n_paths, seed=200,
    )

    rel_diff = abs(result_a["cva"] - result_b["cva"]) / result_a["cva"]
    assert rel_diff < 0.15  # 15% relative tolerance at 5000 paths