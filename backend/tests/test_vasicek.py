"""
Verification tests for models/vasicek.py.

These tests check the closed-form bond price against known analytical
properties of the Vasicek model, and check that the Monte Carlo simulator
converges to the model's analytical mean and variance.
"""

import numpy as np
import pytest

from models.vasicek import bond_price, simulate_paths, time_grid


# Shared "reasonable" parameter set used across tests
KAPPA = 0.3
THETA = 0.06
SIGMA = 0.01
R0 = 0.05


# ---------------------------------------------------------------------------
# bond_price tests
# ---------------------------------------------------------------------------

def test_bond_price_at_maturity_is_one():
    """P(T,T) must equal 1 by definition."""
    P = bond_price(R0, t=2.0, T=2.0, kappa=KAPPA, theta=THETA, sigma=SIGMA)
    assert P == pytest.approx(1.0)


def test_bond_price_decreasing_in_maturity():
    """A dollar further in the future is worth less today."""
    maturities = [0.5, 1, 2, 5, 10, 20]
    prices = [bond_price(R0, 0, T, KAPPA, THETA, SIGMA) for T in maturities]

    for earlier, later in zip(prices, prices[1:]):
        assert earlier > later


def test_bond_price_between_zero_and_one():
    """Discount factors must always lie in (0, 1) for positive maturities."""
    for T in [0.5, 1, 5, 10, 30]:
        P = bond_price(R0, 0, T, KAPPA, THETA, SIGMA)
        assert 0.0 < P < 1.0


def test_long_term_zero_rate_converges_to_analytical_limit():
    """
    As T -> infinity, the continuously-compounded zero rate
    y(0,T) = -ln(P(0,T)) / T converges to:

        y(0, inf) = theta - sigma^2 / (2 * kappa^2)

    This checks the A(t,T) and B(t,T) formulas jointly -- a sign error or
    misplaced sigma^2/kappa^2 term would cause this to fail even though
    simpler monotonicity checks might still pass.
    """
    T = 2000.0  # "infinity" approximation -- convergence is O(1/T)
    P = bond_price(R0, 0, T, KAPPA, THETA, SIGMA)
    y_inf = -np.log(P) / T

    expected = THETA - (SIGMA ** 2) / (2 * KAPPA ** 2)

    assert y_inf == pytest.approx(expected, abs=1e-4)


def test_short_maturity_bond_price_near_one():
    """For very small T, P(0,T) ~ 1 - r0*T (first-order approximation)."""
    T = 1e-4
    P = bond_price(R0, 0, T, KAPPA, THETA, SIGMA)
    approx = 1 - R0 * T
    assert P == pytest.approx(approx, abs=1e-6)


# ---------------------------------------------------------------------------
# simulate_paths tests
# ---------------------------------------------------------------------------

def test_simulated_paths_shape():
    n_paths, n_steps, dt = 100, 10, 0.5
    paths = simulate_paths(n_paths, n_steps, dt, R0, KAPPA, THETA, SIGMA, seed=42)

    assert paths.shape == (n_paths, n_steps + 1)
    # all paths must start at r0
    assert np.allclose(paths[:, 0], R0)


def test_simulated_mean_converges_to_analytical_mean():
    """
    Under Vasicek, E[r(T)] = theta + (r0 - theta) * e^{-kappa*T}.

    With enough paths, the Monte Carlo mean at each timestep should be
    close to this analytical value. If the discretization formula has a
    bug (e.g. wrong decay term), this will fail even if the simulation
    runs without errors.
    """
    n_paths, n_steps, dt = 20_000, 20, 0.25
    paths = simulate_paths(n_paths, n_steps, dt, R0, KAPPA, THETA, SIGMA, seed=123)
    times = time_grid(n_steps, dt)

    simulated_mean = paths.mean(axis=0)
    analytical_mean = THETA + (R0 - THETA) * np.exp(-KAPPA * times)

    # Monte Carlo noise -> allow a small tolerance
    assert np.allclose(simulated_mean, analytical_mean, atol=2e-3)


def test_simulated_variance_converges_to_analytical_variance():
    """
    Under Vasicek, Var[r(T)] = (sigma^2 / (2*kappa)) * (1 - e^{-2*kappa*T}).

    This is independent of r0 and theta, so it isolates the noise term
    (sigma * sqrt[...]) in the discretization formula.
    """
    n_paths, n_steps, dt = 20_000, 20, 0.25
    paths = simulate_paths(n_paths, n_steps, dt, R0, KAPPA, THETA, SIGMA, seed=7)
    times = time_grid(n_steps, dt)

    simulated_var = paths.var(axis=0)
    analytical_var = (SIGMA ** 2 / (2 * KAPPA)) * (1 - np.exp(-2 * KAPPA * times))

    # variance estimates are noisier than means -> looser tolerance
    assert np.allclose(simulated_var, analytical_var, atol=5e-5)


def test_simulation_is_reproducible_with_seed():
    """Same seed -> identical paths. Important for debugging/reporting."""
    p1 = simulate_paths(50, 10, 0.5, R0, KAPPA, THETA, SIGMA, seed=99)
    p2 = simulate_paths(50, 10, 0.5, R0, KAPPA, THETA, SIGMA, seed=99)

    assert np.array_equal(p1, p2)