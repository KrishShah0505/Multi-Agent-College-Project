"""
Core CVA pipeline: wires together Vasicek simulation, swap repricing, and
the hazard rate credit model.

Pipeline:
    simulate_paths()              -> rate_paths   [n_paths x (n_steps+1)]
    exposure_matrix()             -> exposure     [n_paths x (n_steps+1)]
    compute_epe_pfe()             -> EPE, PFE     [n_steps+1]
    compute_cva()                 -> CVA          scalar

run_cva() is the single entry point that runs the full pipeline given
swap + model + credit parameters.
"""

import numpy as np

from models.credit import hazard_rate, marginal_default_probabilities
from models.swap import accrual_fractions, remaining_schedule
from models.vasicek import bond_price, simulate_paths


def price_swap_vectorized(
    r: np.ndarray,
    t: float,
    fixed_rate: float,
    payment_times: np.ndarray,
    notional: float,
    kappa: float,
    theta: float,
    sigma: float,
) -> np.ndarray:
    """
    Vectorized version of models.swap.price_swap: prices the residual swap
    at time t for an ARRAY of short rates r (one per Monte Carlo path).

    Returns an array of the same shape as r. If t is at or after the final
    payment date, returns an array of zeros (no remaining cash flows).

    This must stay mathematically identical to models.swap.price_swap for
    scalar r -- see tests/test_cva.py for the consistency check.
    """
    r = np.asarray(r, dtype=float)
    remaining = remaining_schedule(t, payment_times)

    if len(remaining) == 0:
        return np.zeros_like(r)

    taus = accrual_fractions(t, remaining)

    # discount_factors[i] is an array over paths for remaining time i
    # bond_price broadcasts over r since A(t,T) and B(t,T) don't depend on r
    discount_factors = np.array(
        [bond_price(r, t, T, kappa, theta, sigma) for T in remaining]
    )  # shape (n_remaining, n_paths)

    P_t_Tn = discount_factors[-1]  # shape (n_paths,)

    V_float = notional * (1.0 - P_t_Tn)
    V_fixed = notional * fixed_rate * np.sum(taus[:, None] * discount_factors, axis=0)

    return V_float - V_fixed


def exposure_matrix(
    rate_paths: np.ndarray,
    time_grid: np.ndarray,
    fixed_rate: float,
    payment_times: np.ndarray,
    notional: float,
    kappa: float,
    theta: float,
    sigma: float,
) -> np.ndarray:
    """
    Build the full exposure matrix: exposure[i,j] = max(V_swap on path i at
    time time_grid[j], 0).

    rate_paths: shape (n_paths, n_steps+1)
    time_grid : shape (n_steps+1,) -- must correspond column-for-column to
                rate_paths' second axis.

    Returns
    -------
    exposure : shape (n_paths, n_steps+1)
    """
    n_paths, n_cols = rate_paths.shape
    if len(time_grid) != n_cols:
        raise ValueError("time_grid length must match rate_paths' second axis.")

    exposure = np.zeros_like(rate_paths)

    for j, t in enumerate(time_grid):
        r_col = rate_paths[:, j]
        V = price_swap_vectorized(
            r_col, t, fixed_rate, payment_times, notional, kappa, theta, sigma
        )
        exposure[:, j] = np.maximum(V, 0.0)

    return exposure


def compute_epe_pfe(exposure: np.ndarray, pfe_quantile: float = 0.95):
    """
    Collapse the exposure matrix across paths (axis=0) into EPE and PFE
    curves over time.

    EPE(t) = mean exposure across paths
    PFE(t) = pfe_quantile-th percentile of exposure across paths

    Returns
    -------
    epe, pfe : each shape (n_steps+1,)
    """
    epe = exposure.mean(axis=0)
    pfe = np.percentile(exposure, pfe_quantile * 100, axis=0)
    return epe, pfe


def compute_cva(
    epe: np.ndarray,
    time_grid: np.ndarray,
    hazard: float,
    recovery_rate: float = 0.40,
) -> float:
    """
    CVA = (1 - R) * sum_i [ EPE(t_i) * PD(t_{i-1}, t_i) ]

    where the sum runs over intervals (t_0,t_1], (t_1,t_2], ..., and
    EPE(t_i) is the exposure at the END of each interval.

    epe and time_grid must have the same length (n_steps+1, including t=0).
    The first element of epe (at t=0) is not used directly in the sum --
    only EPE at interval ENDPOINTS t_1..t_n contributes, weighted by the
    marginal default probability of THAT interval.
    """
    if len(epe) != len(time_grid):
        raise ValueError("epe and time_grid must have the same length.")

    marginals = marginal_default_probabilities(time_grid, hazard)  # length n_steps
    epe_at_interval_ends = epe[1:]  # length n_steps, aligned with marginals

    return (1.0 - recovery_rate) * float(np.sum(epe_at_interval_ends * marginals))


def run_cva(
    notional: float,
    fixed_rate: float,
    payment_times: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
    r0: float,
    cds_spread: float,
    recovery_rate: float = 0.40,
    n_paths: int = 1000,
    seed: int | None = None,
    pfe_quantile: float = 0.95,
) -> dict:
    """
    Full pipeline entry point.

    Assumes payment_times is a uniformly-spaced grid (e.g. [0.5, 1.0, ...,
    5.0]) and that the simulation time grid is [0.0] + payment_times --
    i.e. simulation steps coincide exactly with payment/reset dates.

    Returns a dict with:
        time_grid    : (n_steps+1,)
        rate_paths   : (n_paths, n_steps+1)
        exposure     : (n_paths, n_steps+1)
        epe, pfe     : (n_steps+1,)
        cva          : float
        hazard       : float (implied lambda)
    """
    payment_times = np.asarray(payment_times, dtype=float)

    if len(payment_times) < 1:
        raise ValueError("payment_times must be non-empty.")

    diffs = np.diff(np.concatenate(([0.0], payment_times)))
    if not np.allclose(diffs, diffs[0]):
        raise ValueError("run_cva requires a uniformly-spaced payment schedule.")

    dt = diffs[0]
    n_steps = len(payment_times)

    time_grid = np.concatenate(([0.0], payment_times))

    rate_paths = simulate_paths(n_paths, n_steps, dt, r0, kappa, theta, sigma, seed=seed)

    exposure = exposure_matrix(
        rate_paths, time_grid, fixed_rate, payment_times, notional, kappa, theta, sigma
    )

    epe, pfe = compute_epe_pfe(exposure, pfe_quantile=pfe_quantile)

    hazard = hazard_rate(cds_spread, recovery_rate)
    cva = compute_cva(epe, time_grid, hazard, recovery_rate)

    return {
        "time_grid": time_grid,
        "rate_paths": rate_paths,
        "exposure": exposure,
        "epe": epe,
        "pfe": pfe,
        "cva": cva,
        "hazard": hazard,
    }