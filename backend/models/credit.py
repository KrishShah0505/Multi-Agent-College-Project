"""
Flat hazard rate credit model.

Default is modeled as the first arrival of a Poisson process with constant
intensity (hazard rate) lambda:

    P(survive to T)   = exp(-lambda * T)
    P(default by T)   = 1 - exp(-lambda * T)
    P(default in (t1,t2] | survived to t1) -> marginal probability:
                        exp(-lambda*t1) - exp(-lambda*t2)

lambda is implied from a CDS spread via:

    lambda = cds_spread / (1 - recovery_rate)

Provides:
- hazard_rate(cds_spread, recovery_rate)
- survival_probability(t, hazard)
- default_probability(t, hazard)              -- cumulative, P(default by t)
- marginal_default_probability(t_prev, t_curr, hazard)
"""

import numpy as np


def hazard_rate(cds_spread: float, recovery_rate: float = 0.40) -> float:
    """
    Implied flat hazard rate from a CDS spread.

    Parameters
    ----------
    cds_spread    : annualized CDS spread, in decimal (e.g. 0.0150 for 150bps)
    recovery_rate : assumed recovery on default, in decimal (default 0.40)

    Returns
    -------
    lambda : annualized default intensity
    """
    if not (0.0 <= recovery_rate < 1.0):
        raise ValueError("recovery_rate must be in [0, 1).")
    if cds_spread < 0:
        raise ValueError("cds_spread must be non-negative.")

    return cds_spread / (1.0 - recovery_rate)


def survival_probability(t: float, hazard: float) -> float:
    """P(survive to time t) = exp(-lambda * t)."""
    return np.exp(-hazard * t)


def default_probability(t: float, hazard: float) -> float:
    """Cumulative P(default by time t) = 1 - exp(-lambda * t)."""
    return 1.0 - survival_probability(t, hazard)


def marginal_default_probability(t_prev: float, t_curr: float, hazard: float) -> float:
    """
    P(survive to t_prev AND default in (t_prev, t_curr])
        = P(survive to t_prev) - P(survive to t_curr)
        = exp(-lambda * t_prev) - exp(-lambda * t_curr)

    t_curr must be >= t_prev.
    """
    if t_curr < t_prev:
        raise ValueError("t_curr must be >= t_prev.")

    return survival_probability(t_prev, hazard) - survival_probability(t_curr, hazard)


def marginal_default_probabilities(time_grid: np.ndarray, hazard: float) -> np.ndarray:
    """
    Vectorized marginal default probabilities over a time grid
    [t_0, t_1, ..., t_n] (typically t_0 = 0).

    Returns an array of length n (one per interval (t_{i-1}, t_i]),
    NOT length n+1 -- there's no "interval before t_0".
    """
    time_grid = np.asarray(time_grid, dtype=float)
    survival = survival_probability(time_grid, hazard)
    return survival[:-1] - survival[1:]