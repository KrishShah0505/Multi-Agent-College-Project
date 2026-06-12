"""
Vasicek short-rate model.

dr(t) = kappa * (theta - r(t)) dt + sigma dW(t)

Provides:
- bond_price(r, t, T, kappa, theta, sigma): closed-form zero-coupon bond price P(t,T)
- simulate_paths(...): exact-discretization Monte Carlo simulation of r(t)
"""

import numpy as np


def B(t: float, T: float, kappa: float) -> float:
    """B(t,T) term in the Vasicek bond price formula."""
    return (1 - np.exp(-kappa * (T - t))) / kappa


def A(t: float, T: float, kappa: float, theta: float, sigma: float) -> float:
    """A(t,T) term in the Vasicek bond price formula."""
    b = B(t, T, kappa)
    term1 = (theta - (sigma ** 2) / (2 * kappa ** 2)) * (b - (T - t))
    term2 = (sigma ** 2 * b ** 2) / (4 * kappa)
    return np.exp(term1 - term2)


def bond_price(r: float, t: float, T: float, kappa: float, theta: float, sigma: float) -> float:
    """
    Closed-form Vasicek zero-coupon bond price P(t,T).

    Parameters
    ----------
    r     : short rate at time t
    t     : current time (years)
    T     : maturity (years)
    kappa : speed of mean reversion
    theta : long-run mean rate
    sigma : volatility of the short rate

    Returns
    -------
    P(t,T) : price today (at time t) of $1 received at time T
    """
    if t >= T:
        # P(T,T) = 1 by definition; also guards against negative (T-t)
        return 1.0

    b = B(t, T, kappa)
    a = A(t, T, kappa, theta, sigma)
    return a * np.exp(-b * r)


def simulate_paths(
    n_paths: int,
    n_steps: int,
    dt: float,
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate short-rate paths under Vasicek using the EXACT discretization
    (not Euler-Maruyama), so there is no discretization bias even for large dt.

    r(t+dt) = r(t) * e^{-kappa*dt}
              + theta * (1 - e^{-kappa*dt})
              + sigma * sqrt[(1 - e^{-2*kappa*dt}) / (2*kappa)] * Z,   Z ~ N(0,1)

    Parameters
    ----------
    n_paths : number of Monte Carlo paths
    n_steps : number of time steps (the output will have n_steps + 1 columns,
              including t=0)
    dt      : size of each time step in years (e.g. 0.5 for 6-monthly)
    r0      : initial short rate
    kappa, theta, sigma : Vasicek parameters
    seed    : optional RNG seed for reproducibility

    Returns
    -------
    rate_paths : ndarray of shape (n_paths, n_steps + 1)
                  rate_paths[:, 0] == r0 for all paths
    """
    rng = np.random.default_rng(seed)

    rate_paths = np.empty((n_paths, n_steps + 1), dtype=float)
    rate_paths[:, 0] = r0

    # Precompute constants that don't depend on the simulated rate
    decay = np.exp(-kappa * dt)
    mean_shift = theta * (1 - decay)
    var_term = (sigma ** 2) * (1 - np.exp(-2 * kappa * dt)) / (2 * kappa)
    std_term = np.sqrt(var_term)

    for step in range(1, n_steps + 1):
        z = rng.standard_normal(n_paths)
        rate_paths[:, step] = (
            rate_paths[:, step - 1] * decay
            + mean_shift
            + std_term * z
        )

    return rate_paths


def time_grid(n_steps: int, dt: float) -> np.ndarray:
    """Convenience helper: returns [0, dt, 2*dt, ..., n_steps*dt]."""
    return np.arange(n_steps + 1) * dt