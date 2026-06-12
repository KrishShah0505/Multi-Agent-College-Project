"""
Vanilla fixed-for-floating interest rate swap pricing under Vasicek.

Conventions (Option A, as agreed):
- Time is represented as floats (years from today), e.g. [0.5, 1.0, ..., 5.0]
- Accrual fractions (tau) are simply differences between consecutive
  schedule times -- no calendar/day-count logic.
- The floating leg is assumed to reset to par at the valuation date, i.e.
  V_float(t) = Notional * (P(t,t) - P(t, T_n)) = Notional * (1 - P(t,T_n))
  This is the standard simplification: valuation coincides with a reset date.

Provides:
- remaining_schedule(t, payment_times)
- accrual_fractions(t, remaining)
- price_swap(...): MtM value of the residual swap at time t, given short rate r
- par_swap_rate(...): fixed rate that makes V_swap(0) = 0
"""

import numpy as np

from models.vasicek import bond_price


def remaining_schedule(t: float, payment_times: np.ndarray) -> np.ndarray:
    """
    Payment dates strictly after t -- i.e. the residual swap's cash flow
    schedule as seen from time t.
    """
    payment_times = np.asarray(payment_times, dtype=float)
    return payment_times[payment_times > t + 1e-12]


def accrual_fractions(t: float, remaining: np.ndarray) -> np.ndarray:
    """
    tau_i for each remaining payment date: the gap between it and the
    previous date in the schedule, with t itself standing in as the
    "previous date" for the first remaining payment.

    e.g. t=0, remaining=[0.5, 1.0, ..., 5.0] -> taus = [0.5, 0.5, ..., 0.5]
    e.g. t=0.3, remaining=[0.5, 1.0, ...]     -> taus = [0.2, 0.5, ...]
    """
    if len(remaining) == 0:
        return np.array([])
    prev = np.concatenate(([t], remaining[:-1]))
    return remaining - prev


def price_swap(
    r: float,
    t: float,
    fixed_rate: float,
    payment_times: np.ndarray,
    notional: float,
    kappa: float,
    theta: float,
    sigma: float,
) -> float:
    """
    Mark-to-market value of the RESIDUAL swap at time t, from the fixed
    payer's perspective (pay fixed, receive floating):

        V_swap(t) = V_float(t) - V_fixed(t)

    V_float(t) = Notional * (1 - P(t, T_n))
    V_fixed(t) = Notional * fixed_rate * sum(tau_i * P(t, t_i))

    where T_n is the final remaining payment date and the sum runs over
    all remaining payment dates t_i.

    If t is at or after the final payment date, the swap has no remaining
    cash flows and its value is 0.
    """
    remaining = remaining_schedule(t, payment_times)
    if len(remaining) == 0:
        return 0.0

    taus = accrual_fractions(t, remaining)
    discount_factors = np.array(
        [bond_price(r, t, T, kappa, theta, sigma) for T in remaining]
    )

    P_t_Tn = discount_factors[-1]

    V_float = notional * (1.0 - P_t_Tn)
    V_fixed = notional * fixed_rate * np.sum(taus * discount_factors)

    return V_float - V_fixed


def par_swap_rate(
    payment_times: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
    r0: float,
    t: float = 0.0,
) -> float:
    """
    The fixed rate R that makes V_swap(t) = 0 at the given valuation time
    (typically t=0, inception):

        R = (1 - P(t, T_n)) / sum(tau_i * P(t, t_i))

    Computed using the short rate r0 at time t.
    """
    remaining = remaining_schedule(t, payment_times)
    if len(remaining) == 0:
        raise ValueError("No remaining payment dates after t; cannot compute par rate.")

    taus = accrual_fractions(t, remaining)
    discount_factors = np.array(
        [bond_price(r0, t, T, kappa, theta, sigma) for T in remaining]
    )

    P_t_Tn = discount_factors[-1]
    numerator = 1.0 - P_t_Tn
    denominator = np.sum(taus * discount_factors)

    return numerator / denominator


def generate_payment_schedule(tenor_years: float, freq_months: int = 6) -> np.ndarray:
    """
    Generate a uniformly-spaced payment schedule as floats (years from
    today), e.g. tenor_years=5, freq_months=6 -> [0.5, 1.0, ..., 5.0].

    Raises ValueError if tenor_years is not (to within floating point
    tolerance) an integer multiple of the payment frequency.
    """
    if tenor_years <= 0:
        raise ValueError("tenor_years must be positive.")
    if freq_months <= 0:
        raise ValueError("freq_months must be positive.")

    dt = freq_months / 12.0
    n_steps_float = tenor_years / dt

    n_steps = round(n_steps_float)
    if not np.isclose(n_steps_float, n_steps, atol=1e-6):
        raise ValueError(
            f"tenor_years ({tenor_years}) must be an integer multiple of "
            f"freq_months/12 ({dt})."
        )

    return np.round(np.arange(1, n_steps + 1) * dt, 10)