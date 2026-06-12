"""
Verification tests for models/swap.py.

The central check: if the fixed rate is set to the par swap rate at
inception, the swap's value at inception must be (approximately) zero.
This is the standard "no-arbitrage at trade date" identity and catches
sign errors or mismatched discount factor usage between the floating and
fixed leg formulas.
"""

import numpy as np
import pytest

from models.swap import (
    accrual_fractions,
    generate_payment_schedule,
    par_swap_rate,
    price_swap,
    remaining_schedule,
)


NOTIONAL = 1_000_000.0


# ---------------------------------------------------------------------------
# remaining_schedule / accrual_fractions tests
# ---------------------------------------------------------------------------

def test_remaining_schedule_at_inception():
    payment_times = np.array([0.5, 1.0, 1.5, 2.0])
    remaining = remaining_schedule(0.0, payment_times)
    assert np.array_equal(remaining, payment_times)


def test_remaining_schedule_drops_past_payments():
    payment_times = np.array([0.5, 1.0, 1.5, 2.0])
    remaining = remaining_schedule(1.0, payment_times)
    assert np.array_equal(remaining, np.array([1.5, 2.0]))


def test_remaining_schedule_empty_after_maturity():
    payment_times = np.array([0.5, 1.0, 1.5, 2.0])
    remaining = remaining_schedule(2.0, payment_times)
    assert len(remaining) == 0


def test_accrual_fractions_at_inception_regular_schedule():
    payment_times = np.array([0.5, 1.0, 1.5, 2.0])
    remaining = remaining_schedule(0.0, payment_times)
    taus = accrual_fractions(0.0, remaining)
    assert np.allclose(taus, [0.5, 0.5, 0.5, 0.5])


def test_accrual_fractions_mid_period():
    """If t falls between payment dates, the first tau should shrink."""
    payment_times = np.array([0.5, 1.0, 1.5, 2.0])
    remaining = remaining_schedule(0.3, payment_times)
    taus = accrual_fractions(0.3, remaining)
    # first remaining payment is at 0.5, gap from t=0.3 is 0.2
    assert taus[0] == pytest.approx(0.2)
    assert np.allclose(taus[1:], [0.5, 0.5, 0.5])


# ---------------------------------------------------------------------------
# price_swap edge cases
# ---------------------------------------------------------------------------

def test_price_swap_zero_after_maturity():
    payment_times = np.array([0.5, 1.0, 1.5, 2.0])
    V = price_swap(
        r=0.05, t=2.0, fixed_rate=0.06, payment_times=payment_times,
        notional=NOTIONAL, kappa=0.3, theta=0.06, sigma=0.01,
    )
    assert V == 0.0


# ---------------------------------------------------------------------------
# par_swap_rate / no-arbitrage-at-inception tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kappa, theta, sigma, r0",
    [
        (0.3, 0.06, 0.01, 0.05),   # rates below long-run mean
        (0.3, 0.06, 0.01, 0.06),   # rates at long-run mean
        (0.3, 0.06, 0.01, 0.08),   # rates above long-run mean
        (0.8, 0.04, 0.02, 0.03),   # fast mean reversion, higher vol
        (0.1, 0.05, 0.005, 0.05),  # slow mean reversion, low vol
    ],
)
def test_par_rate_gives_zero_value_at_inception(kappa, theta, sigma, r0):
    payment_times = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

    R = par_swap_rate(payment_times, kappa, theta, sigma, r0, t=0.0)

    V = price_swap(
        r=r0, t=0.0, fixed_rate=R, payment_times=payment_times,
        notional=NOTIONAL, kappa=kappa, theta=theta, sigma=sigma,
    )

    # value should be ~0 relative to notional
    assert V == pytest.approx(0.0, abs=1e-6 * NOTIONAL)


def test_par_rate_is_positive_and_reasonable():
    """
    For r0 near theta, the par rate should be in the same ballpark as
    theta -- not negative, not wildly different.
    """
    payment_times = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    R = par_swap_rate(payment_times, kappa=0.3, theta=0.06, sigma=0.01, r0=0.06, t=0.0)

    assert R > 0
    assert abs(R - 0.06) < 0.02  # within 200bps of theta


# ---------------------------------------------------------------------------
# Sign checks: direction of swap value vs fixed rate
# ---------------------------------------------------------------------------

def test_swap_value_decreases_as_fixed_rate_increases():
    """
    Holding everything else fixed, increasing the fixed rate makes the
    swap more expensive for the fixed payer -> V_swap should decrease.
    """
    payment_times = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    kappa, theta, sigma, r0 = 0.3, 0.06, 0.01, 0.06

    R_par = par_swap_rate(payment_times, kappa, theta, sigma, r0, t=0.0)

    V_below = price_swap(r0, 0.0, R_par - 0.01, payment_times, NOTIONAL, kappa, theta, sigma)
    V_at_par = price_swap(r0, 0.0, R_par, payment_times, NOTIONAL, kappa, theta, sigma)
    V_above = price_swap(r0, 0.0, R_par + 0.01, payment_times, NOTIONAL, kappa, theta, sigma)

    assert V_below > V_at_par > V_above
    assert V_below > 0
    assert V_above < 0


def test_swap_value_increases_with_short_rate_for_fixed_payer():
    """
    If rates rise after inception (while the fixed leg stays locked at
    the original par rate), the floating leg becomes more valuable
    relative to the fixed leg -> V_swap should increase for the fixed
    payer. This is the basic intuition from Layer 2.
    """
    payment_times = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    kappa, theta, sigma, r0 = 0.3, 0.06, 0.01, 0.06

    R = par_swap_rate(payment_times, kappa, theta, sigma, r0, t=0.0)

    V_low_rate = price_swap(0.04, 0.0, R, payment_times, NOTIONAL, kappa, theta, sigma)
    V_high_rate = price_swap(0.08, 0.0, R, payment_times, NOTIONAL, kappa, theta, sigma)

    assert V_high_rate > V_low_rate


# ---------------------------------------------------------------------------
# generate_payment_schedule
# ---------------------------------------------------------------------------

def test_generate_payment_schedule_semiannual():
    schedule = generate_payment_schedule(tenor_years=5, freq_months=6)
    expected = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    assert np.allclose(schedule, expected)


def test_generate_payment_schedule_quarterly():
    schedule = generate_payment_schedule(tenor_years=1, freq_months=3)
    expected = np.array([0.25, 0.5, 0.75, 1.0])
    assert np.allclose(schedule, expected)


def test_generate_payment_schedule_rejects_non_integer_periods():
    with pytest.raises(ValueError):
        generate_payment_schedule(tenor_years=5, freq_months=7)  # 5 / (7/12) not integer


def test_generate_payment_schedule_rejects_non_positive_inputs():
    with pytest.raises(ValueError):
        generate_payment_schedule(tenor_years=0, freq_months=6)
    with pytest.raises(ValueError):
        generate_payment_schedule(tenor_years=5, freq_months=0)