"""
Verification tests for models/credit.py.

Key identities checked:
- hazard_rate formula and its edge cases (recovery_rate bounds)
- survival probability is a proper decreasing probability (1 at t=0, ->0 as t->inf)
- marginal default probabilities sum to total cumulative default probability
  over the full horizon (telescoping sum identity)
- limiting behavior: lambda -> 0 implies no default risk; lambda -> large
  implies near-certain early default
"""

import numpy as np
import pytest

from models.credit import (
    default_probability,
    hazard_rate,
    marginal_default_probabilities,
    marginal_default_probability,
    survival_probability,
)


# ---------------------------------------------------------------------------
# hazard_rate
# ---------------------------------------------------------------------------

def test_hazard_rate_basic_value():
    # 150bps spread, 40% recovery -> lambda = 0.015 / 0.6 = 0.025
    lam = hazard_rate(cds_spread=0.0150, recovery_rate=0.40)
    assert lam == pytest.approx(0.025)


def test_hazard_rate_zero_spread_gives_zero_hazard():
    lam = hazard_rate(cds_spread=0.0, recovery_rate=0.40)
    assert lam == 0.0


def test_hazard_rate_higher_recovery_means_higher_hazard():
    """
    For a fixed observed spread, the spread compensates for expected loss
    = lambda * (1 - R). If recovery R is higher, loss-given-default
    (1-R) is smaller, so lambda must be LARGER to produce the same spread.
    i.e. higher assumed recovery -> higher implied hazard rate.
    """
    lam_low_recovery = hazard_rate(cds_spread=0.02, recovery_rate=0.20)
    lam_high_recovery = hazard_rate(cds_spread=0.02, recovery_rate=0.60)
    assert lam_high_recovery > lam_low_recovery


def test_hazard_rate_rejects_invalid_recovery():
    with pytest.raises(ValueError):
        hazard_rate(0.02, recovery_rate=1.0)
    with pytest.raises(ValueError):
        hazard_rate(0.02, recovery_rate=-0.1)


def test_hazard_rate_rejects_negative_spread():
    with pytest.raises(ValueError):
        hazard_rate(-0.01, recovery_rate=0.4)


# ---------------------------------------------------------------------------
# survival_probability / default_probability
# ---------------------------------------------------------------------------

def test_survival_probability_at_zero_is_one():
    assert survival_probability(0.0, hazard=0.025) == pytest.approx(1.0)


def test_survival_probability_decreasing_in_time():
    lam = 0.025
    times = [0, 1, 2, 5, 10]
    survivals = [survival_probability(t, lam) for t in times]
    for earlier, later in zip(survivals, survivals[1:]):
        assert earlier > later


def test_default_probability_complements_survival():
    lam = 0.025
    for t in [0.5, 1, 5, 10]:
        S = survival_probability(t, lam)
        D = default_probability(t, lam)
        assert S + D == pytest.approx(1.0)


def test_default_probability_zero_at_t_zero():
    assert default_probability(0.0, hazard=0.025) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# marginal default probabilities
# ---------------------------------------------------------------------------

def test_marginal_default_probability_basic():
    lam = 0.025
    pd = marginal_default_probability(0.0, 1.0, lam)
    expected = (1 - np.exp(-lam * 1.0)) - 0  # since survival(0)=1
    assert pd == pytest.approx(expected)


def test_marginal_default_probability_rejects_decreasing_times():
    with pytest.raises(ValueError):
        marginal_default_probability(2.0, 1.0, hazard=0.025)


def test_marginal_probabilities_sum_to_cumulative_default():
    """
    Telescoping sum: sum of marginal PDs over (0,T] intervals should equal
    the cumulative default probability P(default by T_n).

    This is the identity the CVA formula relies on -- if it doesn't hold,
    CVA will silently double-count or undercount default risk.
    """
    lam = 0.025
    time_grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

    marginals = marginal_default_probabilities(time_grid, lam)
    total = np.sum(marginals)

    expected_total = default_probability(time_grid[-1], lam)

    assert total == pytest.approx(expected_total)
    assert len(marginals) == len(time_grid) - 1


def test_marginal_probabilities_all_non_negative():
    lam = 0.025
    time_grid = np.linspace(0, 5, 11)
    marginals = marginal_default_probabilities(time_grid, lam)
    assert np.all(marginals >= 0)


# ---------------------------------------------------------------------------
# limiting behavior
# ---------------------------------------------------------------------------

def test_zero_hazard_means_certain_survival():
    """lambda=0 -> never defaults -> survival is always 1, default always 0."""
    for t in [0, 1, 5, 100]:
        assert survival_probability(t, hazard=0.0) == pytest.approx(1.0)
        assert default_probability(t, hazard=0.0) == pytest.approx(0.0)


def test_large_hazard_means_near_certain_early_default():
    """Very high lambda -> survival probability collapses quickly."""
    lam = 50.0  # absurdly high annual intensity
    S_1yr = survival_probability(1.0, lam)
    assert S_1yr < 1e-10


def test_zero_hazard_marginal_probabilities_are_zero():
    time_grid = np.linspace(0, 5, 6)
    marginals = marginal_default_probabilities(time_grid, hazard=0.0)
    assert np.allclose(marginals, 0.0)