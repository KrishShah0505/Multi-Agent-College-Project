"""20 fixed scenarios (reproducible: explicit values and seeds, no randomness)."""

BASE = dict(notional=1_000_000, tenor_years=5, payment_freq_months=6, fixed_rate=None,
            kappa=0.3, theta=0.06, sigma=0.01, r0=0.05, cds_spread=0.015,
            recovery_rate=0.40, n_paths=2000, pfe_quantile=0.95)

QUESTIONS = [None, "Why is the CVA this size?", "How risky is the counterparty?",
             "When is my exposure highest?"]

_VARIANTS = [
    {},                                                       # S01 base case
    dict(tenor_years=1, payment_freq_months=3, notional=500_000, cds_spread=0.005),
    dict(tenor_years=2, notional=2_000_000),
    dict(tenor_years=3, notional=10_000_000, cds_spread=0.03),
    dict(tenor_years=10, payment_freq_months=12, cds_spread=0.01),
    dict(tenor_years=7, notional=5_000_000, sigma=0.02),
    dict(fixed_rate=0.03),                                    # off-par
    dict(fixed_rate=0.08),                                    # off-par
    dict(recovery_rate=0.20),
    dict(recovery_rate=0.60, cds_spread=0.02),
    dict(tenor_years=4, payment_freq_months=3, notional=250_000, sigma=0.005),
    dict(tenor_years=6, notional=3_000_000, sigma=0.03),
    dict(tenor_years=8, payment_freq_months=12, notional=20_000_000, cds_spread=0.05),
    dict(tenor_years=2, payment_freq_months=3, kappa=0.8),
    dict(kappa=0.05, theta=0.03, r0=0.04),
    dict(tenor_years=3, notional=750_000, r0=0.02, theta=0.05),
    dict(tenor_years=9, fixed_rate=0.045),
    dict(tenor_years=1, notional=100_000, cds_spread=0.0025),
    dict(payment_freq_months=12, sigma=0.015, cds_spread=0.04),
    dict(tenor_years=6, payment_freq_months=3, notional=8_000_000, fixed_rate=0.06, cds_spread=0.012),
]

SCENARIOS = [
    {"id": f"S{i + 1:02d}", "params": {**BASE, **v, "seed": 100 + i}, "question": QUESTIONS[i % 4]}
    for i, v in enumerate(_VARIANTS)
]
