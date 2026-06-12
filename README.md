# CVA Engine

A Monte Carlo Credit Valuation Adjustment (CVA) calculator for a vanilla
fixed-for-floating interest rate swap, built under the Vasicek short-rate
model.

**What this answers:** if I hold an interest rate swap with a counterparty,
how much should I discount its value today to account for the risk that
the counterparty defaults before maturity?

```
CVA = (1 - R) * sum_i [ EPE(t_i) * PD(t_{i-1}, t_i) ]
```

where `EPE` (Expected Positive Exposure) comes from a Monte Carlo
simulation of the swap's mark-to-market value over time, and `PD`
(marginal default probability) comes from a hazard rate implied by the
counterparty's CDS spread.

---

## Why this exists

Post-2008, CVA (and the broader XVA family: DVA, FVA, KVA) became a
standard regulatory and pricing adjustment that every bank applies to OTC
derivatives under Basel III. This project implements the core pipeline —
short-rate simulation, swap repricing, exposure profiling, and credit
adjustment — from first principles, with every step verified against a
closed-form or analytical identity rather than just "it runs."

---

## The pipeline

```
Vasicek short-rate simulation
        |
        v
rate_paths  [n_paths x (n_steps+1)]
        |
        v
Reprice the residual swap at every (path, time)
using the closed-form Vasicek bond price P(t,T)
        |
        v
exposure = max(V_swap, 0)   [n_paths x (n_steps+1)]
        |
        v
EPE(t) = mean(exposure)        PFE(t) = 95th pct(exposure)
        |
        v
lambda = CDS_spread / (1 - recovery_rate)
PD(t_prev, t_curr) = e^(-lambda*t_prev) - e^(-lambda*t_curr)
        |
        v
CVA = (1 - recovery) * sum( EPE(t_i) * PD(t_{i-1}, t_i) )
```

---

## Repo structure

```
cva-engine/
├── backend/
│   ├── models/
│   │   ├── vasicek.py    # bond_price(), simulate_paths()
│   │   ├── swap.py       # par_swap_rate(), price_swap(), schedule generation
│   │   └── credit.py      # hazard_rate(), default probabilities
│   ├── core/
│   │   └── cva.py         # exposure_matrix(), EPE/PFE, compute_cva(), run_cva()
│   ├── api/
│   │   ├── main.py        # FastAPI app
│   │   ├── routes.py       # POST /cva
│   │   └── schemas.py      # request/response models
│   ├── tests/              # 80 tests across all modules
│   └── requirements.txt
├── notebooks/
│   ├── validation.py       # Phase 4 validation script
│   └── figures/             # generated plots (see below)
└── README.md
```

---

## Running it

### Tests

```bash
cd backend
pip install -r requirements.txt
pytest -v
```

80 tests, covering:
- Vasicek bond price identities (monotonicity in maturity, long-run zero
  rate limit, simulated mean/variance vs analytical Vasicek moments)
- Swap pricing (par rate gives zero value at inception, across 5 parameter
  sets; sign checks for fixed rate and short rate sensitivity)
- Credit model (hazard rate derivation, marginal default probabilities
  telescoping to cumulative default probability)
- End-to-end CVA pipeline, including a **deterministic closed-form check**:
  when sigma=0 and r0=theta, simulated EPE matches `max(price_swap(...), 0)`
  computed directly, to within floating-point precision
- API request validation and response shapes

### API

```bash
cd backend
uvicorn api.main:app --reload
```

Then `POST /cva`:

```json
{
  "notional": 1000000,
  "tenor_years": 5,
  "payment_freq_months": 6,
  "kappa": 0.3,
  "theta": 0.06,
  "sigma": 0.01,
  "r0": 0.05,
  "cds_spread": 0.015,
  "recovery_rate": 0.40,
  "n_paths": 5000,
  "seed": 42
}
```

If `fixed_rate` is omitted, the par swap rate at inception is used (i.e. a
fairly-priced swap at t=0). Response includes `time_grid`, `epe`, `pfe`,
`cva`, `hazard`, `fixed_rate_used`, and `par_rate`.

### Validation plots

```bash
cd notebooks
python validation.py
```

Regenerates the four figures in `notebooks/figures/`.

---

## Validation

### Monte Carlo convergence

CVA stabilizes as `n_paths` increases — mean across 25 seeds goes from
~$506 (std ~$58, ~12% relative) at 100 paths to ~$523 (std ~$3.5, ~0.7%
relative) at 20,000 paths.

![Convergence](notebooks/figures/convergence.png)

### Sensitivity

CVA vs CDS spread is approximately linear (correlation ~0.999 with a
linear fit over 0-500bps), consistent with the marginal default
probability being approximately linear in the hazard rate for small
lambda. CVA vs sigma is monotonically increasing.

![Sensitivity](notebooks/figures/sensitivity.png)

### Convexity gap

EPE sits above the exposure computed along the *expected* rate path
(Jensen's inequality: `E[max(V,0)] >= max(E[V],0)`). At the peak (t=2yr),
roughly half of EPE comes purely from rate uncertainty rather than the
expected path — this is the mechanism behind the CVA-vs-sigma sensitivity
above.

![Convexity gap](notebooks/figures/convexity_gap.png)

### Deterministic baseline (sigma=0, r0=theta)

When sigma=0 and r0=theta, every path is identical and EPE matches the
closed-form swap value to ~1e-11 (floating point noise). The value is
exactly zero for *every* timestep, not just t=0 — under a flat curve with
a uniform-tau schedule, the par swap rate is tenor-independent
(`R = 2(1/x - 1)`, `x = e^{-theta*tau}`), so the residual swap is always at
par.

![Deterministic baseline](notebooks/figures/deterministic_baseline.png)

---

## Known simplifications

These are deliberate scope choices, documented so they read as informed
trade-offs rather than gaps:

- **Time convention:** payment dates are floats (years from today, e.g.
  `[0.5, 1.0, ..., 5.0]`), not calendar dates with day-count conventions
  (ACT/360, ACT/365). A production system would use real calendars; this
  project isolates the stochastic/pricing math from calendar engineering.

- **Floating leg valuation:** assumes the valuation date coincides with a
  reset date, so `V_float(t) = Notional * (1 - P(t, T_n))`. This is exact
  at every timestep used in the simulation (since the simulation grid
  *is* the reset grid), but would need an accrual correction for
  off-grid valuation dates.

- **Single-factor short-rate model:** Vasicek allows negative rates
  (a known and, post-2008, less severe critique). Hull-White or a
  multi-factor model (e.g. G2++) would be the natural production
  extension.

- **Flat hazard rate:** counterparty default intensity is assumed
  constant, implied from a single CDS spread. A real CVA desk would
  bootstrap a full hazard rate term structure from the CDS curve.

- **CVA only:** this implements CVA in isolation. Full XVA also includes
  DVA (own default risk), FVA (funding cost), and KVA (capital cost) —
  out of scope here but conceptually additive adjustments on the same
  exposure profile.

- **CVA timing convention:** the formula uses EPE at interval *endpoints*
  weighted by that interval's marginal default probability. A trapezoidal
  (average of interval endpoints) convention is equally defensible and
  would give a slightly different number.