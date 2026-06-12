"""
Phase 4 validation script.

Generates the plots that justify the CVA numbers, not just the code that
produces them:

1. Monte Carlo convergence: CVA vs n_paths, across multiple seeds, showing
   the estimate stabilizing as n_paths increases.
2. Sensitivity: CVA vs CDS spread (expect roughly linear), CVA vs sigma
   (expect increasing -- more rate vol -> more exposure -> more CVA).
3. Deterministic baseline: sigma=0 case, EPE/PFE curves collapse onto the
   closed-form swap value curve exactly (no Monte Carlo noise at all).

Run with: python notebooks/validation.py
Outputs PNGs to notebooks/figures/
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# allow running as a script from notebooks/ with backend/ on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from core.cva import run_cva  # noqa: E402
from models.swap import par_swap_rate  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Shared base parameters
NOTIONAL = 1_000_000.0
PAYMENT_TIMES = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
KAPPA, THETA, SIGMA, R0 = 0.3, 0.06, 0.01, 0.05
CDS_SPREAD = 0.015
RECOVERY = 0.40

FIXED_RATE = par_swap_rate(PAYMENT_TIMES, KAPPA, THETA, SIGMA, R0, t=0.0)


# ---------------------------------------------------------------------------
# 1. Convergence: CVA vs n_paths, multiple seeds
# ---------------------------------------------------------------------------

def plot_convergence():
    path_counts = [100, 250, 500, 1000, 2500, 5000, 10000, 20000]
    seeds = list(range(1, 26))  # 25 seeds for a less noisy std estimate

    cva_grid = np.zeros((len(seeds), len(path_counts)))

    for i, seed in enumerate(seeds):
        for j, n_paths in enumerate(path_counts):
            result = run_cva(
                notional=NOTIONAL, fixed_rate=FIXED_RATE, payment_times=PAYMENT_TIMES,
                kappa=KAPPA, theta=THETA, sigma=SIGMA, r0=R0,
                cds_spread=CDS_SPREAD, recovery_rate=RECOVERY,
                n_paths=n_paths, seed=seed,
            )
            cva_grid[i, j] = result["cva"]

    mean_cva = cva_grid.mean(axis=0)
    std_cva = cva_grid.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, seed in enumerate(seeds):
        ax.plot(path_counts, cva_grid[i], "-", alpha=0.15, color="steelblue",
                label="individual seeds" if i == 0 else None)

    ax.plot(path_counts, mean_cva, "o-", color="darkred", linewidth=2.5, label="mean across seeds")
    ax.fill_between(path_counts, mean_cva - std_cva, mean_cva + std_cva, color="darkred", alpha=0.15, label="±1 std")

    ax.set_xscale("log")
    ax.set_xlabel("Number of Monte Carlo paths")
    ax.set_ylabel("CVA ($)")
    ax.set_title("Monte Carlo Convergence: CVA vs n_paths")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "convergence.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"[convergence] saved to {path}")
    print(f"[convergence] mean CVA at n=100:    {mean_cva[0]:.2f}  (std {std_cva[0]:.2f})")
    print(f"[convergence] mean CVA at n=20000:  {mean_cva[-1]:.2f}  (std {std_cva[-1]:.2f})")
    rel_change = abs(mean_cva[-1] - mean_cva[-2]) / mean_cva[-1]
    print(f"[convergence] relative change 10000->20000 paths: {rel_change:.4%}")


# ---------------------------------------------------------------------------
# 2. Sensitivity: CVA vs CDS spread, CVA vs sigma
# ---------------------------------------------------------------------------

def plot_sensitivity():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- CVA vs CDS spread ---
    spreads = np.linspace(0.0, 0.05, 11)  # 0 to 500 bps
    cvas_spread = []
    for spread in spreads:
        result = run_cva(
            notional=NOTIONAL, fixed_rate=FIXED_RATE, payment_times=PAYMENT_TIMES,
            kappa=KAPPA, theta=THETA, sigma=SIGMA, r0=R0,
            cds_spread=spread, recovery_rate=RECOVERY,
            n_paths=5000, seed=42,
        )
        cvas_spread.append(result["cva"])

    axes[0].plot(spreads * 1e4, cvas_spread, "o-", color="darkgreen")
    axes[0].set_xlabel("CDS spread (bps)")
    axes[0].set_ylabel("CVA ($)")
    axes[0].set_title("CVA vs CDS Spread\n(expect ~linear)")
    axes[0].grid(alpha=0.3)

    # check approximate linearity via correlation with a straight line
    corr = np.corrcoef(spreads, cvas_spread)[0, 1]
    axes[0].text(0.05, 0.9, f"corr w/ linear fit: {corr:.4f}",
                  transform=axes[0].transAxes, fontsize=9,
                  bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # --- CVA vs sigma ---
    sigmas = np.linspace(0.0, 0.03, 11)
    cvas_sigma = []
    for sig in sigmas:
        # NOTE: fixed_rate is held at the sigma=SIGMA par rate throughout,
        # NOT recomputed per sigma. See note printed below.
        result = run_cva(
            notional=NOTIONAL, fixed_rate=FIXED_RATE, payment_times=PAYMENT_TIMES,
            kappa=KAPPA, theta=THETA, sigma=sig, r0=R0,
            cds_spread=CDS_SPREAD, recovery_rate=RECOVERY,
            n_paths=5000, seed=42,
        )
        cvas_sigma.append(result["cva"])

    axes[1].plot(sigmas, cvas_sigma, "o-", color="darkorange")
    axes[1].set_xlabel("sigma (short rate volatility)")
    axes[1].set_ylabel("CVA ($)")
    axes[1].set_title("CVA vs Rate Volatility\n(expect increasing)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "sensitivity.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"[sensitivity] saved to {path}")
    print(f"[sensitivity] CVA(spread=0) = {cvas_spread[0]:.4f} (should be ~0)")
    print(f"[sensitivity] CVA vs spread correlation with linear: {corr:.4f}")
    print(f"[sensitivity] CVA(sigma=0)  = {cvas_sigma[0]:.4f}")
    print(f"[sensitivity] CVA(sigma=0.03) = {cvas_sigma[-1]:.4f}")
    is_monotonic = all(b >= a for a, b in zip(cvas_sigma, cvas_sigma[1:]))
    print(f"[sensitivity] CVA monotonically increasing in sigma: {is_monotonic}")
    print(
        "[sensitivity] NOTE: fixed_rate is held constant at the sigma=0.01 par "
        "rate for all sigma values. At sigma=0, this means the swap is NOT "
        "at-par (since the par rate itself depends on sigma), so CVA(sigma=0) "
        "is not exactly 0 -- it reflects the off-market value of a swap struck "
        "at a rate that was fair under a different (higher) volatility "
        "assumption. This is intentional: holding the CONTRACT fixed while "
        "varying the MODEL is the realistic scenario (you don't re-paper a "
        "trade every time your vol estimate changes)."
    )


# ---------------------------------------------------------------------------
# 3. Deterministic baseline (sigma=0, r0=theta): exact closed-form match
# ---------------------------------------------------------------------------

def plot_deterministic_baseline():
    from models.swap import price_swap  # local import to avoid clutter above

    r0_det = THETA
    sigma_det = 0.0
    fixed_rate_det = par_swap_rate(PAYMENT_TIMES, KAPPA, THETA, sigma_det, r0_det, t=0.0)

    result = run_cva(
        notional=NOTIONAL, fixed_rate=fixed_rate_det, payment_times=PAYMENT_TIMES,
        kappa=KAPPA, theta=THETA, sigma=sigma_det, r0=r0_det,
        cds_spread=CDS_SPREAD, recovery_rate=RECOVERY,
        n_paths=2000, seed=1,
    )

    time_grid = result["time_grid"]
    closed_form = np.array([
        max(price_swap(THETA, t, fixed_rate_det, PAYMENT_TIMES, NOTIONAL, KAPPA, THETA, sigma_det), 0.0)
        for t in time_grid
    ])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_grid, result["epe"], "o-", label="Simulated EPE", color="steelblue", markersize=8)
    ax.plot(time_grid, result["pfe"], "x--", label="Simulated PFE", color="steelblue", alpha=0.5)
    ax.plot(time_grid, closed_form, "-", label="Closed-form max(V_swap, 0)", color="darkred", linewidth=2)

    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Exposure ($)")
    ax.set_title("Deterministic Baseline (sigma=0, r0=theta)\nSimulated EPE/PFE vs Closed-Form Swap Value")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.text(
        0.02, 0.95,
        "V_swap(t) = 0 for ALL t, not just t=0.\n"
        "Reason: under a flat curve (r=theta const),\n"
        "P(t,T) depends only on (T-t), so the par\n"
        "rate is tenor-independent for a uniform-tau\n"
        "schedule: R = 2(1/x - 1), x = e^(-theta*tau).\n"
        "Max |EPE - closed form| ~ 1e-11 (float noise).",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
    )

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "deterministic_baseline.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    max_abs_diff = np.max(np.abs(result["epe"] - closed_form))
    print(f"[deterministic] saved to {path}")
    print(f"[deterministic] max |EPE - closed_form| = {max_abs_diff:.10f}")
    print(f"[deterministic] EPE == PFE everywhere: {np.allclose(result['epe'], result['pfe'])}")


def plot_convexity_gap():
    """
    For the base case (sigma=0.01, r0=0.05), compare:
      - Simulated EPE/PFE (from full Monte Carlo)
      - "Exposure along the mean rate path": price the swap deterministically
        at each t using r_mean(t) = theta + (r0-theta)*e^{-kappa*t}, the
        analytical expectation of r(t) under Vasicek -- then take max(.,0).

    Jensen's inequality guarantees E[max(V(r),0)] >= max(E[V(r)],0) when
    V is convex in r (which it is here, via the exponential discount
    factors). So EPE should sit AT OR ABOVE this "mean path" curve
    everywhere. The gap between them is the part of EPE that exists
    PURELY because of rate uncertainty (sigma>0) -- this is the same
    effect driving the increasing CVA-vs-sigma curve in plot_sensitivity.
    """
    from models.swap import price_swap

    result = run_cva(
        notional=NOTIONAL, fixed_rate=FIXED_RATE, payment_times=PAYMENT_TIMES,
        kappa=KAPPA, theta=THETA, sigma=SIGMA, r0=R0,
        cds_spread=CDS_SPREAD, recovery_rate=RECOVERY,
        n_paths=20_000, seed=1,
    )
    time_grid = result["time_grid"]

    mean_path_exposure = np.array([
        max(price_swap(
            THETA + (R0 - THETA) * np.exp(-KAPPA * t),
            t, FIXED_RATE, PAYMENT_TIMES, NOTIONAL, KAPPA, THETA, SIGMA,
        ), 0.0)
        for t in time_grid
    ])

    gap = result["epe"] - mean_path_exposure

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_grid, result["pfe"], "x--", label="Simulated PFE (95th pct)", color="steelblue", alpha=0.5)
    ax.plot(time_grid, result["epe"], "o-", label="Simulated EPE", color="steelblue", markersize=6)
    ax.plot(time_grid, mean_path_exposure, "-", label="Exposure along mean rate path\n(no rate uncertainty)", color="darkred", linewidth=2)
    ax.fill_between(time_grid, mean_path_exposure, result["epe"], alpha=0.15, color="darkred", label="Convexity gap")

    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Exposure ($)")
    ax.set_title(
        "Convexity Gap: EPE >= Exposure(E[r(t)])\n"
        "(Jensen's inequality -- this is WHY CVA increases with sigma)"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "convexity_gap.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)

    print(f"[convexity] saved to {path}")
    print(f"[convexity] gap (EPE - mean-path exposure) by timestep:")
    print(f"  {np.round(gap, 1)}")
    print(f"[convexity] all gaps >= 0: {np.all(gap >= -1e-6)}")


if __name__ == "__main__":
    print(f"Par fixed rate (sigma={SIGMA}): {FIXED_RATE:.6f}\n")
    plot_convergence()
    print()
    plot_sensitivity()
    print()
    plot_convexity_gap()
    print()
    plot_deterministic_baseline()