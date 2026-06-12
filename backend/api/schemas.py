"""
Pydantic schemas for the CVA API.
"""

from typing import Optional

from pydantic import BaseModel, Field


class CVARequest(BaseModel):
    # Swap parameters
    notional: float = Field(..., gt=0, description="Swap notional")
    tenor_years: float = Field(..., gt=0, description="Swap maturity in years")
    payment_freq_months: int = Field(
        6, gt=0, description="Payment/reset frequency in months (e.g. 6 for semi-annual)"
    )
    fixed_rate: Optional[float] = Field(
        None,
        description="Fixed leg rate. If omitted, the par swap rate at t=0 is used.",
    )

    # Vasicek parameters
    kappa: float = Field(..., gt=0, description="Mean reversion speed")
    theta: float = Field(..., description="Long-run mean short rate")
    sigma: float = Field(..., ge=0, description="Short rate volatility")
    r0: float = Field(..., description="Initial short rate")

    # Credit parameters
    cds_spread: float = Field(..., ge=0, description="Counterparty CDS spread (decimal, e.g. 0.015)")
    recovery_rate: float = Field(0.40, ge=0, lt=1, description="Assumed recovery rate on default")

    # Simulation parameters
    n_paths: int = Field(1000, gt=0, le=50_000, description="Number of Monte Carlo paths")
    seed: Optional[int] = Field(None, description="Optional RNG seed for reproducibility")
    pfe_quantile: float = Field(0.95, gt=0, lt=1, description="Quantile used for PFE")


class CVAResponse(BaseModel):
    time_grid: list[float]
    epe: list[float]
    pfe: list[float]
    cva: float
    hazard: float
    fixed_rate_used: float
    par_rate: float