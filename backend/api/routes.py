"""
API routes for the CVA engine.
"""

from fastapi import APIRouter, HTTPException

from api.schemas import CVARequest, CVAResponse
from core.cva import run_cva
from models.swap import generate_payment_schedule, par_swap_rate

router = APIRouter()


@router.post("/cva", response_model=CVAResponse)
def compute_cva(request: CVARequest) -> CVAResponse:
    """
    Run the full CVA pipeline for a vanilla fixed-for-floating IR swap:
    Vasicek simulation -> exposure matrix -> EPE/PFE -> CVA.

    If `fixed_rate` is not supplied, the par swap rate at inception
    (computed from r0) is used -- i.e. a fairly-priced swap at t=0.
    """
    try:
        payment_times = generate_payment_schedule(
            request.tenor_years, request.payment_freq_months
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    par_rate = par_swap_rate(
        payment_times, request.kappa, request.theta, request.sigma, request.r0, t=0.0
    )
    fixed_rate = request.fixed_rate if request.fixed_rate is not None else par_rate

    try:
        result = run_cva(
            notional=request.notional,
            fixed_rate=fixed_rate,
            payment_times=payment_times,
            kappa=request.kappa,
            theta=request.theta,
            sigma=request.sigma,
            r0=request.r0,
            cds_spread=request.cds_spread,
            recovery_rate=request.recovery_rate,
            n_paths=request.n_paths,
            seed=request.seed,
            pfe_quantile=request.pfe_quantile,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return CVAResponse(
        time_grid=result["time_grid"].tolist(),
        epe=result["epe"].tolist(),
        pfe=result["pfe"].tolist(),
        cva=result["cva"],
        hazard=result["hazard"],
        fixed_rate_used=fixed_rate,
        par_rate=par_rate,
    )