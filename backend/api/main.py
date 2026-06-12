"""
FastAPI application entry point for the CVA engine.

Run with:
    uvicorn api.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

app = FastAPI(
    title="CVA Engine",
    description="Monte Carlo CVA calculator for vanilla interest rate swaps under Vasicek.",
    version="0.1.0",
)

# Permissive CORS for local development. Tighten allow_origins before
# deploying behind a real frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"} 