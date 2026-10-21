"""
FastAPI service for the agent layer.

Run from the repo root:
    uvicorn agents.app:app --reload --port 8001
"""
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents import config
from agents.llm import GroqClient, LLMError
from agents.orchestrator import EngineError, run_pipeline

app = FastAPI(title="CVA Agents", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ExplainRequest(BaseModel):
    params: dict[str, Any]          # same body as POST /cva on the engine
    question: Optional[str] = None  # what the reader wants to understand


@app.get("/health")
def health():
    return {"status": "ok", "model": config.GROQ_MODEL, "groq_key_set": bool(config.GROQ_API_KEY)}


@app.post("/explain")
def explain(req: ExplainRequest):
    try:
        llm = GroqClient()
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    try:
        return run_pipeline(req.params, llm, req.question)
    except EngineError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
