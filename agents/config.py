"""Settings, read from environment variables (and agents/.env if present)."""
import os
from pathlib import Path

# CrewAI sends anonymous usage data and offers tracing by default; keep both off.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"))
except ImportError:  # python-dotenv is optional; plain environment variables still work
    pass

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Model ids on Groq change over time. Check console.groq.com/docs/models and
# override with GROQ_MODEL if this default has been retired.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_OPENAI_BASE = "https://api.groq.com/openai/v1"   # CrewAI talks to Groq through its OpenAI-compatible API

# "crewai" (default): the three agents run as a CrewAI crew inside a CrewAI Flow.
# "direct": the earlier framework-free implementation (kept as the single-prompt baseline
# and as a fallback).
AGENT_BACKEND = os.getenv("AGENT_BACKEND", "crewai").lower()
CREW_VERBOSE = os.getenv("CREW_VERBOSE", "false").lower() == "true"   # true prints each agent's work
GUARDRAIL_RETRIES = int(os.getenv("GUARDRAIL_RETRIES", "2"))

CVA_API_URL = os.getenv("CVA_API_URL", "http://localhost:8000")

MAX_RETRIES = int(os.getenv("AGENT_MAX_RETRIES", "2"))
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "30"))
# Minimum seconds between LLM calls (free tiers are rate limited per minute).
LLM_MIN_INTERVAL_S = float(os.getenv("LLM_MIN_INTERVAL_S", "0"))
