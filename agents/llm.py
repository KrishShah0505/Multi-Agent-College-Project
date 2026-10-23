"""Thin LLM client. Agents only need `complete_json(system, user) -> dict`."""
import json
import time
from typing import Protocol

import httpx

from agents import config


class LLMError(Exception):
    """The LLM call failed or did not return usable JSON."""


class LLMFatalError(LLMError):
    """Retrying cannot help (bad key, unknown model, bad request)."""


class LLMClient(Protocol):
    calls: int
    tokens: int

    def complete_json(self, system: str, user: str) -> dict: ...


class GroqClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.model = model or config.GROQ_MODEL
        if not self.api_key:
            raise LLMError("GROQ_API_KEY is not set.")
        self.calls = 0
        self.tokens = 0
        self._last_call = 0.0

    def complete_json(self, system: str, user: str) -> dict:
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        for attempt in range(4):
            wait = config.LLM_MIN_INTERVAL_S - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()
            self.calls += 1
            try:
                r = httpx.post(config.GROQ_URL, json=payload, headers=headers,
                               timeout=config.LLM_TIMEOUT_S)
            except httpx.HTTPError as exc:
                raise LLMError(f"Groq request failed: {exc}") from exc

            if r.status_code == 429 and attempt < 3:  # rate limited: back off
                time.sleep(float(r.headers.get("retry-after", 2 ** (attempt + 1))))
                continue
            if r.status_code in (400, 401, 403, 404):
                raise LLMFatalError(f"Groq HTTP {r.status_code}: {r.text[:300]}")
            if r.status_code != 200:
                raise LLMError(f"Groq HTTP {r.status_code}: {r.text[:200]}")

            body = r.json()
            self.tokens += body.get("usage", {}).get("total_tokens", 0)
            try:
                return json.loads(body["choices"][0]["message"]["content"])
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                raise LLMError("Model did not return valid JSON.") from exc
        raise LLMError("Groq rate limit not cleared after retries.")
