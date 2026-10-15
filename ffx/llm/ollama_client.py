"""Local inference via Ollama, with grammar-constrained structured output.

Always pass a schema produced by :func:`ffx.llm.schema.strict_schema` rather than a raw
``model_json_schema()`` - the reasoning, and the measurements behind it, are in that
module's docstring.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import httpx

from ffx.config import Settings, get_settings
from ffx.llm.base import LLMError, LLMResponse


class OllamaClient:
    provider = "ollama"

    def __init__(self, settings: Optional[Settings] = None, model: Optional[str] = None) -> None:
        self.settings = settings or get_settings()
        self.model = model or self.settings.pipeline_model
        self._client = httpx.Client(
            base_url=self.settings.ollama_host,
            timeout=self.settings.request_timeout_s,
        )

    def complete(
        self,
        prompt: str,
        *,
        schema: Optional[dict[str, Any]] = None,
        system: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                # seed + temperature=0 make the *first* recording reproducible. Run-to-run
                # determinism thereafter comes from the replay cache, which works for any
                # provider - including hosted models that no longer expose sampling knobs.
                "temperature": self.settings.temperature,
                "seed": self.settings.seed,
                "num_ctx": self.settings.num_ctx,
                "num_predict": max_tokens,
            },
        }
        if schema is not None:
            payload["format"] = schema

        started = time.perf_counter()
        try:
            response = self._client.post("/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama request failed: {exc}") from exc

        if response.status_code != 200:
            raise LLMError(f"ollama returned {response.status_code}: {response.text[:400]}")

        body = response.json()
        latency_ms = int((time.perf_counter() - started) * 1000)
        return LLMResponse(
            text=body.get("message", {}).get("content", ""),
            model=self.model,
            provider=self.provider,
            tokens_in=int(body.get("prompt_eval_count") or 0),
            tokens_out=int(body.get("eval_count") or 0),
            latency_ms=latency_ms,
            extra={"done_reason": body.get("done_reason")},
        )

    def close(self) -> None:
        self._client.close()

    def health(self) -> tuple[bool, str]:
        """Cheap readiness probe used by the CLI before a long corpus run."""
        try:
            tags = self._client.get("/api/tags", timeout=10.0)
            tags.raise_for_status()
        except httpx.HTTPError as exc:
            return False, f"Ollama unreachable at {self.settings.ollama_host}: {exc}"
        available = [m.get("name") for m in tags.json().get("models", [])]
        if self.model not in available:
            return False, f"model {self.model!r} not pulled. Available: {available or 'none'}"
        return True, f"ollama ok, model {self.model}"
