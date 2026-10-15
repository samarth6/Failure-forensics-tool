"""One OpenAI-compatible adapter, covering every free tier worth using.

Google AI Studio, Groq and OpenRouter all speak the OpenAI chat-completions shape, so
they differ only by ``base_url`` and model name:

===========  ===========================================================
provider     base_url
===========  ===========================================================
gemini       https://generativelanguage.googleapis.com/v1beta/openai/
groq         https://api.groq.com/openai/v1
openrouter   https://openrouter.ai/api/v1
===========  ===========================================================

Only the judge uses this, and only on traces that already look like failures, so free
tier rate limits are not a practical constraint. If no key is configured the caller
falls back to the local model - see :func:`ffx.analyzer.judge.build_judge`.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

import httpx

from ffx.llm.base import LLMError, LLMResponse

PROVIDER_BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


class OpenAICompatibleClient:
    provider = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout_s: float = 120.0,
        label: str = "openai_compatible",
    ) -> None:
        if not base_url or not model or not api_key:
            raise LLMError("cloud judge requires base_url, model and api_key")
        self.model = model
        self.provider = label
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_s,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
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
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "ffx_response", "schema": schema, "strict": True},
            }

        started = time.perf_counter()
        response = self._post(payload)

        # Not every free tier implements json_schema; json_object is far more widely
        # supported. Degrade rather than fail, and let schema validation catch the rest.
        if response.status_code == 400 and schema is not None:
            payload["response_format"] = {"type": "json_object"}
            payload["messages"][-1]["content"] = (
                prompt + "\n\nRespond with JSON matching this schema:\n" + json.dumps(schema)
            )
            response = self._post(payload)

        if response.status_code != 200:
            raise LLMError(f"cloud judge returned {response.status_code}: {response.text[:400]}")

        body = response.json()
        usage = body.get("usage") or {}
        try:
            text = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise LLMError(f"unexpected cloud judge response shape: {str(body)[:300]}") from exc

        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.provider,
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    def list_models(self) -> list[str]:
        """Model ids this key can use, from the provider's OpenAI-compatible /models.

        Free-tier model names are retired regularly, so `doctor` shows this list when a
        configured model is rejected instead of leaving the user to guess a replacement.
        """
        try:
            response = self._client.get("/models")
        except httpx.HTTPError as exc:
            raise LLMError(f"could not list models: {exc}") from exc
        if response.status_code != 200:
            raise LLMError(f"listing models returned {response.status_code}: {response.text[:200]}")
        ids = [m.get("id", "") for m in response.json().get("data", [])]
        # Gemini prefixes ids with "models/"; the chat endpoint accepts them without it.
        return sorted({i.removeprefix("models/") for i in ids if i})

    def _post(self, payload: dict[str, Any]) -> httpx.Response:
        try:
            return self._client.post("/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"cloud judge request failed: {exc}") from exc

    def close(self) -> None:
        self._client.close()
