"""Try several LLM providers in order until one answers.

The judge runs Groq -> Gemini -> local. Free tiers rate-limit and occasionally go down;
a diagnosis run should degrade to the next provider rather than lose the verdict.

Caching note
------------
Wrap the *whole chain* in one :class:`~ffx.llm.replay.ReplayCache`, not each provider.
The chain's ``provider``/``model`` are a fixed roster string, so the cache key does not
depend on which backend happened to answer. If Gemini answered because Groq was
rate-limited, a replay returns that answer without touching the network - whereas
per-provider caches would miss on Groq and re-try it live on every run.

The response still records who actually answered (``response.provider`` and
``response.extra["answered_by"]``), and that survives the cache round-trip.
"""

from __future__ import annotations

from typing import Any, Optional

from ffx.llm.base import LLMError, LLMResponse


class FallbackClient:
    def __init__(self, clients: list[Any]) -> None:
        if not clients:
            raise ValueError("FallbackClient needs at least one client")
        self.clients = clients
        self.provider = "fallback[" + ">".join(c.provider for c in clients) + "]"
        self.model = "|".join(f"{c.provider}:{c.model}" for c in clients)
        self.failures: dict[str, int] = {c.provider: 0 for c in clients}

    def complete(
        self,
        prompt: str,
        *,
        schema: Optional[dict[str, Any]] = None,
        system: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        errors: list[str] = []
        for client in self.clients:
            try:
                response = client.complete(prompt, schema=schema, system=system, max_tokens=max_tokens)
            except LLMError as exc:
                # Any provider error (429, 5xx, timeout, bad key) moves on to the next.
                self.failures[client.provider] += 1
                errors.append(f"{client.provider}: {str(exc)[:200]}")
                continue
            response.provider = client.provider
            response.extra = {**response.extra, "answered_by": client.provider, "fell_through": errors}
            return response
        raise LLMError("every judge provider failed: " + " | ".join(errors))

    def close(self) -> None:
        for client in self.clients:
            closer = getattr(client, "close", None)
            if callable(closer):
                closer()
