"""Provider-agnostic LLM interface.

An observability tool should not be welded to one vendor, so every call in this project
goes through :class:`LLMClient`. Three implementations exist:

* :class:`ffx.llm.ollama_client.OllamaClient` - local, free, grammar-constrained.
* :class:`ffx.llm.cloud_judge.OpenAICompatibleClient` - any OpenAI-shaped free tier
  (Google AI Studio, Groq, OpenRouter) selected purely by ``base_url``.
* :class:`ffx.llm.replay.ReplayCache` - a decorator over either of the above that makes
  repeat runs free and byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


class LLMError(RuntimeError):
    """Transport or provider failure. Distinct from a schema-validation failure, which
    is a *result* worth recording (the Prompt Failure category) rather than a crash."""


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    from_cache: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    """Anything that can turn a prompt plus a JSON schema into text."""

    provider: str
    model: str

    def complete(
        self,
        prompt: str,
        *,
        schema: Optional[dict[str, Any]] = None,
        system: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Return the model's raw text.

        ``schema``, when supplied, must be *enforced* by the implementation (grammar
        constraint or provider-side structured output) rather than merely requested in
        the prompt. Implementations do not parse or validate - that is the caller's job,
        because a validation failure is data the tracer needs to see.
        """
        ...
