"""Bounded parse-and-repair loop.

A small model will occasionally emit JSON that satisfies the grammar but not the
semantic constraints Pydantic enforces (``confidence`` outside 1-5 is the common one).
Rather than crash, we re-ask once or twice with the validation error attached.

Every attempt is returned, not just the winner. This matters: an exhausted repair loop
*is* the Prompt Failure category, and the attempt history is the evidence for it. A
silently-retried call that eventually succeeds would erase a real quality signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, ValidationError

from ffx.llm.base import LLMError, LLMResponse


@dataclass
class Attempt:
    index: int
    prompt: str
    raw_response: str
    ok: bool
    error: str = ""
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    from_cache: bool = False


@dataclass
class ParseOutcome:
    """Result of asking a model for a typed object."""

    parsed: Optional[BaseModel]
    attempts: list[Attempt] = field(default_factory=list)
    response: Optional[LLMResponse] = None
    # Set when the *transport* failed (provider down, out of memory, timeout) rather
    # than the model producing bad output. The two must never be confused: an OOM on a
    # memory-starved box is not a prompt failure, and diagnosing it as one would blame
    # the pipeline for the hardware.
    transport_error: str = ""

    @property
    def ok(self) -> bool:
        return self.parsed is not None

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def last_error(self) -> str:
        for attempt in reversed(self.attempts):
            if not attempt.ok:
                return attempt.error
        return ""


def complete_typed(
    client: Any,
    *,
    prompt: str,
    output_model: type[BaseModel],
    schema: dict[str, Any],
    system: Optional[str] = None,
    max_tokens: int = 2048,
    max_repair_attempts: int = 2,
) -> ParseOutcome:
    """Ask ``client`` for JSON, validate it, and re-ask on failure up to a limit."""
    outcome = ParseOutcome(parsed=None)
    current_prompt = prompt

    for index in range(max_repair_attempts + 1):
        try:
            response = client.complete(
                current_prompt, schema=schema, system=system, max_tokens=max_tokens
            )
        except LLMError as exc:
            outcome.attempts.append(
                Attempt(index=index, prompt=current_prompt, raw_response="", ok=False, error=str(exc))
            )
            outcome.transport_error = str(exc)
            break

        try:
            parsed = output_model.model_validate_json(response.text)
        except ValidationError as exc:
            error = _summarise(exc)
            outcome.attempts.append(
                Attempt(
                    index=index,
                    prompt=current_prompt,
                    raw_response=response.text,
                    ok=False,
                    error=error,
                    latency_ms=response.latency_ms,
                    tokens_in=response.tokens_in,
                    tokens_out=response.tokens_out,
                    from_cache=response.from_cache,
                )
            )
            outcome.response = response
            current_prompt = (
                f"{prompt}\n\nYour previous answer was rejected:\n{response.text[:1500]}\n\n"
                f"Errors:\n{error}\n\nReturn corrected JSON only."
            )
            continue

        outcome.attempts.append(
            Attempt(
                index=index,
                prompt=current_prompt,
                raw_response=response.text,
                ok=True,
                latency_ms=response.latency_ms,
                tokens_in=response.tokens_in,
                tokens_out=response.tokens_out,
                from_cache=response.from_cache,
            )
        )
        outcome.parsed = parsed
        outcome.response = response
        break

    return outcome


def _summarise(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors()[:6]:
        location = ".".join(str(p) for p in err.get("loc", ()))
        parts.append(f"- {location}: {err.get('msg')}")
    return "\n".join(parts)
