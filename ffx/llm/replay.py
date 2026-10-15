"""Content-addressed replay cache.

This is the component that makes Phase 5 regression tracking meaningful. A regression
run only tells you something if a re-run is comparable to the previous one; caching the
exact bytes a given (provider, model, prompt, schema, options) tuple produced guarantees
that, and it works for *any* provider - including hosted models that have removed
sampling controls and so cannot be pinned with a seed.

The secondary benefit is cost and time: the first corpus run takes ~40 minutes on this
hardware, and every subsequent one is near-instant and free.

Cache entries are plain JSON on disk so they can be inspected, diffed and deleted by
hand. Deleting ``data/llm_cache`` forces a genuine re-record.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

from ffx.config import CACHE_DIR
from ffx.llm.base import LLMResponse


def cache_key(
    *,
    provider: str,
    model: str,
    prompt: str,
    system: Optional[str],
    schema: Optional[dict[str, Any]],
    options: dict[str, Any],
) -> str:
    payload = json.dumps(
        {
            "provider": provider,
            "model": model,
            "prompt": prompt,
            "system": system or "",
            # sort_keys matters: an unstable dict ordering would silently miss the cache
            # on every run and quietly reintroduce nondeterminism.
            "schema": schema or {},
            "options": options,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ReplayCache:
    """Wraps any :class:`~ffx.llm.base.LLMClient` and memoises it to disk."""

    def __init__(self, inner: Any, cache_dir: Optional[Path] = None, enabled: bool = True) -> None:
        self.inner = inner
        self.provider = getattr(inner, "provider", "unknown")
        self.model = getattr(inner, "model", "unknown")
        self.cache_dir = cache_dir or CACHE_DIR
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def complete(
        self,
        prompt: str,
        *,
        schema: Optional[dict[str, Any]] = None,
        system: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        options = {"max_tokens": max_tokens}
        key = cache_key(
            provider=self.provider,
            model=self.model,
            prompt=prompt,
            system=system,
            schema=schema,
            options=options,
        )
        path = self._path(key)

        if self.enabled and path.exists():
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                self.hits += 1
                return LLMResponse(
                    text=record["text"],
                    model=record.get("model", self.model),
                    provider=record.get("provider", self.provider),
                    tokens_in=record.get("tokens_in", 0),
                    tokens_out=record.get("tokens_out", 0),
                    latency_ms=record.get("latency_ms", 0),
                    from_cache=True,
                    extra=record.get("extra", {}),
                )
            except (json.JSONDecodeError, KeyError):
                # A truncated or hand-edited entry should not abort a long run.
                path.unlink(missing_ok=True)

        self.misses += 1
        response = self.inner.complete(prompt, schema=schema, system=system, max_tokens=max_tokens)

        if self.enabled:
            record = {
                "text": response.text,
                "model": response.model,
                "provider": response.provider,
                "tokens_in": response.tokens_in,
                "tokens_out": response.tokens_out,
                "latency_ms": response.latency_ms,
                "extra": response.extra,
                "recorded_at": time.time(),
                # Kept purely so a human browsing the cache can tell entries apart.
                "prompt_preview": prompt[:300],
            }
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)

        return response

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}

    def close(self) -> None:
        closer = getattr(self.inner, "close", None)
        if callable(closer):
            closer()
