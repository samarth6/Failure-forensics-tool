"""Context-manager instrumentation.

Usage::

    tracer = Tracer(store)
    with tracer.trace(doc_id="doc_001", run_id=run_id) as trace:
        with tracer.span(StepName.INTAKE, input_payload={...}) as rec:
            outcome = complete_typed(...)
            rec.record(outcome, prompt=prompt, system=system)

The active trace lives in a ``ContextVar`` so step functions never need a tracer handed
to them explicitly - they stay pure, typed functions, which is the point of Phase 1.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional

from pydantic import BaseModel

from ffx.llm.repair import ParseOutcome
from ffx.models import STEP_ORDER, StepName, dump, utcnow
from ffx.tracing.span import Attempt, Span, SpanError, SpanStatus, Trace, TraceStatus

_current_trace: ContextVar[Optional[Trace]] = ContextVar("ffx_current_trace", default=None)


def current_trace() -> Optional[Trace]:
    return _current_trace.get()


class StepFailed(RuntimeError):
    """A step produced no usable output; downstream steps cannot run."""

    def __init__(self, step: StepName, message: str) -> None:
        super().__init__(f"{step.value}: {message}")
        self.step = step


class SpanRecorder:
    """Handle yielded by :meth:`Tracer.span` for the step to report what happened."""

    def __init__(self, span: Span) -> None:
        self.span = span

    def record(
        self,
        outcome: ParseOutcome,
        *,
        prompt: str,
        system: str = "",
        provider: str = "",
        model: str = "",
    ) -> None:
        span = self.span
        span.prompt = prompt
        span.system_prompt = system
        span.provider = provider
        span.model = model
        span.attempts = [
            Attempt(
                index=a.index,
                raw_response=a.raw_response,
                ok=a.ok,
                error=a.error,
                latency_ms=a.latency_ms,
                from_cache=a.from_cache,
            )
            for a in outcome.attempts
        ]
        span.tokens_in = sum(a.tokens_in for a in outcome.attempts)
        span.tokens_out = sum(a.tokens_out for a in outcome.attempts)
        span.from_cache = bool(outcome.attempts) and all(a.from_cache for a in outcome.attempts)
        span.raw_response = outcome.attempts[-1].raw_response if outcome.attempts else ""

        if outcome.parsed is None:
            span.status = SpanStatus.ERROR
            error_type = "LLMTransportError" if outcome.transport_error else "SchemaValidationError"
            span.error = SpanError(type=error_type, message=outcome.last_error or "no output")
            return

        span.output_payload = dump(outcome.parsed)
        span.self_confidence = getattr(outcome.parsed, "confidence", None)
        # Output that only validated after a re-ask is usable but suspect.
        span.status = SpanStatus.DEGRADED if outcome.attempt_count > 1 else SpanStatus.OK

    def set_output(self, output: BaseModel) -> None:
        """For deterministic (non-LLM) steps, and for counterfactual splices."""
        self.span.output_payload = dump(output)
        self.span.self_confidence = getattr(output, "confidence", None)
        self.span.status = SpanStatus.OK


class Tracer:
    def __init__(self, store: Any = None, exporter: Any = None) -> None:
        self.store = store
        self.exporter = exporter

    @contextmanager
    def trace(
        self,
        doc_id: str,
        *,
        run_id: str = "",
        pipeline_version: str = "",
        metadata: Optional[dict[str, Any]] = None,
        parent_trace_id: Optional[str] = None,
    ) -> Iterator[Trace]:
        trace = Trace(
            doc_id=doc_id,
            run_id=run_id,
            pipeline_version=pipeline_version,
            metadata=metadata or {},
            parent_trace_id=parent_trace_id,
        )
        token = _current_trace.set(trace)
        started = time.perf_counter()
        try:
            yield trace
        except StepFailed:
            trace.status = TraceStatus.ERROR
            # Swallowed deliberately: a step that cannot produce output is a finding to
            # record, not a reason to abort a 50-document batch.
        finally:
            trace.ended_at = utcnow()
            trace.duration_ms = int((time.perf_counter() - started) * 1000)
            if trace.status == TraceStatus.RUNNING:
                trace.status = TraceStatus.OK
            _current_trace.reset(token)
            if self.store is not None:
                self.store.save_trace(trace)
            if self.exporter is not None:
                self.exporter.export(trace)

    @contextmanager
    def span(self, step: StepName, input_payload: Any) -> Iterator[SpanRecorder]:
        trace = current_trace()
        if trace is None:
            raise RuntimeError("tracer.span() used outside tracer.trace()")

        parent = trace.spans[-1].span_id if trace.spans else None
        span = Span(
            trace_id=trace.trace_id,
            parent_span_id=parent,
            step_name=step,
            step_index=STEP_ORDER.index(step),
            input_payload=dump(input_payload) if input_payload is not None else {},
        )
        trace.spans.append(span)
        recorder = SpanRecorder(span)
        started = time.perf_counter()
        try:
            yield recorder
        except StepFailed:
            raise
        except Exception as exc:  # noqa: BLE001 - every failure must land in the trace
            span.status = SpanStatus.ERROR
            span.error = SpanError(type=type(exc).__name__, message=str(exc)[:1000])
            raise StepFailed(step, str(exc)) from exc
        finally:
            span.ended_at = utcnow()
            span.duration_ms = int((time.perf_counter() - started) * 1000)

        if span.status == SpanStatus.ERROR:
            raise StepFailed(step, span.error.message if span.error else "no output")
