"""Optional OpenTelemetry bridge.

The custom span model in :mod:`ffx.tracing.span` is the source of truth. OTel span
attributes are meant to be small primitives, and the analyzer needs full prompts, raw
responses and multi-KB payloads to be queryable - so those stay in ffx's own store.

This bridge re-emits each finished trace as OTel spans carrying identifiers, timings,
token counts, status and confidence. That is enough to view a pipeline run in Jaeger or
any OTLP backend, and keeps "OpenTelemetry-compatible" an honest claim, without bending
the analyzer around OTel's data model.

Off by default. Enable with ``FFX_OTEL=1`` after ``pip install -r requirements-otel.txt``.
Set ``OTEL_EXPORTER_OTLP_ENDPOINT`` to point at a collector (Jaeger all-in-one listens
on http://localhost:4318); without it spans are printed to the console.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from ffx.tracing.span import SpanStatus, Trace


def _to_ns(dt: Any) -> int:
    return int(dt.timestamp() * 1_000_000_000)


class OTelExporter:
    def __init__(self, service_name: str = "ffx-pipeline") -> None:
        try:
            from opentelemetry import trace as ot_trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,
                ConsoleSpanExporter,
                SimpleSpanProcessor,
            )
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "OpenTelemetry is optional: pip install -r requirements-otel.txt"
            ) from exc

        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        else:
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

        self._ot_trace = ot_trace
        self._provider = provider
        self._tracer = provider.get_tracer("ffx")

    def export(self, trace: Trace) -> None:
        ot = self._ot_trace
        end = trace.ended_at or trace.started_at
        root = self._tracer.start_span(
            f"pipeline {trace.doc_id}",
            start_time=_to_ns(trace.started_at),
            attributes={
                "ffx.trace_id": trace.trace_id,
                "ffx.doc_id": trace.doc_id,
                "ffx.run_id": trace.run_id,
                "ffx.status": trace.status.value,
                "ffx.total_tokens": trace.total_tokens,
            },
        )
        parent_ctx = ot.set_span_in_context(root)
        for span in trace.spans:
            attributes: dict[str, Any] = {
                "ffx.span_id": span.span_id,
                "ffx.step": span.step_name.value,
                "ffx.status": span.status.value,
                "ffx.attempts": span.attempt_count,
                "gen_ai.request.model": span.model,
                "gen_ai.usage.input_tokens": span.tokens_in,
                "gen_ai.usage.output_tokens": span.tokens_out,
            }
            if span.self_confidence is not None:
                attributes["ffx.self_confidence"] = span.self_confidence
            child = self._tracer.start_span(
                span.step_name.value,
                context=parent_ctx,
                start_time=_to_ns(span.started_at),
                attributes=attributes,
            )
            if span.status == SpanStatus.ERROR:
                child.set_status(ot.Status(ot.StatusCode.ERROR, span.error.message if span.error else ""))
            child.end(end_time=_to_ns(span.ended_at or span.started_at))
            parent_ctx = ot.set_span_in_context(child)
        root.end(end_time=_to_ns(end))

    def shutdown(self) -> None:
        self._provider.shutdown()


def maybe_exporter() -> Optional[OTelExporter]:
    if os.environ.get("FFX_OTEL") == "1":
        return OTelExporter()
    return None
