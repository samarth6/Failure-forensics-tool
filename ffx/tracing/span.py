"""Span and Trace: the unit of observability.

A Trace is one document's journey through the pipeline. A Span is one step's execution
within it, captured completely enough that the step can be *replayed* - input, prompt,
raw response, parsed output, every repair attempt, timings and token counts.

Signals are attached to spans after the fact by the analyzer. They are append-only:
the analyzer adds evidence, it never rewrites what the pipeline recorded.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from ffx.models import FailureCategory, StepName, utcnow


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class SpanStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"  # produced output, but only after repair attempts
    ERROR = "error"  # produced no usable output


class TraceStatus(str, Enum):
    RUNNING = "running"
    OK = "ok"
    FAILED = "failed"  # completed, but the analyzer judged the result bad
    ERROR = "error"  # a step could not produce output at all


class SignalKind(str, Enum):
    DETERMINISTIC = "deterministic"
    SELF_REPORT = "self_report"
    LLM_JUDGE = "llm_judge"
    HUMAN = "human"


class SignalSeverity(str, Enum):
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"


class Signal(BaseModel):
    """One piece of evidence about a span's quality.

    ``score`` is this signal's contribution to span quality on 0..1 (1 = healthy).
    ``weight`` expresses how much the signal should be trusted; deterministic signals
    carry far more weight than self-reported confidence, by design.
    """

    name: str
    kind: SignalKind
    severity: SignalSeverity
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(default=1.0, ge=0.0)
    detail: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)
    # Which failure category this signal is evidence *for* when it fails. The taxonomy
    # is then read straight off the evidence rather than decided by a separate model call.
    implicates: Optional[FailureCategory] = None
    # A value the analyzer believes the step should have produced, when it can tell.
    proposed_correction: Optional[dict[str, Any]] = None


class Attempt(BaseModel):
    index: int
    raw_response: str
    ok: bool
    error: str = ""
    latency_ms: int = 0
    from_cache: bool = False


class SpanError(BaseModel):
    type: str
    message: str


class Span(BaseModel):
    span_id: str = Field(default_factory=lambda: new_id("span"))
    trace_id: str
    parent_span_id: Optional[str] = None
    step_name: StepName
    step_index: int

    started_at: datetime = Field(default_factory=utcnow)
    ended_at: Optional[datetime] = None
    duration_ms: int = 0
    status: SpanStatus = SpanStatus.OK

    input_payload: dict[str, Any] = Field(default_factory=dict)
    output_payload: Optional[dict[str, Any]] = None
    prompt: str = ""
    system_prompt: str = ""
    raw_response: str = ""

    provider: str = ""
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    attempts: list[Attempt] = Field(default_factory=list)
    from_cache: bool = False

    self_confidence: Optional[int] = None
    error: Optional[SpanError] = None
    signals: list[Signal] = Field(default_factory=list)

    # Counterfactual bookkeeping. A spliced span holds a *corrected* output injected by
    # the analyzer rather than one the model produced; a copied span is carried over
    # unchanged from the original trace. Neither may be counted as model behaviour.
    spliced: bool = False
    copied_from_span_id: Optional[str] = None

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)


class Trace(BaseModel):
    trace_id: str = Field(default_factory=lambda: new_id("trace"))
    doc_id: str
    run_id: str = ""
    pipeline_version: str = ""
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: Optional[datetime] = None
    duration_ms: int = 0
    status: TraceStatus = TraceStatus.RUNNING
    spans: list[Span] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # If this trace is a counterfactual re-run, the trace it was derived from.
    parent_trace_id: Optional[str] = None

    def span(self, step: StepName) -> Optional[Span]:
        for span in self.spans:
            if span.step_name == step:
                return span
        return None

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens_in + s.tokens_out for s in self.spans)
