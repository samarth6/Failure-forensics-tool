"""Data model for a diagnosis: the analyzer's structured answer to "where did this break?"."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from ffx.models import FailureCategory, StepName, utcnow


class SpanRole(str, Enum):
    HEALTHY = "healthy"
    LOW_CONFIDENCE = "low_confidence"
    ROOT_CAUSE = "root_cause"
    CANDIDATE = "candidate"  # damaged independently, but not the top-ranked origin
    PROPAGATION = "propagation"  # below threshold only because its input was bad
    NOT_RUN = "not_run"


class SpanAssessment(BaseModel):
    span_id: str
    step_name: StepName
    q_in: float
    local_quality: float
    q_out: float
    local_damage: float
    role: SpanRole
    failed_signals: list[str] = Field(default_factory=list)


class RootCauseCandidate(BaseModel):
    """A step that failed its *own* checks.

    ``own_damage`` is ``1 - local_quality`` and deliberately ignores upstream quality.
    Ranking on ``q_in - q_out`` instead (as a first version did) makes damage scale with
    ``q_in``, so once an upstream step has dragged quality down, a later step failing
    badly on its own can barely register - and an independent second failure gets
    misfiled as mere propagation.
    """

    rank: int
    span_id: str
    step_name: StepName
    own_damage: float
    category: FailureCategory
    subtype: str


class EvidenceLink(BaseModel):
    order: int
    span_id: str
    step_name: StepName
    role: str  # "origin" | "propagation" | "independent" | "symptom"
    claim: str
    signal: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)


class CounterfactualVerdict(str, Enum):
    CONFIRMED = "confirmed"  # correcting the step fixed every final-output fact it touches
    PARTIAL = "partial"  # fixed some of them
    REFUTED = "refuted"  # fixed none: the final output does not depend on this step's error
    CONTAINED = "contained"  # the error is real but never reached the final output
    INCONCLUSIVE = "inconclusive"  # the re-run itself crashed (e.g. model server OOM)
    NOT_APPLICABLE = "not_applicable"  # terminal step, or no correction available


class CounterfactualResult(BaseModel):
    """Did intervening at the suspected step change the final output as predicted?

    Fidelity is measured against one fixed reference - the corrected facts - for both
    the original trace and the re-run. Comparing local span quality instead would be
    circular: splicing a corrected output removes that span's own penalty, so "quality
    recovered" would be guaranteed and the test would confirm every diagnosis.
    """

    verdict: CounterfactualVerdict
    splice_step: StepName
    correction_source: str = ""  # human | answer_key | judge | rule
    corrected_output: Optional[dict[str, Any]] = None
    counterfactual_trace_id: Optional[str] = None
    fidelity_before: Optional[float] = None
    fidelity_after: Optional[float] = None
    reference: dict[str, Any] = Field(default_factory=dict)
    checks_before: list[dict[str, Any]] = Field(default_factory=list)
    checks_after: list[dict[str, Any]] = Field(default_factory=list)
    # The final-output checks this correction is predicted to fix - the actual test.
    implicated_checks: list[str] = Field(default_factory=list)
    fixed_checks: list[str] = Field(default_factory=list)
    # Checks still failing after the fix that this correction does not touch: evidence
    # of a second, independent failure rather than a refutation of this one.
    remaining_failures: list[str] = Field(default_factory=list)
    new_downstream_failures: list[str] = Field(default_factory=list)
    note: str = ""


class Diagnosis(BaseModel):
    trace_id: str
    doc_id: str
    created_at: datetime = Field(default_factory=utcnow)

    is_failure: bool
    terminal_quality: float
    root_span_id: Optional[str] = None
    root_step: Optional[StepName] = None
    category: FailureCategory = FailureCategory.NO_FAILURE
    subtype: str = ""
    confidence: float = 0.0
    explanation: str = ""

    candidates: list[RootCauseCandidate] = Field(default_factory=list)
    assessments: list[SpanAssessment] = Field(default_factory=list)
    evidence_chain: list[EvidenceLink] = Field(default_factory=list)
    propagation_path: list[str] = Field(default_factory=list)
    counterfactuals: list[CounterfactualResult] = Field(default_factory=list)

    judge_provider: str = ""
    judge_calls: int = 0
    diagnose_ms: int = 0

    @property
    def counterfactual(self) -> Optional[CounterfactualResult]:
        """The counterfactual for the finally-chosen root cause, if one was run."""
        for cf in self.counterfactuals:
            if cf.splice_step == self.root_step:
                return cf
        return None
