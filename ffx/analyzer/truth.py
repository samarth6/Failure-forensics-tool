"""Ground truth for the synthetic corpus - used to *score* the diagnoser, never to drive it.

The diagnosis engine sees signals only. This module looks at the answer key and works
out, independently, whether each trace really failed and where the damage really began.
Comparing the two gives the headline accuracy numbers: detection precision/recall and
root-step accuracy.

Keeping the answer key out of the root-cause *decision* is what makes those numbers
honest. (The counterfactual may use the answer key as its correction source - it tests
a candidate, it does not choose one - and records that it did.)
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ffx.analyzer.grounding import FIELD_KINDS, FieldKind, ground, values_equal
from ffx.models import AnswerKey, FailureCategory, StepName
from ffx.tracing.span import SpanStatus, Trace

KEY_FIELDS = ["claimant_name", "policy_number", "claim_number", "date_of_loss", "reported_amount"]


class TruthLabel(BaseModel):
    is_failure: bool
    root_step: Optional[StepName] = None
    category: FailureCategory = FailureCategory.NO_FAILURE
    reasons: list[str] = Field(default_factory=list)


def label(trace: Trace, key: AnswerKey) -> TruthLabel:
    source = trace.metadata.get("source_text", "")
    failures: list[tuple[StepName, FailureCategory, str]] = []

    def fail(step: StepName, category: FailureCategory, reason: str) -> None:
        failures.append((step, category, reason))

    for span in trace.spans:
        if span.status == SpanStatus.ERROR and not span.spliced:
            infra = span.error is not None and span.error.type == "LLMTransportError"
            fail(
                span.step_name,
                FailureCategory.INFRASTRUCTURE_ERROR if infra else FailureCategory.PROMPT_FAILURE,
                f"{span.step_name.value} {'hit a model-server failure' if infra else 'produced no valid output'}",
            )

    intake = trace.span(StepName.INTAKE)
    lost_at_intake: set[str] = set()
    if intake is not None and intake.output_payload:
        doc_type = intake.output_payload.get("detected_doc_type")
        is_claim = key.doc_type.value == "claim_notice"
        if is_claim != (doc_type == "claim_notice"):
            fail(StepName.INTAKE, FailureCategory.MISCLASSIFICATION, f"doc_type {doc_type} but truth is {key.doc_type.value}")
        normalized = intake.output_payload.get("normalized_text", "")
        for field in KEY_FIELDS:
            truth = getattr(key, field)
            kind = FIELD_KINDS[field]
            if truth and ground(truth, source, kind).grounded and not ground(truth, normalized, kind).grounded:
                lost_at_intake.add(field)
        if lost_at_intake:
            fail(StepName.INTAKE, FailureCategory.CONTEXT_LOSS, f"intake lost {sorted(lost_at_intake)}")

    extraction = trace.span(StepName.EXTRACTION)
    peril_ok = True
    if extraction is not None and extraction.output_payload:
        fields = extraction.output_payload.get("fields", {})
        for field in KEY_FIELDS:
            if field in lost_at_intake:
                continue  # extraction could not have got it right
            truth, got = getattr(key, field), (fields.get(field) or "").strip()
            kind = FIELD_KINDS[field]
            if truth and not got:
                fail(StepName.EXTRACTION, FailureCategory.CONTEXT_LOSS, f"missed {field}")
            elif got and not values_equal(got, truth, kind):
                # Loose match for free text (names): OCR noise in the source is not the model's fault.
                if kind == FieldKind.TEXT and truth and ground(got, truth, kind).grounded:
                    continue
                fail(StepName.EXTRACTION, FailureCategory.EXTRACTION_HALLUCINATION, f"{field}={got!r}, truth {truth!r}")
        if key.doc_type.value == "claim_notice" and fields.get("peril") != key.peril.value:
            peril_ok = False
            fail(StepName.EXTRACTION, FailureCategory.MISCLASSIFICATION, f"peril={fields.get('peril')}, truth {key.peril.value}")

    classification = trace.span(StepName.CLASSIFICATION)
    if classification is not None and classification.output_payload and peril_ok and not classification.spliced:
        got = classification.output_payload.get("claim_category")
        if got != key.claim_category.value:
            fail(StepName.CLASSIFICATION, FailureCategory.MISCLASSIFICATION, f"category={got}, truth {key.claim_category.value}")

    if not failures:
        return TruthLabel(is_failure=False)
    from ffx.models import STEP_ORDER

    failures.sort(key=lambda f: STEP_ORDER.index(f[0]))
    root_step, category, _ = failures[0]
    return TruthLabel(
        is_failure=True,
        root_step=root_step,
        category=category,
        reasons=[f"{s.value}: {r}" for s, _, r in failures],
    )
