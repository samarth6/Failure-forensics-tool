"""Failure taxonomy: read the category straight off the evidence.

There is no separate "classify this failure" model call. Each signal declares which
category it is evidence for; the root-cause span's category is that of its heaviest
failed signal. So every category assignment can be traced to a specific check with a
specific detail string - and a human reviewing it can see exactly why.
"""

from __future__ import annotations

from typing import Optional

from ffx.models import FailureCategory, StepName
from ffx.tracing.span import Signal

CATEGORY_LABELS = {
    FailureCategory.EXTRACTION_HALLUCINATION: "Extraction Hallucination",
    FailureCategory.MISCLASSIFICATION: "Misclassification",
    FailureCategory.PROPAGATION_ERROR: "Propagation Error",
    FailureCategory.PROMPT_FAILURE: "Prompt Failure",
    FailureCategory.CONTEXT_LOSS: "Context Loss",
    FailureCategory.SUMMARY_HALLUCINATION: "Summary Hallucination",
    FailureCategory.INFRASTRUCTURE_ERROR: "Infrastructure Error",
    FailureCategory.NO_FAILURE: "No Failure",
}

CATEGORY_DESCRIPTIONS = {
    FailureCategory.EXTRACTION_HALLUCINATION: "A value was extracted that its input does not support: invented, or the wrong one of several candidates.",
    FailureCategory.MISCLASSIFICATION: "A judgement (document type, peril, category, severity, routing) contradicts the step's own input.",
    FailureCategory.PROPAGATION_ERROR: "The step behaved correctly given what it received; the damage came from upstream.",
    FailureCategory.PROMPT_FAILURE: "The step could not produce valid structured output, or needed repair to do so.",
    FailureCategory.CONTEXT_LOSS: "Information available to the step was dropped: context-window overflow, lost facts, or facts not carried forward.",
    FailureCategory.SUMMARY_HALLUCINATION: "The summary states facts that are not present in its input.",
    FailureCategory.INFRASTRUCTURE_ERROR: "The model server or provider failed (crash, out of memory, outage) - not the pipeline's fault.",
    FailureCategory.NO_FAILURE: "No failure detected.",
}

# Used only when a damaged span has no failed signal that names a category.
STEP_DEFAULT_CATEGORY = {
    StepName.INTAKE: FailureCategory.CONTEXT_LOSS,
    StepName.EXTRACTION: FailureCategory.EXTRACTION_HALLUCINATION,
    StepName.CLASSIFICATION: FailureCategory.MISCLASSIFICATION,
    StepName.SUMMARIZATION: FailureCategory.SUMMARY_HALLUCINATION,
}


def signal_severity_mass(signal: Signal) -> float:
    """How much a failed signal pulls quality down: trust times shortfall."""
    return signal.weight * (1.0 - signal.score)


def heaviest_failure(signals: list[Signal]) -> Optional[Signal]:
    failed = [s for s in signals if not s.passed and s.implicates is not None]
    if not failed:
        return None
    return max(failed, key=signal_severity_mass)


def categorize(step: StepName, signals: list[Signal]) -> tuple[FailureCategory, str]:
    """(category, subtype) for a damaged span. Subtype is the signal name that decided it."""
    top = heaviest_failure(signals)
    if top is None:
        return STEP_DEFAULT_CATEGORY[step], "unattributed_quality_drop"
    return top.implicates, top.name  # type: ignore[return-value]
