"""Eval cases: a human-confirmed failure, frozen as a step-level regression test.

A case is deliberately *step-level*, not end-to-end. It stores the exact input the
failing step received, so a fix is re-tested in seconds against the precise input that
broke it - one model call, not a whole document's pipeline.

Assertions are checked by code, never by an LLM. Re-running a case must give the same
verdict every time the step gives the same output; a model-graded assertion would add a
second source of variance to a test whose whole job is to detect change.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from ffx.analyzer.grounding import FieldKind, ground, values_equal
from ffx.analyzer.signals import STEP_SIGNALS
from ffx.models import FailureCategory, StepName, STEP_ORDER, utcnow
from ffx.tracing.span import Span


class AssertionKind(str, Enum):
    PRODUCES_OUTPUT = "produces_output"  # the step returned a schema-valid output at all
    EQUALS = "equals"  # a field must hold the expected value
    CONTAINS = "contains"  # a text field must contain a fact
    CHECK_PASSES = "check_passes"  # a deterministic signal that failed must now pass


class Assertion(BaseModel):
    kind: AssertionKind
    description: str
    path: str = ""  # dotted path into the output, e.g. "fields.policy_number"
    expected: str = ""
    # How to compare: a grounding kind ("2024-03-14" equals "March 14, 2024"), or None
    # for exact string equality - right for enums such as claim_category.
    value_kind: Optional[FieldKind] = None
    signal: str = ""  # for CHECK_PASSES: the signal name, e.g. "value_grounded:policy_number"


class AssertionResult(BaseModel):
    description: str
    passed: bool
    detail: str = ""


class EvalCase(BaseModel):
    case_id: str
    source_trace_id: str
    source_flag_id: str
    doc_id: str
    step_name: StepName
    category: FailureCategory
    verdict: str  # the human verdict it came from: confirm | override
    created_at: datetime = Field(default_factory=utcnow)

    # Provenance: which prompts and model produced the failure.
    pipeline_version: str = ""
    model: str = ""

    step_input: dict[str, Any]
    bad_output: Optional[dict[str, Any]] = None  # None when the step produced nothing
    expected_output: Optional[dict[str, Any]] = None
    expected_source: str = ""  # human | answer_key | judge | rule | "" (none available)
    assertions: list[Assertion]
    note: str = ""
    # Every confirmed trace whose step received this exact input. The same document
    # reviewed in two runs is one failure, so it is one case - not two that inflate counts.
    seen_in: list[str] = Field(default_factory=list)

    @property
    def traces(self) -> list[str]:
        return self.seen_in or [self.source_trace_id]  # cases saved before seen_in existed


# --------------------------------------------------------------------------
# Checking
# --------------------------------------------------------------------------


def get_path(output: dict[str, Any], path: str) -> Any:
    value: Any = output
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def same_value(actual: str, expected: str, kind: Optional[FieldKind]) -> bool:
    if kind is None:
        return actual == expected
    if values_equal(actual, expected, kind):
        return True
    # Names: "Ms Nadia Osei" for "Nadia Osei" is the right person, not a regression.
    return kind == FieldKind.TEXT and bool(actual and expected) and ground(expected, actual, kind).grounded


def check(assertion: Assertion, step: StepName, step_input: dict[str, Any], output: Optional[dict[str, Any]]) -> AssertionResult:
    """Evaluate one assertion against one output of ``step``."""
    if assertion.kind == AssertionKind.PRODUCES_OUTPUT:
        return AssertionResult(
            description=assertion.description,
            passed=output is not None,
            detail="" if output is not None else "the step produced no valid output",
        )
    if output is None:
        return AssertionResult(description=assertion.description, passed=False, detail="no output to check")

    if assertion.kind == AssertionKind.EQUALS:
        actual = get_path(output, assertion.path)
        actual = "" if actual is None else str(actual)
        ok = same_value(actual, assertion.expected, assertion.value_kind)
        return AssertionResult(
            description=assertion.description, passed=ok, detail="" if ok else f"got {actual!r}",
        )

    if assertion.kind == AssertionKind.CONTAINS:
        text = str(get_path(output, assertion.path) or "")
        ok = ground(assertion.expected, text, assertion.value_kind or FieldKind.TEXT).grounded
        return AssertionResult(
            description=assertion.description, passed=ok, detail="" if ok else f"{assertion.expected!r} is missing",
        )

    # CHECK_PASSES: re-derive the step's own deterministic checks on this output. A
    # signal that does not fire at all (e.g. no value, so nothing to ground) counts as
    # a pass - other assertions cover whether the value should have been there.
    span = Span(
        trace_id="eval",
        step_name=step,
        step_index=STEP_ORDER.index(step),
        input_payload=step_input,
        output_payload=output,
    )
    failing = [s for s in STEP_SIGNALS[step](span) if s.name == assertion.signal and not s.passed]
    return AssertionResult(
        description=assertion.description,
        passed=not failing,
        detail=failing[0].detail if failing else "",
    )


def check_all(case: EvalCase, output: Optional[dict[str, Any]]) -> list[AssertionResult]:
    return [check(a, case.step_name, case.step_input, output) for a in case.assertions]
