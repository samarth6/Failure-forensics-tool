"""Confirmed flag -> eval case.

The human review in the explorer is the gate: only a trace a reviewer has confirmed as a
failure (or overridden to a different root step) becomes a test. The case targets the
step the human named, with the input that step actually received.

Every generated assertion is validated before the case is saved:

* it must **pass** on the expected output - otherwise it tests something the correct
  answer does not satisfy, and the case could never be fixed;
* at least one must **fail** on the bad output - otherwise the case does not capture the
  failure it was made from, and would pass from day one.

A failure that cannot be expressed that way is reported, not saved.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from ffx.analyzer.counterfactual import build_correction
from ffx.analyzer.grounding import FIELD_KINDS, ground
from ffx.analyzer.signals import STEP_SIGNALS, deterministic_signals, salient_items
from ffx.analyzer.truth import KEY_FIELDS
from ffx.config import get_settings
from ffx.evals.cases import Assertion, AssertionKind, EvalCase, check, same_value
from ffx.models import STEP_ORDER, AnswerKey, FailureCategory, StepName
from ffx.tracing.span import SignalSeverity, Span, Trace
from ffx.tracing.store import TraceStore

CASE_VERDICTS = {"confirm", "override"}
MAX_CONTAINS = 8  # intake can lose many facts; a handful of assertions says it clearly
CLASSIFICATION_KEYS = ["claim_category", "severity", "routing_queue"]


def case_id_for(step: StepName, step_input: dict[str, Any]) -> str:
    """Identity by content: the step plus the exact input it received.

    Keyed on the trace instead, the same document confirmed in two corpus runs became
    two identical cases (observed: doc_005). ``received_at`` is excluded because it is
    bookkeeping that never reaches the prompt.
    """
    material = json.dumps({k: v for k, v in step_input.items() if k != "received_at"}, sort_keys=True, ensure_ascii=False)
    return "case_" + hashlib.sha1(f"{step.value}:{material}".encode("utf-8")).hexdigest()[:10]


# --------------------------------------------------------------------------
# Assertions
# --------------------------------------------------------------------------


def _diff_assertions(step: StepName, step_input: dict[str, Any], bad: Optional[dict[str, Any]], expected: dict[str, Any]) -> list[Assertion]:
    """What the expected output gets right that the bad output got wrong."""
    out: list[Assertion] = []
    bad = bad or {}

    if step == StepName.INTAKE:
        if bad.get("detected_doc_type") != expected.get("detected_doc_type"):
            out.append(
                Assertion(
                    kind=AssertionKind.EQUALS, path="detected_doc_type", expected=expected["detected_doc_type"],
                    description=f"detected_doc_type = {expected['detected_doc_type']}",
                )
            )
        good_text, bad_text = expected.get("normalized_text", ""), bad.get("normalized_text", "")
        for value, kind in salient_items(step_input.get("source_text", "")):
            if ground(value, good_text, kind).grounded and not ground(value, bad_text, kind).grounded:
                out.append(
                    Assertion(
                        kind=AssertionKind.CONTAINS, path="normalized_text", expected=value, value_kind=kind,
                        description=f"normalized_text keeps {value!r}",
                    )
                )
        return out[: MAX_CONTAINS + 1]

    if step == StepName.EXTRACTION:
        good_fields, bad_fields = expected.get("fields", {}), bad.get("fields", {})
        for field in KEY_FIELDS:
            want, got = good_fields.get(field, "") or "", bad_fields.get(field, "") or ""
            kind = FIELD_KINDS[field]
            if not bad or not same_value(got, want, kind):
                out.append(
                    Assertion(
                        kind=AssertionKind.EQUALS, path=f"fields.{field}", expected=want, value_kind=kind,
                        description=f"{field} = {want!r}" if want else f"{field} is left empty (the document does not state it)",
                    )
                )
        if good_fields.get("peril") != bad_fields.get("peril"):
            out.append(
                Assertion(
                    kind=AssertionKind.EQUALS, path="fields.peril", expected=str(good_fields.get("peril")),
                    description=f"peril = {good_fields.get('peril')}",
                )
            )
        return out

    if step == StepName.CLASSIFICATION:
        for key in CLASSIFICATION_KEYS:
            if bad.get(key) != expected.get(key):
                out.append(
                    Assertion(kind=AssertionKind.EQUALS, path=key, expected=str(expected.get(key)), description=f"{key} = {expected.get(key)}")
                )
    return out


def _signal_assertions(step: StepName, step_input: dict[str, Any], bad: Optional[dict[str, Any]]) -> list[Assertion]:
    """The step's own deterministic checks that failed on the bad output must pass."""
    if bad is None:
        return []
    span = Span(trace_id="eval", step_name=step, step_index=STEP_ORDER.index(step), input_payload=step_input, output_payload=bad)
    out: list[Assertion] = []
    seen: set[str] = set()
    for sig in STEP_SIGNALS[step](span):
        if sig.passed or sig.severity == SignalSeverity.INFO or sig.name in seen:
            continue
        seen.add(sig.name)
        out.append(
            Assertion(kind=AssertionKind.CHECK_PASSES, signal=sig.name, description=f"check {sig.name} passes (was: {sig.detail[:90]})")
        )
    return out


# --------------------------------------------------------------------------
# One flag -> one case
# --------------------------------------------------------------------------


def case_from_flag(
    trace: Trace,
    flag: dict[str, Any],
    *,
    answer_key: Optional[AnswerKey] = None,
) -> tuple[Optional[EvalCase], str]:
    """Build the case for a flag, or return (None, why not)."""
    if flag.get("verdict") not in CASE_VERDICTS:
        return None, "marked not a failure: nothing to regression-test"
    if not flag.get("step_name") or not flag.get("human_category"):
        return None, "the flag names no step or category"
    step = StepName(flag["step_name"])
    category = FailureCategory(flag["human_category"])
    if category == FailureCategory.INFRASTRUCTURE_ERROR:
        return None, "an infrastructure failure is the machine, not the model: re-running the step tests nothing"

    span = trace.span(step)
    if span is None:
        return None, f"{step.value} did not run in this trace"

    if not span.signals:
        # Never diagnosed: derive the deterministic signals, so rule-based corrections
        # are available exactly as they would be after `ffx diagnose`.
        span = span.model_copy(deep=True)
        span.signals = deterministic_signals(span, get_settings())
    step_input = {k: v for k, v in span.input_payload.items() if k != "_meta"}
    bad = span.output_payload
    built = build_correction(span, trace, answer_key=answer_key, human_output=flag.get("corrected_output"))
    expected, source = (built[0].model_dump(mode="json"), built[1]) if built else (None, "")

    candidates = [
        Assertion(kind=AssertionKind.PRODUCES_OUTPUT, description=f"{step.value} returns a schema-valid output"),
        *(_diff_assertions(step, step_input, bad, expected) if expected else []),
        *_signal_assertions(step, step_input, bad),
    ]
    # Keep only what the correct answer satisfies...
    kept = [a for a in candidates if expected is None or check(a, step, step_input, expected).passed]
    # ...and require that something catches the failure.
    if not any(not check(a, step, step_input, bad).passed for a in kept):
        return None, (
            f"no assertion separates the bad {step.value} output from a correct one"
            + ("" if expected else " (and no corrected output is available - add one in the review panel)")
        )

    case = EvalCase(
        case_id=case_id_for(step, step_input),
        source_trace_id=trace.trace_id,
        source_flag_id=flag["flag_id"],
        doc_id=trace.doc_id,
        step_name=step,
        category=category,
        verdict=flag["verdict"],
        pipeline_version=trace.pipeline_version,
        model=span.model,
        step_input=step_input,
        bad_output=bad,
        expected_output=expected,
        expected_source=source,
        assertions=kept,
        note=flag.get("note", ""),
    )
    return case, f"{len(kept)} assertions, expected output from {source or 'none (checks only)'}"


# --------------------------------------------------------------------------
# Keeping the case set in step with the flags
# --------------------------------------------------------------------------


def sync_trace(store: TraceStore, trace_id: str, *, answer_key: Optional[AnswerKey] = None) -> dict[str, Any]:
    """Make the trace's eval case reflect its *latest* human verdict.

    A reviewer who later changes "confirm" to "not a failure" retires the case; one who
    overrides to another step moves it.
    """
    flags = store.list_flags(trace_id)  # newest first
    trace = store.load_trace(trace_id)
    if not flags or trace is None:
        return {"trace_id": trace_id, "case_id": None, "action": "skipped", "reason": "no flag or no trace"}

    case, reason = case_from_flag(trace, flags[0], answer_key=answer_key)
    detached, deleted = _detach(store, trace_id)
    if case is None:
        return {"trace_id": trace_id, "doc_id": trace.doc_id, "case_id": None, "action": "retired" if deleted else "skipped", "reason": reason}

    existing = store.load_eval_case_json(case.case_id)
    others = [t for t in EvalCase.model_validate_json(existing).traces if t != trace_id] if existing else []
    case.seen_in = [*others, trace_id]
    store.save_eval_case(case)
    if case.case_id in detached:
        action = "updated"
    elif others:
        action = "merged"  # the same failure, already a case from another trace
        reason = f"same input as a case already confirmed in {len(others)} other trace(s); {reason}"
    else:
        action = "created"
    return {"trace_id": trace_id, "doc_id": trace.doc_id, "case_id": case.case_id, "step": case.step_name.value, "action": action, "reason": reason}


def _detach(store: TraceStore, trace_id: str) -> tuple[list[str], list[str]]:
    """Withdraw ``trace_id``'s support from every case it backs.

    A case still confirmed by another trace survives; one left with no trace is deleted.
    Returns (case ids touched, case ids deleted).
    """
    touched, deleted = [], []
    for case in load_cases(store):
        if trace_id not in case.traces:
            continue
        touched.append(case.case_id)
        remaining = [t for t in case.traces if t != trace_id]
        if remaining:
            case.seen_in = remaining
            if case.source_trace_id == trace_id:
                case.source_trace_id = remaining[-1]
            store.save_eval_case(case)
        else:
            store.delete_eval_case(case.case_id)
            deleted.append(case.case_id)
    return touched, deleted


def sync_cases(store: TraceStore, answer_keys: Optional[dict[str, AnswerKey]] = None) -> list[dict[str, Any]]:
    """Regenerate cases for every flagged trace."""
    answer_keys = answer_keys or {}
    trace_ids = list(dict.fromkeys(f["trace_id"] for f in store.list_flags()))
    out = []
    for trace_id in trace_ids:
        trace = store.load_trace(trace_id)
        key = answer_keys.get(trace.doc_id) if trace else None
        out.append(sync_trace(store, trace_id, answer_key=key))
    return out


def load_cases(store: TraceStore) -> list[EvalCase]:
    return [EvalCase.model_validate_json(body) for body in store.list_eval_case_json()]
