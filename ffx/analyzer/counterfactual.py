"""Counterfactual verification: intervene at the suspected step and watch the final output.

The question a counterfactual answers is causal: *if step X had produced the correct
output, would the final output have been right?*

How it is measured, and why
---------------------------
We fix one **reference** - the corrected facts - and score the *final* outputs of both
the original trace and the re-run against that same reference ("terminal fidelity").

The obvious alternative, "does span quality recover after the splice", is circular. The
signals are local by design, so propagation steps already score healthy; splicing a
corrected output at X merely deletes X's own penalty, and quality "recovers" every time.
That would confirm any diagnosis, right or wrong.

Terminal fidelity instead only rises if the correction actually flows through to what
the pipeline finally says. If the downstream steps ignore or overwrite X's output, it
does not move - and the diagnosis is refuted.

What it cannot do
-----------------
Without an answer key, the correction comes from the judge or a rule. The counterfactual
then proves that X *causally drives* the final output toward that correction - not that
the correction is itself right. The human review in the trace explorer is the final
authority on that, and flags feed back in here as the highest-priority correction source.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel

from ffx.analyzer.diagnosis_models import CounterfactualResult, CounterfactualVerdict
from ffx.analyzer.grounding import (
    FIELD_KINDS,
    FieldKind,
    find_amounts,
    find_dates,
    find_identifiers,
    ground,
    identifier_prefix,
    ocr_canonical,
    values_equal,
)
from ffx.analyzer.signals import PERIL_TO_CATEGORIES, salient_items
from ffx.models import (
    AnswerKey,
    ClassificationResult,
    ExtractionResult,
    FieldEvidence,
    IntakeResult,
    StepName,
)
from ffx.tracing.span import SignalSeverity, Span, Trace





REFERENCE_FIELDS = ["claimant_name", "policy_number", "claim_number", "date_of_loss", "reported_amount"]
MUST_STATE = {"claimant_name", "policy_number", "date_of_loss", "reported_amount"}


# --------------------------------------------------------------------------
# Building a corrected output
# --------------------------------------------------------------------------


def strip_boilerplate(text: str) -> str:
    """What intake should have done: drop long paragraphs that carry no facts."""
    kept = []
    for para in re.split(r"\n\s*\n", text):
        if len(para) > 250 and not salient_items(para):
            continue
        kept.append(para)
    return "\n\n".join(kept)


def repair_identifiers(text: str, key: AnswerKey) -> str:
    """Replace OCR-garbled identifiers with the answer key's canonical spelling."""
    truth = [v for v in (key.policy_number, key.claim_number) if v]
    for token in find_identifiers(text):
        for value in truth:
            if token != value and ocr_canonical(token) == ocr_canonical(value):
                text = text.replace(token, value)
    return text


def _sentence_containing(value: str, text: str, kind: FieldKind) -> str:
    for sentence in re.split(r"(?<=[.!?\n])\s+", text):
        if ground(value, sentence, kind).grounded:
            return sentence.strip()
    return value


def _apply_corrections(original: dict[str, Any], corrections: dict[str, Any]) -> dict[str, Any]:
    patched = {**original, "fields": {**original.get("fields", {})}} if "fields" in original else dict(original)
    target = patched["fields"] if "fields" in patched else patched
    for key, value in corrections.items():
        target[key] = value
    return patched


def build_correction(
    span: Span,
    trace: Trace,
    *,
    answer_key: Optional[AnswerKey] = None,
    human_output: Optional[dict[str, Any]] = None,
) -> Optional[tuple[BaseModel, str]]:
    """Return (corrected output, source) for ``span``, or None if we cannot tell."""
    step = span.step_name
    if step == StepName.SUMMARIZATION:
        return None

    model_cls = {
        StepName.INTAKE: IntakeResult,
        StepName.EXTRACTION: ExtractionResult,
        StepName.CLASSIFICATION: ClassificationResult,
    }[step]

    # 1. A human correction beats everything.
    if human_output:
        base = span.output_payload or {}
        merged = _apply_corrections(base, human_output.get("fields", human_output)) if base else human_output
        try:
            return model_cls.model_validate(merged), "human"
        except Exception:  # noqa: BLE001 - a malformed flag should not abort diagnosis
            pass

    # 2. The corpus answer key.
    if answer_key is not None:
        if step == StepName.INTAKE:
            source = span.input_payload.get("source_text", "")
            return (
                IntakeResult(
                    normalized_text=repair_identifiers(strip_boilerplate(source), answer_key),
                    detected_doc_type=answer_key.doc_type,
                    quality_flags=[],
                    confidence=5,
                ),
                "answer_key",
            )
        if step == StepName.EXTRACTION:
            text = span.input_payload.get("normalized_text", "")
            values = {f: getattr(answer_key, f) for f in REFERENCE_FIELDS + ["loss_location"]}
            evidence = [
                FieldEvidence(field=f, value=v, source_quote=_sentence_containing(v, text, FIELD_KINDS.get(f, FieldKind.TEXT)))
                for f, v in values.items()
                if v
            ]
            return (
                ExtractionResult(
                    fields={**values, "peril": answer_key.peril},
                    evidence=evidence,
                    missing_fields=[f for f, v in values.items() if not v],
                    confidence=5,
                ),
                "answer_key",
            )
        if step == StepName.CLASSIFICATION:
            return (
                ClassificationResult(
                    claim_category=answer_key.claim_category,
                    severity=answer_key.severity,
                    routing_queue=answer_key.routing_queue,
                    rationale="corrected from answer key",
                    confidence=5,
                ),
                "answer_key",
            )

    # 3. Corrections proposed by failed signals (judge first, then rules).
    if span.output_payload is None:
        return None
    proposals: dict[str, Any] = {}
    source = "rule"
    for sig in sorted(span.signals, key=lambda s: s.kind.value != "llm_judge"):
        if sig.passed or not sig.proposed_correction:
            continue
        for key, value in sig.proposed_correction.items():
            if key not in proposals:
                proposals[key] = value
                if sig.kind.value == "llm_judge":
                    source = "judge"
    if not proposals:
        if step == StepName.INTAKE:
            source_text = span.input_payload.get("source_text", "")
            patched = {**span.output_payload, "normalized_text": strip_boilerplate(source_text)}
            return IntakeResult.model_validate(patched), "rule"
        return None
    try:
        return model_cls.model_validate(_apply_corrections(span.output_payload, proposals)), source
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# Reference and terminal fidelity
# --------------------------------------------------------------------------


def build_reference(
    step: StepName,
    corrected: BaseModel,
    original: Trace,
    answer_key: Optional[AnswerKey],
) -> dict[str, Any]:
    """The facts the final output *should* state if the correction is right."""
    if answer_key is not None:
        facts = {f: getattr(answer_key, f) for f in REFERENCE_FIELDS}
        return {"facts": facts, "category": answer_key.claim_category.value, "source": "answer_key"}

    extraction = original.span(StepName.EXTRACTION)
    fields = dict((extraction.output_payload or {}).get("fields", {})) if extraction else {}
    category = None
    if isinstance(corrected, ExtractionResult):
        fields = corrected.fields.model_dump(mode="json")
    if isinstance(corrected, ClassificationResult):
        category = corrected.claim_category.value
    else:
        options = PERIL_TO_CATEGORIES.get(fields.get("peril", ""), set())
        category = next(iter(options)) if len(options) == 1 else None
    if isinstance(corrected, IntakeResult):
        # We do not know the right field values, only which facts intake lost.
        intake = original.span(StepName.INTAKE)
        lost = []
        for sig in intake.signals if intake else []:
            lost.extend(sig.evidence.get("lost", []))
        return {"facts": {}, "lost_facts": lost, "category": category, "source": "rule"}
    facts = {f: fields.get(f, "") for f in REFERENCE_FIELDS}
    return {"facts": facts, "category": category, "source": "correction"}


def _summary_text(trace: Trace) -> Optional[str]:
    span = trace.span(StepName.SUMMARIZATION)
    if span is None or not span.output_payload:
        return None
    out = span.output_payload
    return "\n".join([out.get("headline", ""), out.get("body", ""), *out.get("key_facts", [])])


def _conflicts(field: str, value: str, summary: str) -> list[str]:
    kind = FIELD_KINDS[field]
    if kind == FieldKind.IDENTIFIER:
        prefix = identifier_prefix(value) if value else {"policy_number": "P01", "claim_number": "C1M"}[field]
        return [t for t in find_identifiers(summary) if identifier_prefix(t) == prefix and not (value and values_equal(t, value, kind))]
    if kind == FieldKind.DATE:
        return [span for d, span in find_dates(summary) if not (value and values_equal(d.isoformat(), value, kind))]
    if kind == FieldKind.AMOUNT:
        return [raw for a, raw in find_amounts(summary, money_only=True) if not (value and values_equal(str(a), value, kind))]
    return []


def terminal_fidelity(trace: Trace, reference: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    """Score the final outputs against the reference. Each check is tagged with the
    field it tests, so the verdict can look only at the checks a correction touches."""
    summary = _summary_text(trace)
    checks: list[dict[str, Any]] = []
    if summary is None:
        return 0.0, [{"check": "pipeline produced a final summary", "field": "*", "passed": False}]

    for field, value in reference.get("facts", {}).items():
        kind = FIELD_KINDS[field]
        if value and field in MUST_STATE:
            checks.append({"check": f"states {field} = {value}", "field": field, "passed": ground(value, summary, kind).grounded})
        if kind in (FieldKind.IDENTIFIER, FieldKind.DATE, FieldKind.AMOUNT):
            conflicts = _conflicts(field, value, summary)
            label = f"no conflicting {field}" if value else f"does not invent a {field}"
            checks.append({"check": label, "field": field, "passed": not conflicts, "found": conflicts})

    for value in reference.get("lost_facts", []):
        checks.append({"check": f"recovers lost fact {value}", "field": f"lost:{value}", "passed": ground(value, summary).grounded})

    category = reference.get("category")
    cls = trace.span(StepName.CLASSIFICATION)
    if category and cls is not None and cls.output_payload:
        actual = cls.output_payload.get("claim_category")
        checks.append({"check": f"claim_category = {category}", "field": "claim_category", "passed": actual == category, "found": actual})

    if not checks:
        return 1.0, checks
    return sum(1 for c in checks if c["passed"]) / len(checks), checks


def touched_fields(step: StepName, original_span: Span, corrected: BaseModel, reference: dict[str, Any]) -> set[str]:
    """Which final-output facts does this correction predict will change?"""
    touched: set[str] = set()
    original = original_span.output_payload or {}

    if isinstance(corrected, ExtractionResult):
        before = original.get("fields", {})
        after = corrected.fields.model_dump(mode="json")
        for field in REFERENCE_FIELDS:
            if not values_equal(before.get(field, ""), after.get(field, ""), FIELD_KINDS[field]):
                touched.add(field)
        if before.get("peril") != after.get("peril"):
            touched.add("claim_category")

    elif isinstance(corrected, IntakeResult):
        old_text = original.get("normalized_text", "")
        new_text = corrected.normalized_text
        for field, value in reference.get("facts", {}).items():
            kind = FIELD_KINDS[field]
            if value and not ground(value, old_text, kind).grounded and ground(value, new_text, kind).grounded:
                touched.add(field)
        touched.update(f"lost:{v}" for v in reference.get("lost_facts", []))
        if original.get("detected_doc_type") != corrected.detected_doc_type.value:
            touched.add("claim_category")

    elif isinstance(corrected, ClassificationResult):
        if original.get("claim_category") != corrected.claim_category.value:
            touched.add("claim_category")

    return touched


def judge_counterfactual(
    *,
    step: StepName,
    original: Trace,
    original_span: Span,
    rerun: Optional[Trace],
    reference: dict[str, Any],
    correction_source: str,
    corrected: BaseModel,
) -> CounterfactualResult:
    """The causal test: did correcting ``step`` fix the final-output facts it touches?

    Only the *implicated* checks decide the verdict. An earlier version compared global
    fidelity before and after against a fixed delta; on a real trace with nine checks,
    fixing exactly the predicted fact moved fidelity by 0.11 and was wrongly "refuted" -
    while the checks still failing belonged to a second, independent error elsewhere.
    """
    before, checks_before = terminal_fidelity(original, reference)
    touched = touched_fields(step, original_span, corrected, reference)
    result = CounterfactualResult(
        verdict=CounterfactualVerdict.NOT_APPLICABLE,
        splice_step=step,
        correction_source=correction_source,
        corrected_output=corrected.model_dump(mode="json"),
        fidelity_before=round(before, 3),
        reference=reference,
        checks_before=checks_before,
        implicated_checks=[c["check"] for c in checks_before if c["field"] in touched],
    )
    if rerun is None:
        return result

    after, checks_after = terminal_fidelity(rerun, reference)
    after_by_check = {c["check"]: c for c in checks_after}
    result.counterfactual_trace_id = rerun.trace_id
    result.fidelity_after = round(after, 3)
    result.checks_after = checks_after
    result.new_downstream_failures = [
        f"{s.step_name.value}:{sig.name}"
        for s in rerun.spans
        if not s.spliced and s.copied_from_span_id is None
        for sig in s.signals
        if not sig.passed and sig.severity == SignalSeverity.CRITICAL
    ]

    implicated_failing = [c for c in checks_before if c["field"] in touched and not c["passed"]]
    fixed = [c["check"] for c in implicated_failing if after_by_check.get(c["check"], {}).get("passed")]
    result.fixed_checks = fixed
    result.remaining_failures = [c["check"] for c in checks_after if not c["passed"] and c["field"] not in touched]
    remaining = f" Still failing, independently of this step: {'; '.join(result.remaining_failures)}." if result.remaining_failures else ""

    if not touched:
        result.verdict = CounterfactualVerdict.NOT_APPLICABLE
        result.note = f"The correction to {step.value} changes nothing the final output states."
    elif not implicated_failing:
        result.verdict = CounterfactualVerdict.CONTAINED
        result.note = (
            f"The error at {step.value} is real but did not reach the final output: every final fact it "
            f"touches ({', '.join(sorted(touched))}) was already correct."
        )
    elif len(fixed) == len(implicated_failing):
        downstream = [f for f in result.new_downstream_failures if not f.startswith("summarization:summary_retention")]
        result.verdict = CounterfactualVerdict.PARTIAL if downstream else CounterfactualVerdict.CONFIRMED
        result.note = (
            f"Correcting {step.value} fixed {len(fixed)}/{len(implicated_failing)} affected final-output "
            f"fact(s): {'; '.join(fixed)}." + remaining
            + (f" But the re-run introduced new failures: {', '.join(downstream)}." if downstream else "")
        )
    elif fixed:
        result.verdict = CounterfactualVerdict.PARTIAL
        result.note = f"Correcting {step.value} fixed {len(fixed)} of {len(implicated_failing)} affected final-output facts." + remaining
    else:
        result.verdict = CounterfactualVerdict.REFUTED
        result.note = (
            f"Correcting {step.value} fixed none of the {len(implicated_failing)} final-output fact(s) it "
            f"touches; the final output does not depend on this step's error."
        )
    return result
