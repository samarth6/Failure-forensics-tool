"""The backward walk, verified on hand-built traces - no model, no network.

Each test builds a trace whose failure origin is known by construction, then checks the
engine names the right step, the right category, and labels downstream steps as
propagation rather than as additional root causes.
"""

from typing import Any, Optional

import pytest

from ffx.analyzer.diagnose import FAIL_THRESHOLD, Analyzer, local_quality
from ffx.analyzer.diagnosis_models import SpanRole
from ffx.models import FailureCategory, StepName
from ffx.tracing.span import Signal, SignalKind, SignalSeverity, Span, SpanError, SpanStatus, Trace
from ffx.tracing.store import TraceStore

SOURCE = (
    "Subject: New claim - Maria Okafor\n\nI am writing to report a loss at 14 Ash Lane, Bristol. "
    "My policy number is POL-44821. On 14 March 2024 a kitchen fire started by an unattended pan "
    "caused smoke and heat damage to the kitchen. I would like to claim $4,500.00. "
    "Your reference for this is CLM-7781.\n\nKind regards,\nMaria Okafor"
)

GOOD_FIELDS = {
    "claimant_name": "Maria Okafor",
    "policy_number": "POL-44821",
    "claim_number": "CLM-7781",
    "date_of_loss": "2024-03-14",
    "reported_amount": "4500.00",
    "loss_location": "14 Ash Lane, Bristol",
    "peril": "fire",
}


def _evidence(fields: dict[str, str]) -> list[dict[str, str]]:
    quotes = {
        "claimant_name": "Subject: New claim - Maria Okafor",
        "policy_number": "My policy number is POL-44821.",
        "claim_number": "Your reference for this is CLM-7781.",
        "date_of_loss": "On 14 March 2024 a kitchen fire started by an unattended pan",
        "reported_amount": "I would like to claim $4,500.00.",
        "loss_location": "I am writing to report a loss at 14 Ash Lane, Bristol.",
    }
    return [
        {"field": f, "value": v, "source_quote": quotes.get(f, v)}
        for f, v in fields.items()
        if f != "peril" and v
    ]


def _span(trace_id: str, step: StepName, idx: int, inp: dict[str, Any], out: Optional[dict[str, Any]], **kw) -> Span:
    return Span(
        trace_id=trace_id,
        step_name=step,
        step_index=idx,
        input_payload={**inp, "_meta": {"num_ctx": 4096, "prompt_chars": 2000, "done_reason": "stop"}},
        output_payload=out,
        tokens_in=500,
        tokens_out=200,
        self_confidence=(out or {}).get("confidence"),
        **kw,
    )


def build_trace(
    *,
    normalized: str = SOURCE,
    fields: Optional[dict[str, str]] = None,
    category: str = "property",
    severity: str = "low",
    routing: str = "fast_track",
    summary_body: Optional[str] = None,
    doc_type: str = "claim_notice",
    stop_after: Optional[StepName] = None,
) -> Trace:
    fields = {**GOOD_FIELDS, **(fields or {})}
    trace = Trace(doc_id="doc_test", metadata={"source_text": SOURCE, "channel": "email"})
    tid = trace.trace_id

    trace.spans.append(
        _span(tid, StepName.INTAKE, 0, {"doc_id": "doc_test", "source_text": SOURCE},
              {"normalized_text": normalized, "detected_doc_type": doc_type, "quality_flags": [], "confidence": 4})
    )
    if stop_after == StepName.INTAKE:
        trace.spans.append(
            _span(tid, StepName.EXTRACTION, 1, {"normalized_text": normalized}, None,
                  status=SpanStatus.ERROR, error=SpanError(type="SchemaValidationError", message="confidence: must be <= 5"))
        )
        return trace
    trace.spans.append(
        _span(tid, StepName.EXTRACTION, 1, {"normalized_text": normalized},
              {"fields": fields, "evidence": _evidence(fields), "missing_fields": [], "confidence": 5})
    )
    trace.spans.append(
        _span(tid, StepName.CLASSIFICATION, 2, {"doc_type": doc_type, "quality_flags": [], "fields": fields},
              {"claim_category": category, "severity": severity, "routing_queue": routing, "rationale": "", "confidence": 4})
    )
    body = summary_body or (
        f"{fields['claimant_name']} reports a {fields['peril']} on {fields['date_of_loss']} under policy "
        f"{fields['policy_number']}, claiming ${fields['reported_amount']}."
    )
    trace.spans.append(
        _span(tid, StepName.SUMMARIZATION, 3,
              {"fields": fields, "missing_fields": [], "classification": {"claim_category": category}},
              {"headline": "Claim summary", "body": body, "key_facts": [], "open_questions": [], "confidence": 4})
    )
    return trace


@pytest.fixture
def analyzer(tmp_path):
    store = TraceStore(db_path=tmp_path / "t.db", trace_dir=tmp_path / "traces")
    return Analyzer(store, runner=None, judge=None)


def role_of(diagnosis, step: StepName) -> SpanRole:
    return next(a.role for a in diagnosis.assessments if a.step_name == step)


# --------------------------------------------------------------------------


def test_clean_trace_is_not_a_failure(analyzer):
    d = analyzer.analyze(build_trace())
    assert not d.is_failure, d.explanation
    assert d.terminal_quality >= FAIL_THRESHOLD
    assert d.root_step is None


def test_fabricated_value_is_root_caused_to_extraction_and_propagates(analyzer):
    # Extraction invents a policy number; classification and summarization faithfully
    # use it. Only extraction is the origin - the rest must read as propagation.
    trace = build_trace(fields={"policy_number": "POL-12345"})
    d = analyzer.analyze(trace)

    assert d.is_failure
    assert d.root_step == StepName.EXTRACTION
    assert d.category == FailureCategory.EXTRACTION_HALLUCINATION
    assert d.subtype == "value_grounded:policy_number"
    assert role_of(d, StepName.CLASSIFICATION) == SpanRole.PROPAGATION
    assert role_of(d, StepName.SUMMARIZATION) == SpanRole.PROPAGATION

    propagation = [link for link in d.evidence_chain if link.role == "propagation"]
    assert any("POL-12345" in link.claim for link in propagation), "the bad value should be traced downstream"


def test_intake_context_loss_is_upstream_of_everything(analyzer):
    # Intake drops the sentences with the policy number and amount.
    lossy = SOURCE.replace("My policy number is POL-44821. ", "").replace("I would like to claim $4,500.00. ", "")
    trace = build_trace(normalized=lossy, fields={"policy_number": "", "reported_amount": ""})
    d = analyzer.analyze(trace)

    assert d.is_failure
    assert d.root_step == StepName.INTAKE
    assert d.category == FailureCategory.CONTEXT_LOSS


def test_misclassification_with_clean_extraction(analyzer):
    # Peril is fire, extraction is correct, but classification calls it an auto claim.
    d = analyzer.analyze(build_trace(category="auto"))
    assert d.root_step == StepName.CLASSIFICATION
    assert d.category == FailureCategory.MISCLASSIFICATION
    assert role_of(d, StepName.EXTRACTION) in (SpanRole.HEALTHY, SpanRole.LOW_CONFIDENCE)


def test_summary_that_invents_a_fact(analyzer):
    body = "Maria Okafor reports a fire on 2024-03-14 under policy POL-44821, claiming $45,000.00."
    d = analyzer.analyze(build_trace(summary_body=body))
    assert d.root_step == StepName.SUMMARIZATION
    assert d.category == FailureCategory.SUMMARY_HALLUCINATION


def test_step_that_cannot_produce_output_is_a_prompt_failure(analyzer):
    d = analyzer.analyze(build_trace(stop_after=StepName.INTAKE))
    assert d.is_failure
    assert d.root_step == StepName.EXTRACTION
    assert d.category == FailureCategory.PROMPT_FAILURE
    assert role_of(d, StepName.SUMMARIZATION) == SpanRole.NOT_RUN


def test_model_server_crash_is_infrastructure_not_prompt_failure(analyzer):
    # Observed: Ollama ran out of memory mid-run. That is the machine failing, not the
    # prompt - it must never be diagnosed as a Prompt Failure.
    trace = build_trace(stop_after=StepName.INTAKE)
    trace.spans[1].error = SpanError(type="LLMTransportError", message="ollama returned 500: failed to allocate buffer")
    d = analyzer.analyze(trace)
    assert d.root_step == StepName.EXTRACTION
    assert d.category == FailureCategory.INFRASTRUCTURE_ERROR


def test_off_domain_document_treated_as_claim(analyzer):
    receipt = "THE OLIVE BRANCH TRATTORIA\nTable 7\n2 x Margherita 22.00\nTOTAL 64.68\nThank you for dining with us!"
    trace = build_trace()
    trace.metadata["source_text"] = receipt
    trace.spans[0].input_payload["source_text"] = receipt
    trace.spans[0].output_payload["normalized_text"] = receipt
    d = analyzer.analyze(trace)
    assert d.root_step == StepName.INTAKE
    assert d.category == FailureCategory.MISCLASSIFICATION


def test_real_chain_name_lost_at_intake_then_invented_then_misclassified(analyzer):
    """Regression test for the first real failure the tool found (corpus doc_005).

    Intake stripped the subject line and sign-off - the only places the claimant's name
    appeared. Extraction then invented "John Doe". Classification independently called a
    home burglary an auto claim. The origin is intake, not extraction (whose drop was
    larger), and the classification fault is a separate candidate, not propagation.
    """
    source = (
        "Subject: New claim - Nadia Osei\n\nDear Claims Team,\n\nI am writing to report a loss at "
        "5 Beacon Hill, Exeter. My policy number is POL-32026. On February 29, 2024 burglars forced "
        "the back door and stole jewellery, a laptop and two bicycles. I would like to claim "
        "USD 1,100.00 to cover the damage. Your reference for this is CLM-81074.\n\nKind regards,\nNadia Osei\n"
    )
    stripped = source.split("Dear Claims Team,\n\n", 1)[1].split("\n\nKind regards", 1)[0]
    trace = build_trace(
        normalized=stripped,
        fields={
            "claimant_name": "John Doe", "policy_number": "POL-32026", "claim_number": "CLM-81074",
            "date_of_loss": "2024-02-29", "reported_amount": "1100.00", "loss_location": "5 Beacon Hill, Exeter",
            "peril": "theft",
        },
        category="auto",
    )
    trace.metadata["source_text"] = source
    trace.spans[0].input_payload["source_text"] = source

    d = analyzer.analyze(trace)
    assert d.root_step == StepName.INTAKE, d.explanation
    assert d.category == FailureCategory.CONTEXT_LOSS
    assert role_of(d, StepName.EXTRACTION) == SpanRole.CANDIDATE  # an amplification, still its own fault
    assert role_of(d, StepName.CLASSIFICATION) == SpanRole.CANDIDATE  # independent, not propagation
    assert [c.step_name for c in d.candidates] == [StepName.INTAKE, StepName.EXTRACTION, StepName.CLASSIFICATION]


# --------------------------------------------------------------------------


def _sig(score: float, *, passed: bool, severity=SignalSeverity.INFO, weight: Optional[float] = None) -> Signal:
    return Signal(
        name="s", kind=SignalKind.DETERMINISTIC, severity=severity, passed=passed, score=score,
        weight={SignalSeverity.CRITICAL: 1.0, SignalSeverity.WARN: 0.5, SignalSeverity.INFO: 0.25}[severity]
        if weight is None else weight,
    )


def test_one_critical_failure_is_not_diluted_by_many_passes():
    signals = [_sig(1.0, passed=True) for _ in range(20)]
    signals.append(_sig(0.0, passed=False, severity=SignalSeverity.CRITICAL))
    assert local_quality(signals) < FAIL_THRESHOLD


def test_self_report_barely_moves_quality():
    signals = [_sig(1.0, passed=True), _sig(0.0, passed=False, weight=0.15)]
    assert local_quality(signals) > 0.6


def test_signals_are_all_local():
    # Wrong-but-consistent downstream step: classification routes a (fabricated) policy
    # correctly given its input. Its own local quality must stay high.
    trace = build_trace(fields={"policy_number": "POL-12345"})
    from ffx.analyzer.signals import deterministic_signals
    from ffx.config import get_settings

    cls = trace.spans[2]
    cls.signals = deterministic_signals(cls, get_settings())
    assert local_quality(cls.signals) > 0.85
