"""Root-cause diagnosis: from signals to "it broke *here*, and here is the proof".

The maths
---------
For each span ``i``, ``local_quality(i)`` summarises its signals - which, by design, all
judge the span against *its own input*. Two separate questions are asked of it:

**Did this step fail on its own?** ``own_damage(i) = 1 - local_quality(i)``. This is
independent of upstream quality, and a span with ``own_damage >= OWN_DAMAGE_MIN`` is a
root-cause *candidate*.

**How did quality flow?** Quality compounds down the chain, which drives the overall
failure verdict and the quality curve shown in the explorer::

    q_in(0)  = 1.0                       # the raw document is ground truth
    q_in(i)  = q_out(i - 1)
    q_out(i) = q_in(i) * local_quality(i)

A step below the failure line whose own checks passed received garbage and faithfully
passed it on: a *Propagation Error*, not an origin.

The root cause is the **first significant quality drop** - the earliest candidate - as
the brief specifies. An early version ranked by largest drop instead; a real trace showed
why that is wrong: intake silently dropped the claimant's name, and extraction then
invented "John Doe". Extraction's drop was larger, but it only happened because of
intake's. The earliest causal failure is the origin; later ones are amplifications or
independent faults, and are reported as further candidates.

Candidates are then *tested*, not asserted: a counterfactual re-run (see
:mod:`ffx.analyzer.counterfactual`) checks whether correcting the step actually fixes
the final-output facts it touches. A refuted or contained candidate yields to the next.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional

from ffx.analyzer.counterfactual import build_correction, build_reference, judge_counterfactual
from ffx.analyzer.diagnosis_models import (
    CounterfactualResult,
    CounterfactualVerdict,
    Diagnosis,
    EvidenceLink,
    RootCauseCandidate,
    SpanAssessment,
    SpanRole,
)
from ffx.analyzer.grounding import FIELD_KINDS, FieldKind, ground
from ffx.analyzer.judge import Judge
from ffx.analyzer.signals import deterministic_signals
from ffx.analyzer.taxonomy import CATEGORY_LABELS, categorize, signal_severity_mass
from ffx.config import Settings, get_settings
from ffx.models import STEP_ORDER, AnswerKey, FailureCategory, RawDocument, StepName
from ffx.tracing.span import Signal, SignalSeverity, Span, Trace, TraceStatus
from ffx.tracing.store import TraceStore

FAIL_THRESHOLD = 0.6
OWN_DAMAGE_MIN = 0.3  # local_quality below 0.7: the step failed its own checks
CRITICAL_CAP_MARGIN = 0.2
MAX_COUNTERFACTUALS = 3


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def local_quality(signals: list[Signal]) -> float:
    """Weighted mean of signal scores, capped by any failed critical signal.

    The cap matters: fifteen passing checks must not dilute one proven hallucination
    back above the failure threshold.
    """
    weighted = [s for s in signals if s.weight > 0]
    if not weighted:
        return 1.0
    base = sum(s.weight * s.score for s in weighted) / sum(s.weight for s in weighted)
    critical = [s.score for s in weighted if not s.passed and s.severity == SignalSeverity.CRITICAL]
    if critical:
        base = min(base, min(critical) + CRITICAL_CAP_MARGIN)
    return round(base, 4)


def assess(trace: Trace) -> tuple[list[SpanAssessment], float]:
    assessments: list[SpanAssessment] = []
    q_in = 1.0
    for span in trace.spans:
        lq = local_quality(span.signals)
        q_out = round(q_in * lq, 4)
        assessments.append(
            SpanAssessment(
                span_id=span.span_id,
                step_name=span.step_name,
                q_in=round(q_in, 4),
                local_quality=lq,
                q_out=q_out,
                local_damage=round(q_in - q_out, 4),
                role=SpanRole.HEALTHY,
                failed_signals=[s.name for s in span.signals if not s.passed and s.weight > 0],
            )
        )
        q_in = q_out

    ran = {a.step_name for a in assessments}
    for step in STEP_ORDER:
        if step not in ran:
            assessments.append(
                SpanAssessment(
                    span_id="", step_name=step, q_in=0.0, local_quality=0.0, q_out=0.0,
                    local_damage=0.0, role=SpanRole.NOT_RUN,
                )
            )

    complete = len(trace.spans) == len(STEP_ORDER)
    terminal = assessments[len(trace.spans) - 1].q_out if complete else 0.0
    return assessments, round(terminal, 4)


def rank_candidates(trace: Trace, assessments: list[SpanAssessment]) -> list[RootCauseCandidate]:
    """Every step that failed its own checks, in pipeline order.

    Pipeline order *is* the ranking: an upstream failure can explain a downstream one,
    never the reverse, so the earliest candidate is the first hypothesis to test.
    """
    ran = [a for a in assessments if a.role != SpanRole.NOT_RUN]
    damaged = [a for a in ran if 1 - a.local_quality >= OWN_DAMAGE_MIN]
    if not damaged and ran:
        # The trace failed on accumulated small drops with no single step failing its
        # own checks. Report the weakest step rather than nothing.
        damaged = [min(ran, key=lambda a: a.local_quality)]
    out = []
    for rank, a in enumerate(sorted(damaged, key=lambda a: STEP_ORDER.index(a.step_name)), start=1):
        span = next(s for s in trace.spans if s.span_id == a.span_id)
        category, subtype = categorize(span.step_name, span.signals)
        out.append(
            RootCauseCandidate(
                rank=rank, span_id=a.span_id, step_name=a.step_name,
                own_damage=round(1 - a.local_quality, 4), category=category, subtype=subtype,
            )
        )
    return out


def assign_roles(trace: Trace, assessments: list[SpanAssessment], root: Optional[RootCauseCandidate], candidates: list[RootCauseCandidate]) -> list[str]:
    candidate_ids = {c.span_id for c in candidates}
    root_index = STEP_ORDER.index(root.step_name) if root else None
    propagation: list[str] = []
    for a in assessments:
        if a.role == SpanRole.NOT_RUN:
            continue
        span = next(s for s in trace.spans if s.span_id == a.span_id)
        index = STEP_ORDER.index(a.step_name)
        if root and a.span_id == root.span_id:
            a.role = SpanRole.ROOT_CAUSE
        elif a.span_id in candidate_ids:
            a.role = SpanRole.CANDIDATE
        elif root_index is not None and index > root_index and a.q_out < FAIL_THRESHOLD:
            # Below the failure line, but its own checks passed: it inherited the damage.
            a.role = SpanRole.PROPAGATION
            propagation.append(a.span_id)
        elif (span.self_confidence is not None and span.self_confidence <= 2) or a.local_quality < 0.85:
            a.role = SpanRole.LOW_CONFIDENCE
        else:
            a.role = SpanRole.HEALTHY
    return propagation


# --------------------------------------------------------------------------
# Evidence chain
# --------------------------------------------------------------------------


def _bad_values(span: Span) -> list[tuple[str, FieldKind]]:
    """The concrete wrong values an origin span produced, so we can follow them downstream."""
    out: list[tuple[str, FieldKind]] = []
    for sig in span.signals:
        if sig.passed or sig.weight == 0:
            continue
        field = sig.evidence.get("field")
        kind = FIELD_KINDS.get(field or "", FieldKind.TEXT)
        for key in ("value", "chosen"):
            value = sig.evidence.get(key)
            if value and isinstance(value, str):
                out.append((value, kind))
    return list(dict.fromkeys(out))


def _where_found(value: str, kind: FieldKind, payload: Optional[dict[str, Any]]) -> list[str]:
    if not payload:
        return []
    hits = []
    for key, item in payload.items():
        text = json.dumps(item, ensure_ascii=False) if not isinstance(item, str) else item
        if ground(value, text, kind).grounded:
            hits.append(key)
    return hits


def build_evidence_chain(
    trace: Trace,
    assessments: list[SpanAssessment],
    root: Optional[RootCauseCandidate],
    terminal_quality: float,
) -> list[EvidenceLink]:
    if root is None:
        return []
    by_id = {s.span_id: s for s in trace.spans}
    lookup = {a.span_id: a for a in assessments}
    root_span = by_id[root.span_id]
    links: list[EvidenceLink] = []

    failed = sorted(
        [s for s in root_span.signals if not s.passed and s.weight > 0],
        key=signal_severity_mass,
        reverse=True,
    )
    for sig in failed[:3]:
        links.append(
            EvidenceLink(
                order=len(links) + 1,
                span_id=root_span.span_id,
                step_name=root_span.step_name,
                role="origin",
                signal=sig.name,
                claim=f"{root_span.step_name.value}: {sig.detail}",
                evidence=sig.evidence,
            )
        )

    bad = _bad_values(root_span)
    for span in trace.spans[STEP_ORDER.index(root.step_name) + 1 :]:
        a = lookup[span.span_id]
        if a.role == SpanRole.PROPAGATION:
            carried = [(v, _where_found(v, k, span.output_payload)) for v, k in bad]
            carried = [(v, where) for v, where in carried if where]
            if carried:
                what = "; ".join(f"{v!r} appears in its {', '.join(where)}" for v, where in carried)
                claim = f"{span.step_name.value} carried the error forward: {what}. Its own checks passed (local quality {a.local_quality:.2f})."
            else:
                claim = (
                    f"{span.step_name.value} consumed the damaged input and its own checks passed "
                    f"(local quality {a.local_quality:.2f}) - a propagation error, not an origin."
                )
            links.append(
                EvidenceLink(
                    order=len(links) + 1, span_id=span.span_id, step_name=span.step_name,
                    role="propagation", claim=claim, evidence={"carried": [v for v, _ in carried]},
                )
            )
        elif a.role == SpanRole.CANDIDATE:
            top = max((s for s in span.signals if not s.passed and s.weight > 0), key=signal_severity_mass, default=None)
            if top:
                links.append(
                    EvidenceLink(
                        order=len(links) + 1, span_id=span.span_id, step_name=span.step_name,
                        role="independent", signal=top.name,
                        claim=f"{span.step_name.value} also failed on its own: {top.detail}",
                        evidence=top.evidence,
                    )
                )

    last = trace.spans[-1]
    if trace.status == TraceStatus.ERROR:
        symptom = f"The pipeline stopped at {last.step_name.value}; no final output was produced."
    else:
        headline = (last.output_payload or {}).get("headline", "")
        symptom = f"Final output quality {terminal_quality:.2f} (failure threshold {FAIL_THRESHOLD})." + (
            f" Final headline: {headline!r}" if headline else ""
        )
    links.append(
        EvidenceLink(order=len(links) + 1, span_id=last.span_id, step_name=last.step_name, role="symptom", claim=symptom)
    )
    return links


# --------------------------------------------------------------------------
# Confidence and explanation
# --------------------------------------------------------------------------


def diagnosis_confidence(
    candidates: list[RootCauseCandidate],
    root_span: Optional[Span],
    counterfactual: Optional[CounterfactualResult],
) -> float:
    if not candidates or root_span is None:
        return 0.0
    top = next(c for c in candidates if c.span_id == root_span.span_id)
    failed = [s for s in root_span.signals if not s.passed and s.weight > 0]
    # Proven by code (a value absent from the input) beats suspected by a model.
    deterministic = any(s.kind.value == "deterministic" and s.severity == SignalSeverity.CRITICAL for s in failed)
    earliest = top.rank == 1

    conf = 0.45 + 0.3 * top.own_damage + (0.1 if deterministic else 0.0) + (0.05 if earliest else 0.0)
    if counterfactual is not None:
        if counterfactual.verdict == CounterfactualVerdict.CONFIRMED:
            conf = max(conf, 0.93)
        elif counterfactual.verdict == CounterfactualVerdict.PARTIAL:
            conf = min(max(conf, 0.7), 0.85)
        elif counterfactual.verdict == CounterfactualVerdict.REFUTED:
            conf = min(conf, 0.35)
    return round(max(0.05, min(0.99, conf)), 3)


def explain(
    root: Optional[RootCauseCandidate],
    root_span: Optional[Span],
    propagation: list[str],
    trace: Trace,
    counterfactual: Optional[CounterfactualResult],
) -> str:
    if root is None or root_span is None:
        return "No failure detected: every step passed its checks and the final output is above threshold."
    top = max((s for s in root_span.signals if not s.passed and s.weight > 0), key=signal_severity_mass, default=None)
    parts = [f"Root cause: {root.step_name.value} ({CATEGORY_LABELS[root.category]})."]
    if top:
        parts.append(top.detail.rstrip(".") + ".")
    if propagation:
        names = [s.step_name.value for s in trace.spans if s.span_id in propagation]
        parts.append(
            f"The error then propagated through {', '.join(names)}, "
            f"{'which' if len(names) > 1 else 'which'} behaved correctly given the input it received."
        )
    if counterfactual is not None:
        parts.append(f"Counterfactual {counterfactual.verdict.value}: {counterfactual.note}")
    return " ".join(parts)


# --------------------------------------------------------------------------
# The analyzer
# --------------------------------------------------------------------------


class Analyzer:
    def __init__(
        self,
        store: TraceStore,
        *,
        runner: Any = None,
        judge: Optional[Judge] = None,
        settings: Optional[Settings] = None,
        answer_keys: Optional[dict[str, AnswerKey]] = None,
        run_counterfactuals: bool = True,
    ) -> None:
        self.store = store
        self.runner = runner
        self.judge = judge
        self.settings = settings or get_settings()
        self.answer_keys = answer_keys or {}
        self.run_counterfactuals = run_counterfactuals and runner is not None

    def attach_signals(self, trace: Trace) -> int:
        """Fill in every span's signals. Returns the number of judge calls made."""
        calls_before = self.judge.calls if self.judge else 0
        for span in trace.spans:
            span.signals = deterministic_signals(span, self.settings)
            if self.judge is not None and not span.spliced and span.copied_from_span_id is None:
                span.signals.extend(self.judge.review(span, span.signals))
            elif span.copied_from_span_id is not None:
                # Copied spans are unchanged from the original: re-derive deterministic
                # signals but skip the judge (it would only repeat itself).
                pass
        return (self.judge.calls - calls_before) if self.judge else 0

    def _human_correction(self, trace_id: str, step: StepName) -> Optional[dict[str, Any]]:
        for flag in self.store.list_flags(trace_id):
            if flag.get("step_name") == step.value and flag.get("corrected_output"):
                return flag["corrected_output"]
        return None

    def _counterfactual(self, trace: Trace, candidate: RootCauseCandidate) -> CounterfactualResult:
        span = next(s for s in trace.spans if s.span_id == candidate.span_id)
        answer_key = self.answer_keys.get(trace.doc_id)
        if span.step_name == StepName.SUMMARIZATION:
            return CounterfactualResult(
                verdict=CounterfactualVerdict.NOT_APPLICABLE,
                splice_step=span.step_name,
                note="The root cause is the final step; there is nothing downstream to re-run.",
            )
        built = build_correction(
            span, trace, answer_key=answer_key, human_output=self._human_correction(trace.trace_id, span.step_name)
        )
        if built is None:
            return CounterfactualResult(
                verdict=CounterfactualVerdict.NOT_APPLICABLE,
                splice_step=span.step_name,
                note="No correction available for this step (no answer key, human flag or proposed fix).",
            )
        corrected, source = built
        reference = build_reference(span.step_name, corrected, trace, answer_key)
        doc = RawDocument(
            doc_id=trace.doc_id,
            source_text=trace.metadata.get("source_text", ""),
            channel=trace.metadata.get("channel", "email"),
        )
        rerun = None
        for _attempt in range(2):
            rerun = self.runner.run_counterfactual(
                trace, doc, splice_step=span.step_name, corrected_output=corrected,
                reason=f"test root-cause candidate {candidate.rank}",
            )
            self.attach_signals(rerun)
            self.store.save_trace(rerun)
            if not _infrastructure_failure(rerun):
                break
            # A model-server crash (observed: out-of-memory on a ~1 GiB-free machine) is
            # usually transient. Retry once; LLM calls that already succeeded are cached.
        if rerun is not None and _infrastructure_failure(rerun):
            return CounterfactualResult(
                verdict=CounterfactualVerdict.INCONCLUSIVE,
                splice_step=span.step_name,
                correction_source=source,
                corrected_output=corrected.model_dump(mode="json"),
                counterfactual_trace_id=rerun.trace_id,
                note=(
                    f"The re-run from corrected {span.step_name.value} could not complete: "
                    f"{_infrastructure_failure(rerun)}. This says nothing about the diagnosis either way."
                ),
            )
        return judge_counterfactual(
            step=span.step_name, original=trace, original_span=span, rerun=rerun,
            reference=reference, correction_source=source, corrected=corrected,
        )

    def analyze(self, trace: Trace) -> Diagnosis:
        started = time.perf_counter()
        judge_calls = self.attach_signals(trace)
        assessments, terminal = assess(trace)
        is_failure = terminal < FAIL_THRESHOLD or trace.status == TraceStatus.ERROR
        candidates = rank_candidates(trace, assessments) if is_failure else []

        root: Optional[RootCauseCandidate] = candidates[0] if candidates else None
        counterfactuals: list[CounterfactualResult] = []
        if root is not None and self.run_counterfactuals:
            # Counterfactuals *eliminate*; they never promote. Walk candidates in
            # pipeline order and take the first one not ruled out. An earlier version
            # jumped to any later candidate that passed its test - which let extraction
            # overtake an intake step whose own test had merely crashed, never failed.
            eliminated = (CounterfactualVerdict.REFUTED, CounterfactualVerdict.CONTAINED)
            root = None
            for candidate in candidates[:MAX_COUNTERFACTUALS]:
                result = self._counterfactual(trace, candidate)
                counterfactuals.append(result)
                if result.verdict not in eliminated:
                    root = candidate
                    break
            # Every tested candidate eliminated: report the earliest, with low confidence.
            root = root or candidates[0]

        if root is not None and root.rank != 1:
            # A lower-ranked candidate survived the test the top one failed: promote it.
            candidates = [root] + [c for c in candidates if c.span_id != root.span_id]
            for i, c in enumerate(candidates, start=1):
                c.rank = i

        propagation = assign_roles(trace, assessments, root, [c for c in candidates if c is not root])
        root_span = next((s for s in trace.spans if root and s.span_id == root.span_id), None)
        chosen_cf = next((cf for cf in counterfactuals if root and cf.splice_step == root.step_name), None)
        chain = build_evidence_chain(trace, assessments, root, terminal)
        if chosen_cf is not None:
            chain.append(
                EvidenceLink(
                    order=len(chain) + 1,
                    span_id=root.span_id if root else "",
                    step_name=root.step_name if root else StepName.SUMMARIZATION,
                    role="verification",
                    claim=f"Counterfactual {chosen_cf.verdict.value}: {chosen_cf.note}",
                    evidence={"counterfactual_trace_id": chosen_cf.counterfactual_trace_id, "correction_source": chosen_cf.correction_source},
                )
            )

        if is_failure and trace.status == TraceStatus.OK:
            trace.status = TraceStatus.FAILED
        self.store.save_trace(trace)

        diagnosis = Diagnosis(
            trace_id=trace.trace_id,
            doc_id=trace.doc_id,
            is_failure=is_failure,
            terminal_quality=terminal,
            root_span_id=root.span_id if root else None,
            root_step=root.step_name if root else None,
            category=root.category if root else FailureCategory.NO_FAILURE,
            subtype=root.subtype if root else "",
            confidence=diagnosis_confidence(candidates, root_span, chosen_cf) if root else round(terminal, 3),
            explanation=explain(root, root_span, propagation, trace, chosen_cf),
            candidates=candidates,
            assessments=assessments,
            evidence_chain=chain,
            propagation_path=propagation,
            counterfactuals=counterfactuals,
            judge_provider=self.judge.provider if self.judge else "",
            judge_calls=judge_calls,
            diagnose_ms=int((time.perf_counter() - started) * 1000),
        )
        self.store.save_diagnosis(diagnosis)
        return diagnosis

    def analyze_run(
        self,
        run_id: str,
        *,
        on_diagnosis: Optional[Callable[[Diagnosis, Trace], None]] = None,
        only_doc_ids: Optional[set[str]] = None,
    ) -> list[Diagnosis]:
        out = []
        for row in self.store.list_traces(run_id=run_id):
            if only_doc_ids and row["doc_id"] not in only_doc_ids:
                continue
            trace = self.store.load_trace(row["trace_id"])
            if trace is None:
                continue
            diagnosis = self.analyze(trace)
            out.append(diagnosis)
            if on_diagnosis:
                on_diagnosis(diagnosis, trace)
        return out


def _infrastructure_failure(trace: Trace) -> str:
    """The model-server error that stopped ``trace``, or "" if it completed normally."""
    for span in trace.spans:
        if span.error is not None and span.error.type == "LLMTransportError":
            return f"{span.step_name.value}: {span.error.message[:160]}"
    return ""


def load_diagnosis(store: TraceStore, trace_id: str) -> Optional[Diagnosis]:
    body = store.load_diagnosis_json(trace_id)
    return Diagnosis.model_validate_json(body) if body else None
