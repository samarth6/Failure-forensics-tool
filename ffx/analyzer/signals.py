"""Deterministic signals: evidence about each span that needs no LLM.

The one rule every check here obeys
-----------------------------------
**A span is judged against its own input, never against the raw document.**

That is what makes the backward walk work. If extraction invents a policy number and
classification faithfully routes on it, classification's checks all pass - it did its
job with what it was given - so its *local* quality stays high while the trace's
cumulative quality stays low. The diagnoser reads that pattern as "propagation", not as
a second root cause. A check that compared classification against the source document
would blame both steps and make origin-finding impossible.

Every signal also declares which failure category it implicates, so the taxonomy is read
straight off the evidence (see :mod:`ffx.analyzer.taxonomy`).
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from ffx.analyzer.grounding import (
    FIELD_KINDS,
    FieldKind,
    candidates,
    find_amounts,
    find_dates,
    find_identifiers,
    find_names,
    ground,
    parse_amount,
    quote_grounding,
    values_equal,
)
from ffx.config import Settings
from ffx.models import FailureCategory, Peril, StepName
from ffx.tracing.span import Signal, SignalKind, SignalSeverity, Span, SpanStatus

SEVERITY_WEIGHT = {SignalSeverity.CRITICAL: 1.0, SignalSeverity.WARN: 0.5, SignalSeverity.INFO: 0.25}
SELF_REPORT_WEIGHT = 0.15


def make_signal(
    name: str,
    *,
    passed: bool,
    score: float,
    severity: SignalSeverity = SignalSeverity.INFO,
    detail: str = "",
    implicates: Optional[FailureCategory] = None,
    kind: SignalKind = SignalKind.DETERMINISTIC,
    weight: Optional[float] = None,
    evidence: Optional[dict[str, Any]] = None,
    correction: Optional[dict[str, Any]] = None,
) -> Signal:
    return Signal(
        name=name,
        kind=kind,
        severity=severity,
        passed=passed,
        score=max(0.0, min(1.0, score)),
        weight=SEVERITY_WEIGHT[severity] if weight is None else weight,
        detail=detail,
        evidence=evidence or {},
        implicates=implicates,
        proposed_correction=correction,
    )


# --------------------------------------------------------------------------
# Signals common to every span
# --------------------------------------------------------------------------


def schema_signal(span: Span) -> Signal:
    if span.status == SpanStatus.ERROR and span.error and span.error.type == "LLMTransportError":
        return make_signal(
            "infrastructure",
            passed=False,
            score=0.0,
            severity=SignalSeverity.CRITICAL,
            detail=f"the model server failed, not the model: {span.error.message[:200]}",
            implicates=FailureCategory.INFRASTRUCTURE_ERROR,
        )
    if span.status == SpanStatus.ERROR:
        return make_signal(
            "schema_valid",
            passed=False,
            score=0.0,
            severity=SignalSeverity.CRITICAL,
            detail=f"no valid output after {span.attempt_count} attempt(s): "
            + (span.error.message[:200] if span.error else ""),
            implicates=FailureCategory.PROMPT_FAILURE,
            evidence={"attempts": span.attempt_count},
        )
    if span.attempt_count > 1:
        return make_signal(
            "schema_valid",
            passed=False,
            score=0.6,
            severity=SignalSeverity.WARN,
            detail=f"output only validated on attempt {span.attempt_count}",
            implicates=FailureCategory.PROMPT_FAILURE,
            evidence={"errors": [a.error for a in span.attempts if not a.ok]},
        )
    return make_signal("schema_valid", passed=True, score=1.0, detail="valid on first attempt")


def self_confidence_signal(span: Span) -> Optional[Signal]:
    """Weak by design: logged for the calibration analysis, barely moves span quality."""
    if span.self_confidence is None:
        return None
    c = span.self_confidence
    return make_signal(
        "self_confidence",
        passed=c >= 3,
        score=(c - 1) / 4,
        kind=SignalKind.SELF_REPORT,
        weight=SELF_REPORT_WEIGHT,
        detail=f"model reported confidence {c}/5",
    )


def context_signals(span: Span, settings: Settings) -> list[Signal]:
    meta = span.input_payload.get("_meta", {}) if isinstance(span.input_payload, dict) else {}
    num_ctx = int(meta.get("num_ctx") or settings.num_ctx)
    prompt_chars = int(meta.get("prompt_chars") or len(span.prompt))
    est_tokens = int(prompt_chars / 3.6)
    out: list[Signal] = []

    # Ollama silently truncates an over-long prompt. prompt_eval_count pinned at the
    # window size, or a prompt that plainly cannot fit, means the step never saw all
    # of its input - regardless of how confident its output looks.
    overflowed = span.tokens_in >= num_ctx - 64 or est_tokens > num_ctx
    if overflowed:
        out.append(
            make_signal(
                "context_window",
                passed=False,
                score=0.1,
                severity=SignalSeverity.CRITICAL,
                detail=f"prompt ~{est_tokens} tokens vs num_ctx {num_ctx}; model saw {span.tokens_in}. "
                "Part of the input was silently dropped.",
                implicates=FailureCategory.CONTEXT_LOSS,
                evidence={"estimated_tokens": est_tokens, "tokens_in": span.tokens_in, "num_ctx": num_ctx},
            )
        )
    elif span.tokens_in:
        out.append(
            make_signal(
                "context_window",
                passed=True,
                score=1.0,
                detail=f"{span.tokens_in}/{num_ctx} tokens",
            )
        )

    if meta.get("done_reason") == "length":
        out.append(
            make_signal(
                "output_budget",
                passed=False,
                score=0.4,
                severity=SignalSeverity.WARN,
                detail=f"generation stopped at the {meta.get('max_tokens')}-token budget",
                implicates=FailureCategory.PROMPT_FAILURE,
            )
        )
    return out


# --------------------------------------------------------------------------
# Intake
# --------------------------------------------------------------------------

CLAIM_WORDS = {
    "claim", "claiming", "policy", "insured", "insurer", "loss", "damage", "damaged", "incident",
    "accident", "injured", "injury", "stolen", "theft", "burglars", "fire", "flood", "flooded",
    "leak", "leaked", "burst", "storm", "collided", "compensation", "poliza", "siniestro",
}


def _claim_word_count(text: str) -> int:
    words = set(re.findall(r"[a-záéíóúñ]+", text.casefold()))
    return len(words & CLAIM_WORDS)


def salient_items(text: str) -> list[tuple[str, FieldKind]]:
    """Identifiers, dates, money amounts and proper names: the facts intake must never
    lose or alter."""
    items: list[tuple[str, FieldKind]] = []
    seen: set[str] = set()
    for name in find_names(text):
        if name not in seen:
            seen.add(name)
            items.append((name, FieldKind.TEXT))
    for token in find_identifiers(text):
        if token not in seen:
            seen.add(token)
            items.append((token, FieldKind.IDENTIFIER))
    for d, span in find_dates(text):
        key = d.isoformat()
        if key not in seen:
            seen.add(key)
            items.append((span, FieldKind.DATE))
    for amount, span in find_amounts(text, money_only=True):
        key = f"amt:{amount}"
        if key not in seen:
            seen.add(key)
            items.append((span, FieldKind.AMOUNT))
    return items


def intake_signals(span: Span) -> list[Signal]:
    if not span.output_payload:
        return []
    source = span.input_payload.get("source_text", "")
    normalized = span.output_payload.get("normalized_text", "")
    doc_type = span.output_payload.get("detected_doc_type", "unknown")
    out: list[Signal] = []

    # Retention: every salient fact in the source must survive into normalized_text.
    items = salient_items(source)
    if items:
        lost_items = [(value, kind) for value, kind in items if not ground(value, normalized, kind).grounded]
        lost = [value for value, _ in lost_items]
        retention = 1 - len(lost) / len(items)
        if not lost:
            out.append(make_signal("intake_retention", passed=True, score=1.0, detail=f"all {len(items)} salient facts retained"))
        else:
            # Intake is told to keep every fact. Losing any identifier, date or amount -
            # or more than a tenth of the facts - is a critical failure, not a nuisance.
            critical = retention < 0.9 or any(kind != FieldKind.TEXT for _, kind in lost_items)
            out.append(
                make_signal(
                    "intake_retention",
                    passed=False,
                    score=retention * (0.5 if critical else 1.0),
                    severity=SignalSeverity.CRITICAL if critical else SignalSeverity.WARN,
                    detail=f"lost {len(lost)}/{len(items)} salient facts: {', '.join(lost[:6])}",
                    implicates=FailureCategory.CONTEXT_LOSS,
                    evidence={"lost": lost, "total": len(items)},
                )
            )

    # Fidelity: intake must not introduce or alter facts (an over-eager "OCR fix").
    introduced = [value for value, kind in salient_items(normalized) if not ground(value, source, kind).grounded]
    if introduced:
        out.append(
            make_signal(
                "intake_fidelity",
                passed=False,
                score=0.1,
                severity=SignalSeverity.CRITICAL,
                detail=f"normalized text contains facts not in the source: {', '.join(introduced[:6])}",
                implicates=FailureCategory.CONTEXT_LOSS,
                evidence={"introduced": introduced},
            )
        )
    elif normalized:
        out.append(make_signal("intake_fidelity", passed=True, score=1.0, detail="no facts introduced"))

    # Document type vs content.
    claim_words = _claim_word_count(source)
    if doc_type == "claim_notice" and claim_words <= 1:
        out.append(
            make_signal(
                "doc_type_consistency",
                passed=False,
                score=0.1,
                severity=SignalSeverity.CRITICAL,
                detail=f"classified as claim_notice but the text contains {claim_words} claim-related term(s)",
                implicates=FailureCategory.MISCLASSIFICATION,
                evidence={"claim_words": claim_words},
                correction={"detected_doc_type": "correspondence"},
            )
        )
    elif doc_type != "claim_notice" and claim_words >= 4:
        out.append(
            make_signal(
                "doc_type_consistency",
                passed=False,
                score=0.2,
                severity=SignalSeverity.CRITICAL,
                detail=f"classified as {doc_type} but the text contains {claim_words} claim-related terms",
                implicates=FailureCategory.MISCLASSIFICATION,
                evidence={"claim_words": claim_words},
                correction={"detected_doc_type": "claim_notice"},
            )
        )
    else:
        out.append(
            make_signal("doc_type_consistency", passed=True, score=1.0, detail=f"{doc_type}, {claim_words} claim terms")
        )
    return out


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

PERIL_KEYWORDS = {
    Peril.FIRE.value: {"fire", "smoke", "flame", "flames", "burn", "burned", "burnt", "scorched", "blaze"},
    Peril.WATER.value: {"water", "leak", "leaked", "leaking", "flood", "flooded", "burst", "pipe", "hose", "soaking"},
    Peril.THEFT.value: {"stolen", "theft", "burglars", "burglary", "broken", "robbed", "forced"},
    Peril.COLLISION.value: {"collided", "collision", "rear-ended", "crash", "crashed", "vehicle", "van", "car"},
    Peril.WIND.value: {"storm", "wind", "winds", "gale", "hail", "tornado", "chimney"},
    Peril.LIABILITY.value: {
        "slipped", "tripped", "fell", "injured", "injury", "fractured", "solicitor", "compensation",
        "customer", "visitor", "driver",
    },
}


def peril_evidence(text: str) -> dict[str, list[str]]:
    words = set(re.findall(r"[a-z]+(?:-[a-z]+)?", text.casefold()))
    return {peril: sorted(words & kws) for peril, kws in PERIL_KEYWORDS.items() if words & kws}


def extraction_signals(span: Span) -> list[Signal]:
    if not span.output_payload:
        return []
    text = span.input_payload.get("normalized_text", "")
    fields: dict[str, str] = span.output_payload.get("fields", {})
    evidence = span.output_payload.get("evidence", [])
    by_field: dict[str, dict[str, Any]] = {}
    for ev in evidence:
        by_field.setdefault(ev.get("field", ""), ev)
    out: list[Signal] = []

    for field, kind in FIELD_KINDS.items():
        value = (fields.get(field) or "").strip()
        cands = candidates(field, text, like=value) if kind != FieldKind.TEXT else []

        if not value:
            if cands:
                out.append(
                    make_signal(
                        f"missed_field:{field}",
                        passed=False,
                        score=0.4,
                        severity=SignalSeverity.WARN,
                        detail=f"left empty, but the input contains {', '.join(cands)}",
                        implicates=FailureCategory.CONTEXT_LOSS,
                        evidence={"field": field, "candidates": cands, "needs_judge": True},
                    )
                )
            continue

        # (1) The value itself must be present in the step's input.
        g = ground(value, text, kind)
        if not g.grounded:
            out.append(
                make_signal(
                    f"value_grounded:{field}",
                    passed=False,
                    score=0.0,
                    severity=SignalSeverity.CRITICAL,
                    detail=f"{value!r} does not appear in the step input. {g.detail}".strip(),
                    implicates=FailureCategory.EXTRACTION_HALLUCINATION,
                    evidence={"field": field, "value": value, "candidates": cands, "needs_judge": len(cands) > 0},
                    correction={field: cands[0] if len(cands) == 1 else ""},
                )
            )
        else:
            out.append(
                make_signal(
                    f"value_grounded:{field}",
                    passed=True,
                    score=g.score,
                    detail=f"{value!r} grounded ({g.level.value}{': ' + g.matched if g.matched and g.matched != value else ''})",
                    evidence={"field": field, "value": value, "level": g.level.value},
                )
            )

        # (2) The cited evidence must be real and must support the value.
        ev = by_field.get(field)
        if ev is None:
            out.append(
                make_signal(
                    f"evidence:{field}",
                    passed=False,
                    score=0.7,
                    severity=SignalSeverity.INFO,
                    detail="no evidence entry cited for this value",
                    evidence={"field": field},
                )
            )
        else:
            quote = ev.get("source_quote", "")
            q = quote_grounding(quote, text)
            supports = ground(value, quote, kind).grounded if quote else False
            if not q.grounded:
                out.append(
                    make_signal(
                        f"evidence:{field}",
                        passed=False,
                        score=0.2,
                        severity=SignalSeverity.WARN,
                        detail=f"cited quote {quote[:80]!r} is not in the input - fabricated evidence",
                        implicates=FailureCategory.EXTRACTION_HALLUCINATION,
                        evidence={"field": field, "quote": quote, "match": q.detail},
                    )
                )
            elif not supports:
                out.append(
                    make_signal(
                        f"evidence:{field}",
                        passed=False,
                        score=0.5,
                        severity=SignalSeverity.WARN,
                        detail=f"quote is real but does not contain {value!r}",
                        implicates=FailureCategory.EXTRACTION_HALLUCINATION,
                        evidence={"field": field, "quote": quote},
                    )
                )
            else:
                out.append(make_signal(f"evidence:{field}", passed=True, score=q.score, detail="quote real and supports value"))

        # (3) A genuine choice between several candidates is flagged for the judge.
        if g.grounded and len(cands) > 1:
            out.append(
                make_signal(
                    f"ambiguous:{field}",
                    passed=True,
                    score=0.9,
                    severity=SignalSeverity.INFO,
                    detail=f"input offers {len(cands)} candidates {cands}; chose {value!r}",
                    evidence={"field": field, "candidates": cands, "chosen": value, "needs_judge": True},
                )
            )

    out.extend(_peril_signals(fields.get("peril", "unknown"), text))
    return out


def _peril_signals(chosen: str, text: str) -> list[Signal]:
    present = peril_evidence(text)
    if not present:
        return []
    evidence = {"chosen": chosen, "keywords": present}
    if chosen in present and len(present) == 1:
        return [make_signal("peril_support", passed=True, score=1.0, detail=f"{chosen} supported by {present[chosen]}")]
    if chosen in present:
        return [
            make_signal(
                "peril_support",
                passed=True,
                score=0.85,
                detail=f"{chosen} supported, but vocabulary for {sorted(present)} is present",
                evidence={**evidence, "field": "peril", "needs_judge": True},
            )
        ]
    if len(present) == 1:
        only = next(iter(present))
        return [
            make_signal(
                "peril_support",
                passed=False,
                score=0.1,
                severity=SignalSeverity.CRITICAL,
                detail=f"chose {chosen!r} but the input only describes {only} ({', '.join(present[only])})",
                implicates=FailureCategory.MISCLASSIFICATION,
                evidence=evidence,
                correction={"peril": only},
            )
        ]
    return [
        make_signal(
            "peril_support",
            passed=False,
            score=0.35,
            severity=SignalSeverity.WARN,
            detail=f"chose {chosen!r} with no supporting vocabulary; input mentions {sorted(present)}",
            implicates=FailureCategory.MISCLASSIFICATION,
            evidence={**evidence, "field": "peril", "needs_judge": True},
        )
    ]


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

# Theft maps to property only. Vehicles are stolen too, but classification's input has
# no vehicle field, so nothing it *receives* can justify "auto" - and every check here
# judges a step against its own input. (Observed: a home burglary of jewellery, a
# laptop and bicycles classified as auto, then summarised as "theft of his vehicle".)
PERIL_TO_CATEGORIES = {
    "fire": {"property"},
    "water": {"property"},
    "wind": {"property"},
    "theft": {"property"},
    "collision": {"auto"},
    "liability": {"liability"},
}
SEVERITY_ORDER = ["low", "medium", "high", "critical"]


def expected_severity(amount: str) -> Optional[str]:
    value = parse_amount(amount) if amount else None
    if value is None:
        return None
    if value < 5_000:
        return "low"
    if value < 25_000:
        return "medium"
    if value < 100_000:
        return "high"
    return "critical"


def classification_signals(span: Span) -> list[Signal]:
    if not span.output_payload:
        return []
    inp = span.input_payload
    fields = inp.get("fields", {})
    doc_type = inp.get("doc_type", "claim_notice")
    category = span.output_payload.get("claim_category")
    severity = span.output_payload.get("severity")
    routing = span.output_payload.get("routing_queue")
    out: list[Signal] = []

    if doc_type != "claim_notice":
        expected = {"unknown"}
    else:
        expected = PERIL_TO_CATEGORIES.get(fields.get("peril", ""), set())
    if expected and category not in expected:
        out.append(
            make_signal(
                "category_consistency",
                passed=False,
                score=0.1,
                severity=SignalSeverity.CRITICAL,
                detail=f"category {category!r} contradicts its input (doc_type={doc_type}, peril={fields.get('peril')}); expected {sorted(expected)}",
                implicates=FailureCategory.MISCLASSIFICATION,
                correction={"claim_category": next(iter(expected))} if len(expected) == 1 else None,
            )
        )
    elif expected:
        out.append(make_signal("category_consistency", passed=True, score=1.0, detail=f"{category} consistent with input"))

    exp_sev = expected_severity(fields.get("reported_amount", ""))
    if exp_sev and severity in SEVERITY_ORDER:
        gap = abs(SEVERITY_ORDER.index(severity) - SEVERITY_ORDER.index(exp_sev))
        if gap:
            out.append(
                make_signal(
                    "severity_consistency",
                    passed=False,
                    score=0.5 if gap == 1 else 0.2,
                    severity=SignalSeverity.WARN if gap == 1 else SignalSeverity.CRITICAL,
                    detail=f"severity {severity!r} but amount {fields.get('reported_amount')} implies {exp_sev!r}",
                    implicates=FailureCategory.MISCLASSIFICATION,
                    correction={"severity": exp_sev},
                )
            )
        else:
            out.append(make_signal("severity_consistency", passed=True, score=1.0, detail=f"{severity} matches amount"))

    if routing == "fast_track" and severity in ("high", "critical"):
        out.append(
            make_signal(
                "routing_consistency",
                passed=False,
                score=0.2,
                severity=SignalSeverity.CRITICAL,
                detail=f"fast_track routing for a {severity}-severity claim",
                implicates=FailureCategory.MISCLASSIFICATION,
                correction={"routing_queue": "complex"},
            )
        )
    elif routing == "fast_track" and not fields.get("policy_number"):
        out.append(
            make_signal(
                "routing_consistency",
                passed=False,
                score=0.5,
                severity=SignalSeverity.WARN,
                detail="fast_track routing with no policy number",
                implicates=FailureCategory.MISCLASSIFICATION,
                correction={"routing_queue": "standard"},
            )
        )
    elif routing:
        out.append(make_signal("routing_consistency", passed=True, score=1.0, detail=f"{routing} plausible"))
    return out


# --------------------------------------------------------------------------
# Summarization
# --------------------------------------------------------------------------

SUMMARY_FIELDS = ["claimant_name", "policy_number", "date_of_loss", "reported_amount"]
_MONEY_SHAPED = re.compile(r"(?<![\w-])\d{1,3}(?:,\d{3})+(?:\.\d{2})?(?![\w-])|(?<![\w.-])\d+\.\d{2}(?![\w-])")


def _summary_text(output: dict[str, Any]) -> str:
    return "\n".join([output.get("headline", ""), output.get("body", ""), *output.get("key_facts", [])])


def summarization_signals(span: Span) -> list[Signal]:
    if not span.output_payload:
        return []
    fields = span.input_payload.get("fields", {})
    summary = _summary_text(span.output_payload)
    input_text = json.dumps(
        {k: v for k, v in span.input_payload.items() if k != "_meta"}, ensure_ascii=False
    )
    out: list[Signal] = []

    # Retention: the facts the adjuster needs must survive into the brief.
    expected = [(f, fields.get(f, "")) for f in SUMMARY_FIELDS if fields.get(f)]
    if expected:
        dropped = [f for f, v in expected if not ground(v, summary, FIELD_KINDS[f]).grounded]
        ratio = 1 - len(dropped) / len(expected)
        if dropped:
            critical = ratio < 0.75
            out.append(
                make_signal(
                    "summary_retention",
                    passed=False,
                    score=ratio * (0.5 if critical else 1.0),
                    severity=SignalSeverity.CRITICAL if critical else SignalSeverity.WARN,
                    detail=f"summary drops {', '.join(dropped)}",
                    implicates=FailureCategory.CONTEXT_LOSS,
                    evidence={"dropped": dropped},
                )
            )
        else:
            out.append(make_signal("summary_retention", passed=True, score=1.0, detail=f"all {len(expected)} key facts present"))

    # Support: every identifier, date and amount the brief asserts must come from its input.
    unsupported: list[str] = []
    for token in find_identifiers(summary):
        if not ground(token, input_text, FieldKind.IDENTIFIER).grounded:
            unsupported.append(token)
    for d, span_text in find_dates(summary):
        if not any(values_equal(d.isoformat(), v, FieldKind.DATE) for v in [fields.get("date_of_loss", "")]):
            unsupported.append(span_text)
    stripped = _IDENT_OR_DATE.sub(" ", summary)
    claimed = fields.get("reported_amount", "")
    for m in _MONEY_SHAPED.finditer(stripped):
        if not values_equal(m.group(0), claimed, FieldKind.AMOUNT):
            unsupported.append(m.group(0))
    for amount, raw in find_amounts(stripped, money_only=True):
        if raw not in unsupported and not values_equal(str(amount), claimed, FieldKind.AMOUNT):
            unsupported.append(raw)

    if unsupported:
        out.append(
            make_signal(
                "summary_support",
                passed=False,
                score=0.1,
                severity=SignalSeverity.CRITICAL,
                detail=f"brief asserts facts absent from its input: {', '.join(dict.fromkeys(unsupported))}",
                implicates=FailureCategory.SUMMARY_HALLUCINATION,
                evidence={"unsupported": list(dict.fromkeys(unsupported))},
            )
        )
    else:
        out.append(make_signal("summary_support", passed=True, score=1.0, detail="every stated fact traceable to input"))
    return out


_IDENT_OR_DATE = re.compile(
    r"\b[A-Za-z0-9]{2,5}-[A-Za-z0-9]{3,}\b|\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b"
)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

STEP_SIGNALS = {
    StepName.INTAKE: intake_signals,
    StepName.EXTRACTION: extraction_signals,
    StepName.CLASSIFICATION: classification_signals,
    StepName.SUMMARIZATION: summarization_signals,
}


def deterministic_signals(span: Span, settings: Settings) -> list[Signal]:
    """All code-only evidence for one span. Spliced (corrected) spans get none - their
    output is by construction what the step should have produced."""
    if span.spliced:
        return [make_signal("counterfactual_splice", passed=True, score=1.0, detail="corrected output injected by analyzer")]
    signals = [schema_signal(span)]
    confidence = self_confidence_signal(span)
    if confidence:
        signals.append(confidence)
    signals.extend(context_signals(span, settings))
    signals.extend(STEP_SIGNALS[span.step_name](span))
    return signals
