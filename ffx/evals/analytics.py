"""Failure analytics: what fails, where, how fast we find it, and whether to trust it.

Everything here is computed from records that already exist - diagnoses, traces, human
flags, eval results and manual MTTR timings. Nothing calls a model.

Three kinds of reference are used, and every number says which one it is scored against:

* **ground truth** - the corpus answer key, via :mod:`ffx.analyzer.truth`;
* **human** - the reviewer's latest verdict in the explorer;
* **diagnosis** - the tool's own verdict (used only where nothing better exists).
"""

from __future__ import annotations

from collections import Counter
from statistics import mean, median
from typing import Any, Optional

from ffx.analyzer.diagnose import Diagnosis
from ffx.analyzer.diagnosis_models import SpanRole
from ffx.analyzer.grounding import FIELD_KINDS
from ffx.analyzer.taxonomy import CATEGORY_LABELS
from ffx.analyzer.truth import TruthLabel, label
from ffx.evals.cases import same_value
from ffx.evals.runner import regression_matrix
from ffx.models import STEP_ORDER, AnswerKey, FailureCategory
from ffx.tracing.span import SignalKind, Trace
from ffx.tracing.store import TraceStore

CONFIDENCE_BINS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0001)]


def _rate(part: int, whole: int) -> Optional[float]:
    return round(part / whole, 3) if whole else None


def _percentile(values: list[int], q: float) -> Optional[int]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def default_run_id(store: TraceStore) -> Optional[str]:
    """The newest run that has been diagnosed - analytics on an undiagnosed run is empty."""
    for run in store.list_runs():
        if store.list_diagnosis_rows(run["run_id"]):
            return run["run_id"]
    return None


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def category_mix(failures: list[Diagnosis]) -> list[dict[str, Any]]:
    counts = Counter(d.category for d in failures)
    return [
        {"category": c.value, "label": CATEGORY_LABELS[c], "count": n, "share": _rate(n, len(failures))}
        for c, n in counts.most_common()
    ]


def per_step(diagnoses: list[Diagnosis]) -> list[dict[str, Any]]:
    out = []
    for step in STEP_ORDER:
        roots = sum(1 for d in diagnoses if d.is_failure and d.root_step == step)
        own = sum(1 for d in diagnoses if d.is_failure and any(c.step_name == step for c in d.candidates))
        propagated = sum(
            1 for d in diagnoses for a in d.assessments if a.step_name == step and a.role == SpanRole.PROPAGATION
        )
        out.append(
            {
                "step": step.value,
                "root_cause": roots,
                "failed_own_checks": own,
                "propagation": propagated,
                "own_failure_rate": _rate(own, len(diagnoses)),
            }
        )
    return out


def mttr(failures: list[Diagnosis], baselines: list[dict[str, Any]]) -> dict[str, Any]:
    times = [d.diagnose_ms for d in failures]
    with_cf = [d.diagnose_ms for d in failures if any(cf.counterfactual_trace_id for cf in d.counterfactuals)]
    without_cf = [d.diagnose_ms for d in failures if not any(cf.counterfactual_trace_id for cf in d.counterfactuals)]
    automated = {
        "n": len(times),
        "mean_ms": round(mean(times)) if times else None,
        "median_ms": round(median(times)) if times else None,
        "p95_ms": _percentile(times, 0.95),
        "with_counterfactual_median_ms": round(median(with_cf)) if with_cf else None,
        "signals_only_median_ms": round(median(without_cf)) if without_cf else None,
    }
    minutes = [b["minutes"] for b in baselines]
    manual = {"n": len(minutes), "mean_minutes": round(mean(minutes), 1) if minutes else None, "entries": baselines}
    speedup = None
    if minutes and automated["mean_ms"]:
        speedup = round(mean(minutes) * 60_000 / automated["mean_ms"])
    return {
        "automated": automated,
        "manual": manual,
        "speedup": speedup,
        "note": None if minutes else "No manual baseline yet. Time yourself root-causing a trace by hand, then record it.",
    }


def versus_truth(diagnoses: list[Diagnosis], truths: dict[str, TruthLabel]) -> Optional[dict[str, Any]]:
    scored = [(d, truths[d.trace_id]) for d in diagnoses if d.trace_id in truths]
    if not scored:
        return None
    tp = sum(1 for d, t in scored if d.is_failure and t.is_failure)
    fp = sum(1 for d, t in scored if d.is_failure and not t.is_failure)
    fn = sum(1 for d, t in scored if not d.is_failure and t.is_failure)
    tn = sum(1 for d, t in scored if not d.is_failure and not t.is_failure)
    both = [(d, t) for d, t in scored if d.is_failure and t.is_failure]
    root_ok = [(d, t) for d, t in both if d.root_step == t.root_step]
    confusion = Counter(
        ((t.root_step.value if t.root_step else "none"), (d.root_step.value if d.root_step else "none"))
        for d, t in scored
        if d.is_failure or t.is_failure
    )
    return {
        "n": len(scored),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": _rate(tp, tp + fp),
        "recall": _rate(tp, tp + fn),
        "root_accuracy": _rate(len(root_ok), len(both)),
        "category_accuracy": _rate(sum(1 for d, t in root_ok if d.category == t.category), len(root_ok)),
        "confusion": [{"truth": a, "diagnosed": b, "count": n} for (a, b), n in sorted(confusion.items())],
    }


def versus_human(diagnoses: list[Diagnosis], flags: list[dict[str, Any]]) -> dict[str, Any]:
    latest: dict[str, dict[str, Any]] = {}
    for flag in flags:  # newest first
        latest.setdefault(flag["trace_id"], flag)
    by_trace = {d.trace_id: d for d in diagnoses}
    counts = Counter()
    for trace_id, flag in latest.items():
        d = by_trace.get(trace_id)
        if d is None:
            continue
        counts["reviewed"] += 1
        verdict = flag["verdict"]
        if verdict == "confirm":
            counts["agree"] += 1
        elif verdict == "not_a_failure":
            counts["agree" if not d.is_failure else "false_alarm"] += 1
        elif not d.is_failure:
            counts["missed"] += 1
        elif flag.get("step_name") != (d.root_step.value if d.root_step else None):
            counts["wrong_step"] += 1
        else:
            counts["wrong_category"] += 1
    return {
        "reviewed": counts["reviewed"],
        "agree": counts["agree"],
        "wrong_step": counts["wrong_step"],
        "wrong_category": counts["wrong_category"],
        "false_alarm": counts["false_alarm"],
        "missed": counts["missed"],
        "agreement": _rate(counts["agree"], counts["reviewed"]),
    }


def judge_accuracy(traces: list[Trace], keys: dict[str, AnswerKey]) -> dict[str, Any]:
    """Score every judge verdict against the answer key.

    precision: when the judge said extraction was wrong, how often it really was.
    recall:    of the judged fields extraction really got wrong, how many the judge caught.
    """
    verdicts = discarded = scored = correct = flagged = flagged_right = really_wrong = caught = 0
    providers: Counter = Counter()
    for trace in traces:
        key = keys.get(trace.doc_id)
        for span in trace.spans:
            for sig in span.signals:
                if sig.kind != SignalKind.LLM_JUDGE or "judge_choice" not in sig.evidence:
                    continue
                verdicts += 1
                providers[sig.evidence.get("judge_provider", "?")] += 1
                if sig.weight == 0:
                    discarded += 1
                    continue
                field = sig.evidence.get("field", "")
                if key is None or not hasattr(key, field):
                    continue
                truth = getattr(key, field)
                truth = truth.value if hasattr(truth, "value") else str(truth)
                kind = FIELD_KINDS.get(field)
                scored += 1
                correct += same_value(str(sig.evidence["judge_choice"]), truth, kind)
                extraction_wrong = not same_value(str(sig.evidence.get("chosen") or ""), truth, kind)
                really_wrong += extraction_wrong
                if not sig.passed:
                    flagged += 1
                    flagged_right += extraction_wrong
                    caught += extraction_wrong
    return {
        "verdicts": verdicts,
        "discarded_unquoted": discarded,
        "scored": scored,
        "accuracy": _rate(correct, scored),
        "precision": _rate(flagged_right, flagged),
        "recall": _rate(caught, really_wrong),
        "by_provider": dict(providers),
    }


def calibration(diagnoses: list[Diagnosis], traces: dict[str, Trace], truths: dict[str, TruthLabel]) -> dict[str, Any]:
    # 1. The model's self-reported 1-5 confidence vs whether that step actually failed.
    buckets = {c: {"confidence": c, "n": 0, "failed": 0} for c in range(1, 6)}
    used_truth = used_diag = 0
    for d in diagnoses:
        trace = traces.get(d.trace_id)
        if trace is None:
            continue
        truth = truths.get(d.trace_id)
        candidate_steps = {c.step_name for c in d.candidates}
        for span in trace.spans:
            if span.self_confidence not in buckets or span.spliced:
                continue
            if truth is not None:
                failed = span.step_name in truth.failed_steps
                used_truth += 1
            else:
                failed = span.step_name in candidate_steps
                used_diag += 1
            b = buckets[span.self_confidence]
            b["n"] += 1
            b["failed"] += failed
    self_report = [{**b, "failure_rate": _rate(b["failed"], b["n"])} for b in buckets.values()]

    # 2. The diagnoser's own confidence vs whether its root step was right.
    bins = []
    for lo, hi in CONFIDENCE_BINS:
        members = [d for d in diagnoses if d.is_failure and lo <= d.confidence < hi and d.trace_id in truths]
        right = sum(1 for d in members if truths[d.trace_id].is_failure and truths[d.trace_id].root_step == d.root_step)
        bins.append(
            {
                "range": f"{lo:.2f}-{min(hi, 1.0):.2f}",
                "n": len(members),
                "mean_confidence": round(mean(d.confidence for d in members), 3) if members else None,
                "root_correct": right,
                "accuracy": _rate(right, len(members)),
            }
        )
    basis = "ground truth" if used_truth and not used_diag else "diagnosis" if used_diag and not used_truth else "mixed"
    return {"self_report": self_report, "self_report_basis": basis, "diagnosis_confidence": bins}


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def compute(store: TraceStore, answer_keys: Optional[dict[str, AnswerKey]] = None, run_id: Optional[str] = None) -> dict[str, Any]:
    answer_keys = answer_keys or {}
    run_id = run_id or default_run_id(store)
    rows = store.list_diagnosis_rows(run_id) if run_id else []
    diagnoses = [Diagnosis.model_validate_json(r["body"]) for r in rows]
    diagnoses.sort(key=lambda d: d.doc_id)
    traces = {d.trace_id: t for d in diagnoses if (t := store.load_trace(d.trace_id)) is not None}
    truths = {
        tid: label(trace, answer_keys[trace.doc_id]) for tid, trace in traces.items() if trace.doc_id in answer_keys
    }
    failures = [d for d in diagnoses if d.is_failure]
    run_trace_ids = {d.trace_id for d in diagnoses}
    flags = [f for f in store.list_flags() if f["trace_id"] in run_trace_ids]
    matrix = regression_matrix(store)
    latest_run = matrix["runs"][-1] if matrix["runs"] else None

    return {
        "run_id": run_id,
        "overview": {
            "traces": len(store.list_traces(run_id=run_id)) if run_id else 0,
            "diagnosed": len(diagnoses),
            "failures": len(failures),
            "failure_rate": _rate(len(failures), len(diagnoses)),
        },
        "category_mix": category_mix(failures),
        "human_category_mix": [
            {"category": c, "label": CATEGORY_LABELS[FailureCategory(c)], "count": n}
            for c, n in Counter(
                f["human_category"]
                for f in {f["trace_id"]: f for f in reversed(flags)}.values()
                if f["verdict"] in ("confirm", "override") and f.get("human_category")
            ).most_common()
        ],
        "per_step": per_step(diagnoses),
        "mttr": mttr(failures, store.list_baselines()),
        "versus_truth": versus_truth(diagnoses, truths),
        "versus_human": versus_human(diagnoses, flags),
        "judge": judge_accuracy(list(traces.values()), answer_keys),
        "calibration": calibration(diagnoses, traces, truths),
        "evals": {
            "cases": len(matrix["cases"]),
            "status": matrix["summary"],
            "runs": len(matrix["runs"]),
            "latest_pass_rate": _rate(latest_run["passed"], latest_run["total"]) if latest_run else None,
        },
    }
