"""Regression runs: re-run every eval case's step with the *current* prompts and model.

Each case re-executes one step on its frozen input. The prompt is rebuilt from today's
templates, so:

* unchanged prompts and model -> the replay cache answers, and the case reproduces its
  original failure byte for byte (a free self-check that the case is sound);
* an edited prompt or a different model -> a cache miss, a real model call, and a
  genuine new verdict. That is the "did my fix work, and did it break anything else?"
  question the regression matrix answers.

Runs are recorded per case, so the matrix shows each case's history across runs:
fixed, regressed, still failing, or passing.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Any, Callable, Optional

from pydantic import ValidationError

from ffx.config import Settings, get_settings
from ffx.evals.cases import EvalCase, check_all
from ffx.evals.generate import load_cases
from ffx.pipeline.runner import PipelineRunner, build_pipeline_llm
from ffx.pipeline.steps import STEPS, pipeline_version
from ffx.tracing.store import TraceStore
from ffx.tracing.tracer import StepFailed, Tracer

PASS, FAIL, ERROR = "pass", "fail", "error"


class EvalRunner:
    def __init__(self, store: TraceStore, *, llm: Any = None, settings: Optional[Settings] = None) -> None:
        self.store = store
        self.settings = settings or get_settings()
        self.llm = llm if llm is not None else build_pipeline_llm(self.settings)
        # No store on this tracer: an eval execution is a test, not a pipeline run, and
        # must not show up among the traces being diagnosed.
        self.pipeline = PipelineRunner(self.llm, Tracer(store=None), self.settings)

    def run_case(self, case: EvalCase, *, eval_run_id: str = "") -> dict[str, Any]:
        spec = STEPS[case.step_name]
        try:
            step_input = spec.input_model.model_validate(case.step_input)
        except ValidationError as exc:
            # The step's input schema changed since the case was recorded. The case is
            # stale, not the prompt broken: report it, and keep running the others.
            return {
                "case_id": case.case_id,
                "passed": False,
                "detail": {
                    "outcome": ERROR, "assertions": [], "output": None, "from_cache": False, "attempts": 0,
                    "duration_ms": 0, "error": {"type": "StaleCase", "message": str(exc)[:300]},
                },
            }
        tracer = self.pipeline.tracer
        with tracer.trace(case.doc_id, run_id=eval_run_id, pipeline_version=self.pipeline.version, metadata={"eval_case": case.case_id}) as trace:
            try:
                self.pipeline.execute_step(case.step_name, step_input)
            except StepFailed:
                pass  # recorded on the span; a step that produces nothing fails produces_output
        span = trace.spans[-1]

        if span.error is not None and span.error.type == "LLMTransportError":
            # The model server failed: says nothing about the prompt. Never count it as a regression.
            outcome, assertions = ERROR, []
        else:
            assertions = check_all(case, span.output_payload)
            outcome = PASS if all(a.passed for a in assertions) else FAIL

        return {
            "case_id": case.case_id,
            "passed": outcome == PASS,
            "detail": {
                "outcome": outcome,
                "assertions": [a.model_dump() for a in assertions],
                "output": span.output_payload,
                "error": span.error.model_dump() if span.error else None,
                "from_cache": span.from_cache,
                "attempts": span.attempt_count,
                "duration_ms": span.duration_ms,
            },
        }

    def run(
        self,
        *,
        label: str = "",
        cases: Optional[list[EvalCase]] = None,
        on_result: Optional[Callable[[EvalCase, dict[str, Any]], None]] = None,
    ) -> dict[str, Any]:
        cases = load_cases(self.store) if cases is None else cases
        eval_run_id = "eval_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        results = []
        for case in cases:
            result = self.run_case(case, eval_run_id=eval_run_id)
            results.append(result)
            if on_result:
                on_result(case, result)
        version = self.pipeline.version
        self.store.save_eval_run(eval_run_id, label=label, model=self.llm.model, results=results, pipeline_version=version)
        return {
            "eval_run_id": eval_run_id,
            "label": label,
            "model": self.llm.model,
            "pipeline_version": version,
            "total": len(results),
            "passed": sum(1 for r in results if r["passed"]),
            "errors": sum(1 for r in results if r["detail"]["outcome"] == ERROR),
        }


def make_eval_runner(store: Optional[TraceStore] = None, *, model: Optional[str] = None) -> EvalRunner:
    """``model`` evaluates the cases against a different local model than the pipeline's."""
    settings = get_settings()
    if model:
        settings = dataclasses.replace(settings, pipeline_model=model)
    return EvalRunner(store or TraceStore(), settings=settings)


# --------------------------------------------------------------------------
# Regression matrix
# --------------------------------------------------------------------------


def case_status(history: list[str]) -> str:
    """Where a case stands, from its outcomes in run order.

    Errored runs are skipped: a crashed model server is not evidence about the prompt.
    """
    real = [h for h in history if h != ERROR]
    if not history:
        return "not_run"
    if not real:
        return "error"
    if len(real) == 1:
        return "passing" if real[-1] == PASS else "failing"
    previous, latest = real[-2], real[-1]
    if previous == FAIL and latest == PASS:
        return "fixed"
    if previous == PASS and latest == FAIL:
        return "regressed"
    return "passing" if latest == PASS else "failing"


def regression_matrix(store: TraceStore) -> dict[str, Any]:
    cases = load_cases(store)
    runs = store.list_eval_runs()
    cells: dict[str, dict[str, dict[str, Any]]] = {}
    for row in store.eval_results_matrix():
        detail = row["detail"]
        cells.setdefault(row["case_id"], {})[row["eval_run_id"]] = {
            "outcome": detail.get("outcome", PASS if row["passed"] else FAIL),
            "failed": [a["description"] for a in detail.get("assertions", []) if not a["passed"]],
            "from_cache": detail.get("from_cache"),
        }

    rows = []
    for case in cases:
        by_run = cells.get(case.case_id, {})
        history = [by_run[r["eval_run_id"]]["outcome"] for r in runs if r["eval_run_id"] in by_run]
        rows.append(
            {
                "case_id": case.case_id,
                "doc_id": case.doc_id,
                "source_trace_id": case.source_trace_id,
                "step_name": case.step_name.value,
                "category": case.category.value,
                "expected_source": case.expected_source,
                "assertion_count": len(case.assertions),
                "seen_in": len(case.traces),
                "created_at": case.created_at.isoformat(),
                "cells": by_run,
                "status": case_status(history),
            }
        )

    # Per run: what changed relative to each case's previous run.
    transitions = []
    last: dict[str, str] = {}
    for run in runs:
        fixed = regressed = 0
        for case in cases:
            cell = cells.get(case.case_id, {}).get(run["eval_run_id"])
            if cell is None or cell["outcome"] == ERROR:
                continue
            before = last.get(case.case_id)
            fixed += before == FAIL and cell["outcome"] == PASS
            regressed += before == PASS and cell["outcome"] == FAIL
            last[case.case_id] = cell["outcome"]
        transitions.append({"eval_run_id": run["eval_run_id"], "fixed": fixed, "regressed": regressed})

    summary: dict[str, int] = {}
    for row in rows:
        summary[row["status"]] = summary.get(row["status"], 0) + 1
    return {
        "runs": [{**r, **t} for r, t in zip(runs, transitions)],
        "cases": rows,
        "summary": summary,
        "current_pipeline_version": pipeline_version(),
    }
