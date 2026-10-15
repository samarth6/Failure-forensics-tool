"""HTTP API for the trace explorer.

Thin by design: every answer here is computed by code that already exists and is tested
- the store, the analyzer, the correction builder, the ground-truth labeller. The API's
job is to assemble those into the shapes the explorer renders.

Run with ``python -m ffx serve``. In production the built frontend (``web/dist``) is
served from the same process, so the explorer needs no second server - which matters on
a machine with ~1 GiB of free RAM.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ffx.analyzer.counterfactual import build_correction
from ffx.analyzer.diagnose import load_diagnosis
from ffx.analyzer.taxonomy import CATEGORY_DESCRIPTIONS, CATEGORY_LABELS
from ffx.analyzer.truth import label as truth_label
from ffx.config import PROJECT_ROOT, get_settings
from ffx.models import STEP_ORDER, AnswerKey, CorpusDocument, FailureCategory, StepName, utcnow
from ffx.tracing.span import Trace, new_id
from ffx.tracing.store import TraceStore

WEB_DIST = PROJECT_ROOT / "web" / "dist"

app = FastAPI(title="Failure Forensics Tool", version="0.1.0")

# The Vite dev server runs on another port while the frontend is being developed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Shared state
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def store() -> TraceStore:
    return TraceStore()


@lru_cache(maxsize=1)
def corpus() -> dict[str, CorpusDocument]:
    from ffx.corpus.generate import corpus_index, load_corpus

    try:
        return corpus_index(load_corpus())
    except FileNotFoundError:
        return {}


def answer_key(doc_id: str) -> Optional[AnswerKey]:
    doc = corpus().get(doc_id)
    return doc.answer_key if doc else None


def load_trace_or_404(trace_id: str) -> Trace:
    trace = store().load_trace(trace_id)
    if trace is None:
        raise HTTPException(404, f"no trace {trace_id}")
    return trace


# Ground truth needs the full trace, which is a file read and parse. The trace list asks
# for it on every row, so memoise per trace file version (mtime) rather than per request.
_truth_cache: dict[str, tuple[float, Optional[dict[str, Any]]]] = {}


def truth_for(trace_id: str, doc_id: str) -> Optional[dict[str, Any]]:
    key = answer_key(doc_id)
    if key is None:
        return None
    path = store().trace_dir / f"{trace_id}.json"
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    cached = _truth_cache.get(trace_id)
    if cached and cached[0] == mtime:
        return cached[1]
    trace = store().load_trace(trace_id)
    result = truth_label(trace, key).model_dump(mode="json") if trace else None
    _truth_cache[trace_id] = (mtime, result)
    return result


def match_verdict(row: dict[str, Any], truth: Optional[dict[str, Any]]) -> Optional[str]:
    """How the diagnosis compares to ground truth - the same buckets the CLI prints."""
    if truth is None or row.get("is_failure") is None:
        return None
    diagnosed = bool(row["is_failure"])
    if truth["is_failure"] and diagnosed:
        return "root_ok" if truth["root_step"] == row.get("root_step") else "root_wrong"
    if truth["is_failure"]:
        return "missed"
    return "false_alarm" if diagnosed else "ok"


# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    settings = get_settings()
    return {
        "steps": [s.value for s in STEP_ORDER],
        "categories": {
            c.value: {"label": CATEGORY_LABELS[c], "description": CATEGORY_DESCRIPTIONS[c]} for c in FailureCategory
        },
        "pipeline_model": settings.pipeline_model,
        "judge_order": settings.cloud_judges + ["local"],
        "corpus_size": len(corpus()),
    }


@app.get("/api/runs")
def runs() -> list[dict[str, Any]]:
    return store().list_runs()


# --------------------------------------------------------------------------
# Traces
# --------------------------------------------------------------------------


@app.get("/api/traces")
def traces(run_id: Optional[str] = None) -> list[dict[str, Any]]:
    run_id = run_id or store().latest_run_id()
    rows = store().list_traces(run_id=run_id) if run_id else []
    out = []
    for row in rows:
        truth = truth_for(row["trace_id"], row["doc_id"])
        out.append({**row, "truth": truth, "match": match_verdict(row, truth)})
    return out


@app.get("/api/traces/{trace_id}")
def trace_detail(trace_id: str) -> dict[str, Any]:
    trace = load_trace_or_404(trace_id)
    diagnosis = load_diagnosis(store(), trace_id)
    key = answer_key(trace.doc_id)
    cdoc = corpus().get(trace.doc_id)

    counterfactuals = []
    for cf in diagnosis.counterfactuals if diagnosis else []:
        rerun = store().load_trace(cf.counterfactual_trace_id) if cf.counterfactual_trace_id else None
        counterfactuals.append({"result": cf.model_dump(mode="json"), "trace": rerun.model_dump(mode="json") if rerun else None})

    return {
        "trace": trace.model_dump(mode="json"),
        "diagnosis": diagnosis.model_dump(mode="json") if diagnosis else None,
        "counterfactuals": counterfactuals,
        "truth": truth_for(trace_id, trace.doc_id),
        "answer_key": key.model_dump(mode="json") if key else None,
        "failure_mode": cdoc.failure_mode.value if cdoc else trace.metadata.get("failure_mode"),
        "flags": store().list_flags(trace_id),
    }


@app.get("/api/traces/{trace_id}/diff/{step}")
def step_diff(trace_id: str, step: StepName) -> dict[str, Any]:
    """Received vs produced vs should-have-produced, for one step.

    "Should have produced" is the same correction the counterfactual uses, in the same
    priority order: a human's correction, then the corpus answer key, then a fix proposed
    by the judge or a rule. The source is returned so the UI can say where it came from.
    """
    trace = load_trace_or_404(trace_id)
    span = trace.span(step)
    if span is None:
        raise HTTPException(404, f"step {step.value} did not run in this trace")

    human = next(
        (f["corrected_output"] for f in store().list_flags(trace_id) if f.get("step_name") == step.value and f.get("corrected_output")),
        None,
    )
    built = build_correction(span, trace, answer_key=answer_key(trace.doc_id), human_output=human)
    received = {k: v for k, v in span.input_payload.items() if k != "_meta"}
    return {
        "step": step.value,
        "received": received,
        "produced": span.output_payload,
        "should_have_produced": built[0].model_dump(mode="json") if built else None,
        "source": built[1] if built else None,
        "note": None if built else (
            "The final step has nothing downstream; there is no reference output for it."
            if step == StepName.SUMMARIZATION
            else "No correction available: no answer key, human flag, or proposed fix."
        ),
    }


# --------------------------------------------------------------------------
# Human review
# --------------------------------------------------------------------------


class FlagRequest(BaseModel):
    """A human verdict on a diagnosis. Feeds Phase 5: confirmed flags become eval cases.

    * ``confirm``       - the diagnosis is right.
    * ``override``      - it failed, but the root cause is a different step / category.
    * ``not_a_failure`` - the trace was actually fine (a false alarm).
    """

    verdict: str = Field(pattern="^(confirm|override|not_a_failure)$")
    step_name: Optional[StepName] = None
    human_category: Optional[FailureCategory] = None
    corrected_output: Optional[dict[str, Any]] = None
    note: str = ""


@app.post("/api/traces/{trace_id}/flags")
def add_flag(trace_id: str, body: FlagRequest) -> dict[str, Any]:
    trace = load_trace_or_404(trace_id)
    diagnosis = load_diagnosis(store(), trace_id)

    step, category = body.step_name, body.human_category
    if body.verdict == "confirm":
        if diagnosis is None or not diagnosis.is_failure:
            raise HTTPException(400, "nothing to confirm: this trace has no failure diagnosis")
        step = step or diagnosis.root_step
        category = category or diagnosis.category
    elif body.verdict == "override" and (step is None or category is None):
        raise HTTPException(400, "an override needs both step_name and human_category")
    elif body.verdict == "not_a_failure":
        category = FailureCategory.NO_FAILURE

    if body.corrected_output is not None:
        if step is None:
            raise HTTPException(400, "a corrected_output needs the step it corrects")
        span = trace.span(step)
        if span is None:
            raise HTTPException(400, f"step {step.value} did not run in this trace")
        # build_correction falls back to other sources when the human's output does not
        # validate - right for diagnosis, wrong here. Only a "human"-sourced result
        # proves the submitted correction is itself a valid step output.
        built = build_correction(span, trace, human_output=body.corrected_output)
        if built is None or built[1] != "human":
            raise HTTPException(422, f"corrected_output is not a valid {step.value} output")

    flag = {
        "flag_id": new_id("flag"),
        "trace_id": trace_id,
        "step_name": step.value if step else None,
        "verdict": body.verdict,
        "human_category": category.value if category else None,
        "corrected_output": body.corrected_output,
        "note": body.note,
        "created_at": utcnow().isoformat(),
    }
    store().save_flag(flag)
    return flag


@app.get("/api/flags")
def flags() -> list[dict[str, Any]]:
    return store().list_flags()


@app.post("/api/traces/{trace_id}/diagnose")
async def rediagnose(trace_id: str, counterfactual: bool = True, judge: bool = True) -> dict[str, Any]:
    """Re-run diagnosis - e.g. after a human correction, so the counterfactual uses it.

    Can take minutes: counterfactual re-runs call the local model. Runs in a worker
    thread so the server stays responsive.
    """
    trace = load_trace_or_404(trace_id)

    def work() -> dict[str, Any]:
        from ffx.analyzer.diagnose import Analyzer
        from ffx.analyzer.judge import build_judge
        from ffx.pipeline.runner import make_runner

        runner, _ = make_runner(store())
        analyzer = Analyzer(
            store(),
            runner=runner if counterfactual else None,
            judge=build_judge() if judge else None,
            answer_keys={trace.doc_id: answer_key(trace.doc_id)} if answer_key(trace.doc_id) else {},
        )
        return analyzer.analyze(trace).model_dump(mode="json")

    return await run_in_threadpool(work)


# --------------------------------------------------------------------------
# Frontend
# --------------------------------------------------------------------------

if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        # Client-side routing: any non-API path serves the app shell. Unknown API paths
        # must still 404, not quietly return the HTML page with a 200.
        if path.startswith("api/"):
            raise HTTPException(404, f"no endpoint /{path}")
        candidate = WEB_DIST / path
        if path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(WEB_DIST / "index.html")
