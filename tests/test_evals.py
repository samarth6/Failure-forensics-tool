"""Phase 5: confirmed flags become eval cases; regression runs track them over time.

No model and no network: the "model" in the runner tests is a fake that returns a
scripted output, which is exactly what a replay-cached step looks like to the runner.
"""

import json
import sqlite3

import pytest

from ffx.analyzer.diagnose import Analyzer
from ffx.evals.analytics import compute
from ffx.evals.cases import AssertionKind, check_all
from ffx.evals.generate import case_from_flag, load_cases, sync_trace
from ffx.evals.runner import EvalRunner, case_status, regression_matrix
from ffx.llm.base import LLMError, LLMResponse
from ffx.models import AnswerKey, StepName
from ffx.tracing.store import TraceStore
from tests.test_diagnose import GOOD_FIELDS, SOURCE, _evidence, build_trace


@pytest.fixture
def store(tmp_path):
    return TraceStore(db_path=tmp_path / "t.db", trace_dir=tmp_path / "traces")


def flag(trace_id, verdict="confirm", step="extraction", category="extraction_hallucination", **kw):
    return {
        "flag_id": f"flag_{verdict}", "trace_id": trace_id, "verdict": verdict, "step_name": step,
        "human_category": category, "note": "", "created_at": kw.pop("created_at", "2026-11-11T10:00:00"), **kw,
    }


KEY = AnswerKey(
    claimant_name="Maria Okafor", policy_number="POL-44821", claim_number="CLM-7781",
    date_of_loss="2024-03-14", reported_amount="4500.00", loss_location="14 Ash Lane, Bristol",
    peril="fire", claim_category="property", severity="low", routing_queue="fast_track",
)


class FakeLLM:
    """Returns scripted outputs in order; the last one repeats."""

    provider, model = "fake", "fake-model"

    def __init__(self, *outputs):
        self.outputs = list(outputs)

    def complete(self, prompt, *, schema=None, system=None, max_tokens=2048):
        out = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        if isinstance(out, Exception):
            raise out
        return LLMResponse(text=json.dumps(out), model=self.model, provider=self.provider)


def extraction_output(**fields):
    merged = {**GOOD_FIELDS, **fields}
    return {"fields": merged, "evidence": _evidence(merged), "missing_fields": [], "confidence": 5}


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def test_confirmed_hallucination_becomes_a_case_that_catches_it():
    trace = build_trace(fields={"policy_number": "POL-12345"})
    case, reason = case_from_flag(trace, flag(trace.trace_id))
    assert case is not None, reason
    assert case.step_name == StepName.EXTRACTION
    assert "_meta" not in case.step_input
    assert case.expected_source == "rule"
    described = [a.description for a in case.assertions]
    assert "policy_number = 'POL-44821'" in described
    assert any(a.kind == AssertionKind.CHECK_PASSES and a.signal == "value_grounded:policy_number" for a in case.assertions)
    # The two validation rules every saved case obeys.
    assert all(r.passed for r in check_all(case, case.expected_output))
    assert not all(r.passed for r in check_all(case, case.bad_output))


def test_intake_context_loss_case_asserts_the_lost_facts():
    lossy = SOURCE.replace("My policy number is POL-44821. ", "")
    trace = build_trace(normalized=lossy, fields={"policy_number": ""})
    case, reason = case_from_flag(trace, flag(trace.trace_id, step="intake", category="context_loss"), answer_key=KEY)
    assert case is not None, reason
    assert case.expected_source == "answer_key"
    contains = [a for a in case.assertions if a.kind == AssertionKind.CONTAINS]
    assert [a.expected for a in contains] == ["POL-44821"]


def test_not_a_failure_and_infrastructure_make_no_case():
    trace = build_trace(fields={"policy_number": "POL-12345"})
    assert case_from_flag(trace, flag(trace.trace_id, verdict="not_a_failure"))[0] is None
    assert case_from_flag(trace, flag(trace.trace_id, category="infrastructure_error"))[0] is None


def test_case_that_cannot_catch_the_failure_is_not_saved():
    # A "confirmed" failure on a trace whose extraction is actually correct: nothing
    # separates the output from a correct one, so a case would pass from day one.
    trace = build_trace()
    case, reason = case_from_flag(trace, flag(trace.trace_id))
    assert case is None
    assert "no assertion separates" in reason


def test_changing_the_verdict_retires_the_case(store):
    trace = build_trace(fields={"policy_number": "POL-12345"})
    store.save_trace(trace)
    store.save_flag(flag(trace.trace_id))
    assert sync_trace(store, trace.trace_id)["action"] == "created"
    store.save_flag(flag(trace.trace_id, verdict="not_a_failure", created_at="2026-11-11T11:00:00"))
    assert sync_trace(store, trace.trace_id)["action"] == "retired"
    assert load_cases(store) == []


def test_same_failure_in_two_runs_is_one_case(store):
    # Observed: doc_005 confirmed in two corpus runs made two identical cases.
    first = build_trace(fields={"policy_number": "POL-12345"})
    second = build_trace(fields={"policy_number": "POL-12345"})
    for trace in (first, second):
        store.save_trace(trace)
        store.save_flag(flag(trace.trace_id, flag_id=f"flag_{trace.trace_id}"))
    assert sync_trace(store, first.trace_id)["action"] == "created"
    assert sync_trace(store, second.trace_id)["action"] == "merged"
    cases = load_cases(store)
    assert len(cases) == 1 and cases[0].traces == [first.trace_id, second.trace_id]

    # Withdrawing one trace leaves the case alive while the other still confirms it.
    store.save_flag(flag(first.trace_id, verdict="not_a_failure", flag_id="flag_withdraw", created_at="2026-11-11T11:00:00"))
    assert sync_trace(store, first.trace_id)["action"] == "skipped"
    cases = load_cases(store)
    assert len(cases) == 1 and cases[0].traces == [second.trace_id]
    assert cases[0].source_trace_id == second.trace_id


# --------------------------------------------------------------------------
# Regression runs
# --------------------------------------------------------------------------


@pytest.fixture
def case_store(store):
    trace = build_trace(fields={"policy_number": "POL-12345"})
    store.save_trace(trace)
    store.save_flag(flag(trace.trace_id))
    sync_trace(store, trace.trace_id)
    return store


def run_with(store, output, label=""):
    return EvalRunner(store, llm=FakeLLM(output)).run(label=label)


def test_reproducing_the_failure_then_fixing_it(case_store):
    bad = extraction_output(policy_number="POL-12345")
    good = extraction_output()
    assert run_with(case_store, bad)["passed"] == 0
    assert regression_matrix(case_store)["cases"][0]["status"] == "failing"

    summary = run_with(case_store, good, label="prompt fix")
    assert summary["passed"] == 1
    m = regression_matrix(case_store)
    assert m["cases"][0]["status"] == "fixed"
    assert m["runs"][-1]["fixed"] == 1

    run_with(case_store, bad, label="regression")
    m = regression_matrix(case_store)
    assert m["cases"][0]["status"] == "regressed"
    assert m["runs"][-1]["regressed"] == 1


def test_model_server_crash_is_an_error_not_a_regression(case_store):
    run_with(case_store, extraction_output())
    summary = run_with(case_store, LLMError("ollama returned 500: failed to allocate buffer"))
    assert summary["errors"] == 1
    # The crash is skipped: the case is still judged on its last real result.
    assert regression_matrix(case_store)["cases"][0]["status"] == "passing"


def test_unparseable_output_fails_produces_output(case_store):
    result = EvalRunner(case_store, llm=FakeLLM({"nonsense": True})).run_case(load_cases(case_store)[0])
    assert result["detail"]["outcome"] == "fail"
    assert result["detail"]["assertions"][0]["passed"] is False


@pytest.mark.parametrize(
    "history,status",
    [([], "not_run"), (["fail"], "failing"), (["fail", "pass"], "fixed"), (["pass", "fail"], "regressed"),
     (["fail", "pass", "pass"], "passing"), (["pass", "error"], "passing"), (["error"], "error")],
)
def test_case_status(history, status):
    assert case_status(history) == status


# --------------------------------------------------------------------------
# Analytics and storage
# --------------------------------------------------------------------------


def test_analytics_scores_against_truth_and_baseline(store):
    store.create_run("run_a", pipeline_version="pv", model="fake")
    failing = build_trace(fields={"policy_number": "POL-12345"})
    healthy = build_trace()
    for trace, doc in ((failing, "doc_bad"), (healthy, "doc_ok")):
        trace.doc_id, trace.run_id = doc, "run_a"
        store.save_trace(trace)
        Analyzer(store, runner=None, judge=None).analyze(trace)
    store.add_baseline({"baseline_id": "b1", "minutes": 20, "note": "by hand"})

    a = compute(store, {"doc_bad": KEY, "doc_ok": KEY}, "run_a")
    assert a["overview"]["failures"] == 1
    assert a["category_mix"][0]["category"] == "extraction_hallucination"
    assert a["versus_truth"]["tp"] == 1 and a["versus_truth"]["tn"] == 1
    assert a["versus_truth"]["root_accuracy"] == 1.0
    assert a["mttr"]["manual"]["mean_minutes"] == 20 and a["mttr"]["speedup"] is not None
    assert sum(b["n"] for b in a["calibration"]["self_report"]) == 8  # 4 spans x 2 traces


def test_old_database_gains_the_new_column(tmp_path):
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE eval_runs (eval_run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, label TEXT, model TEXT, total INTEGER, passed INTEGER)")
    TraceStore(db_path=db, trace_dir=tmp_path / "traces")
    with sqlite3.connect(db) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(eval_runs)")}
    assert "pipeline_version" in columns
