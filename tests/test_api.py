"""The explorer API, tested against a temporary store - no model, no real data."""

import pytest
from fastapi.testclient import TestClient

import ffx.api as api
from ffx.analyzer.diagnose import Analyzer
from ffx.models import StepName
from ffx.tracing.store import TraceStore
from tests.test_diagnose import build_trace


@pytest.fixture
def client(tmp_path, monkeypatch):
    store = TraceStore(db_path=tmp_path / "t.db", trace_dir=tmp_path / "traces")
    monkeypatch.setattr(api, "store", lambda: store)
    monkeypatch.setattr(api, "corpus", lambda: {})
    api._truth_cache.clear()

    store.create_run("run_test", pipeline_version="pv_test", model="fake")
    failing = build_trace(fields={"policy_number": "POL-12345"})  # fabricated value
    healthy = build_trace()
    for trace, doc_id in ((failing, "doc_bad"), (healthy, "doc_ok")):
        trace.doc_id = doc_id
        trace.run_id = "run_test"
        store.save_trace(trace)
        Analyzer(store, runner=None, judge=None).analyze(trace)

    c = TestClient(api.app)
    c.failing_id, c.healthy_id = failing.trace_id, healthy.trace_id
    return c


def test_trace_list_includes_diagnosis(client):
    rows = {r["doc_id"]: r for r in client.get("/api/traces?run_id=run_test").json()}
    assert rows["doc_bad"]["is_failure"] == 1
    assert rows["doc_bad"]["root_step"] == "extraction"
    assert rows["doc_ok"]["is_failure"] == 0


def test_trace_detail_bundles_trace_and_diagnosis(client):
    body = client.get(f"/api/traces/{client.failing_id}").json()
    assert len(body["trace"]["spans"]) == 4
    assert body["diagnosis"]["category"] == "extraction_hallucination"
    assert [a["role"] for a in body["diagnosis"]["assessments"]] == [
        "healthy", "root_cause", "propagation", "propagation",
    ]


def test_unknown_trace_is_404(client):
    assert client.get("/api/traces/trace_nope").status_code == 404


def test_diff_proposes_the_rule_based_fix(client):
    # No answer key in this fixture, so the correction comes from the failed signal's
    # own proposal: drop the fabricated value.
    body = client.get(f"/api/traces/{client.failing_id}/diff/extraction").json()
    assert body["produced"]["fields"]["policy_number"] == "POL-12345"
    assert body["should_have_produced"]["fields"]["policy_number"] == ""
    assert body["source"] == "rule"
    assert "_meta" not in body["received"]


def test_confirm_flag_inherits_the_diagnosis(client):
    flag = client.post(f"/api/traces/{client.failing_id}/flags", json={"verdict": "confirm"}).json()
    assert flag["step_name"] == "extraction" and flag["human_category"] == "extraction_hallucination"
    assert client.get(f"/api/traces/{client.failing_id}").json()["flags"][0]["verdict"] == "confirm"


def test_override_needs_step_and_category(client):
    r = client.post(f"/api/traces/{client.failing_id}/flags", json={"verdict": "override", "step_name": "intake"})
    assert r.status_code == 400


def test_cannot_confirm_a_healthy_trace(client):
    assert client.post(f"/api/traces/{client.healthy_id}/flags", json={"verdict": "confirm"}).status_code == 400


def test_human_correction_becomes_the_reference(client):
    corrected = {"fields": {"policy_number": "POL-44821"}}
    r = client.post(
        f"/api/traces/{client.failing_id}/flags",
        json={"verdict": "confirm", "step_name": StepName.EXTRACTION.value, "corrected_output": corrected},
    )
    assert r.status_code == 200
    diff = client.get(f"/api/traces/{client.failing_id}/diff/extraction").json()
    assert diff["source"] == "human"
    assert diff["should_have_produced"]["fields"]["policy_number"] == "POL-44821"


def test_invalid_human_correction_is_rejected(client):
    # Must not be silently replaced by a fallback correction and reported as accepted.
    r = client.post(
        f"/api/traces/{client.failing_id}/flags",
        json={"verdict": "confirm", "step_name": "classification", "corrected_output": {"claim_category": "spaceship"}},
    )
    assert r.status_code == 422


def test_meta_lists_steps_and_categories(client):
    body = client.get("/api/meta").json()
    assert body["steps"] == ["intake", "extraction", "classification", "summarization"]
    assert body["categories"]["context_loss"]["label"] == "Context Loss"
