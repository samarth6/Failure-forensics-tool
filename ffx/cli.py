"""Command-line entry point: ``python -m ffx <command>``. Every command is documented in COMMANDS.md."""

from __future__ import annotations

import sys
from datetime import datetime
from typing import List, Optional

import typer

from ffx.config import get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Failure Forensics Tool")
corpus_app = typer.Typer(no_args_is_help=True, help="Synthetic corpus commands")
app.add_typer(corpus_app, name="corpus")
eval_app = typer.Typer(no_args_is_help=True, help="Eval cases from confirmed flags, and regression runs")
app.add_typer(eval_app, name="eval")
analytics_app = typer.Typer(no_args_is_help=True, help="Failure analytics and the MTTR baseline")
app.add_typer(analytics_app, name="analytics")


def _answer_keys() -> dict:
    from ffx.corpus.generate import corpus_index, load_corpus

    try:
        return {doc_id: c.answer_key for doc_id, c in corpus_index(load_corpus()).items()}
    except FileNotFoundError:
        return {}


def _out(msg: str = "") -> None:
    # Windows consoles default to cp1252; never let a stray glyph crash a 40-minute run.
    sys.stdout.write(msg.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8") + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------- doctor


@app.command()
def doctor() -> None:
    """Check that local inference and the judge are reachable."""
    from ffx.llm.ollama_client import OllamaClient

    from ffx.analyzer.judge import cloud_clients
    from ffx.llm.base import LLMError

    settings = get_settings()
    ok, message = OllamaClient(settings).health()
    _out(("OK   " if ok else "FAIL ") + message)

    clients = cloud_clients(settings)
    if not clients:
        _out("INFO no cloud judge keys (FFX_GROQ_API_KEY / FFX_GEMINI_API_KEY) - the judge will use the local model")
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["ok"]}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    for client in clients:
        # A real round-trip: a wrong key or retired model name should fail here, not
        # silently fall through to the next provider in the middle of a diagnosis run.
        try:
            r = client.complete('Reply with JSON {"answer": "ok"}.', schema=schema, max_tokens=1024)
            _out(f"OK   judge {client.provider}: {client.model} answered in {r.latency_ms} ms")
        except LLMError as exc:
            message = str(exc)
            if " 429" in message or " 503" in message:
                # Rate limit / overload: temporary, and exactly what the fallback covers.
                _out(f"WARN judge {client.provider}: {client.model} is busy right now (temporary) - calls will fall through to the next provider")
                continue
            ok = False
            if " 401" in message or " 403" in message:
                _out(f"FAIL judge {client.provider}: key rejected - check FFX_{client.provider.upper()}_API_KEY in .env")
                continue
            _out(f"FAIL judge {client.provider}: {client.model} - {message[:220]}")
            if " 404" in message:
                # The key works; the model name is retired or wrong. Show real options.
                try:
                    models = client.list_models()
                except LLMError as list_exc:
                    _out(f"     (could not list models: {str(list_exc)[:160]})")
                    continue
                _out(f"     key OK. Set FFX_{client.provider.upper()}_MODEL in .env to one of these {len(models)} models:")
                for name in models:
                    _out(f"       {name}")
    _out(f"INFO judge fallback order: {' > '.join([c.provider for c in clients] + ['local'])}")
    raise typer.Exit(0 if ok else 1)


# ---------------------------------------------------------------- corpus


@corpus_app.command("generate")
def corpus_generate(seed: int = typer.Option(7, help="RNG seed; same seed, same corpus.")) -> None:
    """Write the 220-document synthetic benchmark to data/corpus.json."""
    from ffx.corpus.generate import generate_corpus, save_corpus

    docs = generate_corpus(seed=seed)
    path = save_corpus(docs)
    _out(f"wrote {len(docs)} documents to {path}")
    _print_corpus_stats(docs)


@corpus_app.command("stats")
def corpus_stats() -> None:
    """Show condition, expected-root, category and channel distributions."""
    from ffx.corpus.generate import load_corpus

    try:
        docs = load_corpus()
    except FileNotFoundError as exc:
        _out(str(exc))
        raise typer.Exit(1)
    _print_corpus_stats(docs)


def _print_corpus_stats(docs) -> None:
    from collections import Counter
    from statistics import median

    _out(f"documents: {len(docs)}")
    sections = (
        ("failure conditions", Counter(d.failure_mode.value for d in docs)),
        (
            "designed pressure point",
            Counter(
                d.answer_key.expected_failure_step.value
                if d.answer_key.expected_failure_step
                else "none (control)"
                for d in docs
            ),
        ),
        ("designed failure category", Counter(d.answer_key.expected_category.value for d in docs)),
        ("claim category", Counter(d.answer_key.claim_category.value for d in docs)),
        ("severity", Counter(d.answer_key.severity.value for d in docs)),
        ("peril", Counter(d.answer_key.peril.value for d in docs)),
        ("input channel", Counter(d.document.channel.value for d in docs)),
    )
    for title, counts in sections:
        _out(f"\n{title}:")
        for label, count in counts.most_common():
            pct = 100.0 * count / len(docs) if docs else 0.0
            _out(f"  {count:>3}  {pct:>5.1f}%  {label}")

    if docs:
        lengths = sorted(len(d.document.source_text) for d in docs)
        p95_index = min(len(lengths) - 1, int(0.95 * len(lengths)))
        _out("\ndocument length (characters):")
        _out(
            f"  min={lengths[0]}  median={int(median(lengths))}  "
            f"p95={lengths[p95_index]}  max={lengths[-1]}"
        )


# ---------------------------------------------------------------- run


@app.command()
def run(
    limit: Optional[int] = typer.Option(None, help="Only the first N documents."),
    doc: Optional[List[str]] = typer.Option(None, "--doc", help="Specific doc_id(s). Repeatable."),
    mode: Optional[List[str]] = typer.Option(None, "--mode", help="Only these failure modes. Repeatable."),
    notes: str = typer.Option("", help="Free-text note stored on the run."),
    resume: Optional[str] = typer.Option(None, "--resume", help="Continue an interrupted run: skip documents it already traced."),
    pause: float = typer.Option(0.0, help="Seconds to rest after each document that called the model, to let the GPU cool."),
) -> None:
    """Run corpus documents through the traced pipeline."""
    import time

    from ffx.corpus.generate import load_corpus
    from ffx.pipeline.runner import make_runner
    from ffx.tracing.otel import maybe_exporter

    docs = load_corpus()
    if doc:
        docs = [d for d in docs if d.document.doc_id in set(doc)]
    if mode:
        docs = [d for d in docs if d.failure_mode.value in set(mode)]
    if limit:
        docs = docs[:limit]

    runner, store = make_runner(exporter=maybe_exporter())
    if resume:
        if resume not in {r["run_id"] for r in store.list_runs()}:
            _out(f"no run {resume}")
            raise typer.Exit(1)
        done = {row["doc_id"] for row in store.list_traces(run_id=resume)}
        docs = [d for d in docs if d.document.doc_id not in done]
        _out(f"resuming {resume}: {len(done)} document(s) already traced, {len(docs)} to go")
    if not docs:
        _out("no documents selected" if not resume else "nothing left to run")
        raise typer.Exit(0 if resume else 1)

    ok, message = runner.llm.inner.health()
    if not ok:
        _out(f"FAIL {message}")
        raise typer.Exit(1)

    run_id = resume or "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    if not resume:
        store.create_run(run_id, pipeline_version=runner.version, model=runner.llm.model, notes=notes)
    _out(f"{run_id}  pipeline={runner.version}  model={runner.llm.model}  docs={len(docs)}")

    def progress(trace, cdoc, seconds):
        steps = " ".join(f"{s.step_name.value[:4]}:{s.status.value[:2]}" for s in trace.spans)
        cached = sum(1 for s in trace.spans if s.from_cache)
        _out(
            f"  {cdoc.document.doc_id}  {cdoc.failure_mode.value:<22} {trace.status.value:<6} "
            f"{seconds:6.1f}s  {steps}  cache={cached}/{len(trace.spans)}"
        )
        if pause and cached < len(trace.spans):
            time.sleep(pause)

    runner.run_corpus(docs, run_id=run_id, on_trace=progress)
    _out(f"done. cache {runner.llm.stats()}")


# ---------------------------------------------------------------- diagnose


@app.command()
def diagnose(
    run_id: Optional[str] = typer.Option(None, "--run", help="Run to diagnose (default: latest)."),
    doc: Optional[List[str]] = typer.Option(None, "--doc", help="Only these doc_id(s). Repeatable."),
    counterfactual: bool = typer.Option(True, help="Verify root causes by counterfactual re-run."),
    judge: bool = typer.Option(True, help="Use the LLM judge on ambiguous extractions."),
) -> None:
    """Diagnose every trace in a run, and score the verdicts against corpus ground truth."""
    from ffx.analyzer.diagnose import Analyzer
    from ffx.analyzer.judge import build_judge
    from ffx.analyzer.truth import label
    from ffx.corpus.generate import corpus_index, load_corpus
    from ffx.pipeline.runner import make_runner

    runner, store = make_runner()
    run_id = run_id or store.latest_run_id()
    if not run_id:
        _out("no runs yet - run `python -m ffx run` first")
        raise typer.Exit(1)

    try:
        corpus = corpus_index(load_corpus())
    except FileNotFoundError:
        corpus = {}
    keys = {doc_id: c.answer_key for doc_id, c in corpus.items()}

    analyzer = Analyzer(
        store,
        runner=runner if counterfactual else None,
        judge=build_judge() if judge else None,
        answer_keys=keys,
    )
    _out(f"diagnosing {run_id}  judge={analyzer.judge.provider if analyzer.judge else 'off'}  counterfactual={'on' if counterfactual else 'off'}")
    _out(f"  {'doc':<8} {'mode':<22} {'verdict':<34} {'conf':>5} {'cf':<10} {'truth':<34} match")

    tally = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "root_ok": 0, "root_total": 0}

    def on_diagnosis(d, trace):
        mode = trace.metadata.get("failure_mode", "")
        verdict = f"{d.root_step.value}/{d.category.value}" if d.is_failure and d.root_step else "ok"
        cf = d.counterfactual.verdict.value if d.counterfactual else "-"
        truth_txt, match = "-", ""
        if d.doc_id in keys:
            t = label(trace, keys[d.doc_id])
            truth_txt = f"{t.root_step.value}/{t.category.value}" if t.is_failure and t.root_step else "ok"
            if t.is_failure and d.is_failure:
                tally["tp"] += 1
                tally["root_total"] += 1
                if t.root_step == d.root_step:
                    tally["root_ok"] += 1
                    match = "root OK"
                else:
                    match = "ROOT WRONG"
            elif t.is_failure:
                tally["fn"] += 1
                match = "MISSED"
            elif d.is_failure:
                tally["fp"] += 1
                match = "FALSE ALARM"
            else:
                tally["tn"] += 1
                match = "ok"
        _out(f"  {d.doc_id:<8} {mode:<22} {verdict:<34} {d.confidence:>5.2f} {cf:<10} {truth_txt:<34} {match}")

    analyzer.analyze_run(run_id, on_diagnosis=on_diagnosis, only_doc_ids=set(doc) if doc else None)

    if keys:
        tp, fp, fn, tn = tally["tp"], tally["fp"], tally["fn"], tally["tn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        root_acc = tally["root_ok"] / tally["root_total"] if tally["root_total"] else 0.0
        _out("")
        _out(f"detection: tp={tp} fp={fp} fn={fn} tn={tn}  precision={precision:.2f} recall={recall:.2f}")
        _out(f"root-step accuracy on detected true failures: {tally['root_ok']}/{tally['root_total']} = {root_acc:.2f}")
    if analyzer.judge:
        _out(
            f"judge calls: {analyzer.judge.calls}  answered by {analyzer.judge.answered_by or '-'}  "
            f"cache {analyzer.judge.client.stats()}"
        )


@app.command()
def explain(trace_id: str) -> None:
    """Print the full diagnosis and evidence chain for one trace."""
    from ffx.analyzer.diagnose import load_diagnosis
    from ffx.tracing.store import TraceStore

    d = load_diagnosis(TraceStore(), trace_id)
    if d is None:
        _out(f"no diagnosis for {trace_id} - run `python -m ffx diagnose` first")
        raise typer.Exit(1)
    _out(f"{d.trace_id}  doc={d.doc_id}  failure={d.is_failure}  terminal_quality={d.terminal_quality:.2f}  confidence={d.confidence:.2f}")
    _out("")
    _out(d.explanation)
    _out("")
    _out("span assessments:")
    for a in d.assessments:
        _out(
            f"  {a.step_name.value:<15} {a.role.value:<15} q_in={a.q_in:.2f} local={a.local_quality:.2f} "
            f"q_out={a.q_out:.2f} damage={a.local_damage:.2f}  {', '.join(a.failed_signals[:4])}"
        )
    if d.candidates:
        _out("")
        _out("ranked candidates:")
        for c in d.candidates:
            _out(f"  #{c.rank} {c.step_name.value:<15} own_damage={c.own_damage:.2f}  {c.category.value}  ({c.subtype})")
    _out("")
    _out("evidence chain:")
    for link in d.evidence_chain:
        _out(f"  {link.order}. [{link.role}] {link.claim}")
    for cf in d.counterfactuals:
        _out("")
        _out(f"counterfactual @ {cf.splice_step.value}: {cf.verdict.value}  (correction from {cf.correction_source or '-'})")
        if cf.fidelity_before is not None:
            _out(f"  terminal fidelity {cf.fidelity_before:.2f} -> {cf.fidelity_after if cf.fidelity_after is not None else '-'}")
        _out("  before -> after   (* = a fact this correction predicts it will fix; only these decide the verdict)")
        for before in cf.checks_before:
            after = next((c for c in cf.checks_after if c["check"] == before["check"]), None)
            star = "*" if before["check"] in cf.implicated_checks else " "
            _out(f"   {star} {'ok ' if before['passed'] else '!! '}-> {('ok ' if after['passed'] else '!! ') if after else '   '} {before['check']}")
        _out(f"  {cf.note}")
    _out(f"\ndiagnosed in {d.diagnose_ms} ms  judge={d.judge_provider or 'off'} ({d.judge_calls} calls)")


# ---------------------------------------------------------------- show


@app.command()
def show(trace_id: str) -> None:
    """Print one trace, step by step."""
    from ffx.tracing.store import TraceStore

    trace = TraceStore().load_trace(trace_id)
    if trace is None:
        _out(f"no trace {trace_id}")
        raise typer.Exit(1)
    _out(f"{trace.trace_id}  doc={trace.doc_id}  status={trace.status.value}  {trace.duration_ms} ms")
    for span in trace.spans:
        _out(
            f"\n[{span.step_index}] {span.step_name.value}  status={span.status.value}  "
            f"{span.duration_ms} ms  tokens={span.tokens_in}/{span.tokens_out}  "
            f"attempts={span.attempt_count}  confidence={span.self_confidence}"
        )
        if span.error:
            _out(f"    ERROR {span.error.type}: {span.error.message[:300]}")
        if span.output_payload:
            import json

            _out("    " + json.dumps(span.output_payload, ensure_ascii=False)[:600])
        for sig in span.signals:
            mark = "ok " if sig.passed else "!! "
            _out(f"    {mark}{sig.name} ({sig.kind.value}, {sig.severity.value}) {sig.detail[:160]}")


# ---------------------------------------------------------------- eval


@eval_app.command("generate")
def eval_generate() -> None:
    """(Re)build eval cases from every flagged trace's latest human verdict."""
    from ffx.evals.generate import sync_cases
    from ffx.tracing.store import TraceStore

    report = sync_cases(TraceStore(), _answer_keys())
    if not report:
        _out("no flags yet - review a trace in the explorer (Confirm / Override) first")
        return
    for r in report:
        _out(f"  {r.get('doc_id', '-'):<8} {r['action']:<8} {r.get('case_id') or '-':<16} {r['reason']}")


@eval_app.command("run")
def eval_run(
    label: str = typer.Option("", help="What changed, e.g. 'intake prompt keeps signature'."),
    model: Optional[str] = typer.Option(None, help="Evaluate against another local model."),
) -> None:
    """Re-run every eval case's step with the current prompts and model."""
    from ffx.evals.generate import load_cases
    from ffx.evals.runner import make_eval_runner
    from ffx.tracing.store import TraceStore

    store = TraceStore()
    cases = load_cases(store)
    if not cases:
        _out("no eval cases - run `python -m ffx eval generate` after confirming a flag")
        raise typer.Exit(1)
    runner = make_eval_runner(store, model=model)
    ok, message = runner.llm.inner.health()
    if not ok:
        _out(f"FAIL {message}")
        raise typer.Exit(1)
    _out(f"running {len(cases)} case(s)  model={runner.llm.model}  pipeline={runner.pipeline.version}")

    def on_result(case, result):
        detail = result["detail"]
        failed = [a["description"] for a in detail["assertions"] if not a["passed"]]
        cache = "cache" if detail["from_cache"] else f"{detail['duration_ms']} ms"
        _out(f"  {detail['outcome'].upper():<5} {case.doc_id:<8} {case.step_name.value:<15} {cache:<9} {'; '.join(failed)[:110]}")

    summary = runner.run(label=label, cases=cases, on_result=on_result)
    _out(f"{summary['eval_run_id']}: {summary['passed']}/{summary['total']} passed, {summary['errors']} infrastructure error(s)")


@eval_app.command("matrix")
def eval_matrix() -> None:
    """Show each case's pass/fail history across eval runs."""
    from ffx.evals.runner import regression_matrix
    from ffx.tracing.store import TraceStore

    m = regression_matrix(TraceStore())
    if not m["cases"]:
        _out("no eval cases yet")
        return
    runs = m["runs"]
    mark = {"pass": " ok ", "fail": "FAIL", "error": "err "}
    _out("runs: " + "  ".join(f"[{i}] {r['eval_run_id']} {r['pipeline_version'] or ''} {r['label'] or ''}".rstrip() for i, r in enumerate(runs)))
    _out(f"  {'doc':<8} {'step':<15} " + " ".join(f"[{i}] " for i in range(len(runs))) + " status")
    for row in m["cases"]:
        cells = " ".join(mark[row["cells"][r["eval_run_id"]]["outcome"]] if r["eval_run_id"] in row["cells"] else "  - " for r in runs)
        seen = f"  (confirmed in {row['seen_in']} traces)" if row["seen_in"] > 1 else ""
        _out(f"  {row['doc_id']:<8} {row['step_name']:<15} {cells}  {row['status']}{seen}")
    _out("summary: " + ", ".join(f"{k}={v}" for k, v in sorted(m["summary"].items())))


@eval_app.command("show")
def eval_show(case_id: str) -> None:
    """Print one eval case: its input, expected output and assertions."""
    import json

    from ffx.evals.cases import EvalCase
    from ffx.tracing.store import TraceStore

    body = TraceStore().load_eval_case_json(case_id)
    if body is None:
        _out(f"no case {case_id}")
        raise typer.Exit(1)
    case = EvalCase.model_validate_json(body)
    _out(f"{case.case_id}  doc={case.doc_id}  step={case.step_name.value}  category={case.category.value}  from {case.verdict} flag")
    _out(f"source trace {case.source_trace_id}  pipeline {case.pipeline_version}  model {case.model}")
    _out(f"expected output from: {case.expected_source or 'none (checks only)'}")
    _out("assertions:")
    for a in case.assertions:
        _out(f"  - {a.description}")
    _out("\nbad output:\n  " + json.dumps(case.bad_output, ensure_ascii=False)[:600])
    if case.expected_output:
        _out("\nexpected output:\n  " + json.dumps(case.expected_output, ensure_ascii=False)[:600])


# ---------------------------------------------------------------- analytics


@analytics_app.command("report")
def analytics_report(run_id: Optional[str] = typer.Option(None, "--run", help="Run to analyse (default: newest diagnosed).")) -> None:
    """Category mix, per-step rates, MTTR, accuracy, judge quality and calibration."""
    from ffx.evals.analytics import compute
    from ffx.tracing.store import TraceStore

    a = compute(TraceStore(), _answer_keys(), run_id)
    if not a["run_id"]:
        _out("no diagnosed runs yet - run `python -m ffx diagnose` first")
        raise typer.Exit(1)
    pct = lambda v: "-" if v is None else f"{100 * v:.0f}%"  # noqa: E731
    o = a["overview"]
    _out(f"run {a['run_id']}: {o['diagnosed']}/{o['traces']} diagnosed, {o['failures']} failures ({pct(o['failure_rate'])})")

    _out("\ncategory mix (diagnosed failures):")
    for c in a["category_mix"]:
        _out(f"  {c['count']:>3}  {pct(c['share']):>4}  {c['label']}")
    _out("\nper step:            root  own-fail  propagated")
    for s in a["per_step"]:
        _out(f"  {s['step']:<16} {s['root_cause']:>6} {s['failed_own_checks']:>9} {s['propagation']:>11}")

    m = a["mttr"]
    auto = m["automated"]
    _out(f"\ntime to root cause (automated, {auto['n']} failures): mean {auto['mean_ms']} ms, median {auto['median_ms']} ms, p95 {auto['p95_ms']} ms")
    if m["manual"]["n"]:
        _out(f"manual baseline: {m['manual']['mean_minutes']} min over {m['manual']['n']} timing(s)  ->  {m['speedup']}x faster")
    else:
        _out(f"manual baseline: none. {m['note']}")

    t = a["versus_truth"]
    if t:
        _out(
            f"\nvs ground truth ({t['n']} traces): precision {pct(t['precision'])}, recall {pct(t['recall'])}, "
            f"root-step accuracy {pct(t['root_accuracy'])}, category accuracy {pct(t['category_accuracy'])}"
        )
    h = a["versus_human"]
    _out(
        f"vs human review ({h['reviewed']} reviewed): agreement {pct(h['agreement'])}  "
        f"wrong step {h['wrong_step']}, wrong category {h['wrong_category']}, false alarm {h['false_alarm']}, missed {h['missed']}"
    )
    j = a["judge"]
    _out(
        f"judge: {j['verdicts']} verdicts ({j['discarded_unquoted']} discarded), accuracy {pct(j['accuracy'])}, "
        f"precision {pct(j['precision'])}, recall {pct(j['recall'])}  by provider {j['by_provider']}"
    )

    cal = a["calibration"]
    _out(f"\nself-reported confidence vs step actually failed (basis: {cal['self_report_basis']}):")
    for b in cal["self_report"]:
        _out(f"  {b['confidence']}/5  n={b['n']:>3}  failed {pct(b['failure_rate'])}")
    _out("diagnosis confidence vs root step correct (ground truth):")
    for b in cal["diagnosis_confidence"]:
        _out(f"  {b['range']}  n={b['n']:>3}  accurate {pct(b['accuracy'])}")
    e = a["evals"]
    _out(f"\neval cases: {e['cases']}  statuses {e['status']}  latest run pass rate {pct(e['latest_pass_rate'])}")


@analytics_app.command("baseline")
def analytics_baseline(
    minutes: float = typer.Argument(..., help="How long root-causing the trace took you by hand."),
    trace: Optional[str] = typer.Option(None, help="The trace you debugged."),
    note: str = typer.Option("", help="How you did it, e.g. 'read JSON trace files, no explorer'."),
) -> None:
    """Record one manual debugging timing - the honest denominator for the MTTR claim."""
    from ffx.models import utcnow
    from ffx.tracing.span import new_id
    from ffx.tracing.store import TraceStore

    TraceStore().add_baseline(
        {"baseline_id": new_id("base"), "trace_id": trace, "minutes": minutes, "note": note, "created_at": utcnow().isoformat()}
    )
    _out(f"recorded {minutes} min manual baseline")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Interface to bind. Keep local unless you mean to share it."),
    port: int = typer.Option(8000),
) -> None:
    """Start the API and trace explorer (open http://127.0.0.1:8000)."""
    import uvicorn

    from ffx.api import WEB_DIST

    if not WEB_DIST.exists():
        _out("INFO frontend not built yet - API only. Build it with: cd web && npm install && npm run build")
    _out(f"explorer at http://{host}:{port}   API docs at http://{host}:{port}/docs")
    uvicorn.run("ffx.api:app", host=host, port=port, log_level="warning")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
