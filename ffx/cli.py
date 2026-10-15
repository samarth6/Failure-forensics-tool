"""Command-line entry point: ``python -m ffx <command>``."""

from __future__ import annotations

import sys
from datetime import datetime
from typing import List, Optional

import typer

from ffx.config import get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Failure Forensics Tool")
corpus_app = typer.Typer(no_args_is_help=True, help="Synthetic corpus commands")
app.add_typer(corpus_app, name="corpus")


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
) -> None:
    """Run corpus documents through the traced pipeline."""
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
    if not docs:
        _out("no documents selected")
        raise typer.Exit(1)

    runner, store = make_runner(exporter=maybe_exporter())
    ok, message = runner.llm.inner.health()
    if not ok:
        _out(f"FAIL {message}")
        raise typer.Exit(1)

    run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    store.create_run(run_id, pipeline_version=runner.version, model=runner.llm.model, notes=notes)
    _out(f"{run_id}  pipeline={runner.version}  model={runner.llm.model}  docs={len(docs)}")

    def progress(trace, cdoc, seconds):
        steps = " ".join(f"{s.step_name.value[:4]}:{s.status.value[:2]}" for s in trace.spans)
        cached = sum(1 for s in trace.spans if s.from_cache)
        _out(
            f"  {cdoc.document.doc_id}  {cdoc.failure_mode.value:<22} {trace.status.value:<6} "
            f"{seconds:6.1f}s  {steps}  cache={cached}/{len(trace.spans)}"
        )

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
