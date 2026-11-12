import { useCallback, useEffect, useState } from "react";
import { api, type Meta, type Run, type StepName, type TraceDetail, type TraceRow } from "./api";
import { Counterfactuals, EvidenceChain } from "./components/Evidence";
import FlagPanel from "./components/FlagPanel";
import StepPanel from "./components/StepPanel";
import TraceGraph, { Legend } from "./components/TraceGraph";
import TraceList from "./components/TraceList";
import EvalsView from "./components/EvalsView";
import AnalyticsView from "./components/AnalyticsView";
import { MATCH_LABEL, humanize, pct } from "./format";

// The URL hash holds either a page ("#/evals", "#/analytics") or a selected trace
// ("#trace_..."), so a link opens straight to it and the back button works.
type View = "traces" | "evals" | "analytics";
const hashValue = () => decodeURIComponent(window.location.hash.slice(1));
const viewFromHash = (): View => (hashValue() === "/evals" ? "evals" : hashValue() === "/analytics" ? "analytics" : "traces");
const traceFromHash = () => (hashValue() && !hashValue().startsWith("/") ? hashValue() : null);
const VIEWS: [View, string][] = [
  ["traces", "Traces"],
  ["evals", "Evals"],
  ["analytics", "Analytics"],
];

function Summary({ rows }: { rows: TraceRow[] }) {
  const judged = rows.filter((r) => r.match != null);
  const count = (m: string) => judged.filter((r) => r.match === m).length;
  const ok = count("root_ok");
  const wrong = count("root_wrong");
  const diagnosedFailures = rows.filter((r) => r.is_failure === 1).length;
  return (
    <>
      <span className="stat"><b>{rows.length}</b> traces</span>
      <span className="stat"><b>{diagnosedFailures}</b> diagnosed failures</span>
      {judged.length > 0 && (
        <>
          <span className="stat" title="Root step matches corpus ground truth, among detected true failures">
            root accuracy <b>{ok + wrong ? pct(ok / (ok + wrong)) : "–"}</b> ({ok}/{ok + wrong})
          </span>
          <span className="stat"><b>{count("missed")}</b> missed</span>
          <span className="stat"><b>{count("false_alarm")}</b> false alarms</span>
        </>
      )}
    </>
  );
}

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [runId, setRunId] = useState<string | undefined>();
  const [rows, setRows] = useState<TraceRow[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(traceFromHash());
  const [detail, setDetail] = useState<TraceDetail | null>(null);
  const [step, setStep] = useState<StepName>("intake");
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<View>(viewFromHash());

  useEffect(() => {
    Promise.all([api.meta(), api.runs()])
      .then(([m, r]) => {
        setMeta(m);
        setRuns(r);
        setRunId((current) => current ?? r[0]?.run_id);
      })
      .catch((e: Error) => setError(`Cannot reach the API (${e.message}). Is \`python -m ffx serve\` running?`));
  }, []);

  const loadRows = useCallback(() => {
    if (!runId) return;
    api.traces(runId).then(setRows, (e: Error) => setError(e.message));
  }, [runId]);
  useEffect(loadRows, [loadRows]);

  const loadDetail = useCallback(
    (keepStep = false) => {
      if (!selectedId) return;
      api.trace(selectedId).then(
        (d) => {
          setDetail(d);
          // Open on the root cause - that is the step the reader came to see.
          if (!keepStep) setStep(d.diagnosis?.root_step ?? "intake");
        },
        (e: Error) => setError(e.message),
      );
    },
    [selectedId],
  );
  useEffect(() => loadDetail(), [loadDetail]);

  useEffect(() => {
    const onHash = () => {
      setView(viewFromHash());
      const id = traceFromHash();
      if (id) setSelectedId(id); // leaving for another page keeps the selection
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const select = (id: string) => {
    window.location.hash = id;
    setSelectedId(id);
    setView("traces");
  };

  const go = (v: View) => {
    window.location.hash = v === "traces" ? selectedId ?? "" : `/${v}`;
    setView(v);
  };

  const refresh = () => {
    loadRows();
    loadDetail(true);
  };

  const diagnosis = detail?.diagnosis ?? null;
  const span = detail?.trace.spans.find((s) => s.step_name === step);
  const failed = diagnosis?.is_failure ?? false;
  const rootCf = diagnosis?.counterfactuals.find((c) => c.splice_step === diagnosis.root_step);

  return (
    <div className="app">
      <header className="topbar">
        <h1>Failure Forensics</h1>
        <nav className="nav">
          {VIEWS.map(([v, label]) => (
            <button key={v} className={view === v ? "on" : ""} onClick={() => go(v)}>
              {label}
            </button>
          ))}
        </nav>
        <select value={runId ?? ""} onChange={(e) => setRunId(e.target.value)} disabled={!runs.length}>
          {runs.length === 0 && <option value="">no runs yet</option>}
          {runs.map((r) => (
            <option key={r.run_id} value={r.run_id}>
              {r.run_id} ({r.trace_count}){r.notes ? ` - ${r.notes}` : ""}
            </option>
          ))}
        </select>
        {view === "traces" && <Summary rows={rows} />}
        <span className="spacer" />
        {meta && (
          <span className="stat">
            pipeline <b>{meta.pipeline_model}</b> · judge {meta.judge_order.join(" → ")}
          </span>
        )}
      </header>

      {view === "evals" && (
        <div className="full">
          <EvalsView onOpenTrace={select} />
        </div>
      )}
      {view === "analytics" && (
        <div className="full">
          <AnalyticsView runId={runId} />
        </div>
      )}
      <div className="body" hidden={view !== "traces"}>
        <aside className="sidebar">
          {runs.length === 0 && !error ? (
            <div className="empty">
              No runs yet. Generate traces with <code>python -m ffx run</code>, then <code>python -m ffx diagnose</code>.
            </div>
          ) : (
            <TraceList rows={rows} selectedId={selectedId} onSelect={select} />
          )}
        </aside>

        <main className="main">
          {error && <div className="error-box">{error}</div>}
          {!detail && !error && <div className="empty">Select a trace to inspect it.</div>}

          {detail && (
            <>
              <section className={`banner ${diagnosis ? (failed ? "failure" : "healthy") : ""}`}>
                <div className="head">
                  <span className="title">
                    {!diagnosis
                      ? "Not diagnosed yet"
                      : failed
                        ? `Root cause: ${diagnosis.root_step} · ${meta?.categories[diagnosis.category]?.label ?? humanize(diagnosis.category)}`
                        : "No failure detected"}
                  </span>
                  {diagnosis && <span className="badge role-not_run">confidence {pct(diagnosis.confidence)}</span>}
                  {rootCf && <span className={`badge verdict-${rootCf.verdict}`}>counterfactual {humanize(rootCf.verdict)}</span>}
                  {detail.trace.status === "error" && <span className="badge role-root_cause">pipeline stopped</span>}
                </div>
                <div className="muted small mono">
                  {detail.trace.doc_id} · {detail.trace.trace_id} · {humanize(detail.failure_mode)}
                </div>
                {diagnosis ? (
                  <p>{diagnosis.explanation}</p>
                ) : (
                  <p className="muted">Run <code>python -m ffx diagnose</code> to analyse this trace.</p>
                )}
                <div className="meta">
                  {diagnosis && <span>final quality {pct(diagnosis.terminal_quality)}</span>}
                  {detail.truth && (
                    <span>
                      ground truth:{" "}
                      <b>{detail.truth.is_failure ? `${detail.truth.root_step} · ${humanize(detail.truth.category)}` : "no failure"}</b>
                      {(() => {
                        const row = rows.find((r) => r.trace_id === detail.trace.trace_id);
                        return row?.match ? <span className={`match-${row.match}`}> ({MATCH_LABEL[row.match]})</span> : null;
                      })()}
                    </span>
                  )}
                  {diagnosis && <span>diagnosed in {diagnosis.diagnose_ms} ms</span>}
                  {diagnosis?.judge_provider && <span>judge {diagnosis.judge_provider} ({diagnosis.judge_calls} calls)</span>}
                </div>
                {detail.trace.metadata.source_text && (
                  <details style={{ marginTop: 10 }}>
                    <summary className="small">Source document</summary>
                    <pre style={{ marginTop: 8 }}>{detail.trace.metadata.source_text}</pre>
                  </details>
                )}
              </section>

              <TraceGraph trace={detail.trace} diagnosis={diagnosis} selected={step} onSelect={setStep} />
              <Legend />

              <StepPanel key={`${detail.trace.trace_id}-${step}`} traceId={detail.trace.trace_id} span={span} step={step} />

              {diagnosis && (
                <div className="grid-2">
                  <div className="panel">
                    <h2>Evidence chain</h2>
                    <div className="content"><EvidenceChain diagnosis={diagnosis} /></div>
                  </div>
                  <div className="panel">
                    <h2>Counterfactual verification</h2>
                    <div className="content"><Counterfactuals diagnosis={diagnosis} /></div>
                  </div>
                </div>
              )}

              <FlagPanel
                key={detail.trace.trace_id}
                trace={detail.trace}
                diagnosis={diagnosis}
                flags={detail.flags}
                meta={meta}
                onChanged={refresh}
              />
            </>
          )}
        </main>
      </div>
    </div>
  );
}
