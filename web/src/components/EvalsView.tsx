import { useCallback, useEffect, useState } from "react";
import { api, type CaseStatus, type EvalCaseDetail, type EvalMatrix, type Outcome } from "../api";
import { humanize } from "../format";

const STATUS: Record<CaseStatus, { label: string; cls: string; icon: string }> = {
  fixed: { label: "fixed", cls: "role-healthy", icon: "↑" },
  passing: { label: "passing", cls: "role-healthy", icon: "✓" },
  regressed: { label: "regressed", cls: "role-root_cause", icon: "↓" },
  failing: { label: "failing", cls: "role-root_cause", icon: "✗" },
  error: { label: "error", cls: "role-not_run", icon: "!" },
  not_run: { label: "not run", cls: "role-not_run", icon: "·" },
};

const CELL: Record<Outcome, { icon: string; cls: string; text: string }> = {
  pass: { icon: "✓", cls: "cell-pass", text: "pass" },
  fail: { icon: "✗", cls: "cell-fail", text: "fail" },
  error: { icon: "!", cls: "cell-error", text: "error (model server, not the prompt)" },
};

const SOURCE: Record<string, string> = {
  human: "your correction",
  answer_key: "the corpus answer key",
  judge: "the LLM judge",
  rule: "a deterministic rule",
  "": "none - checks only",
};

function CaseDetail({ caseId, onOpenTrace }: { caseId: string; onOpenTrace: (id: string) => void }) {
  const [detail, setDetail] = useState<EvalCaseDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    setDetail(null);
    api.evalCase(caseId).then(setDetail, (e: Error) => setError(e.message));
  }, [caseId]);

  if (error) return <div className="error-box">{error}</div>;
  if (!detail) return <div className="muted small">Loading…</div>;
  const { case: c, history } = detail;
  const latest = history[history.length - 1];
  const resultFor = (description: string) => latest?.assertions.find((a) => a.description === description);

  return (
    <div className="panel">
      <h2>
        {c.doc_id} · {c.step_name} · {humanize(c.category)}
        <span className="spacer" />
        <button onClick={() => onOpenTrace(c.source_trace_id)}>Open source trace</button>
      </h2>
      <div className="content">
        <p className="small muted" style={{ marginTop: 0 }}>
          Re-runs <b>{c.step_name}</b> on the exact input it received in {c.source_trace_id}. Recorded with pipeline{" "}
          <span className="mono">{c.pipeline_version}</span> and model <span className="mono">{c.model}</span>. Expected
          output from {SOURCE[c.expected_source] ?? c.expected_source}.
          {c.note && <> Reviewer note: {c.note}</>}
        </p>

        <h3 className="h3">Assertions {latest && <span className="muted small">(latest run: {latest.eval_run_id})</span>}</h3>
        <table className="checks">
          <tbody>
            {c.assertions.map((a) => {
              const r = resultFor(a.description);
              return (
                <tr key={a.description}>
                  <td className={`mark ${r ? (r.passed ? "pass" : "fail") : ""}`}>{r ? (r.passed ? "✓" : "✗") : "·"}</td>
                  <td>
                    {a.description}
                    {r && !r.passed && r.detail && <div className="muted small">{r.detail}</div>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {latest?.error && (
          <div className="error-box" style={{ marginTop: 10 }}>
            {latest.error.type}: {latest.error.message}
          </div>
        )}

        <div className="grid-2" style={{ marginTop: 14 }}>
          <div>
            <h3 className="h3">What it produced when it failed</h3>
            <pre>{JSON.stringify(c.bad_output, null, 2)}</pre>
          </div>
          <div>
            <h3 className="h3">{latest?.output ? "What it produces now" : "What it should produce"}</h3>
            <pre>{JSON.stringify(latest?.output ?? c.expected_output, null, 2)}</pre>
          </div>
        </div>
        <details style={{ marginTop: 12 }}>
          <summary className="small">Frozen step input</summary>
          <pre style={{ marginTop: 8 }}>{JSON.stringify(c.step_input, null, 2)}</pre>
        </details>
      </div>
    </div>
  );
}

export default function EvalsView({ onOpenTrace }: { onOpenTrace: (id: string) => void }) {
  const [matrix, setMatrix] = useState<EvalMatrix | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [label, setLabel] = useState("");
  const [model, setModel] = useState("");
  const [busy, setBusy] = useState<"run" | "generate" | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => api.evals().then(setMatrix, (e: Error) => setError(e.message)), []);
  useEffect(() => {
    load();
  }, [load]);

  async function act(kind: "run" | "generate") {
    setBusy(kind);
    setError(null);
    setMessage(null);
    try {
      if (kind === "generate") {
        const report = await api.generateEvals();
        const count = (a: string) => report.filter((r) => r.action === a).length;
        setMessage(
          report.length
            ? `${count("created")} created, ${count("updated")} updated, ${count("merged")} merged into existing cases, ${count("retired")} retired, ${count("skipped")} skipped.`
            : "No flags yet - confirm a diagnosis in the trace view first.",
        );
      } else {
        const s = await api.runEvals(label, model);
        setMessage(`${s.eval_run_id}: ${s.passed}/${s.total} passed${s.errors ? `, ${s.errors} infrastructure error(s)` : ""}.`);
        setLabel("");
      }
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  const runs = matrix?.runs ?? [];
  const cases = matrix?.cases ?? [];

  return (
    <div className="page">
      <div className="panel">
        <h2>Regression matrix</h2>
        <div className="content">
          <p className="small muted" style={{ marginTop: 0 }}>
            Every confirmed failure becomes a step-level test. A run re-executes each case's step on its frozen input with
            today's prompts and model. Unchanged prompts replay from cache and reproduce the failure; an edited prompt or
            another model gets a real model call.
          </p>
          <div className="form-row">
            <button disabled={busy !== null} onClick={() => act("generate")}>
              {busy === "generate" ? "Rebuilding…" : "Rebuild cases from flags"}
            </button>
            <input placeholder="Run label, e.g. 'intake keeps signature'" value={label} onChange={(e) => setLabel(e.target.value)} style={{ minWidth: 240 }} />
            <input placeholder="Model (optional)" value={model} onChange={(e) => setModel(e.target.value)} style={{ width: 150 }} />
            <button className="primary" disabled={busy !== null || !cases.length} onClick={() => act("run")}>
              {busy === "run" ? "Running… (uncached cases call the model)" : "Run evals"}
            </button>
          </div>
          {message && <div className="small" style={{ marginBottom: 10 }}>{message}</div>}
          {error && <div className="error-box">{error}</div>}

          {matrix && (
            <div className="form-row small">
              {(Object.keys(STATUS) as CaseStatus[])
                .filter((s) => matrix.summary[s])
                .map((s) => (
                  <span key={s} className={`badge ${STATUS[s].cls}`}>
                    {STATUS[s].icon} {matrix.summary[s]} {STATUS[s].label}
                  </span>
                ))}
              <span className="muted">
                current pipeline <span className="mono">{matrix.current_pipeline_version}</span>
              </span>
            </div>
          )}

          {matrix && !cases.length && (
            <div className="empty">
              No eval cases yet. Open a failed trace, review it with <b>Confirm</b> or <b>Override</b>, and it becomes a case
              here.
            </div>
          )}

          {cases.length > 0 && (
            <div className="matrix-wrap">
              <table className="matrix">
                <thead>
                  <tr>
                    <th>Case</th>
                    <th>Category</th>
                    {runs.map((r, i) => (
                      <th
                        key={r.eval_run_id}
                        className="run-col"
                        title={`${r.eval_run_id}\n${r.label || "(no label)"}\nmodel ${r.model}\npipeline ${r.pipeline_version ?? "?"}\n${r.passed}/${r.total} passed`}
                      >
                        #{i + 1}
                        {(r.fixed > 0 || r.regressed > 0) && (
                          <div className="small muted">
                            {r.fixed > 0 && `+${r.fixed}`} {r.regressed > 0 && `-${r.regressed}`}
                          </div>
                        )}
                      </th>
                    ))}
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {cases.map((c) => (
                    <tr
                      key={c.case_id}
                      className={c.case_id === selected ? "selected" : ""}
                      onClick={() => setSelected(c.case_id)}
                    >
                      <td>
                        <span className="mono">{c.doc_id}</span> · {c.step_name}
                        <div className="muted small">{c.assertion_count} assertions{c.seen_in > 1 && ` · confirmed in ${c.seen_in} traces`}</div>
                      </td>
                      <td className="small">{humanize(c.category)}</td>
                      {runs.map((r) => {
                        const cell = c.cells[r.eval_run_id];
                        if (!cell) return <td key={r.eval_run_id} className="cell-none">·</td>;
                        const look = CELL[cell.outcome];
                        return (
                          <td
                            key={r.eval_run_id}
                            className={look.cls}
                            title={`${look.text}${cell.from_cache ? " (replayed from cache)" : ""}${cell.failed.length ? `\nfailed: ${cell.failed.join("; ")}` : ""}`}
                          >
                            {look.icon}
                          </td>
                        );
                      })}
                      <td>
                        <span className={`badge ${STATUS[c.status].cls}`}>
                          {STATUS[c.status].icon} {STATUS[c.status].label}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {runs.length > 0 && (
                <div className="muted small" style={{ marginTop: 6 }}>
                  Hover a run number for its label, model and prompt version; hover a cell for the assertions that failed.
                  +n / -n under a run: cases it fixed / broke.
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {selected && <CaseDetail key={`${selected}-${runs.length}`} caseId={selected} onOpenTrace={onOpenTrace} />}
    </div>
  );
}
