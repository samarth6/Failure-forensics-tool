import { useEffect, useState } from "react";
import { api, type Json, type Signal, type Span, type StepDiff, type StepName } from "../api";
import { flatten, humanize, withoutMeta } from "../format";

type Tab = "signals" | "diff" | "io" | "prompt";

const SEVERITY_ORDER = { critical: 0, warn: 1, info: 2 } as const;

function sortSignals(signals: Signal[]): Signal[] {
  // Failures first, worst first; passing checks after.
  return [...signals].sort((a, b) => {
    if (a.passed !== b.passed) return a.passed ? 1 : -1;
    return SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity];
  });
}

function SignalList({ signals }: { signals: Signal[] }) {
  if (!signals.length) return <div className="muted small">No signals - this trace has not been diagnosed yet.</div>;
  return (
    <div>
      {sortSignals(signals).map((s, i) => {
        const cls = s.passed ? "pass" : `fail-${s.severity}`;
        return (
          <div key={`${s.name}-${i}`} className={`signal ${cls}`}>
            <span className="icon">{s.passed ? "✓" : s.severity === "info" ? "·" : "✗"}</span>
            <div>
              <div className="name">
                {s.name}
                {!s.passed && s.implicates ? <span className="muted"> → {humanize(s.implicates)}</span> : null}
              </div>
              <div className="detail">{s.detail}</div>
            </div>
            <div className="kind">
              {humanize(s.kind)}
              <br />
              {s.weight === 0 ? "discarded" : `weight ${s.weight}`}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/** The step's main input text, when it has one; otherwise its JSON. */
function receivedView(received: Json): string {
  for (const key of ["source_text", "normalized_text"]) {
    if (typeof received[key] === "string") return received[key] as string;
  }
  return JSON.stringify(received, null, 2);
}

function comparable(payload: Json | null): Json {
  // Extraction outputs nest the answer under `fields`; evidence quotes and confidence
  // are bookkeeping, not part of what the step should have answered.
  if (!payload) return {};
  const { evidence: _e, confidence: _c, rationale: _r, ...rest } = payload;
  return rest;
}

const SOURCE_TEXT: Record<string, string> = {
  human: "from your correction",
  answer_key: "from the corpus answer key",
  judge: "proposed by the LLM judge",
  rule: "proposed by a deterministic rule",
};

function DiffView({ traceId, step }: { traceId: string; step: StepName }) {
  const [diff, setDiff] = useState<StepDiff | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    setDiff(null);
    setError(null);
    api.diff(traceId, step).then(
      (d) => live && setDiff(d),
      (e: Error) => live && setError(e.message),
    );
    return () => {
      live = false;
    };
  }, [traceId, step]);

  if (error) return <div className="error-box">{error}</div>;
  if (!diff) return <div className="muted small">Loading…</div>;

  const produced = flatten(comparable(diff.produced));
  const expected = diff.should_have_produced ? flatten(comparable(diff.should_have_produced)) : null;
  const keys = [...new Set([...Object.keys(produced), ...Object.keys(expected ?? {})])];
  const mismatches = expected ? keys.filter((k) => (produced[k] ?? "") !== (expected[k] ?? "")).length : 0;

  return (
    <div className="grid-2">
      <div>
        <div className="source-note">
          <b>Received</b> - what this step was given
        </div>
        <pre>{receivedView(diff.received)}</pre>
      </div>
      <div>
        <div className="source-note">
          <b>Produced</b> vs <b>should have produced</b>{" "}
          {diff.source ? `(${SOURCE_TEXT[diff.source] ?? diff.source})` : ""}
          {expected ? ` · ${mismatches} difference${mismatches === 1 ? "" : "s"}` : ""}
        </div>
        {!expected && <div className="muted small" style={{ marginBottom: 8 }}>{diff.note}</div>}
        <table className="diff">
          <thead>
            <tr>
              <th>field</th>
              <th>produced</th>
              {expected && <th>should have produced</th>}
            </tr>
          </thead>
          <tbody>
            {keys.map((k) => {
              const bad = expected !== null && (produced[k] ?? "") !== (expected[k] ?? "");
              return (
                <tr key={k} className={bad ? "mismatch" : ""}>
                  <td className="key">{k}</td>
                  <td className="val produced">{produced[k] ?? <span className="muted">–</span>}</td>
                  {expected && <td className="val expected">{expected[k] ?? <span className="muted">–</span>}</td>}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PromptView({ span }: { span: Span }) {
  return (
    <div>
      <div className="small muted" style={{ marginBottom: 8 }}>
        {span.spliced
          ? "Spliced span: this output was injected by the counterfactual, not produced by the model."
          : `${span.provider || "-"} · ${span.model || "-"} · ${span.tokens_in} in / ${span.tokens_out} out tokens · ${span.attempts.length} attempt(s)${span.from_cache ? " · replayed from cache" : ""}`}
      </div>
      {span.error && (
        <div className="error-box">
          <b>{span.error.type}</b>: {span.error.message}
        </div>
      )}
      {span.system_prompt && (
        <>
          <div className="source-note"><b>System prompt</b></div>
          <pre style={{ marginBottom: 12 }}>{span.system_prompt}</pre>
        </>
      )}
      <div className="source-note"><b>Prompt</b></div>
      <pre style={{ marginBottom: 12 }}>{span.prompt || "(none)"}</pre>
      {span.attempts.map((a) => (
        <div key={a.index} style={{ marginBottom: 12 }}>
          <div className="source-note">
            <b>Raw response, attempt {a.index + 1}</b> {a.ok ? "- valid" : `- rejected: ${a.error}`}
          </div>
          <pre>{a.raw_response || "(empty)"}</pre>
        </div>
      ))}
    </div>
  );
}

export default function StepPanel({ traceId, span, step }: { traceId: string; span: Span | undefined; step: StepName }) {
  const [tab, setTab] = useState<Tab>("signals");
  const failing = span?.signals.filter((s) => !s.passed && s.weight > 0).length ?? 0;

  return (
    <div className="panel">
      <h2>
        <span style={{ textTransform: "capitalize" }}>{step}</span>
        {span && (
          <span className="muted small" style={{ fontWeight: 400 }}>
            {span.status} · {failing} failing check{failing === 1 ? "" : "s"}
          </span>
        )}
      </h2>
      {!span ? (
        <div className="content muted">This step did not run - the pipeline stopped before reaching it.</div>
      ) : (
        <>
          <div className="tabs">
            {(
              [
                ["signals", `Signals (${span.signals.length})`],
                ["diff", "Diff"],
                ["io", "Input & output"],
                ["prompt", "Prompt & raw"],
              ] as [Tab, string][]
            ).map(([key, label]) => (
              <button key={key} className={tab === key ? "on" : ""} onClick={() => setTab(key)}>
                {label}
              </button>
            ))}
          </div>
          <div className="content">
            {tab === "signals" && <SignalList signals={span.signals} />}
            {tab === "diff" && <DiffView traceId={traceId} step={step} />}
            {tab === "io" && (
              <div className="grid-2">
                <div>
                  <div className="source-note"><b>Input</b></div>
                  <pre>{JSON.stringify(withoutMeta(span.input_payload), null, 2)}</pre>
                </div>
                <div>
                  <div className="source-note"><b>Output</b></div>
                  <pre>{span.output_payload ? JSON.stringify(span.output_payload, null, 2) : "(no output)"}</pre>
                </div>
              </div>
            )}
            {tab === "prompt" && <PromptView span={span} />}
          </div>
        </>
      )}
    </div>
  );
}
