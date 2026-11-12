import { useCallback, useEffect, useState } from "react";
import { api, type Analytics } from "../api";
import { pct } from "../format";
import { BarChart, SERIES, StatTile } from "./Charts";

const duration = (ms: number | null | undefined): string => {
  if (ms == null) return "–";
  if (ms < 1000) return `${ms} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  return `${(ms / 60_000).toFixed(1)} min`;
};

const count = (v: number) => String(v);

function MttrPanel({ a, onChanged }: { a: Analytics; onChanged: () => void }) {
  const [minutes, setMinutes] = useState("");
  const [note, setNote] = useState("");
  const [error, setError] = useState<string | null>(null);
  const { automated, manual, speedup } = a.mttr;

  async function record() {
    setError(null);
    const value = Number(minutes);
    if (!(value > 0)) {
      setError("Enter the minutes it took you, e.g. 25.");
      return;
    }
    try {
      await api.addBaseline(value, note);
      setMinutes("");
      setNote("");
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <div className="panel">
      <h2>Mean time to root cause</h2>
      <div className="content">
        <div className="hero">
          {speedup != null ? (
            <>
              <div className="hero-value">{speedup.toLocaleString()}× faster</div>
              <div className="muted">
                {duration(automated.mean_ms)} automated vs {manual.mean_minutes} min by hand (mean of {manual.n} timing
                {manual.n === 1 ? "" : "s"})
              </div>
            </>
          ) : (
            <>
              <div className="hero-value">{duration(automated.median_ms)}</div>
              <div className="muted">median automated diagnosis over {automated.n} failure(s). No manual baseline yet.</div>
            </>
          )}
        </div>
        <div className="tiles">
          <StatTile label="Mean" value={duration(automated.mean_ms)} />
          <StatTile label="p95" value={duration(automated.p95_ms)} />
          <StatTile label="With counterfactual re-run" value={duration(automated.with_counterfactual_median_ms)} sub="median" />
          <StatTile label="Signals only" value={duration(automated.signals_only_median_ms)} sub="median" />
        </div>
        <p className="small muted">
          Automated time includes counterfactual re-runs. Re-runs answered from the replay cache are fast; a cold run calls
          the model and takes much longer. For an honest comparison, time yourself finding a root cause by hand (reading
          trace files, no explorer) and record it here.
        </p>
        <div className="form-row">
          <input placeholder="Minutes by hand" value={minutes} onChange={(e) => setMinutes(e.target.value)} style={{ width: 140 }} />
          <input placeholder="How, e.g. 'read JSON traces for doc_005'" value={note} onChange={(e) => setNote(e.target.value)} style={{ flex: 1, minWidth: 200 }} />
          <button onClick={record}>Record manual timing</button>
        </div>
        {error && <div className="error-box">{error}</div>}
        {manual.entries.length > 0 && (
          <ul className="small muted" style={{ margin: 0, paddingLeft: 18 }}>
            {manual.entries.map((e, i) => (
              <li key={i}>
                {e.minutes} min{e.note && ` - ${e.note}`}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

export default function AnalyticsView({ runId }: { runId?: string }) {
  const [a, setA] = useState<Analytics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const load = useCallback(() => api.analytics(runId).then(setA, (e: Error) => setError(e.message)), [runId]);
  useEffect(() => {
    load();
  }, [load]);

  if (error) return <div className="page"><div className="error-box">{error}</div></div>;
  if (!a) return <div className="page muted">Loading…</div>;
  if (!a.run_id || !a.overview.diagnosed) {
    return (
      <div className="page empty">
        This run has no diagnoses yet. Run <code>python -m ffx diagnose</code> first.
      </div>
    );
  }

  const t = a.versus_truth;
  const h = a.versus_human;
  const j = a.judge;
  const cal = a.calibration;

  return (
    <div className="page">
      <div className="tiles">
        <StatTile
          label="Failures"
          value={`${a.overview.failures} / ${a.overview.diagnosed}`}
          sub={`${pct(a.overview.failure_rate)} of diagnosed traces`}
        />
        <StatTile
          label="Root step right"
          value={pct(t?.root_accuracy)}
          sub={t ? `vs ground truth, ${t.tp} detected failures` : "no answer keys"}
        />
        <StatTile
          label="Detection"
          value={t ? `${pct(t.precision)} / ${pct(t.recall)}` : "–"}
          sub="precision / recall vs ground truth"
        />
        <StatTile label="Human agreement" value={pct(h.agreement)} sub={`${h.reviewed} reviewed`} />
        <StatTile
          label="Eval pass rate"
          value={pct(a.evals.latest_pass_rate)}
          sub={`${a.evals.cases} case(s), ${a.evals.runs} run(s)`}
        />
      </div>

      <MttrPanel a={a} onChanged={load} />

      <div className="grid-2">
        <div className="panel">
          <h2>Failure categories</h2>
          <div className="content">
            {a.category_mix.length ? (
              <BarChart
                rows={a.category_mix.map((c) => ({ label: c.label, sub: pct(c.share), values: [c.count] }))}
                series={[{ name: "Diagnosed failures", color: SERIES[0] }]}
                format={count}
                caption="Root-cause category of each diagnosed failure in this run."
              />
            ) : (
              <div className="muted small">No failures in this run.</div>
            )}
          </div>
        </div>
        <div className="panel">
          <h2>Where failures start, and where they spread</h2>
          <div className="content">
            <BarChart
              rows={a.per_step.map((s) => ({ label: s.step, values: [s.root_cause, s.failed_own_checks, s.propagation] }))}
              series={[
                { name: "Root cause", color: SERIES[0] },
                { name: "Failed own checks", color: SERIES[1] },
                { name: "Propagation", color: SERIES[2] },
              ]}
              format={count}
              caption="Per step, over failed traces. A step that only propagates received damage and passed its own checks."
            />
          </div>
        </div>
      </div>

      <div className="grid-2">
        <div className="panel">
          <h2>Is the model's own confidence worth anything?</h2>
          <div className="content">
            <BarChart
              rows={cal.self_report.map((b) => ({
                label: `${b.confidence}/5`,
                sub: `n=${b.n}`,
                values: [b.n ? b.failure_rate : null],
              }))}
              series={[{ name: "Step actually failed", color: SERIES[0] }]}
              format={pct}
              max={1}
              caption={
                <>
                  Share of steps that really failed, by the confidence the model reported for them (basis:{" "}
                  {cal.self_report_basis}). Calibrated means the bars fall as confidence rises.
                </>
              }
            />
          </div>
        </div>
        <div className="panel">
          <h2>Is the diagnoser's confidence worth anything?</h2>
          <div className="content">
            <BarChart
              rows={cal.diagnosis_confidence.map((b) => ({ label: b.range, sub: `n=${b.n}`, values: [b.n ? b.accuracy : null] }))}
              series={[{ name: "Root step correct", color: SERIES[0] }]}
              format={pct}
              max={1}
              caption="Share of diagnoses whose root step matches ground truth, by the diagnoser's stated confidence. Calibrated means the bars rise with confidence."
            />
          </div>
        </div>
      </div>

      <div className="grid-2">
        <div className="panel">
          <h2>LLM judge vs answer key</h2>
          <div className="content">
            <div className="tiles">
              <StatTile label="Verdicts" value={j.verdicts} sub={`${j.discarded_unquoted} discarded (unquoted)`} />
              <StatTile label="Accuracy" value={pct(j.accuracy)} sub={`${j.scored} scored`} />
              <StatTile label="Precision" value={pct(j.precision)} sub="when it said wrong, was it?" />
              <StatTile label="Recall" value={pct(j.recall)} sub="real errors it caught" />
            </div>
            <div className="muted small">
              Answered by:{" "}
              {Object.entries(j.by_provider)
                .map(([p, n]) => `${p} ${n}`)
                .join(", ") || "–"}
            </div>
          </div>
        </div>
        <div className="panel">
          <h2>Diagnosis vs human review</h2>
          <div className="content">
            <table className="checks">
              <tbody>
                <tr><td>Agreed</td><td>{h.agree}</td></tr>
                <tr><td>Wrong root step</td><td>{h.wrong_step}</td></tr>
                <tr><td>Right step, wrong category</td><td>{h.wrong_category}</td></tr>
                <tr><td>False alarm (not a failure)</td><td>{h.false_alarm}</td></tr>
                <tr><td>Missed failure</td><td>{h.missed}</td></tr>
              </tbody>
            </table>
            {t && t.confusion.length > 0 && (
              <details style={{ marginTop: 10 }}>
                <summary className="small">Root step: ground truth vs diagnosed</summary>
                <table className="checks">
                  <tbody>
                    {t.confusion.map((c) => (
                      <tr key={`${c.truth}-${c.diagnosed}`} className={c.truth === c.diagnosed ? "" : "implicated"}>
                        <td>truth {c.truth}</td>
                        <td>diagnosed {c.diagnosed}</td>
                        <td>{c.count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </details>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
