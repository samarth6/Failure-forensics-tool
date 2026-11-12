import { useState } from "react";
import { api, type Diagnosis, type Flag, type FlagRequest, type Meta, type StepName, type Trace } from "../api";
import { STEPS, humanize } from "../format";

type Verdict = Flag["verdict"];

export default function FlagPanel({
  trace,
  diagnosis,
  flags,
  meta,
  onChanged,
}: {
  trace: Trace;
  diagnosis: Diagnosis | null;
  flags: Flag[];
  meta: Meta | null;
  onChanged: () => void;
}) {
  const failed = diagnosis?.is_failure ?? false;
  const [verdict, setVerdict] = useState<Verdict>(failed ? "confirm" : "override");
  const [step, setStep] = useState<StepName>(diagnosis?.root_step ?? "extraction");
  const [category, setCategory] = useState<string>(diagnosis?.category ?? "extraction_hallucination");
  const [withCorrection, setWithCorrection] = useState(false);
  const [correction, setCorrection] = useState("");
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState<"flag" | "diagnose" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const correctionStep: StepName = verdict === "confirm" ? diagnosis?.root_step ?? step : step;
  const categories = Object.entries(meta?.categories ?? {}).filter(([k]) => k !== "no_failure");

  async function loadCorrection(target: StepName) {
    // Start from the best available reference, so the reviewer edits rather than types.
    try {
      const diff = await api.diff(trace.trace_id, target);
      setCorrection(JSON.stringify(diff.should_have_produced ?? diff.produced ?? {}, null, 2));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function submit() {
    setError(null);
    setMessage(null);
    const body: FlagRequest = { verdict, note };
    if (verdict === "override") {
      body.step_name = step;
      body.human_category = category;
    }
    if (withCorrection && verdict !== "not_a_failure") {
      try {
        body.corrected_output = JSON.parse(correction);
      } catch {
        setError("The corrected output is not valid JSON.");
        return;
      }
      body.step_name = correctionStep;
    }
    setBusy("flag");
    try {
      const saved = await api.flag(trace.trace_id, body);
      const ev = saved.eval_case;
      setMessage(
        ev.action === "created" || ev.action === "updated" || ev.action === "merged"
          ? `Saved. Eval case ${ev.case_id} ${ev.action} for ${ev.step} (${ev.reason}) - see the Evals page.`
          : ev.action === "retired"
            ? `Saved. This trace's eval case was retired (${ev.reason}).`
            : `Saved. No eval case: ${ev.reason}.`,
      );
      setNote("");
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function rediagnose() {
    setError(null);
    setMessage(null);
    setBusy("diagnose");
    try {
      await api.rediagnose(trace.trace_id);
      setMessage("Diagnosis re-run. Counterfactuals now use your latest correction, if you gave one.");
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="panel">
      <h2>Human review</h2>
      <div className="content">
        {flags.length > 0 && (
          <div style={{ marginBottom: 12 }}>
            {flags.map((f) => (
              <div key={f.flag_id} className="flag-item">
                <b>{humanize(f.verdict)}</b>
                {f.step_name && <> · {f.step_name}</>}
                {f.human_category && <> · {humanize(f.human_category)}</>}
                {f.corrected_output && <> · with corrected output</>}
                {f.note && <span className="muted"> - {f.note}</span>}
                <span className="muted small"> ({new Date(f.created_at).toLocaleString()})</span>
              </div>
            ))}
          </div>
        )}

        <div className="form-row">
          {(
            [
              ["confirm", "Confirm diagnosis", !failed],
              ["override", "Override", false],
              ["not_a_failure", "Not a failure", false],
            ] as [Verdict, string, boolean][]
          ).map(([value, label, disabled]) => (
            <label key={value} className={disabled ? "muted" : ""}>
              <input
                type="radio"
                name="verdict"
                checked={verdict === value}
                disabled={disabled}
                onChange={() => setVerdict(value)}
              />
              {label}
            </label>
          ))}
        </div>

        {verdict === "override" && (
          <div className="form-row">
            <label>
              Root step
              <select value={step} onChange={(e) => setStep(e.target.value as StepName)}>
                {STEPS.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </label>
            <label>
              Category
              <select value={category} onChange={(e) => setCategory(e.target.value)}>
                {categories.map(([k, v]) => (
                  <option key={k} value={k}>{v.label}</option>
                ))}
              </select>
            </label>
          </div>
        )}

        {verdict !== "not_a_failure" && (
          <div className="form-row">
            <label>
              <input
                type="checkbox"
                checked={withCorrection}
                onChange={(e) => {
                  setWithCorrection(e.target.checked);
                  if (e.target.checked && !correction) loadCorrection(correctionStep);
                }}
              />
              Provide the corrected output for <b>{correctionStep}</b>
            </label>
          </div>
        )}
        {withCorrection && verdict !== "not_a_failure" && (
          <div style={{ marginBottom: 10 }}>
            <textarea rows={10} value={correction} onChange={(e) => setCorrection(e.target.value)} />
            <div className="muted small">
              Pre-filled with the best available reference. It is validated against the step's schema, then used as the
              counterfactual correction and as the expected output of the generated eval case.
            </div>
          </div>
        )}

        <div className="form-row">
          <input
            style={{ flex: 1, minWidth: 200 }}
            placeholder="Note (optional)"
            value={note}
            onChange={(e) => setNote(e.target.value)}
          />
          <button className="primary" disabled={busy !== null} onClick={submit}>
            {busy === "flag" ? "Saving…" : "Save review"}
          </button>
          <button
            disabled={busy !== null}
            onClick={rediagnose}
            title="Re-runs signals, the judge and counterfactuals. Needs Ollama running; can take a few minutes."
          >
            {busy === "diagnose" ? "Re-diagnosing… (can take minutes)" : "Re-run diagnosis"}
          </button>
        </div>
        {error && <div className="error-box">{error}</div>}
        {message && <div className="muted small">{message}</div>}
      </div>
    </div>
  );
}
