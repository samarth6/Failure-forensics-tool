import type { CounterfactualResult, Diagnosis, EvidenceLink } from "../api";
import { humanize, pct } from "../format";

const LINK_ROLE_CLASS: Record<EvidenceLink["role"], string> = {
  origin: "role-root_cause",
  propagation: "role-propagation",
  independent: "role-candidate",
  symptom: "role-not_run",
  verification: "role-healthy",
};

export function EvidenceChain({ diagnosis }: { diagnosis: Diagnosis }) {
  if (!diagnosis.evidence_chain.length) {
    return <div className="muted small">No failure, so no evidence chain.</div>;
  }
  return (
    <ol className="chain">
      {diagnosis.evidence_chain.map((link) => (
        <li key={link.order}>
          <div>
            <span className={`badge ${LINK_ROLE_CLASS[link.role]}`}>{link.role}</span>
          </div>
          <div>
            <div>{link.claim}</div>
            {link.signal && <div className="muted small mono">{link.signal}</div>}
          </div>
        </li>
      ))}
    </ol>
  );
}

const SOURCE_TEXT: Record<string, string> = {
  human: "your correction",
  answer_key: "the corpus answer key",
  judge: "the LLM judge's proposal",
  rule: "a deterministic rule",
};

function CounterfactualCard({ cf }: { cf: CounterfactualResult }) {
  const after = new Map(cf.checks_after.map((c) => [c.check, c]));
  return (
    <div style={{ padding: "4px 0 14px" }}>
      <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
        <b style={{ textTransform: "capitalize" }}>Corrected {cf.splice_step}</b>
        <span className={`badge verdict-${cf.verdict}`}>{humanize(cf.verdict)}</span>
        {cf.fidelity_before != null && (
          <span className="fidelity">
            {pct(cf.fidelity_before)} → {cf.fidelity_after != null ? pct(cf.fidelity_after) : "–"}
          </span>
        )}
        {cf.correction_source && (
          <span className="muted small">correction from {SOURCE_TEXT[cf.correction_source] ?? cf.correction_source}</span>
        )}
      </div>
      <p className="small" style={{ margin: "8px 0 0" }}>{cf.note}</p>

      {cf.checks_before.length > 0 && (
        <table className="checks">
          <thead>
            <tr className="muted small">
              <td className="mark">before</td>
              <td className="mark">after</td>
              <td>final-output check {cf.implicated_checks.length > 0 && "(highlighted = predicted to change; only these decide the verdict)"}</td>
            </tr>
          </thead>
          <tbody>
            {cf.checks_before.map((c) => {
              const a = after.get(c.check);
              return (
                <tr key={c.check} className={cf.implicated_checks.includes(c.check) ? "implicated" : ""}>
                  <td className={`mark ${c.passed ? "pass" : "fail"}`}>{c.passed ? "✓" : "✗"}</td>
                  <td className={`mark ${a ? (a.passed ? "pass" : "fail") : ""}`}>{a ? (a.passed ? "✓" : "✗") : "–"}</td>
                  <td>{c.check}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}

export function Counterfactuals({ diagnosis }: { diagnosis: Diagnosis }) {
  if (!diagnosis.counterfactuals.length) {
    return (
      <div className="muted small">
        No counterfactual was run. Re-run the diagnosis with the local model available to test the root cause.
      </div>
    );
  }
  return (
    <div>
      <p className="muted small" style={{ marginTop: 0 }}>
        Each candidate is tested by splicing in a corrected output and re-running every later step. A candidate is
        ruled out only if the correction fails to fix the final-output facts it touches.
      </p>
      {diagnosis.counterfactuals.map((cf, i) => (
        <CounterfactualCard key={`${cf.splice_step}-${i}`} cf={cf} />
      ))}
    </div>
  );
}
