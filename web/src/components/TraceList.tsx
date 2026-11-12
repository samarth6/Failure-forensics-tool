import { useState } from "react";
import type { TraceRow } from "../api";
import { MATCH_LABEL, MATCH_SHORT, humanize, pct } from "../format";

type Filter = "all" | "failures" | "disagree" | "flagged";

const FILTERS: [Filter, string, (r: TraceRow) => boolean][] = [
  ["all", "All", () => true],
  ["failures", "Failures", (r) => r.is_failure === 1],
  // Where the diagnosis and corpus ground truth disagree - the rows worth reviewing.
  ["disagree", "Disagrees with truth", (r) => r.match === "root_wrong" || r.match === "missed" || r.match === "false_alarm"],
  ["flagged", "Reviewed", (r) => r.human_verdict != null],
];

export default function TraceList({
  rows,
  selectedId,
  onSelect,
}: {
  rows: TraceRow[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const [filter, setFilter] = useState<Filter>("all");
  const predicate = FILTERS.find(([key]) => key === filter)![2];
  const shown = rows.filter(predicate);

  return (
    <div>
      <div className="filters">
        {FILTERS.map(([key, label, test]) => (
          <button key={key} className={filter === key ? "on" : ""} onClick={() => setFilter(key)}>
            {label} ({rows.filter(test).length})
          </button>
        ))}
      </div>
      {shown.length === 0 && <div className="empty">No traces match this filter.</div>}
      {shown.map((r) => {
        const diagnosed = r.is_failure != null;
        return (
          <button
            key={r.trace_id}
            className={`trace-row${r.trace_id === selectedId ? " selected" : ""}`}
            onClick={() => onSelect(r.trace_id)}
          >
            <div className="line1">
              <span className="doc">{r.doc_id}</span>
              <span className="mode">{humanize(r.failure_mode)}</span>
            </div>
            <div className="line2">
              {!diagnosed && <span className="badge role-not_run">not diagnosed</span>}
              {diagnosed && r.is_failure === 1 && (
                <span className="badge role-root_cause" title={`confidence ${pct(r.confidence)}`}>
                  {r.root_step} · {humanize(r.category)}
                </span>
              )}
              {diagnosed && r.is_failure === 0 && <span className="badge role-healthy">healthy</span>}
              {r.match && (
                <span className={`badge match-${r.match}`} title={MATCH_LABEL[r.match]}>
                  {MATCH_SHORT[r.match]}
                </span>
              )}
              {r.human_verdict && <span className="badge role-propagation">reviewed</span>}
            </div>
          </button>
        );
      })}
    </div>
  );
}
