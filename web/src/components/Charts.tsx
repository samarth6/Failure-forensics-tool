import { useState, type ReactNode } from "react";

// Categorical slots 1-3, validated (CVD + contrast) against this app's light and dark
// panel surfaces. Assigned in fixed order: a series keeps its colour wherever it appears.
export const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];

export interface Series {
  name: string;
  color: string;
}

export interface BarRow {
  label: string;
  sub?: string;
  values: (number | null)[];
}

/** Horizontal bars, one row per category, one thin bar per series. Every bar carries
 *  its value at the tip, so colour is never the only way to read it, and a table view
 *  sits underneath. */
export function BarChart({
  rows,
  series,
  format,
  max,
  caption,
}: {
  rows: BarRow[];
  series: Series[];
  format: (v: number) => string;
  max?: number;
  caption?: ReactNode;
}) {
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  const top = max ?? Math.max(1, ...rows.flatMap((r) => r.values.map((v) => v ?? 0)));
  const grouped = series.length > 1;

  return (
    <figure className="chart">
      {grouped && (
        <div className="chart-legend">
          {series.map((s) => (
            <span key={s.name}>
              <i style={{ background: s.color }} />
              {s.name}
            </span>
          ))}
        </div>
      )}
      <div className="bars" onMouseLeave={() => setTip(null)}>
        {rows.map((row) => (
          <div className="bar-row" key={row.label}>
            <div className="bar-label">
              {row.label}
              {row.sub && <span className="muted"> {row.sub}</span>}
            </div>
            <div className={`bar-group${grouped ? " grouped" : ""}`}>
              {row.values.map((v, i) => (
                <div
                  className="bar-line"
                  key={series[i].name}
                  onMouseMove={(e) =>
                    setTip({
                      x: e.clientX,
                      y: e.clientY,
                      text: `${row.label}${grouped ? ` · ${series[i].name}` : ""}: ${v == null ? "no data" : format(v)}`,
                    })
                  }
                >
                  {v == null ? (
                    <span className="bar-none">no data</span>
                  ) : (
                    <>
                      {v > 0 && (
                        <div
                          className="bar"
                          style={{ width: `calc((100% - 56px) * ${v / top})`, background: series[i].color }}
                        />
                      )}
                      <span className="bar-value">{format(v)}</span>
                    </>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
      {tip && (
        <div className="chart-tip" style={{ left: tip.x + 12, top: tip.y + 12 }}>
          {tip.text}
        </div>
      )}
      {caption && <figcaption>{caption}</figcaption>}
      <details className="chart-table">
        <summary>Table</summary>
        <table>
          <thead>
            <tr>
              <th />
              {series.map((s) => (
                <th key={s.name}>{s.name}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.label}>
                <td>
                  {row.label}
                  {row.sub && <span className="muted"> {row.sub}</span>}
                </td>
                {row.values.map((v, i) => (
                  <td key={i}>{v == null ? "–" : format(v)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}

export function StatTile({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="tile">
      <div className="tile-label">{label}</div>
      <div className="tile-value">{value}</div>
      {sub && <div className="tile-sub">{sub}</div>}
    </div>
  );
}
