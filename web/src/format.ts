import type { Json, Match, Role, StepName } from "./api";

export const STEPS: StepName[] = ["intake", "extraction", "classification", "summarization"];

export const ROLE_LABEL: Record<Role, string> = {
  healthy: "Healthy",
  low_confidence: "Low confidence",
  root_cause: "Root cause",
  candidate: "Also failed",
  propagation: "Propagation",
  not_run: "Not run",
};

export const MATCH_LABEL: Record<Exclude<Match, null>, string> = {
  root_ok: "root matches truth",
  root_wrong: "root differs from truth",
  missed: "missed failure",
  false_alarm: "false alarm",
  ok: "correctly healthy",
};

export const MATCH_SHORT: Record<Exclude<Match, null>, string> = {
  root_ok: "✓ root",
  root_wrong: "✗ root",
  missed: "missed",
  false_alarm: "false alarm",
  ok: "✓",
};

export const pct = (x: number | null | undefined): string => (x == null ? "–" : `${Math.round(x * 100)}%`);

export const humanize = (s: string | null | undefined): string => (s ? s.replace(/_/g, " ") : "–");

/** Flatten nested JSON to dotted paths, for field-by-field comparison. */
export function flatten(value: unknown, prefix = "", out: Record<string, string> = {}): Record<string, string> {
  if (value === null || value === undefined) {
    out[prefix || "(value)"] = "";
  } else if (Array.isArray(value)) {
    if (value.every((v) => typeof v !== "object" || v === null)) {
      out[prefix] = value.join(", ");
    } else {
      value.forEach((v, i) => flatten(v, `${prefix}[${i}]`, out));
    }
  } else if (typeof value === "object") {
    for (const [k, v] of Object.entries(value as Json)) flatten(v, prefix ? `${prefix}.${k}` : k, out);
  } else {
    out[prefix || "(value)"] = String(value);
  }
  return out;
}

export function withoutMeta(payload: Json | null | undefined): Json {
  if (!payload) return {};
  const { _meta: _ignored, ...rest } = payload;
  return rest;
}
