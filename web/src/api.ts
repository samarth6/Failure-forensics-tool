// Types mirror the backend's Pydantic models (ffx/tracing/span.py,
// ffx/analyzer/diagnosis_models.py). Keep them in step: a mismatch should fail the
// TypeScript build, not render quietly wrong.

export type StepName = "intake" | "extraction" | "classification" | "summarization";
export type Json = Record<string, unknown>;

export type Role = "healthy" | "low_confidence" | "root_cause" | "candidate" | "propagation" | "not_run";
export type Verdict = "confirmed" | "partial" | "refuted" | "contained" | "inconclusive" | "not_applicable";
export type Match = "root_ok" | "root_wrong" | "missed" | "false_alarm" | "ok" | null;

export interface Signal {
  name: string;
  kind: "deterministic" | "self_report" | "llm_judge" | "human";
  severity: "info" | "warn" | "critical";
  passed: boolean;
  score: number;
  weight: number;
  detail: string;
  evidence: Json;
  implicates: string | null;
  proposed_correction: Json | null;
}

export interface Attempt {
  index: number;
  raw_response: string;
  ok: boolean;
  error: string;
  latency_ms: number;
  from_cache: boolean;
}

export interface Span {
  span_id: string;
  step_name: StepName;
  step_index: number;
  duration_ms: number;
  status: "ok" | "degraded" | "error";
  input_payload: Json;
  output_payload: Json | null;
  prompt: string;
  system_prompt: string;
  raw_response: string;
  provider: string;
  model: string;
  tokens_in: number;
  tokens_out: number;
  attempts: Attempt[];
  from_cache: boolean;
  self_confidence: number | null;
  error: { type: string; message: string } | null;
  signals: Signal[];
  spliced: boolean;
  copied_from_span_id: string | null;
}

export interface Trace {
  trace_id: string;
  doc_id: string;
  run_id: string;
  status: "running" | "ok" | "failed" | "error";
  duration_ms: number;
  spans: Span[];
  metadata: Json & { source_text?: string; failure_mode?: string };
  parent_trace_id: string | null;
}

export interface Assessment {
  span_id: string;
  step_name: StepName;
  q_in: number;
  local_quality: number;
  q_out: number;
  local_damage: number;
  role: Role;
  failed_signals: string[];
}

export interface Candidate {
  rank: number;
  span_id: string;
  step_name: StepName;
  own_damage: number;
  category: string;
  subtype: string;
}

export interface EvidenceLink {
  order: number;
  span_id: string;
  step_name: StepName;
  role: "origin" | "propagation" | "independent" | "symptom" | "verification";
  claim: string;
  signal: string;
  evidence: Json;
}

export interface Check {
  check: string;
  field: string;
  passed: boolean;
  found?: unknown;
}

export interface CounterfactualResult {
  verdict: Verdict;
  splice_step: StepName;
  correction_source: string;
  corrected_output: Json | null;
  counterfactual_trace_id: string | null;
  fidelity_before: number | null;
  fidelity_after: number | null;
  checks_before: Check[];
  checks_after: Check[];
  implicated_checks: string[];
  fixed_checks: string[];
  remaining_failures: string[];
  new_downstream_failures: string[];
  note: string;
}

export interface Diagnosis {
  trace_id: string;
  is_failure: boolean;
  terminal_quality: number;
  root_span_id: string | null;
  root_step: StepName | null;
  category: string;
  subtype: string;
  confidence: number;
  explanation: string;
  candidates: Candidate[];
  assessments: Assessment[];
  evidence_chain: EvidenceLink[];
  propagation_path: string[];
  counterfactuals: CounterfactualResult[];
  judge_provider: string;
  judge_calls: number;
  diagnose_ms: number;
}

export interface Truth {
  is_failure: boolean;
  root_step: StepName | null;
  category: string;
  reasons: string[];
}

export interface Flag {
  flag_id: string;
  trace_id: string;
  step_name: StepName | null;
  verdict: "confirm" | "override" | "not_a_failure";
  human_category: string | null;
  corrected_output: Json | null;
  note: string;
  created_at: string;
}

export interface TraceRow {
  trace_id: string;
  doc_id: string;
  run_id: string;
  status: string;
  duration_ms: number;
  failure_mode: string | null;
  is_failure: number | null;
  root_step: StepName | null;
  category: string | null;
  confidence: number | null;
  counterfactual: Verdict | null;
  human_verdict: string | null;
  truth: Truth | null;
  match: Match;
}

export interface TraceDetail {
  trace: Trace;
  diagnosis: Diagnosis | null;
  counterfactuals: { result: CounterfactualResult; trace: Trace | null }[];
  truth: Truth | null;
  answer_key: Json | null;
  failure_mode: string | null;
  flags: Flag[];
}

export interface StepDiff {
  step: StepName;
  received: Json;
  produced: Json | null;
  should_have_produced: Json | null;
  source: "human" | "answer_key" | "judge" | "rule" | null;
  note: string | null;
}

export interface Meta {
  steps: StepName[];
  categories: Record<string, { label: string; description: string }>;
  pipeline_model: string;
  judge_order: string[];
  corpus_size: number;
}

export interface Run {
  run_id: string;
  started_at: string;
  pipeline_version: string;
  model: string;
  notes: string;
  trace_count: number;
}

// ---------------------------------------------------------------- Phase 5

export type Outcome = "pass" | "fail" | "error";
export type CaseStatus = "not_run" | "passing" | "failing" | "fixed" | "regressed" | "error";

export interface EvalCaseSync {
  trace_id: string;
  doc_id?: string;
  case_id: string | null;
  step?: StepName;
  action: "created" | "updated" | "merged" | "retired" | "skipped";
  reason: string;
}

export interface FlagResponse extends Flag {
  eval_case: EvalCaseSync;
}

export interface EvalRunRow {
  eval_run_id: string;
  started_at: string;
  label: string;
  model: string;
  pipeline_version: string | null;
  total: number;
  passed: number;
  fixed: number;
  regressed: number;
}

export interface MatrixCell {
  outcome: Outcome;
  failed: string[];
  from_cache: boolean | null;
}

export interface MatrixRow {
  case_id: string;
  doc_id: string;
  source_trace_id: string;
  step_name: StepName;
  category: string;
  expected_source: string;
  assertion_count: number;
  seen_in: number;
  created_at: string;
  cells: Record<string, MatrixCell>;
  status: CaseStatus;
}

export interface EvalMatrix {
  runs: EvalRunRow[];
  cases: MatrixRow[];
  summary: Partial<Record<CaseStatus, number>>;
  current_pipeline_version: string;
}

export interface Assertion {
  kind: "produces_output" | "equals" | "contains" | "check_passes";
  description: string;
  path: string;
  expected: string;
  signal: string;
}

export interface AssertionResult {
  description: string;
  passed: boolean;
  detail: string;
}

export interface EvalCase {
  case_id: string;
  source_trace_id: string;
  doc_id: string;
  step_name: StepName;
  category: string;
  verdict: string;
  created_at: string;
  pipeline_version: string;
  model: string;
  step_input: Json;
  bad_output: Json | null;
  expected_output: Json | null;
  expected_source: string;
  assertions: Assertion[];
  note: string;
}

export interface CaseRunResult {
  eval_run_id: string;
  label: string;
  started_at: string;
  outcome: Outcome;
  assertions: AssertionResult[];
  output: Json | null;
  error: { type: string; message: string } | null;
  from_cache: boolean;
  duration_ms: number;
}

export interface EvalCaseDetail {
  case: EvalCase;
  history: CaseRunResult[];
}

export interface EvalRunSummary {
  eval_run_id: string;
  total: number;
  passed: number;
  errors: number;
  model: string;
  pipeline_version: string;
}

type Rate = number | null;

export interface Analytics {
  run_id: string | null;
  overview: { traces: number; diagnosed: number; failures: number; failure_rate: Rate };
  category_mix: { category: string; label: string; count: number; share: Rate }[];
  human_category_mix: { category: string; label: string; count: number }[];
  per_step: { step: StepName; root_cause: number; failed_own_checks: number; propagation: number; own_failure_rate: Rate }[];
  mttr: {
    automated: {
      n: number;
      mean_ms: number | null;
      median_ms: number | null;
      p95_ms: number | null;
      with_counterfactual_median_ms: number | null;
      signals_only_median_ms: number | null;
    };
    manual: { n: number; mean_minutes: number | null; entries: { minutes: number; note: string; trace_id: string | null }[] };
    speedup: number | null;
    note: string | null;
  };
  versus_truth: {
    n: number;
    tp: number;
    fp: number;
    fn: number;
    tn: number;
    precision: Rate;
    recall: Rate;
    root_accuracy: Rate;
    category_accuracy: Rate;
    confusion: { truth: string; diagnosed: string; count: number }[];
  } | null;
  versus_human: {
    reviewed: number;
    agree: number;
    wrong_step: number;
    wrong_category: number;
    false_alarm: number;
    missed: number;
    agreement: Rate;
  };
  judge: {
    verdicts: number;
    discarded_unquoted: number;
    scored: number;
    accuracy: Rate;
    precision: Rate;
    recall: Rate;
    by_provider: Record<string, number>;
  };
  calibration: {
    self_report: { confidence: number; n: number; failed: number; failure_rate: Rate }[];
    self_report_basis: string;
    diagnosis_confidence: { range: string; n: number; mean_confidence: number | null; root_correct: number; accuracy: Rate }[];
  };
  evals: { cases: number; status: Partial<Record<CaseStatus, number>>; runs: number; latest_pass_rate: Rate };
}

export interface FlagRequest {
  verdict: Flag["verdict"];
  step_name?: StepName;
  human_category?: string;
  corrected_output?: Json;
  note?: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body: keep statusText */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  meta: () => request<Meta>("/api/meta"),
  runs: () => request<Run[]>("/api/runs"),
  traces: (runId?: string) => request<TraceRow[]>(`/api/traces${runId ? `?run_id=${encodeURIComponent(runId)}` : ""}`),
  trace: (id: string) => request<TraceDetail>(`/api/traces/${id}`),
  diff: (id: string, step: StepName) => request<StepDiff>(`/api/traces/${id}/diff/${step}`),
  flag: (id: string, body: FlagRequest) =>
    request<FlagResponse>(`/api/traces/${id}/flags`, { method: "POST", body: JSON.stringify(body) }),
  rediagnose: (id: string) => request<Diagnosis>(`/api/traces/${id}/diagnose`, { method: "POST" }),

  evals: () => request<EvalMatrix>("/api/evals"),
  evalCase: (caseId: string) => request<EvalCaseDetail>(`/api/evals/cases/${caseId}`),
  generateEvals: () => request<EvalCaseSync[]>("/api/evals/generate", { method: "POST" }),
  runEvals: (label: string, model?: string) =>
    request<EvalRunSummary>("/api/evals/run", { method: "POST", body: JSON.stringify({ label, model: model || null }) }),
  analytics: (runId?: string) =>
    request<Analytics>(`/api/analytics${runId ? `?run_id=${encodeURIComponent(runId)}` : ""}`),
  addBaseline: (minutes: number, note: string, traceId?: string) =>
    request<unknown>("/api/analytics/baselines", {
      method: "POST",
      body: JSON.stringify({ minutes, note, trace_id: traceId ?? null }),
    }),
};
