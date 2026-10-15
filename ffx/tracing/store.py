"""Persistence: JSON files for full traces, SQLite for everything queryable.

Why two stores
--------------
* ``data/traces/<trace_id>.json`` is the source of truth for a trace - complete payloads,
  prompts, raw responses, every repair attempt. Plain files are trivially inspectable,
  diffable, and survive a schema change in the index.
* ``data/ffx.db`` indexes traces and spans for filtering and analytics, and owns the
  records that are not traces: diagnoses, human flags, eval cases and eval runs.

A new SQLite connection is opened per operation. At this project's scale that costs
nothing and sidesteps every threading question FastAPI's worker pool would raise.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from ffx.config import DB_PATH, TRACE_DIR, ensure_dirs
from ffx.models import utcnow
from ffx.tracing.span import Trace

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    started_at        TEXT NOT NULL,
    pipeline_version  TEXT,
    model             TEXT,
    notes             TEXT
);

CREATE TABLE IF NOT EXISTS traces (
    trace_id          TEXT PRIMARY KEY,
    doc_id            TEXT NOT NULL,
    run_id            TEXT,
    status            TEXT NOT NULL,
    started_at        TEXT NOT NULL,
    duration_ms       INTEGER,
    total_tokens      INTEGER,
    pipeline_version  TEXT,
    parent_trace_id   TEXT,
    failure_mode      TEXT,
    is_counterfactual INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_traces_run ON traces(run_id);
CREATE INDEX IF NOT EXISTS ix_traces_doc ON traces(doc_id);

CREATE TABLE IF NOT EXISTS spans (
    span_id           TEXT PRIMARY KEY,
    trace_id          TEXT NOT NULL,
    step_name         TEXT NOT NULL,
    step_index        INTEGER NOT NULL,
    status            TEXT NOT NULL,
    duration_ms       INTEGER,
    self_confidence   INTEGER,
    tokens_in         INTEGER,
    tokens_out        INTEGER,
    attempt_count     INTEGER,
    model             TEXT
);
CREATE INDEX IF NOT EXISTS ix_spans_trace ON spans(trace_id);

CREATE TABLE IF NOT EXISTS diagnoses (
    trace_id          TEXT PRIMARY KEY,
    created_at        TEXT NOT NULL,
    is_failure        INTEGER NOT NULL,
    root_step         TEXT,
    category          TEXT,
    confidence        REAL,
    counterfactual    TEXT,
    diagnose_ms       INTEGER,
    body              TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS flags (
    flag_id           TEXT PRIMARY KEY,
    trace_id          TEXT NOT NULL,
    step_name         TEXT,
    verdict           TEXT NOT NULL,
    human_category    TEXT,
    corrected_output  TEXT,
    note              TEXT,
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_flags_trace ON flags(trace_id);

CREATE TABLE IF NOT EXISTS eval_cases (
    case_id           TEXT PRIMARY KEY,
    source_trace_id   TEXT NOT NULL,
    step_name         TEXT NOT NULL,
    category          TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    body              TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_runs (
    eval_run_id       TEXT PRIMARY KEY,
    started_at        TEXT NOT NULL,
    label             TEXT,
    model             TEXT,
    total             INTEGER,
    passed            INTEGER
);

CREATE TABLE IF NOT EXISTS eval_results (
    eval_run_id       TEXT NOT NULL,
    case_id           TEXT NOT NULL,
    passed            INTEGER NOT NULL,
    detail            TEXT,
    PRIMARY KEY (eval_run_id, case_id)
);
"""


class TraceStore:
    def __init__(self, db_path: Optional[Path] = None, trace_dir: Optional[Path] = None) -> None:
        ensure_dirs()
        self.db_path = Path(db_path or DB_PATH)
        self.trace_dir = Path(trace_dir or TRACE_DIR)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ runs

    def create_run(self, run_id: str, *, pipeline_version: str, model: str, notes: str = "") -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?)",
                (run_id, utcnow().isoformat(), pipeline_version, model, notes),
            )

    def list_runs(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT r.*, COUNT(t.trace_id) AS trace_count
                   FROM runs r LEFT JOIN traces t
                     ON t.run_id = r.run_id AND t.is_counterfactual = 0
                   GROUP BY r.run_id ORDER BY r.started_at DESC"""
            ).fetchall()
        return [dict(r) for r in rows]

    def latest_run_id(self) -> Optional[str]:
        runs = self.list_runs()
        return runs[0]["run_id"] if runs else None

    # ---------------------------------------------------------------- traces

    def _trace_path(self, trace_id: str) -> Path:
        return self.trace_dir / f"{trace_id}.json"

    def save_trace(self, trace: Trace) -> None:
        """Write the JSON file (atomically) and upsert the index rows."""
        path = self._trace_path(trace.trace_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)

        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    trace.trace_id,
                    trace.doc_id,
                    trace.run_id,
                    trace.status.value,
                    trace.started_at.isoformat(),
                    trace.duration_ms,
                    trace.total_tokens,
                    trace.pipeline_version,
                    trace.parent_trace_id,
                    trace.metadata.get("failure_mode"),
                    1 if trace.parent_trace_id else 0,
                ),
            )
            conn.execute("DELETE FROM spans WHERE trace_id = ?", (trace.trace_id,))
            conn.executemany(
                "INSERT INTO spans VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        s.span_id,
                        trace.trace_id,
                        s.step_name.value,
                        s.step_index,
                        s.status.value,
                        s.duration_ms,
                        s.self_confidence,
                        s.tokens_in,
                        s.tokens_out,
                        s.attempt_count,
                        s.model,
                    )
                    for s in trace.spans
                ],
            )

    def load_trace(self, trace_id: str) -> Optional[Trace]:
        path = self._trace_path(trace_id)
        if not path.exists():
            return None
        return Trace.model_validate_json(path.read_text(encoding="utf-8"))

    def list_traces(
        self,
        *,
        run_id: Optional[str] = None,
        include_counterfactual: bool = False,
    ) -> list[dict[str, Any]]:
        clauses, params = [], []
        if run_id:
            clauses.append("t.run_id = ?")
            params.append(run_id)
        if not include_counterfactual:
            clauses.append("t.is_counterfactual = 0")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._conn() as conn:
            rows = conn.execute(
                f"""SELECT t.*, d.is_failure, d.root_step, d.category, d.confidence,
                           d.counterfactual,
                           (SELECT f.verdict FROM flags f WHERE f.trace_id = t.trace_id
                             ORDER BY f.created_at DESC LIMIT 1) AS human_verdict
                    FROM traces t LEFT JOIN diagnoses d ON d.trace_id = t.trace_id
                    {where}
                    ORDER BY t.doc_id""",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------- diagnoses

    def save_diagnosis(self, diagnosis: Any) -> None:
        body = diagnosis.model_dump_json()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO diagnoses VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    diagnosis.trace_id,
                    utcnow().isoformat(),
                    1 if diagnosis.is_failure else 0,
                    diagnosis.root_step.value if diagnosis.root_step else None,
                    diagnosis.category.value,
                    diagnosis.confidence,
                    diagnosis.counterfactual.verdict.value if diagnosis.counterfactual else None,
                    diagnosis.diagnose_ms,
                    body,
                ),
            )

    def load_diagnosis_json(self, trace_id: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute("SELECT body FROM diagnoses WHERE trace_id = ?", (trace_id,)).fetchone()
        return row["body"] if row else None

    def list_diagnosis_rows(self, run_id: Optional[str] = None) -> list[dict[str, Any]]:
        sql = """SELECT d.*, t.run_id, t.doc_id, t.failure_mode FROM diagnoses d
                 JOIN traces t ON t.trace_id = d.trace_id WHERE t.is_counterfactual = 0"""
        params: list[Any] = []
        if run_id:
            sql += " AND t.run_id = ?"
            params.append(run_id)
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    # ----------------------------------------------------------------- flags

    def save_flag(self, flag: dict[str, Any]) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO flags VALUES (?,?,?,?,?,?,?,?)",
                (
                    flag["flag_id"],
                    flag["trace_id"],
                    flag.get("step_name"),
                    flag["verdict"],
                    flag.get("human_category"),
                    json.dumps(flag.get("corrected_output")) if flag.get("corrected_output") is not None else None,
                    flag.get("note", ""),
                    flag.get("created_at") or utcnow().isoformat(),
                ),
            )

    def list_flags(self, trace_id: Optional[str] = None) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM flags", []
        if trace_id:
            sql += " WHERE trace_id = ?"
            params.append(trace_id)
        sql += " ORDER BY created_at DESC"
        with self._conn() as conn:
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
        for row in rows:
            if row.get("corrected_output"):
                row["corrected_output"] = json.loads(row["corrected_output"])
        return rows

    # ----------------------------------------------------------------- evals

    def save_eval_case(self, case: Any) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO eval_cases VALUES (?,?,?,?,?,?)",
                (
                    case.case_id,
                    case.source_trace_id,
                    case.step_name.value,
                    case.category.value,
                    case.created_at.isoformat(),
                    case.model_dump_json(),
                ),
            )

    def list_eval_case_json(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT body FROM eval_cases ORDER BY created_at").fetchall()
        return [r["body"] for r in rows]

    def save_eval_run(self, eval_run_id: str, *, label: str, model: str, results: list[dict[str, Any]]) -> None:
        passed = sum(1 for r in results if r["passed"])
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO eval_runs VALUES (?,?,?,?,?,?)",
                (eval_run_id, utcnow().isoformat(), label, model, len(results), passed),
            )
            conn.executemany(
                "INSERT OR REPLACE INTO eval_results VALUES (?,?,?,?)",
                [(eval_run_id, r["case_id"], 1 if r["passed"] else 0, json.dumps(r.get("detail", {}))) for r in results],
            )

    def list_eval_runs(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM eval_runs ORDER BY started_at").fetchall()]

    def eval_results_matrix(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT r.eval_run_id, r.case_id, r.passed, r.detail, e.started_at, e.label
                   FROM eval_results r JOIN eval_runs e ON e.eval_run_id = r.eval_run_id
                   ORDER BY e.started_at"""
            ).fetchall()
        return [dict(r) for r in rows]
