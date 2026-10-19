"""MLForge SQLite Persistence & Run Lifecycle State Machine Module (Milestone 4.4).

Provides durable local persistence for evaluation run records, strict lifecycle state
transitions, indexed summary metrics, and safe orchestration without coupling to the
scientific evaluation engine.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid
import warnings

from mlforge.core.contracts import (
    AcceptanceDecision,
    ArtifactManifest,
    EvaluationResult,
    LifecycleTransitionEvent,
    RunRecord,
    RunState,
    RunSummaryMetric,
)
from mlforge.core.exceptions import (
    ArtifactSecurityError,
    DuplicateRunError,
    LifecycleTransitionError,
    PersistenceError,
    RunNotFoundError,
)

logger = logging.getLogger("mlforge.application.persistence")

DEFAULT_DB_PATH = Path("artifacts/mlforge.db")

# Strict lifecycle transition table: {source_state: {valid_target_states}}
VALID_TRANSITIONS: Dict[RunState, Set[RunState]] = {
    RunState.CREATED: {RunState.RUNNING, RunState.CANCELLED},
    RunState.RUNNING: {RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED},
    RunState.COMPLETED: set(),  # Terminal state (immutable)
    RunState.FAILED: set(),     # Terminal state (immutable)
    RunState.CANCELLED: set(),  # Terminal state (immutable)
}

TERMINAL_STATES = {RunState.COMPLETED, RunState.FAILED, RunState.CANCELLED}

ALLOWED_ORDER_BY = {
    "created_at",
    "started_at",
    "completed_at",
    "updated_at",
    "run_id",
}


def _utc_now_iso() -> str:
    """Return timezone-aware UTC ISO 8601 formatted timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def _validate_safe_artifact_path(
    path_str: str,
    allowed_root: Optional[Union[str, Path]] = None,
) -> str:
    """Validate that artifact path does not contain null bytes or directory traversal.

    Parameters
    ----------
    path_str : str
        Filesystem path to validate.
    allowed_root : str or Path, optional
        If specified, ensures that the canonical path resolves strictly within this directory.
        If omitted, external paths without traversal components are permitted under the
        external storage trust boundary.

    Returns
    -------
    str
        Normalized POSIX path string.

    Raises
    ------
    ArtifactSecurityError
        If path contains null bytes, directory traversal ('..'), or resolves outside allowed_root.
    """
    if not path_str or "\x00" in path_str:
        raise ArtifactSecurityError(f"Invalid artifact path: {path_str!r}")
    p = Path(path_str)
    if ".." in p.parts:
        raise ArtifactSecurityError(f"Directory traversal detected in artifact path: '{path_str}'")

    if allowed_root is not None:
        root_resolved = Path(allowed_root).resolve()
        p_resolved = p.resolve()
        try:
            p_resolved.relative_to(root_resolved)
        except ValueError:
            raise ArtifactSecurityError(
                f"Artifact path '{path_str}' resolves outside allowed root '{root_resolved}'."
            )

    return str(p.as_posix())


class SQLiteRunRepository:
    """Durable SQLite repository for evaluation runs, metrics, and lifecycle transitions."""

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        timeout: float = 30.0,
        artifact_root: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize repository with target database path.

        Parameters
        ----------
        db_path : str or Path, optional
            Path to SQLite database file or ':memory:'. Defaults to 'artifacts/mlforge.db'.
        timeout : float, default=30.0
            Timeout in seconds for SQLite lock acquisition.
        artifact_root : str or Path, optional
            Root directory to confine artifact references to. If None, safe external paths
            are permitted.
        """
        self.timeout = timeout
        self.artifact_root = Path(artifact_root).resolve() if artifact_root is not None else None

        if db_path is None:
            self.db_path = DEFAULT_DB_PATH.resolve()
            self.is_memory = False
            self._keepalive_conn = None
        elif str(db_path) == ":memory:":
            self.db_path = ":memory:"
            self.is_memory = True
            # Shared in-memory URI with instance isolation; maintain keepalive connection
            # to preserve tables and data across per-operation connection lifecycles.
            self._mem_uri = f"file:mlforge_mem_{id(self)}_{uuid.uuid4().hex}?mode=memory&cache=shared"
            self._keepalive_conn = sqlite3.connect(self._mem_uri, uri=True)
            self._keepalive_conn.execute("PRAGMA foreign_keys = ON;")
        else:
            self.db_path = Path(db_path).resolve()
            self.is_memory = False
            self._keepalive_conn = None

    def _get_connection(self) -> sqlite3.Connection:
        """Create and configure a local SQLite connection."""
        if self.is_memory:
            conn = sqlite3.connect(
                self._mem_uri,
                uri=True,
                timeout=self.timeout,
                isolation_level=None,  # Autocommit mode for explicit transaction control
            )
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=self.timeout,
                isolation_level=None,  # Autocommit mode for explicit transaction control
            )

        conn.row_factory = sqlite3.Row

        # Security and integrity pragmas
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)};")
        if not self.is_memory:
            try:
                conn.execute("PRAGMA journal_mode = WAL;")
            except sqlite3.DatabaseError:
                pass

        return conn

    def close(self) -> None:
        """Close keepalive connection for in-memory databases if open."""
        if hasattr(self, "_keepalive_conn") and self._keepalive_conn is not None:
            try:
                self._keepalive_conn.close()
            except Exception:
                pass
            self._keepalive_conn = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def initialize_schema(self) -> None:
        """Initialize SQLite database schema and indices if they do not exist."""
        conn = self._get_connection()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    updated_at TEXT NOT NULL,
                    dataset_checksum TEXT,
                    evaluation_schema_version TEXT,
                    candidate_model_type TEXT,
                    baseline_model_type TEXT,
                    acceptance_outcome TEXT,
                    artifact_path TEXT,
                    error_category TEXT,
                    error_message TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS run_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    fold_count INTEGER,
                    std_dev REAL,
                    ci_lower REAL,
                    ci_upper REAL
                );

                CREATE TABLE IF NOT EXISTS lifecycle_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    from_state TEXT,
                    to_state TEXT NOT NULL,
                    transitioned_at TEXT NOT NULL,
                    reason TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_runs_state ON runs(state);
                CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
                CREATE INDEX IF NOT EXISTS idx_run_metrics_run_id ON run_metrics(run_id);
                CREATE INDEX IF NOT EXISTS idx_run_metrics_name ON run_metrics(metric_name);
                CREATE INDEX IF NOT EXISTS idx_lifecycle_run_id ON lifecycle_events(run_id);
                """
            )
        except Exception as exc:
            raise PersistenceError(f"Failed to initialize database schema: {exc}") from exc
        finally:
            conn.close()

    def create_run(
        self,
        run_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        dataset_checksum: Optional[str] = None,
    ) -> RunRecord:
        """Create a new run record in the CREATED state.

        Raises
        ------
        DuplicateRunError
            If run_id already exists in the database.
        PersistenceError
            If database insertion fails.
        """
        if not run_id or not run_id.strip():
            raise PersistenceError("Run ID cannot be empty or pure whitespace.")
        clean_run_id = run_id.strip()

        now_utc = _utc_now_iso()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)

        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            # Check for duplicate
            cur = conn.execute("SELECT run_id FROM runs WHERE run_id = ?", (clean_run_id,))
            if cur.fetchone() is not None:
                raise DuplicateRunError(f"Run with ID '{clean_run_id}' already exists.")

            conn.execute(
                """
                INSERT INTO runs (
                    run_id, state, created_at, started_at, completed_at, updated_at,
                    dataset_checksum, evaluation_schema_version, candidate_model_type,
                    baseline_model_type, acceptance_outcome, artifact_path,
                    error_category, error_message, metadata_json
                ) VALUES (
                    ?, ?, ?, NULL, NULL, ?,
                    ?, NULL, NULL,
                    NULL, NULL, NULL,
                    NULL, NULL, ?
                )
                """,
                (
                    clean_run_id,
                    RunState.CREATED.value,
                    now_utc,
                    now_utc,
                    dataset_checksum,
                    meta_json,
                ),
            )

            conn.execute(
                """
                INSERT INTO lifecycle_events (run_id, from_state, to_state, transitioned_at, reason)
                VALUES (?, NULL, ?, ?, ?)
                """,
                (clean_run_id, RunState.CREATED.value, now_utc, "Run record initialized."),
            )

            conn.execute("COMMIT;")
        except DuplicateRunError:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        except Exception as exc:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise PersistenceError(f"Failed to create run '{clean_run_id}': {exc}") from exc
        finally:
            conn.close()

        record = self.get_run(clean_run_id)
        if record is None:
            raise PersistenceError(f"Failed to retrieve run '{clean_run_id}' immediately after creation.")
        return record

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        """Retrieve a run record by its unique identifier."""
        if not run_id:
            return None

        conn = self._get_connection()
        try:
            cur = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id.strip(),))
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_record(row)
        except Exception as exc:
            raise PersistenceError(f"Failed to query run '{run_id}': {exc}") from exc
        finally:
            conn.close()

    def transition_state(
        self,
        run_id: str,
        to_state: RunState,
        reason: Optional[str] = None,
        error_category: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> RunRecord:
        """Transition a run to a new lifecycle state according to state machine rules.

        Raises
        ------
        RunNotFoundError
            If run_id does not exist.
        LifecycleTransitionError
            If the transition is prohibited or the run is in a terminal state.
        PersistenceError
            If database update fails.
        """
        clean_run_id = run_id.strip()
        now_utc = _utc_now_iso()

        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.execute("SELECT * FROM runs WHERE run_id = ?", (clean_run_id,))
            row = cur.fetchone()
            if row is None:
                raise RunNotFoundError(f"Run '{clean_run_id}' not found.")

            current_state = RunState(row["state"])

            # 1. Enforce Terminal State Immutability
            if current_state in TERMINAL_STATES:
                raise LifecycleTransitionError(
                    f"Terminal run '{clean_run_id}' in state '{current_state.value}' "
                    f"cannot transition to '{to_state.value}'. Terminal states are immutable."
                )

            # 2. Enforce Valid Transitions
            allowed_targets = VALID_TRANSITIONS.get(current_state, set())
            if to_state not in allowed_targets:
                raise LifecycleTransitionError(
                    f"Illegal lifecycle transition for run '{clean_run_id}': "
                    f"cannot transition from '{current_state.value}' to '{to_state.value}'. "
                    f"Allowed target states: {[s.value for s in allowed_targets]}"
                )

            # 3. Determine Timestamp Updates
            started_at = row["started_at"]
            if to_state == RunState.RUNNING and started_at is None:
                started_at = now_utc

            completed_at = row["completed_at"]
            if to_state in TERMINAL_STATES:
                completed_at = now_utc

            # 4. Update Run Record
            conn.execute(
                """
                UPDATE runs SET
                    state = ?,
                    started_at = ?,
                    completed_at = ?,
                    updated_at = ?,
                    error_category = coalesce(?, error_category),
                    error_message = coalesce(?, error_message)
                WHERE run_id = ?
                """,
                (
                    to_state.value,
                    started_at,
                    completed_at,
                    now_utc,
                    error_category,
                    error_message,
                    clean_run_id,
                ),
            )

            # 5. Record Audit Event
            conn.execute(
                """
                INSERT INTO lifecycle_events (run_id, from_state, to_state, transitioned_at, reason)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    clean_run_id,
                    current_state.value,
                    to_state.value,
                    now_utc,
                    reason or f"Transitioned from {current_state.value} to {to_state.value}.",
                ),
            )

            conn.execute("COMMIT;")
        except (RunNotFoundError, LifecycleTransitionError):
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        except Exception as exc:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise PersistenceError(f"Failed to transition run '{clean_run_id}' to '{to_state.value}': {exc}") from exc
        finally:
            conn.close()

        record = self.get_run(clean_run_id)
        if record is None:
            raise PersistenceError(f"Failed to reload run '{clean_run_id}' after state transition.")
        return record

    def update_run_artifacts(
        self,
        run_id: str,
        artifact_path: Union[str, Path],
        evaluation: Optional[EvaluationResult] = None,
        decision: Optional[AcceptanceDecision] = None,
        manifest: Optional[ArtifactManifest] = None,
        allowed_root: Optional[Union[str, Path]] = None,
    ) -> RunRecord:
        """Update artifact references and evaluation summary metadata for an active run.

        Raises
        ------
        RunNotFoundError
            If run_id does not exist.
        ArtifactSecurityError
            If artifact_path violates safe path constraints or resolves outside allowed_root.
        PersistenceError
            If database update fails.
        """
        clean_run_id = run_id.strip()
        effective_root = allowed_root if allowed_root is not None else self.artifact_root
        safe_path = _validate_safe_artifact_path(str(artifact_path), allowed_root=effective_root)
        now_utc = _utc_now_iso()

        ds_checksum = None
        eval_schema = None
        cand_model = None
        base_model = None
        acceptance_outcome = None

        if evaluation is not None:
            ds_checksum = evaluation.dataset_metadata.checksum_sha256
            eval_schema = getattr(evaluation, "schema_version", None)
            cand_model = evaluation.candidate_result.model_type
            base_model = evaluation.baseline_result.model_type
            if evaluation.acceptance_decision is not None:
                d = evaluation.acceptance_decision
                acceptance_outcome = d.outcome.value if hasattr(d.outcome, "value") else str(d.outcome)

        if decision is not None:
            acceptance_outcome = decision.outcome.value if hasattr(decision.outcome, "value") else str(decision.outcome)

        if manifest is not None:
            if ds_checksum is None:
                ds_checksum = manifest.dataset_checksum
            if eval_schema is None:
                eval_schema = manifest.schema_version
            if cand_model is None:
                cand_model = manifest.candidate_model_type
            if base_model is None:
                base_model = manifest.baseline_model_type

        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.execute("SELECT run_id FROM runs WHERE run_id = ?", (clean_run_id,))
            if cur.fetchone() is None:
                raise RunNotFoundError(f"Run '{clean_run_id}' not found.")

            conn.execute(
                """
                UPDATE runs SET
                    artifact_path = ?,
                    dataset_checksum = coalesce(?, dataset_checksum),
                    evaluation_schema_version = coalesce(?, evaluation_schema_version),
                    candidate_model_type = coalesce(?, candidate_model_type),
                    baseline_model_type = coalesce(?, baseline_model_type),
                    acceptance_outcome = coalesce(?, acceptance_outcome),
                    updated_at = ?
                WHERE run_id = ?
                """,
                (
                    safe_path,
                    ds_checksum,
                    eval_schema,
                    cand_model,
                    base_model,
                    acceptance_outcome,
                    now_utc,
                    clean_run_id,
                ),
            )
            conn.execute("COMMIT;")
        except RunNotFoundError:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        except Exception as exc:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise PersistenceError(f"Failed to update artifacts for run '{clean_run_id}': {exc}") from exc
        finally:
            conn.close()

        record = self.get_run(clean_run_id)
        if record is None:
            raise PersistenceError(f"Failed to reload run '{clean_run_id}' after artifact update.")
        return record

    def record_summary_metrics(self, run_id: str, metrics: List[RunSummaryMetric]) -> None:
        """Insert indexed summary metrics for a completed or evaluated run."""
        if not metrics:
            return
        clean_run_id = run_id.strip()

        conn = self._get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.execute("SELECT run_id FROM runs WHERE run_id = ?", (clean_run_id,))
            if cur.fetchone() is None:
                raise RunNotFoundError(f"Run '{clean_run_id}' not found.")

            # Replace existing metrics for this run to keep idempotent
            conn.execute("DELETE FROM run_metrics WHERE run_id = ?", (clean_run_id,))

            rows = [
                (
                    clean_run_id,
                    m.metric_name,
                    m.metric_value,
                    m.fold_count,
                    m.std_dev,
                    m.ci_lower,
                    m.ci_upper,
                )
                for m in metrics
            ]
            conn.executemany(
                """
                INSERT INTO run_metrics (
                    run_id, metric_name, metric_value, fold_count, std_dev, ci_lower, ci_upper
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.execute("COMMIT;")
        except RunNotFoundError:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise
        except Exception as exc:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise PersistenceError(f"Failed to record summary metrics for run '{clean_run_id}': {exc}") from exc
        finally:
            conn.close()

    def get_run_metrics(self, run_id: str) -> List[RunSummaryMetric]:
        """Retrieve all indexed summary metrics for a given run."""
        clean_run_id = run_id.strip()
        conn = self._get_connection()
        try:
            cur = conn.execute(
                """
                SELECT metric_name, metric_value, fold_count, std_dev, ci_lower, ci_upper
                FROM run_metrics
                WHERE run_id = ?
                ORDER BY metric_name ASC
                """,
                (clean_run_id,),
            )
            return [
                RunSummaryMetric(
                    metric_name=r["metric_name"],
                    metric_value=r["metric_value"],
                    fold_count=r["fold_count"],
                    std_dev=r["std_dev"],
                    ci_lower=r["ci_lower"],
                    ci_upper=r["ci_upper"],
                )
                for r in cur.fetchall()
            ]
        except Exception as exc:
            raise PersistenceError(f"Failed to query metrics for run '{clean_run_id}': {exc}") from exc
        finally:
            conn.close()

    def get_lifecycle_history(self, run_id: str) -> List[LifecycleTransitionEvent]:
        """Retrieve complete audit history of lifecycle state transitions for a run."""
        clean_run_id = run_id.strip()
        conn = self._get_connection()
        try:
            cur = conn.execute(
                """
                SELECT run_id, from_state, to_state, transitioned_at, reason
                FROM lifecycle_events
                WHERE run_id = ?
                ORDER BY id ASC
                """,
                (clean_run_id,),
            )
            events = []
            for r in cur.fetchall():
                from_st = RunState(r["from_state"]) if r["from_state"] else None
                to_st = RunState(r["to_state"])
                events.append(
                    LifecycleTransitionEvent(
                        run_id=r["run_id"],
                        from_state=from_st,
                        to_state=to_st,
                        transitioned_at=r["transitioned_at"],
                        reason=r["reason"],
                    )
                )
            return events
        except Exception as exc:
            raise PersistenceError(f"Failed to query lifecycle history for run '{clean_run_id}': {exc}") from exc
        finally:
            conn.close()

    def list_runs(
        self,
        state: Optional[RunState] = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "created_at",
        descending: bool = True,
    ) -> List[RunRecord]:
        """List runs with stable ordering and pagination."""
        if order_by not in ALLOWED_ORDER_BY:
            raise PersistenceError(f"Invalid order_by column '{order_by}'. Allowed: {ALLOWED_ORDER_BY}")

        direction = "DESC" if descending else "ASC"
        order_clause = f"ORDER BY {order_by} {direction}, run_id ASC"

        params: List[Any] = []
        where_clause = ""
        if state is not None:
            where_clause = "WHERE state = ?"
            params.append(state.value)

        params.extend([max(1, limit), max(0, offset)])

        query = f"SELECT * FROM runs {where_clause} {order_clause} LIMIT ? OFFSET ?"

        conn = self._get_connection()
        try:
            cur = conn.execute(query, params)
            return [self._row_to_record(row) for row in cur.fetchall()]
        except Exception as exc:
            raise PersistenceError(f"Failed to list runs: {exc}") from exc
        finally:
            conn.close()

    def count_runs(self, state: Optional[RunState] = None) -> int:
        """Count total run records, optionally filtered by state."""
        conn = self._get_connection()
        try:
            if state is not None:
                cur = conn.execute("SELECT count(*) as cnt FROM runs WHERE state = ?", (state.value,))
            else:
                cur = conn.execute("SELECT count(*) as cnt FROM runs")
            row = cur.fetchone()
            return int(row["cnt"]) if row else 0
        except Exception as exc:
            raise PersistenceError(f"Failed to count runs: {exc}") from exc
        finally:
            conn.close()

    def recover_stale_runs(self, timeout_seconds: float = 3600.0) -> int:
        """Recover runs stuck in RUNNING state beyond timeout by marking them FAILED.

        Operational Limitation & Liveness Semantics
        -------------------------------------------
        This recovery mechanism relies strictly on elapsed wall-clock time
        (now - updated_at >= timeout_seconds). It cannot inspect operating system
        processes or prove process termination. If invoked with an unsuitable or overly
        aggressive timeout, it may misclassify a long-running, active model evaluation or
        hyperparameter sweep as FAILED. Callers should choose a timeout_seconds value that
        safely exceeds the maximum expected single-run execution duration.

        Parameters
        ----------
        timeout_seconds : float, default=3600.0 (1 hour)
            Maximum permissible elapsed duration in RUNNING state.

        Returns
        -------
        int
            Count of recovered runs transitioned to FAILED.
        """
        now = datetime.now(timezone.utc)
        now_utc = now.isoformat()

        conn = self._get_connection()
        recovered_count = 0
        try:
            conn.execute("BEGIN IMMEDIATE;")
            cur = conn.execute("SELECT run_id, updated_at FROM runs WHERE state = ?", (RunState.RUNNING.value,))
            running_rows = cur.fetchall()

            for row in running_rows:
                r_id = row["run_id"]
                upd_str = row["updated_at"]
                try:
                    upd_dt = datetime.fromisoformat(upd_str)
                    elapsed = (now - upd_dt).total_seconds()
                except Exception:
                    elapsed = timeout_seconds + 1.0

                if elapsed >= timeout_seconds:
                    conn.execute(
                        """
                        UPDATE runs SET
                            state = ?,
                            completed_at = ?,
                            updated_at = ?,
                            error_category = ?,
                            error_message = ?
                        WHERE run_id = ?
                        """,
                        (
                            RunState.FAILED.value,
                            now_utc,
                            now_utc,
                            "INTERRUPTED_CRASH",
                            f"Run timed out after remaining in RUNNING state for {elapsed:.1f}s (threshold: {timeout_seconds}s).",
                            r_id,
                        ),
                    )
                    conn.execute(
                        """
                        INSERT INTO lifecycle_events (run_id, from_state, to_state, transitioned_at, reason)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            r_id,
                            RunState.RUNNING.value,
                            RunState.FAILED.value,
                            now_utc,
                            "Automated stale-run recovery: execution timed out or process was interrupted.",
                        ),
                    )
                    recovered_count += 1

            conn.execute("COMMIT;")
            return recovered_count
        except Exception as exc:
            try:
                conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise PersistenceError(f"Failed to recover stale runs: {exc}") from exc
        finally:
            conn.close()

    def _row_to_record(self, row: sqlite3.Row) -> RunRecord:
        """Convert a sqlite3.Row into a validated RunRecord contract."""
        meta_dict = {}
        if row["metadata_json"]:
            try:
                meta_dict = json.loads(row["metadata_json"])
            except Exception:
                meta_dict = {}

        return RunRecord(
            run_id=row["run_id"],
            state=RunState(row["state"]),
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            updated_at=row["updated_at"],
            dataset_checksum=row["dataset_checksum"],
            evaluation_schema_version=row["evaluation_schema_version"],
            candidate_model_type=row["candidate_model_type"],
            baseline_model_type=row["baseline_model_type"],
            acceptance_outcome=row["acceptance_outcome"],
            artifact_path=row["artifact_path"],
            error_category=row["error_category"],
            error_message=row["error_message"],
            metadata=meta_dict,
        )


def execute_managed_evaluation(
    df: Any,
    target_column: str,
    candidate_hyperparameters: Optional[Dict[str, Any]] = None,
    num_folds: int = 5,
    random_state: int = 42,
    subgroup_columns: Optional[List[str]] = None,
    metadata_df: Optional[Any] = None,
    acceptance_policy: Optional[Any] = None,
    artifacts_dir: Union[str, Path] = "artifacts/runs",
    repository: Optional[SQLiteRunRepository] = None,
    run_id: Optional[str] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[RunRecord, EvaluationResult, Path]:
    """Execute evaluation with durable SQLite lifecycle tracking and atomic artifact serialization.

    This application-level orchestrator coordinates the evaluation pipeline while maintaining
    strict separation between the pure evaluation engine and SQLite persistence.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset for evaluation.
    target_column : str
        Column name of classification target.
    candidate_hyperparameters : dict, optional
        Hyperparameters for candidate classifier.
    num_folds : int, default=5
        Cross-validation fold count.
    random_state : int, default=42
        Master random seed.
    subgroup_columns : list of str, optional
        Columns to evaluate for demographic parity / slice diagnostics.
    metadata_df : pandas.DataFrame, optional
        Supplementary metadata dataframe aligned with df.
    acceptance_policy : AcceptancePolicy, optional
        Policy governing statistical acceptance.
    artifacts_dir : str or Path, default="artifacts/runs"
        Filesystem directory for serialized run evidence.
    repository : SQLiteRunRepository, optional
        Durable persistence repository. If None, default repository is initialized.
    run_id : str, optional
        Explicit run ID. If None, generated from timestamp.
    run_metadata : dict, optional
        User tags or metadata to persist with the run record.

    Returns
    -------
    Tuple[RunRecord, EvaluationResult, Path]
        (completed_run_record, evaluation_result, artifact_directory_path)

    Raises
    ------
    MLForgeError
        If dataset validation, evaluation, or persistence fails.
    """
    import uuid
    from mlforge.application.artifacts import serialize_run_artifacts
    from mlforge.core.exceptions import (
        ArtifactSerializationError,
        ConfigurationError,
        DataLeakageError,
        DatasetValidationError,
    )
    from mlforge.engine.evaluator import evaluate_classification

    # Initialize repository if omitted
    if repository is None:
        repository = SQLiteRunRepository()
        repository.initialize_schema()

    # Generate or sanitize run_id
    if run_id is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        active_run_id = f"run_{ts}_{uuid.uuid4().hex[:8]}"
    else:
        active_run_id = run_id.strip()

    # 1. Create run record in CREATED state
    record = repository.create_run(
        run_id=active_run_id,
        metadata=run_metadata,
    )

    # 2. Transition to RUNNING state
    record = repository.transition_state(
        run_id=active_run_id,
        to_state=RunState.RUNNING,
        reason="Execution started.",
    )

    try:
        # 3. Invoke pure scientific evaluation engine (zero SQLite dependencies)
        evaluation = evaluate_classification(
            df=df,
            target_column=target_column,
            num_folds=num_folds,
            candidate_hyperparameters=candidate_hyperparameters,
            random_state=random_state,
            subgroup_columns=subgroup_columns,
            metadata_df=metadata_df,
            acceptance_policy=acceptance_policy,
        )

        # 4. Serialize run artifacts atomically to filesystem
        cand_pipe = getattr(evaluation.candidate_result, "fitted_pipeline", None)
        base_pipe = getattr(evaluation.baseline_result, "fitted_pipeline", None)

        artifact_path = serialize_run_artifacts(
            evaluation=evaluation,
            acceptance_decision=evaluation.acceptance_decision,
            artifacts_dir=artifacts_dir,
            run_id=active_run_id,
            policy=acceptance_policy,
            candidate_pipeline=cand_pipe,
            baseline_pipeline=base_pipe,
        )

        # 5. Update run record with artifact references and evidence metadata
        record = repository.update_run_artifacts(
            run_id=active_run_id,
            artifact_path=artifact_path,
            evaluation=evaluation,
            decision=evaluation.acceptance_decision,
        )

        # 6. Index summary metrics
        summary_metrics = []
        if evaluation.candidate_result and evaluation.candidate_result.metric_summaries:
            for m_name, m_sum in evaluation.candidate_result.metric_summaries.items():
                ci_l = m_sum.confidence_interval_95[0] if m_sum.confidence_interval_95 else None
                ci_u = m_sum.confidence_interval_95[1] if m_sum.confidence_interval_95 else None
                summary_metrics.append(
                    RunSummaryMetric(
                        metric_name=m_name,
                        metric_value=m_sum.mean,
                        fold_count=m_sum.valid_fold_count,
                        std_dev=m_sum.std,
                        ci_lower=ci_l,
                        ci_upper=ci_u,
                    )
                )
        if summary_metrics:
            repository.record_summary_metrics(active_run_id, summary_metrics)

        # 7. Transition to COMPLETED
        record = repository.transition_state(
            run_id=active_run_id,
            to_state=RunState.COMPLETED,
            reason="Evaluation, artifact serialization, and metric indexing completed successfully.",
        )

        return record, evaluation, artifact_path

    except Exception as exc:
        # Categorize failure safely without leaking secrets or full stack traces
        if isinstance(exc, DatasetValidationError):
            cat = "DATASET_VALIDATION_ERROR"
        elif isinstance(exc, DataLeakageError):
            cat = "DATA_LEAKAGE_ERROR"
        elif isinstance(exc, ConfigurationError):
            cat = "CONFIGURATION_ERROR"
        elif isinstance(exc, ArtifactSerializationError):
            cat = "ARTIFACT_SERIALIZATION_ERROR"
        else:
            cat = "EXECUTION_ERROR"

        safe_msg = str(exc)[:500].strip() or f"Execution failed with {type(exc).__name__}"

        try:
            repository.transition_state(
                run_id=active_run_id,
                to_state=RunState.FAILED,
                reason="Execution failed due to error.",
                error_category=cat,
                error_message=safe_msg,
            )
        except Exception as persist_exc:
            safe_persist_err = str(persist_exc)[:200].strip() or type(persist_exc).__name__
            warn_msg = (
                f"Failed to record FAILED state for run '{active_run_id}' in persistence repository: "
                f"{type(persist_exc).__name__} ({safe_persist_err}). Original error will be re-raised."
            )
            warnings.warn(warn_msg, UserWarning, stacklevel=2)
            logger.warning(warn_msg)

        raise
