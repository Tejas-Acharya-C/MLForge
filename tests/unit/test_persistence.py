"""Unit Tests for SQLite Persistence & Run Lifecycle State Machine (Milestone 4.4).

Verifies database schema creation, atomic transactions, foreign-key enforcement,
lifecycle transitions, terminal-state immutability, safe error recording,
paginated listing, stale run recovery, and application orchestration.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import pytest
import pandas as pd

from mlforge.application.persistence import (
    DEFAULT_DB_PATH,
    SQLiteRunRepository,
    execute_managed_evaluation,
)
from mlforge.core.contracts import (
    AcceptanceOutcome,
    AcceptancePolicy,
    AcceptanceRule,
    CriterionSeverity,
    LifecycleTransitionEvent,
    RuleOperator,
    RunRecord,
    RunState,
    RunSummaryMetric,
)
from mlforge.core.exceptions import (
    ArtifactSecurityError,
    DatasetValidationError,
    DuplicateRunError,
    LifecycleTransitionError,
    PersistenceError,
    RunNotFoundError,
)


@pytest.fixture
def temp_repo(tmp_path: Path) -> SQLiteRunRepository:
    """Fixture providing an initialized SQLiteRunRepository backed by a temporary file."""
    db_file = tmp_path / "test_mlforge.db"
    repo = SQLiteRunRepository(db_path=db_file)
    repo.initialize_schema()
    return repo


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Fixture providing a simple valid tabular dataset for evaluation."""
    return pd.DataFrame({
        "num_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "num_2": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        "cat_1": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


def test_database_initialization_and_repeat_initialization(tmp_path: Path):
    """Database schema initialization is idempotent and creates all required tables and indexes."""
    db_file = tmp_path / "idempotent.db"
    repo = SQLiteRunRepository(db_path=db_file)

    # First initialization
    repo.initialize_schema()
    assert db_file.exists()

    # Repeat initialization should not error or alter schema
    repo.initialize_schema()

    # Verify tables exist
    conn = repo._get_connection()
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = {row["name"] for row in cur.fetchall()}
    conn.close()

    assert "runs" in tables
    assert "run_metrics" in tables
    assert "lifecycle_events" in tables


def test_run_creation_and_duplicate_rejection(temp_repo: SQLiteRunRepository):
    """Creating a run record initializes state to CREATED and rejects duplicate IDs."""
    record = temp_repo.create_run(
        run_id="run_test_001",
        metadata={"user": "alice", "experiment": "baseline"},
        dataset_checksum="abc123sha256",
    )

    assert record.run_id == "run_test_001"
    assert record.state == RunState.CREATED
    assert record.dataset_checksum == "abc123sha256"
    assert record.metadata["user"] == "alice"
    assert record.started_at is None
    assert record.completed_at is None
    assert record.created_at == record.updated_at

    # Attempting to create duplicate run_id must raise DuplicateRunError
    with pytest.raises(DuplicateRunError, match="already exists"):
        temp_repo.create_run(run_id="run_test_001")


def test_run_retrieval_and_nonexistent(temp_repo: SQLiteRunRepository):
    """Retrieving an existing run returns RunRecord; non-existent returns None."""
    temp_repo.create_run("run_exist_1")

    rec = temp_repo.get_run("run_exist_1")
    assert rec is not None
    assert rec.run_id == "run_exist_1"

    missing = temp_repo.get_run("run_does_not_exist")
    assert missing is None


def test_valid_lifecycle_transitions(temp_repo: SQLiteRunRepository):
    """Run transitions smoothly through the valid lifecycle path CREATED -> RUNNING -> COMPLETED."""
    run_id = "lifecycle_run_valid"
    temp_repo.create_run(run_id)

    # 1. Transition to RUNNING
    r_running = temp_repo.transition_state(run_id, RunState.RUNNING, reason="Pipeline started.")
    assert r_running.state == RunState.RUNNING
    assert r_running.started_at is not None
    assert r_running.completed_at is None

    # 2. Transition to COMPLETED
    r_completed = temp_repo.transition_state(run_id, RunState.COMPLETED, reason="Evaluation done.")
    assert r_completed.state == RunState.COMPLETED
    assert r_completed.completed_at is not None
    assert r_completed.updated_at >= r_completed.started_at


def test_invalid_lifecycle_transitions(temp_repo: SQLiteRunRepository):
    """Attempting invalid transitions raises LifecycleTransitionError."""
    run_id = "lifecycle_invalid_trans"
    temp_repo.create_run(run_id)

    # Cannot jump directly from CREATED to COMPLETED
    with pytest.raises(LifecycleTransitionError, match="Illegal lifecycle transition"):
        temp_repo.transition_state(run_id, RunState.COMPLETED)

    # Cannot jump from CREATED to FAILED
    with pytest.raises(LifecycleTransitionError, match="Illegal lifecycle transition"):
        temp_repo.transition_state(run_id, RunState.FAILED)


def test_terminal_state_immutability(temp_repo: SQLiteRunRepository):
    """Terminal states (COMPLETED, FAILED, CANCELLED) cannot transition to any other state."""
    # Test COMPLETED
    temp_repo.create_run("run_term_completed")
    temp_repo.transition_state("run_term_completed", RunState.RUNNING)
    temp_repo.transition_state("run_term_completed", RunState.COMPLETED)

    with pytest.raises(LifecycleTransitionError, match="Terminal run.*cannot transition"):
        temp_repo.transition_state("run_term_completed", RunState.RUNNING)

    with pytest.raises(LifecycleTransitionError, match="Terminal run.*cannot transition"):
        temp_repo.transition_state("run_term_completed", RunState.FAILED)

    # Test FAILED
    temp_repo.create_run("run_term_failed")
    temp_repo.transition_state("run_term_failed", RunState.RUNNING)
    temp_repo.transition_state("run_term_failed", RunState.FAILED)

    with pytest.raises(LifecycleTransitionError, match="Terminal run.*cannot transition"):
        temp_repo.transition_state("run_term_failed", RunState.RUNNING)

    # Test CANCELLED
    temp_repo.create_run("run_term_cancelled")
    temp_repo.transition_state("run_term_cancelled", RunState.CANCELLED)

    with pytest.raises(LifecycleTransitionError, match="Terminal run.*cannot transition"):
        temp_repo.transition_state("run_term_cancelled", RunState.RUNNING)


def test_lifecycle_history_audit_logging(temp_repo: SQLiteRunRepository):
    """Every transition is logged sequentially with accurate from_state, to_state, and timestamps."""
    run_id = "audit_log_run"
    temp_repo.create_run(run_id)
    temp_repo.transition_state(run_id, RunState.RUNNING, reason="Started worker.")
    temp_repo.transition_state(run_id, RunState.CANCELLED, reason="User cancelled.")

    history = temp_repo.get_lifecycle_history(run_id)
    assert len(history) == 3

    assert history[0].from_state is None
    assert history[0].to_state == RunState.CREATED

    assert history[1].from_state == RunState.CREATED
    assert history[1].to_state == RunState.RUNNING
    assert history[1].reason == "Started worker."

    assert history[2].from_state == RunState.RUNNING
    assert history[2].to_state == RunState.CANCELLED
    assert history[2].reason == "User cancelled."


def test_safe_error_recording(temp_repo: SQLiteRunRepository):
    """Failure transitions record structured error category and message without sensitive data."""
    run_id = "fail_run_01"
    temp_repo.create_run(run_id)
    temp_repo.transition_state(run_id, RunState.RUNNING)

    err_msg = "Insufficient positive class samples in fold 2."
    record = temp_repo.transition_state(
        run_id=run_id,
        to_state=RunState.FAILED,
        error_category="DATASET_VALIDATION_ERROR",
        error_message=err_msg,
    )

    assert record.state == RunState.FAILED
    assert record.error_category == "DATASET_VALIDATION_ERROR"
    assert record.error_message == err_msg
    assert record.completed_at is not None


def test_summary_metrics_persistence_and_query(temp_repo: SQLiteRunRepository):
    """Cross-validation summary metrics are indexed and queryable by run ID."""
    run_id = "metrics_run_01"
    temp_repo.create_run(run_id)

    metrics = [
        RunSummaryMetric(metric_name="macro_f1", metric_value=0.885, fold_count=5, std_dev=0.02, ci_lower=0.85, ci_upper=0.91),
        RunSummaryMetric(metric_name="accuracy", metric_value=0.910, fold_count=5, std_dev=0.015, ci_lower=0.89, ci_upper=0.93),
        RunSummaryMetric(metric_name="brier_score", metric_value=0.082, fold_count=5, std_dev=0.005, ci_lower=0.07, ci_upper=0.09),
    ]

    temp_repo.record_summary_metrics(run_id, metrics)

    loaded_metrics = temp_repo.get_run_metrics(run_id)
    assert len(loaded_metrics) == 3
    metric_map = {m.metric_name: m.metric_value for m in loaded_metrics}
    assert metric_map["macro_f1"] == 0.885
    assert metric_map["accuracy"] == 0.910


def test_foreign_key_enforcement(temp_repo: SQLiteRunRepository):
    """Inserting metrics or events for non-existent run_id violates foreign key constraints."""
    conn = temp_repo._get_connection()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO run_metrics (run_id, metric_name, metric_value)
            VALUES ('non_existent_run', 'accuracy', 0.9)
            """
        )
    conn.close()


def test_paginated_and_ordered_run_listing(temp_repo: SQLiteRunRepository):
    """Listing runs supports stable pagination and state filtering."""
    for i in range(15):
        r_id = f"list_run_{i:02d}"
        temp_repo.create_run(r_id)
        if i % 2 == 0:
            temp_repo.transition_state(r_id, RunState.RUNNING)
            temp_repo.transition_state(r_id, RunState.COMPLETED)

    # Total count
    assert temp_repo.count_runs() == 15
    assert temp_repo.count_runs(state=RunState.COMPLETED) == 8
    assert temp_repo.count_runs(state=RunState.CREATED) == 7

    # Pagination: first page (limit 5)
    page_1 = temp_repo.list_runs(limit=5, offset=0)
    assert len(page_1) == 5

    # Pagination: second page (limit 5, offset 5)
    page_2 = temp_repo.list_runs(limit=5, offset=5)
    assert len(page_2) == 5
    # Ensure distinct runs across pages
    ids_1 = {r.run_id for r in page_1}
    ids_2 = {r.run_id for r in page_2}
    assert ids_1.isdisjoint(ids_2)

    # Filter by state
    completed_runs = temp_repo.list_runs(state=RunState.COMPLETED, limit=20)
    assert len(completed_runs) == 8
    assert all(r.state == RunState.COMPLETED for r in completed_runs)


def test_artifact_path_validation_and_traversal_rejection(temp_repo: SQLiteRunRepository):
    """Artifact path updates validate path safety and reject directory traversal."""
    run_id = "artifact_path_run"
    temp_repo.create_run(run_id)

    # Valid safe relative path
    rec = temp_repo.update_run_artifacts(run_id, artifact_path="artifacts/runs/run_01")
    assert rec.artifact_path == "artifacts/runs/run_01"

    # Traversal attempt must be rejected
    with pytest.raises(ArtifactSecurityError, match="Directory traversal"):
        temp_repo.update_run_artifacts(run_id, artifact_path="../../etc/shadow")

    with pytest.raises(ArtifactSecurityError, match="Invalid artifact path"):
        temp_repo.update_run_artifacts(run_id, artifact_path="artifacts/runs/\x00malicious")


def test_stale_run_recovery(temp_repo: SQLiteRunRepository):
    """Runs stuck in RUNNING state beyond timeout are recovered to FAILED."""
    r_normal = "run_fresh"
    r_stale = "run_stale"

    temp_repo.create_run(r_normal)
    temp_repo.transition_state(r_normal, RunState.RUNNING)

    temp_repo.create_run(r_stale)
    temp_repo.transition_state(r_stale, RunState.RUNNING)

    # Backdate updated_at for r_stale by 2 hours
    old_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    conn = temp_repo._get_connection()
    conn.execute("UPDATE runs SET updated_at = ? WHERE run_id = ?", (old_time, r_stale))
    conn.close()

    # Recover with 1 hour (3600s) threshold
    recovered = temp_repo.recover_stale_runs(timeout_seconds=3600.0)
    assert recovered == 1

    # Check stale run is now FAILED
    stale_rec = temp_repo.get_run(r_stale)
    assert stale_rec.state == RunState.FAILED
    assert stale_rec.error_category == "INTERRUPTED_CRASH"
    assert "timed out" in stale_rec.error_message

    # Check fresh run is still RUNNING
    normal_rec = temp_repo.get_run(r_normal)
    assert normal_rec.state == RunState.RUNNING


def test_execute_managed_evaluation_success(temp_repo: SQLiteRunRepository, sample_dataframe: pd.DataFrame, tmp_path: Path):
    """Managed evaluation orchestrator runs evaluation, writes artifacts, and marks run COMPLETED."""
    artifacts_dir = tmp_path / "artifacts" / "runs"
    run_id = "managed_success_01"

    policy = AcceptancePolicy(
        policy_id="test_managed_pol",
        name="Test Managed Policy",
        version="1.0.0",
        rules=[
            AcceptanceRule(
                rule_id="r1",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.10,
                severity=CriterionSeverity.BLOCKER,
            )
        ],
    )

    record, evaluation, art_path = execute_managed_evaluation(
        df=sample_dataframe,
        target_column="target",
        categorical_columns=["cat_1"],
        numeric_columns=["num_1", "num_2"],
        num_folds=2,
        random_state=42,
        acceptance_policy=policy,
        artifacts_dir=artifacts_dir,
        repository=temp_repo,
        run_id=run_id,
        run_metadata={"author": "auditor"},
    )

    # Run record must be COMPLETED
    assert record.state == RunState.COMPLETED
    assert record.run_id == run_id
    assert record.completed_at is not None
    assert record.acceptance_outcome == "PASS"
    assert record.candidate_model_type == "logistic_regression"
    assert record.dataset_checksum is not None

    # Artifacts must exist on disk
    assert art_path.exists()
    assert (art_path / "evaluation.json").exists()
    assert (art_path / "manifest.json").exists()
    assert (art_path / "decision.json").exists()

    # Summary metrics must be indexed in SQLite
    metrics = temp_repo.get_run_metrics(run_id)
    assert len(metrics) > 0
    metric_names = {m.metric_name for m in metrics}
    assert "macro_f1" in metric_names


def test_execute_managed_evaluation_failure_fails_closed(temp_repo: SQLiteRunRepository, sample_dataframe: pd.DataFrame, tmp_path: Path):
    """Managed evaluation failure marks run FAILED with structured error without corrupted records."""
    artifacts_dir = tmp_path / "artifacts" / "runs"
    run_id = "managed_fail_01"

    # Missing target column triggers DatasetValidationError
    with pytest.raises(DatasetValidationError):
        execute_managed_evaluation(
            df=sample_dataframe,
            target_column="non_existent_column",
            num_folds=2,
            artifacts_dir=artifacts_dir,
            repository=temp_repo,
            run_id=run_id,
        )

    # Check run in DB was transitioned to FAILED
    record = temp_repo.get_run(run_id)
    assert record is not None
    assert record.state == RunState.FAILED
    assert record.error_category == "DATASET_VALIDATION_ERROR"
    assert record.completed_at is not None
    assert "not found" in record.error_message


def test_persistence_isolation_evaluation_engine_has_zero_persistence_imports():
    """Verify that pure evaluation engine modules do not import persistence or sqlite3."""
    import sys
    from mlforge.engine import evaluator, metrics, pipeline, protocol, reliability

    for mod in (evaluator, metrics, pipeline, protocol, reliability):
        mod_dict = mod.__dict__
        assert "sqlite3" not in mod_dict
        assert "SQLiteRunRepository" not in mod_dict
        assert "execute_managed_evaluation" not in mod_dict


def test_in_memory_repository_initialization_and_operations():
    """Verify in-memory ':memory:' SQLite repository maintains state across operations."""
    repo = SQLiteRunRepository(":memory:")
    repo.initialize_schema()

    rec = repo.create_run("mem_run_01", metadata={"mode": "in_memory"})
    assert rec.run_id == "mem_run_01"
    assert rec.state == RunState.CREATED

    rec_run = repo.transition_state("mem_run_01", RunState.RUNNING, reason="Started test.")
    assert rec_run.state == RunState.RUNNING

    fetched = repo.get_run("mem_run_01")
    assert fetched is not None
    assert fetched.run_id == "mem_run_01"
    assert fetched.state == RunState.RUNNING
    assert fetched.metadata["mode"] == "in_memory"

    repo.close()


def test_managed_evaluation_secondary_persistence_failure_emits_observable_warning(
    temp_repo: SQLiteRunRepository,
    sample_dataframe: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """When recording FAILED state also fails, primary error is re-raised and warning is emitted."""
    artifacts_dir = tmp_path / "artifacts" / "runs"
    run_id = "managed_sec_fail_01"

    # Monkeypatch transition_state on temp_repo to simulate database write failure when moving to FAILED
    orig_transition = temp_repo.transition_state

    def faulty_transition(run_id: str, to_state: RunState, **kwargs):
        if to_state == RunState.FAILED:
            raise PersistenceError("Simulated disk I/O error during failure transition.")
        return orig_transition(run_id, to_state, **kwargs)

    monkeypatch.setattr(temp_repo, "transition_state", faulty_transition)

    with pytest.warns(UserWarning, match="Failed to record FAILED state for run"):
        with pytest.raises(DatasetValidationError):
            execute_managed_evaluation(
                df=sample_dataframe,
                target_column="non_existent_column",
                num_folds=2,
                artifacts_dir=artifacts_dir,
                repository=temp_repo,
                run_id=run_id,
            )


def test_artifact_path_confinement_with_allowed_root(tmp_path: Path):
    """Verify artifact path confinement when allowed_root is enforced vs omitted."""
    root_dir = tmp_path / "allowed_artifacts"
    root_dir.mkdir(parents=True)
    outside_dir = tmp_path / "external_forbidden"
    outside_dir.mkdir(parents=True)

    repo = SQLiteRunRepository(":memory:", artifact_root=root_dir)
    repo.initialize_schema()
    repo.create_run("run_confine_01")

    # Path inside root succeeds
    valid_path = root_dir / "run_confine_01"
    rec = repo.update_run_artifacts("run_confine_01", artifact_path=valid_path)
    assert rec.artifact_path is not None

    # Path outside root raises ArtifactSecurityError
    forbidden_path = outside_dir / "run_leak"
    with pytest.raises(ArtifactSecurityError, match="resolves outside allowed root"):
        repo.update_run_artifacts("run_confine_01", artifact_path=forbidden_path)

    # When artifact_root is None, external paths without traversal are permitted
    permissive_repo = SQLiteRunRepository(":memory:")
    permissive_repo.initialize_schema()
    permissive_repo.create_run("run_permissive_01")
    rec_perm = permissive_repo.update_run_artifacts("run_permissive_01", artifact_path=forbidden_path)
    assert rec_perm.artifact_path is not None

    repo.close()
    permissive_repo.close()


def test_stale_run_recovery_exact_cutoff_and_transactional_history(temp_repo: SQLiteRunRepository):
    """Verify exact cutoff threshold and transactional lifecycle event logging in stale recovery."""
    r_fresh = "run_cutoff_fresh"
    r_stale = "run_cutoff_stale"

    temp_repo.create_run(r_fresh)
    temp_repo.transition_state(r_fresh, RunState.RUNNING)

    temp_repo.create_run(r_stale)
    temp_repo.transition_state(r_stale, RunState.RUNNING)

    now = datetime.now(timezone.utc)
    # 30 seconds ago (fresh for 60s timeout)
    fresh_time = (now - timedelta(seconds=30)).isoformat()
    # 90 seconds ago (stale for 60s timeout)
    stale_time = (now - timedelta(seconds=90)).isoformat()

    conn = temp_repo._get_connection()
    conn.execute("UPDATE runs SET updated_at = ? WHERE run_id = ?", (fresh_time, r_fresh))
    conn.execute("UPDATE runs SET updated_at = ? WHERE run_id = ?", (stale_time, r_stale))
    conn.close()

    recovered = temp_repo.recover_stale_runs(timeout_seconds=60.0)
    assert recovered == 1

    # Fresh run must remain RUNNING
    assert temp_repo.get_run(r_fresh).state == RunState.RUNNING

    # Stale run must be FAILED with audit event
    stale_rec = temp_repo.get_run(r_stale)
    assert stale_rec.state == RunState.FAILED
    assert stale_rec.error_category == "INTERRUPTED_CRASH"

    # Transactional audit log must be present
    history = temp_repo.get_lifecycle_history(r_stale)
    assert len(history) == 3  # None->CREATED, CREATED->RUNNING, RUNNING->FAILED
    last_event = history[-1]
    assert last_event.from_state == RunState.RUNNING
    assert last_event.to_state == RunState.FAILED
    assert "Automated stale-run recovery" in last_event.reason
