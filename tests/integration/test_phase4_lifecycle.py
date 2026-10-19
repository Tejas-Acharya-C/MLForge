"""Integration Tests for Unified Phase 4 Lifecycle (Milestone 4.5 Task 3).

Validates the full unified MLForge pipeline across real components:
Managed Evaluation (Engine) -> SQLite Persistence (Lifecycle State Machine) ->
Artifact Integrity & Model Provenance -> Offline HTML Report ->
Release Gate Evaluation -> Ed25519 Certificate Issuance ->
Audit Bundle Packaging & Safe Extraction -> Multi-Tiered Verification ->
CLI Verification Subcommands (verify-bundle, verify-release).

Includes comprehensive negative paths:
- Failed / rejected release gate outcome
- Post-issuance evidence tampering
- Untrusted signing keys
- Corrupted / tampered audit bundles
- Incomplete evidence and mismatched run identity
- Persistence lifecycle failure propagation
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Tuple
import zipfile

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from mlforge.application.artifacts import (
    load_fitted_model,
    load_run_artifacts,
    serialize_run_artifacts,
)
from mlforge.application.packaging import (
    build_audit_bundle,
    extract_audit_bundle,
    verify_audit_bundle,
)
from mlforge.application.persistence import (
    SQLiteRunRepository,
    execute_managed_evaluation,
)
from mlforge.application.release_certificates import (
    deserialize_public_key,
    generate_release_keypair,
    issue_release_certificate,
    serialize_public_key,
    verify_release_certificate,
)
from mlforge.application.reporting import generate_html_report
from mlforge.cli import main
from mlforge.core.contracts import (
    AcceptanceDecision,
    AcceptanceOutcome,
    AcceptancePolicy,
    AcceptanceRule,
    CriterionSeverity,
    ReleaseGateResult,
    ReleaseGateStatus,
    ReleasePolicy,
    RuleOperator,
    RunRecord,
    RunState,
)
from mlforge.core.exceptions import (
    ArtifactLoadError,
    ArtifactSecurityError,
    AuditBundleCorruptError,
    CertificateSecurityError,
    DatasetValidationError,
    LifecycleTransitionError,
    ReleaseGateError,
)
from mlforge.engine.release_gate import evaluate_release_gate


@pytest.fixture
def synthetic_phase4_data() -> pd.DataFrame:
    """Generate a deterministic synthetic tabular classification dataset."""
    X, y = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_classes=2,
        weights=[0.6, 0.4],
        random_state=42,
    )
    feature_cols = [f"feat_{i}" for i in range(6)]
    df = pd.DataFrame(X, columns=feature_cols)
    df["target"] = y
    rng = np.random.RandomState(42)
    df["subgroup"] = rng.choice(["Region_A", "Region_B", "Region_C"], size=120)
    return df


@pytest.fixture
def phase4_passing_policy() -> AcceptancePolicy:
    """Acceptance policy with achievable thresholds for candidate model."""
    return AcceptancePolicy(
        policy_id="phase4_acceptance_v1",
        version="1.0.0",
        name="Phase 4 Integration Policy",
        fail_precedence=True,
        rules=[
            AcceptanceRule(
                rule_id="r_acc_floor",
                metric="accuracy",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.55,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="candidate_metric",
            ),
            AcceptanceRule(
                rule_id="r_macro_f1_floor",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.55,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="candidate_metric",
            ),
        ],
    )


def test_phase4_unified_end_to_end_lifecycle(
    synthetic_phase4_data: pd.DataFrame,
    phase4_passing_policy: AcceptancePolicy,
    tmp_path: Path,
):
    """Deterministic end-to-end workflow through all real MLForge Phase 4 subsystems."""
    # -------------------------------------------------------------------------
    # Step 1: Managed Evaluation with SQLite Persistence
    # -------------------------------------------------------------------------
    db_file = tmp_path / "mlforge_phase4.db"
    repo = SQLiteRunRepository(db_path=db_file)
    repo.initialize_schema()

    artifacts_root = tmp_path / "artifacts"
    run_id = "phase4_e2e_run_001"

    record, evaluation, run_dir = execute_managed_evaluation(
        df=synthetic_phase4_data,
        target_column="target",
        num_folds=3,
        random_state=42,
        subgroup_columns=["subgroup"],
        acceptance_policy=phase4_passing_policy,
        artifacts_dir=artifacts_root,
        repository=repo,
        run_id=run_id,
        run_metadata={"environment": "integration_test", "tier": "phase4"},
    )

    # -------------------------------------------------------------------------
    # Step 2: Verify SQLite Persistence and Lifecycle History
    # -------------------------------------------------------------------------
    assert record.run_id == run_id
    assert record.state == RunState.COMPLETED
    assert record.dataset_checksum == evaluation.dataset_metadata.checksum_sha256
    assert record.started_at is not None
    assert record.completed_at is not None

    queried = repo.get_run(run_id)
    assert queried is not None
    assert queried.state == RunState.COMPLETED
    assert Path(queried.artifact_path) == run_dir.resolve()

    history = repo.get_lifecycle_history(run_id)
    assert len(history) == 3
    assert [e.to_state for e in history] == [
        RunState.CREATED,
        RunState.RUNNING,
        RunState.COMPLETED,
    ]

    metrics = repo.get_run_metrics(run_id)
    metric_names = {m.metric_name for m in metrics}
    assert "accuracy" in metric_names
    assert "macro_f1" in metric_names
    acc_metric = next(m for m in metrics if m.metric_name == "accuracy")
    assert acc_metric.fold_count == 3
    assert acc_metric.metric_value > 0.55

    # -------------------------------------------------------------------------
    # Step 3: Verify Artifact Integrity & Model Provenance
    # -------------------------------------------------------------------------
    loaded_artifacts = load_run_artifacts(run_dir)
    assert loaded_artifacts["manifest"].run_id == run_id
    assert loaded_artifacts["manifest"].schema_version is not None
    assert loaded_artifacts["decision"].outcome == AcceptanceOutcome.PASS

    # Model provenance checks
    cand_res = evaluation.candidate_result
    assert cand_res.model_provenance == "final_cv_fold"
    assert cand_res.fitted_fold_index == 2

    # Model deserialization security opt-in
    with pytest.raises(ArtifactSecurityError, match="Refusing to deserialize"):
        load_fitted_model(run_dir, model_role="candidate", trusted=False)

    fitted_model = load_fitted_model(run_dir, model_role="candidate", trusted=True)
    assert hasattr(fitted_model, "predict")
    sample_features = synthetic_phase4_data.drop(columns=["target"]).iloc[:5]
    preds = fitted_model.predict(sample_features)
    assert len(preds) == 5

    # -------------------------------------------------------------------------
    # Step 4: Offline HTML Report Generation
    # -------------------------------------------------------------------------
    report_file = generate_html_report(run_dir)
    assert report_file.exists()
    assert report_file.is_file()
    html_content = report_file.read_text(encoding="utf-8")
    assert run_id in html_content
    assert "Phase 4 Integration Policy" in html_content
    assert "http://" not in html_content.lower()
    assert "https://" not in html_content.lower()

    # -------------------------------------------------------------------------
    # Step 5: Release Gate Evaluation
    # -------------------------------------------------------------------------
    release_policy = ReleasePolicy(
        policy_id="phase4_release_pol_v1",
        version="1.0.0",
        target_environment="production",
        require_acceptance_pass=True,
        require_model_artifact=True,
        require_manifest_integrity=True,
        min_evaluated_folds=2,
    )
    gate_result = evaluate_release_gate(run_dir, policy=release_policy)
    assert gate_result.is_releasable is True
    assert gate_result.overall_status == ReleaseGateStatus.PASSED

    # -------------------------------------------------------------------------
    # Step 6: Ed25519 Release Certificate Issuance
    # -------------------------------------------------------------------------
    priv_key, pub_key, key_id = generate_release_keypair()
    pub_hex = serialize_public_key(pub_key)

    cert = issue_release_certificate(
        source=run_dir,
        private_key=priv_key,
        key_id=key_id,
        release_policy=release_policy,
    )
    assert cert.run_id == run_id
    assert cert.key_id == key_id
    assert cert.release_gate_outcome == ReleaseGateStatus.PASSED
    assert cert.candidate_model_checksum is not None
    assert cert.dataset_checksum == evaluation.dataset_metadata.checksum_sha256
    cert_path = run_dir / "release" / "release_certificate.json"
    assert cert_path.exists()

    # -------------------------------------------------------------------------
    # Step 7: Audit Bundle Packaging (Non-Circular Evidence Layout)
    # -------------------------------------------------------------------------
    # Index the issued certificate into manifest.json before bundle packaging
    manifest_path = run_dir / "manifest.json"
    mdata = json.loads(manifest_path.read_text(encoding="utf-8"))
    cert_rel = "release/release_certificate.json"
    mdata["file_checksums"][cert_rel] = hashlib.sha256(cert_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(mdata, indent=2), encoding="utf-8")

    bundle_path = tmp_path / f"{run_id}_audit_bundle.zip"
    created_bundle = build_audit_bundle(run_dir=run_dir, output_bundle_path=bundle_path)
    assert created_bundle.exists()

    # -------------------------------------------------------------------------
    # Step 8: Verify Bundle and Safely Extract
    # -------------------------------------------------------------------------
    bundle_verif = verify_audit_bundle(bundle_path)
    assert bundle_verif.is_valid is True
    assert bundle_verif.errors == []
    assert bundle_verif.manifest is not None
    assert bundle_verif.manifest.run_id == run_id

    extract_dest = tmp_path / "extracted_bundle"
    extracted_entries = extract_audit_bundle(bundle_path, destination_dir=extract_dest, verify_first=True)
    assert (extract_dest / "manifest.json").exists()
    assert (extract_dest / "release" / "release_certificate.json").exists()
    assert (extract_dest / "models" / "candidate_pipeline.joblib").exists()
    extracted_artifacts = load_run_artifacts(extract_dest)
    assert extracted_artifacts["manifest"].run_id == run_id

    # -------------------------------------------------------------------------
    # Step 9: Multi-Tiered Verification with Explicit Trust Boundaries
    # -------------------------------------------------------------------------
    trusted_registry = {key_id: pub_key}

    # Tiered check against local run directory
    res_dir = verify_release_certificate(
        certificate=cert,
        trusted_keys=trusted_registry,
        expected_source=run_dir,
    )
    assert res_dir.release_authorized is True
    assert res_dir.payload_integrity is True
    assert res_dir.signature_validity is True
    assert res_dir.signer_trust is True
    assert res_dir.evidence_consistency is True

    # Tiered check against ZIP audit bundle
    res_bundle = verify_release_certificate(
        certificate=cert,
        trusted_keys=trusted_registry,
        expected_source=bundle_path,
    )
    assert res_bundle.release_authorized is True
    assert res_bundle.evidence_consistency is True

    # Tiered check against safely extracted bundle directory
    res_extracted = verify_release_certificate(
        certificate=extract_dest / "release" / "release_certificate.json",
        trusted_keys=trusted_registry,
        expected_source=extract_dest,
    )
    assert res_extracted.release_authorized is True

    # -------------------------------------------------------------------------
    # Step 10: CLI Verification Subcommands
    # -------------------------------------------------------------------------
    # CLI in-process verify-bundle
    assert main(["verify-bundle", str(bundle_path)]) == 0

    # CLI in-process verify-release with --public-key against bundle and directory
    assert main(["verify-release", str(bundle_path), "--public-key", pub_hex]) == 0
    assert main(["verify-release", str(run_dir), "--public-key", pub_hex]) == 0

    # CLI in-process verify-release with trusted-keys-file
    trusted_keys_file = tmp_path / "trusted_signers.json"
    trusted_keys_file.write_text(
        json.dumps({key_id: pub_hex}),
        encoding="utf-8",
    )
    assert main(["verify-release", str(bundle_path), "--trusted-keys-file", str(trusted_keys_file)]) == 0

    # CLI subprocess execution: python -m mlforge
    sub_res = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlforge",
            "verify-release",
            str(bundle_path),
            "--public-key",
            pub_hex,
        ],
        capture_output=True,
        text=True,
    )
    assert sub_res.returncode == 0
    assert "[AUTHORIZED] Release authorized" in sub_res.stdout


def test_phase4_failure_gate_rejection_blocks_issuance(
    synthetic_phase4_data: pd.DataFrame,
    tmp_path: Path,
):
    """Release authorization must fail closed when acceptance criteria or release policy fails."""
    repo = SQLiteRunRepository(db_path=tmp_path / "repo.db")
    repo.initialize_schema()

    # Impossible policy: accuracy floor of 0.999
    strict_policy = AcceptancePolicy(
        policy_id="impossible_policy",
        version="1.0.0",
        name="Impossible Policy",
        fail_precedence=True,
        rules=[
            AcceptanceRule(
                rule_id="r_impossible_acc",
                metric="accuracy",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.999,
                severity=CriterionSeverity.BLOCKER,
            )
        ],
    )

    _, eval_res, run_dir = execute_managed_evaluation(
        df=synthetic_phase4_data,
        target_column="target",
        num_folds=3,
        random_state=42,
        acceptance_policy=strict_policy,
        artifacts_dir=tmp_path / "runs",
        repository=repo,
        run_id="failed_gate_run",
    )

    assert eval_res.acceptance_decision.outcome == AcceptanceOutcome.FAIL

    release_policy = ReleasePolicy(require_acceptance_pass=True)
    gate_result = evaluate_release_gate(run_dir, policy=release_policy)
    assert gate_result.is_releasable is False
    assert gate_result.overall_status == ReleaseGateStatus.BLOCKED

    priv_key, _, key_id = generate_release_keypair()
    with pytest.raises(CertificateSecurityError, match="authorization denied: release gate outcome is 'blocked'"):
        issue_release_certificate(
            source=run_dir,
            private_key=priv_key,
            key_id=key_id,
            release_policy=release_policy,
        )


def test_phase4_failure_evidence_tampering_detected(
    synthetic_phase4_data: pd.DataFrame,
    phase4_passing_policy: AcceptancePolicy,
    tmp_path: Path,
):
    """Post-issuance tampering of evidence files must invalidate evidence consistency and CLI exit codes."""
    repo = SQLiteRunRepository(db_path=tmp_path / "repo.db")
    repo.initialize_schema()

    _, _, run_dir = execute_managed_evaluation(
        df=synthetic_phase4_data,
        target_column="target",
        num_folds=3,
        random_state=42,
        acceptance_policy=phase4_passing_policy,
        artifacts_dir=tmp_path / "runs",
        repository=repo,
        run_id="tamper_evidence_run",
    )

    priv_key, pub_key, key_id = generate_release_keypair()
    pub_hex = serialize_public_key(pub_key)
    cert = issue_release_certificate(
        source=run_dir,
        private_key=priv_key,
        key_id=key_id,
        release_policy=ReleasePolicy(require_acceptance_pass=True),
    )

    # Tamper with model artifact
    cand_model_path = run_dir / "models" / "candidate_pipeline.joblib"
    cand_model_path.write_bytes(b"corrupted_candidate_model_payload_12345")

    # Verification must detect model checksum inconsistency
    res = verify_release_certificate(
        certificate=cert,
        trusted_keys={key_id: pub_key},
        expected_source=run_dir,
    )
    assert res.evidence_consistency is False
    assert res.release_authorized is False
    assert any("Candidate model checksum mismatch" in err for err in res.errors)

    # CLI verify-release returns 1 on evidence mismatch
    exit_code = main(["verify-release", str(run_dir), "--public-key", pub_hex])
    assert exit_code == 1


def test_phase4_failure_untrusted_signer_key(
    synthetic_phase4_data: pd.DataFrame,
    phase4_passing_policy: AcceptancePolicy,
    tmp_path: Path,
):
    """Valid certificate signed by untrusted key must never produce release authorization or exit 0."""
    repo = SQLiteRunRepository(db_path=tmp_path / "repo.db")
    repo.initialize_schema()

    _, _, run_dir = execute_managed_evaluation(
        df=synthetic_phase4_data,
        target_column="target",
        num_folds=3,
        random_state=42,
        acceptance_policy=phase4_passing_policy,
        artifacts_dir=tmp_path / "runs",
        repository=repo,
        run_id="untrusted_key_run",
    )

    priv_key, pub_key, key_id = generate_release_keypair()
    cert = issue_release_certificate(
        source=run_dir,
        private_key=priv_key,
        key_id=key_id,
        release_policy=ReleasePolicy(require_acceptance_pass=True),
    )

    # Separate unauthorized keypair
    _, untrusted_pub, untrusted_key_id = generate_release_keypair()
    untrusted_pub_hex = serialize_public_key(untrusted_pub)

    # Verify with trusted registry containing only the unauthorized key
    res = verify_release_certificate(
        certificate=cert,
        trusted_keys={untrusted_key_id: untrusted_pub},
        expected_source=run_dir,
    )
    assert res.signer_trust is False
    assert res.release_authorized is False
    assert any("NOT present in supplied trusted keys registry" in w for w in res.warnings)

    # CLI passing a mismatched public key directly fails verification
    exit_code = main(["verify-release", str(run_dir), "--public-key", untrusted_pub_hex])
    assert exit_code in (1, 2)
    assert exit_code != 0


def test_phase4_failure_corrupted_audit_bundle(
    synthetic_phase4_data: pd.DataFrame,
    phase4_passing_policy: AcceptancePolicy,
    tmp_path: Path,
):
    """Corrupted audit bundle archive fails bundle verification, extraction, and release gate."""
    repo = SQLiteRunRepository(db_path=tmp_path / "repo.db")
    repo.initialize_schema()

    _, _, run_dir = execute_managed_evaluation(
        df=synthetic_phase4_data,
        target_column="target",
        num_folds=3,
        random_state=42,
        acceptance_policy=phase4_passing_policy,
        artifacts_dir=tmp_path / "runs",
        repository=repo,
        run_id="corrupt_bundle_run",
    )

    bundle_zip = tmp_path / "bundle_to_corrupt.zip"
    build_audit_bundle(run_dir=run_dir, output_bundle_path=bundle_zip)

    # Corrupt member payload bytes inside ZIP
    raw = bytearray(bundle_zip.read_bytes())
    with zipfile.ZipFile(bundle_zip, "r") as zf:
        info = zf.getinfo("manifest.json")
        header_offset = info.header_offset
        # Mutate byte in compressed payload data
        data_offset = header_offset + 30 + len(info.filename) + len(info.extra)
        raw[data_offset + 2] ^= 0xFF
    bundle_zip.write_bytes(bytes(raw))

    # 1. verify_audit_bundle fails
    verif = verify_audit_bundle(bundle_zip, raise_on_error=False)
    assert verif.is_valid is False
    assert len(verif.errors) > 0

    # 2. extract_audit_bundle raises
    with pytest.raises((AuditBundleCorruptError, Exception)):
        extract_audit_bundle(bundle_zip, destination_dir=tmp_path / "extract_corrupt", verify_first=True)

    # 3. evaluate_release_gate against corrupt bundle yields BLOCKED
    gate_res = evaluate_release_gate(bundle_zip)
    assert gate_res.is_releasable is False
    assert gate_res.overall_status == ReleaseGateStatus.BLOCKED
    assert any(g.gate_id == "bundle_integrity" and g.status.value == "failed" for g in gate_res.gate_results)

    # 4. CLI verify-bundle returns 1
    assert main(["verify-bundle", str(bundle_zip)]) == 1


def test_phase4_failure_incomplete_evidence_and_run_mismatch(
    synthetic_phase4_data: pd.DataFrame,
    phase4_passing_policy: AcceptancePolicy,
    tmp_path: Path,
):
    """Incomplete evidence directories and cross-run evidence binding must fail closed."""
    repo = SQLiteRunRepository(db_path=tmp_path / "repo.db")
    repo.initialize_schema()

    # Generate Run A
    _, _, run_a_dir = execute_managed_evaluation(
        df=synthetic_phase4_data,
        target_column="target",
        num_folds=3,
        random_state=42,
        acceptance_policy=phase4_passing_policy,
        artifacts_dir=tmp_path / "runs",
        repository=repo,
        run_id="run_a",
    )

    # Generate Run B
    _, _, run_b_dir = execute_managed_evaluation(
        df=synthetic_phase4_data,
        target_column="target",
        num_folds=3,
        random_state=123,
        acceptance_policy=phase4_passing_policy,
        artifacts_dir=tmp_path / "runs",
        repository=repo,
        run_id="run_b",
    )

    priv_key, pub_key, key_id = generate_release_keypair()
    cert_a = issue_release_certificate(
        source=run_a_dir,
        private_key=priv_key,
        key_id=key_id,
        release_policy=ReleasePolicy(require_acceptance_pass=True),
    )

    # Test Mismatched Run Identity: Validate Certificate A against Run B's artifacts
    res_mismatch = verify_release_certificate(
        certificate=cert_a,
        trusted_keys={key_id: pub_key},
        expected_source=run_b_dir,
    )
    assert res_mismatch.evidence_consistency is False
    assert res_mismatch.release_authorized is False
    assert any("Run ID mismatch" in err for err in res_mismatch.errors)

    # Test Incomplete Evidence: Delete decision.json from Run A
    (run_a_dir / "decision.json").unlink()
    res_incomplete = verify_release_certificate(
        certificate=cert_a,
        trusted_keys={key_id: pub_key},
        expected_source=run_a_dir,
    )
    assert res_incomplete.evidence_consistency is False
    assert res_incomplete.release_authorized is False
    assert any("decision.json" in err for err in res_incomplete.errors)


def test_phase4_persistence_lifecycle_and_failure_propagation(
    synthetic_phase4_data: pd.DataFrame,
    tmp_path: Path,
):
    """Repository state machine enforces terminal-state immutability and records failed execution."""
    repo = SQLiteRunRepository(db_path=tmp_path / "lifecycle.db")
    repo.initialize_schema()

    # 1. Terminal state immutability
    rec = repo.create_run("state_immutability_run")
    repo.transition_state("state_immutability_run", to_state=RunState.RUNNING)
    repo.transition_state("state_immutability_run", to_state=RunState.COMPLETED)

    with pytest.raises(LifecycleTransitionError, match="Terminal states are immutable"):
        repo.transition_state("state_immutability_run", to_state=RunState.RUNNING)

    with pytest.raises(LifecycleTransitionError, match="Terminal states are immutable"):
        repo.transition_state("state_immutability_run", to_state=RunState.FAILED)

    # 2. Managed evaluation error propagation and FAILED state persistence
    with pytest.raises(DatasetValidationError):
        execute_managed_evaluation(
            df=synthetic_phase4_data,
            target_column="nonexistent_target_column",
            num_folds=3,
            repository=repo,
            run_id="failing_run",
            artifacts_dir=tmp_path / "runs",
        )

    failed_rec = repo.get_run("failing_run")
    assert failed_rec is not None
    assert failed_rec.state == RunState.FAILED
    assert failed_rec.error_category == "DATASET_VALIDATION_ERROR"
    assert failed_rec.error_message is not None

    hist = repo.get_lifecycle_history("failing_run")
    assert [e.to_state for e in hist] == [RunState.CREATED, RunState.RUNNING, RunState.FAILED]


def test_phase4_benchmark_runner_execution(tmp_path: Path):
    """Verify that run_phase4_benchmarks executes all Phase 4 scenarios and writes structured report."""
    from benchmarks.runner import run_phase4_benchmarks

    report = run_phase4_benchmarks(output_dir=tmp_path, repeats=1, random_state=42)

    assert report["milestone"] == "4.5"
    assert len(report["scenarios"]) == 2
    for sc in report["scenarios"]:
        assert sc["status"] == "SUCCESS"
        assert sc["error_message"] is None
        assert sc["repeats"] == 1
        t = sc["timings_ms"]
        assert t["total_elapsed"]["median"] > 0
        assert t["managed_evaluation_and_persistence"]["median"] > 0
        assert t["offline_html_report_generation"]["median"] > 0
        assert t["release_gate_evaluation"]["median"] > 0
        assert t["certificate_issuance"]["median"] > 0
        assert t["audit_bundle_packaging"]["median"] > 0
        assert t["audit_bundle_verification"]["median"] > 0
        assert t["certificate_verification_directory"]["median"] > 0
        assert t["certificate_verification_bundle"]["median"] > 0
        assert t["audit_bundle_safe_extraction"]["median"] > 0
        assert sc["peak_traced_memory_kb"]["median"] > 0

    assert (tmp_path / "phase4_benchmark_results.json").exists()
