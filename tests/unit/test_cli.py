"""Unit Tests for MLForge Command Line Interface (Milestone 4.5).

Verifies the offline CLI subcommands:
- verify-bundle
- verify-release

Guarantees:
- Exit code 0 on authorized/valid verification.
- Exit code 1 on verification rejection or tampering.
- Exit code 2 on invalid arguments, missing trust material, or missing files.
- Zero model unpickling / deserialization during CLI verification.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlforge.application.artifacts import serialize_run_artifacts
from mlforge.application.packaging import build_audit_bundle
from mlforge.application.release_certificates import (
    generate_release_keypair,
    issue_release_certificate,
    serialize_public_key,
)
from mlforge.cli import main
from mlforge.core.contracts import (
    AcceptanceDecision,
    AcceptanceOutcome,
    AcceptancePolicy,
    AcceptanceRule,
    CriterionEvaluation,
    CriterionOutcome,
    CriterionSeverity,
    FoldResult,
    GateCheckResult,
    GateCheckStatus,
    MetricValue,
    ReleaseGateResult,
    ReleaseGateStatus,
    ReleasePolicy,
    RuleOperator,
)
from tests.unit.test_acceptance import _make_dummy_evaluation


@pytest.fixture
def cli_sample_run(tmp_path: Path):
    """Create a realistic evaluated run with serialized model, decision, and certificate."""
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=42))])
    pipe.fit([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], [0, 0, 1, 1])

    evaluation = _make_dummy_evaluation(cand_f1=0.85, cand_acc=0.88, ece=0.05)
    evaluation.candidate_result.fitted_pipeline = pipe
    evaluation.candidate_result.model_provenance = "final_cv_fold"
    evaluation.candidate_result.fitted_fold_index = 4
    evaluation.candidate_result.fold_results = [
        FoldResult(
            fold_index=i,
            train_indices_count=80,
            val_indices_count=20,
            metrics={"macro_f1": MetricValue(value=0.85), "accuracy": MetricValue(value=0.88)},
            confusion_matrix=[[40, 5], [5, 50]],
            class_labels=["0", "1"],
            fit_duration_seconds=0.01,
            predict_duration_seconds=0.001,
        )
        for i in range(5)
    ]

    policy = AcceptancePolicy(
        policy_id="cli_pol_01",
        version="1.0.0",
        name="CLI Test Policy",
        rules=[
            AcceptanceRule(
                rule_id="r1",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.80,
            )
        ],
    )

    decision = AcceptanceDecision(
        decision_id="dec_cli_01",
        evaluation_id="eval_01",
        policy_id=policy.policy_id,
        policy_version=policy.version,
        outcome=AcceptanceOutcome.PASS,
        criteria_evaluations=[
            CriterionEvaluation(
                rule_id="r1",
                metric="macro_f1",
                severity=CriterionSeverity.BLOCKER,
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold_value=0.80,
                observed_value=0.85,
                outcome=CriterionOutcome.SATISFIED,
                reason_code="THRESHOLD_MET",
                message="Satisfied",
                evidence_source="candidate_metric",
            )
        ],
        summary_rationale="Accepted",
        evaluated_at="2026-10-19T00:00:00Z",
        passed_rules_count=1,
        failed_rules_count=0,
        inconclusive_rules_count=0,
    )

    run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        acceptance_decision=decision,
        policy=policy,
        artifacts_dir=tmp_path / "runs",
        run_id="run_cli_01",
    )

    # Generate Ed25519 signing keypair
    priv_key, pub_key, key_id = generate_release_keypair()
    pub_key_hex = serialize_public_key(pub_key)

    # Issue release certificate
    cert = issue_release_certificate(
        source=run_dir,
        private_key=priv_key,
        key_id=key_id,
        release_policy=ReleasePolicy(require_acceptance_pass=True, require_model_artifact=True),
    )

    # Index certificate into manifest so build_audit_bundle includes it
    cert_file = run_dir / "release" / "release_certificate.json"
    manifest_file = run_dir / "manifest.json"
    mdata = json.loads(manifest_file.read_text(encoding="utf-8"))
    mdata["file_checksums"]["release/release_certificate.json"] = hashlib.sha256(cert_file.read_bytes()).hexdigest()
    manifest_file.write_text(json.dumps(mdata, indent=2), encoding="utf-8")

    # Also build an audit bundle containing the certificate
    bundle_zip = tmp_path / "bundle_with_cert.zip"
    build_audit_bundle(run_dir, output_bundle_path=bundle_zip)

    # Also build a plain audit bundle without certificate
    plain_run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        acceptance_decision=decision,
        policy=policy,
        artifacts_dir=tmp_path / "runs",
        run_id="run_cli_plain",
    )
    plain_bundle_zip = tmp_path / "bundle_plain.zip"
    build_audit_bundle(plain_run_dir, output_bundle_path=plain_bundle_zip)

    return {
        "run_dir": run_dir,
        "cert": cert,
        "cert_path": run_dir / "release" / "release_certificate.json",
        "priv_key": priv_key,
        "pub_key": pub_key,
        "key_id": key_id,
        "pub_key_hex": pub_key_hex,
        "bundle_zip": bundle_zip,
        "plain_bundle_zip": plain_bundle_zip,
    }


# ============================================================================
# Basic CLI Invocations & Usage Errors
# ============================================================================

def test_cli_no_arguments_returns_exit_code_2(capsys):
    """Calling CLI with no arguments prints usage and returns exit code 2."""
    ret = main([])
    assert ret == 2
    captured = capsys.readouterr()
    assert "usage:" in captured.out or "usage:" in captured.err or captured.out.strip() != ""


def test_cli_unknown_subcommand_returns_exit_code_2(capsys):
    """Calling CLI with an unknown subcommand returns exit code 2."""
    ret = main(["unknown-subcommand"])
    assert ret == 2


# ============================================================================
# verify-bundle Tests
# ============================================================================

def test_cli_verify_bundle_success(cli_sample_run, capsys):
    """verify-bundle returns 0 and prints [VALID] for a healthy audit bundle."""
    bundle_path = str(cli_sample_run["bundle_zip"])
    ret = main(["verify-bundle", bundle_path])
    assert ret == 0

    captured = capsys.readouterr()
    assert "[VALID]" in captured.out
    assert "verified successfully" in captured.out


def test_cli_verify_bundle_missing_file_returns_exit_code_2(tmp_path: Path, capsys):
    """verify-bundle returns 2 when the bundle path does not exist."""
    nonexistent = str(tmp_path / "does_not_exist.zip")
    ret = main(["verify-bundle", nonexistent])
    assert ret == 2

    captured = capsys.readouterr()
    assert "not found" in captured.err


def test_cli_verify_bundle_corrupt_zip_returns_exit_code_1(tmp_path: Path, capsys):
    """verify-bundle returns 1 when the file is not a valid ZIP archive."""
    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"not a valid zip")

    ret = main(["verify-bundle", str(corrupt)])
    assert ret == 1

    captured = capsys.readouterr()
    assert "[INVALID]" in captured.err


def test_cli_verify_bundle_tampered_returns_exit_code_1(cli_sample_run, tmp_path: Path, capsys):
    """verify-bundle returns 1 when a member file inside the ZIP has been tampered with."""
    orig_zip = cli_sample_run["bundle_zip"]
    tampered_zip = tmp_path / "tampered.zip"

    with zipfile.ZipFile(orig_zip, "r") as zf_in, zipfile.ZipFile(tampered_zip, "w") as zf_out:
        for item in zf_in.infolist():
            content = zf_in.read(item.filename)
            if item.filename == "evaluation.json":
                content = content + b" "
            zf_out.writestr(item, content)

    ret = main(["verify-bundle", str(tampered_zip)])
    assert ret == 1

    captured = capsys.readouterr()
    assert "[INVALID]" in captured.err
    assert "checksum mismatch" in captured.err or "Size mismatch" in captured.err


def test_cli_verify_bundle_quiet_flag(cli_sample_run, capsys):
    """--quiet suppresses standard diagnostic output."""
    bundle_path = str(cli_sample_run["bundle_zip"])
    ret = main(["verify-bundle", bundle_path, "--quiet"])
    assert ret == 0

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_cli_verify_bundle_resource_limits_exceeded(cli_sample_run, capsys):
    """verify-bundle returns 1 when member exceeds configured --max-member-size."""
    bundle_path = str(cli_sample_run["bundle_zip"])
    ret = main(["verify-bundle", bundle_path, "--max-member-size", "10"])
    assert ret == 1

    captured = capsys.readouterr()
    assert "[INVALID]" in captured.err
    assert "exceeds maximum permitted limit" in captured.err


# ============================================================================
# verify-release Tests
# ============================================================================

def test_cli_verify_release_missing_trust_material(cli_sample_run, capsys):
    """verify-release returns 2 when neither --public-key nor --trusted-keys-file is supplied."""
    target = str(cli_sample_run["run_dir"])
    ret = main(["verify-release", target])
    assert ret == 2

    captured = capsys.readouterr()
    assert "Missing required trust material" in captured.err


def test_cli_verify_release_invalid_public_key_hex(cli_sample_run, capsys):
    """verify-release returns 2 when public key hex is malformed."""
    target = str(cli_sample_run["run_dir"])
    ret = main(["verify-release", target, "--public-key", "invalid_hex_string"])
    assert ret == 2

    captured = capsys.readouterr()
    assert "Invalid --public-key" in captured.err or "Failed to parse" in captured.err


def test_cli_verify_release_directory_success(cli_sample_run, capsys):
    """verify-release succeeds with exit code 0 when given run directory and valid --public-key."""
    target = str(cli_sample_run["run_dir"])
    pub_key_hex = cli_sample_run["pub_key_hex"]

    ret = main(["verify-release", target, "--public-key", pub_key_hex])
    assert ret == 0

    captured = capsys.readouterr()
    assert "[AUTHORIZED]" in captured.out
    assert "Release authorized" in captured.out
    assert "Signer Trust:         TRUSTED" in captured.out


def test_cli_verify_release_trusted_keys_file_success(cli_sample_run, tmp_path: Path, capsys):
    """verify-release succeeds with exit code 0 when using --trusted-keys-file."""
    target = str(cli_sample_run["run_dir"])
    key_file = tmp_path / "trusted_keys.json"
    key_file.write_text(
        json.dumps({cli_sample_run["key_id"]: cli_sample_run["pub_key_hex"]}),
        encoding="utf-8",
    )

    ret = main(["verify-release", target, "--trusted-keys-file", str(key_file)])
    assert ret == 0

    captured = capsys.readouterr()
    assert "[AUTHORIZED]" in captured.out


def test_cli_verify_release_untrusted_public_key_returns_1(cli_sample_run, capsys):
    """verify-release returns 1 when the supplied public key does not match the certificate's key."""
    target = str(cli_sample_run["run_dir"])
    _, other_pub, _ = generate_release_keypair()
    other_pub_hex = serialize_public_key(other_pub)

    ret = main(["verify-release", target, "--public-key", other_pub_hex])
    assert ret == 1

    captured = capsys.readouterr()
    assert "[REJECTED]" in captured.err
    assert "Signer Trust:         UNTRUSTED" in captured.err


def test_cli_verify_release_audit_bundle_target_success(cli_sample_run, capsys):
    """verify-release succeeds with exit code 0 when given a .zip audit bundle containing the certificate."""
    bundle_target = str(cli_sample_run["bundle_zip"])
    pub_key_hex = cli_sample_run["pub_key_hex"]

    ret = main(["verify-release", bundle_target, "--public-key", pub_key_hex])
    assert ret == 0

    captured = capsys.readouterr()
    assert "[AUTHORIZED]" in captured.out


def test_cli_verify_release_bundle_without_certificate_returns_1(cli_sample_run, capsys):
    """verify-release returns 1 when the target audit bundle contains no release certificate."""
    plain_bundle = str(cli_sample_run["plain_bundle_zip"])
    pub_key_hex = cli_sample_run["pub_key_hex"]

    ret = main(["verify-release", plain_bundle, "--public-key", pub_key_hex])
    assert ret == 1

    captured = capsys.readouterr()
    assert "No release certificate found inside audit bundle" in captured.err


def test_cli_verify_release_certificate_without_expected_source_fails_closed(cli_sample_run, capsys):
    """Passing a certificate file without --expected-source fails closed with exit code 1."""
    cert_path = str(cli_sample_run["cert_path"])
    pub_key_hex = cli_sample_run["pub_key_hex"]

    ret = main(["verify-release", cert_path, "--public-key", pub_key_hex])
    assert ret == 1

    captured = capsys.readouterr()
    assert "[REJECTED]" in captured.err
    assert "Evidence Consistency: FAILED" in captured.err


def test_cli_verify_release_tampered_evidence_returns_1(cli_sample_run, tmp_path: Path, capsys):
    """verify-release returns 1 when evidence file in the directory has been modified."""
    # Corrupt evaluation.json by altering dataset_metadata checksum
    eval_file = cli_sample_run["run_dir"] / "evaluation.json"
    eval_dict = json.loads(eval_file.read_text(encoding="utf-8"))
    eval_dict["dataset_metadata"]["checksum_sha256"] = "tampered_dataset_sha256_000000"
    eval_file.write_text(json.dumps(eval_dict, indent=2), encoding="utf-8")

    target = str(cli_sample_run["run_dir"])
    pub_key_hex = cli_sample_run["pub_key_hex"]

    ret = main(["verify-release", target, "--public-key", pub_key_hex])
    assert ret == 1

    captured = capsys.readouterr()
    assert "[REJECTED]" in captured.err
    assert "Evidence Consistency: FAILED" in captured.err


# ============================================================================
# Security: Model Deserialization Prohibition
# ============================================================================

def test_cli_never_deserializes_models(cli_sample_run, monkeypatch):
    """Neither verify-bundle nor verify-release ever deserializes model objects."""
    import joblib
    import pickle

    def forbidden_load(*args, **kwargs):
        raise AssertionError("SECURITY BREACH: Deserialization function called in CLI verification!")

    monkeypatch.setattr(joblib, "load", forbidden_load)
    monkeypatch.setattr(pickle, "load", forbidden_load)
    monkeypatch.setattr(pickle, "loads", forbidden_load)

    # 1. verify-bundle
    bundle_path = str(cli_sample_run["bundle_zip"])
    assert main(["verify-bundle", bundle_path]) == 0

    # 2. verify-release
    target = str(cli_sample_run["run_dir"])
    pub_key_hex = cli_sample_run["pub_key_hex"]
    assert main(["verify-release", target, "--public-key", pub_key_hex]) == 0


# ============================================================================
# Direct Python -m mlforge Invocation
# ============================================================================

def test_cli_python_module_invocation():
    """Verify python -m mlforge invocation via subprocess works cleanly."""
    res = subprocess.run(
        [sys.executable, "-m", "mlforge", "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent.parent),
    )
    assert res.returncode == 0
    assert "verify-bundle" in res.stdout
    assert "verify-release" in res.stdout
