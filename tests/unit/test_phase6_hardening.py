"""Phase 6 System Hardening & End-to-End Security Boundary Tests.

Verifies:
1. Release certificate trust model: mathematical signature validity separated from signer trust.
2. Unknown signers with valid signatures fail authorization (fail-closed).
3. Payload tampering and signature tampering detection.
4. Evidence consistency and artifact integrity enforcement.
5. Server keystore precedence over untrusted caller keys.
6. DNS rebinding Host header defense.
7. Browser cross-origin state-changing Origin header validation.
8. Filesystem path confinement for dataset loading and artifact downloads.
9. Strict run_id validation preventing path traversal injection.
10. Offline healthz and robustness query endpoints.
11. End-to-end lifecycle journey from dataset inspection to certificate verification.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Generator
import urllib.parse

from cryptography.hazmat.primitives.asymmetric import ed25519
from fastapi.testclient import TestClient
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlforge.api.app import create_app
from mlforge.api.routes import get_repository
from mlforge.application.artifacts import serialize_run_artifacts
from mlforge.application.persistence import SQLiteRunRepository
from mlforge.application.release_certificates import (
    deserialize_public_key,
    generate_release_keypair,
    issue_release_certificate,
    serialize_public_key,
    verify_release_certificate,
)
from mlforge.core.contracts import (
    AcceptanceDecision,
    AcceptanceOutcome,
    AcceptancePolicy,
    AcceptanceRule,
    ArtifactManifest,
    CriterionEvaluation,
    CriterionOutcome,
    CriterionSeverity,
    EvaluationResult,
    FoldResult,
    MetricValue,
    ReleaseCertificate,
    ReleaseGateStatus,
    ReleasePolicy,
    RuleOperator,
)
from mlforge.core.exceptions import CertificateSecurityError
from tests.unit.test_acceptance import _make_dummy_evaluation


@pytest.fixture
def hardening_repo(tmp_path: Path) -> SQLiteRunRepository:
    """Isolated SQLite repository for Phase 6 hardening verification."""
    db_file = tmp_path / "hardening_test.db"
    repo = SQLiteRunRepository(db_path=db_file)
    repo.initialize_schema()
    return repo


@pytest.fixture
def hardening_client(hardening_repo: SQLiteRunRepository) -> Generator[TestClient, None, None]:
    """TestClient with isolated repository dependency."""
    app = create_app()
    app.dependency_overrides[get_repository] = lambda: hardening_repo
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _build_passing_run(target_parent: Path, run_id: str = "run_cert_test") -> Path:
    """Construct a fully compliant evaluation run directory with passing gate evidence."""
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=42))])
    pipe.fit([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], [0, 0, 1, 1])

    evaluation = _make_dummy_evaluation(cand_f1=0.88, cand_acc=0.90, base_f1=0.72, base_acc=0.75)
    evaluation.candidate_result.fitted_pipeline = pipe
    evaluation.candidate_result.model_provenance = "final_cv_fold"
    evaluation.candidate_result.fitted_fold_index = 4
    evaluation.candidate_result.fold_results = [
        FoldResult(
            fold_index=i,
            train_indices_count=80,
            val_indices_count=20,
            metrics={"macro_f1": MetricValue(value=0.88)},
            confusion_matrix=[[10, 0], [0, 10]],
            class_labels=["0", "1"],
            fit_duration_seconds=0.01,
            predict_duration_seconds=0.001,
        )
        for i in range(5)
    ]

    policy = AcceptancePolicy(
        policy_id="prod_release_policy",
        version="1.0.0",
        name="Production Release Policy",
        description="Standard acceptance policy",
        rules=[
            AcceptanceRule(
                rule_id="min_f1",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.80,
                severity=CriterionSeverity.BLOCKER,
            )
        ],
    )

    crit = CriterionEvaluation(
        rule_id="min_f1",
        metric="macro_f1",
        severity=CriterionSeverity.BLOCKER,
        operator=RuleOperator.GREATER_THAN_OR_EQUAL,
        threshold_value=0.80,
        observed_value=0.88,
        outcome=CriterionOutcome.SATISFIED,
        reason_code="RULE_SATISFIED",
        message="Candidate F1 0.88 >= 0.80",
        evidence_source="candidate",
    )

    decision = AcceptanceDecision(
        decision_id=f"dec_{run_id}",
        evaluation_id=f"eval_{run_id}",
        policy_id=policy.policy_id,
        policy_version=policy.version,
        outcome=AcceptanceOutcome.PASS,
        criteria_evaluations=[crit],
        summary_rationale="All blocker criteria satisfied.",
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        passed_rules_count=1,
        failed_rules_count=0,
        inconclusive_rules_count=0,
        hash_signature=f"sig_{run_id}",
    )

    return serialize_run_artifacts(
        evaluation=evaluation,
        acceptance_decision=decision,
        policy=policy,
        artifacts_dir=target_parent,
        run_id=run_id,
    )


# ============================================================================
# Workstream B: Certificate Trust & Release Authorization Tests
# ============================================================================

def test_unknown_signer_with_valid_signature_fails_trust_and_authorization(tmp_path: Path):
    """A valid mathematical Ed25519 signature from an UNTRUSTED signer must result in:
    signature_validity=True, signer_trust=False, release_authorized=False (Fail-Closed).
    """
    # 1. Setup fully compliant evidence directory
    run_dir = _build_passing_run(tmp_path / "runs", "run_test_trust")

    # 2. Keypair A (the verifier's trusted keystore)
    priv_trusted, pub_trusted, kid_trusted = generate_release_keypair()
    trusted_registry = {kid_trusted: pub_trusted}

    # 3. Keypair B (an untrusted third-party/attacker signer)
    priv_untrusted, pub_untrusted, kid_untrusted = generate_release_keypair()
    candidate_registry = {kid_untrusted: pub_untrusted}

    # 4. Issue certificate using Keypair B (untrusted signer)
    policy = ReleasePolicy(policy_id="default", version="1.0.0", require_acceptance_pass=True, require_manifest_integrity=True)
    cert = issue_release_certificate(
        source=run_dir,
        private_key=priv_untrusted,
        key_id=kid_untrusted,
        release_policy=policy,
        overwrite=True,
    )

    # 5. Verify against Verifier's trusted registry (which only knows Keypair A)
    # providing candidate_registry for mathematical verification
    result = verify_release_certificate(
        certificate=cert,
        trusted_keys=trusted_registry,
        expected_source=run_dir,
        candidate_keys=candidate_registry,
    )

    # Mathematical signature IS valid
    assert result.payload_integrity is True
    assert result.signature_validity is True
    # But signer is UNTRUSTED
    assert result.signer_trust is False
    assert result.evidence_consistency is True
    # Final authorization MUST FAIL CLOSED
    assert result.release_authorized is False
    assert any("UNTRUSTED" in d or "untrusted" in d.lower() for d in result.details + result.warnings)


def test_tampered_certificate_payload_fails_integrity(tmp_path: Path):
    """Tampering with any field of a signed certificate must cause payload_integrity=False."""
    run_dir = _build_passing_run(tmp_path / "runs", "run_tamper")
    priv, pub, kid = generate_release_keypair()
    policy = ReleasePolicy(policy_id="test_pol", version="1.0.0", require_acceptance_pass=True, require_manifest_integrity=True)
    cert = issue_release_certificate(source=run_dir, private_key=priv, key_id=kid, release_policy=policy, overwrite=True)

    # Tamper with the target environment
    tampered_dict = cert.model_dump()
    tampered_dict["target_environment"] = "production_unauthorized"
    tampered_cert = ReleaseCertificate.model_validate(tampered_dict)

    ver = verify_release_certificate(
        certificate=tampered_cert,
        trusted_keys={kid: pub},
        expected_source=run_dir,
    )
    assert ver.payload_integrity is False
    assert ver.release_authorized is False
    assert any("integrity violation" in err.lower() for err in ver.errors)


def test_tampered_signature_fails_validity(tmp_path: Path):
    """Modifying signature bytes must cause signature_validity=False."""
    run_dir = _build_passing_run(tmp_path / "runs", "run_sig")
    priv, pub, kid = generate_release_keypair()
    policy = ReleasePolicy(policy_id="test_pol", version="1.0.0", require_acceptance_pass=True, require_manifest_integrity=True)
    cert = issue_release_certificate(source=run_dir, private_key=priv, key_id=kid, release_policy=policy, overwrite=True)

    # Invert the last hex characters of signature
    sig = cert.signature
    corrupt_sig = sig[:-4] + ("0000" if sig[-4:] != "0000" else "ffff")
    bad_cert_dict = cert.model_dump()
    bad_cert_dict["signature"] = corrupt_sig
    bad_cert = ReleaseCertificate.model_validate(bad_cert_dict)

    ver = verify_release_certificate(
        certificate=bad_cert,
        trusted_keys={kid: pub},
        expected_source=run_dir,
    )
    assert ver.signature_validity is False
    assert ver.release_authorized is False


# ============================================================================
# Workstream C: API Security, Path Confinement & Origin Defense Tests
# ============================================================================

def test_api_security_origin_header_validation(hardening_client: TestClient):
    """State-changing requests with unauthorized Origin must be rejected with 403 Forbidden."""
    # 1. Cross-origin from unauthorized site -> 403
    resp_blocked = hardening_client.post(
        "/api/v1/datasets/inspect",
        json={"sample_name": "churn", "target_column": "churn_label"},
        headers={"Origin": "http://malicious-tracker.com"},
    )
    assert resp_blocked.status_code == 403
    assert "Cross-origin request" in resp_blocked.json()["detail"]

    # 2. Local allowed origin -> 200
    resp_allowed = hardening_client.post(
        "/api/v1/datasets/inspect",
        json={"sample_name": "churn", "target_column": "churn_label"},
        headers={"Origin": "http://localhost:5173"},
    )
    assert resp_allowed.status_code == 200

    # 3. Direct API call without Origin header (local CLI/curl) -> 200
    resp_direct = hardening_client.post(
        "/api/v1/datasets/inspect",
        json={"sample_name": "churn", "target_column": "churn_label"},
    )
    assert resp_direct.status_code == 200


def test_api_security_host_header_dns_rebinding_defense(hardening_client: TestClient):
    """Requests with untrusted Host header (DNS rebinding) must be rejected with 403 Forbidden."""
    resp = hardening_client.get(
        "/api/v1/datasets/sample",
        headers={"Host": "attacker-dns-rebound.evil.com:8000"},
    )
    assert resp.status_code == 403
    assert "Untrusted host header" in resp.json()["detail"]


def test_api_dataset_path_confinement_defense(hardening_client: TestClient):
    """Dataset file_path outside permitted roots must be rejected with 403 Forbidden."""
    # Attempting to inspect arbitrary system file
    resp = hardening_client.post(
        "/api/v1/datasets/inspect",
        json={"file_path": "C:\\Windows\\System32\\drivers\\etc\\hosts", "target_column": "dummy"},
    )
    assert resp.status_code == 403
    assert "outside permitted directory scope" in resp.json()["detail"]


def test_api_run_id_injection_rejection(hardening_client: TestClient):
    """Malformed run IDs with path traversal or symbols must be rejected with 400 Bad Request."""
    malicious_run_ids = [
        "../../etc/passwd",
        "..\\..\\Windows\\win.ini",
        "eval;drop table runs;",
        "run<script>alert(1)</script>",
        "run name with spaces",
        "run/with/slashes",
        "run\\with\\backslashes",
    ]
    # 1. Body-level injection rejection on creation
    for bad_id in malicious_run_ids:
        resp = hardening_client.post(
            "/api/v1/evaluations",
            json={
                "sample_name": "churn",
                "target_column": "churn_label",
                "run_id": bad_id,
            },
        )
        assert resp.status_code == 400
        assert "Invalid run_id format" in resp.json()["detail"]

    # 2. Path-level injection rejection
    url_bad_ids = [
        "eval;drop",
        "bad$run$id",
        "run@evil",
        "run(parenthesis)",
    ]
    for bad_id in url_bad_ids:
        resp = hardening_client.get(f"/api/v1/evaluations/{bad_id}")
        assert resp.status_code == 400
        assert "Invalid run_id format" in resp.json()["detail"]


def test_api_healthz_endpoint(hardening_client: TestClient):
    """Healthz endpoint returns ok status for UI connectivity checking."""
    resp = hardening_client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["service"] == "mlforge"


def test_api_server_keystore_precedence(hardening_client: TestClient, tmp_path: Path):
    """When MLFORGE_TRUSTED_KEYS_FILE is set on the server, caller-supplied trusted_keys cannot bypass it."""
    # 1. Create a server keystore with Server Key
    priv_srv, pub_srv, kid_srv = generate_release_keypair()
    keystore_file = tmp_path / "server_trusted_signers.json"
    keystore_file.write_text(json.dumps({kid_srv: serialize_public_key(pub_srv)}), encoding="utf-8")

    # 2. Attacker creates keypair and signs certificate
    priv_att, pub_att, kid_att = generate_release_keypair()
    att_pub_hex = serialize_public_key(pub_att)

    # Setup valid run
    run_dir = _build_passing_run(tmp_path / "runs", "run_keystore")

    cert = issue_release_certificate(
        source=run_dir,
        private_key=priv_att,
        key_id=kid_att,
        release_policy=ReleasePolicy(require_acceptance_pass=True, require_manifest_integrity=True),
        overwrite=True,
    )

    # 3. Attacker calls verify-release sending att_pub_hex in trusted_keys
    # With server keystore active in environment
    os.environ["MLFORGE_TRUSTED_KEYS_FILE"] = str(keystore_file)
    try:
        resp = hardening_client.post(
            "/api/v1/verify-release",
            json={
                "certificate": cert.model_dump(),
                "trusted_keys": {kid_att: att_pub_hex},
                "evidence_dir": str(run_dir),
            },
        )
        assert resp.status_code == 200
        ver_res = resp.json()
        # Server keystore took precedence: attacker's key was NOT trusted
        assert ver_res["signer_trust"] is False
        assert ver_res["release_authorized"] is False
    finally:
        os.environ.pop("MLFORGE_TRUSTED_KEYS_FILE", None)


# ============================================================================
# Workstream D: End-to-End Journeys via API
# ============================================================================

def test_e2e_journey_dataset_to_release_verification(hardening_client: TestClient):
    """Complete end-to-end user journey across all lifecycle layers."""
    # Journey 1: Inspect sample dataset
    inspect_resp = hardening_client.post(
        "/api/v1/datasets/inspect",
        json={"sample_name": "churn", "target_column": "churn_label", "k_folds": 2},
    )
    assert inspect_resp.status_code == 200
    meta = inspect_resp.json()
    assert meta["num_rows"] == 500
    assert "churn_label" == meta["target_name"]

    # Journey 2: Submit evaluation and inspect results
    sub_resp = hardening_client.post(
        "/api/v1/evaluations",
        json={
            "sample_name": "churn",
            "target_column": "churn_label",
            "candidate_model_type": "logistic_regression",
            "num_folds": 2,
            "random_state": 42,
            "policy_id": "permissive_v1",
        },
    )
    assert sub_resp.status_code == 201
    run_id = sub_resp.json()["run_id"]

    # Inspect evaluation details
    detail_resp = hardening_client.get(f"/api/v1/evaluations/{run_id}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["has_evaluation"] is True

    # Retrieve fold results
    folds_resp = hardening_client.get(f"/api/v1/evaluations/{run_id}/results/folds")
    assert folds_resp.status_code == 200
    assert len(folds_resp.json()) == 2

    # Journey 3: Run robustness stress tests and retrieve via GET
    rob_resp = hardening_client.post(
        f"/api/v1/evaluations/{run_id}/robustness",
        json={"profile": "noise", "severity_levels": [0.05, 0.10]},
    )
    assert rob_resp.status_code == 200
    rob_data = rob_resp.json()
    assert len(rob_data["curves"]) > 0

    # Retrieve via GET robustness endpoint
    rob_get = hardening_client.get(f"/api/v1/evaluations/{run_id}/robustness")
    assert rob_get.status_code == 200
    assert rob_get.json()["curves"][0]["area_under_degradation_curve"] > 0

    # Journey 4: Artifact ledger and bundle creation
    art_resp = hardening_client.get(f"/api/v1/evaluations/{run_id}/artifacts")
    assert art_resp.status_code == 200
    art_data = art_resp.json()
    assert len(art_data["artifacts"]) >= 3
    paths = [a["path"] for a in art_data["artifacts"]]
    assert "evaluation.json" in paths
    assert "manifest.json" in paths

    bundle_resp = hardening_client.post(f"/api/v1/evaluations/{run_id}/bundle", json={})
    assert bundle_resp.status_code == 200
    bundle_path = bundle_resp.json()["bundle_path"]

    # Journey 5: Certificate issuance and verification
    # Evaluate release gate
    gate_resp = hardening_client.post(
        f"/api/v1/evaluations/{run_id}/release-gate",
        json={"policy": {"require_acceptance_pass": False, "target_environment": "staging"}},
    )
    assert gate_resp.status_code == 200
    assert gate_resp.json()["overall_status"] == "passed"

    # Issue certificate
    cert_resp = hardening_client.post(
        f"/api/v1/evaluations/{run_id}/certificate/issue",
        json={"key_id": "lab_signer_01", "target_environment": "staging", "release_policy": {"require_acceptance_pass": False}},
    )
    assert cert_resp.status_code == 200
    cert = cert_resp.json()["certificate"]
    pub_hex = cert_resp.json()["public_key_hex"]

    # Verify with trusted key -> Authorized
    ver_trusted = hardening_client.post(
        "/api/v1/verify-release",
        json={
            "certificate": cert,
            "trusted_keys": {"lab_signer_01": pub_hex},
        },
    )
    assert ver_trusted.status_code == 200
    v1 = ver_trusted.json()
    assert v1["signature_validity"] is True
    assert v1["signer_trust"] is True
    assert v1["evidence_consistency"] is True
    assert v1["release_authorized"] is True

    # Verify with untrusted candidate key -> Unsigned/Untrusted, NOT authorized
    ver_untrusted = hardening_client.post(
        "/api/v1/verify-release",
        json={
            "certificate": cert,
            "trusted_keys": {},
            "candidate_keys": {"lab_signer_01": pub_hex},
        },
    )
    assert ver_untrusted.status_code == 200
    v2 = ver_untrusted.json()
    assert v2["signature_validity"] is True
    assert v2["signer_trust"] is False
    assert v2["evidence_consistency"] is True
    assert v2["release_authorized"] is False


def test_api_dataset_path_validation_and_size_limits(hardening_client: TestClient, tmp_path: Path):
    """Verify detailed dataset validation: sensitive files, size bounds, file types, and directory traversal."""
    # 1. Sensitive dotfile in permitted temp directory -> 403
    dotfile = tmp_path / ".env"
    dotfile.write_text("SECRET_KEY=12345", encoding="utf-8")
    resp_dot = hardening_client.post(
        "/api/v1/datasets/inspect",
        json={"file_path": str(dotfile), "target_column": "dummy"},
    )
    assert resp_dot.status_code == 403
    assert "sensitive file" in resp_dot.json()["detail"].lower()

    # 2. Database file in permitted directory -> 403
    db_file = tmp_path / "mlforge.db"
    db_file.write_text("sqlite format", encoding="utf-8")
    resp_db = hardening_client.post(
        "/api/v1/datasets/inspect",
        json={"file_path": str(db_file), "target_column": "dummy"},
    )
    assert resp_db.status_code == 403
    assert "sensitive file" in resp_db.json()["detail"].lower()

    # 3. Directory path instead of file -> 404
    dir_path = tmp_path / "somedir"
    dir_path.mkdir()
    resp_dir = hardening_client.post(
        "/api/v1/datasets/inspect",
        json={"file_path": str(dir_path), "target_column": "dummy"},
    )
    assert resp_dir.status_code == 404
    assert "not a regular file" in resp_dir.json()["detail"].lower()

    # 4. Unsupported file type (.txt) -> 400
    txt_file = tmp_path / "data.txt"
    txt_file.write_text("col1,col2\n1,2", encoding="utf-8")
    resp_txt = hardening_client.post(
        "/api/v1/datasets/inspect",
        json={"file_path": str(txt_file), "target_column": "col2"},
    )
    assert resp_txt.status_code == 400
    assert "unsupported file format" in resp_txt.json()["detail"].lower()

    # 5. Valid CSV in permitted directory -> 200
    csv_file = tmp_path / "valid_data.csv"
    csv_rows = ["feat1,feat2,target"] + [f"{float(i)},{float(i*2)},{i % 2}" for i in range(20)]
    csv_file.write_text("\n".join(csv_rows) + "\n", encoding="utf-8")
    resp_csv = hardening_client.post(
        "/api/v1/datasets/inspect",
        json={"file_path": str(csv_file), "target_column": "target", "k_folds": 2},
    )
    assert resp_csv.status_code == 200
    assert resp_csv.json()["num_rows"] == 20

    # 6. MLFORGE_DATA_DIR environment trust boundary
    custom_dir = tmp_path / "custom_data_root"
    custom_dir.mkdir()
    custom_csv = custom_dir / "custom.csv"
    custom_csv.write_text("\n".join(csv_rows) + "\n", encoding="utf-8")

    os.environ["MLFORGE_DATA_DIR"] = str(custom_dir)
    try:
        resp_custom = hardening_client.post(
            "/api/v1/datasets/inspect",
            json={"file_path": str(custom_csv), "target_column": "target", "k_folds": 2},
        )
        assert resp_custom.status_code == 200
        assert resp_custom.json()["num_rows"] == 20
    finally:
        os.environ.pop("MLFORGE_DATA_DIR", None)


def test_api_artifact_download_path_traversal(hardening_client: TestClient, tmp_path: Path):
    """Verify artifact download endpoint strictly defends against path traversal and non-existent files."""
    # 1. Create compliant run with artifacts
    run_dir = _build_passing_run(tmp_path / "runs", "run_art_download")
    resp = hardening_client.post(
        "/api/v1/evaluations",
        json={
            "sample_name": "churn",
            "target_column": "churn_label",
            "run_id": "eval_art_download",
            "num_folds": 2,
        },
    )
    assert resp.status_code == 201

    # 2. Legitimate download -> 200
    legit_resp = hardening_client.get("/api/v1/evaluations/eval_art_download/artifacts/evaluation.json")
    assert legit_resp.status_code == 200

    # 3. Path traversal attempts are blocked (HTTP 400 or 404 from URL normalization, never 200)
    traversal_paths = [
        "../secret.txt",
        "..\\secret.txt",
        "sub%2F..%2Fsecret.txt",
        "%2e%2e%2fsecret.txt",
        "/etc/passwd",
        "\\Windows\\win.ini",
    ]
    for p in traversal_paths:
        trav_resp = hardening_client.get(f"/api/v1/evaluations/eval_art_download/artifacts/{p}")
        assert trav_resp.status_code in (400, 404)
        assert trav_resp.status_code != 200

    # 4. Non-existent file within directory -> 404
    missing_resp = hardening_client.get("/api/v1/evaluations/eval_art_download/artifacts/non_existent.json")
    assert missing_resp.status_code == 404
    assert "not found" in missing_resp.json()["detail"].lower()


def test_api_certificate_and_bundle_path_validation(hardening_client: TestClient, tmp_path: Path):
    """Verify certificate and bundle path validation checks file existence and extensions."""
    # 1. Non-existent certificate path -> 404
    resp_c1 = hardening_client.post(
        "/api/v1/verify-release",
        json={"certificate_path": str(tmp_path / "non_existent.json")},
    )
    assert resp_c1.status_code == 404
    assert "not found" in resp_c1.json()["detail"].lower()

    # 2. Certificate path with wrong extension -> 400
    wrong_ext_cert = tmp_path / "cert.txt"
    wrong_ext_cert.write_text("{}", encoding="utf-8")
    resp_c2 = hardening_client.post(
        "/api/v1/verify-release",
        json={"certificate_path": str(wrong_ext_cert)},
    )
    assert resp_c2.status_code == 400
    assert ".json extension" in resp_c2.json()["detail"]

    # 3. Evidence dir non-existent -> 404
    cert_file = tmp_path / "cert.json"
    cert_file.write_text("{}", encoding="utf-8")
    resp_e1 = hardening_client.post(
        "/api/v1/verify-release",
        json={
            "certificate_path": str(cert_file),
            "evidence_dir": str(tmp_path / "non_existent_dir"),
        },
    )
    assert resp_e1.status_code == 404
    assert "does not exist" in resp_e1.json()["detail"]

    # 4. Evidence dir not a dir or zip -> 400
    evidence_file = tmp_path / "evidence.txt"
    evidence_file.write_text("raw text", encoding="utf-8")
    resp_e2 = hardening_client.post(
        "/api/v1/verify-release",
        json={
            "certificate_path": str(cert_file),
            "evidence_dir": str(evidence_file),
        },
    )
    assert resp_e2.status_code == 400
    assert "must be a directory or .zip" in resp_e2.json()["detail"]

    # 5. Non-existent bundle path -> 404
    resp_b1 = hardening_client.post(
        "/api/v1/verify-bundle",
        json={"bundle_path": str(tmp_path / "missing_bundle.zip")},
    )
    assert resp_b1.status_code == 404
    assert "not found" in resp_b1.json()["detail"]

    # 6. Bundle path with wrong extension -> 400
    bad_bundle_ext = tmp_path / "bundle.tar"
    bad_bundle_ext.write_text("dummy", encoding="utf-8")
    resp_b2 = hardening_client.post(
        "/api/v1/verify-bundle",
        json={"bundle_path": str(bad_bundle_ext)},
    )
    assert resp_b2.status_code == 400
    assert ".zip extension" in resp_b2.json()["detail"]


def test_api_cors_preflight_and_host_validation(hardening_client: TestClient):
    """Verify CORS preflight handling and Host header validation."""
    # 1. Allowed origin preflight -> 200 with Allow-Origin header
    resp_preflight = hardening_client.options(
        "/api/v1/evaluations",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp_preflight.status_code == 200
    assert resp_preflight.headers.get("access-control-allow-origin") == "http://localhost:5173"

    # 2. Blocked origin preflight -> Starlette CORSMiddleware rejects with 400 Disallowed
    resp_bad_preflight = hardening_client.options(
        "/api/v1/evaluations",
        headers={
            "Origin": "http://malicious-external-site.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp_bad_preflight.status_code == 400
    assert resp_bad_preflight.headers.get("access-control-allow-origin") is None
