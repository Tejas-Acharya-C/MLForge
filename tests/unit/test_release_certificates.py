"""Unit Tests for Cryptographically Signed Release Certificates (Milestone 4.3).

Verifies Ed25519 digital signing, deterministic canonicalization, multi-tiered verification,
tampering detection, signer trust boundaries, atomic file generation, and HTML report integration.
"""

import copy
import json
from pathlib import Path
import zipfile
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlforge.application.artifacts import serialize_run_artifacts
from mlforge.application.packaging import build_audit_bundle
from mlforge.application.release_certificates import (
    CERTIFICATE_SCHEMA_VERSION,
    canonicalize_certificate_payload,
    deserialize_public_key,
    generate_release_keypair,
    issue_release_certificate,
    serialize_public_key,
    verify_release_certificate,
)
from mlforge.application.reporting import generate_html_report
from mlforge.core.contracts import (
    AcceptanceDecision,
    AcceptanceOutcome,
    AcceptancePolicy,
    AcceptanceRule,
    CriterionEvaluation,
    CriterionOutcome,
    CriterionSeverity,
    FoldResult,
    MetricValue,
    ReleaseCertificate,
    ReleaseGateStatus,
    ReleasePolicy,
    RuleOperator,
)
from mlforge.core.exceptions import (
    ArtifactSerializationError,
    CertificateError,
    CertificateSecurityError,
)
from tests.unit.test_acceptance import _make_dummy_evaluation


@pytest.fixture
def cert_passing_run(tmp_path: Path) -> Path:
    """Create a fully compliant run with passing acceptance decision and candidate model pipeline."""
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
        decision_id="dec_passing_01",
        evaluation_id="eval_01",
        policy_id=policy.policy_id,
        policy_version=policy.version,
        outcome=AcceptanceOutcome.PASS,
        criteria_evaluations=[crit],
        summary_rationale="All blocker criteria satisfied.",
        evaluated_at="2026-10-19T10:00:00Z",
        passed_rules_count=1,
        failed_rules_count=0,
        inconclusive_rules_count=0,
        hash_signature="sig_passing_01",
    )

    run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        acceptance_decision=decision,
        policy=policy,
        artifacts_dir=tmp_path / "runs",
        run_id="run_cert_test",
    )
    return run_dir


def test_issue_release_certificate_success(cert_passing_run: Path):
    """Issuing a release certificate on a passing run generates a valid signed certificate file."""
    priv_key, pub_key, key_id = generate_release_keypair()

    cert = issue_release_certificate(
        source=cert_passing_run,
        private_key=priv_key,
        key_id=key_id,
        release_policy=ReleasePolicy(require_acceptance_pass=True),
    )

    assert isinstance(cert, ReleaseCertificate)
    assert cert.release_gate_outcome == ReleaseGateStatus.PASSED
    assert cert.key_id == key_id
    assert cert.signature_algorithm == "Ed25519"
    assert len(cert.signature) == 128  # 64 bytes = 128 hex chars
    assert len(cert.payload_digest_sha256) == 64

    # Verify certificate file exists in expected directory
    cert_file = cert_passing_run / "release" / "release_certificate.json"
    assert cert_file.exists()


def test_issue_release_certificate_denied_for_blocked_run(cert_passing_run: Path):
    """Certificate issuance is denied with CertificateSecurityError if release gate fails."""
    # Alter decision to FAIL
    dec_path = cert_passing_run / "decision.json"
    with open(dec_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["outcome"] = "fail"
    with open(dec_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    priv_key, pub_key, key_id = generate_release_keypair()

    with pytest.raises(CertificateSecurityError, match="authorization denied"):
        issue_release_certificate(
            source=cert_passing_run,
            private_key=priv_key,
            key_id=key_id,
            release_policy=ReleasePolicy(require_acceptance_pass=True, require_manifest_integrity=False),
        )

    # Ensure certificate file was NOT written
    cert_file = cert_passing_run / "release" / "release_certificate.json"
    assert not cert_file.exists()


def test_certificate_verification_success_with_trusted_key(cert_passing_run: Path):
    """Verification passes across all tiers when trusted public key is provided."""
    priv_key, pub_key, key_id = generate_release_keypair()

    cert = issue_release_certificate(
        source=cert_passing_run,
        private_key=priv_key,
        key_id=key_id,
        release_policy=ReleasePolicy(require_acceptance_pass=True),
    )

    trusted_keys = {key_id: pub_key}
    ver_res = verify_release_certificate(
        certificate=cert,
        trusted_keys=trusted_keys,
        expected_source=cert_passing_run,
    )

    assert ver_res.payload_integrity is True
    assert ver_res.signature_validity is True
    assert ver_res.signer_trust is True
    assert ver_res.evidence_consistency is True
    assert ver_res.release_authorized is True
    assert len(ver_res.errors) == 0


def test_certificate_signature_verification_fails_after_payload_tampering(cert_passing_run: Path):
    """Modifying any signed field in the certificate invalidates both payload integrity and signature."""
    priv_key, pub_key, key_id = generate_release_keypair()

    cert = issue_release_certificate(
        source=cert_passing_run,
        private_key=priv_key,
        key_id=key_id,
        release_policy=ReleasePolicy(require_acceptance_pass=True),
    )

    # Tamper with dataset checksum
    tampered_dict = cert.model_dump()
    tampered_dict["dataset_checksum"] = "tampered_dataset_hash"
    tampered_cert = ReleaseCertificate.model_validate(tampered_dict)

    trusted_keys = {key_id: pub_key}
    ver_res = verify_release_certificate(tampered_cert, trusted_keys=trusted_keys)

    assert ver_res.payload_integrity is False
    assert ver_res.signature_validity is False
    assert ver_res.release_authorized is False
    assert any("Payload integrity violation" in err for err in ver_res.errors)


def test_certificate_verification_fails_with_wrong_public_key(cert_passing_run: Path):
    """Verifying against a different public key fails cryptographic signature check."""
    priv_key1, pub_key1, key_id1 = generate_release_keypair()
    priv_key2, pub_key2, key_id2 = generate_release_keypair()

    cert = issue_release_certificate(
        source=cert_passing_run,
        private_key=priv_key1,
        key_id=key_id1,
        release_policy=ReleasePolicy(require_acceptance_pass=True),
    )

    # Supply pub_key2 registered under key_id1
    wrong_trusted = {key_id1: pub_key2}
    ver_res = verify_release_certificate(cert, trusted_keys=wrong_trusted)

    assert ver_res.payload_integrity is True
    assert ver_res.signature_validity is False
    assert ver_res.release_authorized is False
    assert any("signature verification failed" in err for err in ver_res.errors)


def test_certificate_untrusted_signer_does_not_authorize_release(cert_passing_run: Path):
    """A valid signature from an untrusted key (not in trusted_keys) results in signer_trust=False."""
    priv_key, pub_key, key_id = generate_release_keypair()

    cert = issue_release_certificate(
        source=cert_passing_run,
        private_key=priv_key,
        key_id=key_id,
        release_policy=ReleasePolicy(require_acceptance_pass=True),
    )

    # Empty trusted keys registry
    ver_res = verify_release_certificate(cert, trusted_keys={})

    assert ver_res.payload_integrity is True
    assert ver_res.signature_validity is False  # Cannot verify without key in registry
    assert ver_res.signer_trust is False
    assert ver_res.release_authorized is False
    assert any("NOT present in supplied trusted keys" in w for w in ver_res.warnings)


def test_certificate_evidence_mismatch_fails_consistency(cert_passing_run: Path, tmp_path: Path):
    """If saved run on disk does not match certificate claims, evidence_consistency fails."""
    priv_key, pub_key, key_id = generate_release_keypair()

    cert = issue_release_certificate(
        source=cert_passing_run,
        private_key=priv_key,
        key_id=key_id,
        release_policy=ReleasePolicy(require_acceptance_pass=True),
    )

    # Tamper with physical file on disk (alter evaluation.json dataset hash)
    eval_file = cert_passing_run / "evaluation.json"
    data = json.loads(eval_file.read_text(encoding="utf-8"))
    data["dataset_metadata"]["checksum_sha256"] = "different_dataset_checksum"
    eval_file.write_text(json.dumps(data), encoding="utf-8")

    trusted_keys = {key_id: pub_key}
    ver_res = verify_release_certificate(
        cert,
        trusted_keys=trusted_keys,
        expected_source=cert_passing_run,
    )

    assert ver_res.evidence_consistency is False
    assert ver_res.release_authorized is False
    assert any("Dataset checksum mismatch" in err for err in ver_res.errors)


def test_canonicalize_payload_rejects_nan_and_infinity():
    """Canonicalization strictly rejects NaN and Infinity floating point values."""
    payload_nan = {"metric": float("nan"), "status": "ok"}
    with pytest.raises(CertificateSecurityError, match="strictly reject NaN and Infinity"):
        canonicalize_certificate_payload(payload_nan)

    payload_inf = {"metric": float("inf"), "status": "ok"}
    with pytest.raises(CertificateSecurityError, match="strictly reject NaN and Infinity"):
        canonicalize_certificate_payload(payload_inf)


def test_canonicalize_payload_deterministic_ordering():
    """Canonicalization guarantees bit-for-bit identical bytes regardless of dict key insertion order."""
    dict1 = {"b": 2, "a": 1, "c": {"z": 10, "y": 9}}
    dict2 = {"a": 1, "c": {"y": 9, "z": 10}, "b": 2}

    b1 = canonicalize_certificate_payload(dict1)
    b2 = canonicalize_certificate_payload(dict2)

    assert b1 == b2
    assert b1 == b'{"a":1,"b":2,"c":{"y":9,"z":10}}'


def test_public_key_hex_serialization_roundtrip():
    """Ed25519 public keys serialize to hex and deserialize losslessly."""
    _, pub_key, _ = generate_release_keypair()
    hex_str = serialize_public_key(pub_key)

    assert len(hex_str) == 64
    recovered_pub = deserialize_public_key(hex_str)
    assert recovered_pub.public_bytes_raw() == pub_key.public_bytes_raw()


def test_private_key_never_written_to_certificate(cert_passing_run: Path):
    """Private key material is never persisted into the release certificate JSON."""
    priv_key, pub_key, key_id = generate_release_keypair()

    cert = issue_release_certificate(
        source=cert_passing_run,
        private_key=priv_key,
        key_id=key_id,
        release_policy=ReleasePolicy(require_acceptance_pass=True),
    )

    cert_json_text = (cert_passing_run / "release" / "release_certificate.json").read_text(encoding="utf-8")
    priv_raw_hex = priv_key.private_bytes_raw().hex()

    assert priv_raw_hex not in cert_json_text
    assert "private" not in cert_json_text.lower()


def test_issue_release_certificate_overwrite_policy(cert_passing_run: Path):
    """Attempting to overwrite an existing certificate raises ArtifactSerializationError unless overwrite=True."""
    priv_key, pub_key, key_id = generate_release_keypair()

    issue_release_certificate(
        source=cert_passing_run,
        private_key=priv_key,
        key_id=key_id,
        release_policy=ReleasePolicy(require_acceptance_pass=True),
        overwrite=False,
    )

    # Second issuance without overwrite=True must fail
    with pytest.raises(ArtifactSerializationError, match="already exists"):
        issue_release_certificate(
            source=cert_passing_run,
            private_key=priv_key,
            key_id=key_id,
            release_policy=ReleasePolicy(require_acceptance_pass=True),
            overwrite=False,
        )

    # Issuance with overwrite=True succeeds
    cert2 = issue_release_certificate(
        source=cert_passing_run,
        private_key=priv_key,
        key_id=key_id,
        release_policy=ReleasePolicy(require_acceptance_pass=True),
        overwrite=True,
    )
    assert isinstance(cert2, ReleaseCertificate)


def test_html_report_renders_release_certificate_when_present(cert_passing_run: Path):
    """When a release certificate exists in the run directory, the HTML report displays it with neutral wording."""
    priv_key, pub_key, key_id = generate_release_keypair()

    cert = issue_release_certificate(
        source=cert_passing_run,
        private_key=priv_key,
        key_id=key_id,
        release_policy=ReleasePolicy(require_acceptance_pass=True),
    )

    report_path = generate_html_report(cert_passing_run)
    html_text = report_path.read_text(encoding="utf-8")

    assert "Signed Release Certificate" in html_text
    assert "CERTIFICATE ISSUED &mdash; VERIFICATION NOT ESTABLISHED" in html_text or "VERIFICATION NOT ESTABLISHED" in html_text
    # Must NOT contain a misleading green authorization badge
    assert '<span class="badge badge-success">RELEASED</span>' not in html_text
    assert cert.certificate_id in html_text
    assert cert.payload_digest_sha256 in html_text
    assert "Ed25519" in html_text
    assert key_id in html_text


def test_html_report_shows_no_certificate_when_absent(cert_passing_run: Path):
    """When no certificate exists, HTML report clearly indicates NO CERTIFICATE ISSUED."""
    cert_file = cert_passing_run / "release" / "release_certificate.json"
    if cert_file.exists():
        cert_file.unlink()

    report_path = generate_html_report(cert_passing_run)
    html_text = report_path.read_text(encoding="utf-8")

    assert "NO CERTIFICATE ISSUED" in html_text
    assert "A passing acceptance decision alone does not authorize model release" in html_text


def test_verify_certificate_fails_closed_when_expected_source_is_none(cert_passing_run: Path):
    """When expected_source is None, evidence consistency fails and release is never authorized."""
    priv_key, pub_key, key_id = generate_release_keypair()

    cert = issue_release_certificate(
        source=cert_passing_run,
        private_key=priv_key,
        key_id=key_id,
        release_policy=ReleasePolicy(require_acceptance_pass=True),
    )

    trusted_keys = {key_id: pub_key}
    ver_res = verify_release_certificate(
        certificate=cert,
        trusted_keys=trusted_keys,
        expected_source=None,  # Omitted source
    )

    assert ver_res.payload_integrity is True
    assert ver_res.signature_validity is True
    assert ver_res.signer_trust is True
    assert ver_res.evidence_consistency is False
    assert ver_res.release_authorized is False
    assert any("expected_source was not provided" in err for err in ver_res.errors)


def test_verify_certificate_with_valid_zip_audit_bundle(cert_passing_run: Path, tmp_path: Path):
    """Release certificate can be verified against a valid .zip audit bundle."""
    priv_key, pub_key, key_id = generate_release_keypair()

    cert = issue_release_certificate(
        source=cert_passing_run,
        private_key=priv_key,
        key_id=key_id,
        release_policy=ReleasePolicy(require_acceptance_pass=True),
    )

    bundle_zip = build_audit_bundle(cert_passing_run, output_bundle_path=tmp_path / "valid_bundle.zip")

    trusted_keys = {key_id: pub_key}
    ver_res = verify_release_certificate(
        certificate=cert,
        trusted_keys=trusted_keys,
        expected_source=bundle_zip,
    )

    assert ver_res.payload_integrity is True
    assert ver_res.signature_validity is True
    assert ver_res.signer_trust is True
    assert ver_res.evidence_consistency is True
    assert ver_res.release_authorized is True
    assert len(ver_res.errors) == 0


def test_verify_certificate_fails_when_zip_bundle_is_tampered(cert_passing_run: Path, tmp_path: Path):
    """A tampered ZIP audit bundle fails evidence verification and denies release authorization."""
    priv_key, pub_key, key_id = generate_release_keypair()

    cert = issue_release_certificate(
        source=cert_passing_run,
        private_key=priv_key,
        key_id=key_id,
        release_policy=ReleasePolicy(require_acceptance_pass=True),
    )

    bundle_zip = build_audit_bundle(cert_passing_run, output_bundle_path=tmp_path / "valid_bundle.zip")

    # Create tampered zip where evaluation.json is modified
    tampered_zip = tmp_path / "tampered_bundle.zip"
    with zipfile.ZipFile(bundle_zip, "r") as zf_in, zipfile.ZipFile(tampered_zip, "w") as zf_out:
        for item in zf_in.infolist():
            content = zf_in.read(item.filename)
            if item.filename == "evaluation.json":
                content = b'{"tampered": true}'
            zf_out.writestr(item, content)

    trusted_keys = {key_id: pub_key}
    ver_res = verify_release_certificate(
        certificate=cert,
        trusted_keys=trusted_keys,
        expected_source=tampered_zip,
    )

    assert ver_res.evidence_consistency is False
    assert ver_res.release_authorized is False
    assert len(ver_res.errors) > 0


def test_verify_certificate_fails_when_expected_source_does_not_exist(cert_passing_run: Path):
    """Non-existent expected source fails closed with evidence_consistency=False."""
    priv_key, pub_key, key_id = generate_release_keypair()

    cert = issue_release_certificate(
        source=cert_passing_run,
        private_key=priv_key,
        key_id=key_id,
        release_policy=ReleasePolicy(require_acceptance_pass=True),
    )

    trusted_keys = {key_id: pub_key}
    ver_res = verify_release_certificate(
        certificate=cert,
        trusted_keys=trusted_keys,
        expected_source=cert_passing_run / "non_existent_directory",
    )

    assert ver_res.evidence_consistency is False
    assert ver_res.release_authorized is False
    assert any("does not exist" in err for err in ver_res.errors)


def test_verify_certificate_fails_on_candidate_model_type_mismatch(cert_passing_run: Path):
    """Mismatch between certificate model type and evidence model type invalidates evidence consistency."""
    priv_key, pub_key, key_id = generate_release_keypair()

    cert = issue_release_certificate(
        source=cert_passing_run,
        private_key=priv_key,
        key_id=key_id,
        release_policy=ReleasePolicy(require_acceptance_pass=True),
    )

    # Alter candidate_model_type in evaluation.json
    eval_file = cert_passing_run / "evaluation.json"
    eval_data = json.loads(eval_file.read_text(encoding="utf-8"))
    eval_data["candidate_result"]["model_type"] = "random_forest_classifier"
    eval_file.write_text(json.dumps(eval_data), encoding="utf-8")

    trusted_keys = {key_id: pub_key}
    ver_res = verify_release_certificate(
        certificate=cert,
        trusted_keys=trusted_keys,
        expected_source=cert_passing_run,
    )

    assert ver_res.evidence_consistency is False
    assert ver_res.release_authorized is False
    assert any("Candidate model type mismatch" in err for err in ver_res.errors)


def test_verify_certificate_fails_on_acceptance_decision_outcome_mismatch(cert_passing_run: Path):
    """Mismatch between certificate acceptance outcome and decision.json invalidates evidence consistency."""
    priv_key, pub_key, key_id = generate_release_keypair()

    cert = issue_release_certificate(
        source=cert_passing_run,
        private_key=priv_key,
        key_id=key_id,
        release_policy=ReleasePolicy(require_acceptance_pass=True),
    )

    # Alter decision.json outcome to fail
    dec_file = cert_passing_run / "decision.json"
    dec_data = json.loads(dec_file.read_text(encoding="utf-8"))
    dec_data["outcome"] = "fail"
    dec_file.write_text(json.dumps(dec_data), encoding="utf-8")

    trusted_keys = {key_id: pub_key}
    ver_res = verify_release_certificate(
        certificate=cert,
        trusted_keys=trusted_keys,
        expected_source=cert_passing_run,
    )

    assert ver_res.evidence_consistency is False
    assert ver_res.release_authorized is False
    assert any("Acceptance outcome mismatch" in err for err in ver_res.errors)
