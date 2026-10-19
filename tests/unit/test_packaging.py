"""Unit Tests for Audit Bundle Packaging & Verification (Milestone 4.1).

Verifies portable ZIP packaging, deterministic manifests, SHA-256 member checksums,
zip-slip path traversal rejection, tampering detection, and verification without model deserialization.
"""

import hashlib
import json
import os
from pathlib import Path
import zipfile
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlforge.application.artifacts import serialize_run_artifacts
from mlforge.application.packaging import (
    CURRENT_BUNDLE_SCHEMA_VERSION,
    build_audit_bundle,
    extract_audit_bundle,
    normalize_bundle_path,
    validate_archive_member_safety,
    verify_audit_bundle,
)
from mlforge.core.contracts import (
    AcceptanceDecision,
    AcceptanceOutcome,
    AcceptancePolicy,
    AcceptanceRule,
    CriterionEvaluation,
    CriterionOutcome,
    CriterionSeverity,
    RuleOperator,
)
from mlforge.core.exceptions import (
    AuditBundleCorruptError,
    AuditBundleError,
    AuditBundleSecurityError,
    AuditBundleTamperError,
)
from tests.unit.test_acceptance import _make_dummy_evaluation


@pytest.fixture
def sample_evaluated_run_dir(tmp_path: Path) -> Path:
    """Create a realistic serialized run directory with evaluation, decision, policy, and model."""
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=42))])
    pipe.fit([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], [0, 0, 1, 1])

    evaluation = _make_dummy_evaluation(cand_f1=0.85, cand_acc=0.88, ece=0.05)
    evaluation.candidate_result.fitted_pipeline = pipe

    policy = AcceptancePolicy(
        policy_id="test_bundle_pol",
        version="1.0.0",
        name="Test Bundle Policy",
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
        decision_id="dec_bundle_01",
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
        run_id="run_for_packaging_01",
    )
    return run_dir


# ============================================================================
# Path Safety & Normalization Tests
# ============================================================================

def test_normalize_bundle_path_clean_cases():
    """Verify clean POSIX relative path normalization."""
    assert normalize_bundle_path("evaluation.json") == "evaluation.json"
    assert normalize_bundle_path("models/candidate_pipeline.joblib") == "models/candidate_pipeline.joblib"
    assert normalize_bundle_path("models\\candidate_pipeline.joblib") == "models/candidate_pipeline.joblib"
    assert normalize_bundle_path("./models//nested/../candidate_pipeline.joblib") == "models/candidate_pipeline.joblib"


def test_normalize_bundle_path_rejects_unsafe_inputs():
    """Verify rejection of null bytes, empty strings, drive letters, and path traversal."""
    with pytest.raises(AuditBundleSecurityError, match="null bytes"):
        normalize_bundle_path("models/\0candidate.joblib")

    with pytest.raises(AuditBundleSecurityError, match="empty"):
        normalize_bundle_path("   ")

    with pytest.raises(AuditBundleSecurityError, match="drive or protocol"):
        normalize_bundle_path("C:/models/candidate.joblib")

    with pytest.raises(AuditBundleSecurityError, match="absolute path"):
        normalize_bundle_path("/etc/passwd")

    with pytest.raises(AuditBundleSecurityError, match="path traversal"):
        normalize_bundle_path("../outside.json")

    with pytest.raises(AuditBundleSecurityError, match="path traversal"):
        normalize_bundle_path("models/../../outside.json")


def test_validate_archive_member_safety_detects_duplicates():
    """Reject duplicate normalized member paths within the same archive."""
    seen: set = set()
    validate_archive_member_safety("models/candidate_pipeline.joblib", seen_paths=seen)

    with pytest.raises(AuditBundleSecurityError, match="duplicate normalized"):
        validate_archive_member_safety("models//candidate_pipeline.joblib", seen_paths=seen)

    with pytest.raises(AuditBundleSecurityError, match="duplicate normalized"):
        validate_archive_member_safety("models/./candidate_pipeline.joblib", seen_paths=seen)


# ============================================================================
# Bundle Build & Verification Lifecycle Tests
# ============================================================================

def test_build_and_verify_audit_bundle_success(sample_evaluated_run_dir: Path, tmp_path: Path):
    """Build audit bundle, verify internal cryptographic consistency, and inspect manifest."""
    bundle_zip = tmp_path / "bundles" / "test_bundle.zip"
    created_path = build_audit_bundle(
        run_dir=sample_evaluated_run_dir,
        output_bundle_path=bundle_zip,
        deterministic=True,
    )
    assert created_path.exists()
    assert created_path == bundle_zip

    # Verify bundle
    result = verify_audit_bundle(created_path, raise_on_error=True)
    assert result.is_valid is True
    assert result.run_id == "run_for_packaging_01"
    assert "manifest.json" in result.verified_files
    assert "evaluation.json" in result.verified_files
    assert "decision.json" in result.verified_files
    assert "policy.json" in result.verified_files
    assert "models/candidate_pipeline.joblib" in result.verified_files
    assert result.manifest is not None
    assert result.manifest.schema_version == CURRENT_BUNDLE_SCHEMA_VERSION
    assert result.manifest.has_models is True
    assert result.manifest.decision_outcome == AcceptanceOutcome.PASS


def test_audit_bundle_safe_extraction_roundtrip(sample_evaluated_run_dir: Path, tmp_path: Path):
    """Safely extract audit bundle into temporary directory and verify byte-for-byte fidelity."""
    bundle_zip = tmp_path / "test_extract.zip"
    build_audit_bundle(sample_evaluated_run_dir, output_bundle_path=bundle_zip)

    extract_dir = tmp_path / "extracted"
    extracted_path = extract_audit_bundle(bundle_zip, destination_dir=extract_dir, verify_first=True)
    assert extracted_path == extract_dir

    # Check that all files exist in extract_dir
    assert (extract_dir / "bundle_manifest.json").exists()
    assert (extract_dir / "manifest.json").exists()
    assert (extract_dir / "evaluation.json").exists()
    assert (extract_dir / "decision.json").exists()
    assert (extract_dir / "policy.json").exists()
    assert (extract_dir / "models" / "candidate_pipeline.joblib").exists()

    # Compare SHA-256 with source run_dir
    for fname in ["manifest.json", "evaluation.json", "decision.json", "policy.json", "models/candidate_pipeline.joblib"]:
        src_bytes = (sample_evaluated_run_dir / fname).read_bytes()
        dst_bytes = (extract_dir / fname).read_bytes()
        assert hashlib.sha256(src_bytes).hexdigest() == hashlib.sha256(dst_bytes).hexdigest()


def test_audit_bundle_verification_never_deserializes_models(sample_evaluated_run_dir: Path, tmp_path: Path, monkeypatch):
    """Verify that verify_audit_bundle NEVER calls joblib.load or pickle.load."""
    bundle_zip = tmp_path / "no_unpickle.zip"
    build_audit_bundle(sample_evaluated_run_dir, output_bundle_path=bundle_zip)

    # Monkeypatch joblib.load and pickle.load to fail immediately
    import pickle
    import joblib

    def forbidden_load(*args, **kwargs):
        raise AssertionError("SECURITY VIOLATION: Deserialization API was called during bundle verification!")

    monkeypatch.setattr(joblib, "load", forbidden_load)
    monkeypatch.setattr(pickle, "load", forbidden_load)
    monkeypatch.setattr(pickle, "loads", forbidden_load)

    # Must succeed without triggering forbidden_load
    result = verify_audit_bundle(bundle_zip, raise_on_error=True)
    assert result.is_valid is True


# ============================================================================
# Tampering, Corruption & Security Defense Tests
# ============================================================================

def test_audit_bundle_detects_tampered_member_bytes(sample_evaluated_run_dir: Path, tmp_path: Path):
    """Tampering with bytes of any member file in ZIP archive raises AuditBundleTamperError."""
    bundle_zip = tmp_path / "tamper_test.zip"
    build_audit_bundle(sample_evaluated_run_dir, output_bundle_path=bundle_zip)

    # Tamper with evaluation.json inside zip
    tampered_zip = tmp_path / "tampered.zip"
    with zipfile.ZipFile(bundle_zip, "r") as zf_in, zipfile.ZipFile(tampered_zip, "w") as zf_out:
        for item in zf_in.infolist():
            content = zf_in.read(item.filename)
            if item.filename == "evaluation.json":
                # Flip single byte without altering length to trigger cryptographic checksum mismatch
                flipped_byte = b"X" if content[-1:] != b"X" else b"Y"
                content = content[:-1] + flipped_byte
            zf_out.writestr(item, content)

    # 1. raise_on_error=True raises AuditBundleTamperError
    with pytest.raises(AuditBundleTamperError, match="checksum mismatch"):
        verify_audit_bundle(tampered_zip, raise_on_error=True)

    # 2. raise_on_error=False returns structured failure report
    res = verify_audit_bundle(tampered_zip, raise_on_error=False)
    assert res.is_valid is False
    assert any("checksum mismatch" in err for err in res.errors)


def test_audit_bundle_detects_tampered_model_bytes(sample_evaluated_run_dir: Path, tmp_path: Path):
    """Tampering with bytes of serialized model file (.joblib) raises AuditBundleTamperError."""
    bundle_zip = tmp_path / "model_tamper.zip"
    build_audit_bundle(sample_evaluated_run_dir, output_bundle_path=bundle_zip)

    tampered_zip = tmp_path / "model_tampered.zip"
    with zipfile.ZipFile(bundle_zip, "r") as zf_in, zipfile.ZipFile(tampered_zip, "w") as zf_out:
        for item in zf_in.infolist():
            content = zf_in.read(item.filename)
            if "candidate_pipeline.joblib" in item.filename:
                content = b"corrupted payload"
            zf_out.writestr(item, content)

    with pytest.raises(AuditBundleTamperError, match="checksum mismatch|Size mismatch"):
        verify_audit_bundle(tampered_zip, raise_on_error=True)


def test_audit_bundle_detects_missing_member(sample_evaluated_run_dir: Path, tmp_path: Path):
    """Archive missing a member declared in bundle manifest raises AuditBundleTamperError."""
    bundle_zip = tmp_path / "missing_member_base.zip"
    build_audit_bundle(sample_evaluated_run_dir, output_bundle_path=bundle_zip)

    omitted_zip = tmp_path / "omitted.zip"
    with zipfile.ZipFile(bundle_zip, "r") as zf_in, zipfile.ZipFile(omitted_zip, "w") as zf_out:
        for item in zf_in.infolist():
            if item.filename != "evaluation.json":
                zf_out.writestr(item, zf_in.read(item.filename))

    with pytest.raises(AuditBundleTamperError, match="missing declared member files"):
        verify_audit_bundle(omitted_zip, raise_on_error=True)


def test_audit_bundle_detects_unindexed_unexpected_member(sample_evaluated_run_dir: Path, tmp_path: Path):
    """Archive containing extra file not declared in bundle manifest raises AuditBundleTamperError."""
    bundle_zip = tmp_path / "unexpected_base.zip"
    build_audit_bundle(sample_evaluated_run_dir, output_bundle_path=bundle_zip)

    injected_zip = tmp_path / "injected.zip"
    with zipfile.ZipFile(bundle_zip, "r") as zf_in, zipfile.ZipFile(injected_zip, "w") as zf_out:
        for item in zf_in.infolist():
            zf_out.writestr(item, zf_in.read(item.filename))
        zf_out.writestr("unauthorized_script.py", b"print('unauthorized')")

    with pytest.raises(AuditBundleTamperError, match="unindexed unexpected files"):
        verify_audit_bundle(injected_zip, raise_on_error=True)


def test_audit_bundle_detects_corrupted_zip(tmp_path: Path):
    """Corrupted or non-ZIP file raises AuditBundleCorruptError."""
    bad_zip = tmp_path / "corrupted.zip"
    bad_zip.write_bytes(b"PK\x03\x04not a valid zip structure definitely corrupt")

    with pytest.raises(AuditBundleCorruptError, match="not a valid or readable ZIP"):
        verify_audit_bundle(bad_zip, raise_on_error=True)


def test_audit_bundle_detects_malformed_manifest(sample_evaluated_run_dir: Path, tmp_path: Path):
    """Malformed bundle_manifest.json raises AuditBundleCorruptError."""
    bundle_zip = tmp_path / "malformed_manifest_base.zip"
    build_audit_bundle(sample_evaluated_run_dir, output_bundle_path=bundle_zip)

    broken_zip = tmp_path / "broken_manifest.zip"
    with zipfile.ZipFile(bundle_zip, "r") as zf_in, zipfile.ZipFile(broken_zip, "w") as zf_out:
        for item in zf_in.infolist():
            if item.filename == "bundle_manifest.json":
                zf_out.writestr(item, b"{'broken': json, invalid")
            else:
                zf_out.writestr(item, zf_in.read(item.filename))

    with pytest.raises(AuditBundleCorruptError, match="Invalid or malformed 'bundle_manifest.json'"):
        verify_audit_bundle(broken_zip, raise_on_error=True)


def test_audit_bundle_rejects_path_traversal(sample_evaluated_run_dir: Path, tmp_path: Path):
    """Archive entry with path traversal (zip slip) raises AuditBundleSecurityError."""
    malicious_zip = tmp_path / "zip_slip.zip"
    with zipfile.ZipFile(malicious_zip, "w") as zf:
        zf.writestr("../evil.txt", b"evil contents")
        zf.writestr("bundle_manifest.json", b"{}")

    with pytest.raises(AuditBundleSecurityError, match="attempts path traversal"):
        verify_audit_bundle(malicious_zip, raise_on_error=True)


def test_audit_bundle_rejects_absolute_path_member(sample_evaluated_run_dir: Path, tmp_path: Path):
    """Archive entry with absolute path raises AuditBundleSecurityError."""
    malicious_zip = tmp_path / "abs_path.zip"
    with zipfile.ZipFile(malicious_zip, "w") as zf:
        zf.writestr("/evil.txt", b"evil contents")
        zf.writestr("bundle_manifest.json", b"{}")

    with pytest.raises(AuditBundleSecurityError, match="must not be an absolute path"):
        verify_audit_bundle(malicious_zip, raise_on_error=True)


def test_audit_bundle_rejects_duplicate_normalized_paths(sample_evaluated_run_dir: Path, tmp_path: Path):
    """Archive with duplicate normalized paths raises AuditBundleSecurityError."""
    dup_zip = tmp_path / "dup_paths.zip"
    with zipfile.ZipFile(dup_zip, "w") as zf:
        zf.writestr("models/candidate_pipeline.joblib", b"content1")
        zf.writestr("models//candidate_pipeline.joblib", b"content2")
        zf.writestr("bundle_manifest.json", b"{}")

    with pytest.raises(AuditBundleSecurityError, match="duplicate normalized member path"):
        verify_audit_bundle(dup_zip, raise_on_error=True)


def test_audit_bundle_atomic_creation_failure_cleanup(sample_evaluated_run_dir: Path, tmp_path: Path, monkeypatch):
    """Failed bundle creation must clean up temporary file and leave no partial target file."""
    target_zip = tmp_path / "atomic_target.zip"

    # Break open during writing by monkeypatching zipfile.ZipFile.writestr
    orig_writestr = zipfile.ZipFile.writestr

    def failing_writestr(self, zinfo_or_arcname, *args, **kwargs):
        name = getattr(zinfo_or_arcname, "filename", str(zinfo_or_arcname))
        if name == "evaluation.json":
            raise IOError("Simulated disk write failure")
        return orig_writestr(self, zinfo_or_arcname, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "writestr", failing_writestr)

    with pytest.raises(AuditBundleError, match="Failed to create audit bundle"):
        build_audit_bundle(sample_evaluated_run_dir, output_bundle_path=target_zip)

    assert not target_zip.exists()
    # Check no lingering temp files in directory
    lingering_tmp = list(tmp_path.glob("*.tmp_*"))
    assert len(lingering_tmp) == 0


def test_audit_bundle_deterministic_builds(sample_evaluated_run_dir: Path, tmp_path: Path):
    """Two deterministic builds of the same run produce bit-for-bit identical ZIP archives."""
    bundle_a = tmp_path / "bundle_a.zip"
    bundle_b = tmp_path / "bundle_b.zip"

    build_audit_bundle(sample_evaluated_run_dir, output_bundle_path=bundle_a, deterministic=True)
    build_audit_bundle(sample_evaluated_run_dir, output_bundle_path=bundle_b, deterministic=True)

    bytes_a = bundle_a.read_bytes()
    bytes_b = bundle_b.read_bytes()

    assert hashlib.sha256(bytes_a).hexdigest() == hashlib.sha256(bytes_b).hexdigest()


def test_extract_audit_bundle_without_verify_first_rejects_traversal(tmp_path: Path):
    """extract_audit_bundle must reject path traversal entries even when verify_first=False."""
    malicious_zip = tmp_path / "zip_slip.zip"
    dest_dir = tmp_path / "extracted"

    with zipfile.ZipFile(malicious_zip, "w") as zf:
        zinfo = zipfile.ZipInfo(filename="../../evil.txt")
        zf.writestr(zinfo, b"malicious content escaping dest")

    with pytest.raises(AuditBundleSecurityError, match="attempts path traversal outside archive root"):
        extract_audit_bundle(malicious_zip, destination_dir=dest_dir, verify_first=False)

    assert not (tmp_path / "evil.txt").exists()


def test_extract_audit_bundle_without_verify_first_rejects_absolute_path(tmp_path: Path):
    """extract_audit_bundle must reject absolute path entries even when verify_first=False."""
    malicious_zip = tmp_path / "absolute_member.zip"
    dest_dir = tmp_path / "extracted"

    with zipfile.ZipFile(malicious_zip, "w") as zf:
        zinfo = zipfile.ZipInfo(filename="/etc/shadow")
        zf.writestr(zinfo, b"evil")

    with pytest.raises(AuditBundleSecurityError, match="must not be an absolute path"):
        extract_audit_bundle(malicious_zip, destination_dir=dest_dir, verify_first=False)


def test_audit_bundle_resource_limits_archive_size(sample_evaluated_run_dir: Path, tmp_path: Path):
    """Archives exceeding max_archive_size_bytes must be rejected before reading contents."""
    bundle_zip = tmp_path / "oversized.zip"
    build_audit_bundle(sample_evaluated_run_dir, output_bundle_path=bundle_zip)

    with pytest.raises(AuditBundleSecurityError, match="exceeds maximum permitted limit"):
        verify_audit_bundle(bundle_zip, max_archive_size_bytes=100)

    res = verify_audit_bundle(bundle_zip, raise_on_error=False, max_archive_size_bytes=100)
    assert not res.is_valid
    assert any("exceeds maximum permitted limit" in err for err in res.errors)

    with pytest.raises(AuditBundleSecurityError, match="exceeds limit"):
        extract_audit_bundle(bundle_zip, destination_dir=tmp_path / "ext", verify_first=False, max_archive_size_bytes=100)


def test_audit_bundle_resource_limits_member_size(sample_evaluated_run_dir: Path, tmp_path: Path):
    """Members exceeding max_member_size_bytes must be rejected."""
    bundle_zip = tmp_path / "oversized_member.zip"
    build_audit_bundle(sample_evaluated_run_dir, output_bundle_path=bundle_zip)

    with pytest.raises(AuditBundleSecurityError, match="uncompressed size .* exceeds maximum permitted limit"):
        verify_audit_bundle(bundle_zip, max_member_size_bytes=50)

    res = verify_audit_bundle(bundle_zip, raise_on_error=False, max_member_size_bytes=50)
    assert not res.is_valid
    assert any("uncompressed size" in err for err in res.errors)

    with pytest.raises(AuditBundleSecurityError, match="uncompressed size .* exceeds limit"):
        extract_audit_bundle(bundle_zip, destination_dir=tmp_path / "ext", verify_first=False, max_member_size_bytes=50)


def test_audit_bundle_resource_limits_total_uncompressed(sample_evaluated_run_dir: Path, tmp_path: Path):
    """Archives with cumulative uncompressed size exceeding limit must be rejected."""
    bundle_zip = tmp_path / "oversized_total.zip"
    build_audit_bundle(sample_evaluated_run_dir, output_bundle_path=bundle_zip)

    with pytest.raises(AuditBundleSecurityError, match="Total uncompressed archive size .* exceeds maximum permitted limit"):
        verify_audit_bundle(bundle_zip, max_total_uncompressed_bytes=100)

    res = verify_audit_bundle(bundle_zip, raise_on_error=False, max_total_uncompressed_bytes=100)
    assert not res.is_valid
    assert any("Total uncompressed archive size" in err for err in res.errors)


def test_audit_bundle_detects_suspicious_compression_ratio(tmp_path: Path):
    """Members with extreme compression ratios (>100:1 for >1MB) must be flagged as suspicious."""
    zip_path = tmp_path / "bomb.zip"
    large_payload = b"0" * (2 * 1024 * 1024)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("evidence/payload.bin", large_payload)

    with pytest.raises(AuditBundleSecurityError, match="suspicious compression ratio"):
        verify_audit_bundle(zip_path, max_compression_ratio=50.0)

    res = verify_audit_bundle(zip_path, raise_on_error=False, max_compression_ratio=50.0)
    assert not res.is_valid
    assert any("suspicious compression ratio" in err for err in res.errors)


def test_audit_bundle_rejects_encrypted_members(sample_evaluated_run_dir: Path, tmp_path: Path):
    """Encrypted ZIP members must be rejected with AuditBundleSecurityError."""
    bundle_zip = tmp_path / "encrypted.zip"
    build_audit_bundle(sample_evaluated_run_dir, output_bundle_path=bundle_zip)

    data = bytearray(bundle_zip.read_bytes())
    cd_idx = data.find(b"PK\x01\x02")
    assert cd_idx != -1
    data[cd_idx + 8] |= 0x1

    tampered_zip = tmp_path / "encrypted_flag.zip"
    tampered_zip.write_bytes(data)

    with pytest.raises(AuditBundleSecurityError, match="is encrypted"):
        verify_audit_bundle(tampered_zip)

    with pytest.raises(AuditBundleSecurityError, match="is encrypted"):
        extract_audit_bundle(tampered_zip, destination_dir=tmp_path / "ext", verify_first=False)


def test_audit_bundle_rejects_symbolic_link_members(sample_evaluated_run_dir: Path, tmp_path: Path):
    """Symbolic link members in ZIP archives must be rejected with AuditBundleSecurityError."""
    bundle_zip = tmp_path / "clean.zip"
    build_audit_bundle(sample_evaluated_run_dir, output_bundle_path=bundle_zip)

    symlink_zip = tmp_path / "symlink.zip"
    with zipfile.ZipFile(bundle_zip, "r") as src_zf, zipfile.ZipFile(symlink_zip, "w") as dst_zf:
        for item in src_zf.infolist():
            content = src_zf.read(item.filename)
            if item.filename == "manifest.json":
                item.external_attr = 0o120777 << 16
            dst_zf.writestr(item, content)

    with pytest.raises(AuditBundleSecurityError, match="is a symbolic link"):
        verify_audit_bundle(symlink_zip)

    with pytest.raises(AuditBundleSecurityError, match="is a symbolic link"):
        extract_audit_bundle(symlink_zip, destination_dir=tmp_path / "ext", verify_first=False)


def test_audit_bundle_preserves_provenance_and_backward_compatibility(sample_evaluated_run_dir: Path, tmp_path: Path):
    """Audit bundle manifests must preserve model provenance fields and remain backward compatible with legacy bundles."""
    manifest_file = sample_evaluated_run_dir / "manifest.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        mdata = json.load(f)
    mdata["model_provenance"] = "final_cv_fold"
    mdata["candidate_model_fold_index"] = 4
    mdata["baseline_model_fold_index"] = 4
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(mdata, f, indent=2)

    bundle_zip = tmp_path / "provenance_bundle.zip"
    build_audit_bundle(sample_evaluated_run_dir, output_bundle_path=bundle_zip)

    res = verify_audit_bundle(bundle_zip)
    assert res.is_valid
    assert res.manifest is not None
    assert res.manifest.model_provenance == "final_cv_fold"
    assert res.manifest.candidate_model_fold_index == 4
    assert res.manifest.baseline_model_fold_index == 4

    legacy_zip = tmp_path / "legacy_bundle.zip"
    with zipfile.ZipFile(bundle_zip, "r") as src_zf, zipfile.ZipFile(legacy_zip, "w") as dst_zf:
        for item in src_zf.infolist():
            content = src_zf.read(item.filename)
            if item.filename == "bundle_manifest.json":
                bm = json.loads(content.decode("utf-8"))
                bm.pop("model_provenance", None)
                bm.pop("candidate_model_fold_index", None)
                bm.pop("baseline_model_fold_index", None)
                content = json.dumps(bm, indent=2).encode("utf-8")
            dst_zf.writestr(item, content)

    legacy_res = verify_audit_bundle(legacy_zip)
    assert legacy_res.is_valid
    assert legacy_res.manifest is not None
    assert legacy_res.manifest.model_provenance is None
    assert legacy_res.manifest.candidate_model_fold_index is None


# ============================================================================
# Milestone 4.5 Packaging Stress, Boundary & Corruption Hardening Tests
# ============================================================================

def test_audit_bundle_deeply_nested_members(sample_evaluated_run_dir: Path, tmp_path: Path):
    """Deeply nested directory structures must be safely indexed, packaged, verified, and extracted."""
    nested_file = sample_evaluated_run_dir / "evidence" / "diagnostics" / "subgroups" / "emea" / "slice.json"
    nested_file.parent.mkdir(parents=True, exist_ok=True)
    nested_file.write_text('{"slice": "emea", "metric": 0.82}', encoding="utf-8")

    # Update manifest to declare the nested file
    manifest_file = sample_evaluated_run_dir / "manifest.json"
    mdata = json.loads(manifest_file.read_text(encoding="utf-8"))
    nested_rel = "evidence/diagnostics/subgroups/emea/slice.json"
    mdata["file_checksums"][nested_rel] = hashlib.sha256(nested_file.read_bytes()).hexdigest()
    manifest_file.write_text(json.dumps(mdata, indent=2), encoding="utf-8")

    bundle_zip = tmp_path / "nested_bundle.zip"
    build_audit_bundle(sample_evaluated_run_dir, output_bundle_path=bundle_zip)

    # Verification passes
    v_res = verify_audit_bundle(bundle_zip, raise_on_error=True)
    assert v_res.is_valid
    assert nested_rel in v_res.verified_files

    # Extraction creates the deep directory hierarchy and verifies byte identity
    extract_dir = tmp_path / "extracted_deep"
    extract_audit_bundle(bundle_zip, destination_dir=extract_dir, verify_first=True)
    extracted_nested = extract_dir / "evidence" / "diagnostics" / "subgroups" / "emea" / "slice.json"
    assert extracted_nested.exists()
    assert extracted_nested.read_text(encoding="utf-8") == nested_file.read_text(encoding="utf-8")


def test_audit_bundle_member_size_exact_boundary(sample_evaluated_run_dir: Path, tmp_path: Path):
    """Member size limit enforcement at exact boundaries: <= limit passes, > limit fails."""
    bundle_zip = tmp_path / "boundary_bundle.zip"
    build_audit_bundle(sample_evaluated_run_dir, output_bundle_path=bundle_zip)

    # Find the largest member size in the bundle
    with zipfile.ZipFile(bundle_zip, "r") as zf:
        max_member_size = max(z.file_size for z in zf.infolist())

    # 1. At boundary: limit == max_member_size -> PASSES
    v_at = verify_audit_bundle(bundle_zip, max_member_size_bytes=max_member_size)
    assert v_at.is_valid

    e_at_dir = tmp_path / "ext_at_bound"
    extract_audit_bundle(bundle_zip, destination_dir=e_at_dir, max_member_size_bytes=max_member_size)
    assert (e_at_dir / "manifest.json").exists()

    # 2. Below boundary: limit == max_member_size + 10 -> PASSES
    v_below = verify_audit_bundle(bundle_zip, max_member_size_bytes=max_member_size + 10)
    assert v_below.is_valid

    # 3. Above boundary: limit == max_member_size - 1 -> FAILS with AuditBundleSecurityError
    with pytest.raises(AuditBundleSecurityError, match="exceeds maximum permitted limit"):
        verify_audit_bundle(bundle_zip, max_member_size_bytes=max_member_size - 1)

    with pytest.raises(AuditBundleSecurityError, match="exceeds limit"):
        extract_audit_bundle(
            bundle_zip,
            destination_dir=tmp_path / "ext_above_bound",
            verify_first=False,
            max_member_size_bytes=max_member_size - 1,
        )


def test_audit_bundle_total_uncompressed_exact_boundary(sample_evaluated_run_dir: Path, tmp_path: Path):
    """Total uncompressed archive size limit at exact boundaries: <= limit passes, > limit fails."""
    bundle_zip = tmp_path / "total_boundary.zip"
    build_audit_bundle(sample_evaluated_run_dir, output_bundle_path=bundle_zip)

    with zipfile.ZipFile(bundle_zip, "r") as zf:
        total_size = sum(z.file_size for z in zf.infolist())

    # 1. At boundary: total_size -> PASSES
    v_at = verify_audit_bundle(bundle_zip, max_total_uncompressed_bytes=total_size)
    assert v_at.is_valid

    # 2. Strictly above boundary: total_size - 1 -> FAILS
    with pytest.raises(AuditBundleSecurityError, match="Total uncompressed archive size .* exceeds maximum permitted limit"):
        verify_audit_bundle(bundle_zip, max_total_uncompressed_bytes=total_size - 1)

    with pytest.raises(AuditBundleSecurityError, match="Total extracted size exceeded limit"):
        extract_audit_bundle(
            bundle_zip,
            destination_dir=tmp_path / "ext_total_fail",
            verify_first=False,
            max_total_uncompressed_bytes=total_size - 1,
        )


def test_audit_bundle_compression_ratio_exact_boundary(tmp_path: Path):
    """Compression ratio limit at exact boundaries for files > 1 MB."""
    zip_path = tmp_path / "ratio_test.zip"
    payload = b"A" * (2 * 1024 * 1024)  # 2 MB compressible payload

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.bin", payload)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zinfo = zf.getinfo("data.bin")
        exact_ratio = zinfo.file_size / zinfo.compress_size

    # 1. At boundary (ratio ceiling is >= exact_ratio) -> passes ratio check
    # Note: it will fail on bundle_manifest missing, but NOT on compression ratio
    res_at = verify_audit_bundle(zip_path, raise_on_error=False, max_compression_ratio=exact_ratio + 0.1)
    assert not any("suspicious compression ratio" in err for err in res_at.errors)

    # 2. Stricter than boundary -> raises AuditBundleSecurityError
    with pytest.raises(AuditBundleSecurityError, match="suspicious compression ratio"):
        verify_audit_bundle(zip_path, max_compression_ratio=exact_ratio - 1.0)


def test_audit_bundle_truncated_archives_fail_closed(sample_evaluated_run_dir: Path, tmp_path: Path):
    """Truncated archives must fail closed with AuditBundleCorruptError in both verify and extract."""
    bundle_zip = tmp_path / "clean_base.zip"
    build_audit_bundle(sample_evaluated_run_dir, output_bundle_path=bundle_zip)
    clean_bytes = bundle_zip.read_bytes()

    # 1. Severely truncated (first 50 bytes)
    trunc_50 = tmp_path / "trunc_50.zip"
    trunc_50.write_bytes(clean_bytes[:50])

    with pytest.raises(AuditBundleCorruptError, match="not a valid or readable ZIP"):
        verify_audit_bundle(trunc_50, raise_on_error=True)

    with pytest.raises(AuditBundleCorruptError):
        extract_audit_bundle(trunc_50, destination_dir=tmp_path / "ext_trunc", verify_first=False)

    # 2. Truncated mid-archive (half bytes)
    trunc_half = tmp_path / "trunc_half.zip"
    trunc_half.write_bytes(clean_bytes[: len(clean_bytes) // 2])

    with pytest.raises(AuditBundleCorruptError):
        verify_audit_bundle(trunc_half, raise_on_error=True)

    with pytest.raises(AuditBundleCorruptError):
        extract_audit_bundle(trunc_half, destination_dir=tmp_path / "ext_trunc_half", verify_first=False)


def test_audit_bundle_corrupt_crc_detected(sample_evaluated_run_dir: Path, tmp_path: Path):
    """ZIP member CRC corruption must be caught by testzip() and raise AuditBundleCorruptError."""
    bundle_zip = tmp_path / "crc_base.zip"
    build_audit_bundle(sample_evaluated_run_dir, output_bundle_path=bundle_zip)

    # The first member starts at offset 0 with local header PK\x03\x04
    data = bytearray(bundle_zip.read_bytes())
    assert data[:4] == b"PK\x03\x04"
    fn_len = int.from_bytes(data[26:28], "little")
    extra_len = int.from_bytes(data[28:30], "little")
    payload_offset = 30 + fn_len + extra_len
    # Flip a byte in the compressed payload while preserving the central directory CRC
    data[payload_offset + 5] ^= 0xFF

    corrupt_zip = tmp_path / "corrupt_crc.zip"
    corrupt_zip.write_bytes(data)

    with pytest.raises(AuditBundleCorruptError, match="ZIP CRC checksum failure|Failed ZIP integrity test"):
        verify_audit_bundle(corrupt_zip, raise_on_error=True)


def test_audit_bundle_streaming_detects_decompression_size_forgery(sample_evaluated_run_dir: Path, tmp_path: Path):
    """Streaming verification must catch members whose actual stream size exceeds declared manifest size."""
    bundle_zip = tmp_path / "spoof_base.zip"
    build_audit_bundle(sample_evaluated_run_dir, output_bundle_path=bundle_zip)

    # Tamper with manifest in bundle to forge size to smaller than actual file
    spoofed_zip = tmp_path / "spoofed.zip"
    with zipfile.ZipFile(bundle_zip, "r") as zf_in, zipfile.ZipFile(spoofed_zip, "w") as zf_out:
        for item in zf_in.infolist():
            content = zf_in.read(item.filename)
            if item.filename == "bundle_manifest.json":
                mdict = json.loads(content.decode("utf-8"))
                for f in mdict["files"]:
                    if f["path"] == "evaluation.json":
                        f["size_bytes"] = 10  # Forged small size
                content = json.dumps(mdict, indent=2).encode("utf-8")
            zf_out.writestr(item, content)

    with pytest.raises(AuditBundleTamperError, match="[Ss]ize mismatch"):
        verify_audit_bundle(spoofed_zip, raise_on_error=True)
