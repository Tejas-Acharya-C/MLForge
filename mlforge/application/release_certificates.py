"""MLForge Cryptographically Signed Release Certificates Module (Milestone 4.3).

Provides deterministic payload canonicalization, Ed25519 digital signature generation,
atomic certificate storage, and multi-tiered cryptographic verification with explicit
trust boundaries.
"""

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import zipfile

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519

from mlforge.application.artifacts import load_run_artifacts
from mlforge.core.contracts import (
    AcceptanceDecision,
    ArtifactManifest,
    CertificateVerificationResult,
    EvaluationResult,
    ReleaseCertificate,
    ReleaseGateResult,
    ReleaseGateStatus,
    ReleasePolicy,
)
from mlforge.core.exceptions import (
    ArtifactLoadError,
    ArtifactSecurityError,
    ArtifactSerializationError,
    CertificateError,
    CertificateSecurityError,
    CertificateVerificationError,
)
from mlforge.engine.release_gate import evaluate_release_gate

CERTIFICATE_SCHEMA_VERSION = "1.0.0"
SIGNATURE_ALGORITHM = "Ed25519"


def generate_release_keypair() -> Tuple[ed25519.Ed25519PrivateKey, ed25519.Ed25519PublicKey, str]:
    """Generate a fresh Ed25519 signing keypair for release certificates.

    SECURITY NOTICE
    ---------------
    Private keys must be stored in secure environment key vaults and NEVER committed
    to version control or persisted in evaluation runs or audit bundles.

    Returns
    -------
    Tuple[Ed25519PrivateKey, Ed25519PublicKey, str]
        (private_key, public_key, key_id_fingerprint)
    """
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    raw_pub = public_key.public_bytes_raw()
    key_id = hashlib.sha256(raw_pub).hexdigest()[:16]
    return private_key, public_key, key_id


def serialize_public_key(public_key: ed25519.Ed25519PublicKey) -> str:
    """Export public key as a hexadecimal string."""
    return public_key.public_bytes_raw().hex()


def deserialize_public_key(key_hex: str) -> ed25519.Ed25519PublicKey:
    """Construct an Ed25519PublicKey from a 64-character hexadecimal string."""
    if not isinstance(key_hex, str) or len(key_hex) != 64:
        raise CertificateSecurityError(
            f"Invalid Ed25519 public key format: expected 64 hex characters (32 bytes), got {len(key_hex) if isinstance(key_hex, str) else type(key_hex)}"
        )
    try:
        raw_bytes = bytes.fromhex(key_hex)
        return ed25519.Ed25519PublicKey.from_public_bytes(raw_bytes)
    except Exception as exc:
        raise CertificateSecurityError(f"Failed to parse Ed25519 public key: {exc}") from exc


def _check_finite_numbers(data: Any, path: str = "root") -> None:
    """Recursively reject non-finite floating point numbers (NaN, Inf, -Inf)."""
    if isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            raise CertificateSecurityError(
                f"Non-finite float value '{data}' detected at path '{path}'. Release certificates strictly reject NaN and Infinity."
            )
    elif isinstance(data, dict):
        for k, v in data.items():
            _check_finite_numbers(v, f"{path}.{k}")
    elif isinstance(data, (list, tuple)):
        for idx, item in enumerate(data):
            _check_finite_numbers(item, f"{path}[{idx}]")


def canonicalize_certificate_payload(payload_dict: Dict[str, Any]) -> bytes:
    """Serialize certificate payload into deterministic canonical UTF-8 bytes.

    Canonical format specifies:
    1. Rejection of NaN, Infinity, -Infinity.
    2. Alphabetical dictionary key sorting (`sort_keys=True`).
    3. Compact JSON formatting with no whitespace (`separators=(',', ':')`).
    4. Exact UTF-8 byte encoding.
    """
    _check_finite_numbers(payload_dict)
    try:
        json_str = json.dumps(payload_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return json_str.encode("utf-8")
    except Exception as exc:
        raise CertificateSecurityError(f"Failed to canonicalize certificate payload: {exc}") from exc


def _extract_signed_payload_dict(cert: ReleaseCertificate) -> Dict[str, Any]:
    """Reconstruct exact dictionary of signed payload fields from certificate object."""
    return {
        "acceptance_decision_outcome": cert.acceptance_decision_outcome,
        "acceptance_policy_id": cert.acceptance_policy_id,
        "acceptance_policy_version": cert.acceptance_policy_version,
        "candidate_model_checksum": cert.candidate_model_checksum,
        "candidate_model_name": cert.candidate_model_name,
        "candidate_model_type": cert.candidate_model_type,
        "certificate_id": cert.certificate_id,
        "certificate_schema_version": cert.certificate_schema_version,
        "dataset_checksum": cert.dataset_checksum,
        "evaluation_schema_version": cert.evaluation_schema_version,
        "fitted_fold_index": cert.fitted_fold_index,
        "gate_summary": cert.gate_summary,
        "issued_at": cert.issued_at,
        "key_id": cert.key_id,
        "model_provenance": cert.model_provenance,
        "release_gate_outcome": cert.release_gate_outcome.value if hasattr(cert.release_gate_outcome, "value") else str(cert.release_gate_outcome),
        "release_policy_id": cert.release_policy_id,
        "release_policy_version": cert.release_policy_version,
        "run_id": cert.run_id,
        "signature_algorithm": cert.signature_algorithm,
        "target_environment": cert.target_environment,
    }


def issue_release_certificate(
    source: Union[str, Path],
    private_key: ed25519.Ed25519PrivateKey,
    key_id: str,
    release_policy: Optional[ReleasePolicy] = None,
    output_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> ReleaseCertificate:
    """Evaluate release gates and, upon conclusive PASS, issue a cryptographically signed release certificate.

    PARAMETERS
    ----------
    source : str or Path
        Directory containing serialized run artifacts.
    private_key : Ed25519PrivateKey
        Private key for signing the canonical release certificate payload.
    key_id : str
        Identifier or fingerprint of the public signing key.
    release_policy : ReleasePolicy, optional
        Release requirements to evaluate against.
    output_dir : str or Path, optional
        Target directory for certificate. Defaults to `<source>/release/`.
    overwrite : bool, default=False
        If False, raises ArtifactSerializationError when certificate file already exists.

    RETURNS
    -------
    ReleaseCertificate
        Cryptographically signed release certificate.

    RAISES
    ------
    CertificateSecurityError
        If the release gate did not pass, or key format is invalid.
    ArtifactSerializationError
        If writing certificate fails or overwrite=False conflicts.
    """
    if not isinstance(private_key, ed25519.Ed25519PrivateKey):
        raise CertificateSecurityError("Signing key must be an instance of cryptography.hazmat.primitives.asymmetric.ed25519.Ed25519PrivateKey")

    src_path = Path(source).resolve()
    if not src_path.exists():
        raise ArtifactLoadError(f"Evidence source path '{src_path}' does not exist.")

    # 1. Run Release Gate Engine
    gate_result = evaluate_release_gate(src_path, release_policy)
    if not gate_result.is_releasable:
        raise CertificateSecurityError(
            f"Release certificate authorization denied: release gate outcome is '{gate_result.overall_status.value}'. Summary: {gate_result.summary}"
        )

    # 2. Extract Evidence Details for Certificate
    if src_path.is_dir():
        loaded = load_run_artifacts(src_path)
        manifest = loaded["manifest"]
        evaluation = loaded["evaluation"]
        decision = loaded.get("decision")
    else:
        raise ArtifactLoadError("Direct certificate issuance requires a local run artifact directory.")

    # Determine candidate model checksum if present
    cand_model_checksum: Optional[str] = None
    file_hashes = getattr(manifest, "file_checksums", getattr(manifest, "artifact_checksums", {}))
    if "models/candidate_pipeline.joblib" in file_hashes:
        cand_model_checksum = file_hashes["models/candidate_pipeline.joblib"]

    cert_id = str(uuid.uuid4())
    issued_at = datetime.now(timezone.utc).isoformat()

    gate_summary = {g.gate_id: g.status.value for g in gate_result.gate_results}

    payload_dict: Dict[str, Any] = {
        "acceptance_decision_outcome": decision.outcome.value if decision else None,
        "acceptance_policy_id": decision.policy_id if decision else None,
        "acceptance_policy_version": decision.policy_version if decision else None,
        "candidate_model_checksum": cand_model_checksum,
        "candidate_model_name": evaluation.candidate_result.model_name,
        "candidate_model_type": evaluation.candidate_result.model_type,
        "certificate_id": cert_id,
        "certificate_schema_version": CERTIFICATE_SCHEMA_VERSION,
        "dataset_checksum": evaluation.dataset_metadata.checksum_sha256,
        "evaluation_schema_version": manifest.schema_version,
        "fitted_fold_index": getattr(manifest, "candidate_model_fold_index", evaluation.candidate_result.fitted_fold_index),
        "gate_summary": gate_summary,
        "issued_at": issued_at,
        "key_id": key_id,
        "model_provenance": getattr(manifest, "model_provenance", evaluation.candidate_result.model_provenance),
        "release_gate_outcome": gate_result.overall_status.value,
        "release_policy_id": gate_result.policy_id,
        "release_policy_version": gate_result.policy_version,
        "run_id": manifest.run_id,
        "signature_algorithm": SIGNATURE_ALGORITHM,
        "target_environment": gate_result.target_environment,
    }

    # 3. Canonicalize & Sign
    canonical_bytes = canonicalize_certificate_payload(payload_dict)
    payload_hash = hashlib.sha256(canonical_bytes).hexdigest()

    signature_bytes = private_key.sign(canonical_bytes)
    signature_hex = signature_bytes.hex()

    certificate = ReleaseCertificate(
        **payload_dict,
        payload_digest_sha256=payload_hash,
        signature=signature_hex,
    )

    # 4. Atomic Filesystem Storage
    if output_dir is not None:
        target_dir = Path(output_dir).resolve()
    else:
        target_dir = src_path / "release"

    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "release_certificate.json"

    if target_file.exists() and not overwrite:
        raise ArtifactSerializationError(
            f"Release certificate already exists at '{target_file}' and overwrite=False."
        )

    tmp_file = target_dir / f"release_certificate.json.tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    try:
        cert_json = certificate.model_dump_json(indent=2)
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(cert_json)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_file, target_file)
    except Exception as exc:
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass
        raise ArtifactSerializationError(f"Failed to write release certificate: {exc}") from exc

    return certificate


def _verify_certificate_claims(
    cert: ReleaseCertificate,
    ev: Optional[EvaluationResult],
    manifest: Optional[ArtifactManifest],
    decision: Optional[AcceptanceDecision],
    cand_model_bytes: Optional[bytes],
    errors: List[str],
    details: List[str],
) -> bool:
    """Compare all claims recorded in a release certificate against authoritative persisted evidence."""
    if ev is None or manifest is None:
        return False

    is_consistent = True

    # 1. Run ID
    if manifest.run_id != cert.run_id:
        is_consistent = False
        errors.append(f"Run ID mismatch: certificate has '{cert.run_id}', evidence has '{manifest.run_id}'.")

    # 2. Dataset Checksum
    ev_dataset_hash = ev.dataset_metadata.checksum_sha256
    if ev_dataset_hash != cert.dataset_checksum:
        is_consistent = False
        errors.append(
            f"Dataset checksum mismatch: certificate has '{cert.dataset_checksum[:12]}', "
            f"evidence has '{ev_dataset_hash[:12]}'."
        )

    # 3. Candidate Model Name & Type
    if ev.candidate_result.model_name != cert.candidate_model_name:
        is_consistent = False
        errors.append(
            f"Candidate model name mismatch: certificate has '{cert.candidate_model_name}', "
            f"evidence has '{ev.candidate_result.model_name}'."
        )

    if ev.candidate_result.model_type != cert.candidate_model_type:
        is_consistent = False
        errors.append(
            f"Candidate model type mismatch: certificate has '{cert.candidate_model_type}', "
            f"evidence has '{ev.candidate_result.model_type}'."
        )

    # 4. Candidate Model Checksum
    if cert.candidate_model_checksum:
        if cand_model_bytes is None:
            is_consistent = False
            errors.append("Candidate model artifact recorded in certificate is missing from evidence.")
        else:
            actual_model_hash = hashlib.sha256(cand_model_bytes).hexdigest()
            if actual_model_hash != cert.candidate_model_checksum:
                is_consistent = False
                errors.append(
                    f"Candidate model checksum mismatch: certificate has '{cert.candidate_model_checksum[:12]}', "
                    f"evidence has '{actual_model_hash[:12]}'."
                )

    # 5. Model Provenance
    ev_prov = getattr(manifest, "model_provenance", None) or ev.candidate_result.model_provenance
    if cert.model_provenance is not None and ev_prov != cert.model_provenance:
        is_consistent = False
        errors.append(
            f"Model provenance mismatch: certificate has '{cert.model_provenance}', "
            f"evidence has '{ev_prov}'."
        )

    # 6. Fitted Fold Index
    ev_fold = getattr(manifest, "candidate_model_fold_index", None)
    if ev_fold is None:
        ev_fold = ev.candidate_result.fitted_fold_index
    if cert.fitted_fold_index is not None and ev_fold != cert.fitted_fold_index:
        is_consistent = False
        errors.append(
            f"Fitted fold index mismatch: certificate has {cert.fitted_fold_index}, "
            f"evidence has {ev_fold}."
        )

    # 7. Evaluation Schema Version
    ev_schema = getattr(manifest, "schema_version", None) or getattr(ev, "schema_version", None)
    if ev_schema != cert.evaluation_schema_version:
        is_consistent = False
        errors.append(
            f"Evaluation schema version mismatch: certificate has '{cert.evaluation_schema_version}', "
            f"evidence has '{ev_schema}'."
        )

    # 8. Acceptance Decision Outcome & Policy
    if cert.acceptance_decision_outcome is not None:
        if decision is None:
            is_consistent = False
            errors.append(
                f"Certificate claims acceptance outcome '{cert.acceptance_decision_outcome}', "
                f"but decision.json is missing in evidence."
            )
        else:
            dec_outcome_str = decision.outcome.value if hasattr(decision.outcome, "value") else str(decision.outcome)
            if dec_outcome_str.lower() != cert.acceptance_decision_outcome.lower():
                is_consistent = False
                errors.append(
                    f"Acceptance outcome mismatch: certificate has '{cert.acceptance_decision_outcome}', "
                    f"evidence has '{dec_outcome_str}'."
                )

            if cert.acceptance_policy_id is not None and decision.policy_id != cert.acceptance_policy_id:
                is_consistent = False
                errors.append(
                    f"Acceptance policy ID mismatch: certificate has '{cert.acceptance_policy_id}', "
                    f"evidence has '{decision.policy_id}'."
                )

            if cert.acceptance_policy_version is not None and decision.policy_version != cert.acceptance_policy_version:
                is_consistent = False
                errors.append(
                    f"Acceptance policy version mismatch: certificate has '{cert.acceptance_policy_version}', "
                    f"evidence has '{decision.policy_version}'."
                )

    if is_consistent:
        details.append("Evidence consistency verified: all certificate claims match authoritative persisted evidence.")

    return is_consistent


def verify_release_certificate(
    certificate: Union[ReleaseCertificate, str, Path],
    trusted_keys: Optional[Dict[str, ed25519.Ed25519PublicKey]] = None,
    expected_source: Optional[Union[str, Path]] = None,
    candidate_keys: Optional[Dict[str, ed25519.Ed25519PublicKey]] = None,
) -> CertificateVerificationResult:
    """Verify a release certificate across payload integrity, signature validity, signer trust, and evidence consistency.

    PARAMETERS
    ----------
    certificate : ReleaseCertificate, str, or Path
        Certificate object, JSON string, or path to release_certificate.json file.
    trusted_keys : dict, optional
        Mapping of authorized key_id string to Ed25519PublicKey instances.
    expected_source : str or Path, optional
        Path to evaluation run directory or .zip audit bundle to verify certificate claims against physical artifacts.
        SECURITY REQUIREMENT: Must be provided to establish evidence consistency and release authorization.
        If omitted, evidence_consistency is set to False (fail-closed).
    candidate_keys : dict, optional
        Mapping of untrusted candidate key_id to Ed25519PublicKey instances. Allows mathematical
        verification of signature validity without conferring signer trust.

    RETURNS
    -------
    CertificateVerificationResult
        Multi-tiered verification verdict.
    """
    # Load certificate if path or string
    cert_obj: ReleaseCertificate
    if isinstance(certificate, (str, Path)):
        p = Path(certificate)
        if p.exists() and p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cert_obj = ReleaseCertificate.model_validate(data)
            except Exception as exc:
                return CertificateVerificationResult(
                    certificate_id="unknown",
                    run_id="unknown",
                    payload_integrity=False,
                    signature_validity=False,
                    signer_trust=False,
                    evidence_consistency=False,
                    release_authorized=False,
                    errors=[f"Failed to parse certificate from file '{p}': {exc}"],
                )
        else:
            try:
                data = json.loads(str(certificate))
                cert_obj = ReleaseCertificate.model_validate(data)
            except Exception as exc:
                return CertificateVerificationResult(
                    certificate_id="unknown",
                    run_id="unknown",
                    payload_integrity=False,
                    signature_validity=False,
                    signer_trust=False,
                    evidence_consistency=False,
                    release_authorized=False,
                    errors=[f"Failed to parse certificate JSON string: {exc}"],
                )
    elif isinstance(certificate, ReleaseCertificate):
        cert_obj = certificate
    else:
        raise CertificateSecurityError(f"Unsupported certificate input type: {type(certificate)}")

    cert_id = cert_obj.certificate_id
    run_id = cert_obj.run_id
    key_id = cert_obj.key_id

    details: List[str] = []
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Payload Integrity
    payload_dict = _extract_signed_payload_dict(cert_obj)
    try:
        canonical_bytes = canonicalize_certificate_payload(payload_dict)
        computed_hash = hashlib.sha256(canonical_bytes).hexdigest()
        if computed_hash == cert_obj.payload_digest_sha256:
            payload_integrity = True
            details.append(f"Payload integrity verified (SHA-256: {computed_hash[:16]}...).")
        else:
            payload_integrity = False
            errors.append(
                f"Payload integrity violation: recorded digest '{cert_obj.payload_digest_sha256[:16]}...' "
                f"does not match recomputed payload hash '{computed_hash[:16]}...'."
            )
    except Exception as exc:
        payload_integrity = False
        errors.append(f"Payload canonicalization error: {exc}")

    # 2. Signature Validity & Signer Trust
    signature_validity = False
    signer_trust = False

    if cert_obj.signature_algorithm != SIGNATURE_ALGORITHM:
        errors.append(f"Unsupported signature algorithm '{cert_obj.signature_algorithm}'; expected '{SIGNATURE_ALGORITHM}'.")
    else:
        try:
            sig_bytes = bytes.fromhex(cert_obj.signature)
        except Exception:
            sig_bytes = b""
            errors.append("Malformed cryptographic signature: not valid hex.")

        if sig_bytes and len(sig_bytes) == 64:
            # Check public key availability across trusted and candidate keys
            pub_key: Optional[ed25519.Ed25519PublicKey] = None
            if trusted_keys and key_id in trusted_keys:
                pub_key = trusted_keys[key_id]
                signer_trust = True
                details.append(f"Signer key_id '{key_id}' is recognized in trusted keys registry.")
            elif candidate_keys and key_id in candidate_keys:
                pub_key = candidate_keys[key_id]
                signer_trust = False
                warnings.append(
                    f"Signer key_id '{key_id}' is NOT present in trusted keys registry; "
                    "verifying mathematical signature validity with untrusted candidate key."
                )
            else:
                signer_trust = False
                if trusted_keys is not None:
                    warnings.append(f"Signer key_id '{key_id}' is NOT present in supplied trusted keys registry.")
                else:
                    warnings.append("No trusted keys registry supplied: signer trust cannot be established.")
                errors.append(f"Public key for key_id '{key_id}' is unavailable for signature verification.")

            if pub_key is not None:
                try:
                    pub_key.verify(sig_bytes, canonical_bytes)
                    signature_validity = True
                    if signer_trust:
                        details.append(f"Ed25519 signature cryptographically verified with trusted public key '{key_id}'.")
                    else:
                        details.append(f"Ed25519 signature mathematically verified with UNTRUSTED candidate public key '{key_id}'.")
                except InvalidSignature:
                    signature_validity = False
                    errors.append("Cryptographic signature verification failed: signature does not match public key.")
                except Exception as exc:
                    signature_validity = False
                    errors.append(f"Signature verification error: {exc}")
        else:
            errors.append(f"Invalid Ed25519 signature byte length: expected 64 bytes, got {len(sig_bytes)}.")

    # 3. Evidence Consistency (Fail-Closed)
    evidence_consistency = False
    if expected_source is None:
        errors.append(
            "Evidence consistency check failed: expected_source was not provided to verify physical evidence against certificate claims."
        )
        warnings.append(
            "Release authorization cannot be established without physical evidence verification (expected_source is None)."
        )
    else:
        try:
            exp_path = Path(expected_source).resolve()
            if not exp_path.exists():
                errors.append(f"Expected source '{exp_path}' does not exist on disk.")
            elif exp_path.is_file() and exp_path.suffix.lower() == ".zip":
                # Verify audit bundle
                from mlforge.application.packaging import verify_audit_bundle

                bundle_res = verify_audit_bundle(exp_path, raise_on_error=False)
                if not bundle_res.is_valid:
                    errors.append(f"Audit bundle archive verification failed: {'; '.join(bundle_res.errors)}")
                else:
                    with zipfile.ZipFile(exp_path, mode="r") as zf:
                        namelist = zf.namelist()
                        ev: Optional[EvaluationResult] = None
                        if "evaluation.json" not in namelist:
                            errors.append("evaluation.json missing in audit bundle.")
                        else:
                            try:
                                ev_data = json.loads(zf.read("evaluation.json").decode("utf-8"))
                                ev = EvaluationResult.model_validate(ev_data)
                            except Exception as exc:
                                errors.append(f"Failed to read evaluation.json from bundle: {exc}")

                        m: Optional[ArtifactManifest] = None
                        if "manifest.json" in namelist:
                            try:
                                m_data = json.loads(zf.read("manifest.json").decode("utf-8"))
                                m = ArtifactManifest.model_validate(m_data)
                            except Exception as exc:
                                errors.append(f"Failed to read manifest.json from bundle: {exc}")

                        dec: Optional[AcceptanceDecision] = None
                        if "decision.json" in namelist:
                            try:
                                dec_data = json.loads(zf.read("decision.json").decode("utf-8"))
                                if isinstance(dec_data.get("outcome"), str):
                                    dec_data["outcome"] = dec_data["outcome"].upper()
                                dec = AcceptanceDecision.model_validate(dec_data)
                            except Exception as exc:
                                errors.append(f"Failed to read decision.json from bundle: {exc}")

                        cand_model_bytes: Optional[bytes] = None
                        if cert_obj.candidate_model_checksum:
                            cand_rel = "models/candidate_pipeline.joblib"
                            if cand_rel not in namelist:
                                errors.append(f"Required model artifact '{cand_rel}' recorded in certificate is missing in bundle.")
                            else:
                                cand_model_bytes = zf.read(cand_rel)

                        evidence_consistency = _verify_certificate_claims(
                            cert=cert_obj,
                            ev=ev,
                            manifest=m,
                            decision=dec,
                            cand_model_bytes=cand_model_bytes,
                            errors=errors,
                            details=details,
                        )

            elif exp_path.is_dir():
                # Read manifest.json directly
                man_file = exp_path / "manifest.json"
                m: Optional[ArtifactManifest] = None
                if not man_file.exists():
                    errors.append("manifest.json missing on disk.")
                else:
                    try:
                        with open(man_file, "r", encoding="utf-8") as f:
                            m = ArtifactManifest.model_validate(json.load(f))
                    except Exception as exc:
                        errors.append(f"Failed to read manifest.json: {exc}")

                # Read evaluation.json directly
                eval_file = exp_path / "evaluation.json"
                ev: Optional[EvaluationResult] = None
                if not eval_file.exists():
                    errors.append("evaluation.json missing on disk.")
                else:
                    try:
                        with open(eval_file, "r", encoding="utf-8") as f:
                            ev = EvaluationResult.model_validate(json.load(f))
                    except Exception as exc:
                        errors.append(f"Failed to read evaluation.json: {exc}")

                # Read decision.json if present
                dec_file = exp_path / "decision.json"
                dec: Optional[AcceptanceDecision] = None
                if dec_file.exists():
                    try:
                        with open(dec_file, "r", encoding="utf-8") as f:
                            dec_dict = json.load(f)
                        if isinstance(dec_dict.get("outcome"), str):
                            dec_dict["outcome"] = dec_dict["outcome"].upper()
                        dec = AcceptanceDecision.model_validate(dec_dict)
                    except Exception as exc:
                        errors.append(f"Failed to read decision.json: {exc}")

                cand_model_bytes: Optional[bytes] = None
                if cert_obj.candidate_model_checksum:
                    cand_rel = "models/candidate_pipeline.joblib"
                    cand_disk = exp_path / cand_rel
                    if not cand_disk.exists():
                        errors.append(f"Model artifact '{cand_rel}' recorded in certificate is missing on disk.")
                    else:
                        cand_model_bytes = cand_disk.read_bytes()

                evidence_consistency = _verify_certificate_claims(
                    cert=cert_obj,
                    ev=ev,
                    manifest=m,
                    decision=dec,
                    cand_model_bytes=cand_model_bytes,
                    errors=errors,
                    details=details,
                )
            else:
                errors.append(f"Expected source '{exp_path}' must be a run directory or a .zip audit bundle.")
        except Exception as exc:
            evidence_consistency = False
            errors.append(f"Failed to verify evidence consistency: {exc}")

    # 4. Overall Authorization Verdict
    gate_passed = (cert_obj.release_gate_outcome == ReleaseGateStatus.PASSED)
    if not gate_passed:
        errors.append(f"Certificate records non-passed release gate outcome '{cert_obj.release_gate_outcome.value}'.")

    release_authorized = (
        payload_integrity
        and signature_validity
        and signer_trust
        and evidence_consistency
        and gate_passed
    )

    return CertificateVerificationResult(
        certificate_id=cert_id,
        run_id=run_id,
        payload_integrity=payload_integrity,
        signature_validity=signature_validity,
        signer_trust=signer_trust,
        evidence_consistency=evidence_consistency,
        release_authorized=release_authorized,
        key_id=key_id,
        details=details,
        errors=errors,
        warnings=warnings,
    )
