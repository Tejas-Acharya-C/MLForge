"""MLForge Automated Release-Gate Engine Module (Milestone 4.3).

Evaluates whether completed model evaluation runs satisfy governance, completeness,
statistical acceptance, artifact integrity, and provenance requirements to authorize release.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from mlforge.core.contracts import (
    AcceptanceDecision,
    AcceptanceOutcome,
    AcceptancePolicy,
    ArtifactManifest,
    AuditBundleManifest,
    EvaluationResult,
    GateCheckResult,
    GateCheckStatus,
    ReleaseGateResult,
    ReleaseGateStatus,
    ReleasePolicy,
)
from mlforge.core.exceptions import (
    ArtifactError,
    ArtifactLoadError,
    ArtifactSecurityError,
    ReleaseGateError,
)

SUPPORTED_SCHEMA_PREFIXES = ("1.",)


def _validate_safe_path(base_dir: Path, relative_str: str) -> Path:
    """Validate relative path within base directory against directory traversal."""
    if not relative_str or "\x00" in relative_str:
        raise ArtifactSecurityError(f"Invalid artifact relative path: {relative_str!r}")
    p = Path(relative_str)
    if p.is_absolute() or ".." in p.parts:
        raise ArtifactSecurityError(f"Path traversal detected in artifact entry: '{relative_str}'")
    resolved = (base_dir / p).resolve()
    base_resolved = base_dir.resolve()
    if not str(resolved).startswith(str(base_resolved)):
        raise ArtifactSecurityError(f"Path '{relative_str}' escapes base directory '{base_dir}'")
    return resolved


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 digest of a file in 64 KB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def evaluate_release_gate(
    source: Union[str, Path, Dict[str, Any]],
    policy: Optional[ReleasePolicy] = None,
) -> ReleaseGateResult:
    """Evaluate whether an evaluation run satisfies configured release authorization requirements.

    PARAMETERS
    ----------
    source : str, Path, or dict
        Path to serialized run directory, path to .zip audit bundle, or dictionary of pre-loaded artifacts.
    policy : ReleasePolicy, optional
        Release requirements configuration. If None, default strict ReleasePolicy is applied.

    RETURNS
    -------
    ReleaseGateResult
        Comprehensive structured authorization verdict with individual gate results and summary.

    RAISES
    ------
    ReleaseGateError
        If evidence loading fails or source path cannot be resolved.
    """
    if policy is None:
        policy = ReleasePolicy()

    now_utc = datetime.now(timezone.utc).isoformat()

    # 1. Load Evidence
    evidence: Dict[str, Any] = {}
    if isinstance(source, dict):
        evidence = source
        manifest = evidence.get("manifest")
        evaluation = evidence.get("evaluation")
        decision = evidence.get("decision")
        saved_policy = evidence.get("policy")
        source_type = evidence.get("source_type", "memory")
        source_path = evidence.get("source_path", Path("."))
    else:
        src_path = Path(source).resolve()
        if not src_path.exists():
            raise ReleaseGateError(f"Evidence source path '{src_path}' does not exist.")

        if src_path.is_file() and src_path.suffix.lower() == ".zip":
            source_type = "audit_bundle"
            source_path = src_path
            from mlforge.application.packaging import verify_audit_bundle
            bundle_res = verify_audit_bundle(src_path, raise_on_error=False)
            if not bundle_res.is_valid:
                # Immediate blocker on corrupted or tampered bundle
                gate_fail = GateCheckResult(
                    gate_id="bundle_integrity",
                    status=GateCheckStatus.FAILED,
                    is_blocking=True,
                    explanation=f"Audit bundle archive verification failed: {'; '.join(bundle_res.errors)}",
                    reason_code="AUDIT_BUNDLE_CORRUPT_OR_TAMPERED",
                    evidence_references={"errors": bundle_res.errors},
                )
                return ReleaseGateResult(
                    run_id=bundle_res.run_id or src_path.stem,
                    overall_status=ReleaseGateStatus.BLOCKED,
                    policy_id=policy.policy_id,
                    policy_version=policy.version,
                    target_environment=policy.target_environment,
                    gate_results=[gate_fail],
                    evaluated_at=now_utc,
                    is_releasable=False,
                    summary="Release blocked: audit bundle failed structural and cryptographic verification.",
                )

            manifest = bundle_res.manifest
            # Load evaluation and decision from bundle
            import zipfile
            with zipfile.ZipFile(src_path, mode="r") as zf:
                try:
                    with zf.open("evaluation.json") as ef:
                        evaluation = EvaluationResult.model_validate(json.loads(ef.read().decode("utf-8")))
                except Exception as exc:
                    raise ReleaseGateError(f"Failed to read evaluation from bundle '{src_path}': {exc}") from exc

                decision = None
                if "decision.json" in zf.namelist():
                    try:
                        with zf.open("decision.json") as df:
                            decision = AcceptanceDecision.model_validate(json.loads(df.read().decode("utf-8")))
                    except Exception as exc:
                        raise ReleaseGateError(f"Failed to read decision from bundle '{src_path}': {exc}") from exc

                saved_policy = None
                if "policy.json" in zf.namelist():
                    try:
                        with zf.open("policy.json") as pf:
                            saved_policy = AcceptancePolicy.model_validate(json.loads(pf.read().decode("utf-8")))
                    except Exception as exc:
                        raise ReleaseGateError(f"Failed to read policy from bundle '{src_path}': {exc}") from exc

        elif src_path.is_dir():
            source_type = "run_dir"
            source_path = src_path
            man_file = src_path / "manifest.json"
            if not man_file.exists():
                raise ReleaseGateError(f"Manifest file 'manifest.json' missing in directory '{src_path}'.")
            try:
                with open(man_file, "r", encoding="utf-8") as f:
                    manifest_dict = json.load(f)
                manifest = ArtifactManifest.model_validate(manifest_dict)
            except Exception as exc:
                raise ReleaseGateError(f"Failed to load manifest from '{man_file}': {exc}") from exc

            # evaluation.json
            eval_file = src_path / "evaluation.json"
            evaluation = None
            if eval_file.exists():
                try:
                    with open(eval_file, "r", encoding="utf-8") as f:
                        eval_dict = json.load(f)
                    evaluation = EvaluationResult.model_validate(eval_dict)
                except Exception:
                    evaluation = None

            # decision.json
            dec_file = src_path / "decision.json"
            decision = None
            if dec_file.exists():
                try:
                    with open(dec_file, "r", encoding="utf-8") as f:
                        dec_dict = json.load(f)
                    if isinstance(dec_dict.get("outcome"), str):
                        dec_dict["outcome"] = dec_dict["outcome"].upper()
                    decision = AcceptanceDecision.model_validate(dec_dict)
                except Exception:
                    decision = None

            # policy.json
            pol_file = src_path / "policy.json"
            saved_policy = None
            if pol_file.exists():
                try:
                    with open(pol_file, "r", encoding="utf-8") as f:
                        pol_dict = json.load(f)
                    saved_policy = AcceptancePolicy.model_validate(pol_dict)
                except Exception:
                    saved_policy = None
        else:
            raise ReleaseGateError(f"Evidence source '{src_path}' must be a directory or a .zip archive.")

    run_id = getattr(manifest, "run_id", "unknown_run")
    gate_checks: List[GateCheckResult] = []

    # -----------------------------------------------------------------------
    # Gate 1: Schema Compatibility
    # -----------------------------------------------------------------------
    schema_ver = getattr(manifest, "schema_version", "")
    if any(schema_ver.startswith(p) for p in SUPPORTED_SCHEMA_PREFIXES):
        gate_checks.append(
            GateCheckResult(
                gate_id="schema_compatibility",
                status=GateCheckStatus.PASSED,
                is_blocking=True,
                explanation=f"Evaluation schema version '{schema_ver}' is supported.",
                reason_code="SCHEMA_SUPPORTED",
                evidence_references={"schema_version": schema_ver},
            )
        )
    else:
        gate_checks.append(
            GateCheckResult(
                gate_id="schema_compatibility",
                status=GateCheckStatus.FAILED,
                is_blocking=True,
                explanation=f"Unsupported schema version '{schema_ver}'. Supported prefixes: {SUPPORTED_SCHEMA_PREFIXES}",
                reason_code="UNSUPPORTED_SCHEMA_VERSION",
                evidence_references={"schema_version": schema_ver},
            )
        )

    # -----------------------------------------------------------------------
    # Gate 2: Evidence Completeness
    # -----------------------------------------------------------------------
    completeness_reasons: List[str] = []
    completeness_inconclusive: List[str] = []

    if not evaluation:
        completeness_reasons.append("Missing required EvaluationResult object.")
    else:
        # Check folds
        fold_results = evaluation.candidate_result.fold_results
        if len(fold_results) < policy.min_evaluated_folds:
            completeness_reasons.append(
                f"Candidate evaluated folds ({len(fold_results)}) below required minimum ({policy.min_evaluated_folds})."
            )

        # Baseline comparison
        if policy.require_baseline_comparison:
            if not evaluation.comparison_report or not evaluation.comparison_report.metric_comparisons:
                completeness_reasons.append("Policy mandates baseline comparison, but comparison report is missing or empty.")

        # Robustness
        if policy.require_robustness_evidence:
            cand_rob = evaluation.candidate_result.robustness
            if not cand_rob or not cand_rob.curves:
                completeness_reasons.append("Policy mandates robustness stress testing, but robustness curves are missing.")

        # Calibration
        if policy.require_calibration_evidence:
            cand_cal = evaluation.candidate_result.calibration
            if not cand_cal or not cand_cal.is_usable:
                completeness_inconclusive.append("Policy mandates calibration analysis, but calibration curve is missing or marked unusable.")

    if completeness_reasons:
        gate_checks.append(
            GateCheckResult(
                gate_id="evidence_completeness",
                status=GateCheckStatus.FAILED,
                is_blocking=True,
                explanation=f"Evaluation evidence is incomplete: {'; '.join(completeness_reasons)}",
                reason_code="EVIDENCE_INCOMPLETE",
                evidence_references={"reasons": completeness_reasons},
            )
        )
    elif completeness_inconclusive:
        gate_checks.append(
            GateCheckResult(
                gate_id="evidence_completeness",
                status=GateCheckStatus.INCONCLUSIVE,
                is_blocking=True,
                explanation=f"Evaluation evidence inconclusive: {'; '.join(completeness_inconclusive)}",
                reason_code="EVIDENCE_INCONCLUSIVE",
                evidence_references={"inconclusives": completeness_inconclusive},
            )
        )
    else:
        gate_checks.append(
            GateCheckResult(
                gate_id="evidence_completeness",
                status=GateCheckStatus.PASSED,
                is_blocking=True,
                explanation="All required evaluation evidence and cross-validation partitions are complete.",
                reason_code="EVIDENCE_COMPLETE",
                evidence_references={"fold_count": len(evaluation.candidate_result.fold_results) if evaluation else 0},
            )
        )

    # -----------------------------------------------------------------------
    # Gate 3: Acceptance Policy Compliance
    # -----------------------------------------------------------------------
    if policy.require_acceptance_pass:
        if not decision:
            gate_checks.append(
                GateCheckResult(
                    gate_id="acceptance_compliance",
                    status=GateCheckStatus.FAILED,
                    is_blocking=True,
                    explanation="Policy mandates conclusive PASS acceptance decision, but decision.json is missing.",
                    reason_code="ACCEPTANCE_DECISION_MISSING",
                )
            )
        else:
            outcome_val = decision.outcome.value if hasattr(decision.outcome, "value") else str(decision.outcome)
            if outcome_val == AcceptanceOutcome.PASS.value:
                if not policy.allow_advisory_warnings and decision.advisory_warnings:
                    gate_checks.append(
                        GateCheckResult(
                            gate_id="acceptance_compliance",
                            status=GateCheckStatus.FAILED,
                            is_blocking=True,
                            explanation=f"Acceptance passed with {len(decision.advisory_warnings)} advisory warnings, but policy forbids advisories.",
                            reason_code="ADVISORY_WARNINGS_FORBIDDEN",
                            evidence_references={"advisories": decision.advisory_warnings},
                        )
                    )
                else:
                    # Check policy ID/version consistency if policy artifact is present
                    pol_conflict = False
                    if saved_policy:
                        if saved_policy.policy_id != decision.policy_id or saved_policy.version != decision.policy_version:
                            pol_conflict = True

                    if pol_conflict:
                        gate_checks.append(
                            GateCheckResult(
                                gate_id="acceptance_compliance",
                                status=GateCheckStatus.FAILED,
                                is_blocking=True,
                                explanation=(
                                    f"Inconsistent policy identity: decision references '{decision.policy_id}:{decision.policy_version}', "
                                    f"while policy.json defines '{saved_policy.policy_id}:{saved_policy.version}'."
                                ),
                                reason_code="POLICY_VERSION_MISMATCH",
                                evidence_references={"decision_policy": decision.policy_id, "file_policy": saved_policy.policy_id},
                            )
                        )
                    else:
                        gate_checks.append(
                            GateCheckResult(
                                gate_id="acceptance_compliance",
                                status=GateCheckStatus.PASSED,
                                is_blocking=True,
                                explanation=f"Acceptance decision conclusively passed ({decision.passed_rules_count} rules passed).",
                                reason_code="ACCEPTANCE_PASSED",
                                evidence_references={
                                    "decision_id": decision.decision_id,
                                    "policy_id": decision.policy_id,
                                    "policy_version": decision.policy_version,
                                    "passed_count": decision.passed_rules_count,
                                },
                            )
                        )
            elif outcome_val == AcceptanceOutcome.FAIL.value:
                gate_checks.append(
                    GateCheckResult(
                        gate_id="acceptance_compliance",
                        status=GateCheckStatus.FAILED,
                        is_blocking=True,
                        explanation=f"Acceptance decision FAILED ({decision.failed_rules_count} rules violated: {decision.summary_rationale}).",
                        reason_code="ACCEPTANCE_FAILED",
                        evidence_references={"failed_count": decision.failed_rules_count, "rationale": decision.summary_rationale},
                    )
                )
            else:  # INCONCLUSIVE
                gate_checks.append(
                    GateCheckResult(
                        gate_id="acceptance_compliance",
                        status=GateCheckStatus.INCONCLUSIVE,
                        is_blocking=True,
                        explanation=f"Acceptance decision is INCONCLUSIVE ({decision.inconclusive_rules_count} rules inconclusive).",
                        reason_code="ACCEPTANCE_INCONCLUSIVE",
                        evidence_references={"inconclusive_count": decision.inconclusive_rules_count},
                    )
                )
    else:
        gate_checks.append(
            GateCheckResult(
                gate_id="acceptance_compliance",
                status=GateCheckStatus.PASSED,
                is_blocking=False,
                explanation="Acceptance pass not mandated by release policy.",
                reason_code="ACCEPTANCE_NOT_MANDATED",
            )
        )

    # -----------------------------------------------------------------------
    # Gate 4: Artifact Integrity & Manifest Checksums
    # -----------------------------------------------------------------------
    integrity_failures: List[str] = []
    if policy.require_manifest_integrity and source_type == "run_dir":
        checksums = getattr(manifest, "file_checksums", getattr(manifest, "artifact_checksums", {}))
        if not checksums:
            integrity_failures.append("Manifest does not record any artifact checksums.")
        else:
            for rel_str, expected_hash in checksums.items():
                try:
                    target_file = _validate_safe_path(source_path, rel_str)
                    if not target_file.exists():
                        integrity_failures.append(f"Manifest entry missing on disk: '{rel_str}'")
                    else:
                        actual_hash = _sha256_file(target_file)
                        if actual_hash != expected_hash:
                            integrity_failures.append(
                                f"Checksum mismatch for '{rel_str}': expected {expected_hash[:12]}, got {actual_hash[:12]}"
                            )
                except ArtifactSecurityError as sec_exc:
                    integrity_failures.append(f"Security error: {sec_exc}")
                except Exception as exc:
                    integrity_failures.append(f"Error checking '{rel_str}': {exc}")

    # Check model artifact presence if required
    if policy.require_model_artifact:
        cand_rel = "models/candidate_pipeline.joblib"
        if source_type == "run_dir":
            cand_path = source_path / cand_rel
            if not cand_path.exists():
                integrity_failures.append(f"Required candidate model artifact missing: '{cand_rel}'")
            elif source_path:
                recorded_hashes = getattr(manifest, "file_checksums", getattr(manifest, "artifact_checksums", {}))
                if cand_rel in recorded_hashes:
                    actual_cand_hash = _sha256_file(cand_path)
                    if actual_cand_hash != recorded_hashes[cand_rel]:
                        integrity_failures.append("Candidate model artifact checksum mismatch.")
        elif source_type == "audit_bundle":
            bundle_files = getattr(manifest, "files", [])
            has_cand = any(getattr(f, "path", "") == cand_rel for f in bundle_files)
            if not has_cand:
                integrity_failures.append(f"Required candidate model artifact missing in bundle: '{cand_rel}'")

    if integrity_failures:
        gate_checks.append(
            GateCheckResult(
                gate_id="artifact_integrity",
                status=GateCheckStatus.FAILED,
                is_blocking=True,
                explanation=f"Artifact integrity verification failed: {'; '.join(integrity_failures)}",
                reason_code="ARTIFACT_INTEGRITY_TAMPERED_OR_MISSING",
                evidence_references={"failures": integrity_failures},
            )
        )
    else:
        gate_checks.append(
            GateCheckResult(
                gate_id="artifact_integrity",
                status=GateCheckStatus.PASSED,
                is_blocking=True,
                explanation="All recorded artifacts exist and match cryptographic SHA-256 digests.",
                reason_code="ARTIFACT_INTEGRITY_VERIFIED",
            )
        )

    # -----------------------------------------------------------------------
    # Gate 5: Provenance Consistency
    # -----------------------------------------------------------------------
    provenance_failures: List[str] = []
    if evaluation:
        # Dataset checksum check
        manifest_ds = getattr(manifest, "dataset_checksum", None)
        eval_ds = evaluation.dataset_metadata.checksum_sha256
        if manifest_ds and manifest_ds != eval_ds:
            provenance_failures.append(
                f"Dataset checksum mismatch: manifest has '{manifest_ds[:12]}', evaluation has '{eval_ds[:12]}'"
            )

        # Model provenance check
        cand_res = evaluation.candidate_result
        if cand_res.model_provenance == "final_cv_fold":
            if cand_res.fitted_fold_index is None:
                provenance_failures.append("Candidate specifies 'final_cv_fold' provenance but fitted_fold_index is missing.")
            elif cand_res.fitted_fold_index >= len(cand_res.fold_results):
                provenance_failures.append(
                    f"Fitted fold index ({cand_res.fitted_fold_index}) exceeds evaluated fold count ({len(cand_res.fold_results)})."
                )

    if provenance_failures:
        gate_checks.append(
            GateCheckResult(
                gate_id="provenance_consistency",
                status=GateCheckStatus.FAILED,
                is_blocking=True,
                explanation=f"Provenance inconsistency detected: {'; '.join(provenance_failures)}",
                reason_code="PROVENANCE_INCONSISTENCY",
                evidence_references={"failures": provenance_failures},
            )
        )
    else:
        gate_checks.append(
            GateCheckResult(
                gate_id="provenance_consistency",
                status=GateCheckStatus.PASSED,
                is_blocking=True,
                explanation="Model provenance, dataset checksum, and fold retention metadata are consistent.",
                reason_code="PROVENANCE_CONSISTENT",
            )
        )

    # -----------------------------------------------------------------------
    # Decision Precedence & Overall Verdict
    # -----------------------------------------------------------------------
    # Precedence: BLOCKED (any blocking FAILED) > INCONCLUSIVE (any blocking INCONCLUSIVE) > PASSED
    has_blocker_fail = any(g.status == GateCheckStatus.FAILED and g.is_blocking for g in gate_checks)
    has_blocker_inconclusive = any(g.status == GateCheckStatus.INCONCLUSIVE and g.is_blocking for g in gate_checks)

    if has_blocker_fail:
        overall_status = ReleaseGateStatus.BLOCKED
        failed_gates = [g.gate_id for g in gate_checks if g.status == GateCheckStatus.FAILED and g.is_blocking]
        summary = f"Release authorization BLOCKED: {len(failed_gates)} gate check(s) failed ({', '.join(failed_gates)})."
    elif has_blocker_inconclusive:
        overall_status = ReleaseGateStatus.INCONCLUSIVE
        inc_gates = [g.gate_id for g in gate_checks if g.status == GateCheckStatus.INCONCLUSIVE and g.is_blocking]
        summary = f"Release authorization INCONCLUSIVE: {len(inc_gates)} gate check(s) could not be conclusively validated ({', '.join(inc_gates)})."
    else:
        overall_status = ReleaseGateStatus.PASSED
        summary = f"Release authorization PASSED: all {len(gate_checks)} gate requirements conclusively satisfied."

    is_releasable = (overall_status == ReleaseGateStatus.PASSED)

    return ReleaseGateResult(
        run_id=run_id,
        overall_status=overall_status,
        policy_id=policy.policy_id,
        policy_version=policy.version,
        target_environment=policy.target_environment,
        gate_results=gate_checks,
        evaluated_at=now_utc,
        is_releasable=is_releasable,
        summary=summary,
    )
