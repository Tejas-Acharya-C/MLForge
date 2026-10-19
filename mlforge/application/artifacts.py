"""MLForge Local Artifact Serialization and Loading Module.

Safely serializes and validates evaluation evidence, reproducibility metadata,
acceptance policies, and decisions into auditable local JSON artifacts.
"""

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from mlforge.core.contracts import (
    AcceptanceDecision,
    AcceptancePolicy,
    ArtifactManifest,
    EvaluationResult,
)
from mlforge.core.exceptions import (
    ArtifactLoadError,
    ArtifactSecurityError,
    ArtifactSerializationError,
)

CURRENT_SCHEMA_VERSION = "1.0.0"


def _validate_finite_json_values(obj: Any) -> None:
    """Recursively verify that no float NaN or +/- Inf values exist in data structure."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ArtifactSerializationError(
                f"Data contains non-standard or non-finite values (NaN/Inf): {obj}"
            )
    elif isinstance(obj, dict):
        for v in obj.values():
            _validate_finite_json_values(v)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _validate_finite_json_values(item)


def _atomic_write_json(
    target_path: Path,
    data: Dict[str, Any],
) -> str:
    """Atomically write JSON data to disk and return its SHA-256 hex digest.

    Parameters
    ----------
    target_path : Path
        Destination file path.
    data : dict
        Serializable dictionary.

    Returns
    -------
    str
        SHA-256 checksum of the canonical serialized UTF-8 bytes.

    Raises
    ------
    ArtifactSerializationError
        If serialization fails (e.g. non-finite values) or write fails.
    """
    _validate_finite_json_values(data)
    tmp_path = target_path.with_name(f"{target_path.name}.tmp_{os.getpid()}")
    try:
        # allow_nan=False strictly rejects NaN and +/- Infinity
        serialized_str = json.dumps(data, indent=2, sort_keys=True, allow_nan=False)
        utf8_bytes = serialized_str.encode("utf-8")
        checksum = hashlib.sha256(utf8_bytes).hexdigest()

        with open(tmp_path, "wb") as f:
            f.write(utf8_bytes)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, target_path)
        return checksum
    except ArtifactSerializationError:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    except ValueError as exc:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise ArtifactSerializationError(
            f"Failed to serialize '{target_path.name}': contains non-standard or non-finite values (NaN/Inf): {exc}"
        ) from exc
    except Exception as exc:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise ArtifactSerializationError(
            f"Atomic file write failed for '{target_path.name}': {exc}"
        ) from exc


def _validate_is_fitted(estimator: Any) -> None:
    """Ensure estimator or pipeline is fitted prior to serialization.

    Raises
    ------
    ArtifactSerializationError
        If estimator is unfitted or invalid.
    """
    if estimator is None:
        raise ArtifactSerializationError("Cannot serialize model: estimator is None.")

    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    try:
        check_is_fitted(estimator)
    except NotFittedError as exc:
        raise ArtifactSerializationError(
            f"Cannot serialize unfitted estimator '{type(estimator).__name__}': {exc}"
        ) from exc
    except TypeError:
        # Check trailing underscore attributes for non-standard or custom estimator wrappers
        fitted_attrs = [v for v in dir(estimator) if v.endswith("_") and not v.startswith("__")]
        if not fitted_attrs:
            raise ArtifactSerializationError(
                f"Estimator '{type(estimator).__name__}' does not have any fitted attributes."
            )


def _atomic_write_model(
    target_path: Path,
    estimator: Any,
) -> str:
    """Atomically serialize fitted estimator to disk using joblib and return SHA-256 digest.

    SECURITY WARNING
    ----------------
    Model files (.joblib) are executable serialization artifacts. Deserializing untrusted
    joblib/pickle files can lead to arbitrary code execution. Never load joblib model
    artifacts from untrusted or unverified sources.

    Parameters
    ----------
    target_path : Path
        Destination .joblib file path.
    estimator : Any
        Fitted estimator or pipeline to persist.

    Returns
    -------
    str
        SHA-256 checksum of the final serialized file bytes.

    Raises
    ------
    ArtifactSerializationError
        If estimator is unfitted or write fails.
    """
    _validate_is_fitted(estimator)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.name}.tmp_{os.getpid()}")
    try:
        import joblib
        joblib.dump(estimator, tmp_path)
        with open(tmp_path, "rb") as f:
            raw_bytes = f.read()
        checksum = hashlib.sha256(raw_bytes).hexdigest()
        os.replace(tmp_path, target_path)
        return checksum
    except ArtifactSerializationError:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    except Exception as exc:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise ArtifactSerializationError(
            f"Atomic model serialization failed for '{target_path.name}': {exc}"
        ) from exc


def _validate_and_resolve_run_dir(
    artifacts_dir: Union[str, Path],
    run_id: Optional[str],
    evaluation: EvaluationResult,
) -> tuple[str, Path]:
    """Validate and resolve target run directory safely within artifacts root.

    Raises
    ------
    ArtifactSerializationError
        If run_id contains invalid characters, path separators, traversal components,
        or resolves outside the intended artifact root.
    """
    if run_id is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dataset_short_hash = evaluation.dataset_metadata.checksum_sha256[:8]
        active_run_id = f"run_{ts}_{dataset_short_hash}"
    else:
        if not isinstance(run_id, str):
            raise ArtifactSerializationError(f"Run ID must be a string, got {type(run_id).__name__}.")
        active_run_id = run_id.strip()
        if not active_run_id:
            raise ArtifactSerializationError("Run ID cannot be empty or pure whitespace.")
        if "\0" in run_id:
            raise ArtifactSerializationError("Run ID must not contain null bytes.")
        if "/" in run_id or "\\" in run_id or os.sep in run_id or (os.altsep and os.altsep in run_id):
            raise ArtifactSerializationError(f"Run ID '{run_id}' contains forbidden path separators.")
        if ":" in run_id:
            raise ArtifactSerializationError(f"Run ID '{run_id}' contains forbidden drive or protocol separators.")
        if active_run_id in (".", "..") or ".." in active_run_id:
            raise ArtifactSerializationError(f"Run ID '{run_id}' contains directory traversal components.")
        if Path(run_id).is_absolute():
            raise ArtifactSerializationError(f"Run ID '{run_id}' must not be an absolute path.")

    root_path = Path(artifacts_dir).resolve()
    target_dir = (root_path / active_run_id).resolve()

    try:
        target_dir.relative_to(root_path)
    except ValueError:
        raise ArtifactSerializationError(
            f"Run ID '{active_run_id}' escapes artifact root '{root_path}'."
        )

    if target_dir == root_path:
        raise ArtifactSerializationError("Run ID cannot resolve directly to the artifact root directory.")

    return active_run_id, target_dir


def serialize_run_artifacts(
    evaluation: EvaluationResult,
    acceptance_decision: Optional[AcceptanceDecision] = None,
    artifacts_dir: Union[str, Path] = "artifacts/runs",
    run_id: Optional[str] = None,
    policy: Optional[AcceptancePolicy] = None,
    candidate_pipeline: Optional[Any] = None,
    baseline_pipeline: Optional[Any] = None,
) -> Path:
    """Serialize complete evaluation evidence, models, and acceptance decisions to a local run directory.

    Parameters
    ----------
    evaluation : EvaluationResult
        Completed model evaluation contract.
    acceptance_decision : AcceptanceDecision, optional
        Evaluated acceptance decision. If None, checks evaluation.acceptance_decision.
    artifacts_dir : str or Path, default="artifacts/runs"
        Root folder for evaluation runs.
    run_id : str, optional
        Unique run identifier. If None, generated from timestamp and dataset checksum.
    policy : AcceptancePolicy, optional
        Explicit acceptance policy evaluated for this run.
    candidate_pipeline : Any, optional
        Explicit fitted candidate pipeline. If None, retrieved from evaluation.candidate_result.fitted_pipeline.
    baseline_pipeline : Any, optional
        Explicit fitted baseline pipeline. If None, retrieved from evaluation.baseline_result.fitted_pipeline.

    Returns
    -------
    Path
        Absolute path to the serialized run directory.

    Raises
    ------
    ArtifactSerializationError
        If inputs are invalid, run_id violates path safety, destination conflicts,
        or an unfitted model is provided.
    """
    if evaluation is None:
        raise ArtifactSerializationError("Cannot serialize artifacts: evaluation result is None.")

    decision = acceptance_decision or evaluation.acceptance_decision

    # Validate and resolve run directory with path confinement defense
    active_run_id, target_dir = _validate_and_resolve_run_dir(
        artifacts_dir=artifacts_dir,
        run_id=run_id,
        evaluation=evaluation,
    )

    # Guard: prevent overwriting a directory created by a different run
    if target_dir.exists():
        manifest_file = target_dir / "manifest.json"
        if manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    existing_manifest = json.load(f)
                if existing_manifest.get("run_id") != active_run_id:
                    raise ArtifactSerializationError(
                        f"Target directory '{target_dir}' already contains artifacts for different run "
                        f"'{existing_manifest.get('run_id')}'. Overwrite denied."
                    )
            except Exception as exc:
                if isinstance(exc, ArtifactSerializationError):
                    raise
                pass

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise ArtifactSerializationError(f"Cannot create artifact directory '{target_dir}': {exc}") from exc

    file_checksums: Dict[str, str] = {}

    try:
        # 1. Serialize evaluation.json
        eval_dict = evaluation.model_dump(mode="json")
        eval_path = target_dir / "evaluation.json"
        file_checksums["evaluation.json"] = _atomic_write_json(eval_path, eval_dict)

        # 2. Serialize decision.json (if present)
        if decision is not None:
            dec_dict = decision.model_dump(mode="json")
            dec_path = target_dir / "decision.json"
            file_checksums["decision.json"] = _atomic_write_json(dec_path, dec_dict)

        # 3. Serialize policy.json (if present)
        if policy is not None:
            pol_dict = policy.model_dump(mode="json")
            pol_path = target_dir / "policy.json"
            file_checksums["policy.json"] = _atomic_write_json(pol_path, pol_dict)

        # 4. Serialize fitted model pipelines (candidate and baseline when available)
        cand_model = candidate_pipeline
        if cand_model is None and evaluation.candidate_result is not None:
            cand_model = getattr(evaluation.candidate_result, "fitted_pipeline", None)

        if cand_model is not None:
            cand_path = target_dir / "models" / "candidate_pipeline.joblib"
            file_checksums["models/candidate_pipeline.joblib"] = _atomic_write_model(cand_path, cand_model)

        base_model = baseline_pipeline
        if base_model is None and evaluation.baseline_result is not None:
            base_model = getattr(evaluation.baseline_result, "fitted_pipeline", None)

        if base_model is not None:
            base_path = target_dir / "models" / "baseline_pipeline.joblib"
            file_checksums["models/baseline_pipeline.joblib"] = _atomic_write_model(base_path, base_model)

        # 5. Serialize manifest.json (indexes all written files and their checksums)
        has_models = bool(
            "models/candidate_pipeline.joblib" in file_checksums
            or "models/baseline_pipeline.joblib" in file_checksums
        )

        cand_fold_idx = None
        if "models/candidate_pipeline.joblib" in file_checksums:
            cand_fold_idx = getattr(evaluation.candidate_result, "fitted_fold_index", None)
            if cand_fold_idx is None and evaluation.reproducibility.num_folds:
                cand_fold_idx = evaluation.reproducibility.num_folds - 1

        base_fold_idx = None
        if "models/baseline_pipeline.joblib" in file_checksums:
            base_fold_idx = getattr(evaluation.baseline_result, "fitted_fold_index", None)
            if base_fold_idx is None and evaluation.reproducibility.num_folds:
                base_fold_idx = evaluation.reproducibility.num_folds - 1

        manifest = ArtifactManifest(
            schema_version=CURRENT_SCHEMA_VERSION,
            run_id=active_run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            dataset_checksum=evaluation.dataset_metadata.checksum_sha256,
            split_strategy=evaluation.reproducibility.split_strategy,
            num_folds=evaluation.reproducibility.num_folds,
            candidate_model_type=evaluation.candidate_result.model_type,
            baseline_model_type=evaluation.baseline_result.model_type,
            policy_id=decision.policy_id if decision else (policy.policy_id if policy else None),
            policy_version=decision.policy_version if decision else (policy.version if policy else None),
            decision_outcome=decision.outcome if decision else None,
            software_versions=evaluation.reproducibility.package_versions,
            file_checksums=file_checksums,
            has_robustness=bool(
                (evaluation.candidate_result and evaluation.candidate_result.robustness is not None)
                or (evaluation.baseline_result and evaluation.baseline_result.robustness is not None)
            ),
            has_models=has_models,
            model_provenance="final_cv_fold" if has_models else None,
            candidate_model_fold_index=cand_fold_idx,
            baseline_model_fold_index=base_fold_idx,
        )

        manifest_path = target_dir / "manifest.json"
        _atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
    except ArtifactSerializationError:
        raise
    except Exception as exc:
        raise ArtifactSerializationError(f"Failed to serialize artifacts: {exc}") from exc

    return target_dir.resolve()


def load_run_artifacts(run_dir: Union[str, Path]) -> Dict[str, Any]:
    """Load and validate serialized run artifacts from disk.

    Parameters
    ----------
    run_dir : str or Path
        Directory containing run artifacts.

    Returns
    -------
    dict
        Dictionary containing loaded and validated objects:
        'manifest' -> ArtifactManifest
        'evaluation' -> EvaluationResult
        'decision' -> Optional[AcceptanceDecision]
        'policy' -> Optional[AcceptancePolicy]

    Raises
    ------
    ArtifactLoadError
        If directory does not exist, schema version is unsupported, or files fail checksum verification.
    """
    path = Path(run_dir)
    if not path.exists() or not path.is_dir():
        raise ArtifactLoadError(f"Artifact directory '{path}' does not exist or is not a directory.")

    manifest_file = path / "manifest.json"
    if not manifest_file.exists():
        raise ArtifactLoadError(f"Missing required artifact manifest: '{manifest_file}'.")

    try:
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_dict = json.load(f)
    except Exception as exc:
        raise ArtifactLoadError(f"Malformed manifest JSON in '{manifest_file}': {exc}") from exc

    # Validate schema version
    schema_version = manifest_dict.get("schema_version")
    if schema_version != CURRENT_SCHEMA_VERSION:
        raise ArtifactLoadError(
            f"Unsupported artifact schema version '{schema_version}'. Supported version: '{CURRENT_SCHEMA_VERSION}'."
        )

    try:
        manifest = ArtifactManifest.model_validate(manifest_dict)
    except Exception as exc:
        raise ArtifactLoadError(f"Invalid artifact manifest structure: {exc}") from exc

    # Integrity verification: verify SHA-256 for all indexed files
    for filename, expected_hash in manifest.file_checksums.items():
        artifact_path = path / filename
        if not artifact_path.exists():
            raise ArtifactLoadError(f"Indexed artifact file '{filename}' was not found in '{path}'.")

        try:
            with open(artifact_path, "rb") as f:
                content_bytes = f.read()
            actual_hash = hashlib.sha256(content_bytes).hexdigest()
        except Exception as exc:
            raise ArtifactLoadError(f"Could not read artifact file '{filename}': {exc}") from exc

        if actual_hash != expected_hash:
            raise ArtifactLoadError(
                f"Integrity check failed for '{filename}': checksum mismatch "
                f"(expected {expected_hash}, computed {actual_hash}). File may have been altered or corrupted."
            )

    # Deserialize evaluation.json
    eval_file = path / "evaluation.json"
    if not eval_file.exists():
        raise ArtifactLoadError(f"Missing evaluation evidence file: '{eval_file}'.")

    try:
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_dict = json.load(f)
        evaluation = EvaluationResult.model_validate(eval_dict)
    except Exception as exc:
        raise ArtifactLoadError(f"Failed to deserialize evaluation evidence from '{eval_file}': {exc}") from exc

    # Deserialize decision.json if present
    decision: Optional[AcceptanceDecision] = None
    dec_file = path / "decision.json"
    if dec_file.exists():
        try:
            with open(dec_file, "r", encoding="utf-8") as f:
                dec_dict = json.load(f)
            decision = AcceptanceDecision.model_validate(dec_dict)
        except Exception as exc:
            raise ArtifactLoadError(f"Failed to deserialize decision from '{dec_file}': {exc}") from exc

    # Deserialize policy.json if present
    policy: Optional[AcceptancePolicy] = None
    pol_file = path / "policy.json"
    if pol_file.exists():
        try:
            with open(pol_file, "r", encoding="utf-8") as f:
                pol_dict = json.load(f)
            policy = AcceptancePolicy.model_validate(pol_dict)
        except Exception as exc:
            raise ArtifactLoadError(f"Failed to deserialize policy from '{pol_file}': {exc}") from exc

    return {
        "manifest": manifest,
        "evaluation": evaluation,
        "decision": decision,
        "policy": policy,
    }


SUPPORTED_MODEL_ROLES = {"candidate", "baseline"}


def load_fitted_model(
    run_dir: Union[str, Path],
    model_role: str = "candidate",
    trusted: bool = False,
) -> Any:
    """Safely verify integrity and optionally deserialize a fitted model pipeline.

    SECURITY WARNING
    ----------------
    Model files (.joblib) are executable serialization artifacts. Deserializing untrusted
    joblib/pickle files can execute arbitrary code. Callers must pass trusted=True only
    after independent verification of the file origin.

    Parameters
    ----------
    run_dir : str or Path
        Directory containing the serialized run artifacts.
    model_role : str, default='candidate'
        Role of model: 'candidate' or 'baseline'.
    trusted : bool, default=False
        Safety opt-in flag confirming model source is trusted.

    Returns
    -------
    Any
        The deserialized fitted scikit-learn pipeline or estimator.

    Raises
    ------
    ArtifactSecurityError
        If model_role is invalid or if trusted is False.
    ArtifactLoadError
        If model file is missing or fails checksum integrity verification.
    """
    # 1. Strict model_role validation prior to path construction or filesystem access
    if not isinstance(model_role, str) or model_role not in SUPPORTED_MODEL_ROLES:
        raise ArtifactSecurityError(
            f"Invalid model role '{model_role}'. Must be one of exact supported roles: {sorted(SUPPORTED_MODEL_ROLES)}."
        )

    # 2. Safety opt-in check
    if not trusted:
        raise ArtifactSecurityError(
            f"Refusing to deserialize '{model_role}' model: joblib files are executable code artifacts. "
            "Callers must explicitly set trusted=True after verifying provenance."
        )

    path = Path(run_dir).resolve()
    rel_path = f"models/{model_role}_pipeline.joblib"
    model_path = (path / rel_path).resolve()

    # Confine resolved path strictly within run directory
    try:
        model_path.relative_to(path)
    except ValueError:
        raise ArtifactSecurityError(
            f"Model path '{model_path}' escapes run directory '{path}'."
        )

    if not model_path.exists():
        raise ArtifactLoadError(f"Model file '{rel_path}' not found in '{run_dir}'.")

    # Verify against manifest checksum if manifest exists
    manifest_file = path / "manifest.json"
    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest_dict = json.load(f)
            expected_hash = manifest_dict.get("file_checksums", {}).get(rel_path)
            if expected_hash is not None:
                with open(model_path, "rb") as f:
                    actual_hash = hashlib.sha256(f.read()).hexdigest()
                if actual_hash != expected_hash:
                    raise ArtifactLoadError(
                        f"Integrity check failed for '{rel_path}': checksum mismatch "
                        f"(expected {expected_hash}, computed {actual_hash})."
                    )
        except ArtifactLoadError:
            raise
        except Exception as exc:
            raise ArtifactLoadError(f"Failed to verify model checksum against manifest: {exc}") from exc

    import joblib
    try:
        return joblib.load(model_path)
    except Exception as exc:
        raise ArtifactLoadError(f"Failed to deserialize model from '{model_path}': {exc}") from exc
