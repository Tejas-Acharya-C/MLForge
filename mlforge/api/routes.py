"""REST API route handlers for MLForge."""

import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, HTMLResponse, Response
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from benchmarks.runner import generate_synthetic_churn_dataset
from mlforge.api.schemas import (
    BundleCreateRequest,
    BundleCreateResponse,
    BundleVerifyRequest,
    CancelEvaluationRequest,
    CertificateIssueRequest,
    CertificateIssueResponse,
    CertificateVerifyRequest,
    CreateEvaluationRequest,
    DatasetInspectRequest,
    EvaluationDetailResponse,
    EvaluationListResponse,
    EvaluationStatusResponse,
    EvaluationSummaryItem,
    PolicyTemplateInfo,
    ReleaseGateEvaluateRequest,
    RobustnessRunRequest,
    SampleDatasetInfo,
)
from mlforge.application.artifacts import (
    load_fitted_model,
    load_run_artifacts,
    serialize_run_artifacts,
)
from mlforge.application.packaging import (
    build_audit_bundle,
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
from mlforge.core.contracts import (
    AcceptanceDecision,
    AcceptanceOutcome,
    AcceptancePolicy,
    AcceptanceRule,
    AuditBundleVerificationResult,
    CertificateVerificationResult,
    CriterionOutcome,
    CriterionSeverity,
    DatasetMetadata,
    EvaluationResult,
    ReleaseCertificate,
    ReleaseGateResult,
    ReleaseGateStatus,
    ReleasePolicy,
    RobustnessCurve,
    RobustnessEvaluationResult,
    RobustnessPerturbationResult,
    RuleOperator,
    RunRecord,
    RunState,
    RunSummaryMetric,
)
from mlforge.core.dataset import validate_and_extract_dataset
from mlforge.core.exceptions import (
    ArtifactSecurityError,
    DatasetValidationError,
    LifecycleTransitionError,
    MLForgeError,
    PersistenceError,
    ReleaseGateError,
    RunNotFoundError,
)
from mlforge.engine.acceptance import evaluate_acceptance_policy
from mlforge.engine.release_gate import evaluate_release_gate
from mlforge.engine.robustness import (
    create_category_dropout_stress_profile,
    create_compound_stress_profile,
    create_missingness_stress_profile,
    create_noise_stress_profile,
    evaluate_stress_profile,
    get_standard_stress_suite,
)

logger = logging.getLogger("mlforge.api.routes")

router = APIRouter(prefix="/api/v1")

# Predefined standard policies
STANDARD_POLICIES: Dict[str, AcceptancePolicy] = {
    "standard_classification_v1": AcceptancePolicy(
        policy_id="standard_classification_v1",
        version="1.0.0",
        name="Standard Tabular Classification Policy",
        description="Standard evaluation policy requiring Macro F1 and Accuracy above baseline with low calibration error.",
        fail_precedence=True,
        rules=[
            AcceptanceRule(
                rule_id="min_macro_f1",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.60,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="candidate_metric",
                description="Candidate Macro F1 must achieve at least 0.60 across validation folds.",
            ),
            AcceptanceRule(
                rule_id="min_accuracy",
                metric="accuracy",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.65,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="candidate_metric",
                description="Candidate Accuracy must reach or exceed 0.65.",
            ),
            AcceptanceRule(
                rule_id="superiority_over_baseline",
                metric="accuracy",
                operator=RuleOperator.STATISTICALLY_GREATER_THAN,
                threshold=0.0,
                alpha=0.05,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="statistical_test",
                description="Candidate Accuracy must be statistically superior to baseline dummy classifier (p < 0.05).",
            ),
            AcceptanceRule(
                rule_id="max_ece",
                metric="expected_calibration_error",
                operator=RuleOperator.LESS_THAN_OR_EQUAL,
                threshold=0.20,
                severity=CriterionSeverity.ADVISORY,
                evidence_source="calibration",
                description="Expected Calibration Error should not exceed 0.20.",
            ),
        ],
    ),
    "strict_zero_leakage_v1": AcceptancePolicy(
        policy_id="strict_zero_leakage_v1",
        version="1.0.0",
        name="Strict Zero-Leakage High-Confidence Policy",
        description="Rigorous production gating policy requiring high F1, statistically significant baseline separation, and strict calibration limits.",
        fail_precedence=True,
        rules=[
            AcceptanceRule(
                rule_id="strict_macro_f1",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.70,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="candidate_metric",
                description="Candidate Macro F1 must reach at least 0.70.",
            ),
            AcceptanceRule(
                rule_id="strict_balanced_acc",
                metric="balanced_accuracy",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.65,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="candidate_metric",
                description="Balanced accuracy must reach at least 0.65 to ensure minority class sensitivity.",
            ),
            AcceptanceRule(
                rule_id="strict_stat_superiority",
                metric="accuracy",
                operator=RuleOperator.STATISTICALLY_GREATER_THAN,
                threshold=0.0,
                alpha=0.05,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="statistical_test",
                description="Candidate Accuracy must significantly outperform baseline (p < 0.05).",
            ),
            AcceptanceRule(
                rule_id="strict_ece_limit",
                metric="expected_calibration_error",
                operator=RuleOperator.LESS_THAN_OR_EQUAL,
                threshold=0.15,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="calibration",
                description="Expected Calibration Error must be <= 0.15.",
            ),
        ],
    ),
    "permissive_v1": AcceptancePolicy(
        policy_id="permissive_v1",
        version="1.0.0",
        name="Permissive Experimental Baseline Policy",
        description="Relaxed criteria for exploratory prototyping and initial baseline sanity checks.",
        fail_precedence=True,
        rules=[
            AcceptanceRule(
                rule_id="perm_macro_f1",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.45,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="candidate_metric",
                description="Candidate Macro F1 must achieve at least 0.45.",
            ),
            AcceptanceRule(
                rule_id="perm_accuracy",
                metric="accuracy",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.50,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="candidate_metric",
                description="Candidate Accuracy must achieve at least 0.50.",
            ),
        ],
    ),
}

# Module-level default repository dependency
def get_repository() -> SQLiteRunRepository:
    """Dependency provider for SQLiteRunRepository."""
    repo = SQLiteRunRepository(db_path=Path("artifacts/mlforge.db"))
    repo.initialize_schema()
    return repo


SAFE_RUN_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


def _validate_run_id(run_id: str) -> str:
    """Validate that a run identifier consists strictly of alphanumeric, underscore, and hyphen characters."""
    if not run_id or not isinstance(run_id, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Run ID must be a non-empty string.",
        )
    clean = run_id.strip()
    if not SAFE_RUN_ID_REGEX.match(clean):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid run_id format '{run_id}'. Run IDs must contain only alphanumeric characters, hyphens, and underscores.",
        )
    return clean


def _resolve_confined_path(path_str: str, allowed_roots: Optional[List[Path]] = None) -> Path:
    """Canonically resolve a path and verify strict confinement within permitted filesystem roots."""
    if not path_str or "\x00" in path_str:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filesystem path: null bytes or empty path.",
        )
    p = Path(path_str.strip()).resolve()
    if allowed_roots is None:
        allowed_roots = [
            Path.cwd().resolve(),
            Path("artifacts").resolve(),
            Path("benchmarks").resolve(),
            Path(tempfile.gettempdir()).resolve(),
        ]
        data_dir_env = os.environ.get("MLFORGE_DATA_DIR")
        if data_dir_env:
            try:
                allowed_roots.append(Path(data_dir_env).resolve())
            except Exception:
                pass

    is_confined = any(p.is_relative_to(root) for root in allowed_roots)
    if not is_confined:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Path '{path_str}' is outside permitted directory scope.",
        )
    return p


def _load_sample_dataset(sample_name: str) -> Tuple[pd.DataFrame, str, Optional[str]]:
    """Return DataFrame, target_column, and default subgroup column for preset benchmarks."""
    name_clean = sample_name.lower().strip()
    if name_clean == "churn":
        df = generate_synthetic_churn_dataset(n_samples=500, random_state=42)
        if "churn" in df.columns:
            df = df.rename(columns={"churn": "churn_label"})
        return df, "churn_label", "region"
    elif name_clean in ("breast_cancer", "cancer"):
        data = load_breast_cancer(as_frame=True)
        df = data.frame.copy()
        df = df.rename(columns={"target": "diagnosis"})
        # Add subgroup slice feature for diagnostics
        rng = np.random.RandomState(42)
        df["institution_tier"] = rng.choice(["Tier_1_Academic", "Tier_2_Community", "Tier_3_Rural"], size=len(df), p=[0.5, 0.3, 0.2])
        return df, "diagnosis", "institution_tier"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown sample dataset preset '{sample_name}'. Available: 'churn', 'breast_cancer'.",
        )


def _load_dataset_from_request(
    file_path: Optional[str],
    sample_name: Optional[str],
) -> pd.DataFrame:
    """Defensively load a DataFrame from local file or sample preset with confinement verification."""
    if sample_name:
        df, _, _ = _load_sample_dataset(sample_name)
        return df

    if not file_path or not file_path.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'file_path' or 'sample_name' must be specified.",
        )

    # Confinement and traversal defense
    p = _resolve_confined_path(file_path)

    if not p.exists() or not p.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset file '{file_path}' does not exist or is not a regular file.",
        )

    # Defend against reading sensitive configuration or system files
    if p.name.startswith(".") or p.name in ("mlforge.db", "mlforge.db-journal", "mlforge.db-wal"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access to sensitive file '{p.name}' is forbidden.",
        )

    # Enforce reasonable file size limit for tabular dataset inspection (500 MB)
    max_dataset_bytes = 500 * 1024 * 1024
    if p.stat().st_size > max_dataset_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Dataset file size ({p.stat().st_size} bytes) exceeds maximum limit ({max_dataset_bytes} bytes).",
        )

    try:
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p)
        elif p.suffix.lower() in (".parquet", ".pq"):
            return pd.read_parquet(p)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format '{p.suffix}'. Expected .csv or .parquet.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to read dataset file: {exc}",
        )


def _get_run_artifact_dir(run_id: str, repo: SQLiteRunRepository) -> Path:
    """Retrieve and validate the artifact directory for a run with path confinement."""
    safe_run_id = _validate_run_id(run_id)
    record = repo.get_run(safe_run_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation run '{safe_run_id}' not found.",
        )
    if not record.artifact_path:
        # Fallback to standard artifact directory
        candidate_dir = (Path("artifacts/runs") / safe_run_id).resolve()
        if candidate_dir.exists() and candidate_dir.is_relative_to(Path.cwd().resolve()):
            return candidate_dir
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No artifacts have been generated for run '{safe_run_id}'.",
        )
    art_dir = Path(record.artifact_path).resolve()
    # Check confinement: must be within workspace, artifacts, or system temp
    allowed = [Path.cwd().resolve(), Path("artifacts").resolve(), Path(tempfile.gettempdir()).resolve()]
    if not any(art_dir.is_relative_to(root) for root in allowed):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Artifact directory '{art_dir}' is outside permitted directory scope.",
        )
    if not art_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact directory '{record.artifact_path}' does not exist on disk.",
        )
    return art_dir


# ============================================================================
# 1. Dataset & Policy Endpoints
# ============================================================================

@router.get("/healthz")
def api_healthz() -> Dict[str, str]:
    """Health check endpoint on /api/v1/healthz."""
    return {
        "status": "ok",
        "service": "mlforge",
        "version": "0.1.0",
    }

@router.get("/datasets/sample", response_model=List[SampleDatasetInfo])
def list_sample_datasets() -> List[SampleDatasetInfo]:
    """List available pre-packaged benchmark datasets for instant inspection and evaluation."""
    return [
        SampleDatasetInfo(
            id="churn",
            name="Customer Churn Benchmark",
            description="Synthetic tabular customer churn dataset (500 rows) with mixed numeric features, categorical contract tier and region, missing values, and binary churn label (30% churn).",
            num_rows=500,
            num_features=6,
            default_target="churn_label",
            default_subgroup="region",
        ),
        SampleDatasetInfo(
            id="breast_cancer",
            name="Breast Cancer Diagnostic Benchmark",
            description="Classic Breast Cancer Wisconsin dataset (569 rows) with 30 continuous features, binary diagnosis, and institution tiers.",
            num_rows=569,
            num_features=31,
            default_target="diagnosis",
            default_subgroup="institution_tier",
        ),
    ]


@router.post("/datasets/inspect", response_model=DatasetMetadata)
def inspect_dataset(req: DatasetInspectRequest) -> DatasetMetadata:
    """Inspect and validate a tabular dataset, extracting schema, class distributions, and data quality warnings."""
    df = _load_dataset_from_request(req.file_path, req.sample_name)
    try:
        _, _, metadata = validate_and_extract_dataset(
            df=df,
            target_column=req.target_column,
            k_folds=req.k_folds,
        )
        return metadata
    except DatasetValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dataset inspection failed: {exc}",
        )


@router.get("/acceptance-policies", response_model=List[PolicyTemplateInfo])
def list_acceptance_policies() -> List[PolicyTemplateInfo]:
    """List predefined versioned acceptance policy templates."""
    results = []
    for pol_id, pol in STANDARD_POLICIES.items():
        results.append(
            PolicyTemplateInfo(
                policy_id=pol.policy_id,
                version=pol.version,
                name=pol.name,
                description=pol.description,
                rules_count=len(pol.rules),
                policy=pol,
            )
        )
    return results


# ============================================================================
# 2. Evaluation Execution & Management
# ============================================================================

@router.post("/evaluations", response_model=EvaluationSummaryItem, status_code=status.HTTP_201_CREATED)
def submit_evaluation(
    req: CreateEvaluationRequest,
    repo: SQLiteRunRepository = Depends(get_repository),
) -> EvaluationSummaryItem:
    """Submit and execute an end-to-end evaluation run with SQLite persistence and atomic artifact generation."""
    # 1. Load dataset
    df = _load_dataset_from_request(req.file_path, req.sample_name)

    # 2. Determine acceptance policy
    policy: Optional[AcceptancePolicy] = None
    if req.custom_policy:
        policy = req.custom_policy
    elif req.policy_id is not None:
        p_id = req.policy_id.strip() if isinstance(req.policy_id, str) else req.policy_id
        if not p_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty policy_id specified. Provide a valid policy ID or omit the field.",
            )
        policy = STANDARD_POLICIES.get(p_id)
        if policy is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown policy_id '{req.policy_id}'. Available: {list(STANDARD_POLICIES.keys())}.",
            )
    else:
        # Caller genuinely omitted policy_id, use default standard policy
        policy = STANDARD_POLICIES.get("standard_classification_v1")

    # 3. Determine run ID with strict format validation
    raw_run_id = req.run_id.strip() if req.run_id else f"eval_{uuid.uuid4().hex[:12]}"
    active_run_id = _validate_run_id(raw_run_id)

    # 4. Prepare metadata
    run_meta = dict(req.run_metadata or {})
    run_meta["source"] = req.sample_name or req.file_path or "inline"
    run_meta["target_column"] = req.target_column
    run_meta["policy_id"] = req.policy_id

    # 5. Execute managed evaluation
    try:
        record, eval_result, art_dir = execute_managed_evaluation(
            df=df,
            target_column=req.target_column,
            candidate_hyperparameters=req.candidate_hyperparameters,
            num_folds=req.num_folds,
            random_state=req.random_state,
            subgroup_columns=req.subgroup_columns,
            acceptance_policy=policy,
            artifacts_dir="artifacts/runs",
            repository=repo,
            run_id=active_run_id,
            run_metadata=run_meta,
        )
    except DatasetValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Dataset validation failed: {exc}",
        )
    except MLForgeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Evaluation failed: {exc}",
        )
    except Exception as exc:
        logger.exception("Unexpected error during managed evaluation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal evaluation execution error: {exc}",
        )

    # 6. Generate self-contained offline HTML report
    try:
        generate_html_report(
            source=art_dir,
            output_path=art_dir / "report.html",
        )
    except Exception as exc:
        logger.warning(f"Non-fatal failure generating HTML report: {exc}")

    # 7. Collect summary metrics
    metrics = repo.get_run_metrics(record.run_id)
    metric_dict = {m.metric_name: m.metric_value for m in metrics}

    return EvaluationSummaryItem(
        run_id=record.run_id,
        state=record.state,
        created_at=record.created_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
        updated_at=record.updated_at,
        dataset_checksum=record.dataset_checksum,
        candidate_model_type=record.candidate_model_type,
        baseline_model_type=record.baseline_model_type,
        acceptance_outcome=record.acceptance_outcome,
        artifact_path=record.artifact_path,
        error_category=record.error_category,
        error_message=record.error_message,
        metadata=record.metadata,
        summary_metrics=metric_dict,
        metric_details=metrics,
    )


@router.get("/evaluations", response_model=EvaluationListResponse)
def list_evaluations(
    state: Optional[str] = Query(None, description="Filter by RunState: 'created', 'running', 'completed', 'failed', 'cancelled'"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    order_by: str = Query("created_at"),
    descending: bool = Query(True),
    repo: SQLiteRunRepository = Depends(get_repository),
) -> EvaluationListResponse:
    """List historical evaluation runs with pagination, state filtering, and indexed summary metrics."""
    run_state = None
    if state and state.lower() != "all":
        try:
            run_state = RunState(state.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid state filter '{state}'. Valid states: {[s.value for s in RunState]}",
            )

    records = repo.list_runs(
        state=run_state,
        limit=limit,
        offset=offset,
        order_by=order_by,
        descending=descending,
    )
    total_count = repo.count_runs(state=run_state)

    items: List[EvaluationSummaryItem] = []
    for r in records:
        metrics = repo.get_run_metrics(r.run_id)
        metric_dict = {m.metric_name: m.metric_value for m in metrics}
        items.append(
            EvaluationSummaryItem(
                run_id=r.run_id,
                state=r.state,
                created_at=r.created_at,
                started_at=r.started_at,
                completed_at=r.completed_at,
                updated_at=r.updated_at,
                dataset_checksum=r.dataset_checksum,
                candidate_model_type=r.candidate_model_type,
                baseline_model_type=r.baseline_model_type,
                acceptance_outcome=r.acceptance_outcome,
                artifact_path=r.artifact_path,
                error_category=r.error_category,
                error_message=r.error_message,
                metadata=r.metadata,
                summary_metrics=metric_dict,
                metric_details=metrics,
            )
        )

    return EvaluationListResponse(
        items=items,
        total=total_count,
        limit=limit,
        offset=offset,
    )


@router.get("/evaluations/{run_id}", response_model=EvaluationDetailResponse)
def get_evaluation_detail(
    run_id: str,
    repo: SQLiteRunRepository = Depends(get_repository),
) -> EvaluationDetailResponse:
    """Retrieve complete metadata, lifecycle history, metrics, and artifact availability for an evaluation run."""
    run_id = _validate_run_id(run_id)
    record = repo.get_run(run_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation run '{run_id}' not found.",
        )

    history = repo.get_lifecycle_history(run_id)
    metrics = repo.get_run_metrics(run_id)

    # Check artifact directory files
    art_list: List[Dict[str, Any]] = []
    has_eval = False
    has_dec = False
    has_rep = False
    has_cert = False
    has_bun = False

    if record.artifact_path:
        p = Path(record.artifact_path)
        if p.exists() and p.is_dir():
            for f in sorted(p.glob("*")):
                if f.is_file():
                    sz = f.stat().st_size
                    art_list.append({"name": f.name, "size_bytes": sz, "relative_path": str(f.name)})
                    if f.name == "evaluation.json":
                        has_eval = True
                    elif f.name == "decision.json":
                        has_dec = True
                    elif f.name == "report.html":
                        has_rep = True
                    elif f.name == "release_certificate.json":
                        has_cert = True
                    elif f.name.endswith(".zip") and "bundle" in f.name:
                        has_bun = True

    return EvaluationDetailResponse(
        record=record,
        lifecycle_history=history,
        metrics=metrics,
        artifacts=art_list,
        has_evaluation=has_eval,
        has_decision=has_dec,
        has_report=has_rep,
        has_certificate=has_cert,
        has_bundle=has_bun,
    )


@router.get("/evaluations/{run_id}/status", response_model=EvaluationStatusResponse)
def get_evaluation_status(
    run_id: str,
    repo: SQLiteRunRepository = Depends(get_repository),
) -> EvaluationStatusResponse:
    """Lightweight polling endpoint returning the current lifecycle state of a run."""
    run_id = _validate_run_id(run_id)
    record = repo.get_run(run_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation run '{run_id}' not found.",
        )
    return EvaluationStatusResponse(
        run_id=record.run_id,
        state=record.state,
        started_at=record.started_at,
        completed_at=record.completed_at,
        error_message=record.error_message,
    )


@router.post("/evaluations/{run_id}/cancel", response_model=EvaluationStatusResponse)
def cancel_evaluation(
    run_id: str,
    req: Optional[CancelEvaluationRequest] = None,
    repo: SQLiteRunRepository = Depends(get_repository),
) -> EvaluationStatusResponse:
    """Abort an active evaluation run according to state machine transition rules."""
    run_id = _validate_run_id(run_id)
    reason = req.reason if req and req.reason else "Cancelled by user via UI"
    try:
        updated = repo.transition_state(
            run_id=run_id,
            to_state=RunState.CANCELLED,
            reason=reason,
        )
        return EvaluationStatusResponse(
            run_id=updated.run_id,
            state=updated.state,
            started_at=updated.started_at,
            completed_at=updated.completed_at,
            error_message=updated.error_message,
        )
    except RunNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation run '{run_id}' not found.",
        )
    except LifecycleTransitionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )


# ============================================================================
# 3. Scientific Results & Diagnostics Endpoints
# ============================================================================

@router.get("/evaluations/{run_id}/results")
def get_evaluation_results(
    run_id: str,
    repo: SQLiteRunRepository = Depends(get_repository),
) -> Dict[str, Any]:
    """Retrieve candidate vs baseline comparison results, metric summaries, and statistical test outputs."""
    art_dir = _get_run_artifact_dir(run_id, repo)
    eval_file = art_dir / "evaluation.json"
    if not eval_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation data file 'evaluation.json' not found in run artifacts.",
        )
    try:
        data = json.loads(eval_file.read_text(encoding="utf-8"))
        return data
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read evaluation data: {exc}",
        )


@router.get("/evaluations/{run_id}/results/folds")
def get_fold_results(
    run_id: str,
    repo: SQLiteRunRepository = Depends(get_repository),
) -> List[Dict[str, Any]]:
    """Retrieve detailed fold-by-fold metrics, confusion matrices, and validation splits."""
    art_dir = _get_run_artifact_dir(run_id, repo)
    eval_file = art_dir / "evaluation.json"
    if not eval_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="evaluation.json not found.",
        )
    try:
        data = json.loads(eval_file.read_text(encoding="utf-8"))
        cand = data.get("candidate_result", {})
        folds = cand.get("fold_results", [])
        return folds
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read fold results: {exc}",
        )


@router.get("/evaluations/{run_id}/diagnostics/calibration")
def get_calibration_diagnostics(
    run_id: str,
    repo: SQLiteRunRepository = Depends(get_repository),
) -> Dict[str, Any]:
    """Retrieve probability calibration curve bins, ECE, MCE, and Brier score."""
    art_dir = _get_run_artifact_dir(run_id, repo)
    eval_file = art_dir / "evaluation.json"
    if not eval_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="evaluation.json not found.",
        )
    data = json.loads(eval_file.read_text(encoding="utf-8"))
    cand = data.get("candidate_result", {})
    calib = cand.get("calibration")
    if calib is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Calibration diagnostics were not computed or are unavailable for this run.",
        )
    return calib


@router.get("/evaluations/{run_id}/diagnostics/subgroups")
def get_subgroup_diagnostics(
    run_id: str,
    repo: SQLiteRunRepository = Depends(get_repository),
) -> Dict[str, Any]:
    """Retrieve subgroup slice error rates and demographic disparity diagnostics."""
    art_dir = _get_run_artifact_dir(run_id, repo)
    eval_file = art_dir / "evaluation.json"
    if not eval_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="evaluation.json not found.",
        )
    data = json.loads(eval_file.read_text(encoding="utf-8"))
    cand = data.get("candidate_result", {})
    err_diag = cand.get("error_diagnostics", {})
    subgroups = err_diag.get("subgroup_diagnostics")
    if subgroups is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Subgroup slice diagnostics were not evaluated for this run.",
        )
    return {"subgroup_diagnostics": subgroups}


@router.get("/evaluations/{run_id}/acceptance", response_model=AcceptanceDecision)
def get_acceptance_decision(
    run_id: str,
    repo: SQLiteRunRepository = Depends(get_repository),
) -> AcceptanceDecision:
    """Retrieve the formal acceptance decision, rule evaluation checklist, and cryptographic decision signature."""
    art_dir = _get_run_artifact_dir(run_id, repo)
    dec_file = art_dir / "decision.json"
    if not dec_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No formal acceptance decision found for run '{run_id}'.",
        )
    try:
        data = json.loads(dec_file.read_text(encoding="utf-8"))
        return AcceptanceDecision(**data)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read decision artifact: {exc}",
        )


@router.get("/evaluations/{run_id}/report.html")
def get_html_report(
    run_id: str,
    repo: SQLiteRunRepository = Depends(get_repository),
) -> HTMLResponse:
    """Serve the self-contained offline HTML audit report."""
    art_dir = _get_run_artifact_dir(run_id, repo)
    report_file = art_dir / "report.html"
    if not report_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Offline HTML report has not been generated for run '{run_id}'.",
        )
    return HTMLResponse(
        content=report_file.read_text(encoding="utf-8"),
        status_code=status.HTTP_200_OK,
    )


@router.get("/evaluations/{run_id}/artifacts")
def list_run_artifacts(
    run_id: str,
    repo: SQLiteRunRepository = Depends(get_repository),
) -> Dict[str, Any]:
    """List all persisted artifact files with sizes and SHA-256 digests."""
    art_dir = _get_run_artifact_dir(run_id, repo)
    files = {}
    artifacts_list = []

    for f in sorted(art_dir.glob("**/*")):
        if f.is_file():
            rel = str(f.relative_to(art_dir).as_posix())
            sz = f.stat().st_size
            h = hashlib.sha256(f.read_bytes()).hexdigest()
            files[rel] = {"sha256": h, "size_bytes": sz}
            artifacts_list.append({
                "path": rel,
                "name": f.name,
                "size_bytes": sz,
                "sha256": h,
            })

    return {
        "run_id": run_id,
        "files": files,
        "artifacts": artifacts_list,
        "file_checksums": {k: v["sha256"] for k, v in files.items()},
    }


@router.get("/evaluations/{run_id}/artifacts/{file_path:path}")
def download_run_artifact(
    run_id: str,
    file_path: str,
    repo: SQLiteRunRepository = Depends(get_repository),
) -> FileResponse:
    """Safely retrieve or download an individual artifact file."""
    art_dir = _get_run_artifact_dir(run_id, repo)
    if ".." in file_path or "\x00" in file_path or file_path.startswith("/") or file_path.startswith("\\"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid path traversal.",
        )
    target = (art_dir / file_path).resolve()
    if not target.is_relative_to(art_dir.resolve()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Artifact access outside run directory forbidden.",
        )
    if not target.exists() or not target.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact file '{file_path}' not found.",
        )
    return FileResponse(path=str(target), filename=target.name)


# ============================================================================
# 4. Robustness & Stress Testing
# ============================================================================

@router.get("/evaluations/{run_id}/robustness", response_model=RobustnessEvaluationResult)
def get_robustness_results(
    run_id: str,
    repo: SQLiteRunRepository = Depends(get_repository),
) -> RobustnessEvaluationResult:
    """Retrieve previously recorded robustness stress testing curves for an evaluation run."""
    art_dir = _get_run_artifact_dir(run_id, repo)
    rob_file = art_dir / "robustness.json"
    if rob_file.exists():
        try:
            data = json.loads(rob_file.read_text(encoding="utf-8"))
            return RobustnessEvaluationResult.model_validate(data)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to read robustness artifact: {exc}",
            )

    eval_file = art_dir / "evaluation.json"
    if eval_file.exists():
        try:
            ev_data = json.loads(eval_file.read_text(encoding="utf-8"))
            cand = ev_data.get("candidate_result", {})
            rob = cand.get("robustness")
            if rob is not None:
                return RobustnessEvaluationResult.model_validate(rob)
        except Exception:
            pass

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Robustness diagnostics have not been evaluated for run '{run_id}'.",
    )

@router.post("/evaluations/{run_id}/robustness", response_model=RobustnessEvaluationResult)
def run_robustness_stress_test(
    run_id: str,
    req: RobustnessRunRequest,
    repo: SQLiteRunRepository = Depends(get_repository),
) -> RobustnessEvaluationResult:
    """Execute controlled perturbation stress testing on the retained candidate model pipeline."""
    art_dir = _get_run_artifact_dir(run_id, repo)
    record = repo.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")

    # 1. Load fitted model
    try:
        pipeline = load_fitted_model(art_dir, "candidate", trusted=True)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fitted model artifact unavailable: {exc}. Ensure model serialization was enabled.",
        )

    # 2. Re-create validation slice from original dataset
    src = record.metadata.get("source", "churn")
    target_col = record.metadata.get("target_column", "churn_label")
    try:
        df = _load_dataset_from_request(None, src if src in ("churn", "breast_cancer") else None)
    except Exception:
        # Fallback to churn dataset if source not found
        df = generate_synthetic_churn_dataset(n_samples=500, random_state=req.random_state)
        target_col = "churn_label"

    X, y, _ = validate_and_extract_dataset(df, target_column=target_col)

    # 3. Build perturbation config
    profile_type = req.profile.lower().strip()
    grid = req.severity_levels or [0.05, 0.10, 0.25, 0.50]

    if profile_type == "noise":
        cfg = create_noise_stress_profile(severity_levels=grid, random_state=req.random_state)
    elif profile_type == "missingness":
        cfg = create_missingness_stress_profile(severity_levels=grid, random_state=req.random_state)
    elif profile_type == "category_dropout":
        cfg = create_category_dropout_stress_profile(severity_levels=grid, random_state=req.random_state)
    elif profile_type == "compound":
        cfg = create_compound_stress_profile(severity_levels=grid, random_state=req.random_state)
    else:
        cfg = get_standard_stress_suite(random_state=req.random_state)

    # 4. Evaluate stress testing
    try:
        res = evaluate_stress_profile(
            estimator=pipeline,
            X_val=X,
            y_val=y,
            profile=cfg,
            metric_name="accuracy",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stress testing evaluation failed: {exc}",
        )

    # 5. Persist robustness curves in robustness.json without mutating evaluation.json
    try:
        rob_file = art_dir / "robustness.json"
        rob_file.write_text(json.dumps(res.model_dump(), indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning(f"Could not save robustness.json: {exc}")

    return res


# ============================================================================
# 5. Release Readiness, Certificates & Bundles
# ============================================================================

@router.post("/evaluations/{run_id}/release-gate", response_model=ReleaseGateResult)
def evaluate_run_release_gate(
    run_id: str,
    req: ReleaseGateEvaluateRequest,
    repo: SQLiteRunRepository = Depends(get_repository),
) -> ReleaseGateResult:
    """Evaluate release governance requirements and evidence integrity for an evaluation run."""
    art_dir = _get_run_artifact_dir(run_id, repo)
    policy = req.policy or ReleasePolicy(
        policy_id="default_strict_policy",
        version="1.0.0",
        target_environment="production",
        require_acceptance_pass=True,
        require_manifest_integrity=True,
        require_model_artifact=False,
    )
    try:
        gate_res = evaluate_release_gate(source=art_dir, policy=policy)
        # Save gate result to artifacts
        gate_file = art_dir / "release_gate.json"
        gate_file.write_text(json.dumps(gate_res.model_dump(), indent=2), encoding="utf-8")
        return gate_res
    except ReleaseGateError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Release gate evaluation error: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Release gate evaluation failed: {exc}",
        )


@router.post("/evaluations/{run_id}/certificate/issue", response_model=CertificateIssueResponse)
def issue_run_certificate(
    run_id: str,
    req: CertificateIssueRequest,
    repo: SQLiteRunRepository = Depends(get_repository),
) -> CertificateIssueResponse:
    """Cryptographically issue a signed Ed25519 release certificate authorizing model deployment."""
    art_dir = _get_run_artifact_dir(run_id, repo)

    # 1. First ensure release gate has been evaluated
    gate_file = art_dir / "release_gate.json"
    if not gate_file.exists():
        # Evaluate default gate
        gate_res = evaluate_release_gate(source=art_dir)
        gate_file.write_text(json.dumps(gate_res.model_dump(), indent=2), encoding="utf-8")
    else:
        gate_res = ReleaseGateResult(**json.loads(gate_file.read_text(encoding="utf-8")))

    # 2. Generate local signing keypair
    private_key, public_key, default_key_id = generate_release_keypair()
    pub_hex = serialize_public_key(public_key)
    active_key_id = req.key_id if req.key_id else default_key_id

    # 3. Issue certificate
    try:
        cert = issue_release_certificate(
            source=art_dir,
            private_key=private_key,
            key_id=active_key_id,
            release_policy=req.release_policy,
            output_dir=art_dir,
            overwrite=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to issue release certificate: {exc}",
        )

    # 4. Save certificate to run artifact directory
    cert_path = art_dir / "release_certificate.json"
    cert_path.write_text(json.dumps(cert.model_dump(), indent=2), encoding="utf-8")

    return CertificateIssueResponse(
        certificate=cert,
        public_key_hex=pub_hex,
        certificate_path=str(cert_path.as_posix()),
    )


@router.post("/verify-release", response_model=CertificateVerificationResult)
def verify_release(
    req: CertificateVerifyRequest,
    repo: SQLiteRunRepository = Depends(get_repository),
) -> CertificateVerificationResult:
    """Perform multi-tiered cryptographic verification on a release certificate."""
    cert_obj = None
    if req.certificate:
        cert_obj = ReleaseCertificate(**req.certificate)
    elif req.certificate_path:
        p = _resolve_confined_path(req.certificate_path)
        if not p.exists() or not p.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Certificate path '{req.certificate_path}' not found or is not a regular file.",
            )
        if p.suffix.lower() != ".json":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Certificate file '{req.certificate_path}' must have a .json extension.",
            )
        cert_obj = p
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'certificate' or 'certificate_path' must be provided.",
        )

    # 1. Parse client trusted keys
    parsed_trusted_keys = {}
    for kid, khex in (req.trusted_keys or {}).items():
        try:
            parsed_trusted_keys[kid] = deserialize_public_key(khex)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid public key hex for key '{kid}': {exc}",
            )

    # 2. Check server-authoritative trusted keys file if configured via environment
    env_keys_file = os.environ.get("MLFORGE_TRUSTED_KEYS_FILE")
    if env_keys_file:
        from mlforge.cli import _load_trusted_keys
        try:
            server_keys = _load_trusted_keys(trusted_keys_file=env_keys_file)
            if server_keys:
                parsed_trusted_keys = server_keys
        except Exception as exc:
            logger.warning(f"Failed to load MLFORGE_TRUSTED_KEYS_FILE: {exc}")

    # 3. Parse candidate keys (untrusted keys for mathematical signature verification)
    parsed_candidate_keys = {}
    for kid, khex in (req.candidate_keys or {}).items():
        try:
            parsed_candidate_keys[kid] = deserialize_public_key(khex)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid public key hex for candidate key '{kid}': {exc}",
            )

    # 4. Resolve evidence source path safely
    source_path = req.evidence_dir or req.bundle_path
    if source_path:
        source_p = _resolve_confined_path(source_path)
        if not source_p.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evidence source path '{source_path}' does not exist.",
            )
        if not (source_p.is_dir() or (source_p.is_file() and source_p.suffix.lower() == ".zip")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Evidence source path '{source_path}' must be a directory or .zip bundle.",
            )
        source_path = str(source_p)
    elif isinstance(cert_obj, ReleaseCertificate):
        # Auto-resolve local run artifact directory if available in repository
        try:
            run_rec = repo.get_run(cert_obj.run_id)
            if run_rec and run_rec.artifact_path and Path(run_rec.artifact_path).exists():
                source_path = str(Path(run_rec.artifact_path).resolve())
        except Exception:
            source_path = None

    try:
        result = verify_release_certificate(
            certificate=cert_obj,
            trusted_keys=parsed_trusted_keys,
            expected_source=source_path,
            candidate_keys=parsed_candidate_keys,
        )
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Release certificate verification failed: {exc}",
        )


@router.post("/evaluations/{run_id}/bundle", response_model=BundleCreateResponse)
def create_audit_bundle_endpoint(
    run_id: str,
    req: Optional[BundleCreateRequest] = None,
    repo: SQLiteRunRepository = Depends(get_repository),
) -> BundleCreateResponse:
    """Package run evidence into an offline, tamper-evident audit bundle archive."""
    art_dir = _get_run_artifact_dir(run_id, repo)
    out_bundle = art_dir / f"audit_bundle_{run_id}.zip"

    try:
        bundle_path = build_audit_bundle(
            run_dir=art_dir,
            output_bundle_path=out_bundle,
        )
        sz = bundle_path.stat().st_size
        h = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
        import zipfile
        with zipfile.ZipFile(bundle_path, "r") as zf:
            member_count = len(zf.namelist())

        return BundleCreateResponse(
            bundle_path=str(bundle_path.as_posix()),
            size_bytes=sz,
            checksum_sha256=h,
            member_count=member_count,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create audit bundle: {exc}",
        )


@router.get("/evaluations/{run_id}/bundle.zip")
def download_audit_bundle(
    run_id: str,
    repo: SQLiteRunRepository = Depends(get_repository),
) -> FileResponse:
    """Download the audit bundle ZIP archive."""
    art_dir = _get_run_artifact_dir(run_id, repo)
    out_bundle = art_dir / f"audit_bundle_{run_id}.zip"
    if not out_bundle.exists():
        # Build bundle on demand
        build_audit_bundle(
            run_dir=art_dir,
            output_bundle_path=out_bundle,
        )

    return FileResponse(
        path=str(out_bundle),
        filename=f"audit_bundle_{run_id}.zip",
        media_type="application/zip",
    )


@router.post("/verify-bundle", response_model=AuditBundleVerificationResult)
def verify_audit_bundle_endpoint(req: BundleVerifyRequest) -> AuditBundleVerificationResult:
    """Verify cryptographic integrity, manifest alignment, and archive safety of an audit bundle."""
    p = _resolve_confined_path(req.bundle_path)
    if not p.exists() or not p.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bundle file '{req.bundle_path}' not found or is not a regular file.",
        )
    if p.suffix.lower() != ".zip":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Bundle file '{req.bundle_path}' must have a .zip extension.",
        )
    try:
        res = verify_audit_bundle(p)
        return res
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Bundle verification error: {exc}",
        )
