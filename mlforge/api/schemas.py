"""Pydantic request and response schemas for the MLForge REST API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from mlforge.core.contracts import (
    AcceptanceDecision,
    AcceptancePolicy,
    AuditBundleVerificationResult,
    CertificateVerificationResult,
    ComparisonReport,
    DatasetMetadata,
    EvaluationResult,
    LifecycleTransitionEvent,
    ModelEvaluationResult,
    ReleaseCertificate,
    ReleaseGateResult,
    ReleasePolicy,
    RobustnessEvaluationResult,
    RunRecord,
    RunState,
    RunSummaryMetric,
)


class DatasetInspectRequest(BaseModel):
    """Request to inspect a local tabular dataset or a sample preset."""
    file_path: Optional[str] = Field(None, description="Local path to CSV dataset")
    sample_name: Optional[str] = Field(None, description="Preset benchmark dataset name: 'churn' or 'breast_cancer'")
    target_column: str = Field(..., description="Target column name to extract and validate")
    k_folds: Optional[int] = Field(5, ge=2, description="Target fold count to validate class support")


class SampleDatasetInfo(BaseModel):
    """Metadata describing a pre-packaged benchmark dataset."""
    id: str
    name: str
    description: str
    num_rows: int
    num_features: int
    default_target: str
    default_subgroup: Optional[str] = None


class CreateEvaluationRequest(BaseModel):
    """Request to submit and execute a new model evaluation run."""
    file_path: Optional[str] = Field(None, description="Local path to CSV dataset")
    sample_name: Optional[str] = Field(None, description="Sample dataset preset: 'churn' or 'breast_cancer'")
    target_column: str = Field(..., description="Target classification column name")
    candidate_model_type: str = Field(default="logistic_regression", description="Model architecture type")
    candidate_hyperparameters: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 200},
        description="Candidate model hyperparameters",
    )
    num_folds: int = Field(default=5, ge=2, le=20, description="Cross-validation fold count")
    random_state: int = Field(default=42, description="Deterministic random seed")
    subgroup_columns: Optional[List[str]] = Field(None, description="Columns to partition for subgroup slice diagnostics")
    policy_id: Optional[str] = Field(default="standard_classification_v1", description="Acceptance policy template to apply")
    custom_policy: Optional[AcceptancePolicy] = Field(None, description="Custom acceptance policy definition")
    run_id: Optional[str] = Field(None, description="Explicit run ID; if omitted, generated automatically")
    run_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom user metadata tags")


class EvaluationSummaryItem(BaseModel):
    """Compact summary of a persisted evaluation run for list views."""
    run_id: str
    state: RunState
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    updated_at: str
    dataset_checksum: Optional[str] = None
    candidate_model_type: Optional[str] = None
    baseline_model_type: Optional[str] = None
    acceptance_outcome: Optional[str] = None
    artifact_path: Optional[str] = None
    error_category: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    summary_metrics: Dict[str, Optional[float]] = Field(default_factory=dict)
    metric_details: List[RunSummaryMetric] = Field(default_factory=list)


class EvaluationListResponse(BaseModel):
    """Paginated list of historical evaluation runs."""
    items: List[EvaluationSummaryItem]
    total: int
    limit: int
    offset: int


class EvaluationStatusResponse(BaseModel):
    """Lightweight run execution status response for polling."""
    run_id: str
    state: RunState
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


class EvaluationDetailResponse(BaseModel):
    """Detailed metadata for a single evaluation run."""
    record: RunRecord
    lifecycle_history: List[LifecycleTransitionEvent]
    metrics: List[RunSummaryMetric]
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    has_evaluation: bool = False
    has_decision: bool = False
    has_report: bool = False
    has_certificate: bool = False
    has_bundle: bool = False


class PolicyTemplateInfo(BaseModel):
    """Information regarding a versioned acceptance policy template."""
    policy_id: str
    version: str
    name: str
    description: Optional[str] = None
    rules_count: int
    policy: AcceptancePolicy


class RobustnessRunRequest(BaseModel):
    """Request to execute a stress test on a completed evaluation's retained model."""
    profile: str = Field(default="standard_suite", description="Stress profile: 'noise', 'missingness', 'category_dropout', 'compound', or 'standard_suite'")
    severity_levels: Optional[List[float]] = Field(None, description="Custom non-decreasing severity levels")
    random_state: int = Field(default=42, description="Random seed")


class ReleaseGateEvaluateRequest(BaseModel):
    """Request to evaluate release readiness criteria for an evaluation run."""
    policy: Optional[ReleasePolicy] = Field(None, description="Optional custom release policy; default strict policy if omitted")


class CertificateIssueRequest(BaseModel):
    """Request to cryptographically issue an Ed25519 release certificate."""
    key_id: str = Field(default="mlforge-local-signer-01", description="Identifier for the signing key")
    target_environment: str = Field(default="production", description="Target deployment tier")
    release_policy: Optional[ReleasePolicy] = Field(None, description="Optional release policy overrides")


class CertificateIssueResponse(BaseModel):
    """Response containing the issued release certificate and public key."""
    certificate: ReleaseCertificate
    public_key_hex: str
    certificate_path: str


class CertificateVerifyRequest(BaseModel):
    """Request to verify an Ed25519 release certificate."""
    certificate: Optional[Dict[str, Any]] = Field(None, description="Certificate JSON object")
    certificate_path: Optional[str] = Field(None, description="Filesystem path to certificate JSON")
    trusted_keys: Dict[str, str] = Field(default_factory=dict, description="Mapping of authorized key_id to 64-hex Ed25519 public key")
    candidate_keys: Dict[str, str] = Field(default_factory=dict, description="Mapping of untrusted candidate key_id to 64-hex public key for signature verification without conferring trust")
    evidence_dir: Optional[str] = Field(None, description="Optional path to directory containing evidence artifacts")
    bundle_path: Optional[str] = Field(None, description="Optional path to ZIP bundle containing evidence artifacts")
    require_passed_gate: bool = Field(default=True, description="Whether to require overall_status == PASSED for authorization")


class BundleCreateRequest(BaseModel):
    """Request to package run artifacts into a tamper-evident audit bundle."""
    include_certificate: bool = Field(default=True, description="Whether to include release_certificate.json if present")


class BundleCreateResponse(BaseModel):
    """Response after creating an audit bundle."""
    bundle_path: str
    size_bytes: int
    checksum_sha256: str
    member_count: int
    bundle_size_bytes: Optional[int] = None
    bundle_sha256: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        if self.bundle_size_bytes is None:
            self.bundle_size_bytes = self.size_bytes
        if self.bundle_sha256 is None:
            self.bundle_sha256 = self.checksum_sha256


class BundleVerifyRequest(BaseModel):
    """Request to verify an audit bundle archive."""
    bundle_path: str = Field(..., description="Filesystem path to the audit bundle ZIP archive")


class CancelEvaluationRequest(BaseModel):
    """Request to cancel an active evaluation."""
    reason: Optional[str] = Field("Cancelled by user via UI", description="Operational cancellation reason")
