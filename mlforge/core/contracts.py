"""MLForge Core Data Contracts.

Typed, immutable data structures representing dataset metadata, fold-level metrics,
reproducibility configurations, and aggregated evaluation results.
"""

from enum import Enum
import math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetMetadata(BaseModel):
    """Metadata and distribution statistics for an ingested tabular dataset."""
    num_rows: int = Field(..., ge=1, description="Total sample count")
    num_features: int = Field(..., ge=1, description="Total feature count excluding target")
    feature_names: List[str] = Field(..., description="Ordered list of feature column names")
    feature_types: Dict[str, str] = Field(..., description="Inferred dtype per feature column")
    numeric_feature_count: int = Field(..., ge=0)
    categorical_feature_count: int = Field(..., ge=0)
    target_name: str = Field(..., description="Name of the target column")
    target_type: str = Field(..., description="Data type of the target column")
    class_distribution: Dict[str, int] = Field(..., description="Observed sample counts per class")
    class_labels: List[str] = Field(..., description="Unique sorted class labels")
    imbalance_ratio: float = Field(..., ge=1.0, description="Dominant class count / minority class count")
    missing_value_counts: Dict[str, int] = Field(..., description="Count of missing values per column")
    constant_columns: List[str] = Field(default_factory=list, description="Columns with only 1 unique value")
    checksum_sha256: str = Field(..., description="Cryptographic SHA256 hash of dataset values")
    warnings: List[str] = Field(default_factory=list, description="Actionable advisory warnings")


class MetricValue(BaseModel):
    """Encapsulates a single metric score or an explicit undefined condition."""
    value: Optional[float] = Field(None, description="Computed scalar value; None if undefined")
    is_undefined: bool = Field(default=False, description="True if mathematically or logically undefined")
    undefined_reason: Optional[str] = Field(None, description="Diagnostic explanation if undefined")


class CalibrationBin(BaseModel):
    """Reliability bin summary containing sample counts, predicted confidence, and observed positive rate."""
    bin_index: int = Field(..., ge=0, description="0-indexed bin number")
    lower_bound: float = Field(..., ge=0.0, le=1.0, description="Inclusive or left bin boundary")
    upper_bound: float = Field(..., ge=0.0, le=1.0, description="Right bin boundary")
    sample_count: int = Field(..., ge=0, description="Total samples falling within bin")
    mean_predicted_prob: Optional[float] = Field(
        None, description="Average predicted probability of samples in bin; None if empty"
    )
    observed_positive_rate: Optional[float] = Field(
        None, description="Proportion of actual positive samples in bin; None if empty"
    )
    is_empty: bool = Field(default=False, description="True if no samples fall into bin")


class CalibrationCurveResult(BaseModel):
    """Numerical data and summary metrics for probability calibration and reliability diagrams."""
    bins: List[CalibrationBin] = Field(..., description="Ordered list of reliability diagram bins")
    expected_calibration_error: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Sample-weighted Expected Calibration Error (ECE)"
    )
    max_calibration_error: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Maximum Calibration Error across non-empty bins (MCE)"
    )
    brier_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Mean squared difference between predicted probabilities and binary outcomes"
    )
    log_loss: Optional[float] = Field(
        None, ge=0.0, description="Cross-entropy loss for predicted probabilities"
    )
    bin_strategy: str = Field(default="uniform", description="Binning strategy: 'uniform' or 'quantile'")
    num_bins: int = Field(default=10, ge=2, description="Target number of bins")
    total_samples: int = Field(..., ge=0, description="Total evaluated observations")
    positive_class_label: str = Field(..., description="Explicitly identified positive class label")
    is_usable: bool = Field(default=True, description="True if calibration diagnostics could be reliably calculated")
    unusable_reason: Optional[str] = Field(None, description="Reason calibration metrics are unavailable")
    warnings: List[str] = Field(default_factory=list, description="Advisory or sample-support warnings")


class ThresholdMetrics(BaseModel):
    """Classification metrics and contingency counts at a single decision threshold."""
    threshold: float = Field(..., ge=0.0, le=1.0, description="Classification decision cutoff")
    evaluated_samples: int = Field(..., ge=0, description="Total evaluated observations")
    true_positives: int = Field(..., ge=0)
    false_positives: int = Field(..., ge=0)
    true_negatives: int = Field(..., ge=0)
    false_negatives: int = Field(..., ge=0)
    precision: Optional[float] = Field(None, description="Precision / Positive Predictive Value; None if undefined")
    recall: Optional[float] = Field(None, description="Recall / Sensitivity; None if undefined")
    specificity: Optional[float] = Field(None, description="Specificity / True Negative Rate; None if undefined")
    f1_score: Optional[float] = Field(None, description="Harmonic mean of precision and recall; None if undefined")
    balanced_accuracy: Optional[float] = Field(None, description="Mean of sensitivity and specificity; None if undefined")
    predicted_positive_rate: float = Field(..., ge=0.0, le=1.0, description="(TP + FP) / total_samples")
    precision_undefined_reason: Optional[str] = Field(None)
    recall_undefined_reason: Optional[str] = Field(None)
    specificity_undefined_reason: Optional[str] = Field(None)
    f1_undefined_reason: Optional[str] = Field(None)
    balanced_accuracy_undefined_reason: Optional[str] = Field(None)


class ThresholdAnalysisResult(BaseModel):
    """Multi-threshold classification analysis evaluating trade-offs across decision cutoffs."""
    positive_class_label: str = Field(..., description="Target class considered positive")
    evaluated_samples: int = Field(..., ge=0, description="Total evaluated observations")
    threshold_metrics: List[ThresholdMetrics] = Field(..., description="Evaluated threshold points")
    warnings: List[str] = Field(default_factory=list, description="Threshold evaluation notices")


class FoldErrorDiagnostic(BaseModel):
    """Error counts and rates for a single cross-validation fold."""
    fold_index: int = Field(..., ge=0)
    total_samples: int = Field(..., ge=1)
    error_count: int = Field(..., ge=0)
    error_rate: float = Field(..., ge=0.0, le=1.0)
    false_positives: Optional[int] = Field(None, ge=0)
    false_negatives: Optional[int] = Field(None, ge=0)


class SubgroupMetricDiagnostic(BaseModel):
    """Descriptive performance breakdown for a single category or subgroup within a metadata feature."""
    subgroup_column: str = Field(..., description="Name of the metadata column partitioned on")
    subgroup_value: str = Field(..., description="Category value defining this subgroup")
    sample_count: int = Field(..., ge=0, description="Samples belonging to this subgroup")
    error_count: int = Field(..., ge=0, description="Misclassified samples in subgroup")
    error_rate: float = Field(..., ge=0.0, le=1.0, description="error_count / sample_count")
    accuracy: Optional[float] = Field(None, description="Subgroup accuracy; None if insufficient support")
    precision: Optional[float] = Field(None, description="Subgroup precision if binary and supported")
    recall: Optional[float] = Field(None, description="Subgroup recall if binary and supported")
    is_usable: bool = Field(default=True, description="True if sample size meets minimum support threshold")
    unusable_reason: Optional[str] = Field(None, description="Reason if sample size is insufficient or single-class")


class ErrorDiagnosticsResult(BaseModel):
    """Comprehensive error diagnostic summary including fold-level and subgroup-level failure analysis."""
    total_samples: int = Field(..., ge=0)
    total_errors: int = Field(..., ge=0)
    overall_error_rate: float = Field(..., ge=0.0, le=1.0)
    fold_diagnostics: List[FoldErrorDiagnostic] = Field(default_factory=list)
    subgroup_diagnostics: Optional[Dict[str, List[SubgroupMetricDiagnostic]]] = Field(
        None, description="Mapping of metadata column name to list of subgroup diagnostics"
    )
    warnings: List[str] = Field(default_factory=list)


class FoldResult(BaseModel):
    """Evaluation result for a single cross-validation fold."""
    fold_index: int = Field(..., ge=0, description="0-indexed fold number")
    train_indices_count: int = Field(..., ge=1)
    val_indices_count: int = Field(..., ge=1)
    metrics: Dict[str, MetricValue] = Field(..., description="Computed metrics mapped to MetricValue")
    confusion_matrix: List[List[int]] = Field(..., description="2D contingency matrix [true_idx][pred_idx]")
    class_labels: List[str] = Field(..., description="Class ordering corresponding to confusion matrix")
    fit_duration_seconds: float = Field(..., ge=0.0)
    predict_duration_seconds: float = Field(..., ge=0.0)
    partition_fingerprint: Optional[str] = Field(
        None, description="Deterministic SHA256 fingerprint of validation split indices"
    )
    calibration: Optional[CalibrationCurveResult] = Field(
        None, description="Fold-level probability calibration results if probabilities available"
    )
    threshold_analysis: Optional[ThresholdAnalysisResult] = Field(
        None, description="Fold-level threshold analysis results if binary classification"
    )
    error_diagnostic: Optional[FoldErrorDiagnostic] = Field(
        None, description="Fold-level error diagnostic breakdown"
    )
    val_indices: Optional[List[int]] = Field(
        None, description="Original row indices in validation split for out-of-fold aggregation"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)
    fitted_pipeline: Optional[Any] = Field(
        None, exclude=True, description="Fitted pipeline instance on this fold partition"
    )


class MetricSummary(BaseModel):
    """Aggregate statistics across cross-validation folds for a specific metric."""
    metric_name: str
    mean: Optional[float] = Field(None, description="Mean score across valid folds")
    std: Optional[float] = Field(None, description="Sample standard deviation across valid folds")
    median: Optional[float] = Field(None, description="Median score across valid folds")
    iqr: Optional[float] = Field(None, description="Interquartile range across valid folds")
    min_value: Optional[float] = Field(None, description="Minimum score across valid folds")
    max_value: Optional[float] = Field(None, description="Maximum score across valid folds")
    confidence_interval_95: Optional[List[float]] = Field(
        None, min_length=2, max_length=2, description="[lower_ci, upper_ci] bounds"
    )
    valid_fold_count: int = Field(default=0, ge=0)
    undefined_fold_count: int = Field(default=0, ge=0)
    undefined_reasons: List[str] = Field(default_factory=list)


class PerturbationType(str, Enum):
    """Categorization of feature perturbation and corruption methods."""
    GAUSSIAN_NOISE = "gaussian_noise"
    MISSINGNESS_INJECTION = "missingness_injection"
    CATEGORY_DROPOUT = "category_dropout"
    COMPOUND = "compound"


class PerturbationConfig(BaseModel):
    """Deterministic configuration for a controlled feature corruption experiment."""
    perturbation_type: PerturbationType
    severity_levels: List[float] = Field(
        default_factory=lambda: [0.05, 0.10, 0.25],
        description="Severity levels: noise scale multiplier or missingness/dropout fraction in [0, 1]",
    )
    target_columns: Optional[List[str]] = Field(
        None, description="Subset of feature columns to perturb; if None, all eligible features"
    )
    random_state: int = Field(default=42, description="Random seed for deterministic corruption")
    compound_steps: Optional[List["PerturbationConfig"]] = Field(
        None, description="Ordered list of component perturbation configs if perturbation_type is COMPOUND"
    )

    @model_validator(mode="after")
    def validate_configuration(self) -> "PerturbationConfig":
        if not self.severity_levels:
            raise ValueError("severity_levels must contain at least one severity level.")

        if self.perturbation_type == PerturbationType.COMPOUND and self.compound_steps is not None:
            if len(self.compound_steps) == 0:
                raise ValueError("compound_steps must not be empty if explicitly provided.")
            for step in self.compound_steps:
                if step.perturbation_type == PerturbationType.COMPOUND:
                    raise ValueError("Nested compound steps are not permitted.")

        seen = set()
        for idx, sev in enumerate(self.severity_levels):
            if not isinstance(sev, (int, float)) or math.isnan(sev) or math.isinf(sev):
                raise ValueError(f"Severity levels must be finite numbers; received {sev}.")
            f_sev = float(sev)
            if self.perturbation_type == PerturbationType.GAUSSIAN_NOISE:
                if f_sev < 0.0:
                    raise ValueError(f"Severity for GAUSSIAN_NOISE must be >= 0.0; received {sev}.")
            elif self.perturbation_type in (
                PerturbationType.MISSINGNESS_INJECTION,
                PerturbationType.CATEGORY_DROPOUT,
                PerturbationType.COMPOUND,
            ):
                if f_sev < 0.0 or f_sev > 1.0:
                    raise ValueError(
                        f"Severity for {self.perturbation_type.value} must be in [0.0, 1.0]; received {sev}."
                    )
            if idx > 0 and f_sev < float(self.severity_levels[idx - 1]):
                raise ValueError(
                    f"severity_levels must be in non-decreasing sorted order; received {self.severity_levels}."
                )
            if f_sev in seen:
                raise ValueError(f"Duplicate severity level found in severity_levels: {f_sev}.")
            seen.add(f_sev)
        return self


class RobustnessPerturbationResult(BaseModel):
    """Point evaluation of model performance under a specific perturbation at a single severity."""
    perturbation_type: PerturbationType
    severity_level: float = Field(..., ge=0.0, description="Severity parameter, e.g. 0.05, 0.10, 0.25")
    metric_name: str = Field(default="accuracy")
    clean_score: Optional[float] = Field(None, description="Baseline unperturbed performance score")
    perturbed_score: Optional[float] = Field(None, description="Performance score under perturbation")
    retained_ratio: Optional[float] = Field(
        None, description="Retained performance ratio: perturbed_score / clean_score"
    )
    performance_drop_percentage: Optional[float] = Field(
        None, description="Relative performance drop percentage: ((clean - perturbed) / clean) * 100"
    )
    is_usable: bool = Field(default=True, description="True if perturbation evaluation completed validly")
    unusable_reason: Optional[str] = Field(None, description="Diagnostic explanation if evaluation failed or undefined")
    warnings: List[str] = Field(default_factory=list, description="Advisory notices during perturbation")


class RobustnessCurve(BaseModel):
    """Performance degradation trajectory across incremental perturbation severities."""
    perturbation_type: PerturbationType
    metric_name: str
    clean_score: Optional[float] = None
    points: List[RobustnessPerturbationResult] = Field(default_factory=list)
    area_under_degradation_curve: Optional[float] = Field(
        None, description="Normalized area under retention curve across evaluated severities"
    )
    warnings: List[str] = Field(default_factory=list)

    @property
    def aurdc(self) -> Optional[float]:
        """Alias for area_under_degradation_curve (Area Under Robustness Degradation Curve)."""
        return self.area_under_degradation_curve


class RobustnessEvaluationResult(BaseModel):
    """Aggregated robustness assessment across one or more perturbation dimensions."""
    curves: List[RobustnessCurve] = Field(default_factory=list)
    summary_retention: Dict[str, float] = Field(
        default_factory=dict, description="Summary retention score per perturbation type at maximum severity"
    )
    is_usable: bool = Field(default=True)
    warnings: List[str] = Field(default_factory=list)


class ModelEvaluationResult(BaseModel):
    """Full evaluation record for an evaluated model (baseline or candidate)."""
    model_name: str
    model_type: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    fold_results: List[FoldResult]
    metric_summaries: Dict[str, MetricSummary]
    aggregate_confusion_matrix: List[List[int]]
    calibration: Optional[CalibrationCurveResult] = Field(
        None, description="Aggregated out-of-fold calibration diagnostics"
    )
    threshold_analysis: Optional[ThresholdAnalysisResult] = Field(
        None, description="Aggregated out-of-fold threshold sweep diagnostics"
    )
    error_diagnostics: Optional[ErrorDiagnosticsResult] = Field(
        None, description="Aggregated out-of-fold error diagnostics and subgroup breakdown"
    )
    robustness: Optional[RobustnessEvaluationResult] = Field(
        None, description="Controlled stress testing and perturbation evaluation"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)
    fitted_pipeline: Optional[Any] = Field(
        None, exclude=True, description="Representative or final fitted pipeline from cross-validation run"
    )
    model_provenance: Optional[str] = Field(
        None, description="Training partition provenance of the retained fitted estimator, e.g. 'final_cv_fold'"
    )
    fitted_fold_index: Optional[int] = Field(
        None, description="0-indexed cross-validation fold partition from which fitted_pipeline was retained"
    )


class ReproducibilityMetadata(BaseModel):
    """Audit trail recording environment and configuration for reproducibility."""
    random_seed: int
    split_strategy: str
    num_folds: int
    package_versions: Dict[str, str]
    platform_system: str
    python_version: str


class MetricDirection(str, Enum):
    """Directional interpretation of raw difference for a metric."""
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    NOT_APPLICABLE = "not_applicable"


class TestStatus(str, Enum):
    """Execution status of a statistical hypothesis test."""
    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"


# Avoid Pytest collecting TestStatus as a test suite
TestStatus.__test__ = False


class PairedMetricDifference(BaseModel):
    """Raw paired differences between candidate and baseline fold scores."""
    metric_name: str
    direction: MetricDirection
    raw_deltas: List[Optional[float]] = Field(
        ...,
        description="Per-fold difference: candidate - baseline. None if either fold metric was undefined.",
    )
    mean_delta: Optional[float] = Field(None, description="Sample mean of valid paired differences")
    std_delta: Optional[float] = Field(None, description="Sample standard deviation of valid paired differences")
    expected_folds: int = Field(..., ge=1, description="Total expected cross-validation folds")
    paired_folds: int = Field(..., ge=0, description="Number of aligned folds compared")
    usable_folds: int = Field(..., ge=0, description="Folds where both models had valid scalar values")
    is_usable: bool = Field(..., description="True if usable_folds == expected_folds and usable_folds >= 2")
    unusable_reason: Optional[str] = Field(None, description="Explanation if paired difference is unusable")


class StatisticalTestResult(BaseModel):
    """Machine-readable statistical comparison outcome."""
    test_name: str = Field(..., description="Statistical test method identifier")
    status: TestStatus = Field(..., description="Execution status of the statistical test")
    statistic: Optional[float] = Field(None, description="Calculated test statistic; None if unavailable")
    p_value: Optional[float] = Field(None, description="Calculated two-sided p-value; None if unavailable")
    degrees_of_freedom: Optional[float] = Field(None, description="Degrees of freedom if applicable")
    alpha: float = Field(default=0.05, description="Significance threshold for hypothesis testing")
    statistically_significant: Optional[bool] = Field(
        None, description="True if p_value < alpha; None if test is unavailable"
    )
    alternative_hypothesis: str = Field(default="two-sided")
    assumptions_and_limitations: List[str] = Field(
        default_factory=list, description="Method assumptions and cross-validation caveats"
    )
    warnings: List[str] = Field(default_factory=list, description="Methodological or numerical warnings")
    unavailable_reason: Optional[str] = Field(None, description="Reason test could not be computed")


class MetricComparisonResult(BaseModel):
    """Comprehensive comparison for a specific metric between candidate and baseline."""
    metric_name: str
    direction: MetricDirection
    paired_differences: PairedMetricDifference
    corrected_resampled_t_test: Optional[StatisticalTestResult] = None
    wilcoxon_signed_rank_test: Optional[StatisticalTestResult] = None


class ComparisonReport(BaseModel):
    """Root contract for paired model comparison across all compatible metrics."""
    candidate_model_name: str
    baseline_model_name: str
    dataset_checksum: str
    num_folds: int
    metric_comparisons: Dict[str, MetricComparisonResult]
    warnings: List[str] = Field(default_factory=list)


class AcceptanceOutcome(str, Enum):
    """Three-state evaluation outcome for an acceptance decision."""
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


# Avoid Pytest collecting AcceptanceOutcome as a test suite
AcceptanceOutcome.__test__ = False


class CriterionSeverity(str, Enum):
    """Enforcement severity of an acceptance rule."""
    BLOCKER = "blocker"
    ADVISORY = "advisory"


class CriterionOutcome(str, Enum):
    """Outcome for a single evaluated criterion."""
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    INCONCLUSIVE = "inconclusive"
    SKIPPED = "skipped"


class RuleOperator(str, Enum):
    """Comparison operator applied to evaluate criteria."""
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN_OR_EQUAL = "<="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    EQUALS = "=="
    NOT_EQUALS = "!="
    STATISTICALLY_GREATER_THAN = "statistically_greater_than"
    STATISTICALLY_LESS_THAN = "statistically_less_than"
    STATISTICALLY_SIGNIFICANT = "statistically_significant"
    CI_DISJOINT_SUPERIOR = "ci_disjoint_superior"


class AcceptanceRule(BaseModel):
    """Specification of a single acceptance threshold or validation requirement."""
    rule_id: str = Field(..., description="Unique programmatic identifier for the criterion")
    metric: str = Field(..., description="Target metric, attribute, or property evaluated")
    operator: RuleOperator = Field(..., description="Comparison operation")
    threshold: Optional[float] = Field(None, description="Numerical reference threshold. For statistical comparisons, specifies the minimum required effect size unless min_effect_size is provided")
    alpha: Optional[float] = Field(None, description="Statistical significance alpha threshold (defaults to test alpha or 0.05)")
    min_effect_size: Optional[float] = Field(None, description="Explicit minimum required effect size (mean difference). If specified, overrides threshold for effect size.")
    severity: CriterionSeverity = Field(
        default=CriterionSeverity.BLOCKER,
        description="'blocker' determines Pass/Fail/Inconclusive; 'advisory' emits warnings only"
    )
    evidence_source: str = Field(
        default="candidate_metric",
        description="Source of verification data: 'candidate_metric', 'baseline_contrast', 'calibration', 'statistical_test', 'data_quality', 'robustness'"
    )
    description: Optional[str] = Field(None, description="Human-readable rationale for this requirement")


class AcceptancePolicy(BaseModel):
    """Versioned collection of criteria defining the conditions for model acceptance."""
    policy_id: str = Field(..., description="Unique identifier of this acceptance policy")
    version: str = Field(default="1.0.0", description="SemVer policy version string")
    name: str = Field(..., description="Descriptive name of the policy")
    description: Optional[str] = Field(None, description="Context and intended deployment profile")
    fail_precedence: bool = Field(
        default=True,
        description="If True, any conclusive blocker violation yields FAIL even if another blocker is INCONCLUSIVE"
    )
    rules: List[AcceptanceRule] = Field(default_factory=list, description="Ordered criteria to evaluate")


class CriterionEvaluation(BaseModel):
    """Machine-readable assessment of an individual acceptance criterion."""
    rule_id: str
    metric: str
    severity: CriterionSeverity
    operator: RuleOperator
    threshold_value: Optional[float] = None
    observed_value: Optional[float] = None
    outcome: CriterionOutcome
    reason_code: str = Field(..., description="Machine-readable diagnostic code")
    message: str = Field(..., description="Human-readable explanation of criterion evaluation")
    evidence_source: str


class AcceptanceDecision(BaseModel):
    """Deterministic, auditable decision record for an evaluated model against an acceptance policy."""
    decision_id: str = Field(..., description="Unique identifier for this decision event")
    evaluation_id: Optional[str] = Field(None, description="Associated evaluation run identifier")
    policy_id: str = Field(..., description="Policy identifier used for evaluation")
    policy_version: str = Field(..., description="Version of the evaluated policy")
    outcome: AcceptanceOutcome = Field(..., description="Final decision verdict: PASS, FAIL, or INCONCLUSIVE")
    criteria_evaluations: List[CriterionEvaluation] = Field(
        ..., description="Granular assessment of every evaluated policy rule"
    )
    summary_rationale: str = Field(..., description="Comprehensive explanation of the final decision")
    evaluated_at: str = Field(..., description="ISO 8601 UTC timestamp of decision execution")
    passed_rules_count: int = Field(..., ge=0)
    failed_rules_count: int = Field(..., ge=0)
    inconclusive_rules_count: int = Field(..., ge=0)
    advisory_warnings: List[str] = Field(default_factory=list, description="Advisory notices that did not block decision")
    hash_signature: Optional[str] = Field(
        None, description="SHA256 signature over canonical criterion evaluations and decision metadata"
    )


class ArtifactManifest(BaseModel):
    """Metadata manifest indexing all serialized artifact files for a run."""
    schema_version: str = Field(default="1.0.0", description="Schema version of artifact serialization")
    run_id: str = Field(..., description="Unique run identifier")
    created_at: str = Field(..., description="ISO 8601 UTC timestamp of serialization")
    dataset_checksum: str = Field(..., description="SHA256 hash of dataset")
    split_strategy: str = Field(..., description="CV split strategy")
    num_folds: int = Field(..., ge=1)
    candidate_model_type: str = Field(..., description="Candidate model architecture")
    baseline_model_type: str = Field(..., description="Baseline model architecture")
    policy_id: Optional[str] = None
    policy_version: Optional[str] = None
    decision_outcome: Optional[AcceptanceOutcome] = None
    software_versions: Dict[str, str] = Field(default_factory=dict)
    file_checksums: Dict[str, str] = Field(
        default_factory=dict, description="Relative filename to SHA256 content checksum"
    )
    has_robustness: bool = Field(
        default=False,
        description="True if candidate or baseline robustness evaluation curves are included in evaluation.json"
    )
    has_models: bool = Field(
        default=False,
        description="True if fitted model pipelines (.joblib) are serialized under models/"
    )
    model_provenance: Optional[str] = Field(
        None, description="Training partition provenance of serialized estimators, e.g. 'final_cv_fold'"
    )
    candidate_model_fold_index: Optional[int] = Field(
        None, description="0-indexed fold from which the candidate estimator was retained"
    )
    baseline_model_fold_index: Optional[int] = Field(
        None, description="0-indexed fold from which the baseline estimator was retained"
    )


class ArtifactReference(BaseModel):
    """Reference metadata describing an individual artifact within a run or bundle."""
    path: str = Field(..., description="Relative POSIX path of the artifact file")
    sha256: str = Field(..., description="SHA-256 hex digest of file contents")
    size_bytes: int = Field(..., ge=0, description="Size of file in bytes")
    artifact_type: str = Field(
        default="evidence",
        description="Type of artifact: e.g. evidence, model, manifest, report",
    )


class AuditBundleFileEntry(BaseModel):
    """Inventory entry for a packaged member file in an audit bundle archive."""
    path: str = Field(..., description="Normalized relative POSIX path of the file inside the bundle")
    sha256: str = Field(..., description="SHA-256 hex digest of file contents")
    size_bytes: int = Field(..., ge=0, description="Size of file in bytes")
    artifact_type: str = Field(
        default="evidence",
        description="Type of artifact: e.g. evidence, model, manifest, report",
    )


class AuditBundleManifest(BaseModel):
    """Manifest describing the contents and integrity metadata of an audit bundle."""
    schema_version: str = Field(default="1.0.0", description="Schema version of the bundle manifest")
    bundle_id: str = Field(..., description="Unique bundle identifier, typically matching or derived from run_id")
    run_id: str = Field(..., description="Evaluated run identifier")
    created_at: str = Field(..., description="ISO 8601 UTC timestamp of bundle creation")
    files: List[AuditBundleFileEntry] = Field(
        default_factory=list, description="Inventory of packaged files with relative paths, checksums, and sizes"
    )
    dataset_checksum: Optional[str] = Field(None, description="SHA-256 checksum of source dataset")
    split_strategy: Optional[str] = None
    num_folds: Optional[int] = None
    candidate_model_type: Optional[str] = None
    baseline_model_type: Optional[str] = None
    policy_id: Optional[str] = None
    policy_version: Optional[str] = None
    decision_outcome: Optional[AcceptanceOutcome] = None
    has_models: bool = Field(default=False, description="True if fitted model artifacts (.joblib) are included")
    has_robustness: bool = Field(default=False, description="True if robustness stress curves are included")
    model_provenance: Optional[str] = Field(
        None, description="Training partition provenance of serialized estimators, e.g. 'final_cv_fold'"
    )
    candidate_model_fold_index: Optional[int] = Field(
        None, description="0-indexed fold from which the candidate estimator was retained"
    )
    baseline_model_fold_index: Optional[int] = Field(
        None, description="0-indexed fold from which the baseline estimator was retained"
    )
    software_versions: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AuditBundleVerificationResult(BaseModel):
    """Structured report returned by audit bundle integrity verification."""
    is_valid: bool = Field(..., description="True if all archive entries match manifest checksums and pass security checks")
    bundle_id: str = Field(default="", description="Bundle identifier from manifest")
    run_id: str = Field(default="", description="Run identifier from manifest")
    verified_files: List[str] = Field(default_factory=list, description="List of normalized relative paths verified")
    errors: List[str] = Field(default_factory=list, description="Verification failure errors or tampering notices")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings or advisory notices")
    manifest: Optional[AuditBundleManifest] = Field(None, description="Parsed bundle manifest if valid")


class EvaluationResult(BaseModel):
    """Root contract returned by the classification evaluation engine."""
    dataset_metadata: DatasetMetadata
    reproducibility: ReproducibilityMetadata
    candidate_result: ModelEvaluationResult
    baseline_result: ModelEvaluationResult
    comparison_report: Optional[ComparisonReport] = None
    acceptance_decision: Optional[AcceptanceDecision] = None
    warnings: List[str] = Field(default_factory=list)


class ReleaseGateStatus(str, Enum):
    """Overall outcome of a release-gate evaluation."""
    PASSED = "passed"
    BLOCKED = "blocked"
    INCONCLUSIVE = "inconclusive"


class GateCheckStatus(str, Enum):
    """Outcome of an individual gate check rule."""
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class ReleasePolicy(BaseModel):
    """Configuration governing release authorization criteria."""
    policy_id: str = Field(default="default_release_policy", description="Unique identifier for the release policy")
    version: str = Field(default="1.0.0", description="Semantic version of the release policy")
    target_environment: str = Field(default="production", description="Target deployment tier, e.g. staging, production")
    require_acceptance_pass: bool = Field(default=True, description="Candidate must have a conclusive PASS acceptance decision")
    require_model_artifact: bool = Field(default=False, description="Candidate model serialized pipeline must exist and match checksum")
    require_baseline_comparison: bool = Field(default=False, description="Comparison against baseline must be completed and statistically evaluated")
    require_robustness_evidence: bool = Field(default=False, description="Candidate robustness stress testing must be present and evaluated")
    require_calibration_evidence: bool = Field(default=False, description="Candidate probability calibration must be present and usable")
    require_manifest_integrity: bool = Field(default=True, description="All artifacts in manifest must exist and match cryptographic digests")
    allow_advisory_warnings: bool = Field(default=True, description="If False, presence of any advisory warning blocks release")
    min_evaluated_folds: int = Field(default=2, ge=1, description="Minimum number of valid CV folds required")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary user metadata")


class GateCheckResult(BaseModel):
    """Detailed evaluation result for a single release requirement."""
    gate_id: str = Field(..., description="Unique identifier of the gate check")
    status: GateCheckStatus = Field(..., description="Evaluation verdict of this check")
    is_blocking: bool = Field(default=True, description="If True, failure blocks release")
    explanation: str = Field(..., description="Human-readable explanation of check outcome")
    reason_code: Optional[str] = Field(None, description="Standardized machine-readable reason code")
    evidence_references: Dict[str, Any] = Field(default_factory=dict, description="Relevant evidence values or paths")


class ReleaseGateResult(BaseModel):
    """Comprehensive release gate authorization decision."""
    run_id: str = Field(..., description="Evaluation run identifier")
    overall_status: ReleaseGateStatus = Field(..., description="Final release authorization status")
    policy_id: str = Field(..., description="Evaluated release policy identifier")
    policy_version: str = Field(..., description="Release policy version")
    target_environment: str = Field(..., description="Target deployment environment")
    gate_results: List[GateCheckResult] = Field(default_factory=list, description="Ordered evaluation of individual gates")
    evaluated_at: str = Field(..., description="ISO 8601 UTC timestamp of gate evaluation")
    is_releasable: bool = Field(..., description="True iff overall_status is PASSED")
    summary: str = Field(..., description="Concise rationale for release decision")


class ReleaseCertificate(BaseModel):
    """Cryptographically signed release certificate authorizing model deployment."""
    certificate_schema_version: str = Field(default="1.0.0", description="Certificate contract schema version")
    certificate_id: str = Field(..., description="Unique UUIDv4 identifier for this certificate")
    run_id: str = Field(..., description="Evaluation run identifier")
    dataset_checksum: str = Field(..., description="SHA-256 checksum of source dataset")
    candidate_model_name: str = Field(..., description="Name of candidate model")
    candidate_model_type: str = Field(..., description="Model class or pipeline type")
    candidate_model_checksum: Optional[str] = Field(None, description="SHA-256 digest of serialized candidate model artifact")
    model_provenance: Optional[str] = Field(None, description="Model partition provenance, e.g. final_cv_fold")
    fitted_fold_index: Optional[int] = Field(None, description="Fold index from which model was retained")
    evaluation_schema_version: str = Field(..., description="Schema version of evaluation artifact")
    acceptance_decision_outcome: Optional[str] = Field(None, description="Acceptance decision outcome, e.g. pass")
    acceptance_policy_id: Optional[str] = Field(None, description="Acceptance policy ID if evaluated")
    acceptance_policy_version: Optional[str] = Field(None, description="Acceptance policy version")
    release_gate_outcome: ReleaseGateStatus = Field(..., description="Release gate verdict (must be passed for valid release)")
    release_policy_id: str = Field(..., description="Release policy ID evaluated")
    release_policy_version: str = Field(..., description="Release policy version")
    target_environment: str = Field(..., description="Authorized target deployment tier")
    gate_summary: Dict[str, str] = Field(default_factory=dict, description="Summary mapping of gate_id to status")
    issued_at: str = Field(..., description="ISO 8601 UTC timestamp of certificate issuance")
    payload_digest_sha256: str = Field(..., description="SHA-256 digest of canonical signed payload")
    signature_algorithm: str = Field(default="Ed25519", description="Digital signature algorithm used")
    key_id: str = Field(..., description="Identifier or fingerprint of the public signing key")
    signature: str = Field(..., description="Hex-encoded Ed25519 cryptographic signature")


class CertificateVerificationResult(BaseModel):
    """Multi-tiered verification outcome for a release certificate."""
    certificate_id: str = Field(default="", description="Certificate identifier verified")
    run_id: str = Field(default="", description="Run identifier from certificate")
    payload_integrity: bool = Field(..., description="True if canonical payload hash matches recorded payload_digest_sha256")
    signature_validity: bool = Field(..., description="True if cryptographic signature is mathematically valid for public key")
    signer_trust: bool = Field(..., description="True if signing public key is explicitly present in trusted keys registry")
    evidence_consistency: bool = Field(..., description="True if certificate references match actual artifacts on disk")
    release_authorized: bool = Field(..., description="True iff all 4 verification tiers pass AND release_gate_outcome is PASSED")
    key_id: str = Field(default="", description="Signing key identifier")
    details: List[str] = Field(default_factory=list, description="Diagnostic details for each verification tier")
    errors: List[str] = Field(default_factory=list, description="Fatal verification errors or security failures")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings or trust caveats")


class RunState(str, Enum):
    """Lifecycle states of an evaluation run."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunRecord(BaseModel):
    """Durable metadata record of an evaluation run in local persistence."""
    run_id: str = Field(..., description="Unique run identifier")
    state: RunState = Field(..., description="Current lifecycle state")
    created_at: str = Field(..., description="ISO 8601 UTC creation timestamp")
    started_at: Optional[str] = Field(None, description="ISO 8601 UTC start timestamp")
    completed_at: Optional[str] = Field(None, description="ISO 8601 UTC terminal completion/failure timestamp")
    updated_at: str = Field(..., description="ISO 8601 UTC last update timestamp")
    dataset_checksum: Optional[str] = Field(None, description="SHA-256 digest of input dataset")
    evaluation_schema_version: Optional[str] = Field(None, description="Evaluation contract schema version")
    candidate_model_type: Optional[str] = Field(None, description="Candidate model architecture or pipeline type")
    baseline_model_type: Optional[str] = Field(None, description="Baseline model architecture or pipeline type")
    acceptance_outcome: Optional[str] = Field(None, description="Evaluated acceptance outcome (PASS, FAIL, INCONCLUSIVE)")
    artifact_path: Optional[str] = Field(None, description="Relative or safe filesystem reference to run artifact directory")
    error_category: Optional[str] = Field(None, description="Standardized error category if run failed")
    error_message: Optional[str] = Field(None, description="Sanitized, safe failure explanation (no secrets or raw stack traces)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary run tags or configuration metadata")


class RunSummaryMetric(BaseModel):
    """Indexed cross-validation summary metric for fast filtering and ranking."""
    metric_name: str = Field(..., description="Standard metric identifier")
    metric_value: Optional[float] = Field(None, description="Aggregated metric score across folds")
    fold_count: Optional[int] = Field(None, description="Number of evaluated folds")
    std_dev: Optional[float] = Field(None, description="Standard deviation across folds")
    ci_lower: Optional[float] = Field(None, description="95% confidence interval lower bound")
    ci_upper: Optional[float] = Field(None, description="95% confidence interval upper bound")


class LifecycleTransitionEvent(BaseModel):
    """Audit log entry capturing a run state machine transition."""
    run_id: str = Field(..., description="Run identifier")
    from_state: Optional[RunState] = Field(None, description="Source lifecycle state")
    to_state: RunState = Field(..., description="Target lifecycle state")
    transitioned_at: str = Field(..., description="ISO 8601 UTC transition timestamp")
    reason: Optional[str] = Field(None, description="Operational explanation or trigger rationale")
