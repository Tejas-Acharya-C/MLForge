# MLForge — Versioned Data Contracts

**Document Version:** 2.0.0
**Schema Version:** `v1.0.0`
**Status:** Implemented & Active Specification

---

## 1. Specification Overview

All data exchanged between the frontend, API, persistence layer, and evaluation engine adheres to immutable, versioned contracts. Contracts are formalized in Python (Pydantic v2 compatible) and mapped to JSON Schemas for frontend TypeScript generation.

Every contract includes a top-level or embedded `schema_version` (e.g. `"1.0.0"`) to facilitate schema migration and historical reproducibility.

---

## 2. Core Evaluation Configurations

### A. Evaluation Configuration (`EvaluationConfig`)

```python
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"

class EvaluationConfig(BaseModel):
    schema_version: str = Field(default="1.0.0", description="Contract version")
    evaluation_id: Optional[str] = Field(None, description="System-assigned UUID")
    name: str = Field(..., min_length=1, max_length=120, description="Human-readable run title")
    description: Optional[str] = Field(None, max_length=1000)
    dataset: "DatasetConfig"
    protocol: "SplitConfig"
    candidate_model: "ModelConfig"
    baseline_model: "ModelConfig"
    metrics: List[str] = Field(
        default=["accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall"],
        description="Metric keys to evaluate"
    )
    acceptance_policy_id: Optional[str] = Field(None, description="Policy key to evaluate against")
    random_seed: int = Field(default=42, description="Master random seed for reproducibility")
```

### B. Dataset Configuration & Metadata

```python
class DatasetConfig(BaseModel):
    source_type: str = Field("file", description="Source format: 'file', 'memory', 'synthetic'")
    file_path: Optional[str] = Field(None, description="Path relative to MLForge data directory")
    target_column: str = Field(..., min_length=1, description="Target label column name")
    feature_columns: Optional[List[str]] = Field(None, description="Explicit feature list; if omitted, all non-target columns")
    categorical_columns: Optional[List[str]] = Field(default_factory=list)
    numeric_columns: Optional[List[str]] = Field(default_factory=list)
    subgroup_columns: Optional[List[str]] = Field(default_factory=list, description="Columns tracked for subgroup slice analysis")

class DatasetMetadata(BaseModel):
    num_rows: int
    num_features: int
    feature_names: List[str]
    target_name: str
    target_type: str
    class_distribution: Dict[str, int]
    missing_value_counts: Dict[str, int]
    constant_columns: List[str]
    checksum_sha256: str = Field(..., description="Cryptographic hash of the raw input dataset")
```

### C. Validation & Split Configuration (`SplitConfig`)

```python
class SplitStrategy(str, Enum):
    STRATIFIED_K_FOLD = "stratified_k_fold"
    K_FOLD = "k_fold"
    TRAIN_TEST_SPLIT = "train_test_split"

class SplitConfig(BaseModel):
    strategy: SplitStrategy = Field(default=SplitStrategy.STRATIFIED_K_FOLD)
    num_folds: int = Field(default=5, ge=2, le=20, description="K folds for cross-validation")
    test_size: Optional[float] = Field(default=0.2, ge=0.05, le=0.5, description="Holdout fraction if using train_test_split")
    shuffle: bool = Field(default=True)
    random_state: int = Field(default=42)
```

### D. Model Configuration (`ModelConfig`)

```python
class ModelType(str, Enum):
    DUMMY_CLASSIFIER = "dummy_classifier"
    LOGISTIC_REGRESSION = "logistic_regression"

class PreprocessingConfig(BaseModel):
    numeric_imputer: str = Field("median", description="none, mean, median, constant")
    categorical_imputer: str = Field("most_frequent", description="none, most_frequent, constant")
    numeric_scaler: str = Field("standard", description="none, standard, minmax, robust")
    categorical_encoder: str = Field("onehot", description="onehot, ordinal, none")

class ModelConfig(BaseModel):
    model_type: ModelType
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
```

---

## 3. Results & Metric Contracts

### A. Fold-Level Result (`FoldResult`)

```python
class MetricValue(BaseModel):
    value: Optional[float] = Field(None, description="Numeric metric score; None if undefined")
    is_undefined: bool = Field(default=False)
    undefined_reason: Optional[str] = Field(None, description="Diagnostic reason if division by zero or single class")

class FoldResult(BaseModel):
    fold_index: int = Field(..., ge=0)
    train_row_count: int
    val_row_count: int
    metrics: Dict[str, MetricValue]
    confusion_matrix: List[List[int]] = Field(..., description="2D contingency matrix [true_idx][pred_idx]")
    class_labels: List[str]
    fit_duration_seconds: float
    predict_duration_seconds: float
```

### B. Metric Summary & Aggregation (`MetricSummary`, `AggregatedResults`)

```python
class MetricSummary(BaseModel):
    metric_name: str
    mean: float
    std: float
    median: float
    iqr: float
    min_value: float
    max_value: float
    confidence_interval_95: List[float] = Field(..., min_length=2, max_length=2, description="[lower_ci, upper_ci]")
    undefined_count: int = Field(default=0, description="Number of folds where metric was undefined")

class StatisticalTestResult(BaseModel):
    test_name: str = Field(..., description="e.g. 'Nadeau-Bengio Corrected Resampled t-test'")
    statistic: float
    p_value: float
    degrees_of_freedom: Optional[float] = None
    statistically_significant: bool = Field(..., description="True if p_value < alpha threshold")
    alpha: float = Field(default=0.05)
    interpretation: str

class AggregatedResults(BaseModel):
    model_name: str
    model_type: str
    num_folds: int
    metric_summaries: Dict[str, MetricSummary]
    aggregate_confusion_matrix: List[List[int]]
    statistical_comparison_vs_baseline: Optional[Dict[str, StatisticalTestResult]] = None
```

---

## 4. Diagnostics & Reliability Contracts

```python
class CalibrationBin(BaseModel):
    bin_index: int
    lower_bound: float
    upper_bound: float
    mean_predicted_prob: float
    empirical_accuracy: float
    sample_count: int

class CalibrationDiagnostics(BaseModel):
    bins: List[CalibrationBin]
    expected_calibration_error: float = Field(..., description="ECE weighted score")
    maximum_calibration_error: float = Field(..., description="MCE score")
    brier_score: float

class SubgroupSliceMetric(BaseModel):
    subgroup_feature: str
    subgroup_value: str
    sample_count: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float

class RobustnessPerturbationResult(BaseModel):
    perturbation_type: str = Field(..., description="e.g. 'gaussian_noise', 'missingness_injection'")
    severity_level: float = Field(..., description="Severity parameter, e.g. 0.05, 0.10, 0.25")
    retained_accuracy: float
    performance_drop_percentage: float

class DiagnosticResult(BaseModel):
    calibration: Optional[CalibrationDiagnostics] = None
    subgroups: Optional[List[SubgroupSliceMetric]] = None
    robustness: Optional[List[RobustnessPerturbationResult]] = None
    high_confidence_errors_count: int = Field(default=0, description="Errors where pred_proba >= 0.85")
```

---

## 5. Acceptance Policy & Decision Contracts

```python
class AcceptanceOutcome(str, Enum):
    PASS = "Pass"
    FAIL = "Fail"
    INCONCLUSIVE = "Inconclusive"

class RuleOperator(str, Enum):
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN_OR_EQUAL = "<="
    STATISTICALLY_GREATER_THAN = "statistically_greater_than"

class AcceptanceRule(BaseModel):
    rule_id: str
    metric: str
    operator: RuleOperator
    threshold: Optional[float]
    severity: str = Field("blocker", description="'blocker' causes Fail; 'advisory' flags warning")
    evidence_source: str = Field(
        "candidate_metric",
        description="Source: 'candidate_metric', 'baseline_contrast', 'calibration', 'statistical_test', 'data_quality', 'robustness'",
    )
    description: Optional[str] = None

class AcceptancePolicy(BaseModel):
    policy_id: str
    version: str = Field(default="1.0.0")
    name: str
    rules: List[AcceptanceRule]

class CriterionEvaluation(BaseModel):
    rule_id: str
    metric: str
    observed_value: Optional[float]
    threshold_value: float
    operator: RuleOperator
    satisfied: bool
    is_inconclusive: bool = False
    notes: Optional[str] = None

class AcceptanceDecision(BaseModel):
    decision_id: str
    evaluation_id: str
    policy_id: str
    policy_version: str
    outcome: AcceptanceOutcome
    criteria_evaluations: List[CriterionEvaluation]
    summary_rationale: str
    evaluated_at: str
    hash_signature: str = Field(..., description="SHA256 signature over evaluation metadata and criteria")
```

---

## 6. Run Status & Lifecycle Contracts

```python
class RunState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class EvaluationRunStatus(BaseModel):
    evaluation_id: str
    state: RunState
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    current_fold: Optional[int] = None
    total_folds: Optional[int] = None
    current_action: str = Field(default="Initializing")
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error_message: Optional[str] = None
    cancellation_reason: Optional[str] = None
```

---

## 7. Artifact Reference Manifest

```python
class ArtifactType(str, Enum):
    CONFIG_DUMP = "config_dump"
    FOLD_METRICS = "fold_metrics"
    AGGREGATED_METRICS = "aggregated_metrics"
    DIAGNOSTICS_PAYLOAD = "diagnostics_payload"
    FITTED_PIPELINE = "fitted_pipeline"
    ACCEPTANCE_EVIDENCE = "acceptance_evidence"
    OFFLINE_HTML_REPORT = "offline_html_report"

class ArtifactReference(BaseModel):
    artifact_id: str
    evaluation_id: str
    artifact_type: ArtifactType
    relative_path: str = Field(..., description="Path relative to artifacts root")
    file_size_bytes: int
    sha256_checksum: str
    mime_type: str
    created_at: str

class ArtifactManifest(BaseModel):
    schema_version: str = Field(default="1.0.0")
    run_id: str
    created_at: str
    dataset_checksum: str
    split_strategy: str
    num_folds: int
    candidate_model_type: str
    baseline_model_type: str
    policy_id: Optional[str] = None
    policy_version: Optional[str] = None
    decision_outcome: Optional[AcceptanceOutcome] = None
    software_versions: Dict[str, str] = Field(default_factory=dict)
    file_checksums: Dict[str, str] = Field(default_factory=dict)
    has_robustness: bool = Field(default=False, description="True if candidate or baseline robustness curves are serialized in evaluation.json")
```

---

## 8. Error Responses

```python
class ErrorDetail(BaseModel):
    loc: List[str]
    msg: str
    type: str

class APIErrorResponse(BaseModel):
    error_code: str
    message: str
    details: Optional[List[ErrorDetail]] = None
    recovery_guidance: Optional[str] = None
    timestamp: str
```

---

## 9. Robustness & Stress Testing Contracts (Phase 3)

Controlled tabular feature perturbation and robustness degradation schemas for model reliability auditing.

### A. Perturbation Configuration Contracts

```python
class PerturbationType(str, Enum):
    GAUSSIAN_NOISE = "gaussian_noise"
    MISSINGNESS_INJECTION = "missingness_injection"
    CATEGORY_DROPOUT = "category_dropout"
    COMPOUND = "compound"

class PerturbationConfig(BaseModel):
    perturbation_type: PerturbationType
    severity_levels: List[float] = Field(
        default_factory=lambda: [0.05, 0.10, 0.25],
        description="Severity levels: noise scale multiplier (>= 0.0) or masking/dropout fraction in [0.0, 1.0]",
    )
    target_columns: Optional[List[str]] = Field(
        None, description="Subset of feature columns to perturb; if None, all eligible features"
    )
    random_state: int = Field(default=42, description="Random seed for deterministic corruption")
    compound_steps: Optional[List["PerturbationConfig"]] = Field(
        default=None,
        description="Sub-perturbation configurations executed sequentially when perturbation_type is COMPOUND",
    )
```

**Validation Invariants:**
- `severity_levels` must contain at least one element.
- Severity levels must be strictly sorted in non-decreasing order without duplicates.
- Gaussian noise severity must satisfy $\text{severity} \ge 0.0$.
- Missingness injection and category dropout must satisfy $0.0 \le \text{severity} \le 1.0$.
- Compound perturbation severity must satisfy $0.0 \le \text{severity} \le 1.0$ (scaled to constituent step severities).
- `compound_steps` must not contain nested `COMPOUND` perturbations (rejects recursive chaining).

### B. Robustness Evidence & Aggregation Contracts

```python
class RobustnessPerturbationResult(BaseModel):
    perturbation_type: PerturbationType
    severity_level: float = Field(..., ge=0.0)
    metric_name: str = Field(default="accuracy")
    clean_score: Optional[float] = None
    perturbed_score: Optional[float] = None
    retained_ratio: Optional[float] = None
    performance_drop_percentage: Optional[float] = None
    is_usable: bool = Field(default=True)
    unusable_reason: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

class RobustnessCurve(BaseModel):
    perturbation_type: PerturbationType
    metric_name: str
    clean_score: Optional[float] = None
    points: List[RobustnessPerturbationResult] = Field(default_factory=list)
    area_under_degradation_curve: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)

    @property
    def aurdc(self) -> Optional[float]:
        """Convenience property alias for area_under_degradation_curve (AURDC)."""
        return self.area_under_degradation_curve

class RobustnessEvaluationResult(BaseModel):
    curves: List[RobustnessCurve] = Field(default_factory=list)
    summary_retention: Dict[str, float] = Field(
        default_factory=dict, description="Summary retention scores: per-curve retention at max severity and per-curve/overall AURDC"
    )
    is_usable: bool = Field(default=True)
    warnings: List[str] = Field(default_factory=list)
```

**Standard Stress Testing Profiles:**
- **Standard Grid:** `STANDARD_SEVERITY_GRID = [0.05, 0.10, 0.25, 0.50]`.
- **Factory Functions:**
  - `create_noise_stress_profile(target_columns=None, severities=None, random_state=42)`
  - `create_missingness_stress_profile(target_columns=None, severities=None, random_state=42)`
  - `create_category_dropout_stress_profile(target_columns=None, severities=None, random_state=42)`
  - `create_compound_stress_profile(target_columns=None, severities=None, random_state=42, compound_steps=None)`
  - `get_standard_stress_suite(numeric_columns=None, categorical_columns=None, random_state=42)`: Full stress-testing profile suite.
  - `evaluate_stress_profile(estimator, X_val, y_val, profiles=None, metric="accuracy", ...)`: Executes standardized profile evaluation.

**Metric-Direction Semantics:**
- **Higher-is-better metrics (`accuracy`, `balanced_accuracy`, `macro_f1`, `roc_auc`):**
  $$\text{retained\_ratio} = \frac{\text{perturbed\_score}}{\text{clean\_score}}, \quad \text{drop\_pct} = \frac{\text{clean\_score} - \text{perturbed\_score}}{\text{clean\_score}} \times 100\%$$
- **Lower-is-better loss metrics (`log_loss`, `brier_score`):**
  $$\text{retained\_ratio} = \frac{\text{clean\_score}}{\text{perturbed\_score}}, \quad \text{drop\_pct} = \frac{\text{perturbed\_score} - \text{clean\_score}}{\text{clean\_score}} \times 100\%$$
  Positive `drop_pct` indicates performance degradation (loss inflation); `retained_ratio < 1.0` indicates efficiency loss.

**Area Under Robustness Degradation Curve (AURDC):**
- Normalized numerical trapezoidal integration over severity range $[s_0, s_k]$:
  $$\text{AURDC} = \frac{1}{s_k - s_0} \int_{s_0}^{s_k} R(s)\, ds = \frac{1}{s_k - s_0} \sum_{i=1}^{k} \frac{R(s_{i-1}) + R(s_i)}{2} (s_i - s_{i-1})$$
  where $R(s)$ represents `retained_ratio` at severity $s$.
- Range: for monotonically degrading models, $\text{AURDC} \in [0.0, 1.0]$. Values $> 1.0$ indicate performance improvement under perturbation.
- Handled defensively across NumPy 1.x and NumPy 2.x via `getattr(np, "trapezoid", getattr(np, "trapz", None))`.

**Undefined Evidence & Survivorship Bias Defense:**
- If clean score is zero or negative, `is_usable=False` with explicit reason (`ZERO_BASELINE_SCORE` or `NEGATIVE_BASELINE_SCORE`).
- If any requested perturbation point in a curve fails or is unusable, the aggregate `area_under_degradation_curve` is strictly set to `None` with an explicit warning to prevent survivorship bias from masking catastrophic failures.
- If all points are usable, normalized trapezoidal integration computes AURDC across evaluated severities.

**Reproducibility Invariant:**
- Column-level seeds are derived deterministically via SHA-256 digests over canonical column identifiers and random state, guaranteeing identical perturbations across distinct Python processes and machines.
- Compound perturbations derive step seeds via `(random_state + step_idx * 10007) & 0xFFFFFFFF` to ensure cross-step independence without correlation artifacts.

### C. Robustness Acceptance Policy & Artifact Integration Contracts (Milestone 3.3)

Robustness stress-testing criteria evaluated against candidate model robustness evidence during acceptance policy execution.

```python
# Rule Specification Example
rule = AcceptanceRule(
    rule_id="max_noise_drop_15pct",
    metric="gaussian_noise_max_drop_pct",  # or "gaussian_noise:0.10:drop_pct", "mean_aurdc"
    operator=RuleOperator.LESS_THAN_OR_EQUAL,
    threshold=15.0,
    severity=CriterionSeverity.BLOCKER,
    evidence_source="robustness",
    description="Maximum permitted performance degradation under Gaussian noise is 15%",
)
```

**Supported Metric Syntax:**
- `mean_aurdc`: Overall mean AURDC across all evaluated perturbation curves.
- `<perturbation_type>_aurdc` or `aurdc:<perturbation_type>`: Curve-specific AURDC score.
- `<perturbation_type>_max_retention` or `<perturbation_type>:max_retention`: Retention ratio at maximum evaluated severity.
- `<perturbation_type>_max_drop_pct` or `<perturbation_type>:max_drop_pct`: Percentage performance drop at maximum evaluated severity.
- `<perturbation_type>:<severity>:drop_pct`: Percentage performance drop at a specific evaluated severity level.
- `<perturbation_type>:<severity>:retention`: Retained performance ratio at a specific evaluated severity level.

**Evaluation Semantics & Invariants:**
- If candidate robustness evidence is missing (`robustness is None`), the rule evaluates to `INCONCLUSIVE` with reason `ROBUSTNESS_EVIDENCE_MISSING`.
- If the requested perturbation curve was not evaluated, the rule evaluates to `INCONCLUSIVE` with reason `CURVE_NOT_FOUND`.
- If the requested severity level was not tested, the rule evaluates to `INCONCLUSIVE` with reason `SEVERITY_POINT_NOT_FOUND`.
- If an evaluation point or AURDC is unusable or failed, the rule evaluates to `INCONCLUSIVE` with reason `POINT_UNUSABLE` or `ROBUSTNESS_AURDC_UNAVAILABLE`.
- Non-finite observed values or thresholds evaluate to `INCONCLUSIVE` (`NON_FINITE_VALUE` or `INVALID_THRESHOLD`).
- Missing numerical threshold evaluates to `INCONCLUSIVE` (`THRESHOLD_MISSING`).
- Blocker violations yield overall decision `FAIL`; advisory violations emit warnings in `advisory_warnings` without blocking acceptance.

**Artifact Serialization:**
- Candidate and baseline robustness evidence are integrated directly into `evaluation.json` under `candidate_result.robustness` and `baseline_result.robustness`.
- `manifest.json` indexes `evaluation.json` with a SHA-256 checksum and sets `has_robustness=True` if robustness curves are present.
- Unusable points and undefined retention values serialize faithfully as JSON `null` without numeric zero coercion.

---

## 8. Offline HTML Audit Report Contracts (Milestone 4.2)

### A. Report Generation Public API
The reporting module transforms persisted run artifacts or verified `.zip` audit bundles into a self-contained, offline HTML audit report.

```python
from mlforge.application.reporting import generate_html_report, load_report_evidence

# Generate report from a local artifact run directory
report_path = generate_html_report(
    source="artifacts/runs/run_20261019_103000",
    output_path=None,   # Defaults to <source>/report.html
    overwrite=True,
)

# Generate report directly from a verified .zip audit bundle
report_path = generate_html_report(
    source="audit_bundles/run_20261019_103000.zip",
    output_path="reports/run_20261019_103000_report.html",
    overwrite=True,
)
```

### B. Core Principles & Guarantees
1. **100% Offline Independence:**
   - Zero external network requests, zero CDN links, zero remote web fonts, zero remote images, and zero remote analytics.
   - All styling (CSS) and visualization graphics (vector SVGs for calibration, thresholds, and stress trajectories) are completely inline.
2. **Safe from Code Execution:**
   - Model binary files (`*.joblib`, `*.pickle`) are NEVER loaded or deserialized during report generation. Only non-executable JSON evidence and cryptographic manifests are parsed.
3. **Immutability of Evaluation Artifacts:**
   - Report generation does not mutate `manifest.json`, `evaluation.json`, `decision.json`, or any other persisted artifact. The report is an on-demand presentation projection, avoiding circular hashing or invalidated bundle checksums.
4. **Atomic Output Delivery:**
   - Reports are written to a temporary sibling file (`<target>.tmp_<pid>_<uuid>`) and atomically replaced via `os.replace`, ensuring an interrupted write never produces a corrupted partial report.
5. **Untrusted Input Defense (XSS & Injection):**
   - All user, model, dataset, policy, and rule text strings are HTML-escaped using `html.escape(..., quote=True)`.
   - Embedded JSON data is sanitized against script injection (`</script>` -> `<\/script>`).
   - Relative and output paths are strictly validated against directory traversal attacks.

### C. 4-Tier Provenance & Trust Model
The report explicitly presents artifact verification across four distinct evidentiary tiers:
- **Tier 1 (File Existence):** Verified locally by confirming the required artifact files exist on the filesystem.
- **Tier 2 (Hash Match):** Verified locally by computing SHA-256 checksums over each artifact file and matching against `manifest.json`.
- **Tier 3 (Bundle Consistency):** Verified locally by validating archive structure, size bounds, and inner manifest integrity without arbitrary path extraction.
- **Tier 4 (External Trust / Author Authenticity):** Explicitly marked as **Unverified** locally unless an external digital signature / PKI mechanism is present. Cryptographic checksum matches certify integrity, NOT publisher identity.

---

## 9. Automated Release-Gate Engine & Signed Release Certificates Contracts (Milestone 4.3)

### A. Architectural Distinction: Acceptance vs. Release Authorization
MLForge maintains a strict boundary between metric-level statistical acceptance and release authorization:
- **Statistical Acceptance Engine (`mlforge.engine.acceptance`):** Answers: *Did the model meet configured metric thresholds, confidence intervals, and statistical tests on evaluation folds?* (Returns `PASS`, `FAIL`, or `INCONCLUSIVE`).
- **Release-Gate Engine (`mlforge.engine.release_gate`):** Answers: *Is this completed run sufficiently complete, valid, traceable, and compliant with release requirements to authorize a release?* (Returns `PASSED`, `BLOCKED`, or `INCONCLUSIVE`).

A statistical `PASS` does not grant release authorization if required artifacts are missing, provenance is inconsistent, manifests fail SHA-256 integrity, or schemas are incompatible.

### B. Core Release-Gate Contracts

```python
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ReleaseGateStatus(str, Enum):
    PASSED = "PASSED"
    BLOCKED = "BLOCKED"
    INCONCLUSIVE = "INCONCLUSIVE"

class GateCheckStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"
    INCONCLUSIVE = "INCONCLUSIVE"
    SKIPPED = "SKIPPED"

class ReleasePolicy(BaseModel):
    schema_version: str = Field(default="1.0.0")
    policy_id: str = Field(default="default_release_policy")
    require_acceptance_pass: bool = Field(default=True)
    require_model_artifact: bool = Field(default=False)
    require_baseline_comparison: bool = Field(default=False)
    require_robustness_evidence: bool = Field(default=False)
    require_calibration_evidence: bool = Field(default=False)
    require_manifest_verification: bool = Field(default=True)

class GateCheckResult(BaseModel):
    check_id: str
    category: str  # schema_compatibility, evidence_completeness, acceptance_compliance, artifact_integrity, provenance_consistency
    status: GateCheckStatus
    is_blocking: bool = True
    explanation: str
    evidence_reference: Optional[str] = None
    reason_code: Optional[str] = None

class ReleaseGateResult(BaseModel):
    schema_version: str = Field(default="1.0.0")
    run_id: str
    status: ReleaseGateStatus
    policy_id: str
    is_releasable: bool
    summary: str
    gate_checks: List[GateCheckResult]
    blocking_failures: List[str]
    warnings: List[str]
    inconclusive_reasons: List[str]
    evaluated_at: str
```

### C. Release Decision Semantics & Precedence
1. **`BLOCKED` takes absolute precedence:** Any conclusive failure of a blocking check (e.g., acceptance `FAIL`, SHA-256 hash mismatch, schema incompatibility, missing required model artifact) forces overall status to `BLOCKED`.
2. **`INCONCLUSIVE` takes secondary precedence:** If any blocking check cannot be conclusively evaluated (e.g., corrupted evidence, acceptance `INCONCLUSIVE`, unreadable file), and no conclusive blocker failure exists, status is `INCONCLUSIVE`.
3. **`PASSED` requires complete clearance:** All blocking checks must pass. Warnings on non-blocking checks do not block release authorization.
4. **No Certificate on Block/Inconclusive:** A certificate can **never** be issued for `BLOCKED` or `INCONCLUSIVE` runs.

### D. Cryptographically Signed Release Certificate Contracts

```python
class ReleaseCertificate(BaseModel):
    schema_version: str = Field(default="1.0.0")
    certificate_id: str
    run_id: str
    dataset_checksum: str
    candidate_model_type: str
    model_artifact_checksum: Optional[str] = None
    model_provenance: Optional[str] = None
    fitted_fold_index: Optional[int] = None
    evaluation_schema_version: str
    acceptance_outcome: Optional[str] = None
    acceptance_policy_id: Optional[str] = None
    release_gate_status: str
    release_policy_id: str
    released_at: str
    canonical_payload_sha256: str
    signature_algorithm: str = Field(default="ed25519")
    signing_key_id: str
    signature: str

class CertificateVerificationResult(BaseModel):
    schema_version: str = Field(default="1.0.0")
    certificate_id: str
    run_id: str
    is_valid: bool
    payload_integrity_valid: bool
    signature_valid: bool
    signer_trusted: bool
    evidence_consistent: bool
    release_authorized: bool
    signing_key_id: str
    algorithm: str
    failure_reasons: List[str]
    warnings: List[str]
    verified_at: str
```

### E. Canonicalization & Signature Guarantees
- **Canonical Payload Encoding:** UTF-8 encoded JSON representation with sorted keys, separators `(',', ':')`, and strict rejection of non-finite numbers (`NaN`, `Infinity`).
- **Algorithm:** Ed25519 (Edwards-curve Digital Signature Algorithm over Curve25519) providing 128-bit security level with constant-time operations and high resilience against side-channel vulnerabilities.
- **Excluded Fields:** `signature` and `canonical_payload_sha256` are excluded from the canonical payload to prevent circular dependencies.
- **Verification Boundaries (Fail-Closed):**
  - `signature_valid`: Confirms the signature matches the public key and payload hash (cryptographic authenticity).
  - `signer_trust`: Confirms the public key matches an explicitly supplied trusted public key or key registry (trust authorization).
  - `evidence_consistent`: Compares all signed certificate claims (run ID, dataset SHA-256, candidate model name/type, model SHA-256, model provenance, fitted fold index, schema version, and acceptance outcome/policies) against authoritative persisted evidence on disk or inside a verified `.zip` audit bundle.
  - **Fail-Closed Requirement:** If `expected_source` is omitted (`None`), `evidence_consistency` is strictly `False` and release authorization is denied. Physical evidence must be explicitly supplied and verified.
  - `release_authorized`: Confirms `release_gate_status == "passed"` and all four verification tiers (`payload_integrity`, `signature_validity`, `signer_trust`, `evidence_consistency`) pass.

### F. Persisted Artifact Inventory & Configuration Note
- **Authoritative Artifacts:** A completed evaluation run consists of `manifest.json`, `evaluation.json`, and conditionally `decision.json`, `policy.json`, and fitted model files (`models/candidate_pipeline.joblib`).
- **Evaluation Configuration Storage:** Evaluation configuration and reproducibility parameters are strictly captured inside `evaluation.json` (under `reproducibility`, `dataset_metadata`, and model result configurations). There is no standalone `config.json` artifact; claims and gate checks verify parameters directly from `evaluation.json` and `manifest.json`.

---

## 10. Local SQLite Persistence & Run Lifecycle State Machine Contracts (Milestone 4.4)

### A. Architectural Separation & Persistence Boundary
MLForge maintains strict architectural decoupling between the core scientific evaluation engine and persistence:
- **Core Evaluation Engine (`mlforge.engine`):** Operates on in-memory data structures, contracts, and pure functions. It has zero SQLite imports, zero database schema dependencies, and zero side effects.
- **Durable Persistence Module (`mlforge.application.persistence`):** Provides local SQLite-backed tracking of run lifecycle states, audit logs, metadata, cross-validation summary metrics, and artifact references.
- **Application Orchestrator (`execute_managed_evaluation`):** Coordinates lifecycle state transitions (`CREATED` -> `RUNNING` -> `COMPLETED` / `FAILED`), invokes the pure evaluation engine, writes atomic filesystem artifacts via `serialize_run_artifacts`, and updates persistence records.

### B. Database Location & Configuration
- **Default Database Location:** `Path("artifacts/mlforge.db")`
- **Configurability:** `SQLiteRunRepository(db_path=..., artifact_root=...)` accepts any local filesystem `Path`, string, or `":memory:"`.
- **In-Memory SQLite Semantics:** When configured with `":memory:"`, the repository allocates a dedicated shared in-memory URI (`file:mlforge_mem_<id>?mode=memory&cache=shared`) and maintains an internal keepalive connection to preserve tables and data across per-operation connection lifecycles without writing to NTFS filesystem paths.
- **Filesystem Safety:** Database files reside under `artifacts/` which is ignored by Git, ensuring local persistence data is never tracked in repository source control.

### C. Run Lifecycle State Machine

```
         ┌───────────┐
         │  CREATED  │
         └─────┬─────┘
               │
       ┌───────┴───────┐
       ▼               ▼
 ┌───────────┐   ┌───────────┐
 │  RUNNING  │   │ CANCELLED │ (Terminal)
 └─────┬─────┘   └───────────┘
       │
 ┌─────┼───────────────┐
 ▼     ▼               ▼
┌───────────┐   ┌───────────┐   ┌───────────┐
│ COMPLETED │   │  FAILED   │   │ CANCELLED │ (Terminal)
└───────────┘   └───────────┘   └───────────┘
 (Terminal)      (Terminal)
```

#### Lifecycle State Definitions (`RunState`)
- `CREATED`: Run metadata and identifier registered in database; execution has not started.
- `RUNNING`: Cross-validation, model fitting, and evaluation are actively in progress.
- `COMPLETED`: Evaluation finished successfully, all required filesystem artifacts serialized, and summary metrics indexed. (Terminal)
- `FAILED`: Evaluation, dataset validation, or artifact serialization encountered an error. Structured error category and sanitized message recorded. (Terminal)
- `CANCELLED`: Evaluation cancelled or abandoned prior to completion. (Terminal)

#### State Transition Rules
| Current State | Allowed Next States | Trigger / Condition |
| :--- | :--- | :--- |
| `CREATED` | `RUNNING`, `CANCELLED` | Execution begins or user cancels before start. |
| `RUNNING` | `COMPLETED`, `FAILED`, `CANCELLED` | Success, unhandled error, timeout recovery, or cancellation. |
| `COMPLETED` | *None* | **Terminal & Immutable.** Reopening or modifying state raises `LifecycleTransitionError`. |
| `FAILED` | *None* | **Terminal & Immutable.** Reopening or modifying state raises `LifecycleTransitionError`. |
| `CANCELLED` | *None* | **Terminal & Immutable.** Reopening or modifying state raises `LifecycleTransitionError`. |

#### Stale Run Recovery Semantics & Operational Limitation
- Active runs left in `RUNNING` state due to process termination or unexpected host shutdown can be recovered via `recover_stale_runs(timeout_seconds=...)`.
- Interrupted runs are transitioned to `FAILED` with category `"INTERRUPTED_CRASH"` and audit history logging.
- `RUNNING` runs are never automatically restarted or silently overwritten.
- **Operational Limitation:** `recover_stale_runs` relies strictly on elapsed wall-clock duration (`now - updated_at >= timeout_seconds`). It does not inspect operating system PIDs or heartbeat signals. If invoked with a timeout shorter than a long-running active evaluation, it may misclassify that active evaluation as `FAILED`. Callers must ensure `timeout_seconds` exceeds the maximum expected evaluation runtime.

### D. Relational Schema & Indices

```sql
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    updated_at TEXT NOT NULL,
    dataset_checksum TEXT,
    evaluation_schema_version TEXT,
    candidate_model_type TEXT,
    baseline_model_type TEXT,
    acceptance_outcome TEXT,
    artifact_path TEXT,
    error_category TEXT,
    error_message TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS run_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    metric_name TEXT NOT NULL,
    metric_value REAL,
    fold_count INTEGER,
    std_dev REAL,
    ci_lower REAL,
    ci_upper REAL
);

CREATE TABLE IF NOT EXISTS lifecycle_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    from_state TEXT,
    to_state TEXT NOT NULL,
    transitioned_at TEXT NOT NULL,
    reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_state ON runs(state);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
CREATE INDEX IF NOT EXISTS idx_run_metrics_run_id ON run_metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_run_metrics_name ON run_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_lifecycle_run_id ON lifecycle_events(run_id);
```

### E. Security, Transaction & Concurrency Guarantees
1. **Parameterized SQL Exclusively:** All queries use parameterized SQL (`?` placeholders). String concatenation or formatting of user input into SQL is strictly forbidden.
2. **Foreign Key Enforcement:** Foreign keys are explicitly enabled (`PRAGMA foreign_keys = ON;`) on every connection. Deleting or cascading runs maintains database integrity.
3. **Atomic State Transitions:** All lifecycle transitions and audit event entries execute inside atomic transactions (`BEGIN IMMEDIATE;` ... `COMMIT;`). Failed operations trigger automatic rollback.
4. **Transaction Boundary Isolation:** Database transactions are never held open during long-running model fitting, cross-validation, or filesystem I/O.
5. **Artifact-Path Confinement & Trust Boundaries:** `artifact_path` references are validated before persistence, rejecting `..` traversal sequences and null bytes. When an `artifact_root` is configured on the repository, paths are strictly verified to resolve within the root after canonical resolution. When `artifact_root` is omitted, external paths without traversal components are permitted under the documented external storage trust boundary.
6. **Observable Failure Handling:** Errors captured during failure transitions record a high-level category (`DATASET_VALIDATION_ERROR`, `DATA_LEAKAGE_ERROR`, `ARTIFACT_SERIALIZATION_ERROR`, `CONFIGURATION_ERROR`, `EXECUTION_ERROR`) and a sanitized message (truncated to 500 characters). If recording the `FAILED` state in SQLite also encounters an error, the secondary failure emits an observable `UserWarning` and warning log without masking or altering the primary evaluation exception.
7. **Explicit Out of Scope:** Multi-node distributed clustering, PostgreSQL/MySQL engines, Celery/Redis queues, asynchronous HTTP APIs, and ORM abstractions are intentionally out of scope for Milestone 4.4.
