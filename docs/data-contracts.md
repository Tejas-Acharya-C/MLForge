# MLForge — Versioned Data Contracts

**Document Version:** 1.0.0  
**Phase:** 1 (Foundation & Architecture)  
**Schema Version:** `v1.0.0`  
**Status:** Approved Specification  

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
    threshold: float
    severity: str = Field("blocker", description="'blocker' causes Fail; 'advisory' flags warning")
    description: str

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
    evaluation_id: str
    artifacts: List[ArtifactReference]
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
