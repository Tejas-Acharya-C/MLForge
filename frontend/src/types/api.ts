/**
 * MLForge Typed Frontend API Contracts.
 * Directly mirrors the backend contracts and REST API schemas.
 */

export type RunState = 'created' | 'running' | 'completed' | 'failed' | 'cancelled';
export type AcceptanceOutcome = 'PASS' | 'FAIL' | 'INCONCLUSIVE';
export type CriterionOutcome = 'satisfied' | 'violated' | 'inconclusive' | 'skipped';
export type CriterionSeverity = 'blocker' | 'advisory';
export type RuleOperator = '>=' | '<=' | '>' | '<' | '==' | '!=' | 'statistically_greater_than' | 'statistically_less_than' | 'statistically_significant' | 'ci_disjoint_superior';
export type ReleaseGateStatus = 'passed' | 'blocked' | 'inconclusive';

export interface DatasetMetadata {
  num_rows: number;
  num_features: number;
  feature_names: string[];
  feature_types: Record<string, string>;
  numeric_feature_count: number;
  categorical_feature_count: number;
  target_name: string;
  target_type: string;
  class_distribution: Record<string, number>;
  class_labels: string[];
  imbalance_ratio: number;
  missing_value_counts: Record<string, number>;
  constant_columns: string[];
  checksum_sha256: string;
  warnings: string[];
}

export interface SampleDatasetInfo {
  id: string;
  name: string;
  description: string;
  num_rows: number;
  num_features: number;
  default_target: string;
  default_subgroup?: string;
}

export interface MetricSummary {
  metric_name: string;
  mean: number | null;
  std: number | null;
  median: number | null;
  iqr: number | null;
  min_value: number | null;
  max_value: number | null;
  confidence_interval_95: [number, number] | null;
  valid_fold_count: number;
  undefined_fold_count: number;
  undefined_reasons: string[];
}

export interface CalibrationBin {
  bin_index: number;
  lower_bound: number;
  upper_bound: number;
  sample_count: number;
  mean_predicted_prob: number | null;
  observed_positive_rate: number | null;
  is_empty: boolean;
}

export interface CalibrationCurveResult {
  bins: CalibrationBin[];
  expected_calibration_error: number | null;
  max_calibration_error: number | null;
  brier_score: number | null;
  log_loss: number | null;
  bin_strategy: string;
  num_bins: number;
  total_samples: number;
  positive_class_label: string;
  is_usable: boolean;
  warnings: string[];
}

export interface SubgroupMetricDiagnostic {
  subgroup_column: string;
  subgroup_value: string;
  sample_count: number;
  error_count: number;
  error_rate: number;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  is_usable: boolean;
  unusable_reason?: string;
}

export interface StatisticalTestResult {
  test_name: string;
  status: string;
  statistic: number | null;
  p_value: number | null;
  degrees_of_freedom: number | null;
  alpha: number;
  statistically_significant: boolean | null;
  interpretation?: string;
  warnings: string[];
  assumptions_and_limitations: string[];
}

export interface MetricComparisonResult {
  metric_name: string;
  direction: string;
  paired_differences: {
    raw_deltas: (number | null)[];
    mean_delta: number | null;
    std_delta: number | null;
    is_usable: boolean;
  };
  corrected_resampled_t_test?: StatisticalTestResult;
  wilcoxon_signed_rank_test?: StatisticalTestResult;
}

export interface ComparisonReport {
  candidate_model_name: string;
  baseline_model_name: string;
  dataset_checksum: string;
  num_folds: number;
  metric_comparisons: Record<string, MetricComparisonResult>;
  warnings: string[];
}

export interface FoldResult {
  fold_index: number;
  train_indices_count: number;
  val_indices_count: number;
  metrics: Record<string, { value: number | null; is_undefined: boolean; undefined_reason?: string }>;
  confusion_matrix: number[][];
  class_labels: string[];
  fit_duration_seconds: number;
  predict_duration_seconds: number;
  partition_fingerprint?: string;
}

export interface ModelEvaluationResult {
  model_name: string;
  model_type: string;
  hyperparameters: Record<string, any>;
  fold_results: FoldResult[];
  metric_summaries: Record<string, MetricSummary>;
  aggregate_confusion_matrix: number[][];
  calibration?: CalibrationCurveResult;
  error_diagnostics?: {
    total_samples: number;
    total_errors: number;
    overall_error_rate: number;
    subgroup_diagnostics?: Record<string, SubgroupMetricDiagnostic[]>;
  };
}

export interface CriterionEvaluation {
  rule_id: string;
  metric: string;
  severity: CriterionSeverity;
  operator: RuleOperator;
  threshold_value: number | null;
  observed_value: number | null;
  outcome: CriterionOutcome;
  reason_code: string;
  message: string;
  evidence_source: string;
}

export interface AcceptanceDecision {
  decision_id: string;
  evaluation_id?: string;
  policy_id: string;
  policy_version: string;
  outcome: AcceptanceOutcome;
  criteria_evaluations: CriterionEvaluation[];
  summary_rationale: string;
  evaluated_at: string;
  passed_rules_count: number;
  failed_rules_count: number;
  inconclusive_rules_count: number;
  advisory_warnings: string[];
  hash_signature?: string;
}

export interface AcceptanceRule {
  rule_id: string;
  metric: string;
  operator: RuleOperator;
  threshold?: number;
  alpha?: number;
  min_effect_size?: number;
  severity: CriterionSeverity;
  evidence_source: string;
  description?: string;
}

export interface AcceptancePolicy {
  policy_id: string;
  version: string;
  name: string;
  description?: string;
  fail_precedence: boolean;
  rules: AcceptanceRule[];
}

export interface PolicyTemplateInfo {
  policy_id: string;
  version: string;
  name: string;
  description?: string;
  rules_count: number;
  policy: AcceptancePolicy;
}

export interface RunSummaryMetric {
  metric_name: string;
  metric_value: number | null;
  fold_count?: number;
  std_dev?: number;
  ci_lower?: number;
  ci_upper?: number;
}

export interface LifecycleTransitionEvent {
  run_id: string;
  from_state?: RunState;
  to_state: RunState;
  transitioned_at: string;
  reason?: string;
}

export interface RunRecord {
  run_id: string;
  state: RunState;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  updated_at: string;
  dataset_checksum?: string;
  evaluation_schema_version?: string;
  candidate_model_type?: string;
  baseline_model_type?: string;
  acceptance_outcome?: string;
  artifact_path?: string;
  error_category?: string;
  error_message?: string;
  metadata: Record<string, any>;
}

export interface EvaluationSummaryItem {
  run_id: string;
  state: RunState;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  updated_at: string;
  dataset_checksum?: string;
  candidate_model_type?: string;
  baseline_model_type?: string;
  acceptance_outcome?: string;
  artifact_path?: string;
  error_category?: string;
  error_message?: string;
  metadata: Record<string, any>;
  summary_metrics: Record<string, number | null>;
  metric_details: RunSummaryMetric[];
}

export interface EvaluationListResponse {
  items: EvaluationSummaryItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface EvaluationDetailResponse {
  record: RunRecord;
  lifecycle_history: LifecycleTransitionEvent[];
  metrics: RunSummaryMetric[];
  artifacts: { name: string; size_bytes: number; relative_path: string }[];
  has_evaluation: boolean;
  has_decision: boolean;
  has_report: boolean;
  has_certificate: boolean;
  has_bundle: boolean;
}

export interface RobustnessPerturbationResult {
  perturbation_type: string;
  severity_level: number;
  metric_name: string;
  clean_score: number | null;
  perturbed_score: number | null;
  retained_ratio: number | null;
  performance_drop_percentage: number | null;
  is_usable: boolean;
  unusable_reason?: string;
}

export interface RobustnessCurve {
  perturbation_type: string;
  metric_name: string;
  clean_score?: number;
  points: RobustnessPerturbationResult[];
  area_under_degradation_curve?: number;
  aurdc?: number;
}

export interface RobustnessEvaluationResult {
  curves: RobustnessCurve[];
  summary_retention: Record<string, number>;
  is_usable: boolean;
  warnings: string[];
}

export interface GateCheckResult {
  gate_id: string;
  status: 'passed' | 'failed' | 'inconclusive';
  is_blocking: boolean;
  explanation: string;
  reason_code?: string;
  evidence_references: Record<string, any>;
}

export interface ReleaseGateResult {
  run_id: string;
  overall_status: ReleaseGateStatus;
  policy_id: string;
  policy_version: string;
  target_environment: string;
  gate_results: GateCheckResult[];
  evaluated_at: string;
  is_releasable: boolean;
  summary: string;
}

export interface ReleaseCertificate {
  certificate_schema_version: string;
  certificate_id: string;
  run_id: string;
  dataset_checksum: string;
  candidate_model_name: string;
  candidate_model_type: string;
  candidate_model_checksum?: string;
  model_provenance?: string;
  fitted_fold_index?: number;
  evaluation_schema_version: string;
  acceptance_decision_outcome?: string;
  acceptance_policy_id?: string;
  acceptance_policy_version?: string;
  release_gate_outcome: ReleaseGateStatus;
  release_policy_id: string;
  release_policy_version: string;
  target_environment: string;
  gate_summary: Record<string, string>;
  issued_at: string;
  payload_digest_sha256: string;
  signature_algorithm: string;
  key_id: string;
  signature: string;
}

export interface CertificateVerificationResult {
  certificate_id: string;
  run_id: string;
  payload_integrity: boolean;
  signature_validity: boolean;
  signer_trust: boolean;
  evidence_consistency: boolean;
  release_authorized: boolean;
  key_id: string;
  details: string[];
  errors: string[];
  warnings: string[];
}

export interface AuditBundleVerificationResult {
  is_valid: boolean;
  bundle_id: string;
  run_id: string;
  verified_files: string[];
  errors: string[];
  warnings: string[];
}

export type EvaluationResults = any;
export type FoldMetrics = FoldResult;
export type CalibrationDiagnostics = CalibrationCurveResult;
export type SubgroupDiagnostics = any;
export type EvaluationRunSummary = any;
export type ArtifactFile = { path: string; size_bytes: number; sha256: string };
export type AuditBundleResult = any;
export type AcceptanceReport = any;
export type ReleaseGateEvaluationResult = any;
export interface EvaluationStatusResponse {
  run_id: string;
  state: RunState;
  started_at?: string;
  completed_at?: string;
  acceptance_outcome?: string;
  error_message?: string;
}

export type VerificationResult = CertificateVerificationResult;
export type ReleasePolicy = any;
