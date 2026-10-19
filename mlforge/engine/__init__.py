"""MLForge Scientific Evaluation Engine Module.

Contains protocols, pipelines, metrics, and orchestrator for model evaluation.
"""

from mlforge.engine.acceptance import evaluate_acceptance_policy
from mlforge.engine.diagnostics import (
    DEFAULT_THRESHOLDS,
    aggregate_error_diagnostics,
    analyze_classification_thresholds,
    compute_fold_error_diagnostic,
    compute_subgroup_diagnostics,
)
from mlforge.engine.evaluator import evaluate_classification
from mlforge.engine.metrics import (
    aggregate_metric_folds,
    calculate_fold_classification_metrics,
)
from mlforge.engine.pipeline import build_fold_pipeline
from mlforge.engine.protocol import FoldSplit, generate_stratified_folds
from mlforge.engine.reliability import compute_calibration_curve
from mlforge.engine.robustness import (
    STANDARD_SEVERITY_GRID,
    apply_category_dropout,
    apply_compound_perturbation,
    apply_gaussian_noise,
    apply_missingness_injection,
    calculate_aurdc,
    calculate_retention_metrics,
    create_category_dropout_stress_profile,
    create_compound_stress_profile,
    create_missingness_stress_profile,
    create_noise_stress_profile,
    evaluate_model_robustness,
    evaluate_perturbation_step,
    evaluate_robustness_curve,
    evaluate_stress_profile,
    get_standard_stress_suite,
    perturb_dataset,
)
from mlforge.engine.release_gate import evaluate_release_gate
from mlforge.engine.statistics import (
    METRIC_DIRECTIONS,
    compare_model_evaluations,
    compute_paired_metric_differences,
    nadeau_bengio_corrected_t_test,
    validate_evaluation_compatibility,
    wilcoxon_signed_rank_test,
)

__all__ = [
    "FoldSplit",
    "generate_stratified_folds",
    "build_fold_pipeline",
    "calculate_fold_classification_metrics",
    "aggregate_metric_folds",
    "evaluate_classification",
    "METRIC_DIRECTIONS",
    "validate_evaluation_compatibility",
    "compute_paired_metric_differences",
    "nadeau_bengio_corrected_t_test",
    "wilcoxon_signed_rank_test",
    "compare_model_evaluations",
    "compute_calibration_curve",
    "analyze_classification_thresholds",
    "compute_fold_error_diagnostic",
    "compute_subgroup_diagnostics",
    "aggregate_error_diagnostics",
    "DEFAULT_THRESHOLDS",
    "evaluate_acceptance_policy",
    "STANDARD_SEVERITY_GRID",
    "apply_gaussian_noise",
    "apply_missingness_injection",
    "apply_category_dropout",
    "apply_compound_perturbation",
    "perturb_dataset",
    "calculate_retention_metrics",
    "calculate_aurdc",
    "evaluate_perturbation_step",
    "evaluate_robustness_curve",
    "evaluate_model_robustness",
    "create_noise_stress_profile",
    "create_missingness_stress_profile",
    "create_category_dropout_stress_profile",
    "create_compound_stress_profile",
    "get_standard_stress_suite",
    "evaluate_stress_profile",
    "evaluate_release_gate",
]
