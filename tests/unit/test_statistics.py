"""Unit tests for baseline contrast and statistical significance engine.

Covers:
- Pairing compatibility and validation
- Raw delta direction and preservation across metric types
- Nadeau-Bengio corrected resampled t-test with hand-checkable verification
- Wilcoxon signed-rank test with zero differences, ties, and caveats
- Integration with evaluate_classification
"""

import math
from typing import Dict, List, Optional

import numpy as np
import pytest
from scipy import stats

from mlforge.core.contracts import (
    DatasetMetadata,
    FoldResult,
    MetricDirection,
    MetricValue,
    ModelEvaluationResult,
    PairedMetricDifference,
    ReproducibilityMetadata,
    TestStatus,
)
from mlforge.core.exceptions import IncompatibleEvaluationPairError
from mlforge.engine.evaluator import evaluate_classification
from mlforge.engine.statistics import (
    METRIC_DIRECTIONS,
    compare_model_evaluations,
    compute_paired_metric_differences,
    nadeau_bengio_corrected_t_test,
    validate_evaluation_compatibility,
    wilcoxon_signed_rank_test,
)


# ==========================================
# Fixtures and Helpers
# ==========================================

def make_dummy_metadata(checksum: str = "abc123sha") -> DatasetMetadata:
    return DatasetMetadata(
        num_rows=100,
        num_features=5,
        feature_names=["f1", "f2", "f3", "f4", "f5"],
        feature_types={"f1": "float64"},
        numeric_feature_count=5,
        categorical_feature_count=0,
        target_name="target",
        target_type="int64",
        class_distribution={"0": 60, "1": 40},
        class_labels=["0", "1"],
        imbalance_ratio=1.5,
        missing_value_counts={},
        constant_columns=[],
        checksum_sha256=checksum,
    )


def make_dummy_reproducibility(num_folds: int = 5, seed: int = 42) -> ReproducibilityMetadata:
    return ReproducibilityMetadata(
        random_seed=seed,
        split_strategy="stratified_k_fold",
        num_folds=num_folds,
        package_versions={"numpy": "2.0.0"},
        platform_system="Windows",
        python_version="3.11.9",
    )


def make_mock_fold_results(
    metric_values: Dict[str, List[float]],
    num_folds: Optional[int] = None,
    train_count: int = 80,
    val_count: int = 20,
    fingerprint_suffix: str = "",
    omit_fingerprints: bool = False,
) -> List[FoldResult]:
    if num_folds is None:
        first_key = next(iter(metric_values))
        num_folds = len(metric_values[first_key])

    folds = []
    for i in range(num_folds):
        m_dict = {
            m_name: MetricValue(value=vals[i], is_undefined=False)
            for m_name, vals in metric_values.items()
        }
        folds.append(
            FoldResult(
                fold_index=i,
                train_indices_count=train_count,
                val_indices_count=val_count,
                metrics=m_dict,
                confusion_matrix=[[10, 2], [3, 5]],
                class_labels=["0", "1"],
                fit_duration_seconds=0.01,
                predict_duration_seconds=0.005,
                partition_fingerprint=None if omit_fingerprints else f"fp_{i}{fingerprint_suffix}",
            )
        )
    return folds


# ==========================================
# 1. Pairing and Compatibility Tests
# ==========================================

def test_pairing_compatible_evaluations():
    """Valid evaluations pass compatibility check without exception."""
    meta = make_dummy_metadata()
    repro = make_dummy_reproducibility()

    cand_folds = make_mock_fold_results({"accuracy": [0.85] * 5})
    base_folds = make_mock_fold_results({"accuracy": [0.80] * 5})

    cand = ModelEvaluationResult(
        model_name="Candidate", model_type="logistic", fold_results=cand_folds,
        metric_summaries={}, aggregate_confusion_matrix=[]
    )
    base = ModelEvaluationResult(
        model_name="Baseline", model_type="dummy", fold_results=base_folds,
        metric_summaries={}, aggregate_confusion_matrix=[]
    )

    validate_evaluation_compatibility(cand, base, meta, repro)


def test_pairing_fails_on_dataset_checksum_mismatch():
    """Incompatible dataset checksum raises IncompatibleEvaluationPairError."""
    meta = make_dummy_metadata(checksum="")
    repro = make_dummy_reproducibility()
    cand = ModelEvaluationResult(model_name="C", model_type="l", fold_results=[], metric_summaries={}, aggregate_confusion_matrix=[])
    base = ModelEvaluationResult(model_name="B", model_type="d", fold_results=[], metric_summaries={}, aggregate_confusion_matrix=[])

    with pytest.raises(IncompatibleEvaluationPairError, match="cryptographic checksum"):
        validate_evaluation_compatibility(cand, base, meta, repro)


def test_pairing_fails_on_fold_count_mismatch():
    """Candidate with 5 folds and baseline with 4 folds raises error."""
    meta = make_dummy_metadata()
    repro = make_dummy_reproducibility(num_folds=5)

    cand_folds = make_mock_fold_results({"accuracy": [0.8] * 5}, num_folds=5)
    base_folds = make_mock_fold_results({"accuracy": [0.8] * 4}, num_folds=4)

    cand = ModelEvaluationResult(model_name="C", model_type="l", fold_results=cand_folds, metric_summaries={}, aggregate_confusion_matrix=[])
    base = ModelEvaluationResult(model_name="B", model_type="d", fold_results=base_folds, metric_summaries={}, aggregate_confusion_matrix=[])

    with pytest.raises(IncompatibleEvaluationPairError, match="Baseline fold count"):
        validate_evaluation_compatibility(cand, base, meta, repro)


def test_pairing_fails_on_partition_size_mismatch():
    """Different validation slice sizes per fold raise IncompatibleEvaluationPairError."""
    meta = make_dummy_metadata()
    repro = make_dummy_reproducibility(num_folds=5)

    cand_folds = make_mock_fold_results({"accuracy": [0.8] * 5}, val_count=20)
    base_folds = make_mock_fold_results({"accuracy": [0.8] * 5}, val_count=25)

    cand = ModelEvaluationResult(model_name="C", model_type="l", fold_results=cand_folds, metric_summaries={}, aggregate_confusion_matrix=[])
    base = ModelEvaluationResult(model_name="B", model_type="d", fold_results=base_folds, metric_summaries={}, aggregate_confusion_matrix=[])

    with pytest.raises(IncompatibleEvaluationPairError, match="validation partition size mismatch"):
        validate_evaluation_compatibility(cand, base, meta, repro)


# ==========================================
# 2. Raw Delta Direction and Preservation Tests
# ==========================================

def test_raw_delta_direction_higher_is_better():
    """Raw delta is strictly candidate - baseline. Positive delta indicates candidate improvement for accuracy."""
    cand_folds = make_mock_fold_results({"accuracy": [0.85, 0.88]})
    base_folds = make_mock_fold_results({"accuracy": [0.80, 0.80]})

    diff = compute_paired_metric_differences(cand_folds, base_folds, "accuracy")
    assert diff.direction == MetricDirection.HIGHER_IS_BETTER
    assert diff.raw_deltas == [0.05, 0.08]
    assert diff.mean_delta == pytest.approx(0.065)


def test_raw_delta_direction_lower_is_better_never_inverted():
    """Raw delta is candidate - baseline for loss metrics; never inverted."""
    # Candidate log loss is lower (0.25 vs 0.35) -> raw delta is -0.10
    cand_folds = make_mock_fold_results({"log_loss": [0.25, 0.30]})
    base_folds = make_mock_fold_results({"log_loss": [0.35, 0.30]})

    diff = compute_paired_metric_differences(cand_folds, base_folds, "log_loss")
    assert diff.direction == MetricDirection.LOWER_IS_BETTER
    # Negative delta = improvement in loss, but raw_delta remains strictly cand - base
    assert diff.raw_deltas == [-0.10, 0.0]
    assert diff.mean_delta == pytest.approx(-0.05)


def test_undefined_metric_fold_handled_without_coercion():
    """If a fold has an undefined metric on either model, raw delta is None and fold is marked unusable."""
    cand_folds = make_mock_fold_results({"macro_f1": [0.80, 0.85, 0.90]})
    # Baseline has undefined metric on fold 1
    base_folds = [
        FoldResult(fold_index=0, train_indices_count=80, val_indices_count=20,
                   metrics={"macro_f1": MetricValue(value=0.50, is_undefined=False)},
                   confusion_matrix=[], class_labels=["0", "1"], fit_duration_seconds=0.1, predict_duration_seconds=0.01),
        FoldResult(fold_index=1, train_indices_count=80, val_indices_count=20,
                   metrics={"macro_f1": MetricValue(value=None, is_undefined=True, undefined_reason="Zero positives")},
                   confusion_matrix=[], class_labels=["0", "1"], fit_duration_seconds=0.1, predict_duration_seconds=0.01),
        FoldResult(fold_index=2, train_indices_count=80, val_indices_count=20,
                   metrics={"macro_f1": MetricValue(value=0.52, is_undefined=False)},
                   confusion_matrix=[], class_labels=["0", "1"], fit_duration_seconds=0.1, predict_duration_seconds=0.01),
    ]

    diff = compute_paired_metric_differences(cand_folds, base_folds, "macro_f1")
    assert diff.expected_folds == 3
    assert diff.usable_folds == 2
    assert diff.raw_deltas[1] is None
    assert diff.is_usable is False  # Cannot perform paired t-test with missing fold
    assert "undefined values" in diff.unusable_reason


# ==========================================
# 3. Nadeau-Bengio Corrected t-test Tests
# ==========================================

def test_hand_checkable_nadeau_bengio_t_test():
    """TC-AGG-03: Hand-calculated analytical verification of Nadeau-Bengio corrected t-test.

    Setup:
      m = 5 folds
      d = [0.05, 0.05, 0.07, 0.06, 0.07]
      mean(d) = 0.06
      d - mean = [-0.01, -0.01, 0.01, 0.00, 0.01]
      sum((d - mean)^2) = 0.0004
      s_d^2 = 0.0004 / 4 = 0.0001
      test_train_ratio = 1/4 = 0.25 (standard 5-fold CV)
      corr_factor = 1/5 + 1/4 = 0.20 + 0.25 = 0.45
      SE_corr = sqrt(0.45 * 0.0001) = sqrt(0.000045) = 0.006708203932
      t = 0.06 / 0.006708203932 = 8.94427191
      df = 4
      p_value = 2 * sf(8.94427191, df=4) = 0.0008581
    """
    paired_diff = PairedMetricDifference(
        metric_name="accuracy",
        direction=MetricDirection.HIGHER_IS_BETTER,
        raw_deltas=[0.05, 0.05, 0.07, 0.06, 0.07],
        mean_delta=0.06,
        std_delta=0.01,
        expected_folds=5,
        paired_folds=5,
        usable_folds=5,
        is_usable=True,
    )

    test_train_ratio = 0.25  # 20 val / 80 train

    res = nadeau_bengio_corrected_t_test(paired_diff, test_train_ratio=test_train_ratio, alpha=0.05)

    assert res.status == TestStatus.COMPLETED
    assert res.degrees_of_freedom == 4.0
    assert res.statistic == pytest.approx(8.944272, abs=1e-4)
    assert res.p_value == pytest.approx(0.000858, abs=1e-5)
    assert res.statistically_significant is True
    assert res.alternative_hypothesis == "two-sided"
    assert any("Nadeau, C., & Bengio, Y. (2003)" in a for a in res.assumptions_and_limitations)


def test_t_test_identical_scores_zero_variance():
    """Identical scores on all folds yield t=0.0, p=1.0, not significant."""
    paired_diff = PairedMetricDifference(
        metric_name="accuracy",
        direction=MetricDirection.HIGHER_IS_BETTER,
        raw_deltas=[0.0, 0.0, 0.0, 0.0, 0.0],
        mean_delta=0.0,
        std_delta=0.0,
        expected_folds=5,
        paired_folds=5,
        usable_folds=5,
        is_usable=True,
    )

    res = nadeau_bengio_corrected_t_test(paired_diff, test_train_ratio=0.25)
    assert res.status == TestStatus.COMPLETED
    assert res.statistic == 0.0
    assert res.p_value == 1.0
    assert res.statistically_significant is False
    assert any("Zero variance" in w for w in res.warnings)


def test_t_test_zero_variance_nonzero_mean():
    """Constant non-zero delta across all folds produces unavailable result (SE=0)."""
    paired_diff = PairedMetricDifference(
        metric_name="accuracy",
        direction=MetricDirection.HIGHER_IS_BETTER,
        raw_deltas=[0.05, 0.05, 0.05, 0.05],
        mean_delta=0.05,
        std_delta=0.0,
        expected_folds=4,
        paired_folds=4,
        usable_folds=4,
        is_usable=True,
    )

    res = nadeau_bengio_corrected_t_test(paired_diff, test_train_ratio=0.33)
    assert res.status == TestStatus.UNAVAILABLE
    assert "standard error is zero" in res.unavailable_reason


def test_t_test_invalid_ratio_unavailable():
    """Negative or zero test/train ratio returns UNAVAILABLE status."""
    paired_diff = PairedMetricDifference(
        metric_name="accuracy",
        direction=MetricDirection.HIGHER_IS_BETTER,
        raw_deltas=[0.05, 0.06],
        expected_folds=2,
        paired_folds=2,
        usable_folds=2,
        is_usable=True,
    )

    res = nadeau_bengio_corrected_t_test(paired_diff, test_train_ratio=-0.5)
    assert res.status == TestStatus.UNAVAILABLE
    assert "Invalid test/train" in res.unavailable_reason


def test_t_test_unusable_differences():
    """Unusable paired differences return UNAVAILABLE status."""
    paired_diff = PairedMetricDifference(
        metric_name="accuracy",
        direction=MetricDirection.HIGHER_IS_BETTER,
        raw_deltas=[0.05, None, 0.06],
        expected_folds=3,
        paired_folds=3,
        usable_folds=2,
        is_usable=False,
        unusable_reason="Fold 1 was undefined",
    )

    res = nadeau_bengio_corrected_t_test(paired_diff, test_train_ratio=0.5)
    assert res.status == TestStatus.UNAVAILABLE
    assert "Fold 1 was undefined" in res.unavailable_reason


# ==========================================
# 4. Wilcoxon Signed-Rank Test Tests
# ==========================================

def test_wilcoxon_valid_paired_differences():
    """Wilcoxon test executes with valid positive differences."""
    # 6 folds with strictly positive differences
    deltas = [0.05, 0.08, 0.03, 0.06, 0.07, 0.04]
    paired_diff = PairedMetricDifference(
        metric_name="accuracy",
        direction=MetricDirection.HIGHER_IS_BETTER,
        raw_deltas=deltas,
        expected_folds=6,
        paired_folds=6,
        usable_folds=6,
        is_usable=True,
    )

    res = wilcoxon_signed_rank_test(paired_diff)
    assert res.status == TestStatus.COMPLETED
    assert res.statistic is not None
    assert res.p_value is not None
    assert res.alternative_hypothesis == "two-sided"
    assert any("independence assumption" in a for a in res.assumptions_and_limitations)


def test_wilcoxon_all_zero_differences():
    """Wilcoxon handles all-zero differences by returning stat=0, p=1.0."""
    paired_diff = PairedMetricDifference(
        metric_name="accuracy",
        direction=MetricDirection.HIGHER_IS_BETTER,
        raw_deltas=[0.0, 0.0, 0.0, 0.0],
        expected_folds=4,
        paired_folds=4,
        usable_folds=4,
        is_usable=True,
    )

    res = wilcoxon_signed_rank_test(paired_diff)
    assert res.status == TestStatus.COMPLETED
    assert res.statistic == 0.0
    assert res.p_value == 1.0
    assert res.statistically_significant is False


def test_wilcoxon_tied_differences_and_small_sample():
    """Tied differences and small sample size flag appropriate caveats."""
    paired_diff = PairedMetricDifference(
        metric_name="accuracy",
        direction=MetricDirection.HIGHER_IS_BETTER,
        raw_deltas=[0.05, -0.05, 0.05],
        expected_folds=3,
        paired_folds=3,
        usable_folds=3,
        is_usable=True,
    )

    res = wilcoxon_signed_rank_test(paired_diff)
    assert res.status == TestStatus.COMPLETED
    assert any("very small" in w for w in res.warnings)


# ==========================================
# 5. Integration and End-to-End Evaluation Tests
# ==========================================

def test_evaluate_classification_includes_comparison_report():
    """End-to-end evaluation automatically produces paired comparison report."""
    from sklearn.datasets import make_classification
    import pandas as pd

    X_raw, y_raw = make_classification(
        n_samples=60, n_features=6, n_informative=4, n_redundant=1, n_classes=2, weights=[0.6, 0.4], random_state=42
    )
    df = pd.DataFrame(X_raw, columns=[f"f_{i}" for i in range(6)])
    df["target"] = y_raw

    res = evaluate_classification(df, target_column="target", num_folds=3, random_state=42)

    assert res.comparison_report is not None
    cr = res.comparison_report
    assert cr.candidate_model_name == "Candidate: LogisticRegression"
    assert cr.baseline_model_name == "Baseline: DummyClassifier"
    assert cr.num_folds == 3
    assert cr.dataset_checksum == res.dataset_metadata.checksum_sha256

    # Check accuracy comparison
    acc_comp = cr.metric_comparisons["accuracy"]
    assert acc_comp.direction == MetricDirection.HIGHER_IS_BETTER
    assert len(acc_comp.paired_differences.raw_deltas) == 3
    assert acc_comp.corrected_resampled_t_test is not None
    assert acc_comp.corrected_resampled_t_test.status == TestStatus.COMPLETED

    # Check that macro_f1 on baseline is undefined and comparison gracefully reports unavailable
    f1_comp = cr.metric_comparisons["macro_f1"]
    assert f1_comp.paired_differences.is_usable is False
    assert f1_comp.corrected_resampled_t_test.status == TestStatus.UNAVAILABLE
    assert "Insufficient usable paired observations" in f1_comp.corrected_resampled_t_test.unavailable_reason


# ==========================================
# 6. Milestone 2.3.1 Regression Tests
# ==========================================

def test_pairing_fails_on_duplicate_fold_ids():
    """Duplicate fold IDs in candidate or baseline evaluation raise IncompatibleEvaluationPairError."""
    meta = make_dummy_metadata()
    repro = make_dummy_reproducibility(num_folds=3)

    # Candidate has duplicate fold index 1
    cand_folds = [
        FoldResult(fold_index=0, train_indices_count=80, val_indices_count=20, metrics={}, confusion_matrix=[], class_labels=["0", "1"], fit_duration_seconds=0.01, predict_duration_seconds=0.01),
        FoldResult(fold_index=1, train_indices_count=80, val_indices_count=20, metrics={}, confusion_matrix=[], class_labels=["0", "1"], fit_duration_seconds=0.01, predict_duration_seconds=0.01),
        FoldResult(fold_index=1, train_indices_count=80, val_indices_count=20, metrics={}, confusion_matrix=[], class_labels=["0", "1"], fit_duration_seconds=0.01, predict_duration_seconds=0.01),
    ]
    base_folds = make_mock_fold_results({"acc": [0.8] * 3}, num_folds=3)

    cand = ModelEvaluationResult(model_name="C", model_type="l", fold_results=cand_folds, metric_summaries={}, aggregate_confusion_matrix=[])
    base = ModelEvaluationResult(model_name="B", model_type="d", fold_results=base_folds, metric_summaries={}, aggregate_confusion_matrix=[])

    with pytest.raises(IncompatibleEvaluationPairError, match="duplicate fold IDs"):
        validate_evaluation_compatibility(cand, base, meta, repro)


def test_pairing_fails_on_mismatched_or_missing_fold_ids():
    """Fold IDs not covering [0..K-1] raise IncompatibleEvaluationPairError."""
    meta = make_dummy_metadata()
    repro = make_dummy_reproducibility(num_folds=3)

    cand_folds = [
        FoldResult(fold_index=0, train_indices_count=80, val_indices_count=20, metrics={}, confusion_matrix=[], class_labels=["0", "1"], fit_duration_seconds=0.01, predict_duration_seconds=0.01),
        FoldResult(fold_index=1, train_indices_count=80, val_indices_count=20, metrics={}, confusion_matrix=[], class_labels=["0", "1"], fit_duration_seconds=0.01, predict_duration_seconds=0.01),
        FoldResult(fold_index=5, train_indices_count=80, val_indices_count=20, metrics={}, confusion_matrix=[], class_labels=["0", "1"], fit_duration_seconds=0.01, predict_duration_seconds=0.01),
    ]
    base_folds = make_mock_fold_results({"acc": [0.8] * 3}, num_folds=3)

    cand = ModelEvaluationResult(model_name="C", model_type="l", fold_results=cand_folds, metric_summaries={}, aggregate_confusion_matrix=[])
    base = ModelEvaluationResult(model_name="B", model_type="d", fold_results=base_folds, metric_summaries={}, aggregate_confusion_matrix=[])

    with pytest.raises(IncompatibleEvaluationPairError, match="expected range"):
        validate_evaluation_compatibility(cand, base, meta, repro)


def test_pairing_succeeds_with_reordered_fold_results():
    """Candidate and baseline fold results in different list orders align correctly by fold_index."""
    cand_folds = [
        FoldResult(fold_index=2, train_indices_count=80, val_indices_count=20,
                   metrics={"accuracy": MetricValue(value=0.90, is_undefined=False)},
                   confusion_matrix=[], class_labels=["0", "1"], fit_duration_seconds=0.01, predict_duration_seconds=0.01, partition_fingerprint="fp_2"),
        FoldResult(fold_index=0, train_indices_count=80, val_indices_count=20,
                   metrics={"accuracy": MetricValue(value=0.80, is_undefined=False)},
                   confusion_matrix=[], class_labels=["0", "1"], fit_duration_seconds=0.01, predict_duration_seconds=0.01, partition_fingerprint="fp_0"),
        FoldResult(fold_index=1, train_indices_count=80, val_indices_count=20,
                   metrics={"accuracy": MetricValue(value=0.85, is_undefined=False)},
                   confusion_matrix=[], class_labels=["0", "1"], fit_duration_seconds=0.01, predict_duration_seconds=0.01, partition_fingerprint="fp_1"),
    ]
    base_folds = [
        FoldResult(fold_index=0, train_indices_count=80, val_indices_count=20,
                   metrics={"accuracy": MetricValue(value=0.70, is_undefined=False)},
                   confusion_matrix=[], class_labels=["0", "1"], fit_duration_seconds=0.01, predict_duration_seconds=0.01, partition_fingerprint="fp_0"),
        FoldResult(fold_index=1, train_indices_count=80, val_indices_count=20,
                   metrics={"accuracy": MetricValue(value=0.75, is_undefined=False)},
                   confusion_matrix=[], class_labels=["0", "1"], fit_duration_seconds=0.01, predict_duration_seconds=0.01, partition_fingerprint="fp_1"),
        FoldResult(fold_index=2, train_indices_count=80, val_indices_count=20,
                   metrics={"accuracy": MetricValue(value=0.80, is_undefined=False)},
                   confusion_matrix=[], class_labels=["0", "1"], fit_duration_seconds=0.01, predict_duration_seconds=0.01, partition_fingerprint="fp_2"),
    ]

    diff = compute_paired_metric_differences(cand_folds, base_folds, "accuracy")
    assert diff.is_usable is True
    # Folds 0, 1, 2 deltas: (0.80-0.70=0.10, 0.85-0.75=0.10, 0.90-0.80=0.10)
    assert diff.raw_deltas == [0.10, 0.10, 0.10]
    assert diff.mean_delta == pytest.approx(0.10)


def test_pairing_fails_on_partition_fingerprint_mismatch():
    """Equal partition counts with different partition fingerprints raise IncompatibleEvaluationPairError."""
    meta = make_dummy_metadata()
    repro = make_dummy_reproducibility(num_folds=3)

    cand_folds = make_mock_fold_results({"acc": [0.8] * 3}, num_folds=3, fingerprint_suffix="_cand")
    base_folds = make_mock_fold_results({"acc": [0.8] * 3}, num_folds=3, fingerprint_suffix="_base")

    cand = ModelEvaluationResult(model_name="C", model_type="l", fold_results=cand_folds, metric_summaries={}, aggregate_confusion_matrix=[])
    base = ModelEvaluationResult(model_name="B", model_type="d", fold_results=base_folds, metric_summaries={}, aggregate_confusion_matrix=[])

    with pytest.raises(IncompatibleEvaluationPairError, match="partition fingerprint mismatch"):
        validate_evaluation_compatibility(cand, base, meta, repro)


def test_self_comparison_same_object_rejected():
    """Attempting to compare an evaluation with itself raises IncompatibleEvaluationPairError."""
    meta = make_dummy_metadata()
    repro = make_dummy_reproducibility(num_folds=3)
    cand_folds = make_mock_fold_results({"acc": [0.8] * 3}, num_folds=3)
    cand = ModelEvaluationResult(model_name="C", model_type="l", fold_results=cand_folds, metric_summaries={}, aggregate_confusion_matrix=[])

    with pytest.raises(IncompatibleEvaluationPairError, match="Cannot compare model evaluation against itself"):
        validate_evaluation_compatibility(cand, cand, meta, repro)


def test_unsupported_protocol_split_strategy_rejected():
    """Evaluating a split strategy other than stratified_k_fold or k_fold is rejected."""
    meta = make_dummy_metadata()
    repro = ReproducibilityMetadata(
        random_seed=42,
        split_strategy="time_series_split",
        num_folds=3,
        package_versions={"numpy": "2.0.0"},
        platform_system="Windows",
        python_version="3.11.9",
    )
    cand_folds = make_mock_fold_results({"acc": [0.8] * 3}, num_folds=3)
    base_folds = make_mock_fold_results({"acc": [0.8] * 3}, num_folds=3)

    cand = ModelEvaluationResult(model_name="C", model_type="l", fold_results=cand_folds, metric_summaries={}, aggregate_confusion_matrix=[])
    base = ModelEvaluationResult(model_name="B", model_type="d", fold_results=base_folds, metric_summaries={}, aggregate_confusion_matrix=[])

    with pytest.raises(IncompatibleEvaluationPairError, match="unsupported for paired comparison"):
        validate_evaluation_compatibility(cand, base, meta, repro)


def test_t_test_rejects_unsupported_split_strategy():
    """Nadeau-Bengio corrected t-test returns UNAVAILABLE when given an unsupported split strategy."""
    paired_diff = PairedMetricDifference(
        metric_name="acc",
        direction=MetricDirection.HIGHER_IS_BETTER,
        raw_deltas=[0.05, 0.05, 0.05],
        expected_folds=3,
        paired_folds=3,
        usable_folds=3,
        is_usable=True,
    )
    res = nadeau_bengio_corrected_t_test(paired_diff, test_train_ratio=0.5, split_strategy="rolling_window")
    assert res.status == TestStatus.UNAVAILABLE
    assert "Unsupported evaluation protocol" in res.unavailable_reason


def test_infinite_metric_values_handled_as_unusable_without_nan_status():
    """Infinite metric values (+inf / -inf) are marked unusable and never return COMPLETED with NaN."""
    cand_folds = [
        FoldResult(fold_index=0, train_indices_count=80, val_indices_count=20,
                   metrics={"loss": MetricValue(value=float("inf"), is_undefined=False)},
                   confusion_matrix=[], class_labels=["0", "1"], fit_duration_seconds=0.01, predict_duration_seconds=0.01),
        FoldResult(fold_index=1, train_indices_count=80, val_indices_count=20,
                   metrics={"loss": MetricValue(value=0.50, is_undefined=False)},
                   confusion_matrix=[], class_labels=["0", "1"], fit_duration_seconds=0.01, predict_duration_seconds=0.01),
        FoldResult(fold_index=2, train_indices_count=80, val_indices_count=20,
                   metrics={"loss": MetricValue(value=0.45, is_undefined=False)},
                   confusion_matrix=[], class_labels=["0", "1"], fit_duration_seconds=0.01, predict_duration_seconds=0.01),
    ]
    base_folds = make_mock_fold_results({"loss": [0.40, 0.40, 0.40]}, num_folds=3)

    diff = compute_paired_metric_differences(cand_folds, base_folds, "loss")
    assert diff.raw_deltas[0] is None  # Inf rejected
    assert diff.is_usable is False

    t_res = nadeau_bengio_corrected_t_test(diff, test_train_ratio=0.25)
    assert t_res.status == TestStatus.UNAVAILABLE
    assert not math.isnan(t_res.statistic) if t_res.statistic is not None else True
    assert t_res.p_value is None

    w_res = wilcoxon_signed_rank_test(diff)
    assert w_res.status == TestStatus.UNAVAILABLE
    assert w_res.p_value is None


def test_wilcoxon_hand_checkable_exact_statistic_and_pvalue():
    """Exact hand-calculated verification of two-sided Wilcoxon signed-rank test.

    Setup:
      n = 6 paired observations, all positive deltas: [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
      All positive differences -> W = min(sum(pos_ranks), sum(neg_ranks)) = 0.0
      Two-sided p-value: 2 * (1/2)^6 = 2 * 1/64 = 0.03125
    """
    paired_diff = PairedMetricDifference(
        metric_name="accuracy",
        direction=MetricDirection.HIGHER_IS_BETTER,
        raw_deltas=[0.05, 0.08, 0.03, 0.06, 0.07, 0.04],
        expected_folds=6,
        paired_folds=6,
        usable_folds=6,
        is_usable=True,
    )
    res = wilcoxon_signed_rank_test(paired_diff, alpha=0.05)
    assert res.status == TestStatus.COMPLETED
    assert res.statistic == pytest.approx(0.0)
    assert res.p_value == pytest.approx(0.03125, abs=1e-5)
    assert res.statistically_significant is True


def test_wilcoxon_low_power_warning_at_n5():
    """For n=5, two-sided Wilcoxon cannot mathematically reach p < 0.05, so low-power warning is emitted."""
    paired_diff = PairedMetricDifference(
        metric_name="accuracy",
        direction=MetricDirection.HIGHER_IS_BETTER,
        raw_deltas=[0.05, 0.08, 0.03, 0.06, 0.07],
        expected_folds=5,
        paired_folds=5,
        usable_folds=5,
        is_usable=True,
    )
    res = wilcoxon_signed_rank_test(paired_diff, alpha=0.05)
    assert res.status == TestStatus.COMPLETED
    # At n=5, min p-value is 0.0625
    assert res.p_value == pytest.approx(0.0625, abs=1e-4)
    assert res.statistically_significant is False
    assert any("cannot mathematically achieve p < 0.05" in w for w in res.warnings)


def test_evaluate_classification_preserves_models_on_comparison_failure(monkeypatch):
    """If comparison validation fails in evaluate_classification, model results are preserved with diagnostic warning."""
    import pandas as pd
    from sklearn.datasets import make_classification
    import mlforge.engine.evaluator as evaluator_module

    X_raw, y_raw = make_classification(n_samples=40, n_features=4, n_classes=2, random_state=42)
    df = pd.DataFrame(X_raw, columns=[f"f_{i}" for i in range(4)])
    df["target"] = y_raw

    # Simulate an incompatible evaluation pair error during comparison
    def mock_compare(*args, **kwargs):
        raise IncompatibleEvaluationPairError("Simulated pairing incompatibility.")

    monkeypatch.setattr("mlforge.engine.statistics.compare_model_evaluations", mock_compare)

    res = evaluate_classification(df, target_column="target", num_folds=3, random_state=42)

    assert res.candidate_result is not None
    assert res.baseline_result is not None
    assert res.comparison_report is None
    assert any("Statistical comparison could not be performed: Simulated pairing incompatibility." in w for w in res.warnings)


# ==========================================
# 7. Milestone 2.3.2 Fingerprint Consistency Tests
# ==========================================

def test_pairing_fails_when_candidate_has_fingerprints_and_baseline_has_none():
    """Asymmetric fingerprints (candidate present, baseline absent) raise IncompatibleEvaluationPairError."""
    meta = make_dummy_metadata()
    repro = make_dummy_reproducibility(num_folds=3)

    cand_folds = make_mock_fold_results({"acc": [0.8] * 3}, num_folds=3, omit_fingerprints=False)
    base_folds = make_mock_fold_results({"acc": [0.8] * 3}, num_folds=3, omit_fingerprints=True)

    cand = ModelEvaluationResult(model_name="C", model_type="l", fold_results=cand_folds, metric_summaries={}, aggregate_confusion_matrix=[])
    base = ModelEvaluationResult(model_name="B", model_type="d", fold_results=base_folds, metric_summaries={}, aggregate_confusion_matrix=[])

    with pytest.raises(IncompatibleEvaluationPairError, match="asymmetric partition fingerprints"):
        validate_evaluation_compatibility(cand, base, meta, repro)


def test_pairing_fails_when_baseline_has_fingerprints_and_candidate_has_none():
    """Asymmetric fingerprints (candidate absent, baseline present) raise IncompatibleEvaluationPairError."""
    meta = make_dummy_metadata()
    repro = make_dummy_reproducibility(num_folds=3)

    cand_folds = make_mock_fold_results({"acc": [0.8] * 3}, num_folds=3, omit_fingerprints=True)
    base_folds = make_mock_fold_results({"acc": [0.8] * 3}, num_folds=3, omit_fingerprints=False)

    cand = ModelEvaluationResult(model_name="C", model_type="l", fold_results=cand_folds, metric_summaries={}, aggregate_confusion_matrix=[])
    base = ModelEvaluationResult(model_name="B", model_type="d", fold_results=base_folds, metric_summaries={}, aggregate_confusion_matrix=[])

    with pytest.raises(IncompatibleEvaluationPairError, match="asymmetric partition fingerprints"):
        validate_evaluation_compatibility(cand, base, meta, repro)


def test_pairing_succeeds_when_all_fingerprints_match():
    """Matching fingerprints across all folds complete without unverified warnings."""
    meta = make_dummy_metadata()
    repro = make_dummy_reproducibility(num_folds=3)

    cand_folds = make_mock_fold_results({"accuracy": [0.85, 0.86, 0.87]}, num_folds=3)
    base_folds = make_mock_fold_results({"accuracy": [0.80, 0.81, 0.82]}, num_folds=3)

    cand = ModelEvaluationResult(model_name="Candidate", model_type="l", fold_results=cand_folds, metric_summaries={}, aggregate_confusion_matrix=[])
    base = ModelEvaluationResult(model_name="Baseline", model_type="d", fold_results=base_folds, metric_summaries={}, aggregate_confusion_matrix=[])

    report = compare_model_evaluations(cand, base, meta, repro)
    assert report is not None
    assert not any("Partition verification notice" in w for w in report.warnings)


def test_pairing_fails_when_single_fold_fingerprint_mismatches():
    """If 2 folds match but 1 fold mismatches, IncompatibleEvaluationPairError is raised for that fold."""
    meta = make_dummy_metadata()
    repro = make_dummy_reproducibility(num_folds=3)

    cand_folds = make_mock_fold_results({"acc": [0.8] * 3}, num_folds=3)
    base_folds = make_mock_fold_results({"acc": [0.8] * 3}, num_folds=3)
    # Tamper fold 2 fingerprint on baseline
    base_folds[2] = FoldResult(
        fold_index=2, train_indices_count=80, val_indices_count=20, metrics={}, confusion_matrix=[],
        class_labels=["0", "1"], fit_duration_seconds=0.01, predict_duration_seconds=0.01,
        partition_fingerprint="fp_different"
    )

    cand = ModelEvaluationResult(model_name="C", model_type="l", fold_results=cand_folds, metric_summaries={}, aggregate_confusion_matrix=[])
    base = ModelEvaluationResult(model_name="B", model_type="d", fold_results=base_folds, metric_summaries={}, aggregate_confusion_matrix=[])

    with pytest.raises(IncompatibleEvaluationPairError, match="Fold 2 partition fingerprint mismatch"):
        validate_evaluation_compatibility(cand, base, meta, repro)


def test_pairing_warns_when_both_sides_have_no_fingerprints():
    """When both candidate and baseline omit fingerprints, comparison proceeds with an unverified notice."""
    meta = make_dummy_metadata()
    repro = make_dummy_reproducibility(num_folds=3)

    cand_folds = make_mock_fold_results({"accuracy": [0.85, 0.86, 0.87]}, num_folds=3, omit_fingerprints=True)
    base_folds = make_mock_fold_results({"accuracy": [0.80, 0.81, 0.82]}, num_folds=3, omit_fingerprints=True)

    cand = ModelEvaluationResult(model_name="Candidate", model_type="l", fold_results=cand_folds, metric_summaries={}, aggregate_confusion_matrix=[])
    base = ModelEvaluationResult(model_name="Baseline", model_type="d", fold_results=base_folds, metric_summaries={}, aggregate_confusion_matrix=[])

    report = compare_model_evaluations(cand, base, meta, repro)
    assert report is not None
    assert any("pairing was verified by sample counts only" in w for w in report.warnings)


def test_pairing_partial_fingerprints_across_folds_handled_consistently():
    """Consistent partial fingerprinting across folds cannot bypass checks or suppress warnings."""
    meta = make_dummy_metadata()
    repro = make_dummy_reproducibility(num_folds=3)

    # Fold 0 has matching fingerprints on both; Fold 1 and 2 omit fingerprints on both
    cand_folds = [
        FoldResult(fold_index=0, train_indices_count=80, val_indices_count=20, metrics={"accuracy": MetricValue(value=0.85)}, confusion_matrix=[], class_labels=["0", "1"], fit_duration_seconds=0.01, predict_duration_seconds=0.01, partition_fingerprint="fp_0"),
        FoldResult(fold_index=1, train_indices_count=80, val_indices_count=20, metrics={"accuracy": MetricValue(value=0.86)}, confusion_matrix=[], class_labels=["0", "1"], fit_duration_seconds=0.01, predict_duration_seconds=0.01, partition_fingerprint=None),
        FoldResult(fold_index=2, train_indices_count=80, val_indices_count=20, metrics={"accuracy": MetricValue(value=0.87)}, confusion_matrix=[], class_labels=["0", "1"], fit_duration_seconds=0.01, predict_duration_seconds=0.01, partition_fingerprint=None),
    ]
    base_folds = [
        FoldResult(fold_index=0, train_indices_count=80, val_indices_count=20, metrics={"accuracy": MetricValue(value=0.80)}, confusion_matrix=[], class_labels=["0", "1"], fit_duration_seconds=0.01, predict_duration_seconds=0.01, partition_fingerprint="fp_0"),
        FoldResult(fold_index=1, train_indices_count=80, val_indices_count=20, metrics={"accuracy": MetricValue(value=0.81)}, confusion_matrix=[], class_labels=["0", "1"], fit_duration_seconds=0.01, predict_duration_seconds=0.01, partition_fingerprint=None),
        FoldResult(fold_index=2, train_indices_count=80, val_indices_count=20, metrics={"accuracy": MetricValue(value=0.82)}, confusion_matrix=[], class_labels=["0", "1"], fit_duration_seconds=0.01, predict_duration_seconds=0.01, partition_fingerprint=None),
    ]

    cand = ModelEvaluationResult(model_name="Candidate", model_type="l", fold_results=cand_folds, metric_summaries={}, aggregate_confusion_matrix=[])
    base = ModelEvaluationResult(model_name="Baseline", model_type="d", fold_results=base_folds, metric_summaries={}, aggregate_confusion_matrix=[])

    report = compare_model_evaluations(cand, base, meta, repro)
    assert report is not None
    # Must report partial verification notice (1 of 3 folds)
    assert any("Only 1 of 3 folds contained cryptographic partition fingerprints" in w for w in report.warnings)
