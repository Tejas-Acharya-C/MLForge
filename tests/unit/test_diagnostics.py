"""Unit Tests for Threshold Analysis, Error Diagnostics, and Subgroup Breakdown (Milestone 2.4)."""

import numpy as np
import pandas as pd
import pytest

from mlforge.engine.diagnostics import (
    DEFAULT_THRESHOLDS,
    aggregate_error_diagnostics,
    analyze_classification_thresholds,
    compute_fold_error_diagnostic,
    compute_subgroup_diagnostics,
)
from mlforge.engine.evaluator import evaluate_classification


def test_hand_calculated_threshold_metrics():
    """Verify threshold metrics against exact hand-calculated reference values at threshold 0.5.

    Scenario:
    N = 4, y_true = [1, 1, 0, 0], y_prob = [0.9, 0.4, 0.6, 0.1].
    Threshold = 0.5:
    y_pred_bin = [1, 0, 1, 0] (prob >= 0.5)

    Index 0: y=1, pred=1 -> TP
    Index 1: y=1, pred=0 -> FN
    Index 2: y=0, pred=1 -> FP
    Index 3: y=0, pred=0 -> TN

    TP = 1, FP = 1, TN = 1, FN = 1.
    Precision = 1 / (1 + 1) = 0.5.
    Recall = 1 / (1 + 1) = 0.5.
    Specificity = 1 / (1 + 1) = 0.5.
    F1 = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5.
    Balanced Accuracy = (0.5 + 0.5) / 2 = 0.5.
    Predicted Positive Rate = (1 + 1) / 4 = 0.5.
    """
    y_true = [1, 1, 0, 0]
    y_prob = [0.9, 0.4, 0.6, 0.1]

    res = analyze_classification_thresholds(
        y_true=y_true,
        y_prob=y_prob,
        positive_label=1,
        thresholds=[0.5],
    )

    assert len(res.threshold_metrics) == 1
    m = res.threshold_metrics[0]
    assert m.threshold == 0.5
    assert m.evaluated_samples == 4
    assert m.true_positives == 1
    assert m.false_positives == 1
    assert m.true_negatives == 1
    assert m.false_negatives == 1
    assert m.precision == 0.5
    assert m.recall == 0.5
    assert m.specificity == 0.5
    assert m.f1_score == 0.5
    assert m.balanced_accuracy == 0.5
    assert m.predicted_positive_rate == 0.5
    assert m.precision_undefined_reason is None
    assert m.recall_undefined_reason is None


def test_threshold_boundary_cases_0_and_1():
    """Verify threshold analysis at boundary conditions 0.0 and 1.0.

    At threshold 0.0: all samples predicted positive.
    At threshold 1.0: no samples predicted positive (unless prob == 1.0).
    Precision must be undefined (None) when no samples are predicted positive.
    """
    y_true = [1, 1, 0, 0]
    y_prob = [0.9, 0.8, 0.3, 0.2]  # No sample has prob == 1.0

    res = analyze_classification_thresholds(
        y_true=y_true,
        y_prob=y_prob,
        positive_label=1,
        thresholds=[0.0, 1.0],
    )

    t0 = res.threshold_metrics[0]
    assert t0.threshold == 0.0
    assert t0.true_positives == 2
    assert t0.false_positives == 2
    assert t0.true_negatives == 0
    assert t0.false_negatives == 0
    assert t0.recall == 1.0
    assert t0.specificity == 0.0
    assert t0.predicted_positive_rate == 1.0
    assert t0.precision == 0.5
    assert t0.balanced_accuracy == 0.5

    t1 = res.threshold_metrics[1]
    assert t1.threshold == 1.0
    assert t1.true_positives == 0
    assert t1.false_positives == 0
    assert t1.true_negatives == 2
    assert t1.false_negatives == 2
    assert t1.predicted_positive_rate == 0.0
    assert t1.recall == 0.0
    assert t1.specificity == 1.0
    # Precision must be undefined (never coerced to 0.0)
    assert t1.precision is None
    assert t1.precision_undefined_reason is not None
    # F1 must be undefined
    assert t1.f1_score is None
    assert t1.f1_undefined_reason is not None
    assert t1.balanced_accuracy == 0.5


def test_zero_denominator_metrics_remain_undefined():
    """Single-class true targets must yield undefined recall or specificity without crashing."""
    # All true samples are 0 (no actual positives)
    y_true = [0, 0, 0, 0]
    y_prob = [0.2, 0.4, 0.6, 0.8]

    res = analyze_classification_thresholds(
        y_true=y_true,
        y_prob=y_prob,
        positive_label=1,
        thresholds=[0.5],
    )

    m = res.threshold_metrics[0]
    # Recall denominator (TP + FN) is 0 -> undefined
    assert m.recall is None
    assert m.recall_undefined_reason is not None
    assert m.balanced_accuracy is None
    assert m.balanced_accuracy_undefined_reason is not None


def test_invalid_thresholds_rejected():
    """Invalid thresholds (out of range, NaN, non-finite, empty) must raise ValueError."""
    y_true = [0, 1]
    y_prob = [0.2, 0.8]

    with pytest.raises(ValueError, match="Each threshold must be a finite float in"):
        analyze_classification_thresholds(y_true, y_prob, thresholds=[-0.1])

    with pytest.raises(ValueError, match="Each threshold must be a finite float in"):
        analyze_classification_thresholds(y_true, y_prob, thresholds=[1.05])

    with pytest.raises(ValueError, match="Each threshold must be a finite float in"):
        analyze_classification_thresholds(y_true, y_prob, thresholds=[np.nan])

    with pytest.raises(ValueError, match="Threshold list must contain at least one threshold"):
        analyze_classification_thresholds(y_true, y_prob, thresholds=[])


def test_fold_error_diagnostic():
    """Verify fold error diagnostic computes accurate error counts, rates, FP, and FN."""
    y_true = [1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 1, 1]  # Errors at index 1 (FN) and index 3 (FP)

    diag = compute_fold_error_diagnostic(
        y_true=y_true,
        y_pred=y_pred,
        fold_index=0,
        positive_label=1,
    )

    assert diag.fold_index == 0
    assert diag.total_samples == 5
    assert diag.error_count == 2
    assert diag.error_rate == 0.4
    assert diag.false_positives == 1
    assert diag.false_negatives == 1


def test_subgroup_diagnostics_support_threshold():
    """Subgroups below minimum sample support must be flagged as unusable without fabricated metrics."""
    # 25 samples
    # Group A: 20 samples, 2 errors
    # Group B: 3 samples, 1 error (< min_subgroup_size=5)
    # Group C: 7 samples, all actual class 0
    meta = pd.DataFrame({
        "segment": ["A"] * 20 + ["B"] * 3 + ["C"] * 7,
    })
    # Targets
    y_true = np.array([1] * 10 + [0] * 10 + [1, 0, 0] + [0] * 7)
    # Predictions: 2 errors in A (indices 0 and 10), 1 error in B (index 20), 0 in C
    y_pred = y_true.copy()
    y_pred[0] = 0   # FN
    y_pred[10] = 1  # FP
    y_pred[20] = 0  # FN

    diags = compute_subgroup_diagnostics(
        y_true=y_true,
        y_pred=y_pred,
        metadata_df=meta,
        subgroup_columns=["segment"],
        positive_label=1,
        min_subgroup_size=5,
    )

    assert "segment" in diags
    seg_list = diags["segment"]
    by_val = {d.subgroup_value: d for d in seg_list}

    # Group A
    assert by_val["A"].sample_count == 20
    assert by_val["A"].error_count == 2
    assert by_val["A"].error_rate == 0.1
    assert by_val["A"].is_usable is True
    assert by_val["A"].accuracy == 0.9

    # Group B (< 5 samples)
    assert by_val["B"].sample_count == 3
    assert by_val["B"].error_count == 1
    assert by_val["B"].is_usable is False
    assert by_val["B"].accuracy is None
    assert "below minimum support threshold" in by_val["B"].unusable_reason

    # Group C (>= 5 samples, but single class)
    assert by_val["C"].sample_count == 7
    assert by_val["C"].error_count == 0
    assert by_val["C"].is_usable is True
    assert by_val["C"].accuracy == 1.0
    assert by_val["C"].precision is None  # single-class
    assert "single class" in by_val["C"].unusable_reason


def test_subgroup_diagnostics_missing_values_categorized():
    """Null/NaN values in metadata features must be assigned to '<MISSING>' category."""
    meta = pd.DataFrame({"category": ["X", None, "X", None, "X", "X"]})
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [1, 1, 0, 0, 1, 0]

    diags = compute_subgroup_diagnostics(
        y_true=y_true,
        y_pred=y_pred,
        metadata_df=meta,
        subgroup_columns=["category"],
        min_subgroup_size=2,
    )

    by_val = {d.subgroup_value: d for d in diags["category"]}
    assert "<MISSING>" in by_val
    assert by_val["<MISSING>"].sample_count == 2
    assert by_val["<MISSING>"].is_usable is True


def test_end_to_end_evaluator_integration():
    """Verify that evaluate_classification produces valid calibration, threshold, and subgroup outputs."""
    df = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        "f2": [0.5, 1.5, 0.2, 2.5, 0.1, 1.1, 0.3, 2.3, 0.4, 1.4, 0.2, 2.2],
        "region": ["East", "West", "East", "West", "East", "West", "East", "West", "East", "West", "East", "West"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })

    result = evaluate_classification(
        df=df,
        target_column="target",
        num_folds=3,
        subgroup_columns=["region"],
        random_state=42,
    )

    cand = result.candidate_result
    # Calibration
    assert cand.calibration is not None
    assert cand.calibration.is_usable is True
    assert cand.calibration.expected_calibration_error is not None
    assert cand.calibration.brier_score is not None

    # Threshold analysis
    assert cand.threshold_analysis is not None
    assert len(cand.threshold_analysis.threshold_metrics) == len(DEFAULT_THRESHOLDS)

    # Error diagnostics
    assert cand.error_diagnostics is not None
    assert len(cand.error_diagnostics.fold_diagnostics) == 3
    assert cand.error_diagnostics.subgroup_diagnostics is not None
    assert "region" in cand.error_diagnostics.subgroup_diagnostics

    # Folds also have fold-level calibration and diagnostics
    for fr in cand.fold_results:
        assert fr.error_diagnostic is not None
        assert fr.calibration is not None
        assert fr.threshold_analysis is not None
        assert fr.val_indices is not None
