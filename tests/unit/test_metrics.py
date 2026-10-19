"""Unit tests for classification metrics and aggregation.

Covers test cases TC-MET-01, TC-MET-02, TC-MET-03, TC-AGG-01, TC-AGG-02.
"""

import numpy as np
import pytest

from mlforge.core.contracts import MetricValue
from mlforge.engine.metrics import (
    aggregate_metric_folds,
    calculate_fold_classification_metrics,
)


def test_hand_checkable_binary_metrics():
    """Verify precision, recall, f1, and confusion matrix against manual calculations.

    True:  [0, 0, 1, 1]
    Pred:  [0, 1, 0, 1]
    CM:
      True 0 -> Pred 0: 1 (TN), Pred 1: 1 (FP)
      True 1 -> Pred 0: 1 (FN), Pred 1: 1 (TP)
    Class 0: Prec = 1/2 = 0.5, Rec = 1/2 = 0.5, F1 = 0.5
    Class 1: Prec = 1/2 = 0.5, Rec = 1/2 = 0.5, F1 = 0.5
    Macro Prec = 0.5, Macro Rec = 0.5, Macro F1 = 0.5, Acc = 2/4 = 0.5
    """
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
        [0.6, 0.4],
        [0.1, 0.9],
    ])
    class_labels = ["0", "1"]

    metrics, cm = calculate_fold_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        class_labels=class_labels,
    )

    assert metrics["accuracy"].value == pytest.approx(0.5)
    assert metrics["balanced_accuracy"].value == pytest.approx(0.5)
    assert metrics["macro_precision"].value == pytest.approx(0.5)
    assert metrics["macro_recall"].value == pytest.approx(0.5)
    assert metrics["macro_f1"].value == pytest.approx(0.5)
    assert cm == [[1, 1], [1, 1]]
    assert metrics["roc_auc"].value is not None


def test_tc_met_01_zero_predicted_positives_undefined_precision():
    """TC-MET-01: Zero predicted positive samples yields undefined precision with explicit diagnostic reason."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])  # Model predicts negative for all
    class_labels = ["0", "1"]

    metrics, cm = calculate_fold_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=None,
        class_labels=class_labels,
    )

    # Class 1 was never predicted: precision denominator TP + FP = 0
    assert metrics["macro_precision"].is_undefined is True
    assert metrics["macro_precision"].value is None
    assert "Zero predicted positive samples" in metrics["macro_precision"].undefined_reason

    # Macro F1 should also be undefined due to undefined precision
    assert metrics["macro_f1"].is_undefined is True
    assert metrics["macro_f1"].value is None

    # Accuracy is still valid (2/4 = 0.5)
    assert metrics["accuracy"].is_undefined is False
    assert metrics["accuracy"].value == 0.5


def test_tc_met_02_single_class_in_fold_undefined_roc_auc():
    """TC-MET-02: Validation fold containing only 1 distinct class makes ROC-AUC undefined."""
    # Validation fold only contains class 0
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    y_proba = np.array([[0.9, 0.1]] * 4)
    class_labels = ["0", "1"]

    metrics, cm = calculate_fold_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        class_labels=class_labels,
    )

    assert metrics["roc_auc"].is_undefined is True
    assert metrics["roc_auc"].value is None
    assert "fewer than 2 distinct classes" in metrics["roc_auc"].undefined_reason


def test_tc_met_03_boundary_probabilities_safe_clipping():
    """TC-MET-03: Probabilities of exactly 0.0 or 1.0 are clipped; log loss avoids inf/NaN."""
    y_true = np.array([0, 1])
    # Predicted with 100% false confidence
    y_proba = np.array([
        [0.0, 1.0],  # true 0, predicted 1 with prob 1.0
        [1.0, 0.0],  # true 1, predicted 0 with prob 1.0
    ])
    class_labels = ["0", "1"]

    metrics, _ = calculate_fold_classification_metrics(
        y_true=y_true,
        y_pred=np.array([1, 0]),
        y_proba=y_proba,
        class_labels=class_labels,
    )

    assert metrics["log_loss"].is_undefined is False
    assert metrics["log_loss"].value is not None
    assert not np.isinf(metrics["log_loss"].value)
    assert not np.isnan(metrics["log_loss"].value)
    assert metrics["log_loss"].value > 30.0  # High penalty, but finite


def test_tc_agg_01_metric_aggregation_correctness():
    """TC-AGG-01: Aggregation correctly computes mean, sample std, median, IQR, and 95% CI."""
    # 5 folds with known values: [0.80, 0.82, 0.84, 0.86, 0.88]
    # mean = 0.84, median = 0.84, min = 0.80, max = 0.88
    fold_values = [MetricValue(value=v, is_undefined=False) for v in [0.80, 0.82, 0.84, 0.86, 0.88]]

    summary = aggregate_metric_folds("macro_f1", fold_values)

    assert summary.metric_name == "macro_f1"
    assert summary.valid_fold_count == 5
    assert summary.undefined_fold_count == 0
    assert summary.mean == pytest.approx(0.84, abs=1e-5)
    assert summary.median == pytest.approx(0.84, abs=1e-5)
    assert summary.min_value == pytest.approx(0.80, abs=1e-5)
    assert summary.max_value == pytest.approx(0.88, abs=1e-5)
    assert summary.confidence_interval_95 is not None
    assert summary.confidence_interval_95[0] < summary.mean < summary.confidence_interval_95[1]


def test_tc_agg_02_undefined_metric_aggregation():
    """TC-AGG-02: Aggregation preserves undefined fold count and does not coerce to zero."""
    # 4 valid folds and 1 undefined fold
    fold_values = [
        MetricValue(value=0.80, is_undefined=False),
        MetricValue(value=0.82, is_undefined=False),
        MetricValue(value=None, is_undefined=True, undefined_reason="Zero positive predictions"),
        MetricValue(value=0.86, is_undefined=False),
        MetricValue(value=0.88, is_undefined=False),
    ]

    summary = aggregate_metric_folds("macro_f1", fold_values)

    assert summary.valid_fold_count == 4
    assert summary.undefined_fold_count == 1
    assert "Zero positive predictions" in summary.undefined_reasons
    # Mean computed over the 4 valid folds: (0.80 + 0.82 + 0.86 + 0.88)/4 = 0.84
    assert summary.mean == pytest.approx(0.84, abs=1e-5)
    # Does not coerce undefined fold to 0.0 (which would drag mean down to 0.672)
    assert summary.mean != pytest.approx(0.672, abs=1e-2)
