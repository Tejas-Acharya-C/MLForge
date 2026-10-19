"""MLForge Classification Metrics and Aggregation.

Computes point metrics, probability metrics, and confusion matrices with explicit
handling and diagnosis of mathematically or logically undefined values.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from mlforge.core.contracts import MetricSummary, MetricValue


def calculate_fold_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    class_labels: List[Any],
) -> Tuple[Dict[str, MetricValue], List[List[int]]]:
    """Calculate fold-level classification metrics with explicit undefined handling.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth class labels for the fold.
    y_pred : np.ndarray
        Predicted class labels for the fold.
    y_proba : Optional[np.ndarray]
        Predicted class probabilities [n_samples, n_classes].
    class_labels : List[Any]
        Fixed ordered list of class labels across all folds.

    Returns
    -------
    Tuple[Dict[str, MetricValue], List[List[int]]]
        - Dictionary of metric names mapped to MetricValue objects.
        - 2D confusion matrix array [true_idx][pred_idx].
    """
    metrics: Dict[str, MetricValue] = {}

    # Normalize all labels to string type for strict, consistent matching
    str_class_labels = [str(l) for l in class_labels]
    y_true_str = np.asarray(y_true).astype(str)
    y_pred_str = np.asarray(y_pred).astype(str)

    # 1. Standard Accuracy
    acc = float(accuracy_score(y_true_str, y_pred_str))
    metrics["accuracy"] = MetricValue(value=round(acc, 6), is_undefined=False)

    # 2. Balanced Accuracy
    try:
        bal_acc = float(balanced_accuracy_score(y_true_str, y_pred_str))
        metrics["balanced_accuracy"] = MetricValue(value=round(bal_acc, 6), is_undefined=False)
    except Exception as exc:
        metrics["balanced_accuracy"] = MetricValue(
            value=None, is_undefined=True, undefined_reason=f"Failed to compute balanced accuracy: {exc}"
        )

    # 3. Macro Precision (handles zero division when no samples are predicted for a class)
    try:
        per_class_prec = precision_score(
            y_true_str, y_pred_str, labels=str_class_labels, average=None, zero_division=np.nan
        )
        if np.isnan(per_class_prec).any():
            nan_classes = [str_class_labels[i] for i, v in enumerate(per_class_prec) if np.isnan(v)]
            metrics["macro_precision"] = MetricValue(
                value=None,
                is_undefined=True,
                undefined_reason=(
                    f"Zero predicted positive samples for class(es) {nan_classes} "
                    "(zero division in precision calculation)."
                ),
            )
        else:
            metrics["macro_precision"] = MetricValue(
                value=round(float(np.mean(per_class_prec)), 6), is_undefined=False
            )
    except Exception as exc:
        metrics["macro_precision"] = MetricValue(
            value=None, is_undefined=True, undefined_reason=f"Precision computation error: {exc}"
        )

    # 4. Macro Recall (handles zero division if a class has zero actual samples in validation fold)
    try:
        per_class_rec = recall_score(
            y_true_str, y_pred_str, labels=str_class_labels, average=None, zero_division=np.nan
        )
        if np.isnan(per_class_rec).any():
            nan_classes = [str_class_labels[i] for i, v in enumerate(per_class_rec) if np.isnan(v)]
            metrics["macro_recall"] = MetricValue(
                value=None,
                is_undefined=True,
                undefined_reason=(
                    f"Zero actual positive samples in validation slice for class(es) {nan_classes}."
                ),
            )
        else:
            metrics["macro_recall"] = MetricValue(
                value=round(float(np.mean(per_class_rec)), 6), is_undefined=False
            )
    except Exception as exc:
        metrics["macro_recall"] = MetricValue(
            value=None, is_undefined=True, undefined_reason=f"Recall computation error: {exc}"
        )

    # 5. Macro F1
    try:
        per_class_f1 = f1_score(
            y_true_str, y_pred_str, labels=str_class_labels, average=None, zero_division=np.nan
        )
        if (
            np.isnan(per_class_f1).any()
            or metrics["macro_precision"].is_undefined
            or metrics["macro_recall"].is_undefined
        ):
            metrics["macro_f1"] = MetricValue(
                value=None,
                is_undefined=True,
                undefined_reason="Precision and/or recall were undefined for one or more classes, resulting in undefined F1.",
            )
        else:
            metrics["macro_f1"] = MetricValue(
                value=round(float(np.mean(per_class_f1)), 6), is_undefined=False
            )
    except Exception as exc:
        metrics["macro_f1"] = MetricValue(
            value=None, is_undefined=True, undefined_reason=f"F1 computation error: {exc}"
        )

    # 6. Confusion Matrix (ensures constant dimensions [n_classes, n_classes])
    cm = confusion_matrix(y_true_str, y_pred_str, labels=str_class_labels).tolist()

    # 7. Probability-based metrics (if y_proba is provided)
    if y_proba is not None:
        # A. Log Loss (Clipped to prevent inf)
        try:
            # Ensure probabilities are clipped to [1e-15, 1 - 1e-15]
            eps = 1e-15
            clipped_proba = np.clip(y_proba, eps, 1.0 - eps)
            # Re-normalize rows to sum to 1.0
            clipped_proba = clipped_proba / clipped_proba.sum(axis=1, keepdims=True)
            ll = float(log_loss(y_true_str, clipped_proba, labels=str_class_labels))
            metrics["log_loss"] = MetricValue(value=round(ll, 6), is_undefined=False)
        except Exception as exc:
            metrics["log_loss"] = MetricValue(
                value=None, is_undefined=True, undefined_reason=f"Log loss error: {exc}"
            )

        # B. Brier Score
        try:
            # Multi-class Brier formulation: 1/N * sum_i sum_c (p_ic - y_ic)^2
            n_samples = len(y_true_str)
            y_true_indices = np.array([str_class_labels.index(label) for label in y_true_str])
            y_one_hot = np.zeros((n_samples, len(str_class_labels)))
            y_one_hot[np.arange(n_samples), y_true_indices] = 1.0

            brier = float(np.mean(np.sum((y_proba - y_one_hot) ** 2, axis=1)))
            metrics["brier_score"] = MetricValue(value=round(brier, 6), is_undefined=False)
        except Exception as exc:
            metrics["brier_score"] = MetricValue(
                value=None, is_undefined=True, undefined_reason=f"Brier score error: {exc}"
            )

        # C. ROC-AUC (OvR Macro)
        # Precondition check: validation slice must have at least 2 distinct classes present!
        unique_y_true = np.unique(y_true_str)
        if len(unique_y_true) < 2:
            metrics["roc_auc"] = MetricValue(
                value=None,
                is_undefined=True,
                undefined_reason="Validation fold contains fewer than 2 distinct classes; ROC-AUC is undefined.",
            )
        else:
            try:
                if len(str_class_labels) == 2:
                    # Binary ROC-AUC: use positive class probability (index 1)
                    pos_idx = 1
                    auc = float(roc_auc_score(y_true_str, y_proba[:, pos_idx], labels=str_class_labels))
                else:
                    # Multiclass OvR Macro ROC-AUC
                    auc = float(
                        roc_auc_score(
                            y_true_str,
                            y_proba,
                            multi_class="ovr",
                            average="macro",
                            labels=str_class_labels,
                        )
                    )
                metrics["roc_auc"] = MetricValue(value=round(auc, 6), is_undefined=False)
            except Exception as exc:
                metrics["roc_auc"] = MetricValue(
                    value=None, is_undefined=True, undefined_reason=f"ROC-AUC computation error: {exc}"
                )

    return metrics, cm


def aggregate_metric_folds(
    metric_name: str,
    fold_metric_values: List[MetricValue],
) -> MetricSummary:
    """Aggregate fold-level metric values into summary statistics.

    Maintains explicit separation between valid and undefined folds. Never coerces
    undefined values to zero.
    """
    valid_values: List[float] = []
    undefined_reasons: List[str] = []

    for mv in fold_metric_values:
        if mv.is_undefined or mv.value is None or np.isnan(mv.value):
            if mv.undefined_reason:
                undefined_reasons.append(mv.undefined_reason)
        else:
            valid_values.append(mv.value)

    valid_count = len(valid_values)
    undefined_count = len(fold_metric_values) - valid_count

    if valid_count == 0:
        return MetricSummary(
            metric_name=metric_name,
            mean=None,
            std=None,
            median=None,
            iqr=None,
            min_value=None,
            max_value=None,
            confidence_interval_95=None,
            valid_fold_count=0,
            undefined_fold_count=undefined_count,
            undefined_reasons=list(set(undefined_reasons)),
        )

    vals = np.array(valid_values, dtype=float)
    mean_val = float(np.mean(vals))
    min_val = float(np.min(vals))
    max_val = float(np.max(vals))
    median_val = float(np.median(vals))

    # Sample standard deviation (ddof=1 if count > 1)
    if valid_count > 1:
        std_val = float(np.std(vals, ddof=1))
        q75, q25 = np.percentile(vals, [75, 25])
        iqr_val = float(q75 - q25)

        # 95% Confidence Interval using Student's t-distribution over cross-validation fold scores.
        # Statistical caveat: Cross-validation folds share training instances ((K-2)/(K-1) overlap),
        # violating the i.i.d. assumption. This uncorrected interval estimates the empirical mean fold
        # score across the observed resampling partition, but underestimates generalization variance
        # (Bengio & Grandvalet, 2004). For rigorous paired contrast against baselines, MLForge uses
        # the Nadeau-Bengio corrected resampled t-test.
        se = std_val / math.sqrt(valid_count)
        t_crit = float(stats.t.ppf(0.975, df=valid_count - 1))
        ci_low = float(mean_val - t_crit * se)
        ci_high = float(mean_val + t_crit * se)
        ci_95 = [round(ci_low, 6), round(ci_high, 6)]
    else:
        std_val = 0.0
        iqr_val = 0.0
        ci_95 = [round(mean_val, 6), round(mean_val, 6)]

    return MetricSummary(
        metric_name=metric_name,
        mean=round(mean_val, 6),
        std=round(std_val, 6),
        median=round(median_val, 6),
        iqr=round(iqr_val, 6),
        min_value=round(min_val, 6),
        max_value=round(max_val, 6),
        confidence_interval_95=ci_95,
        valid_fold_count=valid_count,
        undefined_fold_count=undefined_count,
        undefined_reasons=list(set(undefined_reasons)),
    )
