"""MLForge Classification Error and Threshold Diagnostics Module.

Implements binary classification threshold analysis across configurable decision cutoffs,
per-fold failure summaries, confusion-matrix derived diagnostics, and non-target
subgroup error rate breakdowns.
"""

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from mlforge.core.contracts import (
    ErrorDiagnosticsResult,
    FoldErrorDiagnostic,
    SubgroupMetricDiagnostic,
    ThresholdAnalysisResult,
    ThresholdMetrics,
)

DEFAULT_THRESHOLDS: List[float] = [
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
]


def analyze_classification_thresholds(
    y_true: Union[np.ndarray, Sequence[Any]],
    y_prob: Union[np.ndarray, Sequence[Any]],
    positive_label: Optional[Any] = None,
    thresholds: Optional[Sequence[float]] = None,
) -> ThresholdAnalysisResult:
    """Evaluate binary classification contingency metrics across decision thresholds.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary class labels.
    y_prob : array-like
        Predicted probabilities for the positive class (1D of shape [N] or 2D of shape [N, 2]).
    positive_label : Any, optional
        Target class considered positive. Inferred from labels if None.
    thresholds : sequence of float, optional
        Decision thresholds in [0.0, 1.0]. If None, defaults to [0.0, 0.1, ..., 1.0].

    Returns
    -------
    ThresholdAnalysisResult
        Metrics and contingency counts computed at each decision threshold.

    Raises
    ------
    ValueError
        If inputs are invalid, non-finite, out of bounds, mismatched in length, or empty.
    """
    if y_true is None or y_prob is None:
        raise ValueError("y_true and y_prob must not be None.")

    y_true_arr = np.asarray(y_true)
    try:
        y_prob_arr = np.asarray(y_prob, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Predicted probabilities must be numeric: {exc}")

    n_samples = len(y_true_arr)
    if n_samples == 0:
        raise ValueError("Evaluation sample size must be at least 1.")

    if len(y_prob_arr) != n_samples:
        raise ValueError(
            f"Mismatched lengths: y_true has {n_samples} samples, y_prob has {len(y_prob_arr)} samples."
        )

    # Handle 2D probability arrays from estimators
    if y_prob_arr.ndim == 2:
        if y_prob_arr.shape[1] == 2:
            y_prob_arr = y_prob_arr[:, 1]
        elif y_prob_arr.shape[1] == 1:
            y_prob_arr = y_prob_arr.ravel()
        else:
            raise ValueError(
                f"y_prob must be 1D or 2D with 2 columns; received shape {y_prob_arr.shape}."
            )
    elif y_prob_arr.ndim != 1:
        raise ValueError(f"y_prob must be 1-dimensional; received {y_prob_arr.ndim} dimensions.")

    # Strict numerical safety
    if np.isnan(y_prob_arr).any() or np.isinf(y_prob_arr).any():
        raise ValueError("Non-finite values (NaN, +inf, -inf) are not permitted in predicted probabilities.")

    min_p = float(np.min(y_prob_arr))
    max_p = float(np.max(y_prob_arr))
    if min_p < 0.0 or max_p > 1.0:
        raise ValueError(
            f"Predicted probabilities must fall strictly in [0.0, 1.0]; found range [{min_p}, {max_p}]."
        )

    # Validate thresholds
    eval_thresholds: List[float] = (
        DEFAULT_THRESHOLDS if thresholds is None else [float(t) for t in thresholds]
    )
    if len(eval_thresholds) == 0:
        raise ValueError("Threshold list must contain at least one threshold.")

    for t in eval_thresholds:
        if np.isnan(t) or np.isinf(t) or t < 0.0 or t > 1.0:
            raise ValueError(f"Each threshold must be a finite float in [0.0, 1.0]; received {t}.")

    # Validate binary target
    unique_labels = np.unique(y_true_arr)
    if len(unique_labels) > 2:
        raise ValueError(
            f"Threshold analysis requires binary outcomes; observed {len(unique_labels)} classes: {unique_labels.tolist()}."
        )

    warnings: List[str] = []

    if len(unique_labels) == 1:
        warnings.append(
            f"Validation partition contains only a single class ('{unique_labels[0]}'); "
            "threshold metrics reflect one-sided outcomes."
        )

    if positive_label is not None:
        pos_label = positive_label
        if len(unique_labels) == 2 and pos_label not in unique_labels:
            matched = False
            for ul in unique_labels:
                if str(ul) == str(pos_label):
                    pos_label = ul
                    matched = True
                    break
            if not matched:
                warnings.append(
                    f"Configured positive_label '{pos_label}' was not found in validation labels {unique_labels.tolist()}."
                )
    else:
        if len(unique_labels) == 2:
            str_labels = [str(l) for l in unique_labels]
            if set(str_labels) == {"0", "1"}:
                pos_label = unique_labels[str_labels.index("1")]
            else:
                pos_label = unique_labels[1]
        else:
            pos_label = unique_labels[0]

    pos_label_str = str(pos_label)
    y_bin = (y_true_arr == pos_label).astype(int)

    # Evaluate each threshold
    points: List[ThresholdMetrics] = []

    for t in eval_thresholds:
        # Decision rule: predict positive if p >= t
        # Edge cases:
        # At t=0.0: p >= 0.0 is always true for probabilities in [0.0, 1.0] -> all positive
        # At t=1.0: p >= 1.0 is true only if p == 1.0 -> all negative unless p == 1.0
        y_pred_bin = (y_prob_arr >= t).astype(int)

        tp = int(np.sum((y_bin == 1) & (y_pred_bin == 1)))
        fp = int(np.sum((y_bin == 0) & (y_pred_bin == 1)))
        tn = int(np.sum((y_bin == 0) & (y_pred_bin == 0)))
        fn = int(np.sum((y_bin == 1) & (y_pred_bin == 0)))

        # Exact confusion matrix conservation check
        if tp + fp + tn + fn != n_samples:
            raise RuntimeError(
                f"Internal error: Confusion matrix elements sum to {tp + fp + tn + fn}, expected {n_samples}."
            )

        predicted_pos_rate = float(round((tp + fp) / n_samples, 6))

        # Precision = TP / (TP + FP)
        precision: Optional[float] = None
        precision_reason: Optional[str] = None
        if tp + fp > 0:
            precision = float(round(tp / (tp + fp), 6))
        else:
            precision_reason = "No samples predicted positive (TP + FP = 0); precision is undefined."

        # Recall = TP / (TP + FN)
        recall: Optional[float] = None
        recall_reason: Optional[str] = None
        actual_positives = tp + fn
        if actual_positives > 0:
            recall = float(round(tp / actual_positives, 6))
        else:
            recall_reason = "No actual positive samples in partition (TP + FN = 0); recall is undefined."

        # Specificity = TN / (TN + FP)
        specificity: Optional[float] = None
        specificity_reason: Optional[str] = None
        actual_negatives = tn + fp
        if actual_negatives > 0:
            specificity = float(round(tn / actual_negatives, 6))
        else:
            specificity_reason = "No actual negative samples in partition (TN + FP = 0); specificity is undefined."

        # F1 = 2 * (P * R) / (P + R)
        f1_score: Optional[float] = None
        f1_reason: Optional[str] = None
        if precision is not None and recall is not None:
            if precision + recall > 0.0:
                f1_score = float(round(2.0 * (precision * recall) / (precision + recall), 6))
            else:
                f1_score = 0.0
        else:
            f1_reason = "Precision and/or Recall are undefined; F1 score cannot be computed."

        # Balanced Accuracy = (Recall + Specificity) / 2
        balanced_accuracy: Optional[float] = None
        bal_acc_reason: Optional[str] = None
        if recall is not None and specificity is not None:
            balanced_accuracy = float(round((recall + specificity) / 2.0, 6))
        else:
            bal_acc_reason = "Recall and/or Specificity are undefined; balanced accuracy cannot be computed."

        points.append(
            ThresholdMetrics(
                threshold=round(t, 6),
                evaluated_samples=n_samples,
                true_positives=tp,
                false_positives=fp,
                true_negatives=tn,
                false_negatives=fn,
                precision=precision,
                recall=recall,
                specificity=specificity,
                f1_score=f1_score,
                balanced_accuracy=balanced_accuracy,
                predicted_positive_rate=predicted_pos_rate,
                precision_undefined_reason=precision_reason,
                recall_undefined_reason=recall_reason,
                specificity_undefined_reason=specificity_reason,
                f1_undefined_reason=f1_reason,
                balanced_accuracy_undefined_reason=bal_acc_reason,
            )
        )

    return ThresholdAnalysisResult(
        positive_class_label=pos_label_str,
        evaluated_samples=n_samples,
        threshold_metrics=points,
        warnings=warnings,
    )


def compute_fold_error_diagnostic(
    y_true: Union[np.ndarray, Sequence[Any]],
    y_pred: Union[np.ndarray, Sequence[Any]],
    fold_index: int,
    positive_label: Optional[Any] = None,
) -> FoldErrorDiagnostic:
    """Calculate error counts and breakdown for a single evaluation fold.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels for validation partition.
    y_pred : array-like
        Predicted labels for validation partition.
    fold_index : int
        0-indexed fold number.
    positive_label : Any, optional
        Designated positive class label if binary classification.

    Returns
    -------
    FoldErrorDiagnostic
        Error summary with false positive and false negative counts.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    n_samples = len(y_true_arr)

    if n_samples == 0:
        raise ValueError("Fold evaluation sample size must be at least 1.")

    if len(y_pred_arr) != n_samples:
        raise ValueError(
            f"Mismatched lengths: y_true has {n_samples} samples, y_pred has {len(y_pred_arr)} samples."
        )

    errors = int(np.sum(y_true_arr != y_pred_arr))
    error_rate = float(round(errors / n_samples, 6))

    false_positives: Optional[int] = None
    false_negatives: Optional[int] = None

    if positive_label is not None:
        pos_val = str(positive_label)
        y_true_str = y_true_arr.astype(str)
        y_pred_str = y_pred_arr.astype(str)
        false_positives = int(np.sum((y_true_str != pos_val) & (y_pred_str == pos_val)))
        false_negatives = int(np.sum((y_true_str == pos_val) & (y_pred_str != pos_val)))

    return FoldErrorDiagnostic(
        fold_index=fold_index,
        total_samples=n_samples,
        error_count=errors,
        error_rate=error_rate,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def compute_subgroup_diagnostics(
    y_true: Union[np.ndarray, Sequence[Any]],
    y_pred: Union[np.ndarray, Sequence[Any]],
    metadata_df: Union[pd.DataFrame, Dict[str, Sequence[Any]]],
    subgroup_columns: Sequence[str],
    positive_label: Optional[Any] = None,
    min_subgroup_size: int = 5,
) -> Dict[str, List[SubgroupMetricDiagnostic]]:
    """Compute descriptive error rates and performance metrics across non-target subgroup slices.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    metadata_df : pd.DataFrame or dict
        Non-target tabular attributes corresponding to evaluated observations.
    subgroup_columns : sequence of str
        Metadata column names to evaluate performance slices on.
    positive_label : Any, optional
        Target class considered positive for binary subgroup precision and recall.
    min_subgroup_size : int, default=5
        Minimum observations in subgroup required before calculating classification metrics.

    Returns
    -------
    Dict[str, List[SubgroupMetricDiagnostic]]
        Mapping of subgroup column name to list of category diagnostics.

    Raises
    ------
    ValueError
        If inputs are empty, mismatched in length, or requested columns do not exist.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    n_samples = len(y_true_arr)

    if n_samples == 0:
        raise ValueError("Sample size must be at least 1.")

    if len(y_pred_arr) != n_samples:
        raise ValueError(
            f"Mismatched lengths: y_true has {n_samples} samples, y_pred has {len(y_pred_arr)} samples."
        )

    if not isinstance(metadata_df, pd.DataFrame):
        try:
            metadata_df = pd.DataFrame(metadata_df)
        except Exception as exc:
            raise ValueError(f"Cannot convert metadata input to DataFrame: {exc}")

    if len(metadata_df) != n_samples:
        raise ValueError(
            f"Mismatched lengths: y_true has {n_samples} samples, metadata has {len(metadata_df)} rows."
        )

    results: Dict[str, List[SubgroupMetricDiagnostic]] = {}

    for col in subgroup_columns:
        if col not in metadata_df.columns:
            raise ValueError(f"Subgroup column '{col}' is not present in metadata features.")

        col_diagnostics: List[SubgroupMetricDiagnostic] = []
        series = metadata_df[col]
        # Treat missing / null values as a distinct category representation
        categories = series.fillna("<MISSING>").astype(str).unique()
        categories.sort()

        for cat in categories:
            mask = (series.fillna("<MISSING>").astype(str) == cat).to_numpy()
            sub_count = int(np.sum(mask))

            if sub_count == 0:
                continue

            sub_y_true = y_true_arr[mask]
            sub_y_pred = y_pred_arr[mask]

            sub_errors = int(np.sum(sub_y_true != sub_y_pred))
            sub_error_rate = float(round(sub_errors / sub_count, 6))

            if sub_count < min_subgroup_size:
                col_diagnostics.append(
                    SubgroupMetricDiagnostic(
                        subgroup_column=col,
                        subgroup_value=str(cat),
                        sample_count=sub_count,
                        error_count=sub_errors,
                        error_rate=sub_error_rate,
                        accuracy=None,
                        precision=None,
                        recall=None,
                        is_usable=False,
                        unusable_reason=(
                            f"Subgroup sample size ({sub_count}) is below minimum support "
                            f"threshold ({min_subgroup_size}); metrics withheld to prevent statistical distortion."
                        ),
                    )
                )
            else:
                acc = float(round((sub_count - sub_errors) / sub_count, 6))
                sub_precision: Optional[float] = None
                sub_recall: Optional[float] = None
                unusable_reason: Optional[str] = None

                if positive_label is not None:
                    pos_str = str(positive_label)
                    sub_true_str = sub_y_true.astype(str)
                    sub_pred_str = sub_y_pred.astype(str)

                    sub_tp = int(np.sum((sub_true_str == pos_str) & (sub_pred_str == pos_str)))
                    sub_fp = int(np.sum((sub_true_str != pos_str) & (sub_pred_str == pos_str)))
                    sub_fn = int(np.sum((sub_true_str == pos_str) & (sub_pred_str != pos_str)))

                    actual_pos = sub_tp + sub_fn
                    predicted_pos = sub_tp + sub_fp

                    unique_sub_true = np.unique(sub_true_str)
                    if len(unique_sub_true) < 2:
                        unusable_reason = (
                            f"Subgroup contains only single class '{unique_sub_true[0]}'; "
                            "binary precision and recall are not uniquely defined."
                        )

                    if predicted_pos > 0:
                        sub_precision = float(round(sub_tp / predicted_pos, 6))
                    if actual_pos > 0:
                        sub_recall = float(round(sub_tp / actual_pos, 6))

                col_diagnostics.append(
                    SubgroupMetricDiagnostic(
                        subgroup_column=col,
                        subgroup_value=str(cat),
                        sample_count=sub_count,
                        error_count=sub_errors,
                        error_rate=sub_error_rate,
                        accuracy=acc,
                        precision=sub_precision,
                        recall=sub_recall,
                        is_usable=True,
                        unusable_reason=unusable_reason,
                    )
                )

        results[col] = col_diagnostics

    return results


def aggregate_error_diagnostics(
    fold_diagnostics: List[FoldErrorDiagnostic],
    total_samples: int,
    subgroup_diagnostics: Optional[Dict[str, List[SubgroupMetricDiagnostic]]] = None,
    warnings: Optional[List[str]] = None,
) -> ErrorDiagnosticsResult:
    """Aggregate fold error diagnostics into an overall failure analysis summary.

    Parameters
    ----------
    fold_diagnostics : list of FoldErrorDiagnostic
        Per-fold failure diagnostics.
    total_samples : int
        Total samples across all folds.
    subgroup_diagnostics : dict, optional
        Subgroup diagnostics breakdown.
    warnings : list of str, optional
        Advisory notices.

    Returns
    -------
    ErrorDiagnosticsResult
        Aggregated failure report.
    """
    total_errors = sum(fd.error_count for fd in fold_diagnostics)
    overall_error_rate = float(round(total_errors / total_samples, 6)) if total_samples > 0 else 0.0

    return ErrorDiagnosticsResult(
        total_samples=total_samples,
        total_errors=total_errors,
        overall_error_rate=overall_error_rate,
        fold_diagnostics=fold_diagnostics,
        subgroup_diagnostics=subgroup_diagnostics,
        warnings=warnings or [],
    )
