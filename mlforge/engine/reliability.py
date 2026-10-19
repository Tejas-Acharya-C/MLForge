"""MLForge Probability Calibration and Prediction Reliability Module.

Calculates Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
reliability diagram binning distributions, Brier score, and log-loss for binary
classification probabilities without fitting calibration curves on evaluation data.
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np

from mlforge.core.contracts import CalibrationBin, CalibrationCurveResult


def compute_calibration_curve(
    y_true: Union[np.ndarray, List[Any]],
    y_prob: Union[np.ndarray, List[Any]],
    positive_label: Optional[Any] = None,
    num_bins: int = 10,
    strategy: str = "uniform",
) -> CalibrationCurveResult:
    """Compute reliability diagram bins and calibration statistics for binary classification.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary class labels.
    y_prob : array-like
        Predicted probabilities for the positive class (1D of shape [N] or 2D of shape [N, 2]).
    positive_label : Any, optional
        Explicit label representing the positive class. If None, inferred automatically
        from class values (defaulting to 1 or second sorted label).
    num_bins : int, default=10
        Number of bins for partitioning probability space (>= 2).
    strategy : str, default="uniform"
        Binning strategy: 'uniform' (equal-width) or 'quantile' (equal-frequency).

    Returns
    -------
    CalibrationCurveResult
        Structured reliability contract containing calibration bins, ECE, MCE,
        Brier score, and log loss.

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

    # 1. Handle 2D probability arrays from estimators (e.g. [N, 2])
    if y_prob_arr.ndim == 2:
        if y_prob_arr.shape[1] == 2:
            # By default, scikit-learn order places positive class at column index 1
            y_prob_arr = y_prob_arr[:, 1]
        elif y_prob_arr.shape[1] == 1:
            y_prob_arr = y_prob_arr.ravel()
        else:
            raise ValueError(
                f"y_prob must be 1D or 2D with 2 columns; received shape {y_prob_arr.shape}."
            )
    elif y_prob_arr.ndim != 1:
        raise ValueError(f"y_prob must be 1-dimensional; received {y_prob_arr.ndim} dimensions.")

    # 2. Strict numerical safety checks: finite values and probability bounds
    if np.isnan(y_prob_arr).any() or np.isinf(y_prob_arr).any():
        raise ValueError("Non-finite values (NaN, +inf, -inf) are not permitted in predicted probabilities.")

    min_p = float(np.min(y_prob_arr))
    max_p = float(np.max(y_prob_arr))
    if min_p < 0.0 or max_p > 1.0:
        raise ValueError(
            f"Predicted probabilities must fall strictly in [0.0, 1.0]; found range [{min_p}, {max_p}]."
        )

    # 3. Binning strategy validation
    if num_bins < 2:
        raise ValueError(f"num_bins must be an integer >= 2; received {num_bins}.")

    if strategy not in ("uniform", "quantile"):
        raise ValueError(f"Unsupported binning strategy: '{strategy}'. Supported: 'uniform', 'quantile'.")

    # 4. Target class validation and positive class resolution
    unique_labels = np.unique(y_true_arr)
    if len(unique_labels) > 2:
        raise ValueError(
            f"Calibration analysis requires binary outcomes; observed {len(unique_labels)} classes: {unique_labels.tolist()}."
        )

    warnings: List[str] = []

    if len(unique_labels) == 1:
        warnings.append(
            f"Validation partition contains only a single class ('{unique_labels[0]}'); "
            "calibration metrics reflect one-sided outcomes."
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
            # Convention: if {0, 1} use 1, otherwise second sorted label
            str_labels = [str(l) for l in unique_labels]
            if set(str_labels) == {"0", "1"}:
                pos_label = unique_labels[str_labels.index("1")]
            else:
                pos_label = unique_labels[1]
        else:
            # Single-class edge case
            pos_label = unique_labels[0]

    pos_label_str = str(pos_label)
    y_bin = (y_true_arr == pos_label).astype(int)

    # 5. Core Calibration Metrics: Brier Score & Log Loss
    # Brier Score: 1/N * sum (p_i - y_i)^2
    brier_score = float(round(np.mean((y_prob_arr - y_bin) ** 2), 6))

    # Log Loss: clipped with epsilon = 1e-15 to avoid log(0)
    eps = 1e-15
    clipped_p = np.clip(y_prob_arr, eps, 1.0 - eps)
    ll_val = -float(np.mean(y_bin * np.log(clipped_p) + (1 - y_bin) * np.log(1.0 - clipped_p)))
    log_loss_score = float(round(ll_val, 6))

    # 6. Bin Partitioning
    bins: List[CalibrationBin] = []

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, num_bins + 1)
        # np.digitize maps:
        # p < edges[1] -> 0
        # edges[1] <= p < edges[2] -> 1
        # ...
        # p >= edges[-2] -> num_bins - 1
        # By using edges[1:-1], index 0 is [0.0, edges[1]), index num_bins - 1 is [edges[-2], 1.0]
        bin_assignments = np.digitize(y_prob_arr, edges[1:-1])

        for b_idx in range(num_bins):
            b_low = float(round(edges[b_idx], 6))
            b_high = float(round(edges[b_idx + 1], 6))
            mask = bin_assignments == b_idx
            count = int(np.sum(mask))

            if count > 0:
                mean_p = float(round(np.mean(y_prob_arr[mask]), 6))
                obs_rate = float(round(np.mean(y_bin[mask]), 6))
                is_empty = False
            else:
                mean_p = None
                obs_rate = None
                is_empty = True

            bins.append(
                CalibrationBin(
                    bin_index=b_idx,
                    lower_bound=b_low,
                    upper_bound=b_high,
                    sample_count=count,
                    mean_predicted_prob=mean_p,
                    observed_positive_rate=obs_rate,
                    is_empty=is_empty,
                )
            )

    elif strategy == "quantile":
        # Empirical quantile binning with deduplication for tied probabilities
        quantiles = np.linspace(0.0, 1.0, num_bins + 1)
        raw_edges = np.percentile(y_prob_arr, quantiles * 100)
        unique_edges = np.unique(raw_edges)

        if len(unique_edges) < 2:
            # All probabilities are identical
            warnings.append(
                "All predicted probabilities are identical; quantile binning collapsed to a single bin."
            )
            count = n_samples
            mean_p = float(round(np.mean(y_prob_arr), 6))
            obs_rate = float(round(np.mean(y_bin), 6))
            bins.append(
                CalibrationBin(
                    bin_index=0,
                    lower_bound=min_p,
                    upper_bound=max_p,
                    sample_count=count,
                    mean_predicted_prob=mean_p,
                    observed_positive_rate=obs_rate,
                    is_empty=False,
                )
            )
        else:
            actual_bins = len(unique_edges) - 1
            if actual_bins < num_bins:
                warnings.append(
                    f"Quantile binning requested {num_bins} bins but collapsed to {actual_bins} unique bins due to tied probabilities."
                )
            bin_assignments = np.digitize(y_prob_arr, unique_edges[1:-1])
            for b_idx in range(actual_bins):
                b_low = float(round(unique_edges[b_idx], 6))
                b_high = float(round(unique_edges[b_idx + 1], 6))
                mask = bin_assignments == b_idx
                count = int(np.sum(mask))
                if count > 0:
                    mean_p = float(round(np.mean(y_prob_arr[mask]), 6))
                    obs_rate = float(round(np.mean(y_bin[mask]), 6))
                    is_empty = False
                else:
                    mean_p = None
                    obs_rate = None
                    is_empty = True
                bins.append(
                    CalibrationBin(
                        bin_index=b_idx,
                        lower_bound=b_low,
                        upper_bound=b_high,
                        sample_count=count,
                        mean_predicted_prob=mean_p,
                        observed_positive_rate=obs_rate,
                        is_empty=is_empty,
                    )
                )

    # 7. Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    # Formula: ECE = sum_{b: n_b > 0} (n_b / N) * |observed_rate_b - mean_prob_b|
    weighted_diffs: List[float] = []
    abs_diffs: List[float] = []

    for b in bins:
        if not b.is_empty and b.observed_positive_rate is not None and b.mean_predicted_prob is not None:
            diff = abs(b.observed_positive_rate - b.mean_predicted_prob)
            abs_diffs.append(diff)
            weight = b.sample_count / n_samples
            weighted_diffs.append(weight * diff)

    if weighted_diffs:
        ece = float(round(sum(weighted_diffs), 6))
        mce = float(round(max(abs_diffs), 6))
    else:
        ece = 0.0
        mce = 0.0

    return CalibrationCurveResult(
        bins=bins,
        expected_calibration_error=ece,
        max_calibration_error=mce,
        brier_score=brier_score,
        log_loss=log_loss_score,
        bin_strategy=strategy,
        num_bins=len(bins),
        total_samples=n_samples,
        positive_class_label=pos_label_str,
        is_usable=True,
        unusable_reason=None,
        warnings=warnings,
    )
