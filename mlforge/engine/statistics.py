"""MLForge Baseline Contrast and Statistical Significance Engine.

Implements paired per-fold metric comparisons, oriented raw differences,
Nadeau-Bengio corrected resampled t-tests, and Wilcoxon signed-rank comparisons
with explicit assumptions, limitations, and edge-case handling.
"""

import math
from typing import Dict, List, Optional, Set

import numpy as np
from scipy import stats

from mlforge.core.contracts import (
    ComparisonReport,
    DatasetMetadata,
    FoldResult,
    MetricComparisonResult,
    MetricDirection,
    MetricValue,
    ModelEvaluationResult,
    PairedMetricDifference,
    ReproducibilityMetadata,
    StatisticalTestResult,
    TestStatus,
)
from mlforge.core.exceptions import IncompatibleEvaluationPairError

# Canonical mapping of evaluation metrics to improvement direction
METRIC_DIRECTIONS: Dict[str, MetricDirection] = {
    "accuracy": MetricDirection.HIGHER_IS_BETTER,
    "balanced_accuracy": MetricDirection.HIGHER_IS_BETTER,
    "macro_precision": MetricDirection.HIGHER_IS_BETTER,
    "macro_recall": MetricDirection.HIGHER_IS_BETTER,
    "macro_f1": MetricDirection.HIGHER_IS_BETTER,
    "roc_auc": MetricDirection.HIGHER_IS_BETTER,
    "log_loss": MetricDirection.LOWER_IS_BETTER,
    "brier_score": MetricDirection.LOWER_IS_BETTER,
}

# Split strategies supported for paired comparison and corrected t-test
SUPPORTED_SPLIT_STRATEGIES: Set[str] = {"stratified_k_fold", "k_fold"}


def validate_evaluation_compatibility(
    candidate: ModelEvaluationResult,
    baseline: ModelEvaluationResult,
    dataset_metadata: DatasetMetadata,
    reproducibility: ReproducibilityMetadata,
) -> None:
    """Validate that candidate and baseline evaluations are strictly compatible for pairing.

    Raises
    ------
    IncompatibleEvaluationPairError
        If dataset checksum, fold counts, fold identities, split strategy, or split partitions mismatch.
    """
    if candidate is baseline:
        raise IncompatibleEvaluationPairError(
            "Cannot compare model evaluation against itself: candidate and baseline reference the exact same evaluation object."
        )

    if not dataset_metadata.checksum_sha256:
        raise IncompatibleEvaluationPairError("Dataset metadata is missing cryptographic checksum.")

    if reproducibility.split_strategy not in SUPPORTED_SPLIT_STRATEGIES:
        raise IncompatibleEvaluationPairError(
            f"Evaluation split strategy '{reproducibility.split_strategy}' is unsupported for paired comparison. "
            f"Supported strategies: {sorted(SUPPORTED_SPLIT_STRATEGIES)}."
        )

    cand_folds = candidate.fold_results
    base_folds = baseline.fold_results

    # 1. Fold count validation against protocol
    if len(cand_folds) != reproducibility.num_folds:
        raise IncompatibleEvaluationPairError(
            f"Candidate fold count ({len(cand_folds)}) does not match expected "
            f"protocol fold count ({reproducibility.num_folds})."
        )

    if len(base_folds) != reproducibility.num_folds:
        raise IncompatibleEvaluationPairError(
            f"Baseline fold count ({len(base_folds)}) does not match expected "
            f"protocol fold count ({reproducibility.num_folds})."
        )

    if len(cand_folds) != len(base_folds):
        raise IncompatibleEvaluationPairError(
            f"Fold count mismatch between candidate ({len(cand_folds)}) and baseline ({len(base_folds)})."
        )

    # 2. Check for duplicate fold IDs in either model
    cand_ids = [cf.fold_index for cf in cand_folds]
    base_ids = [bf.fold_index for bf in base_folds]

    if len(set(cand_ids)) != len(cand_ids):
        raise IncompatibleEvaluationPairError("Candidate fold results contain duplicate fold IDs.")

    if len(set(base_ids)) != len(base_ids):
        raise IncompatibleEvaluationPairError("Baseline fold results contain duplicate fold IDs.")

    # 3. Check that the fold ID sets are identical and cover expected range [0..K-1]
    expected_ids = set(range(reproducibility.num_folds))
    if set(cand_ids) != expected_ids:
        raise IncompatibleEvaluationPairError(
            f"Candidate fold IDs do not match expected range [0..{reproducibility.num_folds - 1}]: got {sorted(set(cand_ids))}."
        )
    if set(base_ids) != expected_ids:
        raise IncompatibleEvaluationPairError(
            f"Baseline fold IDs do not match expected range [0..{reproducibility.num_folds - 1}]: got {sorted(set(base_ids))}."
        )

    # 4. Map by fold_index to ensure matching is by fold ID, not list position
    cand_map = {cf.fold_index: cf for cf in cand_folds}
    base_map = {bf.fold_index: bf for bf in base_folds}

    for fold_id in sorted(expected_ids):
        cf = cand_map[fold_id]
        bf = base_map[fold_id]

        if cf.train_indices_count != bf.train_indices_count:
            raise IncompatibleEvaluationPairError(
                f"Fold {fold_id} training partition size mismatch: candidate={cf.train_indices_count} "
                f"vs baseline={bf.train_indices_count}."
            )
        if cf.val_indices_count != bf.val_indices_count:
            raise IncompatibleEvaluationPairError(
                f"Fold {fold_id} validation partition size mismatch: candidate={cf.val_indices_count} "
                f"vs baseline={bf.val_indices_count}."
            )
        if cf.class_labels != bf.class_labels:
            raise IncompatibleEvaluationPairError(
                f"Fold {fold_id} class label ordering mismatch: candidate={cf.class_labels} "
                f"vs baseline={bf.class_labels}."
            )
        # Verify cryptographic partition fingerprint per fold
        c_fp = cf.partition_fingerprint
        b_fp = bf.partition_fingerprint

        if (c_fp is not None and b_fp is None) or (c_fp is None and b_fp is not None):
            raise IncompatibleEvaluationPairError(
                f"Fold {fold_id} has asymmetric partition fingerprints: candidate has "
                f"{'present' if c_fp is not None else 'absent'} fingerprint while baseline has "
                f"{'present' if b_fp is not None else 'absent'} fingerprint. "
                "Both evaluations must provide partition fingerprints or both omit them."
            )
        elif c_fp is not None and b_fp is not None:
            if c_fp != b_fp:
                raise IncompatibleEvaluationPairError(
                    f"Fold {fold_id} partition fingerprint mismatch: candidate={c_fp} "
                    f"vs baseline={b_fp}. Folds do not share identical sample partitions."
                )


def _is_invalid_metric_value(metric: Optional[MetricValue]) -> bool:
    """Return True if metric is missing, undefined, NaN, or non-finite."""
    if metric is None or metric.is_undefined or metric.value is None:
        return True
    val = metric.value
    return np.isnan(val) or not math.isfinite(val)


def compute_paired_metric_differences(
    candidate_folds: List[FoldResult],
    baseline_folds: List[FoldResult],
    metric_name: str,
    direction: Optional[MetricDirection] = None,
) -> PairedMetricDifference:
    """Compute raw paired per-fold differences: raw_delta = candidate - baseline.

    Aligns folds strictly by fold_index (not list position).
    Preserves exact raw difference without inverting signs for loss metrics.
    Rejects None, NaN, and +/-infinity as unusable observations.
    """
    metric_dir = direction or METRIC_DIRECTIONS.get(metric_name, MetricDirection.NOT_APPLICABLE)

    # Index folds by unique fold_index for position-independent alignment
    cand_map = {cf.fold_index: cf for cf in candidate_folds}
    base_map = {bf.fold_index: bf for bf in baseline_folds}
    all_fold_ids = sorted(set(cand_map.keys()).union(base_map.keys()))

    expected_folds = len(cand_map)
    paired_folds = len(all_fold_ids)

    raw_deltas: List[Optional[float]] = []
    valid_deltas: List[float] = []
    unusable_reasons: List[str] = []

    for fold_id in all_fold_ids:
        cf = cand_map.get(fold_id)
        bf = base_map.get(fold_id)

        if cf is None:
            raw_deltas.append(None)
            unusable_reasons.append(f"Fold {fold_id}: missing from candidate evaluation.")
            continue
        if bf is None:
            raw_deltas.append(None)
            unusable_reasons.append(f"Fold {fold_id}: missing from baseline evaluation.")
            continue

        c_metric = cf.metrics.get(metric_name)
        b_metric = bf.metrics.get(metric_name)

        if _is_invalid_metric_value(c_metric):
            raw_deltas.append(None)
            reason = c_metric.undefined_reason if c_metric and c_metric.undefined_reason else "invalid/non-finite or missing"
            unusable_reasons.append(f"Fold {fold_id}: candidate metric was unusable ({reason}).")
        elif _is_invalid_metric_value(b_metric):
            raw_deltas.append(None)
            reason = b_metric.undefined_reason if b_metric and b_metric.undefined_reason else "invalid/non-finite or missing"
            unusable_reasons.append(f"Fold {fold_id}: baseline metric was unusable ({reason}).")
        else:
            delta = float(c_metric.value - b_metric.value)
            if not math.isfinite(delta) or np.isnan(delta):
                raw_deltas.append(None)
                unusable_reasons.append(f"Fold {fold_id}: paired difference evaluated to non-finite value ({delta}).")
            else:
                raw_deltas.append(round(delta, 6))
                valid_deltas.append(delta)

    usable_folds = len(valid_deltas)
    is_usable = (usable_folds == expected_folds) and (usable_folds >= 2)

    unusable_reason: Optional[str] = None
    if not is_usable:
        if usable_folds < 2:
            unusable_reason = (
                f"Insufficient usable paired observations: {usable_folds} valid folds found, "
                "but at least 2 are required for paired statistical comparison."
            )
        else:
            unusable_reason = (
                f"Metric has {expected_folds - usable_folds} fold(s) with invalid or undefined values: "
                + " ".join(list(set(unusable_reasons))[:2])
            )

    mean_delta = round(float(np.mean(valid_deltas)), 6) if usable_folds > 0 else None
    std_delta = round(float(np.std(valid_deltas, ddof=1)), 6) if usable_folds > 1 else (0.0 if usable_folds == 1 else None)

    return PairedMetricDifference(
        metric_name=metric_name,
        direction=metric_dir,
        raw_deltas=raw_deltas,
        mean_delta=mean_delta,
        std_delta=std_delta,
        expected_folds=expected_folds,
        paired_folds=paired_folds,
        usable_folds=usable_folds,
        is_usable=is_usable,
        unusable_reason=unusable_reason,
    )


def nadeau_bengio_corrected_t_test(
    paired_diff: PairedMetricDifference,
    test_train_ratio: float,
    alpha: float = 0.05,
    split_strategy: str = "stratified_k_fold",
) -> StatisticalTestResult:
    """Perform Nadeau and Bengio (2003) corrected resampled t-test for paired CV differences.

    Corrected standard error:
        SE_corr = sqrt((1/m + n_test/n_train) * s_d^2)
        t = mean_d / SE_corr
        df = m - 1
    """
    test_name = "Nadeau-Bengio Corrected Resampled t-test"
    assumptions = [
        "Corrects standard error for overlapping training sets across cross-validation folds.",
        "Formulation is parameterized for single K-fold cross-validation where validation sets partition the data without replacement.",
        "Repeated random train/test splits and repeated K-fold CV are distinct designs requiring different degree-of-freedom and variance formulation.",
        "Assumes paired fold differences follow an approximately normal distribution.",
        "Reference: Nadeau, C., & Bengio, Y. (2003). Inference for the Generalization Error. Machine Learning, 52, 239-281.",
        "Caution: On single K-fold cross-validation with K=5, df=4 has low statistical power.",
    ]

    # Pre-validation of protocol compatibility
    if split_strategy not in SUPPORTED_SPLIT_STRATEGIES:
        return StatisticalTestResult(
            test_name=test_name,
            status=TestStatus.UNAVAILABLE,
            alpha=alpha,
            assumptions_and_limitations=assumptions,
            unavailable_reason=(
                f"Unsupported evaluation protocol '{split_strategy}' for Nadeau-Bengio corrected t-test. "
                f"Supported protocols: {sorted(SUPPORTED_SPLIT_STRATEGIES)}."
            ),
        )

    # Pre-validation of paired differences usability
    if not paired_diff.is_usable:
        return StatisticalTestResult(
            test_name=test_name,
            status=TestStatus.UNAVAILABLE,
            alpha=alpha,
            assumptions_and_limitations=assumptions,
            unavailable_reason=paired_diff.unusable_reason or "Paired metric difference is not usable.",
        )

    if test_train_ratio <= 0.0 or not math.isfinite(test_train_ratio):
        return StatisticalTestResult(
            test_name=test_name,
            status=TestStatus.UNAVAILABLE,
            alpha=alpha,
            assumptions_and_limitations=assumptions,
            unavailable_reason=f"Invalid test/train sample size ratio: {test_train_ratio}. Must be finite and positive.",
        )

    valid_deltas = [d for d in paired_diff.raw_deltas if d is not None]
    m = len(valid_deltas)
    if m < 2:
        return StatisticalTestResult(
            test_name=test_name,
            status=TestStatus.UNAVAILABLE,
            alpha=alpha,
            assumptions_and_limitations=assumptions,
            unavailable_reason=f"Insufficient paired observations: m={m} < 2.",
        )

    # Validate that all deltas are strictly finite
    if any(not math.isfinite(d) for d in valid_deltas):
        return StatisticalTestResult(
            test_name=test_name,
            status=TestStatus.UNAVAILABLE,
            alpha=alpha,
            assumptions_and_limitations=assumptions,
            unavailable_reason="Non-finite values encountered in paired metric differences.",
        )

    arr = np.array(valid_deltas, dtype=float)
    mean_d = float(np.mean(arr))
    var_d = float(np.var(arr, ddof=1))
    df = float(m - 1)

    if not math.isfinite(mean_d) or not math.isfinite(var_d):
        return StatisticalTestResult(
            test_name=test_name,
            status=TestStatus.UNAVAILABLE,
            degrees_of_freedom=df,
            alpha=alpha,
            assumptions_and_limitations=assumptions,
            unavailable_reason="Sample mean difference or variance evaluated to a non-finite value.",
        )

    # Edge case: zero variance
    if var_d == 0.0:
        if mean_d == 0.0:
            return StatisticalTestResult(
                test_name=test_name,
                status=TestStatus.COMPLETED,
                statistic=0.0,
                p_value=1.0,
                degrees_of_freedom=df,
                alpha=alpha,
                statistically_significant=False,
                alternative_hypothesis="two-sided",
                assumptions_and_limitations=assumptions,
                warnings=["Zero variance across all folds: candidate and baseline produced identical scores on every fold."],
            )
        else:
            return StatisticalTestResult(
                test_name=test_name,
                status=TestStatus.UNAVAILABLE,
                degrees_of_freedom=df,
                alpha=alpha,
                assumptions_and_limitations=assumptions,
                unavailable_reason="Zero variance across folds with non-zero mean difference; standard error is zero.",
            )

    # Corrected Standard Error calculation
    corr_factor = (1.0 / float(m)) + float(test_train_ratio)
    if not math.isfinite(corr_factor) or corr_factor <= 0.0:
        return StatisticalTestResult(
            test_name=test_name,
            status=TestStatus.UNAVAILABLE,
            degrees_of_freedom=df,
            alpha=alpha,
            assumptions_and_limitations=assumptions,
            unavailable_reason=f"Invalid correction factor: {corr_factor}.",
        )

    se_corr = math.sqrt(corr_factor * var_d)

    if not math.isfinite(se_corr) or se_corr == 0.0:
        return StatisticalTestResult(
            test_name=test_name,
            status=TestStatus.UNAVAILABLE,
            degrees_of_freedom=df,
            alpha=alpha,
            assumptions_and_limitations=assumptions,
            unavailable_reason="Corrected standard error evaluated to zero or non-finite.",
        )

    t_stat = mean_d / se_corr
    p_val = float(stats.t.sf(abs(t_stat), df=df) * 2.0)

    if not math.isfinite(t_stat) or not math.isfinite(p_val) or np.isnan(p_val):
        return StatisticalTestResult(
            test_name=test_name,
            status=TestStatus.UNAVAILABLE,
            degrees_of_freedom=df,
            alpha=alpha,
            assumptions_and_limitations=assumptions,
            unavailable_reason="Calculated test statistic or p-value evaluated to non-finite value.",
        )

    # Clip to [0.0, 1.0]
    p_val = min(max(p_val, 0.0), 1.0)
    is_significant = bool(p_val < alpha)

    warnings: List[str] = []
    if m <= 5:
        warnings.append(
            f"Low degrees of freedom (df={int(df)}). Single {m}-fold CV has conservative statistical power; "
            "failure to reject null (p >= 0.05) does not demonstrate equivalence."
        )

    return StatisticalTestResult(
        test_name=test_name,
        status=TestStatus.COMPLETED,
        statistic=round(t_stat, 6),
        p_value=round(p_val, 6),
        degrees_of_freedom=df,
        alpha=alpha,
        statistically_significant=is_significant,
        alternative_hypothesis="two-sided",
        assumptions_and_limitations=assumptions,
        warnings=warnings,
    )


def wilcoxon_signed_rank_test(
    paired_diff: PairedMetricDifference,
    alpha: float = 0.05,
) -> StatisticalTestResult:
    """Perform Wilcoxon signed-rank test on paired per-fold metric differences.

    Note: Fold differences in cross-validation violate the i.i.d. assumption.
    This test is provided as a non-parametric diagnostic only.
    """
    test_name = "Wilcoxon Signed-Rank Test"
    assumptions = [
        "Non-parametric paired comparison evaluating median difference between paired observations.",
        "CRITICAL LIMITATION: Standard cross-validation folds violate the independence assumption "
        "due to overlapping training sets. Wilcoxon does NOT adjust for fold correlation.",
        "Zero differences are handled using zero_method='wilcox' (discarded from ranking).",
        "Alternative hypothesis: two-sided (distribution of differences is not symmetric about zero).",
    ]

    if not paired_diff.is_usable:
        return StatisticalTestResult(
            test_name=test_name,
            status=TestStatus.UNAVAILABLE,
            alpha=alpha,
            assumptions_and_limitations=assumptions,
            unavailable_reason=paired_diff.unusable_reason or "Paired metric difference is not usable.",
        )

    valid_deltas = [d for d in paired_diff.raw_deltas if d is not None]
    if not valid_deltas or any(not math.isfinite(d) for d in valid_deltas):
        return StatisticalTestResult(
            test_name=test_name,
            status=TestStatus.UNAVAILABLE,
            alpha=alpha,
            assumptions_and_limitations=assumptions,
            unavailable_reason="Non-finite values encountered in paired differences.",
        )

    arr = np.array(valid_deltas, dtype=float)

    # All-zero differences check
    if np.all(arr == 0.0):
        return StatisticalTestResult(
            test_name=test_name,
            status=TestStatus.COMPLETED,
            statistic=0.0,
            p_value=1.0,
            degrees_of_freedom=None,
            alpha=alpha,
            statistically_significant=False,
            alternative_hypothesis="two-sided",
            assumptions_and_limitations=assumptions,
            warnings=["All paired differences are zero."],
        )

    nonzero_count = int(np.sum(arr != 0.0))
    if nonzero_count == 0:
        return StatisticalTestResult(
            test_name=test_name,
            status=TestStatus.COMPLETED,
            statistic=0.0,
            p_value=1.0,
            alpha=alpha,
            statistically_significant=False,
            alternative_hypothesis="two-sided",
            assumptions_and_limitations=assumptions,
        )

    warnings: List[str] = [
        "Fold dependence caveat: Cross-validation training set overlap violates the Wilcoxon i.i.d. assumption."
    ]
    if nonzero_count < 6:
        warnings.append(
            f"Effective sample size after discarding zero differences is very small (n={nonzero_count} < 6). "
            f"For an exact two-sided Wilcoxon test at alpha={alpha}, n <= 5 cannot mathematically achieve p < {alpha} "
            f"(minimum attainable p-value at n=5 is 0.0625)."
        )

    try:
        res = stats.wilcoxon(arr, zero_method="wilcox", alternative="two-sided")
        stat_val = float(res.statistic)
        p_val = float(res.pvalue)

        if not math.isfinite(stat_val) or not math.isfinite(p_val) or np.isnan(p_val):
            return StatisticalTestResult(
                test_name=test_name,
                status=TestStatus.UNAVAILABLE,
                alpha=alpha,
                assumptions_and_limitations=assumptions,
                unavailable_reason="Calculated Wilcoxon statistic or p-value evaluated to non-finite value.",
                warnings=warnings,
            )

        p_val = min(max(p_val, 0.0), 1.0)
        is_significant = bool(p_val < alpha)

        return StatisticalTestResult(
            test_name=test_name,
            status=TestStatus.COMPLETED,
            statistic=round(stat_val, 6),
            p_value=round(p_val, 6),
            degrees_of_freedom=None,
            alpha=alpha,
            statistically_significant=is_significant,
            alternative_hypothesis="two-sided",
            assumptions_and_limitations=assumptions,
            warnings=warnings,
        )
    except Exception as exc:
        return StatisticalTestResult(
            test_name=test_name,
            status=TestStatus.UNAVAILABLE,
            alpha=alpha,
            assumptions_and_limitations=assumptions,
            unavailable_reason=f"Wilcoxon signed-rank test calculation error: {exc}",
            warnings=warnings,
        )


def compare_model_evaluations(
    candidate: ModelEvaluationResult,
    baseline: ModelEvaluationResult,
    dataset_metadata: DatasetMetadata,
    reproducibility: ReproducibilityMetadata,
    alpha: float = 0.05,
) -> ComparisonReport:
    """Compare candidate and baseline evaluations with paired differences and statistical tests.

    Parameters
    ----------
    candidate : ModelEvaluationResult
        Evaluated candidate model.
    baseline : ModelEvaluationResult
        Evaluated baseline model.
    dataset_metadata : DatasetMetadata
        Dataset metadata including cryptographic SHA256 checksum.
    reproducibility : ReproducibilityMetadata
        Reproducibility settings including split strategy and fold counts.
    alpha : float, default=0.05
        Statistical significance threshold.

    Returns
    -------
    ComparisonReport
        Structured comparison report containing paired differences, t-test, and Wilcoxon outcomes.

    Raises
    ------
    IncompatibleEvaluationPairError
        If evaluations are incompatible for paired comparison.
    """
    # 1. Validate pairing compatibility
    validate_evaluation_compatibility(
        candidate=candidate,
        baseline=baseline,
        dataset_metadata=dataset_metadata,
        reproducibility=reproducibility,
    )

    # 2. Derive average test/train sample size ratio from fold partitions
    ratios = [
        float(cf.val_indices_count) / float(cf.train_indices_count)
        for cf in candidate.fold_results
    ]
    test_train_ratio = float(np.mean(ratios))

    # 3. Identify all distinct scalar metrics across candidate folds
    metric_keys = sorted(list(candidate.fold_results[0].metrics.keys()))
    metric_comparisons: Dict[str, MetricComparisonResult] = {}
    report_warnings: List[str] = []

    for m_name in metric_keys:
        direction = METRIC_DIRECTIONS.get(m_name, MetricDirection.NOT_APPLICABLE)

        # Compute paired per-fold difference
        paired_diff = compute_paired_metric_differences(
            candidate_folds=candidate.fold_results,
            baseline_folds=baseline.fold_results,
            metric_name=m_name,
            direction=direction,
        )

        # Run Nadeau-Bengio corrected resampled t-test
        t_res = nadeau_bengio_corrected_t_test(
            paired_diff=paired_diff,
            test_train_ratio=test_train_ratio,
            alpha=alpha,
            split_strategy=reproducibility.split_strategy,
        )

        # Run Wilcoxon signed-rank test
        w_res = wilcoxon_signed_rank_test(
            paired_diff=paired_diff,
            alpha=alpha,
        )

        if not paired_diff.is_usable and paired_diff.unusable_reason:
            report_warnings.append(f"Metric '{m_name}': {paired_diff.unusable_reason}")

        metric_comparisons[m_name] = MetricComparisonResult(
            metric_name=m_name,
            direction=direction,
            paired_differences=paired_diff,
            corrected_resampled_t_test=t_res,
            wilcoxon_signed_rank_test=w_res,
        )

    # Check for partition fingerprint verification status across all paired folds
    cand_map = {cf.fold_index: cf for cf in candidate.fold_results}
    base_map = {bf.fold_index: bf for bf in baseline.fold_results}
    expected_ids = sorted(cand_map.keys())

    verified_folds = [
        fid for fid in expected_ids
        if cand_map[fid].partition_fingerprint is not None
        and base_map[fid].partition_fingerprint is not None
        and cand_map[fid].partition_fingerprint == base_map[fid].partition_fingerprint
    ]

    if len(verified_folds) == 0:
        report_warnings.append(
            "Partition verification notice: Evaluated folds did not contain cryptographic partition fingerprints; "
            "pairing was verified by sample counts only."
        )
    elif len(verified_folds) < len(expected_ids):
        report_warnings.append(
            f"Partition verification notice: Only {len(verified_folds)} of {len(expected_ids)} folds "
            "contained cryptographic partition fingerprints. Partial partition verification achieved."
        )

    # Check for identical model configuration warning
    if candidate.model_type == baseline.model_type and candidate.hyperparameters == baseline.hyperparameters:
        report_warnings.append(
            "Identical model configuration notice: Candidate and baseline share identical model type and hyperparameters."
        )

    return ComparisonReport(
        candidate_model_name=candidate.model_name,
        baseline_model_name=baseline.model_name,
        dataset_checksum=dataset_metadata.checksum_sha256,
        num_folds=reproducibility.num_folds,
        metric_comparisons=metric_comparisons,
        warnings=list(set(report_warnings)),
    )
