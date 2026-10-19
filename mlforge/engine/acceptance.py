"""MLForge Acceptance Decision Engine Module.

Evaluates trained and evaluated candidate models against versioned, multi-criteria
acceptance policies to yield an auditable, deterministic verdict: PASS, FAIL, or INCONCLUSIVE.
"""

from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Dict, List, Optional
import uuid

from mlforge.core.contracts import (
    AcceptanceDecision,
    AcceptanceOutcome,
    AcceptancePolicy,
    AcceptanceRule,
    CriterionEvaluation,
    CriterionOutcome,
    CriterionSeverity,
    EvaluationResult,
    MetricDirection,
    PerturbationType,
    RobustnessEvaluationResult,
    RuleOperator,
    TestStatus,
)


def _evaluate_candidate_metric_rule(
    rule: AcceptanceRule,
    evaluation: EvaluationResult,
) -> CriterionEvaluation:
    """Evaluate a criterion comparing a candidate summary metric against a scalar threshold."""
    metric_name = rule.metric
    cand_summaries = evaluation.candidate_result.metric_summaries

    if metric_name not in cand_summaries:
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=metric_name,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=rule.threshold,
            observed_value=None,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="EVIDENCE_MISSING",
            message=f"Candidate summary metric '{metric_name}' was not computed in evaluation.",
            evidence_source=rule.evidence_source,
        )

    summary = cand_summaries[metric_name]

    if summary.mean is None or summary.valid_fold_count == 0:
        reasons = "; ".join(summary.undefined_reasons) if summary.undefined_reasons else "all folds undefined"
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=metric_name,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=rule.threshold,
            observed_value=None,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="METRIC_UNDEFINED",
            message=f"Metric '{metric_name}' is undefined across evaluation folds: {reasons}.",
            evidence_source=rule.evidence_source,
        )

    val = summary.mean
    if math.isnan(val) or math.isinf(val):
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=metric_name,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=rule.threshold,
            observed_value=None,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="NON_FINITE_VALUE",
            message=f"Observed value for '{metric_name}' is non-finite ({val}).",
            evidence_source=rule.evidence_source,
        )

    threshold = rule.threshold
    if threshold is None:
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=metric_name,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=None,
            observed_value=val,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="THRESHOLD_MISSING",
            message=f"Rule '{rule.rule_id}' did not specify a numerical threshold.",
            evidence_source=rule.evidence_source,
        )

    satisfied = False
    if rule.operator in (RuleOperator.GREATER_THAN_OR_EQUAL, RuleOperator.STATISTICALLY_GREATER_THAN):
        satisfied = val >= threshold
    elif rule.operator in (RuleOperator.LESS_THAN_OR_EQUAL, RuleOperator.STATISTICALLY_LESS_THAN):
        satisfied = val <= threshold
    elif rule.operator == RuleOperator.GREATER_THAN:
        satisfied = val > threshold
    elif rule.operator == RuleOperator.LESS_THAN:
        satisfied = val < threshold
    elif rule.operator == RuleOperator.EQUALS:
        satisfied = math.isclose(val, threshold, abs_tol=1e-9)
    elif rule.operator == RuleOperator.NOT_EQUALS:
        satisfied = not math.isclose(val, threshold, abs_tol=1e-9)
    else:
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=metric_name,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=threshold,
            observed_value=val,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="UNSUPPORTED_OPERATOR",
            message=f"Operator '{rule.operator.value}' is not supported for candidate metric evaluation.",
            evidence_source=rule.evidence_source,
        )

    outcome = CriterionOutcome.SATISFIED if satisfied else CriterionOutcome.VIOLATED
    code = "THRESHOLD_MET" if satisfied else "THRESHOLD_BREACHED"
    verb = "satisfies" if satisfied else "violates"
    msg = f"Observed {metric_name} ({val:.4f}) {verb} requirement ({rule.operator.value} {threshold})."

    return CriterionEvaluation(
        rule_id=rule.rule_id,
        metric=metric_name,
        severity=rule.severity,
        operator=rule.operator,
        threshold_value=threshold,
        observed_value=val,
        outcome=outcome,
        reason_code=code,
        message=msg,
        evidence_source=rule.evidence_source,
    )


def _evaluate_calibration_rule(
    rule: AcceptanceRule,
    evaluation: EvaluationResult,
) -> CriterionEvaluation:
    """Evaluate a calibration diagnostic criterion (e.g. ECE, Brier score, log loss)."""
    calib = evaluation.candidate_result.calibration
    metric_name = rule.metric.lower()

    if calib is None or not calib.is_usable:
        reason = getattr(calib, "unusable_reason", None) or "calibration diagnostics not available"
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=rule.metric,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=rule.threshold,
            observed_value=None,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="CALIBRATION_UNAVAILABLE",
            message=f"Calibration evidence for '{rule.metric}' unavailable: {reason}.",
            evidence_source=rule.evidence_source,
        )

    val: Optional[float] = None
    if metric_name in ("expected_calibration_error", "ece"):
        val = calib.expected_calibration_error
    elif metric_name in ("max_calibration_error", "mce"):
        val = calib.max_calibration_error
    elif metric_name in ("brier_score", "brier"):
        val = calib.brier_score
    elif metric_name in ("log_loss", "cross_entropy"):
        val = calib.log_loss
    else:
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=rule.metric,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=rule.threshold,
            observed_value=None,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="UNKNOWN_CALIBRATION_METRIC",
            message=f"Unknown calibration metric identifier: '{rule.metric}'.",
            evidence_source=rule.evidence_source,
        )

    if val is None or math.isnan(val) or math.isinf(val):
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=rule.metric,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=rule.threshold,
            observed_value=None,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="METRIC_UNDEFINED",
            message=f"Calibration metric '{rule.metric}' evaluated to non-finite or None value.",
            evidence_source=rule.evidence_source,
        )

    threshold = rule.threshold
    if threshold is None:
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=rule.metric,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=None,
            observed_value=val,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="THRESHOLD_MISSING",
            message=f"Calibration rule '{rule.rule_id}' did not specify a numerical threshold.",
            evidence_source=rule.evidence_source,
        )

    satisfied = False
    if rule.operator in (RuleOperator.LESS_THAN_OR_EQUAL, RuleOperator.STATISTICALLY_LESS_THAN):
        satisfied = val <= threshold
    elif rule.operator == RuleOperator.LESS_THAN:
        satisfied = val < threshold
    elif rule.operator in (RuleOperator.GREATER_THAN_OR_EQUAL, RuleOperator.STATISTICALLY_GREATER_THAN):
        satisfied = val >= threshold
    elif rule.operator == RuleOperator.GREATER_THAN:
        satisfied = val > threshold
    elif rule.operator == RuleOperator.EQUALS:
        satisfied = math.isclose(val, threshold, abs_tol=1e-9)
    elif rule.operator == RuleOperator.NOT_EQUALS:
        satisfied = not math.isclose(val, threshold, abs_tol=1e-9)
    else:
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=rule.metric,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=threshold,
            observed_value=val,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="UNSUPPORTED_OPERATOR",
            message=f"Operator '{rule.operator.value}' is not supported for calibration evaluation.",
            evidence_source=rule.evidence_source,
        )

    outcome = CriterionOutcome.SATISFIED if satisfied else CriterionOutcome.VIOLATED
    code = "CALIBRATION_THRESHOLD_MET" if satisfied else "CALIBRATION_THRESHOLD_BREACHED"
    verb = "satisfies" if satisfied else "breaches"
    msg = f"Observed calibration {rule.metric} ({val:.4f}) {verb} required tolerance ({rule.operator.value} {threshold})."

    return CriterionEvaluation(
        rule_id=rule.rule_id,
        metric=rule.metric,
        severity=rule.severity,
        operator=rule.operator,
        threshold_value=threshold,
        observed_value=val,
        outcome=outcome,
        reason_code=code,
        message=msg,
        evidence_source=rule.evidence_source,
    )


def _evaluate_baseline_contrast_rule(
    rule: AcceptanceRule,
    evaluation: EvaluationResult,
) -> CriterionEvaluation:
    """Evaluate candidate improvement over baseline on a specific metric."""
    report = evaluation.comparison_report
    metric_name = rule.metric

    if report is None:
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=metric_name,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=rule.threshold,
            observed_value=None,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="COMPARISON_UNAVAILABLE",
            message="No paired baseline comparison report was generated.",
            evidence_source=rule.evidence_source,
        )

    if metric_name not in report.metric_comparisons:
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=metric_name,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=rule.threshold,
            observed_value=None,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="COMPARISON_METRIC_MISSING",
            message=f"Metric '{metric_name}' was not compared between candidate and baseline.",
            evidence_source=rule.evidence_source,
        )

    comp = report.metric_comparisons[metric_name]
    diffs = comp.paired_differences

    if not diffs.is_usable or diffs.mean_delta is None:
        reason = diffs.unusable_reason or "paired differences unusable"
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=metric_name,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=rule.threshold,
            observed_value=None,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="COMPARISON_UNDEFINED",
            message=f"Paired comparison for '{metric_name}' is undefined: {reason}.",
            evidence_source=rule.evidence_source,
        )

    mean_delta = diffs.mean_delta
    if math.isnan(mean_delta) or math.isinf(mean_delta):
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=metric_name,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=rule.threshold,
            observed_value=None,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="NON_FINITE_VALUE",
            message=f"Mean paired delta for '{metric_name}' is non-finite ({mean_delta}).",
            evidence_source=rule.evidence_source,
        )

    # 1. Special case: Confidence Interval Disjoint Superiority
    if rule.operator == RuleOperator.CI_DISJOINT_SUPERIOR:
        cand_summary = evaluation.candidate_result.metric_summaries.get(metric_name)
        base_summary = evaluation.baseline_result.metric_summaries.get(metric_name)

        if (
            cand_summary is None
            or base_summary is None
            or cand_summary.confidence_interval_95 is None
            or base_summary.confidence_interval_95 is None
        ):
            return CriterionEvaluation(
                rule_id=rule.rule_id,
                metric=metric_name,
                severity=rule.severity,
                operator=rule.operator,
                threshold_value=None,
                observed_value=mean_delta,
                outcome=CriterionOutcome.INCONCLUSIVE,
                reason_code="CONFIDENCE_INTERVAL_UNAVAILABLE",
                message=f"95% confidence intervals unavailable for candidate or baseline on '{metric_name}'.",
                evidence_source=rule.evidence_source,
            )

        cand_ci = cand_summary.confidence_interval_95
        base_ci = base_summary.confidence_interval_95

        if comp.direction == MetricDirection.HIGHER_IS_BETTER:
            # Candidate lower bound strictly exceeds baseline upper bound
            satisfied = cand_ci[0] > base_ci[1]
        elif comp.direction == MetricDirection.LOWER_IS_BETTER:
            # Candidate upper bound strictly lower than baseline lower bound
            satisfied = cand_ci[1] < base_ci[0]
        else:
            satisfied = cand_ci[0] > base_ci[1]

        outcome = CriterionOutcome.SATISFIED if satisfied else CriterionOutcome.VIOLATED
        code = "CI_DISJOINT_MET" if satisfied else "CI_OVERLAPPING_OR_INFERIOR"
        ci_str = f"Cand 95% CI [{cand_ci[0]:.4f}, {cand_ci[1]:.4f}] vs Base 95% CI [{base_ci[0]:.4f}, {base_ci[1]:.4f}]"
        verb = "demonstrates disjoint superiority over" if satisfied else "fails to achieve disjoint superiority over"
        msg = f"Candidate {verb} baseline on {metric_name} ({ci_str})."

        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=metric_name,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=None,
            observed_value=mean_delta,
            outcome=outcome,
            reason_code=code,
            message=msg,
            evidence_source=rule.evidence_source,
        )

    # 2. Signed Delta Threshold Comparison: raw_delta is candidate - baseline
    threshold = rule.threshold
    if threshold is None:
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=metric_name,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=None,
            observed_value=mean_delta,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="THRESHOLD_MISSING",
            message=f"Baseline contrast rule '{rule.rule_id}' did not specify a numerical delta threshold.",
            evidence_source=rule.evidence_source,
        )

    satisfied = False
    if rule.operator in (RuleOperator.GREATER_THAN_OR_EQUAL, RuleOperator.STATISTICALLY_GREATER_THAN):
        satisfied = mean_delta >= threshold
    elif rule.operator == RuleOperator.GREATER_THAN:
        satisfied = mean_delta > threshold
    elif rule.operator in (RuleOperator.LESS_THAN_OR_EQUAL, RuleOperator.STATISTICALLY_LESS_THAN):
        satisfied = mean_delta <= threshold
    elif rule.operator == RuleOperator.LESS_THAN:
        satisfied = mean_delta < threshold
    elif rule.operator == RuleOperator.EQUALS:
        satisfied = math.isclose(mean_delta, threshold, abs_tol=1e-9)
    elif rule.operator == RuleOperator.NOT_EQUALS:
        satisfied = not math.isclose(mean_delta, threshold, abs_tol=1e-9)
    else:
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=metric_name,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=threshold,
            observed_value=mean_delta,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="UNSUPPORTED_OPERATOR",
            message=f"Operator '{rule.operator.value}' is not supported for baseline contrast evaluation.",
            evidence_source=rule.evidence_source,
        )

    outcome = CriterionOutcome.SATISFIED if satisfied else CriterionOutcome.VIOLATED
    code = "BASELINE_DELTA_MET" if satisfied else "BASELINE_DELTA_BREACHED"
    verb = "satisfies" if satisfied else "violates"
    msg = (
        f"Observed mean delta on {metric_name} ({mean_delta:+.4f}) {verb} baseline delta requirement "
        f"({rule.operator.value} {threshold:+.4f}) [direction: {comp.direction.value}]."
    )

    return CriterionEvaluation(
        rule_id=rule.rule_id,
        metric=metric_name,
        severity=rule.severity,
        operator=rule.operator,
        threshold_value=threshold,
        observed_value=mean_delta,
        outcome=outcome,
        reason_code=code,
        message=msg,
        evidence_source=rule.evidence_source,
    )


def _evaluate_statistical_test_rule(
    rule: AcceptanceRule,
    evaluation: EvaluationResult,
) -> CriterionEvaluation:
    """Evaluate statistical hypothesis test evidence (p-value, significance, direction)."""
    report = evaluation.comparison_report
    metric_name = rule.metric

    if report is None or metric_name not in report.metric_comparisons:
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=metric_name,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=rule.threshold,
            observed_value=None,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="STATISTICAL_TEST_UNAVAILABLE",
            message=f"Statistical comparison evidence unavailable for '{metric_name}'.",
            evidence_source=rule.evidence_source,
        )

    comp = report.metric_comparisons[metric_name]
    test = comp.corrected_resampled_t_test

    if test is None or test.status != TestStatus.COMPLETED or test.p_value is None:
        reason = getattr(test, "unavailable_reason", None) or "statistical test not completed"
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=metric_name,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=rule.threshold,
            observed_value=None,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="STATISTICAL_TEST_UNAVAILABLE",
            message=f"Corrected t-test on '{metric_name}' is unavailable: {reason}.",
            evidence_source=rule.evidence_source,
        )

    p_val = test.p_value
    if (
        p_val is None
        or not isinstance(p_val, (int, float))
        or math.isnan(p_val)
        or math.isinf(p_val)
        or p_val < 0.0
        or p_val > 1.0
    ):
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=metric_name,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=rule.threshold,
            observed_value=p_val if isinstance(p_val, (int, float)) and not math.isnan(p_val) else None,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="INVALID_P_VALUE",
            message=f"Statistical test on '{metric_name}' produced an invalid or non-finite p-value: {p_val}.",
            evidence_source=rule.evidence_source,
        )

    # Statistical significance alpha: explicitly from rule.alpha, or operator=STATISTICALLY_SIGNIFICANT threshold, or test.alpha, or default 0.05
    rule_alpha = getattr(rule, "alpha", None)
    if rule_alpha is not None:
        alpha_thresh = rule_alpha
    elif (
        rule.operator == RuleOperator.STATISTICALLY_SIGNIFICANT
        and getattr(rule, "min_effect_size", None) is None
        and rule.threshold is not None
    ):
        alpha_thresh = rule.threshold
    elif test.alpha is not None:
        alpha_thresh = test.alpha
    else:
        alpha_thresh = 0.05

    if (
        alpha_thresh is None
        or not isinstance(alpha_thresh, (int, float))
        or math.isnan(alpha_thresh)
        or math.isinf(alpha_thresh)
        or alpha_thresh <= 0.0
        or alpha_thresh >= 1.0
    ):
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=metric_name,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=None,
            observed_value=p_val,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="INVALID_ALPHA",
            message=f"Statistical test rule '{rule.rule_id}' has invalid alpha threshold ({alpha_thresh}).",
            evidence_source=rule.evidence_source,
        )

    # Minimum required effect size (mean difference): explicit from rule.min_effect_size or rule.threshold
    rule_min_effect = getattr(rule, "min_effect_size", None)
    if rule_min_effect is not None:
        min_effect = rule_min_effect
    elif rule.operator != RuleOperator.STATISTICALLY_SIGNIFICANT and rule.threshold is not None:
        min_effect = rule.threshold
    elif (
        rule.operator == RuleOperator.STATISTICALLY_SIGNIFICANT
        and rule_alpha is not None
        and rule.threshold is not None
    ):
        min_effect = rule.threshold
    else:
        min_effect = 0.0

    # Direction and effect size check
    mean_delta = comp.paired_differences.mean_delta
    if (
        mean_delta is None
        or not isinstance(mean_delta, (int, float))
        or math.isnan(mean_delta)
        or math.isinf(mean_delta)
    ):
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=metric_name,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=rule.threshold if rule.threshold is not None else alpha_thresh,
            observed_value=None,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="NON_FINITE_DELTA",
            message=f"Paired delta on '{metric_name}' is undefined or non-finite: {mean_delta}.",
            evidence_source=rule.evidence_source,
        )

    is_significant = p_val < alpha_thresh

    if not is_significant:
        outcome = CriterionOutcome.VIOLATED
        code = "SIGNIFICANCE_UNMET"
        msg = (
            f"Statistical significance threshold not met on {metric_name} "
            f"(p={p_val:.4f} >= alpha {alpha_thresh:.4f}). "
            "Note: failure to reject H0 indicates lack of evidence, not proof of equivalence."
        )
    else:
        if rule.operator in (
            RuleOperator.STATISTICALLY_GREATER_THAN,
            RuleOperator.GREATER_THAN,
            RuleOperator.GREATER_THAN_OR_EQUAL,
        ):
            if mean_delta <= 0.0:
                outcome = CriterionOutcome.VIOLATED
                code = "SIGNIFICANT_REGRESSION"
                msg = (
                    f"Candidate exhibits statistically significant divergence/regression on {metric_name} "
                    f"(p={p_val:.4f} < alpha {alpha_thresh:.4f}, mean delta={mean_delta:+.4f} <= 0)."
                )
            elif mean_delta <= min_effect:
                outcome = CriterionOutcome.VIOLATED
                code = "EFFECT_SIZE_UNMET"
                msg = (
                    f"Candidate improvement on {metric_name} is statistically significant "
                    f"(p={p_val:.4f} < alpha {alpha_thresh:.4f}) but mean delta ({mean_delta:+.4f}) "
                    f"does not meet minimum required effect size of {min_effect:.4f}."
                )
            else:
                outcome = CriterionOutcome.SATISFIED
                code = "SIGNIFICANT_IMPROVEMENT"
                msg = (
                    f"Candidate achieves statistically significant improvement on {metric_name} "
                    f"(p={p_val:.4f} < alpha {alpha_thresh:.4f}, mean delta={mean_delta:+.4f} > min effect {min_effect:.4f})."
                )

        elif rule.operator in (
            RuleOperator.STATISTICALLY_LESS_THAN,
            RuleOperator.LESS_THAN,
            RuleOperator.LESS_THAN_OR_EQUAL,
        ):
            if mean_delta >= 0.0:
                outcome = CriterionOutcome.VIOLATED
                code = "SIGNIFICANT_REGRESSION"
                msg = (
                    f"Candidate exhibits statistically significant divergence on {metric_name} "
                    f"(p={p_val:.4f} < alpha {alpha_thresh:.4f}, mean delta={mean_delta:+.4f} >= 0)."
                )
            elif mean_delta >= -min_effect:
                outcome = CriterionOutcome.VIOLATED
                code = "EFFECT_SIZE_UNMET"
                msg = (
                    f"Candidate reduction on {metric_name} is statistically significant "
                    f"(p={p_val:.4f} < alpha {alpha_thresh:.4f}) but mean delta ({mean_delta:+.4f}) "
                    f"does not meet minimum required magnitude of {-min_effect:.4f}."
                )
            else:
                outcome = CriterionOutcome.SATISFIED
                code = "SIGNIFICANT_IMPROVEMENT"
                msg = (
                    f"Candidate achieves statistically significant reduction on {metric_name} "
                    f"(p={p_val:.4f} < alpha {alpha_thresh:.4f}, mean delta={mean_delta:+.4f} < {-min_effect:.4f})."
                )

        elif rule.operator in (RuleOperator.STATISTICALLY_SIGNIFICANT, RuleOperator.NOT_EQUALS):
            if min_effect > 0.0 and abs(mean_delta) <= min_effect:
                outcome = CriterionOutcome.VIOLATED
                code = "EFFECT_SIZE_UNMET"
                msg = (
                    f"Difference on {metric_name} is statistically significant "
                    f"(p={p_val:.4f} < alpha {alpha_thresh:.4f}) but magnitude ({abs(mean_delta):.4f}) "
                    f"does not meet minimum required effect size of {min_effect:.4f}."
                )
            else:
                outcome = CriterionOutcome.SATISFIED
                code = "SIGNIFICANT_IMPROVEMENT"
                msg = (
                    f"Candidate achieves statistically significant difference on {metric_name} "
                    f"(p={p_val:.4f} < alpha {alpha_thresh:.4f}, mean delta={mean_delta:+.4f})."
                )
        else:
            return CriterionEvaluation(
                rule_id=rule.rule_id,
                metric=metric_name,
                severity=rule.severity,
                operator=rule.operator,
                threshold_value=rule.threshold if rule.threshold is not None else alpha_thresh,
                observed_value=p_val,
                outcome=CriterionOutcome.INCONCLUSIVE,
                reason_code="UNSUPPORTED_OPERATOR",
                message=f"Operator '{rule.operator.value}' is not supported for statistical test evaluation.",
                evidence_source=rule.evidence_source,
            )

    # Use explicit threshold and observed values based on rule operator
    if rule.operator == RuleOperator.STATISTICALLY_SIGNIFICANT and getattr(rule, "min_effect_size", None) is None:
        eval_thresh = alpha_thresh
        eval_obs = p_val
    else:
        eval_thresh = min_effect if rule.threshold is not None or getattr(rule, "min_effect_size", None) is not None else alpha_thresh
        eval_obs = mean_delta

    return CriterionEvaluation(
        rule_id=rule.rule_id,
        metric=metric_name,
        severity=rule.severity,
        operator=rule.operator,
        threshold_value=eval_thresh,
        observed_value=eval_obs,
        outcome=outcome,
        reason_code=code,
        message=msg,
        evidence_source=rule.evidence_source,
    )


def _evaluate_data_quality_rule(
    rule: AcceptanceRule,
    evaluation: EvaluationResult,
) -> CriterionEvaluation:
    """Evaluate data-quality and stability requirements (e.g. max undefined folds, fold std)."""
    metric_name = rule.metric
    cand = evaluation.candidate_result

    threshold = rule.threshold
    if threshold is None:
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=metric_name,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=None,
            observed_value=None,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="THRESHOLD_MISSING",
            message=f"Data quality rule '{rule.rule_id}' did not specify a numerical threshold.",
            evidence_source=rule.evidence_source,
        )

    if metric_name == "max_undefined_folds":
        max_allowed = threshold
        total_undefined = sum(
            s.undefined_fold_count for s in cand.metric_summaries.values()
        )
        if rule.operator in (RuleOperator.LESS_THAN_OR_EQUAL, RuleOperator.STATISTICALLY_LESS_THAN):
            satisfied = total_undefined <= max_allowed
        elif rule.operator == RuleOperator.LESS_THAN:
            satisfied = total_undefined < max_allowed
        elif rule.operator == RuleOperator.EQUALS:
            satisfied = math.isclose(total_undefined, max_allowed, abs_tol=1e-9)
        else:
            return CriterionEvaluation(
                rule_id=rule.rule_id,
                metric=metric_name,
                severity=rule.severity,
                operator=rule.operator,
                threshold_value=max_allowed,
                observed_value=float(total_undefined),
                outcome=CriterionOutcome.INCONCLUSIVE,
                reason_code="UNSUPPORTED_OPERATOR",
                message=f"Operator '{rule.operator.value}' is not supported for max_undefined_folds.",
                evidence_source=rule.evidence_source,
            )

        outcome = CriterionOutcome.SATISFIED if satisfied else CriterionOutcome.VIOLATED
        code = "QUALITY_CRITERION_MET" if satisfied else "EXCESSIVE_UNDEFINED_FOLDS"
        msg = f"Observed {total_undefined} total undefined fold instances (limit: {int(max_allowed)})."
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=metric_name,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=max_allowed,
            observed_value=float(total_undefined),
            outcome=outcome,
            reason_code=code,
            message=msg,
            evidence_source=rule.evidence_source,
        )

    elif metric_name == "max_fold_std":
        target_metric = "macro_f1" if "macro_f1" in cand.metric_summaries else "accuracy"
        summary = cand.metric_summaries.get(target_metric)
        if summary is None or summary.std is None:
            return CriterionEvaluation(
                rule_id=rule.rule_id,
                metric=metric_name,
                severity=rule.severity,
                operator=rule.operator,
                threshold_value=threshold,
                observed_value=None,
                outcome=CriterionOutcome.INCONCLUSIVE,
                reason_code="EVIDENCE_MISSING",
                message=f"Sample standard deviation unavailable for '{target_metric}'.",
                evidence_source=rule.evidence_source,
            )
        std_val = summary.std
        max_std = threshold
        if rule.operator in (RuleOperator.LESS_THAN_OR_EQUAL, RuleOperator.STATISTICALLY_LESS_THAN):
            satisfied = std_val <= max_std
        elif rule.operator == RuleOperator.LESS_THAN:
            satisfied = std_val < max_std
        elif rule.operator == RuleOperator.EQUALS:
            satisfied = math.isclose(std_val, max_std, abs_tol=1e-9)
        else:
            return CriterionEvaluation(
                rule_id=rule.rule_id,
                metric=metric_name,
                severity=rule.severity,
                operator=rule.operator,
                threshold_value=max_std,
                observed_value=std_val,
                outcome=CriterionOutcome.INCONCLUSIVE,
                reason_code="UNSUPPORTED_OPERATOR",
                message=f"Operator '{rule.operator.value}' is not supported for max_fold_std.",
                evidence_source=rule.evidence_source,
            )

        outcome = CriterionOutcome.SATISFIED if satisfied else CriterionOutcome.VIOLATED
        code = "STABILITY_MET" if satisfied else "EXCESSIVE_VARIANCE"
        msg = f"Observed fold std for {target_metric} ({std_val:.4f}) {'satisfies' if satisfied else 'exceeds'} stability ceiling ({max_std})."
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=metric_name,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=max_std,
            observed_value=std_val,
            outcome=outcome,
            reason_code=code,
            message=msg,
            evidence_source=rule.evidence_source,
        )

    else:
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=metric_name,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=threshold,
            observed_value=None,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="UNKNOWN_QUALITY_CRITERION",
            message=f"Unknown data quality criterion '{metric_name}'.",
            evidence_source=rule.evidence_source,
        )


def _extract_robustness_metric_value(
    metric_str: str,
    robustness: RobustnessEvaluationResult,
) -> tuple[Optional[float], Optional[str], Optional[str]]:
    """Extract observed numerical value for a robustness metric from evaluation evidence.

    Returns
    -------
    tuple[Optional[float], Optional[str], Optional[str]]
        (observed_value, error_code, error_message)
    """
    norm = metric_str.lower().strip()

    # 1. Direct match in summary_retention dictionary
    if norm in robustness.summary_retention:
        val = robustness.summary_retention[norm]
        if val is None or math.isnan(val) or math.isinf(val):
            return None, "NON_FINITE_VALUE", f"Summary robustness metric '{metric_str}' is non-finite or None."
        return val, None, None

    # Handle mean_aurdc specifically if curves are empty or incomplete
    if norm == "mean_aurdc":
        if not robustness.curves:
            return None, "ROBUSTNESS_EVIDENCE_MISSING", "No robustness curves were evaluated."
        return (
            None,
            "ROBUSTNESS_AURDC_UNAVAILABLE",
            "Mean AURDC is unavailable because one or more robustness curves are incomplete or unusable.",
        )

    # 2. Tokenized parsing (e.g. colon-separated or underscore-separated)
    alias_map = {
        "noise": "gaussian_noise",
        "missingness": "missingness_injection",
        "dropout": "category_dropout",
        "category": "category_dropout",
    }
    valid_types = {e.value for e in PerturbationType}

    # Normalize colon or slash delimiters
    parts = [p.strip() for p in norm.replace("/", ":").split(":") if p.strip()]

    # If 1 part but underscore format, e.g. gaussian_noise_max_drop_pct
    if len(parts) == 1:
        for pt_candidate in valid_types:
            if norm.startswith(pt_candidate + "_"):
                suffix = norm[len(pt_candidate) + 1 :]
                parts = [pt_candidate, suffix]
                break
        if len(parts) == 1:
            for alias_candidate, canon in alias_map.items():
                if norm.startswith(alias_candidate + "_"):
                    suffix = norm[len(alias_candidate) + 1 :]
                    parts = [canon, suffix]
                    break

    # If still 1 part and matches known perturbation type alone (ambiguous measure)
    if len(parts) == 1:
        canon_type = alias_map.get(parts[0], parts[0])
        if canon_type in valid_types:
            return (
                None,
                "AMBIGUOUS_ROBUSTNESS_METRIC",
                f"Robustness metric '{metric_str}' must specify a measurement "
                f"(e.g. '{parts[0]}_aurdc', '{parts[0]}_max_drop_pct', '{parts[0]}_max_retention', or '{parts[0]}:<severity>:drop_pct').",
            )
        return (
            None,
            "UNKNOWN_ROBUSTNESS_METRIC",
            f"Could not resolve robustness metric '{metric_str}'. Supported formats: 'mean_aurdc', '<type>_aurdc', '<type>_max_drop_pct', '<type>_max_retention', '<type>:<severity>:drop_pct', '<type>:<severity>:retention'.",
        )

    # If 2 parts: [perturbation_type, measurement]
    if len(parts) == 2:
        pt_token = None
        measure_token = None
        for p in parts:
            canon = alias_map.get(p, p)
            if canon in valid_types:
                pt_token = canon
            else:
                measure_token = p

        if pt_token is None or measure_token is None:
            return (
                None,
                "UNKNOWN_ROBUSTNESS_METRIC",
                f"Could not parse perturbation type and measurement from '{metric_str}'.",
            )

        matching_curves = [c for c in robustness.curves if c.perturbation_type.value == pt_token]
        if not matching_curves:
            return None, "CURVE_NOT_FOUND", f"Perturbation curve for '{pt_token}' was not evaluated."
        curve = matching_curves[0]

        if measure_token in ("aurdc", "area_under_degradation_curve"):
            if curve.area_under_degradation_curve is None:
                reasons = "; ".join(curve.warnings) or "curve contains unusable points or evaluation failures"
                return None, "ROBUSTNESS_AURDC_UNAVAILABLE", f"AURDC for '{pt_token}' is unavailable: {reasons}."
            return curve.area_under_degradation_curve, None, None

        if measure_token in (
            "max_drop",
            "max_drop_pct",
            "max_drop_percentage",
            "drop_pct",
            "performance_drop_percentage",
            "performance_drop",
        ):
            if not curve.points:
                return None, "POINTS_EMPTY", f"Perturbation curve for '{pt_token}' contains no evaluation points."
            max_pt = max(curve.points, key=lambda p: p.severity_level)
            if not max_pt.is_usable:
                reason = max_pt.unusable_reason or "perturbation evaluation failed"
                return (
                    None,
                    "POINT_UNUSABLE",
                    f"Perturbation point for '{pt_token}' at max severity ({max_pt.severity_level}) is unusable: {reason}.",
                )
            if max_pt.performance_drop_percentage is None:
                return (
                    None,
                    "METRIC_UNDEFINED",
                    f"Performance drop percentage is undefined for '{pt_token}' at max severity ({max_pt.severity_level}).",
                )
            return max_pt.performance_drop_percentage, None, None

        if measure_token in ("max_retention", "retention", "retained_ratio", "min_retention"):
            if not curve.points:
                return None, "POINTS_EMPTY", f"Perturbation curve for '{pt_token}' contains no evaluation points."
            max_pt = max(curve.points, key=lambda p: p.severity_level)
            if not max_pt.is_usable:
                reason = max_pt.unusable_reason or "perturbation evaluation failed"
                return (
                    None,
                    "POINT_UNUSABLE",
                    f"Perturbation point for '{pt_token}' at max severity ({max_pt.severity_level}) is unusable: {reason}.",
                )
            if max_pt.retained_ratio is None:
                return (
                    None,
                    "METRIC_UNDEFINED",
                    f"Retained ratio is undefined for '{pt_token}' at max severity ({max_pt.severity_level}).",
                )
            return max_pt.retained_ratio, None, None

        return None, "UNKNOWN_MEASUREMENT", f"Unknown robustness measurement '{measure_token}' for '{pt_token}'."

    # If 3 parts: [perturbation_type, severity, measurement]
    if len(parts) == 3:
        pt_token = None
        sev_token = None
        measure_token = None
        for p in parts:
            canon = alias_map.get(p, p)
            if canon in valid_types:
                pt_token = canon
                continue
            try:
                sev_token = float(p)
                continue
            except ValueError:
                pass
            measure_token = p

        if pt_token is None or sev_token is None or measure_token is None:
            return (
                None,
                "UNKNOWN_ROBUSTNESS_METRIC",
                f"Could not parse perturbation type, severity, and measurement from '{metric_str}'.",
            )

        matching_curves = [c for c in robustness.curves if c.perturbation_type.value == pt_token]
        if not matching_curves:
            return None, "CURVE_NOT_FOUND", f"Perturbation curve for '{pt_token}' was not evaluated."
        curve = matching_curves[0]

        matching_points = [p for p in curve.points if math.isclose(p.severity_level, sev_token, abs_tol=1e-5)]
        if not matching_points:
            eval_sevs = [p.severity_level for p in curve.points]
            return (
                None,
                "SEVERITY_POINT_NOT_FOUND",
                f"Severity level {sev_token} for '{pt_token}' was not evaluated. Evaluated severities: {eval_sevs}.",
            )
        point = matching_points[0]

        if not point.is_usable:
            reason = point.unusable_reason or "perturbation evaluation failed"
            return (
                None,
                "POINT_UNUSABLE",
                f"Perturbation point for '{pt_token}' at severity {sev_token} is unusable: {reason}.",
            )

        if measure_token in ("drop_pct", "performance_drop_percentage", "drop_percentage", "drop", "max_drop"):
            if point.performance_drop_percentage is None:
                return (
                    None,
                    "METRIC_UNDEFINED",
                    f"Performance drop percentage is undefined for '{pt_token}' at severity {sev_token}.",
                )
            return point.performance_drop_percentage, None, None

        if measure_token in ("retention", "retained_ratio", "retained"):
            if point.retained_ratio is None:
                return (
                    None,
                    "METRIC_UNDEFINED",
                    f"Retained ratio is undefined for '{pt_token}' at severity {sev_token}.",
                )
            return point.retained_ratio, None, None

        if measure_token in ("score", "perturbed_score"):
            if point.perturbed_score is None:
                return (
                    None,
                    "METRIC_UNDEFINED",
                    f"Perturbed score is undefined for '{pt_token}' at severity {sev_token}.",
                )
            return point.perturbed_score, None, None

        return None, "UNKNOWN_MEASUREMENT", f"Unknown robustness measurement '{measure_token}' in '{metric_str}'."

    return (
        None,
        "UNKNOWN_ROBUSTNESS_METRIC",
        f"Could not resolve robustness metric '{metric_str}'. Supported formats: 'mean_aurdc', '<type>_aurdc', '<type>_max_drop_pct', '<type>_max_retention', '<type>:<severity>:drop_pct', '<type>:<severity>:retention'.",
    )


def _evaluate_robustness_rule(
    rule: AcceptanceRule,
    evaluation: EvaluationResult,
) -> CriterionEvaluation:
    """Evaluate a robustness stress-testing criterion against model robustness evidence."""
    cand = evaluation.candidate_result

    if cand.robustness is None:
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=rule.metric,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=rule.threshold,
            observed_value=None,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="ROBUSTNESS_EVIDENCE_MISSING",
            message=f"Candidate model has no robustness evaluation evidence for criterion '{rule.rule_id}'.",
            evidence_source=rule.evidence_source,
        )

    robustness = cand.robustness
    threshold = rule.threshold
    if threshold is None:
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=rule.metric,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=None,
            observed_value=None,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="THRESHOLD_MISSING",
            message=f"Robustness rule '{rule.rule_id}' did not specify a numerical threshold.",
            evidence_source=rule.evidence_source,
        )

    if math.isnan(threshold) or math.isinf(threshold):
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=rule.metric,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=None,
            observed_value=None,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="INVALID_THRESHOLD",
            message=f"Robustness rule '{rule.rule_id}' specified a non-finite threshold ({threshold}).",
            evidence_source=rule.evidence_source,
        )

    val, err_code, err_msg = _extract_robustness_metric_value(rule.metric, robustness)
    if err_code is not None or val is None:
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=rule.metric,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=threshold,
            observed_value=val,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code=err_code or "METRIC_UNDEFINED",
            message=err_msg or f"Robustness metric '{rule.metric}' could not be extracted.",
            evidence_source=rule.evidence_source,
        )

    if math.isnan(val) or math.isinf(val):
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=rule.metric,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=threshold,
            observed_value=None,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="NON_FINITE_VALUE",
            message=f"Observed value for robustness metric '{rule.metric}' is non-finite ({val}).",
            evidence_source=rule.evidence_source,
        )

    satisfied = False
    if rule.operator in (RuleOperator.LESS_THAN_OR_EQUAL, RuleOperator.STATISTICALLY_LESS_THAN):
        satisfied = val <= threshold
    elif rule.operator == RuleOperator.LESS_THAN:
        satisfied = val < threshold
    elif rule.operator in (RuleOperator.GREATER_THAN_OR_EQUAL, RuleOperator.STATISTICALLY_GREATER_THAN):
        satisfied = val >= threshold
    elif rule.operator == RuleOperator.GREATER_THAN:
        satisfied = val > threshold
    elif rule.operator == RuleOperator.EQUALS:
        satisfied = math.isclose(val, threshold, abs_tol=1e-9)
    elif rule.operator == RuleOperator.NOT_EQUALS:
        satisfied = not math.isclose(val, threshold, abs_tol=1e-9)
    else:
        return CriterionEvaluation(
            rule_id=rule.rule_id,
            metric=rule.metric,
            severity=rule.severity,
            operator=rule.operator,
            threshold_value=threshold,
            observed_value=val,
            outcome=CriterionOutcome.INCONCLUSIVE,
            reason_code="UNSUPPORTED_OPERATOR",
            message=f"Operator '{rule.operator.value}' is not supported for robustness evaluation.",
            evidence_source=rule.evidence_source,
        )

    outcome = CriterionOutcome.SATISFIED if satisfied else CriterionOutcome.VIOLATED
    code = "ROBUSTNESS_THRESHOLD_MET" if satisfied else "ROBUSTNESS_THRESHOLD_BREACHED"
    verb = "satisfies" if satisfied else "violates"
    msg = f"Observed robustness {rule.metric} ({val:.4f}) {verb} requirement ({rule.operator.value} {threshold})."

    return CriterionEvaluation(
        rule_id=rule.rule_id,
        metric=rule.metric,
        severity=rule.severity,
        operator=rule.operator,
        threshold_value=threshold,
        observed_value=val,
        outcome=outcome,
        reason_code=code,
        message=msg,
        evidence_source=rule.evidence_source,
    )


def evaluate_acceptance_policy(
    evaluation: EvaluationResult,
    policy: AcceptancePolicy,
    decision_id: Optional[str] = None,
    evaluation_id: Optional[str] = None,
) -> AcceptanceDecision:
    """Evaluate an EvaluationResult against an AcceptancePolicy to yield a deterministic decision.

    Parameters
    ----------
    evaluation : EvaluationResult
        Completed model evaluation contract.
    policy : AcceptancePolicy
        Explicit, versioned acceptance criteria.
    decision_id : str, optional
        Unique decision identifier. Generated with timestamp/uuid if None.
    evaluation_id : str, optional
        Associated evaluation identifier.

    Returns
    -------
    AcceptanceDecision
        Structured decision record with overall outcome (PASS, FAIL, INCONCLUSIVE),
        granular criterion assessments, advisory warnings, and cryptographic signature.

    Raises
    ------
    ValueError
        If evaluation or policy is None.
    """
    if evaluation is None or policy is None:
        raise ValueError("evaluation and policy must not be None.")

    dec_id = decision_id or f"dec_{policy.policy_id}_{uuid.uuid4().hex[:12]}"
    now_utc = datetime.now(timezone.utc).isoformat()

    if not policy.rules:
        return AcceptanceDecision(
            decision_id=dec_id,
            evaluation_id=evaluation_id,
            policy_id=policy.policy_id,
            policy_version=policy.version,
            outcome=AcceptanceOutcome.INCONCLUSIVE,
            criteria_evaluations=[],
            summary_rationale="Acceptance policy contains no defined rules; verdict is inconclusive.",
            evaluated_at=now_utc,
            passed_rules_count=0,
            failed_rules_count=0,
            inconclusive_rules_count=0,
            advisory_warnings=["Policy has zero rules."],
            hash_signature=None,
        )

    criteria_evaluations: List[CriterionEvaluation] = []

    for rule in policy.rules:
        source = rule.evidence_source.lower()
        if source == "candidate_metric":
            crit_eval = _evaluate_candidate_metric_rule(rule, evaluation)
        elif source == "calibration":
            crit_eval = _evaluate_calibration_rule(rule, evaluation)
        elif source == "baseline_contrast":
            crit_eval = _evaluate_baseline_contrast_rule(rule, evaluation)
        elif source == "statistical_test":
            crit_eval = _evaluate_statistical_test_rule(rule, evaluation)
        elif source == "data_quality":
            crit_eval = _evaluate_data_quality_rule(rule, evaluation)
        elif source == "robustness":
            crit_eval = _evaluate_robustness_rule(rule, evaluation)
        else:
            crit_eval = CriterionEvaluation(
                rule_id=rule.rule_id,
                metric=rule.metric,
                severity=rule.severity,
                operator=rule.operator,
                threshold_value=rule.threshold,
                observed_value=None,
                outcome=CriterionOutcome.INCONCLUSIVE,
                reason_code="UNKNOWN_EVIDENCE_SOURCE",
                message=f"Unsupported evidence source: '{rule.evidence_source}'.",
                evidence_source=rule.evidence_source,
            )
        criteria_evaluations.append(crit_eval)

    # Separate blocker rules from advisory rules
    blockers = [c for c in criteria_evaluations if c.severity == CriterionSeverity.BLOCKER]
    advisories = [c for c in criteria_evaluations if c.severity == CriterionSeverity.ADVISORY]

    blocker_violations = [c for c in blockers if c.outcome == CriterionOutcome.VIOLATED]
    blocker_inconclusives = [c for c in blockers if c.outcome == CriterionOutcome.INCONCLUSIVE]
    blocker_satisfied = [c for c in blockers if c.outcome == CriterionOutcome.SATISFIED]

    advisory_warnings: List[str] = []
    for adv in advisories:
        if adv.outcome != CriterionOutcome.SATISFIED:
            advisory_warnings.append(f"Advisory notice [{adv.rule_id}]: {adv.message}")

    # Determine final outcome based on documented precedence
    if policy.fail_precedence:
        # Precedence: FAIL > INCONCLUSIVE > PASS
        if blocker_violations:
            final_outcome = AcceptanceOutcome.FAIL
            failed_rules_str = ", ".join(f"'{c.rule_id}'" for c in blocker_violations)
            rationale = (
                f"Candidate model rejected: {len(blocker_violations)} required criterion/criteria conclusively failed ({failed_rules_str})."
            )
        elif blocker_inconclusives:
            final_outcome = AcceptanceOutcome.INCONCLUSIVE
            inc_rules_str = ", ".join(f"'{c.rule_id}'" for c in blocker_inconclusives)
            rationale = (
                f"Acceptance verdict inconclusive: {len(blocker_inconclusives)} required criterion/criteria could not be evaluated ({inc_rules_str})."
            )
        else:
            final_outcome = AcceptanceOutcome.PASS
            rationale = (
                f"Candidate model accepted: all {len(blocker_satisfied)} required criteria conclusively satisfied."
            )
    else:
        # Precedence: INCONCLUSIVE > FAIL > PASS
        if blocker_inconclusives:
            final_outcome = AcceptanceOutcome.INCONCLUSIVE
            inc_rules_str = ", ".join(f"'{c.rule_id}'" for c in blocker_inconclusives)
            rationale = (
                f"Acceptance verdict inconclusive: {len(blocker_inconclusives)} required criterion/criteria could not be evaluated ({inc_rules_str})."
            )
        elif blocker_violations:
            final_outcome = AcceptanceOutcome.FAIL
            failed_rules_str = ", ".join(f"'{c.rule_id}'" for c in blocker_violations)
            rationale = (
                f"Candidate model rejected: {len(blocker_violations)} required criterion/criteria conclusively failed ({failed_rules_str})."
            )
        else:
            final_outcome = AcceptanceOutcome.PASS
            rationale = (
                f"Candidate model accepted: all {len(blocker_satisfied)} required criteria conclusively satisfied."
            )

    # Collect partition fingerprints if present
    cand_fingerprints = [
        fr.partition_fingerprint
        for fr in evaluation.candidate_result.fold_results
        if fr.partition_fingerprint
    ]
    base_fingerprints = [
        fr.partition_fingerprint
        for fr in evaluation.baseline_result.fold_results
        if fr.partition_fingerprint
    ]

    # Canonical evidence & decision signature payload
    # Note: decision_id is intentionally excluded so equivalent evidence & policy yield identical signatures
    signature_payload = {
        "policy_id": policy.policy_id,
        "policy_version": policy.version,
        "dataset_checksum": evaluation.dataset_metadata.checksum_sha256,
        "candidate_partition_fingerprints": cand_fingerprints,
        "baseline_partition_fingerprints": base_fingerprints,
        "outcome": final_outcome.value,
        "criteria": sorted(
            [
                {
                    "rule_id": c.rule_id,
                    "metric": c.metric,
                    "operator": c.operator.value,
                    "threshold_value": c.threshold_value,
                    "outcome": c.outcome.value,
                    "reason_code": c.reason_code,
                    "observed_value": round(c.observed_value, 6) if c.observed_value is not None else None,
                }
                for c in criteria_evaluations
            ],
            key=lambda x: x["rule_id"],
        ),
    }
    canonical_bytes = json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    hash_sig = hashlib.sha256(canonical_bytes).hexdigest()

    return AcceptanceDecision(
        decision_id=dec_id,
        evaluation_id=evaluation_id,
        policy_id=policy.policy_id,
        policy_version=policy.version,
        outcome=final_outcome,
        criteria_evaluations=criteria_evaluations,
        summary_rationale=rationale,
        evaluated_at=now_utc,
        passed_rules_count=len(blocker_satisfied),
        failed_rules_count=len(blocker_violations),
        inconclusive_rules_count=len(blocker_inconclusives),
        advisory_warnings=advisory_warnings,
        hash_signature=hash_sig,
    )
