"""Unit Tests for Acceptance Decision Engine (Milestone 2.5)."""

import math
from typing import List, Optional
import numpy as np
import pytest

from mlforge.core.contracts import (
    AcceptanceDecision,
    AcceptanceOutcome,
    AcceptancePolicy,
    AcceptanceRule,
    CalibrationCurveResult,
    ComparisonReport,
    CriterionOutcome,
    CriterionSeverity,
    DatasetMetadata,
    EvaluationResult,
    FoldResult,
    MetricComparisonResult,
    MetricDirection,
    MetricSummary,
    MetricValue,
    ModelEvaluationResult,
    PairedMetricDifference,
    PerturbationType,
    ReproducibilityMetadata,
    RobustnessCurve,
    RobustnessEvaluationResult,
    RobustnessPerturbationResult,
    RuleOperator,
    StatisticalTestResult,
    TestStatus,
)
from mlforge.engine.acceptance import evaluate_acceptance_policy


def _make_dummy_evaluation(
    cand_f1: float = 0.85,
    cand_acc: float = 0.88,
    base_f1: float = 0.70,
    base_acc: float = 0.72,
    ece: float = 0.05,
    t_pvalue: float = 0.01,
    test_status: TestStatus = TestStatus.COMPLETED,
    comparison_available: bool = True,
    metric_undefined: bool = False,
    non_finite_metric: bool = False,
    cand_f1_std: float = 0.02,
    cand_f1_ci: Optional[list] = None,
    base_f1_ci: Optional[list] = None,
    robustness: Optional[RobustnessEvaluationResult] = None,
) -> EvaluationResult:
    """Helper to build a controlled EvaluationResult fixture for acceptance testing."""
    ds_meta = DatasetMetadata(
        num_rows=100,
        num_features=4,
        feature_names=["f1", "f2", "f3", "f4"],
        feature_types={"f1": "float64", "f2": "float64", "f3": "float64", "f4": "float64"},
        numeric_feature_count=4,
        categorical_feature_count=0,
        target_name="target",
        target_type="int64",
        class_distribution={"0": 50, "1": 50},
        class_labels=["0", "1"],
        imbalance_ratio=1.0,
        missing_value_counts={},
        checksum_sha256="dummy_sha256",
    )

    repro = ReproducibilityMetadata(
        random_seed=42,
        split_strategy="stratified_k_fold",
        num_folds=5,
        package_versions={"numpy": "2.0.0"},
        platform_system="Windows",
        python_version="3.11.9",
    )

    f1_mean = None if metric_undefined else (float("nan") if non_finite_metric else cand_f1)
    cand_metrics = {
        "macro_f1": MetricSummary(
            metric_name="macro_f1",
            mean=f1_mean,
            std=cand_f1_std,
            confidence_interval_95=cand_f1_ci,
            valid_fold_count=0 if metric_undefined else 5,
            undefined_fold_count=5 if metric_undefined else 0,
            undefined_reasons=["fold undefined"] if metric_undefined else [],
        ),
        "accuracy": MetricSummary(
            metric_name="accuracy",
            mean=cand_acc,
            std=0.01,
            valid_fold_count=5,
            undefined_fold_count=0,
        ),
    }

    base_metrics = {
        "macro_f1": MetricSummary(
            metric_name="macro_f1",
            mean=base_f1,
            std=0.03,
            confidence_interval_95=base_f1_ci,
            valid_fold_count=5,
        ),
        "accuracy": MetricSummary(
            metric_name="accuracy",
            mean=base_acc,
            std=0.02,
            valid_fold_count=5,
        ),
    }

    calib = CalibrationCurveResult(
        bins=[],
        expected_calibration_error=ece,
        max_calibration_error=ece * 1.5,
        brier_score=0.10,
        log_loss=0.35,
        total_samples=100,
        positive_class_label="1",
        is_usable=True,
    )

    cand_result = ModelEvaluationResult(
        model_name="Candidate",
        model_type="logistic_regression",
        fold_results=[],
        metric_summaries=cand_metrics,
        aggregate_confusion_matrix=[[45, 5], [7, 43]],
        calibration=calib,
        robustness=robustness,
    )

    base_result = ModelEvaluationResult(
        model_name="Baseline",
        model_type="dummy_classifier",
        fold_results=[],
        metric_summaries=base_metrics,
        aggregate_confusion_matrix=[[50, 0], [50, 0]],
    )

    comp_report = None
    if comparison_available:
        mean_delta = cand_f1 - base_f1
        paired_diff = PairedMetricDifference(
            metric_name="macro_f1",
            direction=MetricDirection.HIGHER_IS_BETTER,
            raw_deltas=[0.15, 0.14, 0.16, 0.15, 0.15],
            mean_delta=mean_delta,
            std_delta=0.01,
            expected_folds=5,
            paired_folds=5,
            usable_folds=5,
            is_usable=True,
        )
        t_test = StatisticalTestResult(
            test_name="nadeau_bengio_corrected_t_test",
            status=test_status,
            statistic=4.5 if test_status == TestStatus.COMPLETED else None,
            p_value=t_pvalue if test_status == TestStatus.COMPLETED else None,
            degrees_of_freedom=4.0,
            alpha=0.05,
            statistically_significant=(t_pvalue < 0.05) if test_status == TestStatus.COMPLETED else None,
        )
        comp_report = ComparisonReport(
            candidate_model_name="Candidate",
            baseline_model_name="Baseline",
            dataset_checksum="dummy_sha256",
            num_folds=5,
            metric_comparisons={
                "macro_f1": MetricComparisonResult(
                    metric_name="macro_f1",
                    direction=MetricDirection.HIGHER_IS_BETTER,
                    paired_differences=paired_diff,
                    corrected_resampled_t_test=t_test,
                )
            },
        )

    return EvaluationResult(
        dataset_metadata=ds_meta,
        reproducibility=repro,
        candidate_result=cand_result,
        baseline_result=base_result,
        comparison_report=comp_report,
    )


def test_tc_acc_01_all_required_criteria_pass_yields_pass():
    """TC-ACC-01: Candidate exceeding all blocker thresholds with significant superiority yields PASS."""
    evaluation = _make_dummy_evaluation(cand_f1=0.85, cand_acc=0.88, ece=0.05, t_pvalue=0.01)

    policy = AcceptancePolicy(
        policy_id="production_readiness_v1",
        version="1.0.0",
        name="Production Readiness Policy",
        rules=[
            AcceptanceRule(
                rule_id="r1_f1_floor",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.80,
                severity=CriterionSeverity.BLOCKER,
            ),
            AcceptanceRule(
                rule_id="r2_ece_ceiling",
                metric="expected_calibration_error",
                operator=RuleOperator.LESS_THAN_OR_EQUAL,
                threshold=0.10,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="calibration",
            ),
            AcceptanceRule(
                rule_id="r3_baseline_superiority",
                metric="macro_f1",
                operator=RuleOperator.STATISTICALLY_GREATER_THAN,
                threshold=0.05,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="statistical_test",
            ),
        ],
    )

    decision = evaluate_acceptance_policy(evaluation=evaluation, policy=policy)

    assert decision.outcome == AcceptanceOutcome.PASS
    assert decision.passed_rules_count == 3
    assert decision.failed_rules_count == 0
    assert decision.inconclusive_rules_count == 0
    assert "conclusively satisfied" in decision.summary_rationale
    assert decision.hash_signature is not None


def test_tc_acc_02_required_criterion_violated_yields_fail():
    """TC-ACC-02: Hard fail when candidate breaches a required blocker threshold."""
    # cand_f1 is 0.72, breaching required threshold of 0.80
    evaluation = _make_dummy_evaluation(cand_f1=0.72, cand_acc=0.75, ece=0.05)

    policy = AcceptancePolicy(
        policy_id="standard_policy",
        version="1.0.0",
        name="Standard Policy",
        rules=[
            AcceptanceRule(
                rule_id="r1_f1_floor",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.80,
                severity=CriterionSeverity.BLOCKER,
            ),
        ],
    )

    decision = evaluate_acceptance_policy(evaluation=evaluation, policy=policy)

    assert decision.outcome == AcceptanceOutcome.FAIL
    assert decision.failed_rules_count == 1
    assert decision.passed_rules_count == 0
    assert "conclusively failed" in decision.summary_rationale
    assert decision.criteria_evaluations[0].outcome == CriterionOutcome.VIOLATED


def test_tc_acc_03_missing_required_evidence_yields_inconclusive():
    """TC-ACC-03: Missing evidence (e.g. required comparison report absent) yields INCONCLUSIVE."""
    evaluation = _make_dummy_evaluation(cand_f1=0.85, comparison_available=False)

    policy = AcceptancePolicy(
        policy_id="contrast_policy",
        version="1.0.0",
        name="Contrast Policy",
        rules=[
            AcceptanceRule(
                rule_id="r_contrast",
                metric="macro_f1",
                operator=RuleOperator.STATISTICALLY_GREATER_THAN,
                threshold=0.05,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="statistical_test",
            ),
        ],
    )

    decision = evaluate_acceptance_policy(evaluation=evaluation, policy=policy)

    assert decision.outcome == AcceptanceOutcome.INCONCLUSIVE
    assert decision.inconclusive_rules_count == 1
    assert decision.criteria_evaluations[0].reason_code == "STATISTICAL_TEST_UNAVAILABLE"


def test_tc_acc_04_undefined_metric_yields_inconclusive():
    """TC-ACC-04: When a required metric is undefined, verdict is INCONCLUSIVE (never coerced to 0)."""
    evaluation = _make_dummy_evaluation(metric_undefined=True)

    policy = AcceptancePolicy(
        policy_id="f1_policy",
        version="1.0.0",
        name="F1 Policy",
        rules=[
            AcceptanceRule(
                rule_id="r_f1",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.70,
                severity=CriterionSeverity.BLOCKER,
            ),
        ],
    )

    decision = evaluate_acceptance_policy(evaluation=evaluation, policy=policy)

    assert decision.outcome == AcceptanceOutcome.INCONCLUSIVE
    assert decision.criteria_evaluations[0].reason_code == "METRIC_UNDEFINED"
    assert decision.criteria_evaluations[0].observed_value is None


def test_fail_precedence_over_inconclusive():
    """Under fail_precedence=True (default), a conclusive violation yields FAIL even if another rule is INCONCLUSIVE."""
    # macro_f1 fails conclusively (0.65 < 0.80) while statistical test is unavailable
    evaluation = _make_dummy_evaluation(cand_f1=0.65, comparison_available=False)

    policy = AcceptancePolicy(
        policy_id="multi_rule",
        version="1.0.0",
        name="Multi Rule",
        fail_precedence=True,
        rules=[
            AcceptanceRule(
                rule_id="r_fail",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.80,
                severity=CriterionSeverity.BLOCKER,
            ),
            AcceptanceRule(
                rule_id="r_inc",
                metric="macro_f1",
                operator=RuleOperator.STATISTICALLY_GREATER_THAN,
                threshold=0.05,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="statistical_test",
            ),
        ],
    )

    decision = evaluate_acceptance_policy(evaluation=evaluation, policy=policy)

    assert decision.outcome == AcceptanceOutcome.FAIL
    assert decision.failed_rules_count == 1
    assert decision.inconclusive_rules_count == 1


def test_inconclusive_precedence_when_fail_precedence_disabled():
    """When fail_precedence=False, any inconclusive blocker makes final outcome INCONCLUSIVE."""
    evaluation = _make_dummy_evaluation(cand_f1=0.65, comparison_available=False)

    policy = AcceptancePolicy(
        policy_id="strict_inconclusive_policy",
        version="1.0.0",
        name="Strict Policy",
        fail_precedence=False,
        rules=[
            AcceptanceRule(
                rule_id="r_fail",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.80,
                severity=CriterionSeverity.BLOCKER,
            ),
            AcceptanceRule(
                rule_id="r_inc",
                metric="macro_f1",
                operator=RuleOperator.STATISTICALLY_GREATER_THAN,
                threshold=0.05,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="statistical_test",
            ),
        ],
    )

    decision = evaluate_acceptance_policy(evaluation=evaluation, policy=policy)

    assert decision.outcome == AcceptanceOutcome.INCONCLUSIVE


def test_advisory_violation_does_not_fail_and_adds_warning():
    """An advisory rule violation emits warnings but does NOT prevent a PASS verdict."""
    # ECE is 0.12, violating advisory threshold of 0.10, but blocker F1 passes
    evaluation = _make_dummy_evaluation(cand_f1=0.85, ece=0.12)

    policy = AcceptancePolicy(
        policy_id="advisory_policy",
        version="1.0.0",
        name="Advisory Policy",
        rules=[
            AcceptanceRule(
                rule_id="r_blocker",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.80,
                severity=CriterionSeverity.BLOCKER,
            ),
            AcceptanceRule(
                rule_id="r_advisory_ece",
                metric="expected_calibration_error",
                operator=RuleOperator.LESS_THAN_OR_EQUAL,
                threshold=0.10,
                severity=CriterionSeverity.ADVISORY,
                evidence_source="calibration",
            ),
        ],
    )

    decision = evaluate_acceptance_policy(evaluation=evaluation, policy=policy)

    assert decision.outcome == AcceptanceOutcome.PASS
    assert decision.passed_rules_count == 1
    assert len(decision.advisory_warnings) == 1
    assert "Advisory notice [r_advisory_ece]" in decision.advisory_warnings[0]


def test_non_finite_metric_rejected_as_inconclusive():
    """Non-finite metric values (NaN/Inf) are rejected and yield INCONCLUSIVE."""
    evaluation = _make_dummy_evaluation(non_finite_metric=True)

    policy = AcceptancePolicy(
        policy_id="p",
        rules=[
            AcceptanceRule(
                rule_id="r1",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.70,
            )
        ],
        name="P",
    )

    decision = evaluate_acceptance_policy(evaluation=evaluation, policy=policy)

    assert decision.outcome == AcceptanceOutcome.INCONCLUSIVE
    assert decision.criteria_evaluations[0].reason_code == "NON_FINITE_VALUE"


def test_statistical_non_significance_does_not_claim_equivalence():
    """Non-significance yields VIOLATED when significance is required, with explicit note on H0."""
    evaluation = _make_dummy_evaluation(cand_f1=0.85, t_pvalue=0.20)

    policy = AcceptancePolicy(
        policy_id="sig_policy",
        rules=[
            AcceptanceRule(
                rule_id="r_sig",
                metric="macro_f1",
                operator=RuleOperator.STATISTICALLY_GREATER_THAN,
                threshold=0.05,
                evidence_source="statistical_test",
            )
        ],
        name="Sig Policy",
    )

    decision = evaluate_acceptance_policy(evaluation=evaluation, policy=policy)

    assert decision.outcome == AcceptanceOutcome.FAIL
    crit = decision.criteria_evaluations[0]
    assert crit.outcome == CriterionOutcome.VIOLATED
    assert "failure to reject H0 indicates lack of evidence, not proof of equivalence" in crit.message


def test_deterministic_decision_repeatability():
    """Evaluating identical evidence against identical policy must produce bit-for-bit identical decision signatures."""
    evaluation = _make_dummy_evaluation(cand_f1=0.85, cand_acc=0.88, ece=0.05, t_pvalue=0.01)

    policy = AcceptancePolicy(
        policy_id="prod_policy",
        version="1.0.0",
        name="Prod Policy",
        rules=[
            AcceptanceRule(
                rule_id="r1",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.80,
            )
        ],
    )

    d1 = evaluate_acceptance_policy(evaluation=evaluation, policy=policy, decision_id="dec_fixed_id")
    d2 = evaluate_acceptance_policy(evaluation=evaluation, policy=policy, decision_id="dec_fixed_id")

    assert d1.outcome == d2.outcome == AcceptanceOutcome.PASS
    assert d1.hash_signature == d2.hash_signature


def test_decision_signature_deterministic_with_omitted_decision_id():
    """Even when decision_id is omitted (generating distinct UUIDs), the evidence hash signature must remain identical."""
    evaluation = _make_dummy_evaluation(cand_f1=0.85, cand_acc=0.88, ece=0.05, t_pvalue=0.01)

    policy = AcceptancePolicy(
        policy_id="prod_policy",
        version="1.0.0",
        name="Prod Policy",
        rules=[
            AcceptanceRule(
                rule_id="r1",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.80,
            )
        ],
    )

    d1 = evaluate_acceptance_policy(evaluation=evaluation, policy=policy, decision_id=None)
    d2 = evaluate_acceptance_policy(evaluation=evaluation, policy=policy, decision_id=None)

    assert d1.decision_id != d2.decision_id
    assert d1.hash_signature == d2.hash_signature
    assert d1.hash_signature is not None


def test_decision_signature_sensitive_to_evidence_and_policy_changes():
    """Modifying evidence, dataset checksum, partition fingerprints, or policy rules must alter the hash signature."""
    eval_base = _make_dummy_evaluation(cand_f1=0.85)

    policy = AcceptancePolicy(
        policy_id="p1",
        version="1.0.0",
        name="P1",
        rules=[
            AcceptanceRule(
                rule_id="r1",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.80,
            )
        ],
    )

    d_base = evaluate_acceptance_policy(evaluation=eval_base, policy=policy)

    # 1. Evidence change (cand_f1 changes)
    eval_diff_metric = _make_dummy_evaluation(cand_f1=0.86)
    d_diff_metric = evaluate_acceptance_policy(evaluation=eval_diff_metric, policy=policy)
    assert d_diff_metric.hash_signature != d_base.hash_signature

    # 2. Dataset checksum change
    eval_diff_ds = _make_dummy_evaluation(cand_f1=0.85)
    eval_diff_ds.dataset_metadata.checksum_sha256 = "different_checksum"
    d_diff_ds = evaluate_acceptance_policy(evaluation=eval_diff_ds, policy=policy)
    assert d_diff_ds.hash_signature != d_base.hash_signature

    # 3. Policy threshold change
    policy_diff_thresh = AcceptancePolicy(
        policy_id="p1",
        version="1.0.0",
        name="P1",
        rules=[
            AcceptanceRule(
                rule_id="r1",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.75,
            )
        ],
    )
    d_diff_pol = evaluate_acceptance_policy(evaluation=eval_base, policy=policy_diff_thresh)
    assert d_diff_pol.hash_signature != d_base.hash_signature


def test_ci_disjoint_superior_operator_higher_and_lower_is_better():
    """Verify RuleOperator.CI_DISJOINT_SUPERIOR for both higher-is-better and lower-is-better metrics."""
    # 1. Higher is better (macro F1): cand CI [0.82, 0.88] vs base CI [0.65, 0.75] -> DISJOINT MET
    eval_f1_disjoint = _make_dummy_evaluation(
        cand_f1_ci=[0.82, 0.88],
        base_f1_ci=[0.65, 0.75],
    )
    policy_f1 = AcceptancePolicy(
        policy_id="ci_pol",
        rules=[
            AcceptanceRule(
                rule_id="r_ci_f1",
                metric="macro_f1",
                operator=RuleOperator.CI_DISJOINT_SUPERIOR,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="baseline_contrast",
            )
        ],
        name="CI Policy",
    )
    d_f1 = evaluate_acceptance_policy(evaluation=eval_f1_disjoint, policy=policy_f1)
    assert d_f1.outcome == AcceptanceOutcome.PASS
    assert d_f1.criteria_evaluations[0].reason_code == "CI_DISJOINT_MET"

    # 2. Overlapping CI on higher is better: cand CI [0.72, 0.82] vs base CI [0.70, 0.78] -> VIOLATED
    eval_f1_overlap = _make_dummy_evaluation(
        cand_f1_ci=[0.72, 0.82],
        base_f1_ci=[0.70, 0.78],
    )
    d_f1_overlap = evaluate_acceptance_policy(evaluation=eval_f1_overlap, policy=policy_f1)
    assert d_f1_overlap.outcome == AcceptanceOutcome.FAIL
    assert d_f1_overlap.criteria_evaluations[0].reason_code == "CI_OVERLAPPING_OR_INFERIOR"

    # 3. Missing CI evidence -> INCONCLUSIVE
    eval_f1_no_ci = _make_dummy_evaluation(cand_f1_ci=None, base_f1_ci=None)
    d_f1_no_ci = evaluate_acceptance_policy(evaluation=eval_f1_no_ci, policy=policy_f1)
    assert d_f1_no_ci.outcome == AcceptanceOutcome.INCONCLUSIVE
    assert d_f1_no_ci.criteria_evaluations[0].reason_code == "CONFIDENCE_INTERVAL_UNAVAILABLE"


def test_unsupported_operators_rejected_as_inconclusive():
    """Using an unsupported operator for a criterion category must yield INCONCLUSIVE with UNSUPPORTED_OPERATOR."""
    evaluation = _make_dummy_evaluation()

    # CI_DISJOINT_SUPERIOR is comparative and unsupported on a standalone candidate metric
    policy = AcceptancePolicy(
        policy_id="unsupported_op_pol",
        rules=[
            AcceptanceRule(
                rule_id="r_bad_op",
                metric="macro_f1",
                operator=RuleOperator.CI_DISJOINT_SUPERIOR,
                threshold=0.80,
                evidence_source="candidate_metric",
            )
        ],
        name="Bad Op Policy",
    )

    decision = evaluate_acceptance_policy(evaluation=evaluation, policy=policy)
    assert decision.outcome == AcceptanceOutcome.INCONCLUSIVE
    assert decision.criteria_evaluations[0].reason_code == "UNSUPPORTED_OPERATOR"
    assert decision.criteria_evaluations[0].outcome == CriterionOutcome.INCONCLUSIVE


def test_missing_threshold_rejected_across_all_rule_types():
    """Rules without an explicit threshold must evaluate to INCONCLUSIVE (THRESHOLD_MISSING), not silently use defaults."""
    evaluation = _make_dummy_evaluation()

    # Test candidate_metric, calibration, baseline_contrast, and data_quality rules with threshold=None
    rules = [
        AcceptanceRule(
            rule_id="r_cand",
            metric="macro_f1",
            operator=RuleOperator.GREATER_THAN_OR_EQUAL,
            threshold=None,
            evidence_source="candidate_metric",
        ),
        AcceptanceRule(
            rule_id="r_calib",
            metric="expected_calibration_error",
            operator=RuleOperator.LESS_THAN_OR_EQUAL,
            threshold=None,
            evidence_source="calibration",
        ),
        AcceptanceRule(
            rule_id="r_contrast",
            metric="macro_f1",
            operator=RuleOperator.GREATER_THAN_OR_EQUAL,
            threshold=None,
            evidence_source="baseline_contrast",
        ),
        AcceptanceRule(
            rule_id="r_dq",
            metric="max_undefined_folds",
            operator=RuleOperator.LESS_THAN_OR_EQUAL,
            threshold=None,
            evidence_source="data_quality",
        ),
    ]

    for rule in rules:
        policy = AcceptancePolicy(policy_id="test_thresh", rules=[rule], name="T")
        decision = evaluate_acceptance_policy(evaluation=evaluation, policy=policy)
        assert decision.outcome == AcceptanceOutcome.INCONCLUSIVE
        assert decision.criteria_evaluations[0].reason_code == "THRESHOLD_MISSING"


def test_baseline_contrast_signed_delta_operators():
    """Verify signed candidate-minus-baseline delta comparison across operators."""
    # cand_f1 is 0.85, base_f1 is 0.70 -> mean_delta is +0.15
    evaluation = _make_dummy_evaluation(cand_f1=0.85, base_f1=0.70)

    # 1. Delta >= +0.10 -> SATISFIED
    p_ge = AcceptancePolicy(
        policy_id="p_ge",
        rules=[
            AcceptanceRule(
                rule_id="r1",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.10,
                evidence_source="baseline_contrast",
            )
        ],
        name="P",
    )
    d_ge = evaluate_acceptance_policy(evaluation=evaluation, policy=p_ge)
    assert d_ge.outcome == AcceptanceOutcome.PASS
    assert d_ge.criteria_evaluations[0].reason_code == "BASELINE_DELTA_MET"

    # 2. Delta > +0.20 -> VIOLATED
    p_gt = AcceptancePolicy(
        policy_id="p_gt",
        rules=[
            AcceptanceRule(
                rule_id="r1",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN,
                threshold=0.20,
                evidence_source="baseline_contrast",
            )
        ],
        name="P",
    )
    d_gt = evaluate_acceptance_policy(evaluation=evaluation, policy=p_gt)
    assert d_gt.outcome == AcceptanceOutcome.FAIL
    assert d_gt.criteria_evaluations[0].reason_code == "BASELINE_DELTA_BREACHED"

    # 3. Delta == +0.15 -> SATISFIED (boundary equality)
    p_eq = AcceptancePolicy(
        policy_id="p_eq",
        rules=[
            AcceptanceRule(
                rule_id="r1",
                metric="macro_f1",
                operator=RuleOperator.EQUALS,
                threshold=0.15,
                evidence_source="baseline_contrast",
            )
        ],
        name="P",
    )
    d_eq = evaluate_acceptance_policy(evaluation=evaluation, policy=p_eq)
    assert d_eq.outcome == AcceptanceOutcome.PASS


def test_boundary_threshold_equality_and_operators():
    """Verify exact boundary threshold evaluation (val == threshold) across all standard operators."""
    # candidate macro_f1 is exactly 0.80
    evaluation = _make_dummy_evaluation(cand_f1=0.80)

    # Greater than or equal (0.80 >= 0.80 -> SATISFIED)
    p_gte = AcceptancePolicy(
        policy_id="p_gte",
        rules=[AcceptanceRule(rule_id="r", metric="macro_f1", operator=RuleOperator.GREATER_THAN_OR_EQUAL, threshold=0.80)],
        name="P",
    )
    assert evaluate_acceptance_policy(evaluation=evaluation, policy=p_gte).outcome == AcceptanceOutcome.PASS

    # Less than or equal (0.80 <= 0.80 -> SATISFIED)
    p_lte = AcceptancePolicy(
        policy_id="p_lte",
        rules=[AcceptanceRule(rule_id="r", metric="macro_f1", operator=RuleOperator.LESS_THAN_OR_EQUAL, threshold=0.80)],
        name="P",
    )
    assert evaluate_acceptance_policy(evaluation=evaluation, policy=p_lte).outcome == AcceptanceOutcome.PASS

    # Strictly greater than (0.80 > 0.80 -> VIOLATED)
    p_gt = AcceptancePolicy(
        policy_id="p_gt",
        rules=[AcceptanceRule(rule_id="r", metric="macro_f1", operator=RuleOperator.GREATER_THAN, threshold=0.80)],
        name="P",
    )
    assert evaluate_acceptance_policy(evaluation=evaluation, policy=p_gt).outcome == AcceptanceOutcome.FAIL

    # Strictly less than (0.80 < 0.80 -> VIOLATED)
    p_lt = AcceptancePolicy(
        policy_id="p_lt",
        rules=[AcceptanceRule(rule_id="r", metric="macro_f1", operator=RuleOperator.LESS_THAN, threshold=0.80)],
        name="P",
    )
    assert evaluate_acceptance_policy(evaluation=evaluation, policy=p_lt).outcome == AcceptanceOutcome.FAIL

    # Equals (0.80 == 0.80 -> SATISFIED)
    p_eq = AcceptancePolicy(
        policy_id="p_eq",
        rules=[AcceptanceRule(rule_id="r", metric="macro_f1", operator=RuleOperator.EQUALS, threshold=0.80)],
        name="P",
    )
    assert evaluate_acceptance_policy(evaluation=evaluation, policy=p_eq).outcome == AcceptanceOutcome.PASS

    # Not equals (0.80 != 0.80 -> VIOLATED)
    p_neq = AcceptancePolicy(
        policy_id="p_neq",
        rules=[AcceptanceRule(rule_id="r", metric="macro_f1", operator=RuleOperator.NOT_EQUALS, threshold=0.80)],
        name="P",
    )
    assert evaluate_acceptance_policy(evaluation=evaluation, policy=p_neq).outcome == AcceptanceOutcome.FAIL


# ============================================================================
# Section 10: Robustness Acceptance Policy Evaluation (Milestone 3.3)
# ============================================================================

def _make_dummy_robustness_result(
    noise_drops: Optional[List[float]] = None,
    noise_aurdc: Optional[float] = 0.92,
    has_unusable_point: bool = False,
    non_finite_value: bool = False,
) -> RobustnessEvaluationResult:
    """Helper to build a controlled RobustnessEvaluationResult fixture."""
    drops = noise_drops if noise_drops is not None else [2.5, 7.5, 12.5]
    severities = [0.05, 0.10, 0.25]
    clean_score = 0.80

    points = []
    for sev, drop in zip(severities, drops):
        pert_score = clean_score * (1.0 - drop / 100.0)
        ret_ratio = pert_score / clean_score
        if non_finite_value and sev == 0.10:
            ret_ratio = float("nan")

        pt = RobustnessPerturbationResult(
            perturbation_type=PerturbationType.GAUSSIAN_NOISE,
            severity_level=sev,
            metric_name="accuracy",
            clean_score=clean_score,
            perturbed_score=pert_score,
            retained_ratio=ret_ratio,
            performance_drop_percentage=drop,
            is_usable=not (has_unusable_point and sev == 0.10),
            unusable_reason="SIMULATED_FAILURE" if (has_unusable_point and sev == 0.10) else None,
        )
        points.append(pt)

    actual_aurdc = None if has_unusable_point else noise_aurdc
    curve = RobustnessCurve(
        perturbation_type=PerturbationType.GAUSSIAN_NOISE,
        metric_name="accuracy",
        clean_score=clean_score,
        points=points,
        area_under_degradation_curve=actual_aurdc,
        warnings=["Point failed"] if has_unusable_point else [],
    )

    summary = {}
    if not has_unusable_point:
        summary["gaussian_noise_max_retention"] = round(points[-1].retained_ratio, 6)
        summary["gaussian_noise_max_drop_pct"] = round(points[-1].performance_drop_percentage, 6)
        if actual_aurdc is not None:
            summary["gaussian_noise_aurdc"] = actual_aurdc
            summary["mean_aurdc"] = actual_aurdc

    return RobustnessEvaluationResult(
        curves=[curve],
        summary_retention=summary,
        is_usable=not has_unusable_point,
        warnings=["Unusable points"] if has_unusable_point else [],
    )


def test_robustness_rule_mean_aurdc_pass_and_fail():
    """Verify mean_aurdc rule evaluates correctly at threshold boundaries."""
    rob = _make_dummy_robustness_result(noise_aurdc=0.92)
    evaluation = _make_dummy_evaluation(robustness=rob)

    # 1. Pass: mean_aurdc (0.92) >= 0.90
    pol_pass = AcceptancePolicy(
        policy_id="pol_rob_aurdc_pass",
        name="Robustness AURDC Pass",
        rules=[
            AcceptanceRule(
                rule_id="rob_aurdc",
                metric="mean_aurdc",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.90,
                evidence_source="robustness",
            )
        ],
    )
    dec_pass = evaluate_acceptance_policy(evaluation=evaluation, policy=pol_pass)
    assert dec_pass.outcome == AcceptanceOutcome.PASS
    assert dec_pass.criteria_evaluations[0].outcome == CriterionOutcome.SATISFIED
    assert dec_pass.criteria_evaluations[0].reason_code == "ROBUSTNESS_THRESHOLD_MET"
    assert math.isclose(dec_pass.criteria_evaluations[0].observed_value, 0.92, abs_tol=1e-5)

    # 2. Fail: mean_aurdc (0.92) >= 0.95
    pol_fail = AcceptancePolicy(
        policy_id="pol_rob_aurdc_fail",
        name="Robustness AURDC Fail",
        rules=[
            AcceptanceRule(
                rule_id="rob_aurdc",
                metric="mean_aurdc",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.95,
                evidence_source="robustness",
            )
        ],
    )
    dec_fail = evaluate_acceptance_policy(evaluation=evaluation, policy=pol_fail)
    assert dec_fail.outcome == AcceptanceOutcome.FAIL
    assert dec_fail.criteria_evaluations[0].outcome == CriterionOutcome.VIOLATED
    assert dec_fail.criteria_evaluations[0].reason_code == "ROBUSTNESS_THRESHOLD_BREACHED"


def test_robustness_rule_max_drop_percentage():
    """Verify maximum permitted performance drop under noise <= 15% (roadmap example)."""
    # Max drop is 12.5%
    rob = _make_dummy_robustness_result(noise_drops=[2.5, 7.5, 12.5])
    evaluation = _make_dummy_evaluation(robustness=rob)

    # 1. Pass: max drop (12.5%) <= 15.0%
    pol_pass = AcceptancePolicy(
        policy_id="pol_drop_pass",
        name="Max Drop Pass",
        rules=[
            AcceptanceRule(
                rule_id="r_max_drop",
                metric="gaussian_noise_max_drop_pct",
                operator=RuleOperator.LESS_THAN_OR_EQUAL,
                threshold=15.0,
                evidence_source="robustness",
            )
        ],
    )
    dec_pass = evaluate_acceptance_policy(evaluation=evaluation, policy=pol_pass)
    assert dec_pass.outcome == AcceptanceOutcome.PASS
    assert dec_pass.criteria_evaluations[0].outcome == CriterionOutcome.SATISFIED
    assert dec_pass.criteria_evaluations[0].reason_code == "ROBUSTNESS_THRESHOLD_MET"
    assert math.isclose(dec_pass.criteria_evaluations[0].observed_value, 12.5, abs_tol=1e-5)

    # 2. Fail: max drop (12.5%) <= 10.0%
    pol_fail = AcceptancePolicy(
        policy_id="pol_drop_fail",
        name="Max Drop Fail",
        rules=[
            AcceptanceRule(
                rule_id="r_max_drop",
                metric="gaussian_noise_max_drop_pct",
                operator=RuleOperator.LESS_THAN_OR_EQUAL,
                threshold=10.0,
                evidence_source="robustness",
            )
        ],
    )
    dec_fail = evaluate_acceptance_policy(evaluation=evaluation, policy=pol_fail)
    assert dec_fail.outcome == AcceptanceOutcome.FAIL
    assert dec_fail.criteria_evaluations[0].outcome == CriterionOutcome.VIOLATED
    assert dec_fail.criteria_evaluations[0].reason_code == "ROBUSTNESS_THRESHOLD_BREACHED"


def test_robustness_rule_colon_format_and_specific_severity():
    """Verify tokenized formats: e.g. gaussian_noise:0.10:drop_pct and aurdc:gaussian_noise."""
    rob = _make_dummy_robustness_result(noise_drops=[2.5, 7.5, 12.5], noise_aurdc=0.92)
    evaluation = _make_dummy_evaluation(robustness=rob)

    policy = AcceptancePolicy(
        policy_id="pol_colon_syntax",
        name="Colon Syntax Policy",
        rules=[
            # Specific severity drop at 10% severity: observed 7.5% <= 8.0%
            AcceptanceRule(
                rule_id="r_sev_drop",
                metric="gaussian_noise:0.10:drop_pct",
                operator=RuleOperator.LESS_THAN_OR_EQUAL,
                threshold=8.0,
                evidence_source="robustness",
            ),
            # Specific severity retention at 10% severity: observed 0.925 >= 0.90
            AcceptanceRule(
                rule_id="r_sev_ret",
                metric="gaussian_noise:0.10:retention",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.90,
                evidence_source="robustness",
            ),
            # AURDC via colon syntax: observed 0.92 >= 0.85
            AcceptanceRule(
                rule_id="r_aurdc_colon",
                metric="aurdc:gaussian_noise",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.85,
                evidence_source="robustness",
            ),
        ],
    )
    decision = evaluate_acceptance_policy(evaluation=evaluation, policy=policy)
    assert decision.outcome == AcceptanceOutcome.PASS
    assert len(decision.criteria_evaluations) == 3
    assert all(c.outcome == CriterionOutcome.SATISFIED for c in decision.criteria_evaluations)


def test_robustness_rule_missing_robustness_evidence():
    """If candidate model lacks robustness evaluation, outcome is INCONCLUSIVE."""
    evaluation = _make_dummy_evaluation(robustness=None)

    policy = AcceptancePolicy(
        policy_id="pol_missing_rob",
        name="Missing Robustness Evidence Policy",
        rules=[
            AcceptanceRule(
                rule_id="r_rob",
                metric="mean_aurdc",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.90,
                evidence_source="robustness",
            )
        ],
    )
    decision = evaluate_acceptance_policy(evaluation=evaluation, policy=policy)
    assert decision.outcome == AcceptanceOutcome.INCONCLUSIVE
    assert decision.criteria_evaluations[0].outcome == CriterionOutcome.INCONCLUSIVE
    assert decision.criteria_evaluations[0].reason_code == "ROBUSTNESS_EVIDENCE_MISSING"
    assert decision.criteria_evaluations[0].observed_value is None


def test_robustness_rule_missing_curve_and_severity():
    """Requesting an un-evaluated perturbation curve or severity level yields INCONCLUSIVE."""
    rob = _make_dummy_robustness_result()
    evaluation = _make_dummy_evaluation(robustness=rob)

    # 1. Missing curve: compound was not evaluated
    pol_curve = AcceptancePolicy(
        policy_id="pol_missing_curve",
        name="Missing Curve Policy",
        rules=[
            AcceptanceRule(
                rule_id="r_compound",
                metric="compound_aurdc",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.80,
                evidence_source="robustness",
            )
        ],
    )
    dec_curve = evaluate_acceptance_policy(evaluation=evaluation, policy=pol_curve)
    assert dec_curve.outcome == AcceptanceOutcome.INCONCLUSIVE
    assert dec_curve.criteria_evaluations[0].reason_code == "CURVE_NOT_FOUND"

    # 2. Missing severity: severity 0.50 was not evaluated in grid [0.05, 0.10, 0.25]
    pol_sev = AcceptancePolicy(
        policy_id="pol_missing_sev",
        name="Missing Severity Policy",
        rules=[
            AcceptanceRule(
                rule_id="r_sev",
                metric="gaussian_noise:0.50:drop_pct",
                operator=RuleOperator.LESS_THAN_OR_EQUAL,
                threshold=20.0,
                evidence_source="robustness",
            )
        ],
    )
    dec_sev = evaluate_acceptance_policy(evaluation=evaluation, policy=pol_sev)
    assert dec_sev.outcome == AcceptanceOutcome.INCONCLUSIVE
    assert dec_sev.criteria_evaluations[0].reason_code == "SEVERITY_POINT_NOT_FOUND"


def test_robustness_rule_unusable_point_causes_inconclusive():
    """If a requested point or AURDC is unusable, the criterion evaluates to INCONCLUSIVE."""
    rob = _make_dummy_robustness_result(has_unusable_point=True)
    evaluation = _make_dummy_evaluation(robustness=rob)

    # 1. Specific point at severity 0.10 is unusable
    pol_pt = AcceptancePolicy(
        policy_id="pol_pt_unusable",
        name="Point Unusable Policy",
        rules=[
            AcceptanceRule(
                rule_id="r_pt",
                metric="gaussian_noise:0.10:drop_pct",
                operator=RuleOperator.LESS_THAN_OR_EQUAL,
                threshold=10.0,
                evidence_source="robustness",
            )
        ],
    )
    dec_pt = evaluate_acceptance_policy(evaluation=evaluation, policy=pol_pt)
    assert dec_pt.outcome == AcceptanceOutcome.INCONCLUSIVE
    assert dec_pt.criteria_evaluations[0].reason_code == "POINT_UNUSABLE"

    # 2. AURDC is unavailable due to unusable point in curve
    pol_aurdc = AcceptancePolicy(
        policy_id="pol_aurdc_unusable",
        name="AURDC Unusable Policy",
        rules=[
            AcceptanceRule(
                rule_id="r_aurdc",
                metric="aurdc:gaussian_noise",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.85,
                evidence_source="robustness",
            )
        ],
    )
    dec_aurdc = evaluate_acceptance_policy(evaluation=evaluation, policy=pol_aurdc)
    assert dec_aurdc.outcome == AcceptanceOutcome.INCONCLUSIVE
    assert dec_aurdc.criteria_evaluations[0].reason_code == "ROBUSTNESS_AURDC_UNAVAILABLE"


def test_robustness_rule_non_finite_and_missing_threshold():
    """Non-finite thresholds or values cannot accidentally pass and return INCONCLUSIVE."""
    rob = _make_dummy_robustness_result()
    evaluation = _make_dummy_evaluation(robustness=rob)

    # 1. Missing threshold
    pol_no_thresh = AcceptancePolicy(
        policy_id="pol_no_thresh",
        name="No Threshold Policy",
        rules=[
            AcceptanceRule(
                rule_id="r1",
                metric="mean_aurdc",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=None,
                evidence_source="robustness",
            )
        ],
    )
    dec_no_thresh = evaluate_acceptance_policy(evaluation=evaluation, policy=pol_no_thresh)
    assert dec_no_thresh.outcome == AcceptanceOutcome.INCONCLUSIVE
    assert dec_no_thresh.criteria_evaluations[0].reason_code == "THRESHOLD_MISSING"

    # 2. Non-finite threshold (NaN)
    pol_nan_thresh = AcceptancePolicy(
        policy_id="pol_nan_thresh",
        name="NaN Threshold Policy",
        rules=[
            AcceptanceRule(
                rule_id="r2",
                metric="mean_aurdc",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=float("nan"),
                evidence_source="robustness",
            )
        ],
    )
    dec_nan_thresh = evaluate_acceptance_policy(evaluation=evaluation, policy=pol_nan_thresh)
    assert dec_nan_thresh.outcome == AcceptanceOutcome.INCONCLUSIVE
    assert dec_nan_thresh.criteria_evaluations[0].reason_code == "INVALID_THRESHOLD"


def test_robustness_rule_blocker_vs_advisory_precedence():
    """Blocker failure rejects candidate; advisory failure emits warning without blocking."""
    # Max drop is 12.5%
    rob = _make_dummy_robustness_result(noise_drops=[2.5, 7.5, 12.5])
    evaluation = _make_dummy_evaluation(robustness=rob)

    # 1. Blocker violation -> FAIL
    pol_blocker = AcceptancePolicy(
        policy_id="pol_blocker",
        name="Blocker Policy",
        rules=[
            AcceptanceRule(
                rule_id="r_cand_acc",
                metric="accuracy",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.80,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="candidate_metric",
            ),
            AcceptanceRule(
                rule_id="r_rob_drop",
                metric="gaussian_noise_max_drop_pct",
                operator=RuleOperator.LESS_THAN_OR_EQUAL,
                threshold=5.0,  # 12.5% breaches 5.0%
                severity=CriterionSeverity.BLOCKER,
                evidence_source="robustness",
            ),
        ],
    )
    dec_blocker = evaluate_acceptance_policy(evaluation=evaluation, policy=pol_blocker)
    assert dec_blocker.outcome == AcceptanceOutcome.FAIL
    assert dec_blocker.failed_rules_count == 1
    assert dec_blocker.passed_rules_count == 1

    # 2. Advisory violation -> PASS with advisory warning
    pol_advisory = AcceptancePolicy(
        policy_id="pol_advisory",
        name="Advisory Policy",
        rules=[
            AcceptanceRule(
                rule_id="r_cand_acc",
                metric="accuracy",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.80,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="candidate_metric",
            ),
            AcceptanceRule(
                rule_id="r_rob_drop",
                metric="gaussian_noise_max_drop_pct",
                operator=RuleOperator.LESS_THAN_OR_EQUAL,
                threshold=5.0,  # 12.5% breaches 5.0%
                severity=CriterionSeverity.ADVISORY,
                evidence_source="robustness",
            ),
        ],
    )
    dec_advisory = evaluate_acceptance_policy(evaluation=evaluation, policy=pol_advisory)
    assert dec_advisory.outcome == AcceptanceOutcome.PASS
    assert dec_advisory.passed_rules_count == 1
    assert dec_advisory.failed_rules_count == 0
    assert len(dec_advisory.advisory_warnings) == 1
    assert "r_rob_drop" in dec_advisory.advisory_warnings[0]


def test_robustness_rule_deterministic_hash_signature():
    """Identical evaluation and policy with robustness rules produce identical decision signatures."""
    rob = _make_dummy_robustness_result()
    eval1 = _make_dummy_evaluation(robustness=rob)
    eval2 = _make_dummy_evaluation(robustness=rob)

    policy = AcceptancePolicy(
        policy_id="pol_sig_rob",
        name="Signature Determinism Policy",
        rules=[
            AcceptanceRule(
                rule_id="r1",
                metric="mean_aurdc",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.90,
                evidence_source="robustness",
            ),
            AcceptanceRule(
                rule_id="r2",
                metric="gaussian_noise_max_drop_pct",
                operator=RuleOperator.LESS_THAN_OR_EQUAL,
                threshold=15.0,
                evidence_source="robustness",
            ),
        ],
    )

    dec1 = evaluate_acceptance_policy(evaluation=eval1, policy=policy, decision_id="dec_01")
    dec2 = evaluate_acceptance_policy(evaluation=eval2, policy=policy, decision_id="dec_02")

    assert dec1.hash_signature is not None
    assert dec1.hash_signature == dec2.hash_signature
