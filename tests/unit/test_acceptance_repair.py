"""Regression tests for pre-redesign critical defect repairs.

Verifies:
- ACC-001: Statistical significance alpha separation from effect-size threshold.
- STAT-001: Dummy baseline Macro F1 aggregation and usable paired observations.
- STAT-002: Zero-variance paired statistical comparisons adhering to documented methodology.
- API-001: Explicit unknown policy_id rejection vs omitted default behavior.
- SEC-001: Filesystem path confinement across traversal, symlinks, and forbidden files.
- Decision Matrix: Proper evaluation outcomes for PASS, FAIL, and INCONCLUSIVE scenarios.
"""

import math
from pathlib import Path
import tempfile
from typing import Generator
import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlforge.api.app import create_app
from mlforge.api.routes import get_repository, _resolve_confined_path, STANDARD_POLICIES
from mlforge.application.persistence import SQLiteRunRepository
from mlforge.core.contracts import (
    AcceptanceOutcome,
    AcceptancePolicy,
    AcceptanceRule,
    CalibrationCurveResult,
    ComparisonReport,
    CriterionOutcome,
    CriterionSeverity,
    EvaluationResult,
    FoldResult,
    MetricComparisonResult,
    MetricDirection,
    MetricSummary,
    MetricValue,
    ModelEvaluationResult,
    PairedMetricDifference,
    ReproducibilityMetadata,
    RuleOperator,
    StatisticalTestResult,
    TestStatus,
)
from mlforge.engine.acceptance import evaluate_acceptance_policy
from mlforge.engine.metrics import (
    calculate_fold_classification_metrics,
    aggregate_metric_folds,
)
from mlforge.engine.statistics import (
    compute_paired_metric_differences,
    nadeau_bengio_corrected_t_test,
)


from tests.unit.test_acceptance import _make_dummy_evaluation


# ============================================================================
# ACC-001: Acceptance Significance Alpha vs Effect-Size Threshold
# ============================================================================

def test_acc_001_superiority_rule_with_alpha_and_effect_size_passes():
    """ACC-001: Statistical test rule with threshold=0.0 and alpha=0.05 passes when p < 0.05 and delta > 0."""
    eval_res = _make_dummy_evaluation(cand_f1=0.85, base_f1=0.40, t_pvalue=0.001)

    policy = AcceptancePolicy(
        policy_id="test_acc_001_pass",
        name="Test ACC-001 Pass",
        rules=[
            AcceptanceRule(
                rule_id="superiority_over_baseline",
                metric="macro_f1",
                operator=RuleOperator.STATISTICALLY_GREATER_THAN,
                threshold=0.0,
                alpha=0.05,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="statistical_test",
            ),
        ],
    )

    decision = evaluate_acceptance_policy(evaluation=eval_res, policy=policy)
    assert decision.outcome == AcceptanceOutcome.PASS
    assert decision.passed_rules_count == 1
    assert decision.failed_rules_count == 0
    assert decision.criteria_evaluations[0].outcome == CriterionOutcome.SATISFIED


def test_acc_001_superiority_rule_violates_when_non_significant():
    """ACC-001: Statistical test rule fails conclusively (VIOLATED) when p >= alpha."""
    eval_res = _make_dummy_evaluation(cand_f1=0.85, base_f1=0.40, t_pvalue=0.20)

    policy = AcceptancePolicy(
        policy_id="test_acc_001_fail_p",
        name="Test ACC-001 Fail Non-Sig",
        rules=[
            AcceptanceRule(
                rule_id="superiority_over_baseline",
                metric="macro_f1",
                operator=RuleOperator.STATISTICALLY_GREATER_THAN,
                threshold=0.0,
                alpha=0.05,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="statistical_test",
            ),
        ],
    )

    decision = evaluate_acceptance_policy(evaluation=eval_res, policy=policy)
    assert decision.outcome == AcceptanceOutcome.FAIL
    assert decision.failed_rules_count == 1
    assert decision.criteria_evaluations[0].outcome == CriterionOutcome.VIOLATED
    assert decision.criteria_evaluations[0].reason_code == "SIGNIFICANCE_UNMET"


def test_acc_001_superiority_rule_violates_when_regression():
    """ACC-001: Statistical test rule fails when delta is negative even if statistically divergent."""
    eval_res = _make_dummy_evaluation(cand_f1=0.40, base_f1=0.85, t_pvalue=0.001)

    policy = AcceptancePolicy(
        policy_id="test_acc_001_fail_reg",
        name="Test ACC-001 Fail Reg",
        rules=[
            AcceptanceRule(
                rule_id="superiority_over_baseline",
                metric="macro_f1",
                operator=RuleOperator.STATISTICALLY_GREATER_THAN,
                threshold=0.0,
                alpha=0.05,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="statistical_test",
            ),
        ],
    )

    decision = evaluate_acceptance_policy(evaluation=eval_res, policy=policy)
    assert decision.outcome == AcceptanceOutcome.FAIL
    assert decision.criteria_evaluations[0].outcome == CriterionOutcome.VIOLATED
    assert decision.criteria_evaluations[0].reason_code == "SIGNIFICANT_REGRESSION"


# ============================================================================
# STAT-001: Dummy Baseline Metric Aggregation and Usable Observations
# ============================================================================

def test_stat_001_dummy_baseline_metric_semantics():
    """STAT-001: Dummy predicting single class flags macro_precision and macro_f1 as undefined, but accuracy is valid."""
    y_true = np.array(["0", "0", "0", "1"])  # 75% class 0, 25% class 1
    y_pred = np.array(["0", "0", "0", "0"])  # Majority class dummy prediction
    class_labels = ["0", "1"]

    metrics, cm = calculate_fold_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=None,
        class_labels=class_labels,
    )

    # Precision for class 1 has zero predictions (0/0) -> undefined with diagnostic reason
    assert metrics["macro_precision"].is_undefined is True
    assert metrics["macro_precision"].value is None

    # Macro F1 is undefined due to undefined precision per methodology
    assert metrics["macro_f1"].is_undefined is True
    assert metrics["macro_f1"].value is None

    # Accuracy is fully defined (3/4 = 0.75)
    assert metrics["accuracy"].is_undefined is False
    assert metrics["accuracy"].value == pytest.approx(0.75)


def test_stat_001_paired_comparison_with_dummy_baseline_completes_on_accuracy():
    """STAT-001: Paired comparison between candidate and dummy baseline on accuracy has full usable folds and completes."""
    cand_folds = []
    base_folds = []

    for k in range(5):
        c_metrics = {
            "accuracy": MetricValue(value=0.92 + 0.01 * k, is_undefined=False),
        }
        cand_folds.append(FoldResult(
            fold_index=k,
            train_indices_count=80,
            val_indices_count=20,
            metrics=c_metrics,
            confusion_matrix=[[9, 1], [1, 9]],
            class_labels=["0", "1"],
            fit_duration_seconds=0.01,
            predict_duration_seconds=0.005,
        ))

        b_metrics = {
            "accuracy": MetricValue(value=0.75, is_undefined=False),
        }
        base_folds.append(FoldResult(
            fold_index=k,
            train_indices_count=80,
            val_indices_count=20,
            metrics=b_metrics,
            confusion_matrix=[[15, 0], [5, 0]],
            class_labels=["0", "1"],
            fit_duration_seconds=0.001,
            predict_duration_seconds=0.001,
        ))

    paired_diff = compute_paired_metric_differences(cand_folds, base_folds, "accuracy", MetricDirection.HIGHER_IS_BETTER)
    assert paired_diff.is_usable is True
    assert paired_diff.usable_folds == 5
    assert len(paired_diff.raw_deltas) == 5

    t_test = nadeau_bengio_corrected_t_test(paired_diff, test_train_ratio=0.25)
    assert t_test.status == TestStatus.COMPLETED
    assert t_test.p_value is not None
    assert t_test.p_value < 0.001
    assert t_test.statistically_significant is True


# ============================================================================
# STAT-002: Zero-Variance Paired Statistical Comparisons
# ============================================================================

def test_stat_002_zero_variance_zero_mean_completes_non_significant():
    """STAT-002: Identical scores across all folds (mean_delta=0, var=0) yields COMPLETED test with p=1.0."""
    diff = PairedMetricDifference(
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
    res = nadeau_bengio_corrected_t_test(diff, test_train_ratio=0.25)
    assert res.status == TestStatus.COMPLETED
    assert res.statistic == 0.0
    assert res.p_value == 1.0
    assert res.statistically_significant is False


def test_stat_002_zero_variance_nonzero_mean_reports_unavailable():
    """STAT-002: Constant non-zero delta across all folds produces zero SE, returning UNAVAILABLE per methodology."""
    diff = PairedMetricDifference(
        metric_name="accuracy",
        direction=MetricDirection.HIGHER_IS_BETTER,
        raw_deltas=[0.10, 0.10, 0.10, 0.10, 0.10],
        mean_delta=0.10,
        std_delta=0.0,
        expected_folds=5,
        paired_folds=5,
        usable_folds=5,
        is_usable=True,
    )
    res = nadeau_bengio_corrected_t_test(diff, test_train_ratio=0.25)
    assert res.status == TestStatus.UNAVAILABLE
    assert "Zero variance across folds with non-zero mean difference" in res.unavailable_reason
    # Ensure p_value was NOT artificially set to 0.0
    assert res.p_value is None


# ============================================================================
# API-001: Explicit Policy ID Validation
# ============================================================================

@pytest.fixture
def api_client(tmp_path: Path) -> Generator[TestClient, None, None]:
    db_file = tmp_path / "test_api_repair.db"
    repo = SQLiteRunRepository(db_path=db_file)
    repo.initialize_schema()
    app = create_app()
    app.dependency_overrides[get_repository] = lambda: repo
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_api_001_explicit_invalid_policy_id_returns_400(api_client: TestClient):
    """API-001: Submitting an explicit unknown policy_id returns HTTP 400 Bad Request."""
    payload = {
        "sample_name": "churn",
        "target_column": "churn_label",
        "policy_id": "non_existent_policy_template_123",
        "num_folds": 2,
    }
    resp = api_client.post("/api/v1/evaluations", json=payload)
    assert resp.status_code == 400
    assert "Unknown policy_id" in resp.json()["detail"]


def test_api_001_explicit_empty_policy_id_returns_400(api_client: TestClient):
    """API-001: Submitting an empty policy_id string returns HTTP 400 Bad Request."""
    payload = {
        "sample_name": "churn",
        "target_column": "churn_label",
        "policy_id": "   ",
        "num_folds": 2,
    }
    resp = api_client.post("/api/v1/evaluations", json=payload)
    assert resp.status_code == 400
    assert "Empty policy_id specified" in resp.json()["detail"]


# ============================================================================
# SEC-001: Filesystem Path Confinement
# ============================================================================

def test_sec_001_path_traversal_rejected():
    """SEC-001: Path traversal with ../ is strictly rejected with 403 Forbidden."""
    with pytest.raises(Exception) as exc_info:
        _resolve_confined_path("../../outside_secret.csv")
    assert "outside permitted directory scope" in str(exc_info.value)


def test_sec_001_null_byte_rejected():
    """SEC-001: Path containing null byte is rejected with 400 Bad Request."""
    with pytest.raises(Exception) as exc_info:
        _resolve_confined_path("valid/path\x00/file.csv")
    assert "null bytes or empty path" in str(exc_info.value)


def test_sec_001_workspace_confined_path_accepted(tmp_path: Path):
    """SEC-001: Path inside allowed roots is accepted canonically."""
    test_file = tmp_path / "valid_data.csv"
    test_file.write_text("a,b\n1,2")
    resolved = _resolve_confined_path(str(test_file), allowed_roots=[tmp_path])
    assert resolved == test_file.resolve()


# ============================================================================
# Decision Matrix: PASS, FAIL, INCONCLUSIVE
# ============================================================================

def test_decision_matrix_all_criteria_met_yields_pass():
    """Decision Matrix: When all blocker criteria are conclusively met, verdict is PASS."""
    eval_res = _make_dummy_evaluation(cand_f1=0.85, cand_acc=0.88, t_pvalue=0.001)
    policy = AcceptancePolicy(
        policy_id="pass_policy",
        name="Pass Policy",
        rules=[
            AcceptanceRule(rule_id="r_f1", metric="macro_f1", operator=RuleOperator.GREATER_THAN_OR_EQUAL, threshold=0.70),
            AcceptanceRule(rule_id="r_acc", metric="accuracy", operator=RuleOperator.GREATER_THAN_OR_EQUAL, threshold=0.75),
            AcceptanceRule(rule_id="r_sup", metric="macro_f1", operator=RuleOperator.STATISTICALLY_GREATER_THAN, threshold=0.0, alpha=0.05, evidence_source="statistical_test"),
        ],
    )
    decision = evaluate_acceptance_policy(evaluation=eval_res, policy=policy)
    assert decision.outcome == AcceptanceOutcome.PASS
    assert decision.passed_rules_count == 3
    assert decision.failed_rules_count == 0
    assert decision.inconclusive_rules_count == 0


def test_decision_matrix_blocker_violation_yields_fail():
    """Decision Matrix: When a blocker criterion is violated, verdict is FAIL."""
    eval_res = _make_dummy_evaluation(cand_f1=0.60, cand_acc=0.88, t_pvalue=0.001)  # f1 0.60 < required 0.70
    policy = AcceptancePolicy(
        policy_id="fail_policy",
        name="Fail Policy",
        rules=[
            AcceptanceRule(rule_id="r_f1", metric="macro_f1", operator=RuleOperator.GREATER_THAN_OR_EQUAL, threshold=0.70),
        ],
    )
    decision = evaluate_acceptance_policy(evaluation=eval_res, policy=policy)
    assert decision.outcome == AcceptanceOutcome.FAIL
    assert decision.failed_rules_count == 1


def test_decision_matrix_missing_evidence_yields_inconclusive():
    """Decision Matrix: When required evidence is missing/unavailable, verdict is INCONCLUSIVE."""
    eval_res = _make_dummy_evaluation(comparison_available=False)
    policy = AcceptancePolicy(
        policy_id="inc_policy",
        name="Inconclusive Policy",
        rules=[
            AcceptanceRule(rule_id="r_sup", metric="macro_f1", operator=RuleOperator.STATISTICALLY_GREATER_THAN, threshold=0.0, alpha=0.05, evidence_source="statistical_test"),
        ],
    )
    decision = evaluate_acceptance_policy(evaluation=eval_res, policy=policy)
    assert decision.outcome == AcceptanceOutcome.INCONCLUSIVE
    assert decision.inconclusive_rules_count == 1
    assert decision.criteria_evaluations[0].reason_code == "STATISTICAL_TEST_UNAVAILABLE"


# ============================================================================
# Explicit Rule Semantics Verification (Section 1 & 2 Requirements)
# ============================================================================

@pytest.mark.parametrize(
    "thresh,expected_outcome,expected_code",
    [
        (0.0, CriterionOutcome.SATISFIED, "SIGNIFICANT_IMPROVEMENT"),
        (0.01, CriterionOutcome.SATISFIED, "SIGNIFICANT_IMPROVEMENT"),
        (0.05, CriterionOutcome.VIOLATED, "EFFECT_SIZE_UNMET"),
        (0.10, CriterionOutcome.VIOLATED, "EFFECT_SIZE_UNMET"),
        (0.25, CriterionOutcome.VIOLATED, "EFFECT_SIZE_UNMET"),
    ],
)
def test_statistical_rule_thresholds_evaluated_by_contract_small_delta(thresh, expected_outcome, expected_code):
    """Verify that thresholds 0.0, 0.01, 0.05, 0.1, and >0.1 are interpreted strictly as effect sizes.

    With mean_delta = 0.04 and p = 0.001 < alpha (0.05):
    - thresh 0.0 and 0.01 are satisfied (0.04 > 0.0 and 0.04 > 0.01).
    - thresh 0.05, 0.10, 0.25 are violated because 0.04 <= thresh (never guessed as alpha).
    """
    eval_res = _make_dummy_evaluation(cand_f1=0.74, base_f1=0.70, t_pvalue=0.001)  # delta = 0.04
    rule = AcceptanceRule(
        rule_id=f"stat_thresh_{thresh}",
        metric="macro_f1",
        operator=RuleOperator.STATISTICALLY_GREATER_THAN,
        threshold=thresh,
        alpha=0.05,
        evidence_source="statistical_test",
    )
    policy = AcceptancePolicy(policy_id="test_policy", name="Test", rules=[rule])
    decision = evaluate_acceptance_policy(evaluation=eval_res, policy=policy)
    crit = decision.criteria_evaluations[0]
    assert crit.outcome == expected_outcome
    assert crit.reason_code == expected_code


@pytest.mark.parametrize(
    "thresh,expected_outcome,expected_code",
    [
        (0.0, CriterionOutcome.SATISFIED, "SIGNIFICANT_IMPROVEMENT"),
        (0.01, CriterionOutcome.SATISFIED, "SIGNIFICANT_IMPROVEMENT"),
        (0.05, CriterionOutcome.SATISFIED, "SIGNIFICANT_IMPROVEMENT"),
        (0.10, CriterionOutcome.SATISFIED, "SIGNIFICANT_IMPROVEMENT"),
        (0.25, CriterionOutcome.VIOLATED, "EFFECT_SIZE_UNMET"),
    ],
)
def test_statistical_rule_thresholds_evaluated_by_contract_large_delta(thresh, expected_outcome, expected_code):
    """Verify that thresholds 0.0, 0.01, 0.05, 0.1, and >0.1 are interpreted strictly as effect sizes.

    With mean_delta = 0.15 and p = 0.001 < alpha (0.05):
    - thresh 0.0, 0.01, 0.05, 0.10 are satisfied (0.15 > thresh).
    - thresh 0.25 is violated because 0.15 <= 0.25.
    """
    eval_res = _make_dummy_evaluation(cand_f1=0.85, base_f1=0.70, t_pvalue=0.001)  # delta = 0.15
    rule = AcceptanceRule(
        rule_id=f"stat_thresh_{thresh}",
        metric="macro_f1",
        operator=RuleOperator.STATISTICALLY_GREATER_THAN,
        threshold=thresh,
        alpha=0.05,
        evidence_source="statistical_test",
    )
    policy = AcceptancePolicy(policy_id="test_policy", name="Test", rules=[rule])
    decision = evaluate_acceptance_policy(evaluation=eval_res, policy=policy)
    crit = decision.criteria_evaluations[0]
    assert crit.outcome == expected_outcome
    assert crit.reason_code == expected_code


def test_statistical_rule_custom_alpha_distinguishes_significance():
    """Verify that explicit rule.alpha controls significance level independent of effect size threshold."""
    # delta = 0.15, p = 0.03
    eval_res = _make_dummy_evaluation(cand_f1=0.85, base_f1=0.70, t_pvalue=0.03)

    # Policy with alpha = 0.05 -> p (0.03) < 0.05 passes
    rule_05 = AcceptanceRule(
        rule_id="r_alpha_05",
        metric="macro_f1",
        operator=RuleOperator.STATISTICALLY_GREATER_THAN,
        threshold=0.05,
        alpha=0.05,
        evidence_source="statistical_test",
    )
    dec_05 = evaluate_acceptance_policy(evaluation=eval_res, policy=AcceptancePolicy(policy_id="p1", name="p1", rules=[rule_05]))
    assert dec_05.outcome == AcceptanceOutcome.PASS
    assert dec_05.criteria_evaluations[0].outcome == CriterionOutcome.SATISFIED

    # Policy with alpha = 0.01 -> p (0.03) >= 0.01 violates significance
    rule_01 = AcceptanceRule(
        rule_id="r_alpha_01",
        metric="macro_f1",
        operator=RuleOperator.STATISTICALLY_GREATER_THAN,
        threshold=0.05,
        alpha=0.01,
        evidence_source="statistical_test",
    )
    dec_01 = evaluate_acceptance_policy(evaluation=eval_res, policy=AcceptancePolicy(policy_id="p2", name="p2", rules=[rule_01]))
    assert dec_01.outcome == AcceptanceOutcome.FAIL
    assert dec_01.criteria_evaluations[0].outcome == CriterionOutcome.VIOLATED
    assert dec_01.criteria_evaluations[0].reason_code == "SIGNIFICANCE_UNMET"


def test_statistical_rule_zero_improvement_violates():
    """Verify that zero improvement (mean_delta = 0.0) violates STATISTICALLY_GREATER_THAN even if p is low."""
    eval_res = _make_dummy_evaluation(cand_f1=0.70, base_f1=0.70, t_pvalue=0.001)
    rule = AcceptanceRule(
        rule_id="r_zero_delta",
        metric="macro_f1",
        operator=RuleOperator.STATISTICALLY_GREATER_THAN,
        threshold=0.0,
        alpha=0.05,
        evidence_source="statistical_test",
    )
    decision = evaluate_acceptance_policy(evaluation=eval_res, policy=AcceptancePolicy(policy_id="p", name="p", rules=[rule]))
    assert decision.outcome == AcceptanceOutcome.FAIL
    crit = decision.criteria_evaluations[0]
    assert crit.outcome == CriterionOutcome.VIOLATED
    assert crit.reason_code == "SIGNIFICANT_REGRESSION"


@pytest.mark.parametrize("invalid_p", [float("nan"), -0.05, 1.5])
def test_statistical_rule_invalid_p_value_yields_inconclusive(invalid_p):
    """Verify that non-finite or out-of-bounds p-values yield INCONCLUSIVE with INVALID_P_VALUE."""
    eval_res = _make_dummy_evaluation(cand_f1=0.85, base_f1=0.70, t_pvalue=invalid_p)
    rule = AcceptanceRule(
        rule_id="r_invalid_p",
        metric="macro_f1",
        operator=RuleOperator.STATISTICALLY_GREATER_THAN,
        threshold=0.0,
        alpha=0.05,
        evidence_source="statistical_test",
    )
    decision = evaluate_acceptance_policy(evaluation=eval_res, policy=AcceptancePolicy(policy_id="p", name="p", rules=[rule]))
    assert decision.outcome == AcceptanceOutcome.INCONCLUSIVE
    crit = decision.criteria_evaluations[0]
    assert crit.outcome == CriterionOutcome.INCONCLUSIVE
    assert crit.reason_code == "INVALID_P_VALUE"


@pytest.mark.parametrize("invalid_alpha", [0.0, -0.05, 1.5, float("nan")])
def test_statistical_rule_invalid_alpha_yields_inconclusive(invalid_alpha):
    """Verify that non-positive, >= 1.0, or non-finite alpha yields INCONCLUSIVE with INVALID_ALPHA."""
    eval_res = _make_dummy_evaluation(cand_f1=0.85, base_f1=0.70, t_pvalue=0.01)
    rule = AcceptanceRule(
        rule_id="r_invalid_alpha",
        metric="macro_f1",
        operator=RuleOperator.STATISTICALLY_GREATER_THAN,
        threshold=0.0,
        alpha=invalid_alpha,
        evidence_source="statistical_test",
    )
    decision = evaluate_acceptance_policy(evaluation=eval_res, policy=AcceptancePolicy(policy_id="p", name="p", rules=[rule]))
    assert decision.outcome == AcceptanceOutcome.INCONCLUSIVE
    crit = decision.criteria_evaluations[0]
    assert crit.outcome == CriterionOutcome.INCONCLUSIVE
    assert crit.reason_code == "INVALID_ALPHA"


def test_sec_001_sibling_prefix_path_rejected(tmp_path: Path):
    """SEC-001: A sibling directory with common prefix is rejected by is_relative_to."""
    allowed_dir = tmp_path / "allowed_folder"
    allowed_dir.mkdir()
    sibling_dir = tmp_path / "allowed_folder_sibling"
    sibling_dir.mkdir()
    target_file = sibling_dir / "target.csv"
    target_file.write_text("x,y\n1,2")

    with pytest.raises(Exception) as exc_info:
        _resolve_confined_path(str(target_file), allowed_roots=[allowed_dir])
    assert "outside permitted directory scope" in str(exc_info.value)


def test_sec_001_symlink_pointing_outside_rejected(tmp_path: Path):
    """SEC-001: Symlink inside approved root pointing to an external target is rejected after resolution."""
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    secret_file = external_dir / "secret.csv"
    secret_file.write_text("classified,data\n1,0")

    symlink_path = allowed_dir / "link_to_secret.csv"
    try:
        symlink_path.symlink_to(secret_file)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks require administrator privileges on Windows")

    with pytest.raises(Exception) as exc_info:
        _resolve_confined_path(str(symlink_path), allowed_roots=[allowed_dir])
    assert "outside permitted directory scope" in str(exc_info.value)


def test_release_authorization_independent_from_acceptance_pass():
    """Verify that an Acceptance PASS does NOT automatically equal Release Authorization.

    Release Gate independently enforces artifact integrity, calibration, robustness,
    and signer cryptographic authorization.
    """
    from mlforge.engine.release_gate import evaluate_release_gate
    from mlforge.core.contracts import ReleasePolicy, ReleaseGateStatus

    # An evaluation with PASS acceptance outcome
    eval_res = _make_dummy_evaluation(cand_f1=0.85, cand_acc=0.88, ece=0.05, t_pvalue=0.001)
    pass_decision = evaluate_acceptance_policy(
        evaluation=eval_res,
        policy=AcceptancePolicy(
            policy_id="p_pass",
            name="p_pass",
            rules=[
                AcceptanceRule(rule_id="r1", metric="accuracy", operator=RuleOperator.GREATER_THAN_OR_EQUAL, threshold=0.75),
            ],
        ),
    )
    assert pass_decision.outcome == AcceptanceOutcome.PASS

    # When evaluated under ReleasePolicy requiring manifest integrity, missing manifest BLOCKS release
    release_policy = ReleasePolicy(
        policy_id="strict_release",
        require_manifest_integrity=True,
    )
    gate_res = evaluate_release_gate(
        source={"evaluation": eval_res, "decision": pass_decision, "source_type": "run_dir"},
        policy=release_policy,
    )
    # Release is BLOCKED despite Acceptance Decision being PASS
    assert gate_res.overall_status == ReleaseGateStatus.BLOCKED
    assert gate_res.is_releasable is False
    assert any(g.gate_id == "artifact_integrity" and g.status.value == "failed" for g in gate_res.gate_results)
