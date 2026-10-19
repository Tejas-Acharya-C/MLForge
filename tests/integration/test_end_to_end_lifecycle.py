"""Integration tests for MLForge End-to-End Evaluation Lifecycle and Artifacts.

Validates the full scientific pipeline:
Dataset Validation -> Stratified Folds -> Isolated Fold Execution ->
Metric Calculation & Aggregation -> Statistical Tests (Nadeau-Bengio & Wilcoxon) ->
Reliability & Diagnostics -> Acceptance Policy Evaluator -> Local Artifact Persistence & Verification.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from mlforge.application.artifacts import (
    CURRENT_SCHEMA_VERSION,
    load_run_artifacts,
    serialize_run_artifacts,
)
from mlforge.core.contracts import (
    AcceptanceOutcome,
    AcceptancePolicy,
    AcceptanceRule,
    CriterionOutcome,
    CriterionSeverity,
    RuleOperator,
    TestStatus,
)
from mlforge.core.exceptions import ArtifactLoadError
from mlforge.engine.evaluator import evaluate_classification


@pytest.fixture
def mixed_churn_df():
    """Generates a synthetic mixed tabular dataset with missing values and imbalance."""
    rng = np.random.RandomState(42)
    n = 120
    df = pd.DataFrame({
        "num_clean": rng.normal(10.0, 2.0, n),
        "num_nan": rng.normal(50.0, 10.0, n),
        "cat_segment": rng.choice(["SMB", "Enterprise", "Consumer"], size=n, p=[0.4, 0.4, 0.2]),
        "cat_nan": rng.choice(["Tier1", "Tier2"], size=n),
        "churn": [0, 1, 0, 0, 1, 0] * 20,  # 80 zeros, 40 ones
    })
    # Inject missing values
    df.loc[::5, "num_nan"] = np.nan
    df.loc[::7, "cat_nan"] = None
    return df


def test_end_to_end_pipeline_with_acceptance_and_artifacts(mixed_churn_df, tmp_path: Path):
    """Full lifecycle integration test: raw data -> evaluation -> acceptance -> artifact storage -> reload."""
    policy = AcceptancePolicy(
        policy_id="prod_policy_v1",
        version="1.0.0",
        name="Production Classification Policy",
        fail_precedence=True,
        rules=[
            # Blocker 1: Candidate Accuracy floor
            AcceptanceRule(
                rule_id="r1_cand_acc",
                metric="accuracy",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.60,
                severity=CriterionSeverity.BLOCKER,
            ),
            # Blocker 2: Calibration ECE ceiling
            AcceptanceRule(
                rule_id="r2_ece_ceiling",
                metric="expected_calibration_error",
                operator=RuleOperator.LESS_THAN_OR_EQUAL,
                threshold=0.35,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="calibration",
            ),
            # Blocker 3: Baseline superiority via Nadeau-Bengio corrected t-test
            AcceptanceRule(
                rule_id="r3_t_test_superiority",
                metric="accuracy",
                operator=RuleOperator.STATISTICALLY_GREATER_THAN,
                threshold=0.05,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="statistical_test",
            ),
            # Advisory 1: Balanced accuracy recommendation
            AcceptanceRule(
                rule_id="r4_bal_acc_advisory",
                metric="balanced_accuracy",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.55,
                severity=CriterionSeverity.ADVISORY,
            ),
        ],
    )

    # 1. Execute Evaluation Pipeline
    eval_result = evaluate_classification(
        df=mixed_churn_df,
        target_column="churn",
        num_folds=5,
        subgroup_columns=["cat_segment"],
        acceptance_policy=policy,
        random_state=42,
    )

    # 2. Verify Evaluation Results
    # Dataset Metadata
    assert eval_result.dataset_metadata.num_rows == 120
    assert eval_result.dataset_metadata.num_features == 4
    assert eval_result.dataset_metadata.missing_value_counts["num_nan"] > 0
    assert eval_result.dataset_metadata.missing_value_counts["cat_nan"] > 0

    # Folds & Zero Leakage
    cand = eval_result.candidate_result
    base = eval_result.baseline_result
    assert len(cand.fold_results) == 5
    assert len(base.fold_results) == 5

    # Statistical Comparison
    assert eval_result.comparison_report is not None
    assert "accuracy" in eval_result.comparison_report.metric_comparisons
    t_test = eval_result.comparison_report.metric_comparisons["accuracy"].corrected_resampled_t_test
    assert t_test.status == TestStatus.COMPLETED

    # Acceptance Decision
    assert eval_result.acceptance_decision is not None
    decision = eval_result.acceptance_decision
    assert decision.policy_id == "prod_policy_v1"
    assert decision.outcome in (AcceptanceOutcome.PASS, AcceptanceOutcome.FAIL, AcceptanceOutcome.INCONCLUSIVE)
    assert decision.hash_signature is not None
    assert len(decision.criteria_evaluations) == 4

    # 3. Serialize Run Artifacts
    run_id = "lifecycle_run_01"
    run_dir = serialize_run_artifacts(
        evaluation=eval_result,
        acceptance_decision=decision,
        artifacts_dir=tmp_path,
        run_id=run_id,
        policy=policy,
    )

    assert run_dir.exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "evaluation.json").exists()
    assert (run_dir / "decision.json").exists()
    assert (run_dir / "policy.json").exists()

    # 4. Load Run Artifacts and Verify Integrity
    loaded = load_run_artifacts(run_dir)
    assert loaded["manifest"].run_id == run_id
    assert loaded["manifest"].schema_version == CURRENT_SCHEMA_VERSION
    assert loaded["evaluation"].dataset_metadata.checksum_sha256 == eval_result.dataset_metadata.checksum_sha256
    assert loaded["decision"].outcome == decision.outcome
    assert loaded["decision"].hash_signature == decision.hash_signature
    assert loaded["policy"].policy_id == "prod_policy_v1"


def test_end_to_end_blocker_failure_precedence(mixed_churn_df):
    """A blocker rule failure must yield FAIL even if other blocker rules pass."""
    impossible_policy = AcceptancePolicy(
        policy_id="impossible_policy",
        version="1.0.0",
        name="Impossible Policy",
        rules=[
            # Passable blocker
            AcceptanceRule(
                rule_id="r1_pass",
                metric="accuracy",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.50,
                severity=CriterionSeverity.BLOCKER,
            ),
            # Impossible blocker: Candidate will fail 0.999 accuracy
            AcceptanceRule(
                rule_id="r2_impossible",
                metric="accuracy",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.999,
                severity=CriterionSeverity.BLOCKER,
            ),
        ],
    )

    eval_result = evaluate_classification(
        df=mixed_churn_df,
        target_column="churn",
        num_folds=3,
        acceptance_policy=impossible_policy,
        random_state=42,
    )

    assert eval_result.acceptance_decision is not None
    assert eval_result.acceptance_decision.outcome == AcceptanceOutcome.FAIL
    assert eval_result.acceptance_decision.failed_rules_count == 1
    assert eval_result.acceptance_decision.passed_rules_count == 1


def test_end_to_end_advisory_rule_failure_does_not_block_pass(mixed_churn_df):
    """An advisory rule failure must NOT cause FAIL; verdict remains PASS with warning recorded."""
    advisory_policy = AcceptancePolicy(
        policy_id="advisory_policy",
        version="1.0.0",
        name="Advisory Policy",
        rules=[
            # Satisfied blocker
            AcceptanceRule(
                rule_id="r1_blocker",
                metric="accuracy",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.50,
                severity=CriterionSeverity.BLOCKER,
            ),
            # Failed advisory
            AcceptanceRule(
                rule_id="r2_advisory",
                metric="accuracy",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.999,
                severity=CriterionSeverity.ADVISORY,
            ),
        ],
    )

    eval_result = evaluate_classification(
        df=mixed_churn_df,
        target_column="churn",
        num_folds=3,
        acceptance_policy=advisory_policy,
        random_state=42,
    )

    assert eval_result.acceptance_decision is not None
    assert eval_result.acceptance_decision.outcome == AcceptanceOutcome.PASS
    assert eval_result.acceptance_decision.passed_rules_count == 1
    assert eval_result.acceptance_decision.failed_rules_count == 0  # Blocker rule failures are 0
    # Check that the advisory rule violation is explicitly tracked in criteria and advisory warnings
    r2_eval = next(c for c in eval_result.acceptance_decision.criteria_evaluations if c.rule_id == "r2_advisory")
    assert r2_eval.outcome == CriterionOutcome.VIOLATED
    assert len(eval_result.acceptance_decision.advisory_warnings) == 1
    assert "r2_advisory" in eval_result.acceptance_decision.advisory_warnings[0]


def test_end_to_end_undefined_metric_propagation_to_inconclusive(tmp_path: Path):
    """When a baseline has undefined metrics, statistical contrast is unavailable and policy yields INCONCLUSIVE."""
    # DummyClassifier on extreme dataset where minority class is never predicted
    df = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 3,
        "target": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1] * 3,
    })

    # Policy requiring statistically significant macro_f1 superiority
    policy = AcceptancePolicy(
        policy_id="f1_contrast_policy",
        version="1.0.0",
        name="F1 Contrast Policy",
        rules=[
            AcceptanceRule(
                rule_id="r_f1_t_test",
                metric="macro_f1",
                operator=RuleOperator.STATISTICALLY_GREATER_THAN,
                threshold=0.05,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="statistical_test",
            ),
        ],
    )

    eval_result = evaluate_classification(
        df=df,
        target_column="target",
        num_folds=3,
        acceptance_policy=policy,
        random_state=42,
    )

    # Baseline macro_f1 is undefined because Dummy never predicts class 1
    base_f1 = eval_result.baseline_result.metric_summaries["macro_f1"]
    assert base_f1.undefined_fold_count > 0

    # Paired differences cannot be computed with undefined folds -> test unavailable
    assert eval_result.acceptance_decision is not None
    assert eval_result.acceptance_decision.outcome == AcceptanceOutcome.INCONCLUSIVE
    assert eval_result.acceptance_decision.inconclusive_rules_count == 1
    crit = eval_result.acceptance_decision.criteria_evaluations[0]
    assert crit.outcome == CriterionOutcome.INCONCLUSIVE
    assert crit.reason_code == "STATISTICAL_TEST_UNAVAILABLE"

    # Verify serialization preserves undefined state intact
    run_dir = serialize_run_artifacts(
        evaluation=eval_result,
        acceptance_decision=eval_result.acceptance_decision,
        artifacts_dir=tmp_path,
        run_id="undef_run_01",
        policy=policy,
    )
    loaded = load_run_artifacts(run_dir)
    assert loaded["decision"].outcome == AcceptanceOutcome.INCONCLUSIVE


def test_end_to_end_artifact_tamper_detection(mixed_churn_df, tmp_path: Path):
    """Tampering with any serialized artifact file must fail byte-level manifest verification."""
    eval_result = evaluate_classification(
        df=mixed_churn_df,
        target_column="churn",
        num_folds=3,
        random_state=42,
    )

    run_dir = serialize_run_artifacts(
        evaluation=eval_result,
        artifacts_dir=tmp_path,
        run_id="tamper_detect_run",
    )

    # Corrupt evaluation.json by appending invalid whitespace or bytes
    eval_file = run_dir / "evaluation.json"
    with open(eval_file, "r+", encoding="utf-8") as f:
        content = f.read()
        f.seek(0)
        f.write(content + "\n")  # Byte change changes SHA-256

    with pytest.raises(ArtifactLoadError, match="Integrity check failed for 'evaluation.json'"):
        load_run_artifacts(run_dir)


def test_end_to_end_zero_leakage_and_partition_isolation(mixed_churn_df):
    """Verify partition isolation invariants: no overlap, full coverage, and unmutated caller input."""
    df_copy = mixed_churn_df.copy(deep=True)

    eval_result = evaluate_classification(
        df=mixed_churn_df,
        target_column="churn",
        num_folds=4,
        random_state=42,
    )

    # 1. Caller DataFrame must not be mutated
    pd.testing.assert_frame_equal(mixed_churn_df, df_copy)

    # 2. Validation indices across all folds must partition the dataset with zero overlap
    all_val_indices = []
    n_samples = len(mixed_churn_df)

    for fr in eval_result.candidate_result.fold_results:
        val_idx = fr.val_indices
        assert val_idx is not None
        all_val_indices.extend(val_idx)
        # Train and val partition counts sum to total samples
        assert fr.train_indices_count + fr.val_indices_count == n_samples

    assert sorted(all_val_indices) == list(range(n_samples))
    assert len(all_val_indices) == len(set(all_val_indices))
