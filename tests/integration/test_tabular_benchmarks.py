"""Integration tests for Standard Tabular Dataset Benchmarks (Milestone 2.6).

Verifies evaluation on standard tabular benchmark datasets:
1. Synthetic Continuous Binary (Breast Cancer-style, 569 samples, 30 features)
2. Synthetic Multiclass (Iris-style, 150 samples, 4 features, 3 classes)
3. Synthetic Mixed Churn (Mixed features, missing values, class imbalance)
4. Bit-for-bit determinism and cryptographic hash signature reproducibility across runs.
5. Benchmark runner execution and structured reporting.
"""

from pathlib import Path
import numpy as np
import pytest

from benchmarks.runner import (
    generate_breast_cancer_dataset,
    generate_iris_multiclass_dataset,
    generate_synthetic_churn_dataset,
    run_all_benchmarks,
)
from mlforge.application.artifacts import load_run_artifacts, serialize_run_artifacts
from mlforge.core.contracts import (
    AcceptanceOutcome,
    AcceptancePolicy,
    AcceptanceRule,
    CriterionSeverity,
    RuleOperator,
    TestStatus,
)
from mlforge.engine.evaluator import evaluate_classification


def test_standard_breast_cancer_benchmark_evaluation():
    """Verify evaluation on synthetic breast cancer continuous feature benchmark."""
    df = generate_breast_cancer_dataset(n_samples=569, random_state=42)

    policy = AcceptancePolicy(
        policy_id="bc_readiness_policy",
        version="1.0.0",
        name="Breast Cancer Model Policy",
        rules=[
            AcceptanceRule(
                rule_id="r1_f1_floor",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.70,
                severity=CriterionSeverity.BLOCKER,
            ),
            AcceptanceRule(
                rule_id="r2_stat_superiority",
                metric="accuracy",
                operator=RuleOperator.STATISTICALLY_GREATER_THAN,
                threshold=0.05,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="statistical_test",
            ),
            AcceptanceRule(
                rule_id="r3_ece_ceiling",
                metric="expected_calibration_error",
                operator=RuleOperator.LESS_THAN_OR_EQUAL,
                threshold=0.25,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="calibration",
            ),
        ],
    )

    eval_result = evaluate_classification(
        df=df,
        target_column="diagnosis",
        num_folds=5,
        acceptance_policy=policy,
        random_state=42,
    )

    # 1. Dataset Characteristics
    meta = eval_result.dataset_metadata
    assert meta.num_rows == 569
    assert meta.num_features == 30
    assert meta.numeric_feature_count == 30
    assert meta.categorical_feature_count == 0

    # 2. Performance: LogisticRegression easily beats DummyClassifier
    cand = eval_result.candidate_result
    base = eval_result.baseline_result

    cand_acc = cand.metric_summaries["accuracy"].mean
    base_acc = base.metric_summaries["accuracy"].mean
    assert cand_acc is not None and base_acc is not None
    assert cand_acc > 0.75
    assert base_acc < 0.65
    assert cand_acc > base_acc

    # 3. Statistical Test: Nadeau-Bengio corrected t-test significant
    assert eval_result.comparison_report is not None
    t_test = eval_result.comparison_report.metric_comparisons["accuracy"].corrected_resampled_t_test
    assert t_test.status == TestStatus.COMPLETED
    assert t_test.p_value is not None
    assert t_test.p_value < 0.05
    assert t_test.statistically_significant is True

    # 4. Calibration & Diagnostics
    assert cand.calibration is not None
    assert cand.calibration.expected_calibration_error is not None
    assert cand.calibration.expected_calibration_error < 0.20
    assert cand.calibration.brier_score < 0.25

    # 5. Acceptance Decision
    assert eval_result.acceptance_decision is not None
    assert eval_result.acceptance_decision.outcome == AcceptanceOutcome.PASS
    assert eval_result.acceptance_decision.passed_rules_count == 3
    assert eval_result.acceptance_decision.failed_rules_count == 0


def test_standard_iris_multiclass_benchmark_evaluation():
    """Verify evaluation on synthetic multiclass iris benchmark."""
    df = generate_iris_multiclass_dataset(n_samples=150, random_state=42)

    eval_result = evaluate_classification(
        df=df,
        target_column="species",
        num_folds=5,
        random_state=42,
    )

    # 1. Dataset Metadata
    meta = eval_result.dataset_metadata
    assert meta.num_rows == 150
    assert meta.num_features == 4
    assert meta.class_labels == ["class_0", "class_1", "class_2"]
    assert meta.class_distribution == {"class_0": 50, "class_1": 49, "class_2": 51}
    assert sum(meta.class_distribution.values()) == 150

    # 2. Multiclass Confusion Matrix
    cand = eval_result.candidate_result
    cm = cand.aggregate_confusion_matrix
    assert len(cm) == 3
    assert len(cm[0]) == 3
    # Total samples conserved
    assert sum(sum(row) for row in cm) == 150

    # 3. Macro Metrics
    assert cand.metric_summaries["macro_precision"].mean is not None
    assert cand.metric_summaries["macro_recall"].mean is not None
    assert cand.metric_summaries["macro_f1"].mean is not None
    assert cand.metric_summaries["balanced_accuracy"].mean is not None
    assert cand.metric_summaries["roc_auc"].mean is not None  # OvR macro ROC-AUC


def test_synthetic_churn_mixed_tabular_benchmark_evaluation(tmp_path: Path):
    """Verify evaluation on mixed tabular churn benchmark with missingness and subgroup diagnostics."""
    df = generate_synthetic_churn_dataset(n_samples=300, random_state=42)

    policy = AcceptancePolicy(
        policy_id="churn_policy",
        version="1.0.0",
        name="Churn Deployment Policy",
        rules=[
            AcceptanceRule(
                rule_id="r1_f1",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.60,
                severity=CriterionSeverity.BLOCKER,
            ),
        ],
    )

    eval_result = evaluate_classification(
        df=df,
        target_column="churn",
        num_folds=3,
        subgroup_columns=["region"],
        acceptance_policy=policy,
        random_state=42,
    )

    # 1. Mixed Features & Missing Values
    meta = eval_result.dataset_metadata
    assert meta.numeric_feature_count == 4
    assert meta.categorical_feature_count == 2
    assert meta.missing_value_counts["tenure_num_1"] > 0
    assert meta.missing_value_counts["contract_tier"] > 0

    # 2. Subgroup Diagnostics
    cand = eval_result.candidate_result
    assert cand.error_diagnostics is not None
    assert "region" in cand.error_diagnostics.subgroup_diagnostics
    region_diags = cand.error_diagnostics.subgroup_diagnostics["region"]
    assert len(region_diags) == 4  # North, South, East, West

    # 3. Artifact Serialization Roundtrip
    run_dir = serialize_run_artifacts(
        evaluation=eval_result,
        acceptance_decision=eval_result.acceptance_decision,
        artifacts_dir=tmp_path,
        run_id="churn_bench_01",
        policy=policy,
    )
    loaded = load_run_artifacts(run_dir)
    assert loaded["evaluation"].dataset_metadata.checksum_sha256 == meta.checksum_sha256
    assert loaded["decision"].policy_id == "churn_policy"


def test_deterministic_reproducibility_across_benchmark_runs():
    """Two independent runs with identical inputs and seeds must produce bit-for-bit identical results."""
    df = generate_breast_cancer_dataset(n_samples=200, random_state=42)

    policy = AcceptancePolicy(
        policy_id="repro_policy",
        version="1.0.0",
        name="Reproducibility Policy",
        rules=[
            AcceptanceRule(
                rule_id="r1_f1",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.70,
            )
        ],
    )

    run_1 = evaluate_classification(df, target_column="diagnosis", num_folds=3, random_state=42, acceptance_policy=policy)
    run_2 = evaluate_classification(df, target_column="diagnosis", num_folds=3, random_state=42, acceptance_policy=policy)

    # 1. Dataset Checksums
    assert run_1.dataset_metadata.checksum_sha256 == run_2.dataset_metadata.checksum_sha256

    # 2. Partition Fingerprints per fold
    for f1, f2 in zip(run_1.candidate_result.fold_results, run_2.candidate_result.fold_results):
        assert f1.partition_fingerprint == f2.partition_fingerprint
        assert f1.val_indices == f2.val_indices

    # 3. Metric Summaries
    for k in run_1.candidate_result.metric_summaries:
        m1 = run_1.candidate_result.metric_summaries[k]
        m2 = run_2.candidate_result.metric_summaries[k]
        assert m1.mean == m2.mean
        assert m1.std == m2.std
        assert m1.confidence_interval_95 == m2.confidence_interval_95

    # 4. Statistical Test Statistic and P-Value
    assert run_1.comparison_report is not None and run_2.comparison_report is not None
    t1 = run_1.comparison_report.metric_comparisons["accuracy"].corrected_resampled_t_test
    t2 = run_2.comparison_report.metric_comparisons["accuracy"].corrected_resampled_t_test
    assert t1.statistic == t2.statistic
    assert t1.p_value == t2.p_value

    # 5. Acceptance Decision Hash Signature
    assert run_1.acceptance_decision is not None and run_2.acceptance_decision is not None
    assert run_1.acceptance_decision.hash_signature == run_2.acceptance_decision.hash_signature


def test_benchmark_suite_runner_execution(tmp_path: Path):
    """Verify that the benchmark runner executes all scenarios successfully and writes JSON."""
    report = run_all_benchmarks(output_dir=tmp_path, repeats=1, random_state=42)

    assert report["milestone"] == "2.6"
    assert len(report["scenarios"]) == 3
    for sc in report["scenarios"]:
        assert sc["status"] == "SUCCESS"
        assert sc["error_message"] is None
        assert sc["repeats"] == 1
        assert sc["timings_ms"]["total_elapsed"]["median"] > 0
        assert sc["peak_memory_kb"]["median"] > 0

    assert (tmp_path / "benchmark_results.json").exists()
