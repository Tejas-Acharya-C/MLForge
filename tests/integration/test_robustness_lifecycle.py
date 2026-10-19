"""Integration tests for Robustness & Stress Testing Lifecycle (Milestone 3.4).

Verifies the complete Phase 3 scientific pipeline:
Dataset Ingestion -> Stratified CV & Candidate/Baseline Isolation ->
Multi-Feature & Compound Perturbations -> Severity Sweeps ->
Directional Retention & AURDC Integration -> Robustness Acceptance Policy Evaluation ->
Atomic Artifact Serialization with SHA-256 Manifest Integrity -> Round-Trip Reloading.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from benchmarks.runner import (
    generate_breast_cancer_dataset,
    generate_synthetic_churn_dataset,
    run_robustness_benchmarks,
)
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
    MetricDirection,
    PerturbationType,
    RuleOperator,
)
from mlforge.core.exceptions import ArtifactLoadError
from mlforge.engine.acceptance import evaluate_acceptance_policy
from mlforge.engine.evaluator import evaluate_classification
from mlforge.engine.pipeline import build_fold_pipeline
from mlforge.engine.robustness import (
    create_category_dropout_stress_profile,
    create_compound_stress_profile,
    create_missingness_stress_profile,
    create_noise_stress_profile,
    evaluate_model_robustness,
    evaluate_stress_profile,
    get_standard_stress_suite,
)


@pytest.fixture
def synthetic_churn_lifecycle_df():
    """Deterministic synthetic churn dataset with mixed features and missing values."""
    return generate_synthetic_churn_dataset(n_samples=250, random_state=42)


def test_end_to_end_robustness_evaluation_lifecycle(synthetic_churn_lifecycle_df, tmp_path: Path):
    """Verify full end-to-end lifecycle from data through robustness stress testing to artifact verification."""
    df = synthetic_churn_lifecycle_df

    # 1. Core Model Evaluation (Candidate vs Baseline)
    eval_result = evaluate_classification(
        df=df,
        target_column="churn",
        num_folds=5,
        random_state=42,
    )

    assert eval_result.candidate_result is not None
    assert eval_result.baseline_result is not None

    # Prepare validation partition for stress evaluation
    y = df["churn"]
    X = df.drop(columns=["churn"])
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    n_val = 50
    X_train, X_val = X.iloc[:-n_val], X.iloc[-n_val:]
    y_train, y_val = y.iloc[:-n_val], y.iloc[-n_val:]

    # Train a candidate pipeline on training fold
    candidate_pipeline = build_fold_pipeline(
        model_type="logistic_regression",
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=42,
    )
    candidate_pipeline.fit(X_train, y_train)

    # 2. Candidate Robustness Stress Testing (Full Standard Suite)
    cand_robustness = evaluate_stress_profile(
        estimator=candidate_pipeline,
        X_val=X_val,
        y_val=y_val,
        metric_name="accuracy",
    )
    assert cand_robustness.is_usable is True
    assert len(cand_robustness.curves) == 4  # noise, missingness, dropout, compound
    assert "mean_aurdc" in cand_robustness.summary_retention
    assert cand_robustness.summary_retention["mean_aurdc"] > 0.60
    eval_result.candidate_result.robustness = cand_robustness

    # 3. Baseline Robustness Stress Testing (Noise Profile)
    baseline_pipeline = build_fold_pipeline(
        model_type="dummy_classifier",
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        hyperparameters={"strategy": "prior"},
        random_state=42,
    )
    baseline_pipeline.fit(X_train, y_train)

    base_noise_profile = create_noise_stress_profile(random_state=42)
    base_robustness = evaluate_model_robustness(
        estimator=baseline_pipeline,
        X_val=X_val,
        y_val=y_val,
        configs=[base_noise_profile],
        metric_name="accuracy",
    )
    eval_result.baseline_result.robustness = base_robustness

    # 4. Acceptance Policy with Robustness Rules
    policy = AcceptancePolicy(
        policy_id="pol_robustness_lifecycle_v1",
        version="1.0.0",
        name="Production Robustness Gate",
        fail_precedence=True,
        rules=[
            # Blocker: Mean AURDC floor
            AcceptanceRule(
                rule_id="r1_mean_aurdc",
                metric="mean_aurdc",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.65,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="robustness",
            ),
            # Blocker: Gaussian noise max drop ceiling
            AcceptanceRule(
                rule_id="r2_noise_max_drop",
                metric="gaussian_noise_max_drop_pct",
                operator=RuleOperator.LESS_THAN_OR_EQUAL,
                threshold=35.0,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="robustness",
            ),
            # Blocker: Compound stress severity 0.10 drop ceiling
            AcceptanceRule(
                rule_id="r3_compound_drop_10",
                metric="compound:0.10:drop_pct",
                operator=RuleOperator.LESS_THAN_OR_EQUAL,
                threshold=30.0,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="robustness",
            ),
            # Advisory: Category dropout retention recommendation
            AcceptanceRule(
                rule_id="r4_dropout_retention_adv",
                metric="category_dropout:0.25:retention",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.75,
                severity=CriterionSeverity.ADVISORY,
                evidence_source="robustness",
            ),
            # Blocker: Baseline statistical contrast
            AcceptanceRule(
                rule_id="r5_baseline_superiority",
                metric="accuracy",
                operator=RuleOperator.STATISTICALLY_GREATER_THAN,
                threshold=0.05,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="statistical_test",
            ),
        ],
    )

    decision = evaluate_acceptance_policy(
        evaluation=eval_result,
        policy=policy,
        decision_id="dec_robustness_lifecycle_01",
    )

    assert decision.outcome == AcceptanceOutcome.PASS
    assert decision.passed_rules_count >= 4
    assert decision.failed_rules_count == 0
    assert decision.hash_signature is not None

    # 5. Artifact Serialization
    run_dir = serialize_run_artifacts(
        evaluation=eval_result,
        acceptance_decision=decision,
        artifacts_dir=tmp_path,
        run_id="run_robustness_lifecycle_01",
        policy=policy,
    )

    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "evaluation.json").exists()
    assert (run_dir / "decision.json").exists()
    assert (run_dir / "policy.json").exists()

    # 6. Artifact Verification and Round-Trip Reload
    loaded = load_run_artifacts(run_dir)
    manifest = loaded["manifest"]
    loaded_eval = loaded["evaluation"]
    loaded_dec = loaded["decision"]
    loaded_pol = loaded["policy"]

    # Manifest checks
    assert manifest.has_robustness is True
    assert manifest.schema_version == CURRENT_SCHEMA_VERSION
    assert manifest.run_id == "run_robustness_lifecycle_01"
    assert "evaluation.json" in manifest.file_checksums

    # Candidate robustness reload checks
    cand_rob_loaded = loaded_eval.candidate_result.robustness
    assert cand_rob_loaded is not None
    assert len(cand_rob_loaded.curves) == 4
    assert cand_rob_loaded.summary_retention["mean_aurdc"] == pytest.approx(
        cand_robustness.summary_retention["mean_aurdc"], abs=1e-6
    )

    # Baseline robustness reload checks
    base_rob_loaded = loaded_eval.baseline_result.robustness
    assert base_rob_loaded is not None
    assert len(base_rob_loaded.curves) == 1
    assert base_rob_loaded.curves[0].perturbation_type == PerturbationType.GAUSSIAN_NOISE

    # Decision signature integrity
    assert loaded_dec.hash_signature == decision.hash_signature
    assert loaded_dec.outcome == AcceptanceOutcome.PASS
    assert loaded_pol.policy_id == policy.policy_id


def test_robustness_lifecycle_blocker_failure_and_inconclusive_precedence(synthetic_churn_lifecycle_df, tmp_path: Path):
    """Verify that blocker robustness violations trigger FAIL and respect fail_precedence."""
    df = synthetic_churn_lifecycle_df
    eval_result = evaluate_classification(df, target_column="churn", num_folds=3, random_state=42)

    # Attach robustness where mean AURDC is ~0.80
    cand_noise = create_noise_stress_profile(random_state=42)
    rob = evaluate_model_robustness(
        estimator=build_fold_pipeline("logistic_regression", ["tenure_num_0"], ["contract_tier"], random_state=42).fit(
            df[["tenure_num_0", "contract_tier"]], df["churn"]
        ),
        X_val=df[["tenure_num_0", "contract_tier"]],
        y_val=df["churn"],
        configs=[cand_noise],
    )
    eval_result.candidate_result.robustness = rob

    # Policy with unreachable AURDC threshold (0.999) and an unavailable curve
    policy_fail = AcceptancePolicy(
        policy_id="pol_blocker_fail",
        version="1.0.0",
        name="Failing Robustness Policy",
        fail_precedence=True,
        rules=[
            # This blocker will conclusively fail: observed ~0.80 < 0.999
            AcceptanceRule(
                rule_id="r_unreachable_aurdc",
                metric="gaussian_noise_aurdc",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.999,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="robustness",
            ),
            # This blocker will evaluate to INCONCLUSIVE (curve not evaluated)
            AcceptanceRule(
                rule_id="r_missing_curve",
                metric="category_dropout_aurdc",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.50,
                severity=CriterionSeverity.BLOCKER,
                evidence_source="robustness",
            ),
        ],
    )

    # 1. Under fail_precedence=True, FAIL > INCONCLUSIVE -> Outcome is FAIL
    dec_fail = evaluate_acceptance_policy(eval_result, policy_fail)
    assert dec_fail.outcome == AcceptanceOutcome.FAIL
    assert dec_fail.failed_rules_count == 1
    assert dec_fail.inconclusive_rules_count == 1

    # 2. Under fail_precedence=False, INCONCLUSIVE > FAIL -> Outcome is INCONCLUSIVE
    policy_fail.fail_precedence = False
    dec_inc = evaluate_acceptance_policy(eval_result, policy_fail)
    assert dec_inc.outcome == AcceptanceOutcome.INCONCLUSIVE

    # 3. Serialization and reload of failed decision preserves failure state
    run_dir = serialize_run_artifacts(
        evaluation=eval_result,
        acceptance_decision=dec_fail,
        artifacts_dir=tmp_path,
        run_id="run_rob_blocker_fail",
        policy=policy_fail,
    )
    loaded = load_run_artifacts(run_dir)
    assert loaded["decision"].outcome == AcceptanceOutcome.FAIL


def test_robustness_lifecycle_lower_is_better_metric(synthetic_churn_lifecycle_df, tmp_path: Path):
    """Verify robustness evaluation lifecycle using a lower-is-better metric (log_loss)."""
    df = synthetic_churn_lifecycle_df

    # Prepare features and fit pipeline
    y = df["churn"]
    X = df.drop(columns=["churn"])
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    pipeline = build_fold_pipeline(
        model_type="logistic_regression",
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=42,
    )
    pipeline.fit(X.iloc[:180], y.iloc[:180])
    X_val = X.iloc[180:]
    y_val = y.iloc[180:]

    # Evaluate noise profile on log_loss
    noise_cfg = create_noise_stress_profile(target_columns=numeric_features, random_state=42)
    rob_log_loss = evaluate_model_robustness(
        estimator=pipeline,
        X_val=X_val,
        y_val=y_val,
        configs=[noise_cfg],
        metric_name="log_loss",
    )

    assert rob_log_loss.is_usable is True
    curve = rob_log_loss.curves[0]
    assert curve.metric_name == "log_loss"
    assert curve.clean_score is not None
    assert curve.clean_score > 0.0

    # Perturbed loss degrades (increases), drop_pct > 0, retained_ratio < 1
    for pt in curve.points:
        assert pt.metric_name == "log_loss"
        if pt.perturbed_score is not None and pt.perturbed_score > curve.clean_score:
            assert pt.performance_drop_percentage > 0.0
            assert pt.retained_ratio < 1.0

    # Acceptance rule checking max drop on log_loss
    eval_res = evaluate_classification(df, target_column="churn", num_folds=3, random_state=42)
    eval_res.candidate_result.robustness = rob_log_loss

    policy = AcceptancePolicy(
        policy_id="pol_log_loss_robustness",
        version="1.0.0",
        name="Log Loss Policy",
        rules=[
            AcceptanceRule(
                rule_id="r_log_loss_max_drop",
                metric="gaussian_noise_max_drop_pct",
                operator=RuleOperator.LESS_THAN_OR_EQUAL,
                threshold=200.0,  # Bound max loss degradation to 200%
                severity=CriterionSeverity.BLOCKER,
                evidence_source="robustness",
            ),
        ],
    )

    decision = evaluate_acceptance_policy(eval_res, policy)
    assert decision.outcome == AcceptanceOutcome.PASS

    # Serialization roundtrip preserves log_loss evidence
    run_dir = serialize_run_artifacts(eval_res, decision, tmp_path, run_id="run_log_loss_rob", policy=policy)
    loaded = load_run_artifacts(run_dir)
    assert loaded["manifest"].has_robustness is True
    loaded_curve = loaded["evaluation"].candidate_result.robustness.curves[0]
    assert loaded_curve.metric_name == "log_loss"


def test_robustness_lifecycle_tampering_detection(synthetic_churn_lifecycle_df, tmp_path: Path):
    """Verify that tampering with robustness evidence in evaluation.json is detected by manifest checksums."""
    df = synthetic_churn_lifecycle_df
    eval_res = evaluate_classification(df, target_column="churn", num_folds=3, random_state=42)

    pipeline = build_fold_pipeline("logistic_regression", ["tenure_num_0"], ["contract_tier"], random_state=42)
    pipeline.fit(df[["tenure_num_0", "contract_tier"]], df["churn"])
    rob = evaluate_model_robustness(
        pipeline, df[["tenure_num_0", "contract_tier"]], df["churn"],
        configs=[create_noise_stress_profile(random_state=42)],
    )
    eval_res.candidate_result.robustness = rob

    run_dir = serialize_run_artifacts(
        evaluation=eval_res,
        artifacts_dir=tmp_path,
        run_id="run_rob_tamper_test",
    )

    # Tamper with evaluation.json by altering summary_retention
    eval_file = run_dir / "evaluation.json"
    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["candidate_result"]["robustness"]["summary_retention"]["gaussian_noise_aurdc"] = 0.999999
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    with pytest.raises(ArtifactLoadError, match="Integrity check failed for 'evaluation.json'"):
        load_run_artifacts(run_dir)


def test_robustness_lifecycle_deterministic_reproducibility(synthetic_churn_lifecycle_df):
    """Verify bit-for-bit determinism across two independent robustness evaluations with identical seed."""
    df = synthetic_churn_lifecycle_df

    def _execute_lifecycle():
        eval_res = evaluate_classification(df, target_column="churn", num_folds=3, random_state=42)
        y = df["churn"]
        X = df.drop(columns=["churn"])
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = [c for c in X.columns if c not in numeric_features]

        pipeline = build_fold_pipeline("logistic_regression", numeric_features, categorical_features, random_state=42)
        pipeline.fit(X.iloc[:180], y.iloc[:180])
        suite = get_standard_stress_suite(random_state=42)
        rob = evaluate_stress_profile(pipeline, X.iloc[180:], y.iloc[180:], profile=suite)
        eval_res.candidate_result.robustness = rob

        policy = AcceptancePolicy(
            policy_id="pol_repro",
            version="1.0.0",
            name="Repro Policy",
            rules=[
                AcceptanceRule(
                    rule_id="r_aurdc",
                    metric="mean_aurdc",
                    operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                    threshold=0.60,
                    severity=CriterionSeverity.BLOCKER,
                    evidence_source="robustness",
                ),
            ],
        )
        dec = evaluate_acceptance_policy(eval_res, policy, decision_id="dec_fixed_id")
        return eval_res, dec

    res1, dec1 = _execute_lifecycle()
    res2, dec2 = _execute_lifecycle()

    # 1. Curve and point level reproducibility
    rob1 = res1.candidate_result.robustness
    rob2 = res2.candidate_result.robustness
    assert len(rob1.curves) == len(rob2.curves)
    for c1, c2 in zip(rob1.curves, rob2.curves):
        assert c1.perturbation_type == c2.perturbation_type
        assert c1.area_under_degradation_curve == c2.area_under_degradation_curve
        assert len(c1.points) == len(c2.points)
        for p1, p2 in zip(c1.points, c2.points):
            assert p1.severity_level == p2.severity_level
            assert p1.retained_ratio == p2.retained_ratio
            assert p1.performance_drop_percentage == p2.performance_drop_percentage

    # 2. Summary retention scores
    assert rob1.summary_retention == rob2.summary_retention

    # 3. Acceptance decision signature
    assert dec1.hash_signature == dec2.hash_signature
    assert dec1.outcome == dec2.outcome


def test_robustness_benchmark_runner_execution(tmp_path: Path):
    """Verify that run_robustness_benchmarks executes all scenarios and outputs valid timing & memory JSON."""
    report = run_robustness_benchmarks(output_dir=tmp_path, repeats=1, random_state=42)

    assert report["milestone"] == "3.4"
    assert len(report["scenarios"]) == 2
    for sc in report["scenarios"]:
        assert sc["status"] == "SUCCESS"
        assert sc["error_message"] is None
        assert sc["repeats"] == 1
        t = sc["timings_ms"]
        assert t["total_elapsed"]["median"] > 0
        assert t["full_standard_suite"]["median"] > 0
        assert t["acceptance_policy_evaluation"]["median"] > 0
        assert t["artifact_serialization"]["median"] > 0
        assert sc["peak_traced_memory_kb"]["median"] > 0

    assert (tmp_path / "robustness_benchmark_results.json").exists()
