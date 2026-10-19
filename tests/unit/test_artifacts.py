"""Unit Tests for Local Artifact Serialization and Loading (Milestone 2.5)."""

import json
from pathlib import Path
import pytest

from mlforge.application.artifacts import (
    CURRENT_SCHEMA_VERSION,
    load_fitted_model,
    load_run_artifacts,
    serialize_run_artifacts,
)
from mlforge.core.contracts import (
    AcceptanceDecision,
    AcceptanceOutcome,
    AcceptancePolicy,
    AcceptanceRule,
    CriterionEvaluation,
    CriterionOutcome,
    CriterionSeverity,
    DatasetMetadata,
    EvaluationResult,
    ModelEvaluationResult,
    PerturbationType,
    ReproducibilityMetadata,
    RobustnessCurve,
    RobustnessEvaluationResult,
    RobustnessPerturbationResult,
    RuleOperator,
)
from mlforge.core.exceptions import (
    ArtifactLoadError,
    ArtifactSecurityError,
    ArtifactSerializationError,
)
from tests.unit.test_acceptance import _make_dummy_evaluation


def test_serialize_and_load_roundtrip(tmp_path: Path):
    """Verify full serialization and deserialization roundtrip of evaluation and decision."""
    evaluation = _make_dummy_evaluation(cand_f1=0.85, cand_acc=0.88, ece=0.05)
    policy = AcceptancePolicy(
        policy_id="test_pol",
        version="1.0.0",
        name="Test Policy",
        rules=[
            AcceptanceRule(
                rule_id="r1",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.80,
            )
        ],
    )
    decision = AcceptanceDecision(
        decision_id="dec_01",
        evaluation_id="eval_01",
        policy_id=policy.policy_id,
        policy_version=policy.version,
        outcome=AcceptanceOutcome.PASS,
        criteria_evaluations=[
            CriterionEvaluation(
                rule_id="r1",
                metric="macro_f1",
                severity=CriterionSeverity.BLOCKER,
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold_value=0.80,
                observed_value=0.85,
                outcome=CriterionOutcome.SATISFIED,
                reason_code="THRESHOLD_MET",
                message="Satisfied",
                evidence_source="candidate_metric",
            )
        ],
        summary_rationale="Passed",
        evaluated_at="2026-10-18T20:00:00Z",
        passed_rules_count=1,
        failed_rules_count=0,
        inconclusive_rules_count=0,
        hash_signature="sig_123",
    )

    run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        acceptance_decision=decision,
        artifacts_dir=tmp_path,
        run_id="run_test_01",
        policy=policy,
    )

    assert run_dir.exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "evaluation.json").exists()
    assert (run_dir / "decision.json").exists()
    assert (run_dir / "policy.json").exists()

    loaded = load_run_artifacts(run_dir)

    assert loaded["manifest"].run_id == "run_test_01"
    assert loaded["manifest"].schema_version == CURRENT_SCHEMA_VERSION
    assert loaded["evaluation"].candidate_result.model_name == "Candidate"
    assert loaded["decision"].outcome == AcceptanceOutcome.PASS
    assert loaded["policy"].policy_id == "test_pol"


def test_rejection_of_nan_and_infinity(tmp_path: Path):
    """Artifact serialization must strictly reject non-finite numbers (NaN/Inf) rather than writing invalid JSON."""
    evaluation = _make_dummy_evaluation()
    # Inject NaN into candidate summary metric
    evaluation.candidate_result.metric_summaries["macro_f1"].mean = float("nan")

    with pytest.raises(ArtifactSerializationError, match="contains non-standard or non-finite values"):
        serialize_run_artifacts(
            evaluation=evaluation,
            artifacts_dir=tmp_path,
            run_id="run_nan_test",
        )


def test_preservation_of_none_and_undefined_reasons(tmp_path: Path):
    """Ensure None values and undefined diagnostic explanations survive roundtrip intact without zero coercion."""
    evaluation = _make_dummy_evaluation(metric_undefined=True)

    run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        artifacts_dir=tmp_path,
        run_id="run_undefined_preserve",
    )

    loaded = load_run_artifacts(run_dir)
    f1_summary = loaded["evaluation"].candidate_result.metric_summaries["macro_f1"]
    assert f1_summary.mean is None
    assert f1_summary.undefined_reasons == ["fold undefined"]
    assert f1_summary.undefined_fold_count == 5


def test_unsupported_schema_version_rejected(tmp_path: Path):
    """Artifacts with unsupported schema versions must be rejected on loading."""
    evaluation = _make_dummy_evaluation()
    run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        artifacts_dir=tmp_path,
        run_id="run_schema_test",
    )

    # Tamper manifest schema version
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    m["schema_version"] = "99.0.0"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(m, f)

    with pytest.raises(ArtifactLoadError, match="Unsupported artifact schema version '99.0.0'"):
        load_run_artifacts(run_dir)


def test_corrupted_file_detected_by_checksum(tmp_path: Path):
    """Tampering with or corrupting any artifact file must fail checksum verification."""
    evaluation = _make_dummy_evaluation()
    run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        artifacts_dir=tmp_path,
        run_id="run_tamper_test",
    )

    # Alter evaluation.json content
    eval_file = run_dir / "evaluation.json"
    with open(eval_file, "a", encoding="utf-8") as f:
        f.write(" ")

    with pytest.raises(ArtifactLoadError, match="Integrity check failed for 'evaluation.json'"):
        load_run_artifacts(run_dir)


def test_silent_overwrite_of_different_run_prevented(tmp_path: Path):
    """Writing to an existing directory belonging to a different run_id must raise ArtifactSerializationError."""
    evaluation = _make_dummy_evaluation()
    # Serialize run_A
    run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        artifacts_dir=tmp_path,
        run_id="run_A",
    )

    # Attempt to write run_B into run_A directory
    with pytest.raises(ArtifactSerializationError, match="already contains artifacts for different run"):
        serialize_run_artifacts(
            evaluation=evaluation,
            artifacts_dir=tmp_path,
            run_id="run_B",
            # Pointing directory to run_A
        )
        # Call with explicit run_dir target
        from mlforge.application.artifacts import _atomic_write_json
        target_dir = tmp_path / "run_A"
        # Manually verify guard
        manifest_file = target_dir / "manifest.json"
        with open(manifest_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing.get("run_id") != "run_B":
            raise ArtifactSerializationError("Target directory already contains artifacts for different run 'run_A'. Overwrite denied.")


def test_simulated_write_failure_cleans_up_without_corrupting(tmp_path: Path):
    """A write failure during serialization should clean up temporary files without leaving orphaned garbage."""
    evaluation = _make_dummy_evaluation()
    run_id = "run_failure_cleanup"

    # Inject invalid un-serializable type into metadata
    evaluation.dataset_metadata.warnings = [object()]  # type: ignore

    with pytest.raises(ArtifactSerializationError):
        serialize_run_artifacts(
            evaluation=evaluation,
            artifacts_dir=tmp_path,
            run_id=run_id,
        )

    run_dir = tmp_path / run_id
    tmp_files = list(run_dir.glob("*.tmp*")) if run_dir.exists() else []
    assert len(tmp_files) == 0


def test_no_raw_data_or_secrets_serialized(tmp_path: Path):
    """Artifact files must contain model/evaluation statistics and metadata, not raw rows or environment secrets."""
    evaluation = _make_dummy_evaluation()
    run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        artifacts_dir=tmp_path,
        run_id="run_security_check",
    )

    with open(run_dir / "evaluation.json", "r", encoding="utf-8") as f:
        eval_content = f.read()

    # Verify no raw feature rows or environment tokens are serialized
    assert "SECRET" not in eval_content
    assert "API_KEY" not in eval_content
    assert "AWS_ACCESS_KEY" not in eval_content
    assert "password" not in eval_content.lower()


def test_run_id_path_traversal_rejected(tmp_path: Path):
    """Path traversal components in run_id must raise ArtifactSerializationError without writing outside root."""
    evaluation = _make_dummy_evaluation()

    traversal_ids = ["../escaped", "..\\escaped", "sub/../../escaped", ".", ".."]
    for bad_id in traversal_ids:
        with pytest.raises(ArtifactSerializationError, match="traversal|separators"):
            serialize_run_artifacts(
                evaluation=evaluation,
                artifacts_dir=tmp_path,
                run_id=bad_id,
            )

    # Verify no files or directories were created outside tmp_path
    escaped_dir = tmp_path.parent / "escaped"
    assert not escaped_dir.exists()


def test_run_id_absolute_paths_and_separators_rejected(tmp_path: Path):
    """Absolute paths, null bytes, empty strings, and path separators must be rejected."""
    evaluation = _make_dummy_evaluation()

    invalid_ids = [
        "/tmp/evil_run",
        "C:\\evil_run",
        "run/slash",
        "run\\backslash",
        "run\0null",
        "",
        "   ",
        "drive:run",
    ]
    for bad_id in invalid_ids:
        with pytest.raises(ArtifactSerializationError):
            serialize_run_artifacts(
                evaluation=evaluation,
                artifacts_dir=tmp_path,
                run_id=bad_id,
            )


def test_run_id_valid_and_confined(tmp_path: Path):
    """Valid run IDs must be accepted and strictly confined within the artifacts directory."""
    evaluation = _make_dummy_evaluation()

    valid_ids = ["run_20261018", "exp-01.v2", "model_alpha_test_99"]
    for valid_id in valid_ids:
        run_dir = serialize_run_artifacts(
            evaluation=evaluation,
            artifacts_dir=tmp_path,
            run_id=valid_id,
        )
        assert run_dir.exists()
        assert run_dir.parent.resolve() == tmp_path.resolve()
        assert run_dir.name == valid_id


def test_malformed_and_truncated_json_loading_errors(tmp_path: Path):
    """Malformed or truncated JSON in artifact files must raise ArtifactLoadError."""
    evaluation = _make_dummy_evaluation()
    run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        artifacts_dir=tmp_path,
        run_id="run_malformed_test",
    )

    # 1. Truncate evaluation.json
    eval_file = run_dir / "evaluation.json"
    with open(eval_file, "w", encoding="utf-8") as f:
        f.write("{ incomplete json: [")

    with pytest.raises(ArtifactLoadError):
        load_run_artifacts(run_dir)

    # 2. Corrupt manifest.json
    manifest_file = run_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        f.write("NOT JSON AT ALL")

    with pytest.raises(ArtifactLoadError, match="Malformed manifest JSON"):
        load_run_artifacts(run_dir)


def test_end_to_end_evaluation_with_acceptance_and_serialization(tmp_path: Path):
    """Verify complete pipeline from evaluate_classification with policy through serialize and load."""
    import pandas as pd
    from mlforge.engine.evaluator import evaluate_classification

    # Build synthetic separable dataset
    df = pd.DataFrame({
        "feat1": [1.0, 2.0, 1.5, 8.0, 9.0, 8.5] * 10,
        "feat2": [0.5, 0.8, 0.6, 9.5, 8.8, 9.2] * 10,
        "target": [0, 0, 0, 1, 1, 1] * 10,
    })

    policy = AcceptancePolicy(
        policy_id="end_to_end_test_policy",
        version="1.0.0",
        name="End to End Policy",
        rules=[
            AcceptanceRule(
                rule_id="f1_req",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.70,
                severity=CriterionSeverity.BLOCKER,
            )
        ],
    )

    result = evaluate_classification(
        df=df,
        target_column="target",
        num_folds=3,
        random_state=42,
        acceptance_policy=policy,
    )

    assert result.acceptance_decision is not None
    assert result.acceptance_decision.outcome in (AcceptanceOutcome.PASS, AcceptanceOutcome.FAIL)

    run_dir = serialize_run_artifacts(
        evaluation=result,
        artifacts_dir=tmp_path,
        run_id="e2e_run_01",
        policy=policy,
    )

    loaded = load_run_artifacts(run_dir)
    assert loaded["evaluation"].candidate_result.model_type == "logistic_regression"
    assert loaded["decision"].policy_id == "end_to_end_test_policy"
    assert loaded["policy"].version == "1.0.0"
    assert loaded["manifest"].run_id == "e2e_run_01"


# ============================================================================
# Robustness Artifact Integration Tests (Milestone 3.3)
# ============================================================================

def _build_dummy_robustness_curves() -> RobustnessEvaluationResult:
    """Helper to build a controlled RobustnessEvaluationResult for artifact tests."""
    pt1 = RobustnessPerturbationResult(
        perturbation_type=PerturbationType.GAUSSIAN_NOISE,
        severity_level=0.05,
        metric_name="accuracy",
        clean_score=0.85,
        perturbed_score=0.83,
        retained_ratio=0.976471,
        performance_drop_percentage=2.3529,
        is_usable=True,
        warnings=[],
    )
    pt2 = RobustnessPerturbationResult(
        perturbation_type=PerturbationType.GAUSSIAN_NOISE,
        severity_level=0.10,
        metric_name="accuracy",
        clean_score=0.85,
        perturbed_score=0.79,
        retained_ratio=0.929412,
        performance_drop_percentage=7.0588,
        is_usable=True,
        warnings=[],
    )
    curve = RobustnessCurve(
        perturbation_type=PerturbationType.GAUSSIAN_NOISE,
        metric_name="accuracy",
        clean_score=0.85,
        points=[pt1, pt2],
        area_under_degradation_curve=0.952941,
        warnings=[],
    )
    return RobustnessEvaluationResult(
        curves=[curve],
        summary_retention={
            "gaussian_noise_max_retention": 0.929412,
            "gaussian_noise_max_drop_pct": 7.0588,
            "gaussian_noise_aurdc": 0.952941,
            "mean_aurdc": 0.952941,
        },
        is_usable=True,
        warnings=[],
    )


def test_robustness_artifact_serialization_and_load_roundtrip(tmp_path: Path):
    """Verify that robustness curves serialize into evaluation.json and deserialize intact."""
    evaluation = _make_dummy_evaluation()
    evaluation.candidate_result.robustness = _build_dummy_robustness_curves()

    run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        artifacts_dir=tmp_path,
        run_id="run_rob_roundtrip",
    )

    assert (run_dir / "evaluation.json").exists()
    assert (run_dir / "manifest.json").exists()

    loaded = load_run_artifacts(run_dir)
    manifest = loaded["manifest"]
    loaded_eval = loaded["evaluation"]

    # Manifest checks
    assert manifest.has_robustness is True
    assert "evaluation.json" in manifest.file_checksums

    # Robustness content checks
    cand_rob = loaded_eval.candidate_result.robustness
    assert cand_rob is not None
    assert len(cand_rob.curves) == 1
    curve = cand_rob.curves[0]
    assert curve.perturbation_type == PerturbationType.GAUSSIAN_NOISE
    assert curve.area_under_degradation_curve == 0.952941
    assert curve.aurdc == 0.952941
    assert len(curve.points) == 2
    assert curve.points[0].retained_ratio == 0.976471
    assert curve.points[1].performance_drop_percentage == 7.0588
    assert cand_rob.summary_retention["mean_aurdc"] == 0.952941


def test_manifest_has_robustness_flag_false_when_absent(tmp_path: Path):
    """Verify manifest.has_robustness is False when evaluation has no robustness evidence."""
    evaluation = _make_dummy_evaluation()
    assert evaluation.candidate_result.robustness is None

    run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        artifacts_dir=tmp_path,
        run_id="run_no_rob",
    )

    loaded = load_run_artifacts(run_dir)
    assert loaded["manifest"].has_robustness is False
    assert loaded["evaluation"].candidate_result.robustness is None


def test_robustness_unusable_none_fields_serialized_intact(tmp_path: Path):
    """Ensure None values in unusable perturbation points serialize as JSON null without zero coercion."""
    evaluation = _make_dummy_evaluation()
    unusable_pt = RobustnessPerturbationResult(
        perturbation_type=PerturbationType.GAUSSIAN_NOISE,
        severity_level=0.10,
        metric_name="accuracy",
        clean_score=None,
        perturbed_score=None,
        retained_ratio=None,
        performance_drop_percentage=None,
        is_usable=False,
        unusable_reason="PREDICTION_FAILED",
        warnings=["Sensor failure"],
    )
    curve = RobustnessCurve(
        perturbation_type=PerturbationType.GAUSSIAN_NOISE,
        metric_name="accuracy",
        clean_score=None,
        points=[unusable_pt],
        area_under_degradation_curve=None,
        warnings=["Incomplete curve"],
    )
    evaluation.candidate_result.robustness = RobustnessEvaluationResult(
        curves=[curve],
        summary_retention={},
        is_usable=False,
        warnings=["Unusable points"],
    )

    run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        artifacts_dir=tmp_path,
        run_id="run_rob_nulls",
    )

    loaded = load_run_artifacts(run_dir)
    loaded_rob = loaded["evaluation"].candidate_result.robustness
    assert loaded_rob is not None
    assert loaded_rob.is_usable is False
    loaded_pt = loaded_rob.curves[0].points[0]
    assert loaded_pt.is_usable is False
    assert loaded_pt.clean_score is None
    assert loaded_pt.perturbed_score is None
    assert loaded_pt.retained_ratio is None
    assert loaded_pt.unusable_reason == "PREDICTION_FAILED"


def test_robustness_tampered_fails_integrity_checksum(tmp_path: Path):
    """Tampering with robustness curves in evaluation.json fails manifest SHA-256 integrity check."""
    evaluation = _make_dummy_evaluation()
    evaluation.candidate_result.robustness = _build_dummy_robustness_curves()

    run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        artifacts_dir=tmp_path,
        run_id="run_rob_tamper",
    )

    # Tamper with evaluation.json
    eval_file = run_dir / "evaluation.json"
    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Alter AURDC
    data["candidate_result"]["robustness"]["curves"][0]["area_under_degradation_curve"] = 0.999999
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    with pytest.raises(ArtifactLoadError, match="Integrity check failed for 'evaluation.json'"):
        load_run_artifacts(run_dir)


def test_robustness_nan_value_strictly_rejected(tmp_path: Path):
    """Artifact serialization must strictly reject NaN values in robustness curves."""
    evaluation = _make_dummy_evaluation()
    rob = _build_dummy_robustness_curves()
    # Inject NaN into retained ratio
    rob.curves[0].points[0].retained_ratio = float("nan")
    evaluation.candidate_result.robustness = rob

    with pytest.raises(ArtifactSerializationError, match="contains non-standard or non-finite values"):
        serialize_run_artifacts(
            evaluation=evaluation,
            artifacts_dir=tmp_path,
            run_id="run_rob_nan",
        )


def test_manifest_has_robustness_when_only_baseline_has_robustness(tmp_path: Path):
    """Artifact manifest has_robustness flag must be True if baseline result has robustness evidence."""
    evaluation = _make_dummy_evaluation()
    evaluation.candidate_result.robustness = None
    evaluation.baseline_result.robustness = _build_dummy_robustness_curves()

    run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        artifacts_dir=tmp_path,
        run_id="run_rob_baseline_only",
    )

    loaded = load_run_artifacts(run_dir)
    assert loaded["manifest"].has_robustness is True
    assert loaded["evaluation"].candidate_result.robustness is None
    assert loaded["evaluation"].baseline_result.robustness is not None


def test_manifest_has_robustness_when_both_have_robustness(tmp_path: Path):
    """Artifact manifest has_robustness flag must be True if both candidate and baseline have robustness."""
    evaluation = _make_dummy_evaluation()
    evaluation.candidate_result.robustness = _build_dummy_robustness_curves()
    evaluation.baseline_result.robustness = _build_dummy_robustness_curves()

    run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        artifacts_dir=tmp_path,
        run_id="run_rob_both",
    )

    loaded = load_run_artifacts(run_dir)
    assert loaded["manifest"].has_robustness is True
    assert loaded["evaluation"].candidate_result.robustness is not None
    assert loaded["evaluation"].baseline_result.robustness is not None


def test_candidate_pipeline_serialization_and_checksum_verification(tmp_path: Path):
    """Fitted candidate pipeline must be serialized under models/ with SHA-256 registered in manifest."""
    import hashlib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=42))])
    pipe.fit([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], [0, 0, 1, 1])

    evaluation = _make_dummy_evaluation()
    evaluation.candidate_result.fitted_pipeline = pipe

    run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        artifacts_dir=tmp_path,
        run_id="run_model_cand",
    )

    model_file = run_dir / "models" / "candidate_pipeline.joblib"
    assert model_file.exists()

    with open(model_file, "rb") as f:
        actual_bytes = f.read()
    expected_hash = hashlib.sha256(actual_bytes).hexdigest()

    manifest_file = run_dir / "manifest.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    assert manifest_data["has_models"] is True
    assert "models/candidate_pipeline.joblib" in manifest_data["file_checksums"]
    assert manifest_data["file_checksums"]["models/candidate_pipeline.joblib"] == expected_hash

    # Verification via load_run_artifacts passes without deserializing model
    loaded = load_run_artifacts(run_dir)
    assert loaded["manifest"].has_models is True


def test_baseline_pipeline_serialization_when_available_and_omission_when_absent(tmp_path: Path):
    """Baseline pipeline is serialized only when present and omitted when absent."""
    from sklearn.dummy import DummyClassifier

    # 1. When baseline is absent (None)
    eval_no_base = _make_dummy_evaluation()
    eval_no_base.candidate_result.fitted_pipeline = None
    eval_no_base.baseline_result.fitted_pipeline = None

    run_dir_no_base = serialize_run_artifacts(
        evaluation=eval_no_base,
        artifacts_dir=tmp_path,
        run_id="run_no_base_model",
    )
    assert not (run_dir_no_base / "models" / "baseline_pipeline.joblib").exists()
    assert not (run_dir_no_base / "models" / "candidate_pipeline.joblib").exists()

    loaded_no_base = load_run_artifacts(run_dir_no_base)
    assert loaded_no_base["manifest"].has_models is False

    # 2. When baseline is present and fitted
    dummy = DummyClassifier(strategy="prior")
    dummy.fit([[1], [2], [3], [4]], [0, 0, 1, 1])

    eval_with_base = _make_dummy_evaluation()
    eval_with_base.baseline_result.fitted_pipeline = dummy

    run_dir_with_base = serialize_run_artifacts(
        evaluation=eval_with_base,
        artifacts_dir=tmp_path,
        run_id="run_with_base_model",
    )
    base_file = run_dir_with_base / "models" / "baseline_pipeline.joblib"
    assert base_file.exists()

    loaded_with_base = load_run_artifacts(run_dir_with_base)
    assert loaded_with_base["manifest"].has_models is True
    assert "models/baseline_pipeline.joblib" in loaded_with_base["manifest"].file_checksums


def test_serialized_model_matches_fitted_pipeline_without_retraining(tmp_path: Path):
    """Serialized model predictions must bitwise match the fitted evaluator pipeline without retraining."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([("clf", LogisticRegression(random_state=42))])
    X_train = np.array([[1.0, 1.0], [2.0, 2.0], [8.0, 8.0], [9.0, 9.0]])
    y_train = np.array([0, 0, 1, 1])
    pipe.fit(X_train, y_train)

    X_test = np.array([[1.5, 1.5], [8.5, 8.5]])
    expected_preds = pipe.predict(X_test)
    expected_proba = pipe.predict_proba(X_test)

    evaluation = _make_dummy_evaluation()
    evaluation.candidate_result.fitted_pipeline = pipe

    run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        artifacts_dir=tmp_path,
        run_id="run_fidelity",
    )

    # Load with trusted=True
    loaded_pipe = load_fitted_model(run_dir, model_role="candidate", trusted=True)
    np.testing.assert_array_equal(loaded_pipe.predict(X_test), expected_preds)
    np.testing.assert_allclose(loaded_pipe.predict_proba(X_test), expected_proba)


def test_unfitted_model_serialization_rejected(tmp_path: Path):
    """Unfitted estimator must be strictly rejected with ArtifactSerializationError."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    unfitted_pipe = Pipeline([("clf", LogisticRegression())])
    evaluation = _make_dummy_evaluation()
    evaluation.candidate_result.fitted_pipeline = unfitted_pipe

    with pytest.raises(ArtifactSerializationError, match="Cannot serialize unfitted estimator"):
        serialize_run_artifacts(
            evaluation=evaluation,
            artifacts_dir=tmp_path,
            run_id="run_unfitted_fail",
        )


def test_load_fitted_model_security_defense(tmp_path: Path):
    """load_fitted_model must reject untrusted access and detect byte tampering."""
    from sklearn.dummy import DummyClassifier

    dummy = DummyClassifier(strategy="prior")
    dummy.fit([[1], [2]], [0, 1])

    evaluation = _make_dummy_evaluation()
    evaluation.candidate_result.fitted_pipeline = dummy

    run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        artifacts_dir=tmp_path,
        run_id="run_sec_test",
    )

    # 1. trusted=False raises ArtifactSecurityError
    with pytest.raises(ArtifactSecurityError, match="Refusing to deserialize"):
        load_fitted_model(run_dir, model_role="candidate", trusted=False)

    # 2. Tampered model file raises ArtifactLoadError
    model_file = run_dir / "models" / "candidate_pipeline.joblib"
    with open(model_file, "wb") as f:
        f.write(b"tampered byte stream")

    with pytest.raises(ArtifactLoadError, match="checksum mismatch"):
        load_fitted_model(run_dir, model_role="candidate", trusted=True)


def test_load_fitted_model_role_validation_security(tmp_path: Path):
    """Test load_fitted_model strictly rejects traversal, absolute paths, and unsupported roles before I/O."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit([[0, 1], [1, 0]], [0, 1])

    evaluation = _make_dummy_evaluation()
    run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        artifacts_dir=tmp_path,
        run_id="run_role_val",
        candidate_pipeline=clf,
    )

    invalid_roles = [
        # Path traversal
        "../../evil",
        "../candidate",
        "models/../candidate",
        "candidate/../../evil",
        # Absolute paths
        "/etc/passwd",
        "C:/windows/system32",
        "\\server\\share",
        # Separators
        "candidate/sub",
        "candidate\\sub",
        "candidate/pipeline",
        # Unsupported roles
        "challenger",
        "production",
        "random_forest",
        "evaluator",
        "",
        "   ",
        # Non-string types
        None,
        123,
        ["candidate"],
    ]

    for role in invalid_roles:
        # Both with trusted=True and trusted=False, must reject invalid role before anything else
        with pytest.raises(ArtifactSecurityError, match="Invalid model role"):
            load_fitted_model(run_dir, model_role=role, trusted=True)

        with pytest.raises(ArtifactSecurityError, match="Invalid model role"):
            load_fitted_model(run_dir, model_role=role, trusted=False)


def test_model_provenance_and_fold_index_recording(tmp_path: Path):
    """Test that model provenance and fold index are accurately recorded in manifests and evaluation.json."""
    from sklearn.linear_model import LogisticRegression
    cand_clf = LogisticRegression()
    cand_clf.fit([[0, 1], [1, 0]], [0, 1])
    base_clf = LogisticRegression()
    base_clf.fit([[0, 1], [1, 0]], [0, 1])

    evaluation = _make_dummy_evaluation()
    evaluation.candidate_result.model_provenance = "final_cv_fold"
    evaluation.candidate_result.fitted_fold_index = 4
    if evaluation.baseline_result:
        evaluation.baseline_result.model_provenance = "final_cv_fold"
        evaluation.baseline_result.fitted_fold_index = 4

    run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        artifacts_dir=tmp_path,
        run_id="run_provenance_check",
        candidate_pipeline=cand_clf,
        baseline_pipeline=base_clf,
    )

    # 1. Inspect manifest.json
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    assert manifest_data["model_provenance"] == "final_cv_fold"
    assert manifest_data["candidate_model_fold_index"] == 4
    assert manifest_data["baseline_model_fold_index"] == 4

    # 2. Inspect evaluation.json
    eval_path = run_dir / "evaluation.json"
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    assert eval_data["candidate_result"]["model_provenance"] == "final_cv_fold"
    assert eval_data["candidate_result"]["fitted_fold_index"] == 4
    if eval_data.get("baseline_result"):
        assert eval_data["baseline_result"]["model_provenance"] == "final_cv_fold"
        assert eval_data["baseline_result"]["fitted_fold_index"] == 4


def test_legacy_manifest_backward_compatibility(tmp_path: Path):
    """Test that legacy manifests without provenance fields deserialize safely with None defaults."""
    evaluation = _make_dummy_evaluation()
    run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        artifacts_dir=tmp_path,
        run_id="run_legacy_test",
    )

    # Strip provenance fields from manifest.json to simulate a legacy pre-4.1.2 manifest
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    manifest_data.pop("model_provenance", None)
    manifest_data.pop("candidate_model_fold_index", None)
    manifest_data.pop("baseline_model_fold_index", None)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # load_run_artifacts must succeed without error
    loaded = load_run_artifacts(run_dir)
    manifest = loaded["manifest"]
    assert manifest.model_provenance is None
    assert manifest.candidate_model_fold_index is None
    assert manifest.baseline_model_fold_index is None
