"""Unit Tests for Offline HTML Audit Report Generation (Milestone 4.2).

Verifies standalone offline HTML report generation, complete evidence rendering,
defensive handling of missing sections, XSS escaping, offline asset containment,
atomic writes, and zero model deserialization.
"""

import hashlib
import json
from pathlib import Path
import re
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlforge.application.artifacts import serialize_run_artifacts
from mlforge.application.packaging import build_audit_bundle
from mlforge.application.reporting import (
    REPORT_SCHEMA_VERSION,
    generate_html_report,
    load_report_evidence,
)
from mlforge.core.contracts import (
    AcceptanceDecision,
    AcceptanceOutcome,
    AcceptancePolicy,
    AcceptanceRule,
    CalibrationBin,
    CalibrationCurveResult,
    CriterionEvaluation,
    CriterionOutcome,
    CriterionSeverity,
    DatasetMetadata,
    ErrorDiagnosticsResult,
    EvaluationResult,
    FoldErrorDiagnostic,
    FoldResult,
    MetricDirection,
    MetricSummary,
    MetricValue,
    ModelEvaluationResult,
    PerturbationType,
    ReproducibilityMetadata,
    RobustnessCurve,
    RobustnessEvaluationResult,
    RobustnessPerturbationResult,
    RuleOperator,
    SubgroupMetricDiagnostic,
    ThresholdAnalysisResult,
    ThresholdMetrics,
)
from mlforge.core.exceptions import (
    ArtifactLoadError,
    ArtifactSerializationError,
)
from tests.unit.test_acceptance import _make_dummy_evaluation


@pytest.fixture
def sample_full_run_dir(tmp_path: Path) -> Path:
    """Create a fully-populated run directory with candidate, baseline, calibration, diagnostics, and robustness."""
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=42))])
    pipe.fit([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], [0, 0, 1, 1])

    evaluation = _make_dummy_evaluation(cand_f1=0.86, cand_acc=0.89, base_f1=0.72, base_acc=0.75, ece=0.04)
    evaluation.candidate_result.model_name = "Candidate: LogisticRegression"
    evaluation.candidate_result.fitted_pipeline = pipe
    evaluation.candidate_result.model_provenance = "final_cv_fold"
    evaluation.candidate_result.fitted_fold_index = 4

    # Add calibration bins
    bins = [
        CalibrationBin(
            bin_index=0,
            lower_bound=0.0,
            upper_bound=0.5,
            mean_predicted_prob=0.25,
            observed_positive_rate=0.20,
            sample_count=50,
            is_empty=False,
        ),
        CalibrationBin(
            bin_index=1,
            lower_bound=0.5,
            upper_bound=1.0,
            mean_predicted_prob=0.75,
            observed_positive_rate=0.80,
            sample_count=50,
            is_empty=False,
        ),
    ]
    evaluation.candidate_result.calibration = CalibrationCurveResult(
        bins=bins,
        expected_calibration_error=0.04,
        max_calibration_error=0.05,
        brier_score=0.12,
        log_loss=0.35,
        bin_strategy="uniform",
        num_bins=2,
        total_samples=100,
        positive_class_label="1",
        is_usable=True,
    )

    # Add threshold analysis
    evaluation.candidate_result.threshold_analysis = ThresholdAnalysisResult(
        positive_class_label="1",
        evaluated_samples=100,
        threshold_metrics=[
            ThresholdMetrics(
                threshold=0.3,
                evaluated_samples=100,
                true_positives=38,
                false_positives=10,
                true_negatives=50,
                false_negatives=2,
                precision=0.79,
                recall=0.95,
                specificity=0.83,
                f1_score=0.86,
                balanced_accuracy=0.89,
                predicted_positive_rate=0.48,
            ),
            ThresholdMetrics(
                threshold=0.5,
                evaluated_samples=100,
                true_positives=35,
                false_positives=5,
                true_negatives=55,
                false_negatives=5,
                precision=0.88,
                recall=0.88,
                specificity=0.92,
                f1_score=0.88,
                balanced_accuracy=0.90,
                predicted_positive_rate=0.40,
            ),
        ],
    )

    # Add error diagnostics
    evaluation.candidate_result.error_diagnostics = ErrorDiagnosticsResult(
        total_samples=100,
        total_errors=10,
        overall_error_rate=0.10,
        fold_diagnostics=[
            FoldErrorDiagnostic(
                fold_index=0,
                total_samples=20,
                error_count=2,
                error_rate=0.10,
                false_positives=1,
                false_negatives=1,
            )
        ],
        subgroup_diagnostics={
            "region": [
                SubgroupMetricDiagnostic(
                    subgroup_column="region",
                    subgroup_value="North",
                    sample_count=50,
                    error_count=4,
                    error_rate=0.08,
                    accuracy=0.92,
                    precision=0.90,
                    recall=0.93,
                    is_usable=True,
                )
            ]
        },
    )

    # Add robustness curves
    evaluation.candidate_result.robustness = RobustnessEvaluationResult(
        curves=[
            RobustnessCurve(
                perturbation_type=PerturbationType.GAUSSIAN_NOISE,
                metric_name="accuracy",
                clean_score=0.89,
                points=[
                    RobustnessPerturbationResult(
                        perturbation_type=PerturbationType.GAUSSIAN_NOISE,
                        severity_level=0.05,
                        metric_name="accuracy",
                        clean_score=0.89,
                        perturbed_score=0.87,
                        retained_ratio=0.9775,
                        performance_drop_percentage=2.25,
                        is_usable=True,
                    ),
                    RobustnessPerturbationResult(
                        perturbation_type=PerturbationType.GAUSSIAN_NOISE,
                        severity_level=0.10,
                        metric_name="accuracy",
                        clean_score=0.89,
                        perturbed_score=0.84,
                        retained_ratio=0.9438,
                        performance_drop_percentage=5.62,
                        is_usable=True,
                    ),
                ],
                area_under_degradation_curve=0.96,
            )
        ],
        summary_retention={"gaussian_noise": 0.9438},
        is_usable=True,
    )

    policy = AcceptancePolicy(
        policy_id="prod_release_v1",
        version="1.0.0",
        name="Production Model Release Gate",
        rules=[
            AcceptanceRule(
                rule_id="r_f1",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.80,
                severity=CriterionSeverity.BLOCKER,
            )
        ],
    )

    decision = AcceptanceDecision(
        decision_id="dec_report_01",
        evaluation_id="eval_report_01",
        policy_id=policy.policy_id,
        policy_version=policy.version,
        outcome=AcceptanceOutcome.PASS,
        criteria_evaluations=[
            CriterionEvaluation(
                rule_id="r_f1",
                metric="macro_f1",
                severity=CriterionSeverity.BLOCKER,
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold_value=0.80,
                observed_value=0.86,
                outcome=CriterionOutcome.SATISFIED,
                reason_code="THRESHOLD_MET",
                message="Observed macro F1 (0.86) satisfies requirement (>= 0.80).",
                evidence_source="candidate_metric",
            )
        ],
        summary_rationale="All criteria satisfied for production deployment.",
        evaluated_at="2026-10-19T10:00:00Z",
        passed_rules_count=1,
        failed_rules_count=0,
        inconclusive_rules_count=0,
        hash_signature="abcd1234ef5678",
    )

    run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        acceptance_decision=decision,
        policy=policy,
        artifacts_dir=tmp_path / "runs",
        run_id="run_full_report",
    )
    return run_dir


def test_generate_html_report_minimal_run(tmp_path: Path):
    """Minimal evaluation without baseline or policy generates clean report with explicit missing sections."""
    evaluation = _make_dummy_evaluation(comparison_available=False)
    evaluation.warnings = ["Single fold test run."]

    run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        artifacts_dir=tmp_path / "runs",
        run_id="run_minimal",
    )

    report_path = generate_html_report(run_dir)
    assert report_path.exists()
    assert report_path.name == "report.html"

    html_text = report_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_text
    assert "MLForge" in html_text
    assert "run_minimal" in html_text
    assert "No Acceptance Decision Evaluated" in html_text
    assert "Single fold test run." in html_text
    assert "Single-model runs do not generate comparative significance metrics." in html_text


def test_generate_html_report_full_evidence(sample_full_run_dir: Path):
    """Full evaluation run renders all 10 required sections, tables, and vector charts."""
    report_path = generate_html_report(sample_full_run_dir)
    assert report_path.exists()

    html_text = report_path.read_text(encoding="utf-8")

    # 1. Section anchors exist
    expected_anchors = [
        "id=\"executive-summary\"",
        "id=\"performance\"",
        "id=\"comparison\"",
        "id=\"calibration\"",
        "id=\"diagnostics\"",
        "id=\"robustness\"",
        "id=\"acceptance\"",
        "id=\"provenance\"",
        "id=\"methodology\"",
    ]
    for anchor in expected_anchors:
        assert anchor in html_text, f"Missing section anchor: {anchor}"

    # 2. Executive summary details
    assert "Acceptance Decision: PASS" in html_text
    assert "prod_release_v1" in html_text
    assert "Candidate: LogisticRegression" in html_text

    # 3. Inline SVG charts exist
    assert "Probability Calibration Reliability Diagram" in html_text
    assert "Decision Threshold Trade-Off Curve" in html_text
    assert "Robustness Degradation Trajectory" in html_text

    # 4. Calibration & threshold data
    assert "ECE:" in html_text
    assert "0.0400" in html_text

    # 5. Robustness & AURDC
    assert "gaussian_noise" in html_text
    assert "0.9600" in html_text

    # 6. Provenance & Trust Tiers
    assert "Tier 1 (File Existence)" in html_text
    assert "Tier 4 (External Trust / Author Authenticity)" in html_text
    assert "final_cv_fold" in html_text


def test_generate_html_report_from_audit_bundle(sample_full_run_dir: Path, tmp_path: Path):
    """Report can be generated directly from a verified .zip audit bundle."""
    bundle_path = tmp_path / "test_bundle.zip"
    build_audit_bundle(sample_full_run_dir, output_bundle_path=bundle_path)

    report_path = generate_html_report(bundle_path)
    assert report_path.exists()
    assert report_path.suffix == ".html"

    html_text = report_path.read_text(encoding="utf-8")
    assert "Verified Audit Bundle (.zip)" in html_text
    assert "run_full_report" in html_text
    assert "Acceptance Decision: PASS" in html_text


def test_generate_html_report_never_deserializes_models(sample_full_run_dir: Path, monkeypatch):
    """Generating a report must NEVER deserialize joblib or pickle models."""
    import joblib
    import pickle

    def fail_load(*args, **kwargs):
        raise AssertionError("Security violation: Model deserialization function was invoked!")

    monkeypatch.setattr(joblib, "load", fail_load)
    monkeypatch.setattr(pickle, "load", fail_load)

    # Must succeed without invoking joblib.load or pickle.load
    report_path = generate_html_report(sample_full_run_dir)
    assert report_path.exists()


def test_generate_html_report_xss_hostile_strings_escaped(tmp_path: Path):
    """Hostile XSS injection strings in model names, dataset labels, and messages are escaped."""
    pipe = Pipeline([("clf", LogisticRegression())])
    pipe.fit([[1, 2], [3, 4]], [0, 1])

    evaluation = _make_dummy_evaluation()
    evaluation.candidate_result.fitted_pipeline = pipe
    evaluation.candidate_result.model_name = "<script>alert('pwned_model')</script>"
    evaluation.candidate_result.model_type = '"><img src=x onerror=alert(1)>'
    evaluation.warnings = ["<iframe src='javascript:alert(2)'></iframe>"]

    policy = AcceptancePolicy(
        policy_id="pol_<script>",
        version="1.0.0",
        name="<b onmouseover='alert(3)'>Policy</b>",
        rules=[],
    )

    decision = AcceptanceDecision(
        decision_id="dec_01",
        policy_id="pol_<script>",
        policy_version="1.0.0",
        outcome=AcceptanceOutcome.FAIL,
        criteria_evaluations=[
            CriterionEvaluation(
                rule_id="r1",
                metric="f1",
                severity=CriterionSeverity.BLOCKER,
                operator=RuleOperator.EQUALS,
                observed_value=0.5,
                outcome=CriterionOutcome.VIOLATED,
                reason_code="ERR_<svg/onload=alert(4)>",
                message="<a href='javascript:alert(5)'>Click</a>",
                evidence_source="candidate_metric",
            )
        ],
        summary_rationale="Failed: <script>alert(6)</script>",
        evaluated_at="2026-10-19T00:00:00Z",
        passed_rules_count=0,
        failed_rules_count=1,
        inconclusive_rules_count=0,
    )

    run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        acceptance_decision=decision,
        policy=policy,
        artifacts_dir=tmp_path / "runs",
        run_id="run_xss_test",
    )

    report_path = generate_html_report(run_dir)
    html_text = report_path.read_text(encoding="utf-8")

    # Hostile executable HTML must NOT appear raw in the report
    dangerous_patterns = [
        "<script>alert('pwned_model')</script>",
        "<img src=x onerror=alert(1)>",
        "<iframe src=",
        "onmouseover='alert(3)'",
        "<svg/onload=alert(4)>",
        "href='javascript:alert(5)'",
        "<script>alert(6)</script>",
    ]
    for pattern in dangerous_patterns:
        assert pattern not in html_text, f"XSS pattern leaked unescaped: {pattern}"

    # Escaped safe versions must be present
    assert "&lt;script&gt;alert(&#x27;pwned_model&#x27;)&lt;/script&gt;" in html_text
    assert "&lt;iframe src=&#x27;javascript:alert(2)&#x27;&gt;&lt;/iframe&gt;" in html_text


def test_generate_html_report_is_completely_offline(sample_full_run_dir: Path):
    """Generated HTML report must be 100% self-contained with no external resource links."""
    report_path = generate_html_report(sample_full_run_dir)
    html_text = report_path.read_text(encoding="utf-8")

    # Check for external network links in src or href
    assert "http://" not in html_text
    assert "https://" not in html_text
    assert "fonts.googleapis.com" not in html_text
    assert "cdn." not in html_text
    assert "<script src=" not in html_text
    assert "<link rel=\"stylesheet\"" not in html_text


def test_generate_html_report_undefined_metrics_handled_cleanly(tmp_path: Path):
    """Undefined and NaN metrics render as N/A or undefined without crashing."""
    evaluation = _make_dummy_evaluation(metric_undefined=True)

    run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        artifacts_dir=tmp_path / "runs",
        run_id="run_undefined_metric",
    )

    report_path = generate_html_report(run_dir)
    html_text = report_path.read_text(encoding="utf-8")
    assert 'class="val-na">N/A</span>' in html_text


def test_generate_html_report_legacy_manifest_without_provenance(sample_full_run_dir: Path):
    """Legacy manifests lacking newer provenance fields are reported without error."""
    manifest_path = sample_full_run_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        mdata = json.load(f)

    mdata.pop("model_provenance", None)
    mdata.pop("candidate_model_fold_index", None)
    mdata.pop("baseline_model_fold_index", None)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(mdata, f, indent=2)

    report_path = generate_html_report(sample_full_run_dir)
    assert report_path.exists()
    html_text = report_path.read_text(encoding="utf-8")
    assert "Model Estimator Provenance" in html_text


def test_generate_html_report_atomic_failure_cleanup(sample_full_run_dir: Path, tmp_path: Path, monkeypatch):
    """Failure during report write removes partial temporary file."""
    target_path = tmp_path / "failing_report.html"

    # Simulate write failure by monkeypatching open
    orig_open = open

    def failing_open(path, *args, **kwargs):
        if str(path).endswith(".tmp_") or "failing_report.html.tmp_" in str(path):
            raise IOError("Simulated disk full error")
        return orig_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", failing_open)

    with pytest.raises(ArtifactSerializationError, match="Failed to write HTML report"):
        generate_html_report(sample_full_run_dir, output_path=target_path)

    assert not target_path.exists()
    # Ensure no lingering temporary file
    lingering = list(tmp_path.glob("*.tmp_*"))
    assert len(lingering) == 0


def test_generate_html_report_overwrite_flag(sample_full_run_dir: Path, tmp_path: Path):
    """When overwrite=False and report exists, raises ArtifactSerializationError."""
    target_path = tmp_path / "custom_report.html"
    generate_html_report(sample_full_run_dir, output_path=target_path)
    assert target_path.exists()

    with pytest.raises(ArtifactSerializationError, match="already exists and overwrite is set to False"):
        generate_html_report(sample_full_run_dir, output_path=target_path, overwrite=False)


def test_generate_html_report_manifest_not_mutated(sample_full_run_dir: Path):
    """Report generation must not alter manifest.json or create circular checksum issues."""
    manifest_path = sample_full_run_dir / "manifest.json"
    orig_bytes = manifest_path.read_bytes()
    orig_sha256 = hashlib.sha256(orig_bytes).hexdigest()

    generate_html_report(sample_full_run_dir)

    after_bytes = manifest_path.read_bytes()
    after_sha256 = hashlib.sha256(after_bytes).hexdigest()

    assert orig_sha256 == after_sha256, "manifest.json was mutated by report generation!"


def test_generate_html_report_invalid_sources(tmp_path: Path):
    """Invalid or missing sources raise typed ArtifactLoadError."""
    # Missing directory
    with pytest.raises(ArtifactLoadError, match="does not exist"):
        generate_html_report(tmp_path / "non_existent_dir")

    # Directory without evaluation.json
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(ArtifactLoadError):
        generate_html_report(empty_dir)
