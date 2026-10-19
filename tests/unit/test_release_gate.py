"""Unit Tests for Automated Release-Gate Engine (Milestone 4.3).

Verifies governance rules, evidence completeness, acceptance compliance, artifact integrity,
provenance consistency, decision precedence, and path safety.
"""

import copy
import json
from pathlib import Path
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlforge.application.artifacts import serialize_run_artifacts
from mlforge.application.packaging import build_audit_bundle
from mlforge.core.contracts import (
    AcceptanceDecision,
    AcceptanceOutcome,
    AcceptancePolicy,
    AcceptanceRule,
    CriterionEvaluation,
    CriterionOutcome,
    CriterionSeverity,
    EvaluationResult,
    FoldResult,
    GateCheckStatus,
    MetricValue,
    ReleaseGateResult,
    ReleaseGateStatus,
    ReleasePolicy,
    RuleOperator,
)
from mlforge.core.exceptions import (
    ArtifactSecurityError,
    ReleaseGateError,
)
from mlforge.engine.release_gate import evaluate_release_gate
from tests.unit.test_acceptance import _make_dummy_evaluation


@pytest.fixture
def passing_run_dir(tmp_path: Path) -> Path:
    """Create a fully compliant run with passing acceptance decision and serialized model."""
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=42))])
    pipe.fit([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], [0, 0, 1, 1])

    evaluation = _make_dummy_evaluation(cand_f1=0.88, cand_acc=0.90, base_f1=0.72, base_acc=0.75)
    evaluation.candidate_result.fitted_pipeline = pipe
    evaluation.candidate_result.model_provenance = "final_cv_fold"
    evaluation.candidate_result.fitted_fold_index = 4
    evaluation.candidate_result.fold_results = [
        FoldResult(
            fold_index=i,
            train_indices_count=80,
            val_indices_count=20,
            metrics={"macro_f1": MetricValue(value=0.88)},
            confusion_matrix=[[10, 0], [0, 10]],
            class_labels=["0", "1"],
            fit_duration_seconds=0.01,
            predict_duration_seconds=0.001,
        )
        for i in range(5)
    ]

    policy = AcceptancePolicy(
        policy_id="prod_release_policy",
        version="1.0.0",
        name="Production Release Policy",
        description="Standard acceptance policy",
        rules=[
            AcceptanceRule(
                rule_id="min_f1",
                metric="macro_f1",
                operator=RuleOperator.GREATER_THAN_OR_EQUAL,
                threshold=0.80,
                severity=CriterionSeverity.BLOCKER,
            )
        ],
    )

    crit = CriterionEvaluation(
        rule_id="min_f1",
        metric="macro_f1",
        severity=CriterionSeverity.BLOCKER,
        operator=RuleOperator.GREATER_THAN_OR_EQUAL,
        threshold_value=0.80,
        observed_value=0.88,
        outcome=CriterionOutcome.SATISFIED,
        reason_code="RULE_SATISFIED",
        message="Candidate F1 0.88 >= 0.80",
        evidence_source="candidate",
    )

    decision = AcceptanceDecision(
        decision_id="dec_passing_01",
        evaluation_id="eval_01",
        policy_id=policy.policy_id,
        policy_version=policy.version,
        outcome=AcceptanceOutcome.PASS,
        criteria_evaluations=[crit],
        summary_rationale="All blocker criteria satisfied.",
        evaluated_at="2026-10-19T10:00:00Z",
        passed_rules_count=1,
        failed_rules_count=0,
        inconclusive_rules_count=0,
        hash_signature="sig_passing_01",
    )

    run_dir = serialize_run_artifacts(
        evaluation=evaluation,
        acceptance_decision=decision,
        policy=policy,
        artifacts_dir=tmp_path / "runs",
        run_id="run_passing_release",
    )
    return run_dir


def test_release_gate_complete_valid_evidence_passes(passing_run_dir: Path):
    """Passing evaluation satisfies all release gates and receives overall status PASSED."""
    policy = ReleasePolicy(
        require_acceptance_pass=True,
        require_model_artifact=True,
        require_manifest_integrity=True,
    )
    result = evaluate_release_gate(passing_run_dir, policy)

    assert result.overall_status == ReleaseGateStatus.PASSED
    assert result.is_releasable is True
    assert result.run_id == "run_passing_release"
    assert len(result.gate_results) >= 4
    for gate in result.gate_results:
        assert gate.status == GateCheckStatus.PASSED


def test_release_gate_missing_required_evaluation_artifact(tmp_path: Path):
    """Empty or non-existent run directory raises ReleaseGateError."""
    empty_dir = tmp_path / "empty_run"
    empty_dir.mkdir()

    with pytest.raises(ReleaseGateError, match="missing in directory|does not exist|Failed to load"):
        evaluate_release_gate(empty_dir)


def test_release_gate_missing_optional_diagnostics_not_blocking_by_default(passing_run_dir: Path):
    """Missing optional diagnostics do not block release when policy does not mandate them."""
    # Default policy does not require robustness or calibration
    policy = ReleasePolicy(
        require_robustness_evidence=False,
        require_calibration_evidence=False,
    )
    result = evaluate_release_gate(passing_run_dir, policy)
    assert result.overall_status == ReleaseGateStatus.PASSED
    assert result.is_releasable is True


def test_release_gate_missing_evidence_mandatory_under_policy_blocks(passing_run_dir: Path):
    """When policy explicitly mandates robustness evidence, its absence causes BLOCKED."""
    policy = ReleasePolicy(require_robustness_evidence=True)
    result = evaluate_release_gate(passing_run_dir, policy)

    assert result.overall_status == ReleaseGateStatus.BLOCKED
    assert result.is_releasable is False
    comp_gate = next(g for g in result.gate_results if g.gate_id == "evidence_completeness")
    assert comp_gate.status == GateCheckStatus.FAILED
    assert comp_gate.reason_code == "EVIDENCE_INCOMPLETE"


def test_release_gate_failed_acceptance_blocker_yields_blocked(passing_run_dir: Path):
    """An acceptance decision of FAIL conclusively yields ReleaseGateStatus.BLOCKED."""
    dec_path = passing_run_dir / "decision.json"
    with open(dec_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["outcome"] = "fail"
    data["failed_rules_count"] = 1
    data["summary_rationale"] = "Blocker criterion violated."
    with open(dec_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    # Re-evaluate with require_manifest_integrity=False so we isolate acceptance gate
    policy = ReleasePolicy(require_acceptance_pass=True, require_manifest_integrity=False)
    result = evaluate_release_gate(passing_run_dir, policy)

    assert result.overall_status == ReleaseGateStatus.BLOCKED
    assert result.is_releasable is False
    acc_gate = next(g for g in result.gate_results if g.gate_id == "acceptance_compliance")
    assert acc_gate.status == GateCheckStatus.FAILED
    assert acc_gate.reason_code == "ACCEPTANCE_FAILED"


def test_release_gate_inconclusive_acceptance_yields_inconclusive(passing_run_dir: Path):
    """An acceptance decision of INCONCLUSIVE yields ReleaseGateStatus.INCONCLUSIVE."""
    dec_path = passing_run_dir / "decision.json"
    with open(dec_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["outcome"] = "inconclusive"
    data["inconclusive_rules_count"] = 1
    with open(dec_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    policy = ReleasePolicy(require_acceptance_pass=True, require_manifest_integrity=False)
    result = evaluate_release_gate(passing_run_dir, policy)

    assert result.overall_status == ReleaseGateStatus.INCONCLUSIVE
    assert result.is_releasable is False
    acc_gate = next(g for g in result.gate_results if g.gate_id == "acceptance_compliance")
    assert acc_gate.status == GateCheckStatus.INCONCLUSIVE
    assert acc_gate.reason_code == "ACCEPTANCE_INCONCLUSIVE"


def test_release_gate_unsupported_schema_version_blocks(passing_run_dir: Path):
    """Unsupported major schema version blocks release."""
    man_path = passing_run_dir / "manifest.json"
    with open(man_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["schema_version"] = "99.0.0"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    result = evaluate_release_gate(passing_run_dir, ReleasePolicy(require_manifest_integrity=False))
    assert result.overall_status == ReleaseGateStatus.BLOCKED
    assert result.is_releasable is False
    schema_gate = next(g for g in result.gate_results if g.gate_id == "schema_compatibility")
    assert schema_gate.status == GateCheckStatus.FAILED
    assert schema_gate.reason_code == "UNSUPPORTED_SCHEMA_VERSION"


def test_release_gate_manifest_checksum_tampering_blocks(passing_run_dir: Path):
    """Tampering with an evaluation file invalidates manifest integrity check and blocks release."""
    eval_file = passing_run_dir / "evaluation.json"
    content = eval_file.read_text(encoding="utf-8")
    eval_file.write_text(content + " ", encoding="utf-8")  # Append whitespace to alter SHA-256

    policy = ReleasePolicy(require_manifest_integrity=True)
    result = evaluate_release_gate(passing_run_dir, policy)

    assert result.overall_status == ReleaseGateStatus.BLOCKED
    assert result.is_releasable is False
    integ_gate = next(g for g in result.gate_results if g.gate_id == "artifact_integrity")
    assert integ_gate.status == GateCheckStatus.FAILED
    assert integ_gate.reason_code == "ARTIFACT_INTEGRITY_TAMPERED_OR_MISSING"


def test_release_gate_missing_required_model_artifact_blocks(passing_run_dir: Path):
    """When require_model_artifact=True and candidate model file is missing, release is BLOCKED."""
    cand_model = passing_run_dir / "models" / "candidate_pipeline.joblib"
    if cand_model.exists():
        cand_model.unlink()

    policy = ReleasePolicy(require_model_artifact=True, require_manifest_integrity=False)
    result = evaluate_release_gate(passing_run_dir, policy)

    assert result.overall_status == ReleaseGateStatus.BLOCKED
    assert result.is_releasable is False
    integ_gate = next(g for g in result.gate_results if g.gate_id == "artifact_integrity")
    assert integ_gate.status == GateCheckStatus.FAILED


def test_release_gate_model_provenance_fold_index_out_of_bounds_blocks(passing_run_dir: Path):
    """Inconsistent model provenance fitted_fold_index blocks release."""
    eval_path = passing_run_dir / "evaluation.json"
    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 5 folds evaluated (indices 0..4); set fold index to 99
    data["candidate_result"]["fitted_fold_index"] = 99
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    policy = ReleasePolicy(require_manifest_integrity=False)
    result = evaluate_release_gate(passing_run_dir, policy)

    assert result.overall_status == ReleaseGateStatus.BLOCKED
    assert result.is_releasable is False
    prov_gate = next(g for g in result.gate_results if g.gate_id == "provenance_consistency")
    assert prov_gate.status == GateCheckStatus.FAILED
    assert prov_gate.reason_code == "PROVENANCE_INCONSISTENCY"


def test_release_gate_dataset_checksum_mismatch_blocks(passing_run_dir: Path):
    """Mismatch between manifest dataset_checksum and evaluation dataset checksum blocks release."""
    man_path = passing_run_dir / "manifest.json"
    with open(man_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["dataset_checksum"] = "bad_checksum_1234567890abcdef"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    policy = ReleasePolicy(require_manifest_integrity=False)
    result = evaluate_release_gate(passing_run_dir, policy)

    assert result.overall_status == ReleaseGateStatus.BLOCKED
    assert result.is_releasable is False
    prov_gate = next(g for g in result.gate_results if g.gate_id == "provenance_consistency")
    assert prov_gate.status == GateCheckStatus.FAILED


def test_release_gate_advisory_warnings_forbidden_policy_blocks(passing_run_dir: Path):
    """When allow_advisory_warnings=False, acceptance decisions with advisory warnings block release."""
    dec_path = passing_run_dir / "decision.json"
    with open(dec_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["advisory_warnings"] = ["High calibration error observed in fold 2"]
    with open(dec_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    policy = ReleasePolicy(
        require_acceptance_pass=True,
        allow_advisory_warnings=False,
        require_manifest_integrity=False,
    )
    result = evaluate_release_gate(passing_run_dir, policy)

    assert result.overall_status == ReleaseGateStatus.BLOCKED
    assert result.is_releasable is False
    acc_gate = next(g for g in result.gate_results if g.gate_id == "acceptance_compliance")
    assert acc_gate.status == GateCheckStatus.FAILED
    assert acc_gate.reason_code == "ADVISORY_WARNINGS_FORBIDDEN"


def test_release_gate_precedence_blocker_overrides_inconclusive(passing_run_dir: Path):
    """Conclusive blocker failure overrides inconclusive checks (BLOCKED > INCONCLUSIVE)."""
    dec_path = passing_run_dir / "decision.json"
    with open(dec_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["outcome"] = "fail"
    with open(dec_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    # Policy also requires calibration which is missing (would be INCONCLUSIVE)
    policy = ReleasePolicy(
        require_acceptance_pass=True,
        require_calibration_evidence=True,
        require_manifest_integrity=False,
    )
    result = evaluate_release_gate(passing_run_dir, policy)

    assert result.overall_status == ReleaseGateStatus.BLOCKED
    assert result.is_releasable is False


def test_release_gate_audit_bundle_source(passing_run_dir: Path, tmp_path: Path):
    """Release gate evaluates directly from a verified .zip audit bundle."""
    bundle_path = tmp_path / "passing_bundle.zip"
    build_audit_bundle(passing_run_dir, bundle_path)

    policy = ReleasePolicy(require_acceptance_pass=True)
    result = evaluate_release_gate(bundle_path, policy)

    assert result.overall_status == ReleaseGateStatus.PASSED
    assert result.is_releasable is True


def test_release_gate_deterministic_repeated_evaluation(passing_run_dir: Path):
    """Repeated evaluation on identical saved evidence produces identical gate verdicts."""
    policy = ReleasePolicy(require_acceptance_pass=True)
    res1 = evaluate_release_gate(passing_run_dir, policy)
    res2 = evaluate_release_gate(passing_run_dir, policy)

    assert res1.overall_status == res2.overall_status
    assert res1.is_releasable == res2.is_releasable
    assert len(res1.gate_results) == len(res2.gate_results)
    for g1, g2 in zip(res1.gate_results, res2.gate_results):
        assert g1.gate_id == g2.gate_id
        assert g1.status == g2.status
        assert g1.reason_code == g2.reason_code
