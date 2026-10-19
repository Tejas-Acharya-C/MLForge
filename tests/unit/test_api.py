"""Unit tests for the MLForge REST API Layer."""

import json
from pathlib import Path
import tempfile
from typing import Generator

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from mlforge.api.app import create_app
from mlforge.api.routes import get_repository
from mlforge.application.persistence import SQLiteRunRepository


@pytest.fixture
def temp_repo(tmp_path: Path) -> SQLiteRunRepository:
    """Provide an isolated SQLiteRunRepository for API unit tests."""
    db_file = tmp_path / "test_api_mlforge.db"
    repo = SQLiteRunRepository(db_path=db_file)
    repo.initialize_schema()
    return repo


@pytest.fixture
def client(temp_repo: SQLiteRunRepository) -> Generator[TestClient, None, None]:
    """FastAPI TestClient with overridden database repository dependency."""
    app = create_app()
    app.dependency_overrides[get_repository] = lambda: temp_repo
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# ============================================================================
# 1. Health & Datasets & Policies
# ============================================================================

def test_api_root_health(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["application"] == "MLForge"
    assert data["status"] == "ready"


def test_api_sample_datasets(client: TestClient) -> None:
    resp = client.get("/api/v1/datasets/sample")
    assert resp.status_code == 200
    samples = resp.json()
    assert len(samples) >= 2
    ids = [s["id"] for s in samples]
    assert "churn" in ids
    assert "breast_cancer" in ids


def test_api_inspect_sample_churn(client: TestClient) -> None:
    payload = {
        "sample_name": "churn",
        "target_column": "churn_label",
        "k_folds": 5,
    }
    resp = client.post("/api/v1/datasets/inspect", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["num_rows"] == 500
    assert data["target_name"] == "churn_label"
    assert "0" in data["class_distribution"] or 0 in data["class_distribution"]
    assert len(data["feature_names"]) >= 5


def test_api_inspect_invalid_target(client: TestClient) -> None:
    payload = {
        "sample_name": "churn",
        "target_column": "non_existent_target",
    }
    resp = client.post("/api/v1/datasets/inspect", json=payload)
    assert resp.status_code == 422
    assert "not found in dataset" in resp.json()["detail"]


def test_api_list_policies(client: TestClient) -> None:
    resp = client.get("/api/v1/acceptance-policies")
    assert resp.status_code == 200
    policies = resp.json()
    p_ids = [p["policy_id"] for p in policies]
    assert "standard_classification_v1" in p_ids
    assert "strict_zero_leakage_v1" in p_ids


# ============================================================================
# 2. Evaluation Lifecycle & Results
# ============================================================================

def test_api_submit_evaluation_and_results(client: TestClient) -> None:
    # 1. Submit evaluation using sample preset
    submit_req = {
        "sample_name": "churn",
        "target_column": "churn_label",
        "num_folds": 3,
        "random_state": 42,
        "policy_id": "standard_classification_v1",
        "subgroup_columns": ["region"],
    }
    resp = client.post("/api/v1/evaluations", json=submit_req)
    assert resp.status_code == 201
    run_summary = resp.json()
    run_id = run_summary["run_id"]
    assert run_summary["state"] == "completed"
    assert "accuracy" in run_summary["summary_metrics"]
    assert "macro_f1" in run_summary["summary_metrics"]

    # 2. List evaluations
    list_resp = client.get("/api/v1/evaluations")
    assert list_resp.status_code == 200
    runs = list_resp.json()
    assert runs["total"] >= 1
    assert any(item["run_id"] == run_id for item in runs["items"])

    # 3. Get evaluation detail
    detail_resp = client.get(f"/api/v1/evaluations/{run_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["record"]["run_id"] == run_id
    assert detail["has_evaluation"] is True
    assert detail["has_decision"] is True
    assert len(detail["lifecycle_history"]) >= 3

    # 4. Get evaluation status
    status_resp = client.get(f"/api/v1/evaluations/{run_id}/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["state"] == "completed"

    # 5. Get results
    results_resp = client.get(f"/api/v1/evaluations/{run_id}/results")
    assert results_resp.status_code == 200
    res = results_resp.json()
    assert "candidate_result" in res
    assert "baseline_result" in res
    assert "comparison_report" in res

    # 6. Get fold results
    folds_resp = client.get(f"/api/v1/evaluations/{run_id}/results/folds")
    assert folds_resp.status_code == 200
    folds = folds_resp.json()
    assert len(folds) == 3

    # 7. Get calibration diagnostics
    calib_resp = client.get(f"/api/v1/evaluations/{run_id}/diagnostics/calibration")
    assert calib_resp.status_code == 200
    calib = calib_resp.json()
    assert "bins" in calib
    assert "expected_calibration_error" in calib

    # 8. Get subgroup diagnostics
    sub_resp = client.get(f"/api/v1/evaluations/{run_id}/diagnostics/subgroups")
    assert sub_resp.status_code == 200
    sub = sub_resp.json()
    assert "subgroup_diagnostics" in sub
    assert "region" in sub["subgroup_diagnostics"]

    # 9. Get acceptance decision
    dec_resp = client.get(f"/api/v1/evaluations/{run_id}/acceptance")
    assert dec_resp.status_code == 200
    dec = dec_resp.json()
    assert dec["outcome"] in ("PASS", "FAIL", "INCONCLUSIVE")
    assert len(dec["criteria_evaluations"]) >= 2
    assert dec["hash_signature"] is not None

    # 10. Get offline report HTML
    rep_resp = client.get(f"/api/v1/evaluations/{run_id}/report.html")
    assert rep_resp.status_code == 200
    assert "<!DOCTYPE html>" in rep_resp.text
    assert "MLForge" in rep_resp.text

    # 11. Get artifacts listing
    art_resp = client.get(f"/api/v1/evaluations/{run_id}/artifacts")
    assert art_resp.status_code == 200

    # 12. Run robustness stress test
    rob_resp = client.post(
        f"/api/v1/evaluations/{run_id}/robustness",
        json={"profile": "noise", "severity_levels": [0.05, 0.10, 0.25]},
    )
    assert rob_resp.status_code == 200
    rob = rob_resp.json()
    assert len(rob["curves"]) == 1
    assert rob["curves"][0]["perturbation_type"] == "gaussian_noise"

    # 13. Evaluate release gate
    gate_resp = client.post(
        f"/api/v1/evaluations/{run_id}/release-gate",
        json={"policy": {"target_environment": "staging", "require_acceptance_pass": False}},
    )
    assert gate_resp.status_code == 200
    gate = gate_resp.json()
    assert gate["overall_status"] in ("passed", "blocked", "inconclusive")

    # 14. Issue Ed25519 release certificate
    cert_resp = client.post(
        f"/api/v1/evaluations/{run_id}/certificate/issue",
        json={
            "key_id": "test-key-01",
            "target_environment": "staging",
            "release_policy": {"target_environment": "staging", "require_acceptance_pass": False},
        },
    )
    assert cert_resp.status_code == 200
    cert_data = cert_resp.json()
    cert = cert_data["certificate"]
    pub_hex = cert_data["public_key_hex"]
    assert cert["key_id"] == "test-key-01"
    assert cert["signature"] is not None

    # 15. Verify certificate
    verify_resp = client.post(
        "/api/v1/verify-release",
        json={
            "certificate": cert,
            "trusted_keys": {"test-key-01": pub_hex},
            "require_passed_gate": False,
        },
    )
    assert verify_resp.status_code == 200
    ver = verify_resp.json()
    assert ver["payload_integrity"] is True
    assert ver["signature_validity"] is True
    assert ver["signer_trust"] is True

    # 16. Package audit bundle
    bundle_resp = client.post(f"/api/v1/evaluations/{run_id}/bundle", json={})
    assert bundle_resp.status_code == 200
    bundle_info = bundle_resp.json()
    bundle_path = bundle_info["bundle_path"]
    assert Path(bundle_path).exists()

    # 17. Verify audit bundle
    ver_bundle_resp = client.post(
        "/api/v1/verify-bundle",
        json={"bundle_path": bundle_path},
    )
    assert ver_bundle_resp.status_code == 200
    ver_b = ver_bundle_resp.json()
    assert ver_b["is_valid"] is True
    assert ver_b["run_id"] == run_id

    # 18. Download bundle
    dl_resp = client.get(f"/api/v1/evaluations/{run_id}/bundle.zip")
    assert dl_resp.status_code == 200
    assert dl_resp.headers["content-type"] == "application/zip"


def test_api_run_not_found(client: TestClient) -> None:
    resp = client.get("/api/v1/evaluations/non_existent_run_id")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_api_invalid_state_filter(client: TestClient) -> None:
    resp = client.get("/api/v1/evaluations?state=invalid_status")
    assert resp.status_code == 400
    assert "Invalid state filter" in resp.json()["detail"]


def test_api_cancel_completed_run_conflict(client: TestClient) -> None:
    # 1. Create completed run
    submit_req = {
        "sample_name": "churn",
        "target_column": "churn_label",
        "num_folds": 2,
        "random_state": 42,
    }
    resp = client.post("/api/v1/evaluations", json=submit_req)
    assert resp.status_code == 201
    run_id = resp.json()["run_id"]

    # 2. Attempt cancel on terminal run -> 409 Conflict
    cancel_resp = client.post(f"/api/v1/evaluations/{run_id}/cancel", json={"reason": "test abort"})
    assert cancel_resp.status_code == 409
    assert "Terminal run" in cancel_resp.json()["detail"]


def test_api_artifact_path_traversal_forbidden(client: TestClient) -> None:
    submit_req = {
        "sample_name": "churn",
        "target_column": "churn_label",
        "num_folds": 2,
    }
    resp = client.post("/api/v1/evaluations", json=submit_req)
    run_id = resp.json()["run_id"]

    resp_trav = client.get(f"/api/v1/evaluations/{run_id}/artifacts/sub%2F..%2Fsecret.txt")
    assert resp_trav.status_code in (400, 404)
