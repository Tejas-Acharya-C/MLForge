# MLForge — REST API Specification

**Document Version:** 2.0.0
**Base URL:** `http://127.0.0.1:8000/api/v1`
**Status:** Implemented & Active Specification

> [!NOTE]
> This document formalizes the active HTTP endpoints, payload schemas, query parameters, status transitions, and error behaviors for the MLForge FastAPI application layer (`mlforge/api/routes.py`). All documented endpoints are implemented, tested, and actively consumed by the frontend laboratory.

---

## 1. Run Lifecycle State Machine

Evaluation runs proceed through an explicit, finite state machine:

```
          ┌─────────────┐
          │   QUEUED    │
          └──────┬──────┘
                 │ (Worker picks up evaluation)
                 ▼
          ┌─────────────┐       (User abort or timeout)
          │   RUNNING   ├─────────────────────────────────┐
          └──────┬──────┘                                 │
                 │                                        │
        ┌────────┴────────┐                               ▼
        │ (Folds finish   │ (Uncaught error,      ┌──────────────┐
        │  & validation   │  leakage detected,    │  CANCELLED   │
        │  succeeds)      │  or crash)            └──────────────┘
        ▼                 ▼
 ┌─────────────┐   ┌─────────────┐
 │  COMPLETED  │   │   FAILED    │
 └─────────────┘   └─────────────┘
```

### Valid State Transitions
1. `queued` $\to$ `running`
2. `queued` $\to$ `cancelled`
3. `running` $\to$ `completed`
4. `running` $\to$ `failed`
5. `running` $\to$ `cancelled`
6. `completed`, `failed`, `cancelled` are **terminal states** and cannot transition further.

### Completion Integrity Criteria
An evaluation run may transition to `completed` **only if**:
- Every configured fold ($k = 1 \dots K$) executed its fit and validation passes without fatal execution errors.
- Both candidate and baseline models completed execution on identical folds.
- Fold-level metric dictionaries and confusion matrices have been computed and validated against their contracts.
- Aggregated summaries (mean, std, 95% CI) have been generated.
- Acceptance policy evaluation has executed and emitted a signed `AcceptanceDecision`.
- Raw artifacts have been persisted to the local filesystem and registered in the `ArtifactManifest`.

---

## 2. API Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/datasets/inspect` | Inspect local tabular file, return column types and class counts |
| `POST` | `/evaluations` | Submit and queue a new evaluation run |
| `GET` | `/evaluations` | List recent evaluations with pagination and status filters |
| `GET` | `/evaluations/{id}` | Retrieve comprehensive evaluation details and configuration |
| `GET` | `/evaluations/{id}/status` | Lightweight endpoint for polling execution state and progress |
| `POST` | `/evaluations/{id}/cancel` | Abort a running or queued evaluation |
| `GET` | `/evaluations/{id}/results` | Retrieve model comparisons and aggregate metric summaries |
| `GET` | `/evaluations/{id}/results/folds` | Retrieve fold-by-fold metrics and distributions |
| `GET` | `/evaluations/{id}/diagnostics/calibration` | Calibration bins, ECE, MCE, and reliability curve coordinates |
| `GET` | `/evaluations/{id}/diagnostics/subgroups` | Subgroup slice metrics and performance discrepancies |
| `GET` | `/evaluations/{id}/acceptance` | Formal acceptance decision, criteria breakdown, and hash |
| `GET` | `/evaluations/{id}/report.html` | Download or view the self-contained offline HTML acceptance report |
| `GET` | `/evaluations/{id}/artifacts` | List all persisted artifact references and SHA256 checksums |
| `GET` | `/acceptance-policies` | List predefined acceptance policy templates and version rules |

---

## 3. Detailed Endpoint Contracts

### A. Dataset Inspection
- **Endpoint:** `POST /api/v1/datasets/inspect`
- **Request Body:**
  ```json
  {
    "file_path": "data/churn.csv",
    "target_column": "churn_label"
  }
  ```
- **Response (200 OK):** `DatasetMetadata` (see `docs/data-contracts.md`)
- **Error (404 Not Found):** File does not exist under `MLFORGE_DATA_DIR`.
- **Error (422 Unprocessable Entity):** Target column not present in dataset.

### B. Submit Evaluation
- **Endpoint:** `POST /api/v1/evaluations`
- **Request Body:** `EvaluationConfig` (see `docs/data-contracts.md`)
- **Response (202 Accepted):**
  ```json
  {
    "evaluation_id": "eval_7f9c21ba-8e43-41bb-a320-cfc813958012",
    "status": "queued",
    "submitted_at": "2026-10-18T18:45:00Z",
    "status_url": "/api/v1/evaluations/eval_7f9c21ba-8e43-41bb-a320-cfc813958012/status"
  }
  ```
- **Error (400 Bad Request):** Class count validation failed (e.g. fewer than $K$ samples for rare class).

### C. Polling Run Status
- **Endpoint:** `GET /api/v1/evaluations/{id}/status`
- **Response (200 OK):**
  ```json
  {
    "evaluation_id": "eval_7f9c21ba-8e43-41bb-a320-cfc813958012",
    "state": "running",
    "progress_percentage": 60.0,
    "current_fold": 3,
    "total_folds": 5,
    "current_action": "Fitting candidate pipeline on fold 3",
    "started_at": "2026-10-18T18:45:02Z",
    "finished_at": null,
    "error_message": null
  }
  ```

### D. Cancel Evaluation
- **Endpoint:** `POST /api/v1/evaluations/{id}/cancel`
- **Request Body:**
  ```json
  {
    "reason": "Cancelled by user via UI"
  }
  ```
- **Response (200 OK):**
  ```json
  {
    "evaluation_id": "eval_7f9c21ba-8e43-41bb-a320-cfc813958012",
    "state": "cancelled",
    "cancellation_reason": "Cancelled by user via UI"
  }
  ```
- **Error (409 Conflict):** Evaluation is already in terminal state (`completed`, `failed`).

### E. Evaluation Results
- **Endpoint:** `GET /api/v1/evaluations/{id}/results`
- **Response (200 OK):**
  ```json
  {
    "evaluation_id": "eval_7f9c21ba-8e43-41bb-a320-cfc813958012",
    "candidate": {
      "model_type": "logistic_regression",
      "num_folds": 5,
      "metric_summaries": {
        "macro_f1": {
          "metric_name": "macro_f1",
          "mean": 0.8124,
          "std": 0.0210,
          "median": 0.8140,
          "iqr": 0.0185,
          "min_value": 0.7810,
          "max_value": 0.8350,
          "confidence_interval_95": [0.7863, 0.8385],
          "undefined_count": 0
        }
      }
    },
    "baseline": {
      "model_type": "dummy_classifier",
      "num_folds": 5,
      "metric_summaries": {
        "macro_f1": {
          "mean": 0.4620,
          "std": 0.0050,
          "confidence_interval_95": [0.4558, 0.4682]
        }
      }
    },
    "statistical_comparison": {
      "macro_f1": {
        "test_name": "Nadeau-Bengio Corrected Resampled t-test",
        "statistic": 14.82,
        "p_value": 0.00012,
        "statistically_significant": true,
        "alpha": 0.05,
        "interpretation": "Candidate significantly outperforms baseline on Macro F1 (p < 0.001)."
      }
    }
  }
  ```

### F. Acceptance Decision
- **Endpoint:** `GET /api/v1/evaluations/{id}/acceptance`
- **Response (200 OK):**
  ```json
  {
    "decision_id": "dec_3419",
    "evaluation_id": "eval_7f9c21ba-8e43-41bb-a320-cfc813958012",
    "policy_id": "standard_classification_v1",
    "policy_version": "1.0.0",
    "outcome": "Pass",
    "criteria_evaluations": [
      {
        "rule_id": "min_macro_f1",
        "metric": "macro_f1",
        "observed_value": 0.8124,
        "threshold_value": 0.75,
        "operator": ">=",
        "satisfied": true,
        "is_inconclusive": false
      },
      {
        "rule_id": "outperform_dummy_baseline",
        "metric": "macro_f1",
        "observed_value": 0.00012,
        "threshold_value": 0.05,
        "operator": "statistically_greater_than",
        "satisfied": true,
        "is_inconclusive": false
      }
    ],
    "summary_rationale": "Model passed all 2 criteria. Candidate statistically exceeds dummy baseline with 95% CI [0.7863, 0.8385].",
    "evaluated_at": "2026-10-18T18:46:10Z",
    "hash_signature": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  }
  ```

---

## 4. Standard Error Structure

All 4xx and 5xx responses emit a standardized payload:

```json
{
  "error_code": "DATA_LEAKAGE_DETECTED",
  "message": "Column 'user_lifetime_value' exhibits perfect target correlation due to post-hoc feature derivation.",
  "details": [
    {
      "loc": ["dataset", "feature_columns", 12],
      "msg": "Identified as target-derived feature",
      "type": "leakage_violation"
    }
  ],
  "recovery_guidance": "Remove 'user_lifetime_value' from feature_columns in the evaluation configuration.",
  "timestamp": "2026-10-18T18:46:12Z"
}
```
