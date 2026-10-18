# MLForge — Implementation Roadmap & Phase 2 Blueprint

**Document Version:** 1.0.0  
**Phase:** 1 (Foundation & Architecture)  
**Status:** Approved Specification  

---

## 1. Multi-Phase Master Roadmap

MLForge is engineered incrementally to ensure that the core scientific engine is bulletproof before layering on API servers, persistent databases, and web interfaces:

```
┌────────────────────────────────────────────────────────┐
│ Phase 1: Architecture, Contracts, & Design System     │  <-- [COMPLETED]
│ Establish boundaries, contracts, anti-AI tokens, tests │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Phase 2: Classification Evaluation Increment           │  <-- [NEXT PHASE]
│ Core engine, Dummy vs Logistic, leakage-safe CV, tests │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Phase 3: Reliability, Diagnostics, & Robustness Engine │
│ Calibration (ECE), subgroups, noise injection, errors  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Phase 4: Application Layer, REST API, & Persistence    │
│ FastAPI endpoints, run lifecycle, SQLite/PostgreSQL    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Phase 5: Anti-AI Frontend & Interactive Laboratory     │
│ React, Vite, Plotly, 6 screens, URL state, offline HTML│
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Phase 6: System Hardening, Benchmarking, & Packaging   │
│ End-to-end audit, performance profiling, CLI release   │
└────────────────────────────────────────────────────────┘
```

---

## 2. Precise Phase 2 Implementation Plan

### Goal
Implement the first independently testable, scientifically defensible classification evaluation increment comparing `DummyClassifier` against `LogisticRegression` under strict leakage-free cross-validation.

### Milestone Breakdown

#### Milestone 2.1: Dataset Contracts & Validation
- **Deliverables:** `mlforge/core/dataset.py`
- **Scope:**
  - Ingest pandas DataFrame or CSV file.
  - Compute `DatasetMetadata` (feature types, null counts, class counts, SHA256 checksum).
  - Validation guards: check target column existence, detect single-value constant features, enforce minimum class count $\ge K$.
- **Target Unit Tests:** `TC-DS-01` through `TC-DS-04`, `TC-SPLIT-01`.

#### Milestone 2.2: Leakage-Safe Evaluation Protocol & Pipeline Execution
- **Deliverables:** `mlforge/engine/protocol.py`, `mlforge/engine/pipeline.py`
- **Scope:**
  - Implement deterministic stratified $K$-fold generator with fixed `random_state`.
  - Build fold execution runner that isolates preprocessing (imputation, standard scaling, one-hot encoding) inside an `sklearn.pipeline.Pipeline`.
  - Ensure zero leakage: fit transformers strictly on fold training data; transform fold validation data.
  - Execute both candidate (`LogisticRegression`) and baseline (`DummyClassifier`) across identical folds.
- **Target Unit Tests:** `TC-LEAK-01` through `TC-LEAK-04`, `TC-REP-01`.

#### Milestone 2.3: Classification Metric Suite & Edge-Case Safeguards
- **Deliverables:** `mlforge/engine/metrics.py`
- **Scope:**
  - Implement point and fold calculations: Accuracy, Balanced Accuracy, Macro Precision, Macro Recall, Macro F1, Confusion Matrix.
  - Implement probability metrics: Log Loss (with epsilon clipping), Brier Score, ROC-AUC (OvR macro), PR-AUC.
  - Guard against division-by-zero (return 0.0 with `is_undefined=True` and documented diagnostics).
  - Guard against single-class validation folds in ROC-AUC.
  - Compute fold aggregations: mean, standard deviation, median, IQR, min, max, 95% Confidence Interval.
- **Target Unit Tests:** `TC-MET-01` through `TC-MET-03`, `TC-AGG-01`, `TC-AGG-02`.

#### Milestone 2.4: Baseline Contrast & Statistical Significance Engine
- **Deliverables:** `mlforge/engine/statistics.py`
- **Scope:**
  - Implement Nadeau and Bengio (2003) corrected resampled $t$-test for cross-validation folds.
  - Implement Wilcoxon signed-rank test.
  - Compute absolute and relative performance gains ($\Delta$) with confidence intervals.
  - Return structured `StatisticalTestResult` with clear scientific interpretation.
- **Target Unit Tests:** `TC-AGG-03`.

#### Milestone 2.5: Acceptance Decision Evaluator & Artifact Serialization
- **Deliverables:** `mlforge/engine/acceptance.py`, `mlforge/application/artifacts.py`
- **Scope:**
  - Implement Acceptance Policy parser and rule engine.
  - Evaluate criteria against observed values, standard deviations, and statistical tests.
  - Output three-state verdict: `Pass`, `Fail`, or `Inconclusive`.
  - Serialize run configuration, fold results, metric summaries, and acceptance decision to local JSON artifacts under `artifacts/runs/{id}/`.
- **Target Unit Tests:** `TC-ACC-01` through `TC-ACC-04`.

#### Milestone 2.6: Test Suite Validation & Verification Check
- **Deliverables:** `tests/unit/`, `tests/integration/`
- **Scope:**
  - Full suite of automated pytest tests executing the matrix defined in `docs/testing-strategy.md`.
  - Benchmark on standard tabular datasets (e.g. breast cancer / iris / synthetic binary churn).

---

## 3. Phase 2 Entry & Exit Criteria

### Entry Criteria (Must be met before starting Phase 2)
- [x] Architecture, module boundaries, and dependency directions approved.
- [x] Scientific evaluation methodology formalized.
- [x] Data contracts defined with Pydantic schemas.
- [x] Anti-AI design tokens and frontend blueprint documented.
- [x] Testing strategy and quality matrix defined.
- [x] Clean local git repository established with proper `.gitignore` and `.env.example`.

### Exit Criteria (Must be met before declaring Phase 2 complete)
- [ ] Evaluation engine runs in pure Python with zero web or DB dependencies.
- [ ] `DummyClassifier` and `LogisticRegression` evaluated across $K$ folds with verified zero leakage.
- [ ] Undefined metrics handled gracefully with diagnostic flags.
- [ ] Statistical significance test (Nadeau-Bengio) correctly flags whether candidate beats baseline.
- [ ] `Pass`, `Fail`, and `Inconclusive` acceptance decisions generated with cryptographic hash.
- [ ] 100% of Phase 2 test matrix cases passing in pytest.
- [ ] Zero database migrations, FastAPI routes, or React UI code introduced prematurely.

---

## 4. Technical Risks & Mitigations

| Risk | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Fold Correlation Misleading Significance** | High | Enforce Nadeau-Bengio variance correction rather than naive paired t-test. |
| **Data Leakage in Complex Preprocessing** | Critical | Use scikit-learn `Pipeline` objects instantiated freshly inside each fold loop. |
| **Metric Division by Zero on Rare Classes** | Medium | Explicit `MetricValue` contract with `is_undefined` flag; never crash or silently substitute 1.0. |
| **Over-Engineering Scope** | Medium | Strict Phase 2 boundaries: command-line and test-driven execution only; defer REST API to Phase 4. |
