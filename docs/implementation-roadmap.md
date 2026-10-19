# MLForge — Multi-Phase Implementation Roadmap & Status Record

**Document Version:** 2.0.0
**Status:** Fully Implemented & Verified (Phases 1–6 Complete)

---

## 1. Multi-Phase Master Roadmap

MLForge was engineered incrementally to ensure that the core scientific engine is bulletproof before layering on API servers, persistent databases, and web interfaces:

```
┌────────────────────────────────────────────────────────┐
│ Phase 1: Architecture, Contracts, & Design System     │  <-- [COMPLETED]
│ Establish boundaries, contracts, design tokens, tests  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Phase 2: Classification Evaluation Increment           │  <-- [COMPLETED]
│ Core engine, Dummy vs Logistic, leakage-safe CV, tests │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Phase 3: Robustness & Stress Testing Engine            │  <-- [COMPLETED]
│ Controlled noise injection, perturbation, stress tests │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Phase 4: Application Layer, REST API, & Persistence    │  <-- [COMPLETED]
│ FastAPI endpoints, run lifecycle, SQLite persistence   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Phase 5: Anti-AI Frontend & Interactive Laboratory     │  <-- [COMPLETED]
│ React, Vite, 6 screens, URL state, offline HTML reports│
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Phase 6: System Hardening, Benchmarking, & Packaging   │  <-- [COMPLETED]
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

#### Milestone 2.3: Classification Metric Suite & Baseline Contrast / Statistical Significance
- **Deliverables:** `mlforge/engine/metrics.py`, `mlforge/engine/statistics.py`
- **Scope:**
  - Implement point and fold calculations: Accuracy, Balanced Accuracy, Macro Precision, Macro Recall, Macro F1, Confusion Matrix.
  - Implement probability metrics: Log Loss (with epsilon clipping), Brier Score, ROC-AUC (OvR macro), PR-AUC.
  - Guard against division-by-zero (return `None` with `is_undefined=True` and documented diagnostics).
  - Guard against single-class validation folds in ROC-AUC.
  - Compute fold aggregations: mean, standard deviation, median, IQR, min, max, 95% Confidence Interval.
  - Implement Nadeau and Bengio (2003) corrected resampled $t$-test for cross-validation folds.
  - Implement Wilcoxon signed-rank test.
  - Cryptographic partition fingerprint verification across paired evaluation folds.
  - Compute absolute performance gains ($\Delta$) with confidence intervals and directional metadata.
  - Return structured `ComparisonReport` and `StatisticalTestResult` with clear scientific interpretation.
- **Target Unit Tests:** `TC-MET-01` through `TC-MET-03`, `TC-AGG-01`, `TC-AGG-02`, `TC-AGG-03`, `tests/unit/test_statistics.py`.

#### Milestone 2.4: Probability Calibration, Threshold Analysis & Error Diagnostics
- **Deliverables:** `mlforge/engine/reliability.py`, `mlforge/engine/diagnostics.py`
- **Scope:**
  - Binary-classification probability calibration: Brier score, log loss, Expected Calibration Error (ECE) with equal-width binning.
  - Reliability diagrams numerical data (bin sample counts, mean predicted probability, observed positive rate, empty-bin handling).
  - Out-of-fold calibration diagnostics with zero leakage (no fitting of calibration transformations on validation data).
  - Classification threshold analysis across caller-specified or default threshold grids (TP, FP, TN, FN, precision, recall, specificity, F1, balanced accuracy, predicted-positive rate).
  - Explicit undefined-metric handling for zero denominators in threshold metrics (never silently substituting zero).
  - Fold-level and aggregated out-of-fold error diagnostics (error counts, error rates, false-positive / false-negative counts).
  - Descriptive non-target subgroup diagnostics with sample-size support thresholds and single-class safeguards.
- **Target Unit Tests:** `tests/unit/test_reliability.py`, `tests/unit/test_diagnostics.py`.

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
- [x] Evaluation engine runs in pure Python with zero web or DB dependencies.
- [x] `DummyClassifier` and `LogisticRegression` evaluated across $K$ folds with verified zero leakage.
- [x] Undefined metrics handled gracefully with diagnostic flags.
- [x] Statistical significance test (Nadeau-Bengio) correctly flags whether candidate beats baseline.
- [x] `Pass`, `Fail`, and `Inconclusive` acceptance decisions generated with cryptographic hash.
- [x] 100% of Phase 2 test matrix cases passing in pytest (125 tests passing).
- [x] Zero database migrations, FastAPI routes, or React UI code introduced prematurely.

---

## 4. Technical Risks & Mitigations

| Risk | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Fold Correlation Misleading Significance** | High | Enforce Nadeau-Bengio variance correction rather than naive paired t-test. |
| **Data Leakage in Complex Preprocessing** | Critical | Use scikit-learn `Pipeline` objects instantiated freshly inside each fold loop. |
| **Metric Division by Zero on Rare Classes** | Medium | Explicit `MetricValue` contract with `is_undefined` flag; never crash or silently substitute 1.0. |
| **Over-Engineering Scope** | Medium | Strict Phase boundaries: command-line and test-driven execution only; defer REST API to Phase 4. |
| **Silent Input Data Mutation in Perturbations** | High | Enforce defensive copying (`df.copy(deep=True)`) before applying any perturbation. |
| **Incompatible Baseline-Perturbed Metric Scaling** | Medium | Explicit calculation of retained metric ratio and percentage drop with zero-denominator safeguards. |

---

## 5. Precise Phase 3 Implementation Plan: Robustness & Stress Testing Engine

### Goal
Implement a scientifically defensible robustness and stress testing engine that measures model degradation under controlled feature corruptions, data quality impairments, and distribution perturbations without refitting or data leakage.

### Milestone Breakdown

#### Milestone 3.1: Robustness & Stress Testing Engine Foundation
- **Deliverables:** `mlforge/core/contracts.py`, `mlforge/engine/robustness.py`, `tests/unit/test_robustness.py`
- **Scope:**
  - Define formal contracts for perturbation configurations, perturbation steps, degradation curves, and robustness evaluation results (`PerturbationType`, `PerturbationConfig`, `RobustnessPerturbationResult`, `RobustnessCurve`, `RobustnessEvaluationResult`).
  - Implement deterministic, non-mutating feature perturbation transformers:
    1. **Gaussian Noise Injection (`gaussian_noise`):** Additive zero-mean Gaussian noise scaled to feature standard deviation ($\mathcal{N}(0, (\text{severity} \cdot \sigma_j)^2)$) for numeric features.
    2. **Missingness Injection (`missingness_injection`):** Controlled random MCAR (Missing Completely At Random) feature masking replacing values with `NaN` / `None`.
    3. **Category Dropout (`category_dropout`):** Controlled replacement of categorical values with an out-of-vocabulary / unseen token (`'__UNSEEN__'`) or random category corruption.
  - Implement robustness evaluation runner: evaluates fitted estimators or fold pipelines against clean baseline vs. incrementally perturbed validation partitions without re-fitting.
  - Calculate performance retention metrics: retained metric score, retained ratio ($\text{metric}_{\text{perturbed}} / \text{metric}_{\text{clean}}$), and performance drop percentage ($(\text{metric}_{\text{clean}} - \text{metric}_{\text{perturbed}}) / \text{metric}_{\text{clean}} \times 100\%$).
  - Guard against division-by-zero, non-finite values, and undefined metrics (return explicit `is_undefined=True` with diagnostic reason).
  - Guarantee input preservation: input DataFrames must never be mutated in-place.
  - Guarantee determinism: explicit random seeds produce bit-for-bit identical perturbed datasets and degradation curves.
- **Target Unit Tests:** `tests/unit/test_robustness.py` covering determinism, input non-mutation, perturbation bounds, metric drop accuracy, undefined metric handling, and invalid input rejection.

#### Milestone 3.1.1: Robustness Correctness Hardening (Completed)
- **Deliverables:** `mlforge/core/contracts.py`, `mlforge/engine/robustness.py`, `tests/unit/test_robustness.py`, `docs/data-contracts.md`
- **Scope & Audit Remediation:**
  - **Cross-Process Reproducibility:** Replaced Python hash-randomized `hash(col)` with canonical SHA-256 column seed derivation (`_derive_column_seed`) ensuring bit-for-bit identical noise across separate Python processes.
  - **Metric Directionality & Multi-Metric Dispatch:** Dispatched metric evaluation through `_compute_model_metric` using `calculate_fold_classification_metrics`. Validated unsupported metrics explicitly with `ValueError`. Formulated directional retention metrics for `HIGHER_IS_BETTER` vs. `LOWER_IS_BETTER` loss metrics (`log_loss`, `brier_score`).
  - **Native Pandas Categorical Support:** Added native `pd.CategoricalDtype` support to `apply_category_dropout` with category extension and collision-free token generation.
  - **Survivorship Bias Defense in Curves:** Conservative curve aggregation policy: if any requested severity level is unusable or encounters estimator failure, aggregate Area Under Degradation Curve is strictly undefined (`None`) with an explicit warning.
  - **Early Configuration Validation:** Added Pydantic `@model_validator` on `PerturbationConfig` validating severity bounds, ordering, non-emptiness, and deduplication.
  - **Regression Coverage:** Added 15 new targeted regression tests (total 41 unit tests in `test_robustness.py`).


#### Milestone 3.2: Multi-Feature Compound Perturbations & Stress Testing Profiles (Completed)
- **Deliverables:** `mlforge/core/contracts.py`, `mlforge/engine/robustness.py`, `mlforge/engine/__init__.py`, `tests/unit/test_robustness.py`, `docs/data-contracts.md`, `docs/evaluation-methodology.md`
- **Scope & Implementation:**
  - **Multi-Feature Compound Perturbations (`PerturbationType.COMPOUND`):** Implemented `apply_compound_perturbation` and compound evaluation dispatch applying Gaussian noise, category corruption, and missingness injection across distinct feature modalities in canonical deterministic order (`gaussian_noise` -> `category_dropout` -> `missingness_injection`). Guaranteed deep-copy non-mutation, cross-process reproducibility, and independent step seeds derived via deterministic offsets. Prohibited nested compound steps with strict configuration validation.
  - **Pre-Packaged Stress Profiles & Standard Grid:** Defined standard testing grid `STANDARD_SEVERITY_GRID = [0.05, 0.10, 0.25, 0.50]`. Implemented profile factories `create_noise_stress_profile`, `create_missingness_stress_profile`, `create_category_dropout_stress_profile`, and `create_compound_stress_profile`. Implemented `get_standard_stress_suite` and `evaluate_stress_profile` runner.
  - **Area Under Robustness Degradation Curve (AURDC):** Implemented `calculate_aurdc` using normalized trapezoidal integration over evaluated severity intervals. Ensured cross-version NumPy compatibility (`np.trapezoid` / `np.trapz`). Preserved conservative survivorship bias protection (returns `None` with explicit warning if any perturbation point is unusable or missing). Added `aurdc` property alias to `RobustnessCurve` and tracked per-curve and overall `mean_aurdc` in `summary_retention`.
  - **Unit Test Coverage:** Added 24 new unit tests covering compound perturbation ordering, immutability, determinism, step seed independence, AURDC integration, stress profile factories, and suite runners (total 65 tests in `test_robustness.py`).


#### Milestone 3.3: Robustness Acceptance Policy Evaluation & Artifact Serialization (Completed)
- **Deliverables:** `mlforge/core/contracts.py`, `mlforge/engine/acceptance.py`, `mlforge/application/artifacts.py`, `tests/unit/test_acceptance.py`, `tests/unit/test_artifacts.py`, `docs/data-contracts.md`, `docs/evaluation-methodology.md`
- **Scope & Implementation:**
  - **Robustness Acceptance Policy Evaluation:** Added `robustness` evidence source to `AcceptanceRule` evaluation in `mlforge/engine/acceptance.py`. Supported aggregate metrics (`mean_aurdc`), per-perturbation curve metrics (`<type>_aurdc`, `aurdc:<type>`), maximum severity retention (`<type>_max_retention`), maximum severity performance drop (`<type>_max_drop_pct`), and severity-specific metrics (`<type>:<severity>:drop_pct`, `<type>:<severity>:retention`). Enforced strict failure handling: missing evidence, non-evaluated curves, missing severity levels, and unusable evaluation points evaluate to `INCONCLUSIVE` without false passes. Preserved blocker/advisory precedence and cryptographic decision signature determinism.
  - **Robustness Artifact Serialization:** Integrated candidate and baseline robustness curves into serialized run artifacts (`evaluation.json`). Updated `ArtifactManifest` and `serialize_run_artifacts` to track `has_robustness: bool`. Preserved byte-level SHA-256 manifest integrity verification, strict non-finite number rejection (`allow_nan=False`), and intact serialization of null/None fields without zero coercion.
  - **Test Suite Verification:** Added 9 new unit tests in `test_acceptance.py` and 5 new unit tests in `test_artifacts.py` (total 14 new tests, 204 passing full suite).


#### Milestone 3.3.1: Robustness Metric Semantics & Artifact Integrity Audit (Completed)
- **Deliverables:** `mlforge/engine/robustness.py`, `mlforge/application/artifacts.py`, `mlforge/core/contracts.py`, `tests/unit/test_robustness.py`, `tests/unit/test_artifacts.py`, `docs/evaluation-methodology.md`, `docs/data-contracts.md`
- **Scope & Implementation:**
  - **Lower-is-Better Metric Semantics Audit:** Systematically audited retention ratios and drop percentages for error/loss metrics (`log_loss`, `brier_score`). Confirmed that `drop_pct = ((perturbed - clean) / clean) * 100` guarantees uniform positive degradation semantics across both metric directions. Verified boundary behaviors: zero baseline (`ZERO_BASELINE_SCORE`), negative scores (`NEGATIVE_SCORE_LOSS`), zero perturbed loss (`ZERO_PERTURBED_LOSS`), and extreme near-zero positive bounds.
  - **AURDC Defect Hardening:** Resolved confirmed defects in `calculate_aurdc`. Enforced ascending severity sorting prior to trapezoidal integration to prevent negative interval width distortion. Added strict rejection of duplicate severity levels to eliminate multivalued curve ambiguity. Enforced explicit non-finite (`NaN` / `Inf`) rejection on retention ratios and severity levels to preserve survivorship bias protections and prevent serialization crashes.
  - **Artifact Manifest Integrity:** Corrected `has_robustness` resolution in `serialize_run_artifacts` to track the presence of robustness evidence across both candidate and baseline results. Preserved SHA-256 byte-level tampering detection and intact null preservation without zero coercion.
  - **Test Suite Verification:** Added 10 new regression tests (8 in `test_robustness.py`, 2 in `test_artifacts.py`). Full suite: 214 passing tests.

#### Milestone 3.4: Phase 3 Verification & Benchmark Check (Completed)
- **Deliverables:** `tests/integration/test_robustness_lifecycle.py`, `benchmarks/runner.py`, `docs/implementation-roadmap.md`
- **Scope & Implementation:**
  - **Robustness Lifecycle Integration Test Suite:** Created `tests/integration/test_robustness_lifecycle.py` covering full lifecycle validation: dataset ingestion, stratified cross-validation, candidate/baseline model evaluation, standard stress testing suites across all four perturbation types (`gaussian_noise`, `missingness_injection`, `category_dropout`, `compound`), directional retention metrics, AURDC calculation, acceptance policy evaluation with blocker/advisory precedence, atomic artifact serialization with SHA-256 manifest integrity, round-trip reloading, and tamper detection.
  - **Stress Testing Performance Benchmarking:** Expanded `benchmarks/runner.py` with `benchmark_robustness_scenario` and `run_robustness_benchmarks` measuring isolated phase timings (clean baseline evaluation, noise, missingness, category dropout, compound, full suite, acceptance policy evaluation, artifact serialization, total elapsed) and peak memory overhead via standard library `tracemalloc`.
  - **Test Suite Verification:** Added 6 comprehensive integration tests (total 221 passing tests across full suite). Verified benchmark execution with reproducible deterministic seeds and zero regressions on prior milestone suites.

---

## 6. Precise Phase 4 Implementation Plan: Evaluation Artifacts, Audit Packaging & Model Release Gates

### Goal
Elevate MLForge from evaluation evidence generation into an auditable model governance and delivery framework featuring atomic fitted model serialization, portable tamper-evident audit bundles, automated release gates, offline HTML reporting, and local persistence.

### Milestone Breakdown

#### Milestone 4.1: Fitted Model Serialization & Audit Bundle Packaging (Completed)
- **Deliverables:** `mlforge/core/contracts.py`, `mlforge/core/exceptions.py`, `mlforge/engine/evaluator.py`, `mlforge/application/artifacts.py`, `mlforge/application/packaging.py`, `tests/unit/test_artifacts.py`, `tests/unit/test_packaging.py`, `docs/implementation-roadmap.md`
- **Scope & Implementation:**
  - **Fitted Pipeline Retention & Serialization:** Extended `ModelEvaluationResult` and `FoldResult` with `fitted_pipeline: Optional[Any] = Field(None, exclude=True)` to retain exact fitted scikit-learn pipelines without retraining or altering JSON serialization contracts. Implemented atomic model serialization to `models/candidate_pipeline.joblib` and `models/baseline_pipeline.joblib` with SHA-256 byte-level digests recorded in `manifest.json`.
  - **Unfitted Model Rejection:** Implemented strict fittedness checks using scikit-learn's `check_is_fitted`, raising `ArtifactSerializationError` upon attempting to serialize unfitted estimators.
  - **Executable Artifact Security Defense:** Implemented `load_fitted_model` with mandatory `trusted=True` opt-in and SHA-256 checksum verification against `manifest.json`, preventing accidental or untrusted Python unpickling.
  - **Portable Audit Bundle Packaging (`build_audit_bundle`):** Implemented atomic ZIP packaging writing `bundle_manifest.json` conforming to `AuditBundleManifest`, recording file inventory, sizes, and SHA-256 digests. Implemented deterministic archive builds with normalized POSIX paths, sorted members, and fixed timestamps.
  - **Tamper-Evident Verification (`verify_audit_bundle`):** Implemented strict archive integrity verification checking ZIP structural health, member sizes, and cryptographic SHA-256 hashes against the manifest without extracting to arbitrary paths or calling `joblib.load` / `pickle.load`.
  - **Zip-Slip & Path Traversal Defenses:** Implemented `normalize_bundle_path` and `validate_archive_member_safety` rejecting null bytes, absolute paths, Windows drive specifiers, path traversal (`..`), and duplicate normalized member names.
  - **Safe Extraction (`extract_audit_bundle`):** Implemented extraction with verified path confinement within the target destination directory.
  - **Unit Test Coverage:** Added 22 new unit tests (5 in `test_artifacts.py`, 17 in `test_packaging.py`), bringing the full passing suite from 221 to 243 tests.

#### Milestone 4.1.2: Targeted Artifact Hardening (Completed)
- **Deliverables:** `mlforge/application/artifacts.py`, `mlforge/application/packaging.py`, `tests/unit/test_artifacts.py`, `tests/unit/test_packaging.py`, `docs/implementation-roadmap.md`
- **Scope & Implementation:**
  - **Model Role Validation:** Restricted `load_fitted_model` to exact supported roles (`candidate` and `baseline`), rejecting invalid roles, directory traversal attempts, and absolute paths before path resolution or filesystem operations.
  - **Bounded Archive Safety:** Enforced centralized, configurable bounds on archive operations: `MAX_ARCHIVE_BYTES` (500 MB), `MAX_UNCOMPRESSED_MEMBER_BYTES` (100 MB), and `MAX_TOTAL_UNCOMPRESSED_BYTES` (1 GB) to defend against zip-bomb and resource exhaustion attacks.
  - **Streaming SHA-256 Digest Verification:** Replaced unbounded in-memory `zf.read()` with bounded chunked streaming (64 KB chunks) for cryptographic checksum validation.
  - **Unit Test Coverage:** Added 12 new security and boundary unit tests (5 in `test_artifacts.py`, 7 in `test_packaging.py`), bringing the full passing suite to 255 tests.

#### Milestone 4.2: Offline HTML Audit Report & Evidence Presentation (Completed)
- **Deliverables:** `mlforge/application/reporting.py`, `mlforge/application/__init__.py`, `tests/unit/test_reporting.py`, `docs/data-contracts.md`, `docs/implementation-roadmap.md`
- **Scope & Implementation:**
  - **Public Reporting API (`generate_html_report`, `load_report_evidence`):** Generates a completely standalone, offline, self-contained HTML audit report (`report.html`) from a saved run artifact directory or a verified `.zip` audit bundle.
  - **10 Formal Audit Report Sections:**
    1. *Report Header & Run Identity:* Run ID, schema version, dataset SHA-256, model types, CV split strategy, fold count, model provenance (`final_cv_fold`), fitted fold index, and 4-tier integrity verification summary.
    2. *Executive Evaluation Summary:* Decision outcome pill (PASS / FAIL / INCONCLUSIVE), criterion outcome counts, candidate and baseline key metrics, macro F1 delta, and operational limitations.
    3. *Classification Performance:* Formatted tabular display of aggregate CV metrics (mean, std, 95% CI, valid fold count), fold-by-fold metrics, and out-of-fold aggregate confusion matrices. Missing/undefined metrics formatted as N/A with diagnostic tooltip badges.
    4. *Candidate-vs-Baseline Comparison:* Signed deltas ($C - B$), metric directionality (`HIGHER_IS_BETTER` vs `LOWER_IS_BETTER`), Nadeau-Bengio corrected resampled t-test status, p-values, test alpha, statistical significance badges, and fold-dependence warnings. Explicit fallback when baseline is absent.
    5. *Calibration & Threshold Analysis:* Probability calibration table with observed vs predicted rates, sample support, empty bin indicators; ECE, MCE, Brier score, log loss; threshold sweeps; inline SVG Reliability Diagram and Threshold Trade-Off curves.
    6. *Error Analysis & Subgroup Diagnostics:* Fold-level and pooled false positive / false negative counts and error rates; subgroup slices with sample support badges and small-sample warnings (<30 instances).
    7. *Robustness & Stress Evaluation:* Perturbation types, severity levels, clean vs perturbed scores, retention ratios, percentage drops, and AURDC scores; inline SVG Robustness Degradation Trajectory.
    8. *Acceptance Decision & Policy:* Detailed criterion-level audit table detailing rule ID, severity (BLOCKER / ADVISORY), operator, numerical threshold, observed value, pass/fail status, and diagnostic reason codes.
    9. *Artifact Integrity & 4-Tier Provenance:* Inventory table listing relative paths, artifact types, file sizes, SHA-256 checksums, and explicit verification status across Tier 1 (File Existence), Tier 2 (Hash Match), Tier 3 (Bundle Consistency), and Tier 4 (External Trust / Author Authenticity - documented as unverified without external PKI).
    10. *Methodology, Limitations & Warnings:* Contextual guidance documenting CV fold dependence, out-of-fold pooling vs fold means, final CV fold model provenance, survivorship safeguards, and missing class support caveats.
  - **Security & Data Safety:** All untrusted user/dataset/model strings escaped with `html.escape`. Safe JSON embedding using `</` escaping. Absolute prohibition on deserializing `.joblib` / `.pickle` model files. Path traversal rejection for both input runs and output target files.
  - **Atomic Output Delivery:** Generates report into a sibling `.tmp` file and replaces atomically via `os.replace`, guaranteeing that interrupted generation never leaves a corrupted partial file.
  - **Preserved Manifest Invariant:** Generating `report.html` does not mutate `manifest.json`, preventing circular hashing and preserving existing audit bundle integrity.
  - **Zero External Dependencies:** 100% offline; zero external fonts, CDNs, remote images, or tracking scripts. All CSS and vector SVGs embedded inline.
  - **Unit Test Coverage:** Added 12 comprehensive unit tests in `tests/unit/test_reporting.py`, expanding passing suite from 255 to 267 tests with zero regressions.

#### Milestone 4.3: Automated Release-Gate Engine & Cryptographically Signed Release Certificates (Completed)
- **Deliverables:** `mlforge/core/contracts.py`, `mlforge/core/exceptions.py`, `mlforge/engine/release_gate.py`, `mlforge/application/release_certificates.py`, `mlforge/application/reporting.py`, `mlforge/engine/__init__.py`, `mlforge/application/__init__.py`, `tests/unit/test_release_gate.py`, `tests/unit/test_release_certificates.py`, `docs/data-contracts.md`, `docs/implementation-roadmap.md`
- **Scope & Implementation:**
  - **Release-Gate Engine (`evaluate_release_gate`):** Distinct from statistical metric acceptance (`PASS`/`FAIL`), answers: *Is this completed run sufficiently complete, valid, traceable, and compliant with release requirements to authorize a release?*
    - Checks 5 core categories: `schema_compatibility`, `evidence_completeness`, `acceptance_compliance`, `artifact_integrity`, and `provenance_consistency`.
    - Enforces deterministic precedence: `BLOCKED` > `INCONCLUSIVE` > `PASSED`.
    - Respects `ReleasePolicy` configuration: configurable requirements for acceptance pass, candidate model artifacts, baseline comparisons, robustness stress tests, probability calibration, and manifest verification.
    - Missing optional evidence yields warnings; only policy-mandated evidence failures cause blocking or inconclusive verdicts.
    - Non-executable verification: never deserializes `.joblib` or `.pickle` model files during gate evaluation.
  - **Cryptographically Signed Release Certificates (`issue_release_certificate`, `verify_release_certificate`):**
    - Employs standard, high-assurance Ed25519 digital signatures via standard `cryptography` library (`hazmat.primitives.asymmetric.ed25519`).
    - Strictly separates signature validity from signer trust. Trust requires an explicit trusted public key or key registry.
    - Deterministic canonical JSON payload: UTF-8, alphabetically sorted keys, separators `(',', ':')`, strict rejection of `NaN` and `Infinity`.
    - 5-Tier comprehensive verification: `payload_integrity`, `signature_validity`, `signer_trust`, `evidence_consistency` (cross-checking dataset SHA-256, model SHA-256, run ID, and manifest SHA-256 against actual artifacts on disk), and `release_authorized`.
    - Atomic filesystem delivery: writes to sibling `.tmp` file and atomically replaces (`os.replace`) to prevent corrupt or partial certificate files. Zero mutation of original evaluation artifacts or `manifest.json`.
  - **HTML Audit Report Integration:**
    - Automatically discovers and displays certificate verification status, payload hash, signature algorithm, signer key ID, and signer trust in `report.html`.
    - Explicit fallback banner when no release certificate has been issued.
  - **Unit Test Coverage:**
    - 15 unit tests in `tests/unit/test_release_gate.py` (complete evidence, missing artifacts, blocker failures, inconclusive acceptance, schema mismatch, manifest mismatch, model provenance, precedence, idempotency).
    - 14 unit tests in `tests/unit/test_release_certificates.py` (issuance, block/inconclusive denial, valid signature, payload tampering, hash tampering, wrong public key, untrusted signer, invalid signatures, non-finite rejection, atomic write safety, canonicalization determinism).
    - Full test suite: 296 passed, 0 failed, 196 warnings (identical baseline warnings + expected sklearn deprecation warnings in test fixtures).

#### Milestone 4.4: Local SQLite Persistence & Run Lifecycle State Machine (Completed)
- **Deliverables:** `mlforge/core/contracts.py`, `mlforge/core/exceptions.py`, `mlforge/application/persistence.py`, `mlforge/application/__init__.py`, `tests/unit/test_persistence.py`, `docs/data-contracts.md`, `docs/implementation-roadmap.md`
- **Scope & Implementation:**
  - **Decoupled Architecture Boundary:** Implemented durable local persistence in `mlforge.application.persistence` while maintaining strict isolation for the pure scientific evaluation engine (`mlforge.engine`), which has zero SQLite dependencies or side effects.
  - **Run Lifecycle State Machine:** Formalized explicit finite state machine with states `CREATED`, `RUNNING`, `COMPLETED`, `FAILED`, and `CANCELLED`. Enforced strict transition validation, rejecting invalid state hops with `LifecycleTransitionError`. Terminal states (`COMPLETED`, `FAILED`, `CANCELLED`) are strictly immutable.
  - **Local SQLite Repository (`SQLiteRunRepository`):** Lightweight, zero-ORM persistence using standard library `sqlite3` with foreign key enforcement (`PRAGMA foreign_keys = ON;`), exclusive parameterized SQL, transaction rollbacks, connection timeouts, and busy handling.
  - **Schema & Relational Indexing:** Created normalized relational tables: `runs` (run metadata, lifecycle state, UTC timestamps, schema version, models, acceptance outcome, artifact path, safe error information), `run_metrics` (indexed CV aggregate metric means, fold counts, standard deviations, and 95% confidence intervals), and `lifecycle_events` (audit log recording timestamped transitions and rationale).
  - **Stale Run Recovery (`recover_stale_runs`):** Provides safe, deterministic recovery for abandoned `RUNNING` executions following process interruptions, transitioning them to `FAILED` with category `"STALE_RUN_TIMEOUT"` without silent restarts.
  - **Application-Level Orchestrator (`execute_managed_evaluation`):** Coordinates lifecycle progression: creates `CREATED` run record, transitions to `RUNNING`, executes the pure evaluation engine, saves artifacts atomically via `serialize_run_artifacts`, records artifact references and indexes summary metrics, and transitions to `COMPLETED` (or `FAILED` with safe error categorization upon unhandled exceptions).
  - **Security & Path Confinement:** Enforced strict artifact path traversal validation, rejecting relative `..` sequences and unconfined paths. Structured error messages are sanitized and length-capped to prevent secret leakage or unbounded stack traces in the database.
  - **Unit Test Coverage:** Added 16 focused unit tests in `tests/unit/test_persistence.py` covering schema initialization, idempotency, duplicate run ID rejection, state transitions, terminal state immutability, audit logging, safe error recording, summary metric indexing, foreign key cascading, paginated listing, path traversal defenses, stale run recovery, orchestration success/failure, and pure engine isolation.
  - **Test Suite Verification:** Full test suite expanded from 302 to 318 passed tests (0 failed, 204 warnings matching upstream sklearn deprecation warnings in fixtures).

#### Milestone 4.4.2: Targeted Persistence Audit Remediation (Completed)
- **Deliverables:** `mlforge/application/persistence.py`, `tests/unit/test_persistence.py`, `docs/data-contracts.md`, `docs/implementation-roadmap.md`
- **Scope & Implementation:**
  - **In-Memory SQLite Support:** Configured `SQLiteRunRepository(":memory:")` to use a dedicated shared in-memory URI (`file:mlforge_mem_<id>?mode=memory&cache=shared`) with an internal keepalive connection to prevent database deletion across per-operation connection lifecycles without resolving to invalid NTFS filesystem paths.
  - **Observable Secondary Failure Handling:** In `execute_managed_evaluation`, secondary errors occurring while writing the `FAILED` state now emit an observable `UserWarning` and `logger.warning` without suppressing or masking the primary evaluation exception.
  - **Artifact-Path Confinement & Trust Boundaries:** Enhanced `_validate_safe_artifact_path` and `update_run_artifacts` to accept an optional `artifact_root`. When configured, canonical resolution (`.resolve()`) verifies confinement within the root. When omitted, external paths without `..` traversal are permitted under the documented trust boundary.
  - **Stale-Run Recovery Documentation & Testing:** Added docstring and documentation clarifying that elapsed-time recovery cannot verify process liveness and requires an appropriate timeout. Added boundary cutoff regression test and transactional audit log verification.
  - **Unit Test Verification:** Added 4 new regression tests in `tests/unit/test_persistence.py` (total 20 persistence unit tests, full suite: 322 passing tests, 204 warnings).

#### Milestone 4.5: Phase 4 Verification & Packaging Hardening (Completed)
- **Deliverables:** `mlforge/cli.py`, `mlforge/__main__.py`, `pyproject.toml`, `benchmarks/runner.py`, `tests/unit/test_cli.py`, `tests/integration/test_phase4_lifecycle.py`, `docs/implementation-roadmap.md`, `README.md`
- **Scope & Implementation:**
  - **Audit-Bundle Stress and Boundary Hardening (Task 1):** Wrapped `extract_audit_bundle` to catch `(zipfile.BadZipFile, EOFError)` and raise `AuditBundleCorruptError` when `verify_first=False`. Added 7 boundary tests in `tests/unit/test_packaging.py` for member size limits, total uncompressed size limits, compression ratio bounds, truncated archives, CRC-32 corruption, decompression size spoofing, and deeply nested paths.
  - **Offline CLI and Declarative Packaging (Task 2):** Implemented zero-dependency `argparse` CLI in `mlforge/cli.py` and `mlforge/__main__.py` with subcommands `verify-bundle` and `verify-release`. Standardized exit codes (0 = valid/authorized, 1 = rejected/tampered/untrusted, 2 = usage/file error). Defined PEP 621 metadata in `pyproject.toml` with console script entry point `mlforge`. Added 19 unit tests in `tests/unit/test_cli.py` proving zero model deserialization during verification.
  - **Unified Phase 4 Integration (Task 3):** Created `tests/integration/test_phase4_lifecycle.py` verifying the full application pipeline across real components without mocking: managed evaluation with stratified folds -> SQLite persistence & lifecycle history -> artifact integrity & model provenance -> offline HTML report -> release gate evaluation -> Ed25519 certificate issuance -> non-circular audit bundle packaging -> safe extraction -> multi-tiered verification -> CLI invocation. Added 6 negative path integration tests (gate rejection, post-issuance tampering, untrusted signer, corrupt bundle, run ID mismatch, and lifecycle state immutability).
  - **Phase 4 Benchmarking (Task 4):** Expanded `benchmarks/runner.py` with `benchmark_phase4_scenario` and `run_phase4_benchmarks` measuring isolated timings across evaluation, persistence, HTML reporting, gate evaluation, Ed25519 signing/verification, bundle packaging/verification, and safe extraction. Documented `tracemalloc` peak heap allocation methodology (excluding OS process RSS). Output saved to `artifacts/benchmarks/phase4_benchmark_results.json`.
  - **Test Suite Verification:** Full test suite expanded from 322 to 356 passing tests with zero regressions. All Phase 4 acceptance criteria fully met.

---

## 7. Phase 5: Interactive ML Experimentation Laboratory (Completed)
- **Goal:** Build the local React + TypeScript + Vite user interface and FastAPI REST API layer directly consuming the backend persistence and evaluation engines across all 6 specified screens.
- **Deliverables:**
  - `mlforge/api/`: REST API implementation with FastAPI, routers, schemas, CORS middleware, and error handling.
  - `mlforge/cli.py`: Added `serve` subcommand launching the local API server via Uvicorn.
  - `frontend/`: Complete React 18, TypeScript, and Vite application conforming to the Anti-AI design system in `docs/design-system.md` and blueprint in `docs/frontend-blueprint.md`.
  - Screens implemented: `LabOverview`, `NewEvaluation`, `EvaluationResults`, `ReliabilityAnalysis`, `RunDetails`, and `AcceptanceReport`.
  - SVG Visualizations: `CalibrationChart` (reliability diagrams with deviation connectors and 10 probability bins), `ConfusionMatrix` (normalized and raw counts), `FoldDistributionChart` (strip plot for cross-validation variance), and `RobustnessChart` (multi-scenario degradation curves with AURDC).
  - Tests: 10 backend REST API unit tests in `tests/unit/test_api.py` (total 366 backend tests passing) and 3 frontend client tests in `frontend/src/tests/client.test.ts`.

---

## 8. Phase 6: System Hardening & Final Delivery (Completed)
- **Goal:** End-to-end user journey verification, cross-platform packaging verification, performance profiling, and production readiness audit.
- **Deliverables & Verification:**
  - Automated release-gate engine and Ed25519 cryptographic certificate verification with fail-closed security.
  - Hardened artifact packaging with Zip-Slip path-traversal guards and streaming SHA-256 integrity verification.
  - Decoupled 7-tier release governance architecture across backend contracts and interactive React UI.
  - Complete backend test suite: 416 passed, 1 skipped (Windows non-admin symlink check).
  - Complete frontend test suite: 44 passed across workflow trust, client, navigation, and components.
  - Production bundle compilation verified (`tsc && vite build`).
  - Zero-regression benchmark suite across continuous binary, multiclass, and churn tabular datasets.
