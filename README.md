# MLForge — AI Model Evaluation & Reliability Lab

> **A local-only, evidence-backed evaluation and reliability platform for tabular machine learning models.**

---

## 1. Executive Summary & Purpose

**MLForge** is an engineering and scientific platform dedicated to rigorous model evaluation, statistical reasoning, prediction reliability, failure analysis, controlled robustness experiments, and evidence-backed acceptance decisions.

### Crucial Differentiation: MLForge vs DataFlow
- **DataFlow** focuses on data processing, ingestion, feature engineering, and pipeline orchestration.
- **MLForge** focuses strictly on **evaluative truth and model reliability**:
  - Is the model's reported performance statistically defensible, or is it an artifact of fold variance and data leakage?
  - Does the model fail catastrophically under slight feature corruptions or subgroup shifts?
  - Are probability outputs well-calibrated and reliable for downstream decision-making?
  - Does the empirical evidence support an explicit **Pass**, **Fail**, or **Inconclusive** deployment decision against versioned acceptance policies?

---

## 2. Core Architecture & Dependency Direction

MLForge is designed as a **modular monolith** with a strict unidirectional dependency hierarchy:

```
[ Frontend (React + TypeScript + Vite + Plotly) ]
                        │
                        ▼ (HTTP / REST)
       [ Application & API Layer (FastAPI) ]
           │                         │
           ▼ (Persistence/I/O)       ▼ (In-process execution)
 [ Local DB & Artifact Store ]   [ Scientific Evaluation Engine ]
```

### Architectural Safeguards:
1. **Engine Independence**: The core scientific evaluation engine (`mlforge.engine`) has zero dependencies on FastAPI, Pydantic API schemas, SQLAlchemy, React, or database drivers. It can be invoked directly as a pure Python library or CLI tool.
2. **Leakage-Safe Preprocessing**: All data transformations (imputation, scaling, encoding) are encapsulated within `scikit-learn` pipelines and fitted strictly and exclusively on the training partition of each cross-validation fold.
3. **Traceable Acceptance Decisions**: Every acceptance outcome (`Pass`, `Fail`, `Inconclusive`) is cryptographically verifiable against an exact acceptance policy version, observed metric distributions, confidence intervals, and evaluation protocol metadata.
4. **Local-Only Execution**: MLForge runs completely locally with direct in-process execution. No cloud infrastructure, microservices, Celery, or Kubernetes are required.

---

## 3. Documentation Set (Phase 1 Deliverables)

All architectural specifications, scientific methodologies, contracts, design systems, and roadmaps are documented under [`docs/`](file:///e:/Python/Python%20projects/MLForge/docs/):

| Document | Description |
| :--- | :--- |
| [`docs/architecture.md`](file:///e:/Python/Python%20projects/MLForge/docs/architecture.md) | Modular monolith architecture, module boundaries, data flow, storage model, and Architecture Decision Records (ADRs). |
| [`docs/evaluation-methodology.md`](file:///e:/Python/Python%20projects/MLForge/docs/evaluation-methodology.md) | Scientific methodology for tabular classification: baseline comparison, cross-validation, leakage prevention, undefined metrics handling, statistical significance tests, and acceptance criteria. |
| [`docs/data-contracts.md`](file:///e:/Python/Python%20projects/MLForge/docs/data-contracts.md) | Formal, versioned Pydantic/JSON schemas for configurations, fold-level metrics, uncertainty bounds, diagnostics, acceptance decisions, and artifact manifests. |
| [`docs/api-specification.md`](file:///e:/Python/Python%20projects/MLForge/docs/api-specification.md) | REST API endpoints, request/response models, run lifecycle states (`queued`, `running`, `completed`, `failed`, `cancelled`), and transition rules. |
| [`docs/frontend-blueprint.md`](file:///e:/Python/Python%20projects/MLForge/docs/frontend-blueprint.md) | Information architecture and comprehensive screen specifications for all 6 pages (Lab Overview, New Evaluation, Evaluation Results, Reliability Analysis, Run Details, Acceptance Report). |
| [`docs/design-system.md`](file:///e:/Python/Python%20projects/MLForge/docs/design-system.md) | Anti-AI Web Design & Engineering Manual: tinted neutral palette, dual-typeface typography, tabular numerals, surface layering, compact technical tables, and Plotly styling. |
| [`docs/testing-strategy.md`](file:///e:/Python/Python%20projects/MLForge/docs/testing-strategy.md) | Comprehensive test matrix covering leakage safety, numerical edge cases, class imbalance, reproducibility, run lifecycle, and contract verification. |
| [`docs/implementation-roadmap.md`](file:///e:/Python/Python%20projects/MLForge/docs/implementation-roadmap.md) | Multi-phase roadmap with an exact, milestone-level breakdown for Phase 2 (Classification Evaluation Increment). |

---

## 4. Local Development Environment

### Prerequisites
- **Python**: 3.11+
- **Node.js**: v20+ (LTS) or v22+ (for upcoming Phase 5 frontend)
- **Git**: 2.40+

### Installation
Install MLForge in editable mode with development dependencies:
```bash
python -m pip install -e . --no-deps
```
Or with development packages:
```bash
python -m pip install -e ".[dev]"
```

### Environment Configuration
Copy the sample environment file before running any services:
```bash
cp .env.example .env
```

---

## 5. Offline CLI Usage

MLForge provides a minimal, offline command-line interface with zero network dependencies:

### Verify Audit Bundle
Verify ZIP archive integrity, manifest conformity, CRC-32 checksums, and resource bounds:
```bash
mlforge verify-bundle <bundle_path.zip>
# Or via python -m:
python -m mlforge verify-bundle <bundle_path.zip>
```

### Verify Release Authorization
Verify an Ed25519 release certificate against cryptographic signatures, explicit signer trust, and physical evaluation evidence:
```bash
# With explicit public key hex (64 hex characters):
mlforge verify-release <run_dir_or_bundle.zip> --public-key <64_char_hex>

# With trusted keys registry file:
mlforge verify-release <run_dir_or_bundle.zip> --trusted-keys-file <path_to_keys.json>
```

**Security & Trust Rules:**
- Exit code `0`: Release authorized (all 4 verification tiers passed: payload integrity, signature validity, signer trust, and evidence consistency).
- Exit code `1`: Verification failed (invalid signature, untrusted key, tampered evidence, or blocked gate).
- Exit code `2`: Usage or file-not-found error.
- **Fail-Closed Signer Trust**: A valid signature from an untrusted or missing key always fails release authorization (never returns exit code 0).
- **Zero Model Deserialization**: CLI verification verifies model digests via streaming SHA-256 without calling `joblib.load` or `pickle.load`.

---

## 6. Running Tests & Benchmarks

### Full Test Suite
Run the full test suite with standard pytest:
```bash
python -m pytest -v
```

### Performance Benchmarks
Run the unified benchmark suite across classification (Milestone 2.6), robustness (Milestone 3.4), and Phase 4 lifecycle/persistence/packaging (Milestone 4.5):
```bash
python -m benchmarks.runner
```
*Memory Methodology Note*: Memory measurements report peak traced Python heap allocations via standard library `tracemalloc`. This reflects heap allocations and excludes OS process RSS, native C/C++ buffers, and memory-mapped files.

---

## 7. Interactive ML Experimentation Laboratory (Phase 5)

MLForge Phase 5 provides an interactive, dense, technical web application connecting directly to the local FastAPI backend.

### Starting the Local Laboratory

1. **Start the Backend API Core:**
   ```bash
   python -m mlforge serve --host 127.0.0.1 --port 8000
   ```
   *The REST API will be available at `http://127.0.0.1:8000/api/v1` with documentation at `http://127.0.0.1:8000/docs`.*

2. **Start the Frontend Application:**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   *The interactive lab will open at `http://localhost:5173`.*

### Available Laboratory Workflows

- **Lab Overview (`/`)**: Comprehensive ledger of all evaluation runs recorded in local SQLite persistence. Filter by state (`completed`, `running`, `failed`, `cancelled`), inspect Macro F1 with 95% confidence intervals, and review acceptance outcomes.
- **New Evaluation (`/new`)**: Configure dataset parameters (CSV/Parquet file paths or built-in benchmarks like Customer Churn or Breast Cancer), inspect schema and class distributions, configure model hyperparameters (C, solver, penalty, max iterations), tune stratified cross-validation folds, select policy templates, and execute evaluations with live preflight summaries.
- **Evaluation Results (`/results`)**: Empirical performance comparison against dummy baseline across primary and secondary metrics. Includes $\mu \pm \sigma$, 95% confidence intervals, relative $\Delta$, corrected resampled t-test p-values, cross-validation fold stability distributions, and side-by-side confusion matrices.
- **Reliability & Robustness Laboratory (`/reliability`)**: Deep diagnostic analysis of posterior calibration (Expected Calibration Error, Brier score, reliability diagram with 10 confidence bins), subgroup slice disparities across sensitive or demographic cohorts, and live perturbation stress testing (Gaussian noise, missingness, label inversion) with Area Under Retention Degradation Curve (AURDC).
- **Run Details & Artifact Ledger (`/details`)**: Inspect physical disk artifacts, file sizes, and SHA-256 digests. View environment reproducibility manifests, launch standalone offline HTML audit reports, and generate `.zip` audit bundles.
- **Release Readiness & Cryptographic Verification (`/acceptance`)**: Strict 6-state release governance dashboard evaluating criteria rules, blocking failures, and issuing tamper-evident Ed25519 digital certificates with cryptographic signature and manifest verification.

---

## 8. Implementation Progress & Checklist

- [x] **Phase 1**: Architecture, Data Contracts & Design System.
- [x] **Phase 2**: Classification Evaluation Increment (leakage-safe stratified CV, baseline contrast, statistics, calibration).
- [x] **Phase 3**: Robustness & Stress Testing Engine (perturbation profiles, AURDC, retention metrics).
- [x] **Phase 4**: Persistence, Offline Reporting, Security & Packaging:
  - [x] Milestone 4.1: Fitted Model Serialization & Audit Bundle Packaging.
  - [x] Milestone 4.2: Self-Contained Offline HTML Audit Reports.
  - [x] Milestone 4.3: Automated Release Gates & Ed25519 Cryptographic Certificates.
  - [x] Milestone 4.4: Local SQLite Persistence & Run Lifecycle State Machine.
  - [x] Milestone 4.5: Phase 4 Verification, Packaging Hardening, CLI & Benchmarks.
- [x] **Phase 5**: Interactive ML Experimentation Laboratory (FastAPI REST Core, React 18, TypeScript, Vite, Scientific Design System).
- [x] **Phase 6**: System Hardening & End-to-End Verification (Cryptographic trust decoupling, API & filesystem confinement, offline UI hardening, complete test verification).
