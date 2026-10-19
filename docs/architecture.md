# MLForge — Architecture & System Design

**Document Version:** 2.0.0
**Status:** Implemented & Active Specification

---

## 1. System Objective & Philosophy

MLForge is a local-only AI Model Evaluation and Reliability Lab designed for tabular classification and regression. Unlike data-preparation or pipeline tools (such as DataFlow), MLForge is built solely to answer:
1. **Is this model's performance scientifically real and reproducible, or is it an artifact of fold variance, leakage, or lucky splits?**
2. **How does this model fail under edge cases, class imbalance, probability miscalibration, and controlled feature corruptions?**
3. **What empirical evidence justifies accepting, rejecting, or marking an evaluation run inconclusive?**

---

## 2. Dependency Hierarchy & Architectural Principles

MLForge adopts a **modular monolith** architecture enforcing strict unidirectional dependencies:

```
┌────────────────────────────────────────────────────────┐
│             Presentation Layer (Frontend)              │
│       React 18+, TypeScript, Vite, Plotly, Vanilla CSS  │
└───────────────────────────┬────────────────────────────┘
                            │ (HTTP / JSON REST API)
                            ▼
┌────────────────────────────────────────────────────────┐
│             Application & Service Layer                │
│    FastAPI, Pydantic schemas, Run Lifecycle Orchestrator│
└─────────────┬───────────────────────────┬──────────────┘
              │                           │
              ▼                           ▼
┌───────────────────────────┐ ┌──────────────────────────┐
│   Infrastructure Layer    │ │   Evaluation Engine      │
│ SQLAlchemy, SQLite/Postgres│ │   (Core Scientific)      │
│ Local Filesystem Artifacts│ │   Pure Python + NumPy    │
│ (Isolated from DataFlow)  │ │   SciPy + scikit-learn   │
└───────────────────────────┘ └──────────────────────────┘
```

### Inviolable Architectural Rules

1. **Pure Engine Isolation:** The scientific evaluation engine (`mlforge.engine`) has **zero** dependencies on FastAPI, Pydantic HTTP request/response handlers, SQLAlchemy models, database drivers, or frontend libraries. It takes pure Python / NumPy / Pandas data structures and typed configuration dataclasses, returning typed evaluation results.
2. **Deterministic & Leakage-Free Execution:** Preprocessing transformations (scaling, imputation, encoding) must never be fitted globally across a dataset. They exist inside an evaluated fold pipeline and are fitted solely on the training partition of that fold.
3. **Direct Local Execution First:** All evaluation runs execute directly within the local runtime process. No Celery, Redis, Kubernetes, or cloud microservices may be added without an explicit, measured requirement.
4. **Database Isolation:** MLForge maintains its own dedicated database (`mlforge_local.db` or PostgreSQL `mlforge_db`). Under no circumstances will MLForge share database tables or connections with DataFlow.

---

## 3. Module Boundaries & Responsibilities

The codebase is partitioned into distinct subpackages with explicit responsibilities:

### A. Core Foundation (`mlforge.core`)
- **Dataset Contracts & Validation:** Ingests tabular data, validates feature schemas, target data types, detects constant columns, missingness rates, and class distribution counts.
- **Exceptions:** Domain-specific exceptions (`DataLeakageDetectedError`, `InsufficientClassInstancesError`, `MismatchedSchemaError`).

### B. Scientific Evaluation Engine (`mlforge.engine`)
- **Protocol & Splitting (`mlforge.engine.protocol`):** Generates deterministic training and validation splits (e.g. `StratifiedKFold`, repeated CV). Validates fold balance and minimum instances per class.
- **Fold-Local Execution (`mlforge.engine.pipeline`):** Clones estimator and preprocessing transformers per fold. Fits exclusively on the fold training set; predicts on the fold validation set.
- **Metric Computation & Aggregation (`mlforge.engine.metrics`):** Calculates accuracy, balanced accuracy, macro precision, macro recall, macro F1, confusion matrices, and probability metrics (Log Loss, Brier score, ROC-AUC, PR-AUC). Handles undefined metrics explicitly with documented warnings. Aggregates metrics across folds with mean, median, standard deviation, and confidence intervals.
- **Statistical Comparison (`mlforge.engine.statistics`):** Compares candidate models against baselines (`DummyClassifier`) using appropriate statistical tests (paired t-test, Nadeau-Bengio corrected resampled t-test, or Wilcoxon signed-rank test).
- **Calibration & Prediction Reliability (`mlforge.engine.reliability`):** Generates reliability diagrams, Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Brier score decompositions.
- **Error & Subgroup Diagnostics (`mlforge.engine.diagnostics`):** Analyzes prediction confidence vs error distributions, worst-case slices, and performance broken down by categorical feature subgroups.
- **Controlled Robustness Experiments (`mlforge.engine.robustness`):** Evaluates model degradation under controlled feature noise (Gaussian noise, missingness injection, category dropout).
- **Acceptance Policy Evaluator (`mlforge.engine.acceptance`):** Evaluates observed metrics, confidence intervals, and reliability indicators against versioned threshold policies to yield an explicit verdict: `Pass`, `Fail`, or `Inconclusive`.

### C. Application & Lifecycle Layer (`mlforge.application`)
- **Run Lifecycle Manager:** Manages run state machine (`queued` -> `running` -> `completed` | `failed` | `cancelled`). Enforces integrity checks before transitioning to `completed`.
- **Artifact Generator:** Serializes fold predictions, confusion matrices, calibration curves, serialized models, and standalone evidence HTML reports to the local filesystem.

### D. API Layer (`mlforge.api`)
- **FastAPI Endpoints:** HTTP routing, query filtering, request payload validation, HTTP status codes, and error responses.

### E. Application Persistence & Artifact Layer (`mlforge.application.persistence`)
- **Database Repositories:** SQLite persistence for run metadata, configuration snapshots, lifecycle audit trails, and summary metrics.
- **Filesystem Storage:** Directory structure management under `artifacts/` for binary models and JSON evaluation dumps.

---

## 4. Storage Architecture

MLForge employs a dual-storage strategy optimized for local research performance:

```
MLForge Storage
├── Relational Database (SQLite / PostgreSQL)
│   ├── evaluations (id, name, task, target, status, created_at, finished_at)
│   ├── configurations (id, evaluation_id, json_blob, hash)
│   ├── metric_summaries (id, evaluation_id, metric_name, mean, std, ci_low, ci_high)
│   ├── acceptance_decisions (id, evaluation_id, policy_version, outcome, reason)
│   └── artifact_manifest (id, evaluation_id, artifact_type, file_path, checksum)
│
└── Local Filesystem Store (artifacts/)
    └── runs/
        └── {run_id}/
            ├── config.json                 (Verifiable configuration dump)
            ├── fold_results.json           (Full fold-by-fold predictions & metrics)
            ├── diagnostics.json            (Calibration, confusion matrix, subgroups)
            ├── models/
            │   ├── fold_01.joblib          (Fitted pipeline per fold)
            │   └── ...
            ├── acceptance_evidence.json    (Traceable acceptance proof)
            └── report.html                 (Self-contained offline HTML report)
```

---

## 5. Architecture Decision Records (ADRs)

### ADR-001: Independent Scientific Evaluation Engine
- **Context:** Frameworks often tightly couple ML logic with web frameworks (e.g. putting scikit-learn code inside FastAPI routes or Django models). This makes unit testing slow, prevents CLI usage, and risks data leakage.
- **Decision:** The evaluation engine must be a pure Python library. It accepts standard data structures (DataFrames, NumPy arrays, Pydantic/dataclass configs) and returns structured evaluation results.
- **Consequences:** Highly testable in isolation; zero web framework dependencies; enables pure CLI or script execution.

### ADR-002: Direct Local Process Execution vs Distributed Queues
- **Context:** Tabular model evaluation on typical local datasets (thousands to hundreds of thousands of rows) finishes in seconds to minutes. Introducing Celery, Redis, or RabbitMQ adds significant operational overhead, process orchestration failure modes, and dependency complexity.
- **Decision:** Execute evaluation runs in-process or via Python standard library `multiprocessing`/`concurrent.futures` directly on the local machine. Provide explicit timeouts and cancellation flags.
- **Consequences:** Eliminates Redis/Celery operational complexity; zero external process requirements; straightforward local debugging.

### ADR-003: Dual Storage Model (Relational Metadata + Filesystem Artifacts)
- **Context:** Storing large prediction arrays, fold probability matrices, and fitted model pipelines inside a relational database causes database bloat and slow query performance. Conversely, storing everything as raw files makes searching and comparing historical runs cumbersome.
- **Decision:** Store indexed run metadata, status, summary statistics, and acceptance outcomes in a relational database. Store raw fold predictions, diagnostic arrays, serialized pipelines, and HTML reports as structured files on the local filesystem.
- **Consequences:** Fast indexing and filtering; atomic metadata queries; lightweight DB footprint; reproducible artifact archiving.

### ADR-004: Strict Evidence-Based Acceptance Model (Pass / Fail / Inconclusive)
- **Context:** Binary pass/fail decisions often obscure critical uncertainty. If an evaluation run has high fold variance, uncalibrated probabilities, or overlapping confidence intervals with a baseline, it cannot honestly be certified as a "Pass", nor is it necessarily a "Fail".
- **Decision:** Support three explicit acceptance states: `Pass`, `Fail`, and `Inconclusive`. An inconclusive result must never be coerced into Pass or Fail. Each decision must record an immutable trace containing policy version, observed metrics, confidence intervals, and diagnostic flags.
- **Consequences:** Scientifically defensible deployment decisions; prevents deployment of statistically indistinguishable or unstable models.

### ADR-005: Strict Database Isolation from DataFlow
- **Context:** The local workstation may host other projects like DataFlow. Sharing database instances, database names, or ORM migrations risks schema collision and catastrophic data corruption.
- **Decision:** MLForge will use its own distinct SQLite database file (`mlforge_local.db`) or dedicated PostgreSQL database (`mlforge_db`) and independent Alembic migration environment.
- **Consequences:** Complete isolation; zero risk of modifying DataFlow schemas or state.
