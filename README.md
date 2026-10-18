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
- **Node.js**: v20+ (LTS) or v22+
- **Git**: 2.40+

### Environment Configuration
Copy the sample environment file before running any services:
```bash
cp .env.example .env
```

*Note on Database*: In Phase 1 and 2, MLForge utilizes direct local evaluation without requiring an external database. When persistence is introduced in Phase 4, MLForge will maintain its own dedicated database (default SQLite fallback, or PostgreSQL `mlforge_db`) strictly separated from any existing services like DataFlow.

---

## 5. Phase 1 Definition of Done Checklist

- [x] Strict dependency direction established (`Frontend -> API -> Engine`).
- [x] Scientific evaluation methodology formalized (leakage prevention, fold preservation, statistical significance testing).
- [x] Versioned data contracts defined for configs, results, diagnostics, and acceptance.
- [x] Six core frontend screens specified with URL states, error boundaries, and empty states.
- [x] Anti-AI design tokens and engineering manual documented.
- [x] Comprehensive test matrix specified.
- [x] Phase 2 implementation plan detailed into actionable milestones.
- [x] Zero empty module proliferation, premature DB migrations, or pseudo-implementations.
