# MLForge — Frontend Information Architecture & Blueprint

**Document Version:** 2.0.0
**Status:** Implemented & Active Architecture

---

## 1. User Journey & Coherent Lab Workflow

MLForge's interface is structured around a continuous scientific research cycle rather than disjointed dashboard tabs:

```
[ Lab Overview ] ──(New Evaluation)──> [ New Evaluation (Config) ]
       │                                            │
       │ (Inspect existing)                         │ (Execute)
       ▼                                            ▼
[ Evaluation Results ] <─────────────────── [ Run Details ]
       │
       ├─────────────────────────┐
       ▼                         ▼
[ Reliability Analysis ]   [ Acceptance Report ]
```

Users configure an evaluation, observe deterministic execution, evaluate fold-level metrics against baseline anchors, drill into prediction reliability and edge-case errors, and verify cryptographically traceable acceptance decisions.

---

## 2. Screen Specifications

---

### Page 1: Lab Overview (`/`)

#### A. Purpose & Context
Provides researchers with immediate situational awareness over historical and active evaluation runs. Serves as the laboratory launchpad without relying on generic KPI cards or marketing metrics.

#### B. Visual Hierarchy & Major Regions
1. **Header Region:** Laboratory identity ("MLForge Lab"), current environment mode (`local-development`), quick action to "Configure new evaluation".
2. **Filter & Search Toolbar:** Filter by evaluation state (`all`, `completed`, `failed`, `running`), task type (`binary`, `multiclass`), or target dataset. Persisted in URL.
3. **Primary Evaluations Table (Information-Led):**
   - Dense, tabular view with columns: Status indicator, Evaluation Name & ID (monospaced), Target Dataset, Model vs Baseline, Key Metric Summary (Macro F1 with 95% CI), Acceptance Verdict (`Pass` / `Fail` / `Inconclusive`), Execution Timestamp, and Quick Links.
4. **Active Runs Drawer / Sider (Conditional):** If any run is in state `running` or `queued`, an anchored side panel displays real-time progress bar, fold count ($k/K$), and cancel button.
5. **Meaningful First-Run Empty State:** When no evaluations exist, displays an instructional benchmark guide: "No evaluations executed yet. Ingest a tabular dataset or load the sample benchmark to verify baseline separation." Primary button: "Configure first evaluation".

#### C. Primary & Secondary Actions
- **Primary:** "Configure new evaluation" (routes to `/evaluations/new`).
- **Secondary:** "Inspect run" (routes to `/evaluations/{id}`), "View acceptance report" (routes to `/evaluations/{id}/acceptance`), "Cancel execution" (calls `POST /api/v1/evaluations/{id}/cancel`).

#### D. Data Requirements & URL State
- **API Dependencies:** `GET /api/v1/evaluations?page=1&limit=25&state={state}&q={query}`
- **URL Parameters:** `?status=all&page=1&sort=created_at:desc`

#### E. Loading, Empty, & Error States
- **Loading:** Hairline skeleton pulse on table rows (retaining table header layout to prevent content shifts).
- **Empty:** Contextual empty state explaining lack of runs with immediate guidance.
- **Error:** Inline callout with error code, explanation of backend unavailability, and "Retry connection" button.

#### F. Responsive Behavior
Desktop: Full 8-column data grid with monospaced run hashes. Mobile/Tablet: Collapses secondary columns (timestamps, fold details) into expandable drawer rows.

---

### Page 2: New Evaluation (`/evaluations/new`)

#### A. Purpose & Context
An interactive laboratory workbench for specifying dataset inputs, leakage-safe preprocessing pipelines, validation splits, candidate vs baseline hyperparameters, and acceptance policies.

#### B. Visual Hierarchy & Major Regions
1. **Breadcrumb & Header:** `Lab Overview > Configure evaluation`.
2. **Region 1: Dataset & Target Selection:**
   - File selector / path input with instant "Inspect Schema" trigger.
   - Schema Inspector table displaying feature names, types, null counts, and class distribution counts.
   - Target variable picker with immediate validation warning if any class has $< 5$ samples.
3. **Region 2: Model & Preprocessing Pipeline Configuration:**
   - Candidate model specification (`LogisticRegression`: solver, penalty, $C$, max iter).
   - Preprocessing rules (missing value strategy, scaling, categorical encoding).
   - Baseline anchor specification (`DummyClassifier`: strategy `prior`).
4. **Region 3: Validation Protocol & Reproducibility:**
   - Split strategy (`StratifiedKFold`), fold count slider ($2 \le K \le 20$), fixed random seed input (default `42`).
5. **Region 4: Metric Suite & Acceptance Policy:**
   - Checkbox group for evaluation metrics (Accuracy, Balanced Accuracy, Macro F1, Macro Precision, Macro Recall, Log Loss, Brier Score, ECE).
   - Acceptance policy selector (`standard_classification_v1`, `strict_zero_leakage_v1`, `custom`).
6. **Action Footer Bar:** Sticky bottom utility bar with validation summary, estimated execution time, and action button: "Execute evaluation run".

#### C. Primary & Secondary Actions
- **Primary:** "Execute evaluation run" (submits payload to `POST /api/v1/evaluations`, redirects to `/evaluations/{id}/run`).
- **Secondary:** "Load sample classification benchmark", "Reset form", "Validate configuration".

#### D. Data Requirements & URL State
- **API Dependencies:** `POST /api/v1/datasets/inspect`, `GET /api/v1/acceptance-policies`, `POST /api/v1/evaluations`.
- **URL State:** `?preset=binary_churn` (optional pre-fill query).

#### E. Loading, Empty, & Error States
- **Inline Validation:** Instant feedback on field blur (e.g. invalid C parameter $\le 0$, non-existent column name).
- **Error Recovery:** Specific error callout if class count is insufficient: "Class 'fraud' has 3 instances, which is fewer than K=5 folds. Select K=3 or use Repeated Stratified Split."

---

### Page 3: Evaluation Results (`/evaluations/{id}/results`)

#### A. Purpose & Context
The central scientific evaluation screen comparing the candidate estimator against the baseline anchor across all cross-validation folds.

#### B. Visual Hierarchy & Major Regions
1. **Executive Evaluation Header:** Run ID, dataset name, candidate vs baseline badges, overall statistical outcome badge (`Statistically Significant Superiority`, $p < 0.001$).
2. **Region 1: Metric Comparison Grid (Point Estimates + Fold Uncertainty):**
   - Compact comparison table with columns: Metric Name, Baseline Dummy ($\mu \pm \sigma$), Candidate ($\mu \pm \sigma$), 95% Confidence Interval, Absolute Delta ($\Delta$), $p$-value (Nadeau-Bengio corrected), Status.
   - Tabular numerals enabled.
3. **Region 2: Fold-Level Distribution Strip:**
   - Interactive Plotly strip/box plot displaying the metric score for each individual fold ($1 \dots K$). Shows candidate points vs baseline points to immediately reveal fold instability or outliers.
4. **Region 3: Confusion Matrix Analysis:**
   - Normalized and raw contingency matrices side-by-side: Candidate vs Baseline.
   - Interactive hover disclosing exact misclassification counts and false positive/negative percentages.
5. **Region 4: Split & Protocol Metadata Card:**
   - Exact configuration parameters, fold sizes, class balance verification, and random seed.

#### C. Primary & Secondary Actions
- **Primary:** "Proceed to reliability analysis" (routes to `/evaluations/{id}/reliability`).
- **Secondary:** "View acceptance report" (routes to `/evaluations/{id}/acceptance`), "Export evaluation summary (JSON)".

#### D. Data Requirements & URL State
- **API Dependencies:** `GET /api/v1/evaluations/{id}/results`, `GET /api/v1/evaluations/{id}/results/folds`.
- **URL Parameters:** `?metric=macro_f1&view=comparison`

---

### Page 4: Reliability Analysis (`/evaluations/{id}/reliability`)

#### A. Purpose & Context
Investigates how the model behaves beyond aggregate accuracy: probability calibration, high-confidence errors, subgroup disparities, and robustness under controlled noise.

#### B. Visual Hierarchy & Major Regions
1. **Reliability Header:** Reliability score indicator, Expected Calibration Error (ECE) value, Brier Score.
2. **Region 1: Calibration Diagram (Reliability Curve):**
   - Plotly chart: Mean predicted probability vs observed empirical accuracy across 10 bins.
   - Diagonal reference line representing perfect calibration.
   - Gap fill showing under-confidence and over-confidence regions.
3. **Region 2: High-Confidence Failure Analysis:**
   - Table of predictions where model confidence was $\ge 85\%$ but the prediction was wrong.
   - Shows row index, true label, predicted label, predicted probability, and key feature values.
4. **Region 3: Subgroup Slice Diagnostics:**
   - Disaggregated metric table broken down by categorical attributes (e.g. Geography, Age Tier).
   - Highlights worst-performing subgroup and calculates the disparate performance ratio.
5. **Region 4: Controlled Robustness Perturbations (If evaluated):**
   - Line chart depicting performance degradation under incremental Gaussian noise ($0\%, 5\%, 10\%, 25\%$) and feature missingness.

#### C. Primary & Secondary Actions
- **Primary:** "Review acceptance decision" (routes to `/evaluations/{id}/acceptance`).
- **Secondary:** "Toggle calibration bin width (5 / 10 / 20 bins)", "Download failure cases (CSV)".

#### D. Data Requirements & URL State
- **API Dependencies:** `GET /api/v1/evaluations/{id}/diagnostics/calibration`, `GET /api/v1/evaluations/{id}/diagnostics/subgroups`.
- **URL Parameters:** `?slice_feature=geography&calib_bins=10`

---

### Page 5: Run Details (`/evaluations/{id}`)

#### A. Purpose & Context
The auditable operational record of the evaluation run: execution timeline, environment details, lifecycle state machine transitions, exact configuration manifest, and raw artifact downloads.

#### B. Visual Hierarchy & Major Regions
1. **Run Status Banner:** Current state (`completed`, `failed`, `cancelled`), execution duration, compute hostname, Python version, random seed.
2. **Region 1: Execution Timeline & Fold Logs:**
   - Step-by-step audit trail: Dataset loaded $\to$ Split generated $\to$ Fold 1 fitted $\to \dots \to$ Metrics aggregated $\to$ Acceptance evaluated.
   - For failed runs: Expandable traceback box with syntax highlighting and recovery recommendations.
3. **Region 2: Reproducibility Manifest:**
   - Complete JSON dump of the frozen `EvaluationConfig` with one-click "Copy JSON" and "Re-run this evaluation".
   - Dataset SHA256 checksum and package versions (`scikit-learn`, `numpy`, `scipy`).
4. **Region 3: Persisted Artifacts Table:**
   - Table of generated files (`config.json`, `fold_results.json`, `diagnostics.json`, `report.html`, fitted pipelines).
   - File size, SHA256 hash, and direct download links.

#### C. Primary & Secondary Actions
- **Primary:** "Re-run evaluation with same parameters" (redirects to `/evaluations/new?from={id}`).
- **Secondary:** "Download offline HTML report", "Download raw artifact bundle (ZIP)".

#### D. Data Requirements & URL State
- **API Dependencies:** `GET /api/v1/evaluations/{id}`, `GET /api/v1/evaluations/{id}/artifacts`.
- **URL Parameters:** `?tab=artifacts`

---

### Page 6: Acceptance Report (`/evaluations/{id}/acceptance`)

#### A. Purpose & Context
The definitive, evidence-backed evaluation certificate. Presents whether the model passed, failed, or was inconclusive against a formal acceptance policy, along with verifiable cryptographic evidence.

#### B. Visual Hierarchy & Major Regions
1. **Decision Header:**
   - Large, authoritative verdict badge: **`PASS`**, **`FAIL`**, or **`INCONCLUSIVE`**.
   - Applied Policy Name, Policy Version (`v1.0.0`), and SHA256 Decision Hash.
2. **Region 1: Executive Rationale:**
   - Multi-paragraph synthesis explaining *why* the model achieved this verdict based on statistical significance, baseline delta, and reliability constraints.
3. **Region 2: Criteria Evaluation Checklist:**
   - Structured audit table:
     - Rule Description (e.g. "Candidate Macro F1 must exceed 0.75")
     - Metric & Operator (`macro_f1 >= 0.75`)
     - Required Threshold (`0.7500`)
     - Observed Value (`0.8124 [95% CI: 0.7863, 0.8385]`)
     - Outcome Status (`PASSED` with green indicator, `FAILED` with red indicator, `INCONCLUSIVE` with amber indicator)
4. **Region 3: Statistical Evidence & Baseline Contrast:**
   - Paired difference summary, $p$-value, degree of freedom, and confirmation of leakage-free fold isolation.
5. **Region 4: Known Assumptions & Caveats:**
   - Explicitly documented assumptions (e.g., "Assumes tabular feature distribution remains stationary relative to training sample; subgroup sample size for tier 'enterprise' was below 30 samples").

#### C. Primary & Secondary Actions
- **Primary:** "Export standalone HTML report" (downloads `report.html` for offline auditing).
- **Secondary:** "Copy decision verification hash", "Compare with previous run".

#### D. Data Requirements & URL State
- **API Dependencies:** `GET /api/v1/evaluations/{id}/acceptance`, `GET /api/v1/evaluations/{id}/report.html`.
- **URL Parameters:** `?view=printable`
