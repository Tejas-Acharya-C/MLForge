# MLForge — Testing Strategy & Quality Assurance Matrix

**Document Version:** 2.0.0
**Status:** Implemented & Active QA Matrix

---

## 1. Testing Philosophy & Verification Levels

Because MLForge makes scientific claims and automated model acceptance/rejection decisions, its test suite must verify not only functional correctness, but also statistical validity, numerical stability, and total data isolation.

Testing is structured across four rigorous tiers:
1. **Unit & Mathematical Invariant Tests (pytest):** Fast, deterministic tests validating splitting, metric formulas, leakage prevention, and edge-case handling.
2. **Integration & Lifecycle Tests (pytest + HTTPX):** End-to-end evaluation runs comparing candidate vs baseline, testing cancellation, and validating artifact generation.
3. **Contract & Schema Invariant Tests:** Verification that all JSON outputs match Pydantic schemas without property loss.
4. **Frontend Behavior & Accessibility Tests (Vitest + Playwright):** Verification of URL state synchronization, keyboard navigation, focus management, and chart data fidelity.

---

## 2. Comprehensive Test Matrix

### A. Configuration & Dataset Validation
| Test ID | Objective | Test Input / Condition | Expected Behavior |
| :--- | :--- | :--- | :--- |
| `TC-DS-01` | Valid tabular ingestion | Standard CSV with numeric + categorical features and binary target | Emits valid `DatasetMetadata`, correct checksum, and feature list. |
| `TC-DS-02` | Missing target column | CSV lacking specified target column name | Raises `MismatchedSchemaError` with clear error location. |
| `TC-DS-03` | Constant feature detection | Dataset contains column with only 1 unique value | Flags warning and skips or safely handles column in preprocessing. |
| `TC-DS-04` | Non-tabular / Corrupt file | Binary / truncated file passed as CSV | Emits `DatasetReadError` with recovery guidance. |

### B. Class Imbalance & Insufficient Class Counts
| Test ID | Objective | Test Input / Condition | Expected Behavior |
| :--- | :--- | :--- | :--- |
| `TC-SPLIT-01` | Insufficient class instances | A class has 3 samples, but $K=5$ folds are requested | Raises `InsufficientClassInstancesError`; refuses to silently drop samples. |
| `TC-SPLIT-02` | Stratification preservation | 80/20 imbalanced dataset evaluated under 5-fold Stratified CV | Each fold's train and validation sets maintain ~80/20 class ratio ($\pm 1$ sample). |
| `TC-SPLIT-03` | Single-sample class | Extreme multiclass dataset with a singleton class | Caught before fold generation with diagnostic explanation. |

### C. Leakage-Safe Preprocessing Safeguards
| Test ID | Objective | Test Input / Condition | Expected Behavior |
| :--- | :--- | :--- | :--- |
| `TC-LEAK-01` | Imputer parameter isolation | Train set has mean=10.0; validation set has mean=50.0 | Imputer fitted on train uses mean=10.0 to fill missing validation values. |
| `TC-LEAK-02` | Scaler parameter isolation | Train set has std=2.0; validation set has std=10.0 | Scaler applies train $(\mu_{\text{train}}, \sigma_{\text{train}})$ exclusively. |
| `TC-LEAK-03` | Unseen category handling | Validation fold contains category `'Z'` unseen in training fold | OneHotEncoder handles unknown category gracefully (`all zeros`) without crashing. |
| `TC-LEAK-04` | Zero global fitting | Synthetic test spying on estimator `.fit()` | Verifies estimator `.fit()` is called exactly $K$ times on fold slices, never on full $X$. |

### D. Determinism & Reproducibility
| Test ID | Objective | Test Input / Condition | Expected Behavior |
| :--- | :--- | :--- | :--- |
| `TC-REP-01` | Identical seeds yield identical results | Two independent runs on same dataset with seed `42` | Bit-for-bit identical fold metrics, predictions, and confusion matrices. |
| `TC-REP-02` | Seed alteration yields different splits | Run with seed `42` vs run with seed `99` | Generates distinct fold splits while maintaining overall dataset integrity. |

### E. Undefined Metrics & Numerical Stability
| Test ID | Objective | Test Input / Condition | Expected Behavior |
| :--- | :--- | :--- | :--- |
| `TC-MET-01` | Zero predicted positive samples | Candidate predicts class 0 for all validation rows | Precision flags `is_undefined=True`, returns 0.0 with documented warning. |
| `TC-MET-02` | Single class present in fold | Extreme imbalance where validation fold contains only 1 label | ROC-AUC flags `is_undefined=True` for that fold instead of throwing exception. |
| `TC-MET-03` | Probabilities at boundaries | Predicted probability of exactly 0.0 or 1.0 on wrong label | Log loss clips probabilities to $[\epsilon, 1-\epsilon]$, avoiding `inf` or `NaN`. |

### F. Probability Metric Preconditions & Calibration
| Test ID | Objective | Test Input / Condition | Expected Behavior |
| :--- | :--- | :--- | :--- |
| `TC-CAL-01` | ECE calculation accuracy | Known synthetic probabilities and ground truth | Calculates exact bin counts, mean confidence, and expected ECE value. |
| `TC-CAL-02` | Perfect calibration benchmark | Perfectly calibrated synthetic model | ECE approaches $0.0$ ($\le 0.01$). |
| `TC-CAL-03` | Dummy model calibration | Baseline `DummyClassifier(strategy='prior')` | Emits flat probability equal to prior, correctly classified into single bin. |

### G. Fold Aggregation & Uncertainty Bounds
| Test ID | Objective | Test Input / Condition | Expected Behavior |
| :--- | :--- | :--- | :--- |
| `TC-AGG-01` | Metric aggregation correctness | 5 folds with known scalar F1 scores | Mean, std, median, min, max, and 95% CI match analytical formula. |
| `TC-AGG-02` | Handled undefined metrics in aggregation | 1 fold undefined out of 5 | Summary records `undefined_count=1` and computes stats over valid 4 folds. |
| `TC-AGG-03` | Nadeau-Bengio corrected t-test | Candidate vs Baseline fold F1 differences | Emits corrected $t$-statistic, accounting for $(1/K + n_{\text{test}}/n_{\text{train}})$. |

### H. Run Lifecycle & Error Handling
| Test ID | Objective | Test Input / Condition | Expected Behavior |
| :--- | :--- | :--- | :--- |
| `TC-LIFE-01` | Normal run completion | Valid 5-fold evaluation | Moves `queued` $\to$ `running` $\to$ `completed`. Artifacts registered. |
| `TC-LIFE-02` | Mid-run cancellation | User triggers cancel during fold 3 | Moves to `cancelled`, terminates fold process, marks duration. |
| `TC-LIFE-03` | Exception in fold execution | Estimator throws convergence error or memory failure | Moves to `failed`, stores traceback, prevents partial completion. |

### I. Acceptance Policy Boundary Conditions
| Test ID | Objective | Test Input / Condition | Expected Behavior |
| :--- | :--- | :--- | :--- |
| `TC-ACC-01` | Clean Pass | Candidate exceeds all thresholds with $p < 0.01$ | Emits `AcceptanceOutcome.PASS`. |
| `TC-ACC-02` | Hard Fail | Candidate performs worse than `DummyClassifier` | Emits `AcceptanceOutcome.FAIL`. |
| `TC-ACC-03` | Inconclusive (Confidence Overlap) | Candidate mean exceeds baseline, but 95% CI overlaps | Emits `AcceptanceOutcome.INCONCLUSIVE`; never coerces to Pass. |
| `TC-ACC-04` | Inconclusive (High Fold Variance) | Fold standard deviation exceeds policy tolerance | Emits `AcceptanceOutcome.INCONCLUSIVE`. |

### J. Frontend & Contract Invariants
| Test ID | Objective | Test Input / Condition | Expected Behavior |
| :--- | :--- | :--- | :--- |
| `TC-UI-01` | URL State Persistence | User selects filter `?status=failed&metric=macro_f1` | Reloading page preserves identical table filter and chart selection. |
| `TC-UI-02` | Chart Data Agreement | Rendered Plotly confusion matrix vs API JSON | Values rendered on SVG match exact API counts with zero discrepancy. |
| `TC-UI-03` | Keyboard Navigation | Navigate table and open run drawer using keyboard only | Focus ring is visible; <kbd>Escape</kbd> dismisses drawer cleanly. |

---

## 3. Test Execution Guidelines

- Unit tests must be fast (< 100ms per test).
- Integration tests must use fixed synthetic datasets generated via `sklearn.datasets.make_classification(random_state=42)` to avoid external network or disk dependencies.
- No flaky tests: all random seeds must be explicitly injected.
