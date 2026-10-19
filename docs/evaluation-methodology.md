# MLForge — Scientific Evaluation Methodology

**Document Version:** 2.0.0
**Status:** Implemented & Active Methodology

---

## 1. Core Principles of Rigorous Evaluation

In machine learning engineering, an evaluation is only as trustworthy as the safeguards preventing information contamination, statistical misunderstanding, and false certainty. MLForge enforces five foundational principles:

1. **Strict Data Leakage Prevention:** No transformation, imputation, scaling, or feature encoding may observe any sample from a validation fold or held-out test split.
2. **Mandatory Baseline Contrast:** A complex model cannot be judged in isolation. Every candidate must be evaluated in direct comparison with a deterministic or naive baseline (`DummyClassifier`).
3. **Preservation of Fold Variance:** Aggregate scalar metrics (e.g., mean accuracy = 0.84) conceal critical instabilities. Full fold-level metric distributions, worst-case folds, and variance must be recorded and presented.
4. **Statistical Rigor Over P-Hacking:** Cross-validation folds are non-independent. Standard Student's t-tests underestimate variance. Statistical claims must use appropriate corrections (e.g., Nadeau-Bengio corrected resampled t-test) or explicitly document violations.
5. **Three-State Acceptance Integrity:** Evaluation outcomes are categorized as `Pass`, `Fail`, or `Inconclusive`. Unstable, borderline, or under-powered evaluations must be flagged as `Inconclusive`.

---

## 2. Classification Evaluation Architecture

MLForge implements rigorous classification evaluation using contrasting estimators on tabular datasets:

### A. Evaluated Estimators
1. **Baseline Model (`DummyClassifier`):**
   - Strategy: `prior` (predicts most frequent class distribution or dominant class).
   - Serves as the zero-skill anchor. Any candidate model failing to statistically outperform the dummy baseline must be immediately rejected.
2. **Candidate Model (`LogisticRegression`):**
   - Configurable solver (e.g. `lbfgs`), penalty (`l2`), inverse regularization strength `C`, and maximum iterations.
   - Outputs calibrated class probabilities via sigmoid/multinomial softmax.

---

## 3. Evaluation Protocol & Splitting Strategy

### A. Cross-Validation Configuration
- **Default Protocol:** Stratified $K$-Fold Cross-Validation ($K = 5$ or $K = 10$).
- **Seed Control:** Evaluation configs enforce an explicit `random_state` (default: 42) for deterministic splitting.
- **Precondition Checks:**
  - Before splitting, the engine verifies that every class $c \in C$ has a sample count $N_c \ge K$.
  - If $N_c < K$ for any class, the engine raises an `InsufficientClassInstancesError` and refuses to silently drop rows or produce corrupt folds.
  - If stratification is impossible due to severe rarity, the user must explicitly choose an alternative protocol (e.g. Repeated Stratified Train/Test Split or Group/Time-based Split) with documented limitations.

### B. Leakage-Free Pipeline Architecture
To eliminate data snooping and data leakage:

```
Full Dataset (X, y)
        │
        ▼ (Split via StratifiedKFold with fixed random_state)
   ┌─────────────────────────────────────────────────────────┐
   │ Fold k of K                                             │
   │                                                         │
   │  Training Set (X_train_k, y_train_k)                    │
   │          │                                              │
   │          ▼                                              │
   │   ┌──────────────────────────────────────────────────┐  │
   │   │ sklearn.pipeline.Pipeline                        │  │
   │   │  1. SimpleImputer (median / most_frequent)       │  │
   │   │     --> .fit(X_train_k)                          │  │
   │   │  2. StandardScaler / OneHotEncoder               │  │
   │   │     --> .fit(X_train_k)                          │  │
   │   │  3. Estimator (LogisticRegression / Dummy)       │  │
   │   │     --> .fit(X_train_k, y_train_k)               │  │
   │   └──────────────────────────────────────────────────┘  │
   │                                                         │
   │  Validation Set (X_val_k, y_val_k)                      │
   │          │                                              │
   │          ▼                                              │
   │   Pipeline.predict(X_val_k) & .predict_proba(X_val_k)   │
   │   (Strictly uses fitted parameters from X_train_k)      │
   └─────────────────────────────────────────────────────────┘
```

**Leakage Audit Checklist:**
- Imputation statistics (means, medians, modes) are computed **strictly** on `X_train_k`.
- Standardization parameters ($\mu, \sigma$) are computed **strictly** on `X_train_k`.
- Categorical vocabularies are fitted **strictly** on `X_train_k`. Unseen categories in `X_val_k` are encoded via `handle_unknown='ignore'`.
- Target values $y$ are never exposed to any feature transformer.

---

## 4. Evaluated Metric Suite & Edge-Case Handling

### A. Metric Definitions

| Metric | Formula / Description | Edge-Case Safeguard |
| :--- | :--- | :--- |
| **Accuracy** | $\frac{\text{Correct Predictions}}{\text{Total Predictions}}$ | Misleading on imbalanced classes; never used as single acceptance criterion. |
| **Balanced Accuracy** | $\frac{1}{\|C\|} \sum_{c \in C} \text{Recall}_c$ | Sensitive to per-class performance; prevents majority class gaming. |
| **Macro Precision** | $\frac{1}{|C|} \sum_{c \in C} \frac{TP_c}{TP_c + FP_c}$ | Zero division handled: when $TP_c + FP_c = 0$, set to `0.0` with documented warning. |
| **Macro Recall** | $\frac{1}{|C|} \sum_{c \in C} \frac{TP_c}{TP_c + FN_c}$ | Zero division handled: when $TP_c + FN_c = 0$, set to `0.0` with documented warning. |
| **Macro F1** | $\frac{1}{|C|} \sum_{c \in C} 2 \cdot \frac{\text{Prec}_c \cdot \text{Rec}_c}{\text{Prec}_c + \text{Rec}_c}$ | Harmonic mean; zero if both precision and recall are zero. |
| **Confusion Matrix** | Contingency table of true vs predicted labels | Fixed class indexing across all folds even if a fold validation set lacks a class. |

### B. Probability-Based Metrics
When the estimator exposes `.predict_proba()`:

1. **Log Loss (Cross-Entropy Loss):**
   $$\mathcal{L}_{\log} = - \frac{1}{N} \sum_{i=1}^N \sum_{c \in C} y_{ic} \ln(p_{ic})$$
   *Safeguard:* Probabilities clipped to $[\epsilon, 1 - \epsilon]$ where $\epsilon = 10^{-15}$ to prevent $\ln(0)$ infinity errors.
2. **Brier Score (Multi-class / Binary):**
   $$\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{c \in C} (p_{ic} - y_{ic})^2$$
   Measures calibration and refinement simultaneously. For binary classification: $\text{BS} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2 \in [0.0, 1.0]$.
3. **ROC-AUC (Receiver Operating Characteristic):**
   - Binary: Standard area under True Positive Rate vs False Positive Rate.
   - Multiclass: One-vs-Rest (OvR) macro-averaged ROC-AUC.
   *Safeguard:* If a validation fold contains only 1 unique class (e.g. extreme imbalance), ROC-AUC is undefined for that fold. The engine explicitly flags `fold_metric_undefined` rather than crashing or inventing 0.5.
4. **PR-AUC (Precision-Recall AUC / Average Precision):**
   Crucial for imbalanced tabular targets where ROC-AUC gives an overly optimistic assessment.
5. **Expected Calibration Error (ECE) & Reliability Diagrams:**
   Partitions predictions into $M$ confidence bins (default $M=10$) over $[0.0, 1.0]$:
   $$\text{ECE} = \sum_{m: |B_m| > 0} \frac{|B_m|}{N} |\bar{y}_m - \bar{p}_m|$$
   - **Bin Edges & Boundary Conventions:** Equal-width bins are defined by edges $e_m = m / M$. For $m < M-1$, bin $B_m$ spans $[e_m, e_{m+1})$. The final bin $B_{M-1}$ spans $[e_{M-1}, e_M]$ (inclusive of right endpoint $1.0$). Probabilities of exactly $0.0$ fall in bin 0, and $1.0$ fall in bin $M-1$.
   - **Maximum Calibration Error (MCE):** $\text{MCE} = \max_{m: |B_m| > 0} |\bar{y}_m - \bar{p}_m|$.
   - **Empty Bins:** Bins with $|B_m| = 0$ have `is_empty = True`, `mean_predicted_prob = None`, and `observed_positive_rate = None`. Empty bins contribute zero to ECE because their sample weight $|B_m| / N$ is zero.
   - **Quantile Binning:** Optional equal-frequency binning using empirical quantiles, with automatic deduplication for tied probability distributions.
   - **Ranking vs Calibration:** Preserves explicit distinction: a model with ROC-AUC = 1.0 can still have severe miscalibration (e.g., predicted probabilities compressed near 0.51 or 0.99).
   - **Leakage Prevention:** Calibration metrics are evaluated strictly on out-of-fold validation sets. Calibration fitting (e.g. Platt scaling, isotonic regression) is never performed on the evaluated data.
   - **Methodological Caveat:** ECE is bin-count and sample-size dependent; it is not an unbiased universal metric.

### C. Classification Threshold Analysis
Binary classification decisions depend critically on the choice of decision cutoff $\theta \in [0.0, 1.0]$:
$$\hat{y}_i = \mathbb{I}(p_i \ge \theta)$$

For each evaluated threshold $\theta$, the engine derives exact contingency counts and metrics:
- **True Positives ($TP$), False Positives ($FP$), True Negatives ($TN$), False Negatives ($FN$):** Conservation invariant $TP + FP + TN + FN = N$ is strictly verified.
- **Predicted Positive Rate:** $(TP + FP) / N$.
- **Precision / PPV:** $\frac{TP}{TP + FP}$. If $TP + FP = 0$, precision is logically undefined (`None` with explicit reason; never coerced to 0.0).
- **Recall / Sensitivity / TPR:** $\frac{TP}{TP + FN}$. If $TP + FN = 0$ (no actual positives), recall is undefined (`None` with explicit reason).
- **Specificity / TNR:** $\frac{TN}{TN + FP}$. If $TN + FP = 0$ (no actual negatives), specificity is undefined (`None` with explicit reason).
- **F1 Score:** $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$. Undefined if precision or recall is undefined; $0.0$ if both are $0.0$.
- **Balanced Accuracy:** $\frac{\text{Recall} + \text{Specificity}}{2}$. Undefined if recall or specificity is undefined.
- **Boundary Cutoffs:** At $\theta = 0.0$, all observations are predicted positive. At $\theta = 1.0$, only observations with $p_i = 1.0$ are predicted positive.
- **No Automated "Optimal" Threshold:** Threshold analysis is descriptive. Selecting operational cutoffs requires explicit cost-utility trade-offs and independent validation data.

### D. Error Diagnostics & Subgroup Breakdowns
- **Fold-Level Error Diagnostics:** Tracks total misclassifications, error rates, false positives, and false negatives per cross-validation fold.
- **Aggregated Out-of-Fold Failures:** Pools non-overlapping out-of-fold predictions to evaluate dataset-level failure distributions without leakage.
- **Non-Target Subgroup Diagnostics:** Evaluates error rates and classification metrics across categories of explicitly supplied metadata features.
  - *Support Floor:* Subgroups with sample size below minimum support ($N_{\text{sub}} < 5$) are marked `is_usable = False` with metrics withheld to prevent small-sample distortion.
  - *Single-Class Subgroups:* Subgroups containing only one class label report accuracy, but binary precision/recall are flagged as undefined with documented reasons.
  - *Missing Attribute Handling:* Null or missing values in metadata are grouped under a distinct `"<MISSING>"` category rather than silently discarded.
  - *Descriptive Non-Fairness Claim:* Subgroup diagnostics provide descriptive performance breakdowns; they do not certify legal compliance or algorithmic fairness without domain-specific policies.

---

## 5. Statistical Comparison Methodology

### A. The Flaw of Standard Paired t-tests on CV Folds
In standard $K$-fold cross-validation, the training sets of any two folds overlap by $\frac{K-2}{K-1}$ (e.g., 75% overlap in 5-fold CV). This violates the independence assumption of the Student's paired $t$-test, artificially inflating the test statistic and causing an elevated Type I error rate (false discovery of statistical superiority). Similarly, standard non-parametric tests like Wilcoxon assume independent observations; applying Wilcoxon to CV folds does **not** eliminate this dependence.

### B. Paired Comparison Requirements & Integrity Guards
To ensure comparisons represent authentic, controlled contrasts:
1. **Compatible Pairing Contract:** Two model evaluations can be paired only when they share identical:
   - Dataset cryptographic identity (`checksum_sha256`).
   - Target column name, data type, and class label ordering.
   - Cross-validation protocol (`split_strategy` must be `stratified_k_fold` or `k_fold`, `num_folds` matching).
   - Per-fold partition boundaries: both `train_indices_count` / `val_indices_count` matching and identical cryptographic partition fingerprints (`partition_fingerprint`) verifying identical sample index sets.
2. **Domain-Specific Enforcement:** If any pairing condition fails (such as duplicate fold IDs, mismatched fold index sets, partition fingerprint mismatches, or unsupported split strategies), the engine raises an `IncompatibleEvaluationPairError` rather than manufacturing a false comparison or aligning by list position alone.
3. **Self-Comparison Safeguard:** Comparing an evaluation object against itself is rejected with `IncompatibleEvaluationPairError`. Comparing distinct models with identical configurations and metrics emits a prominent advisory notice.

### C. Raw Delta Sign Convention & Metric Directions
MLForge enforces a single, immutable difference convention across all metrics:
$$\text{raw\_delta}_i = \text{candidate\_metric}_i - \text{baseline\_metric}_i$$

The raw delta is **never inverted or reversed** for loss metrics. Instead, metric definitions explicitly carry their improvement direction:
- **Higher-is-Better (`HIGHER_IS_BETTER`):** `accuracy`, `balanced_accuracy`, `macro_precision`, `macro_recall`, `macro_f1`, `roc_auc`. A positive raw delta indicates candidate improvement.
- **Lower-is-Better (`LOWER_IS_BETTER`):** `log_loss`, `brier_score`. A negative raw delta indicates candidate improvement.

### D. Undefined and Non-Finite Metrics in Paired Comparisons
- Undefined, NaN, positive infinite, and negative infinite metric values are rejected as unusable differences. They are **never** silently imputed with zero or mean values.
- If either model has an undefined or non-finite metric on a fold, that fold difference is preserved as `None` with an explicit diagnostic reason.
- If usable folds $< 2$ or fewer than total expected folds, statistical tests return `status = TestStatus.UNAVAILABLE` with an explicit diagnostic reason.
- Under no circumstances will a statistical test return `status = TestStatus.COMPLETED` with a NaN or infinite statistic or $p$-value.
- Metric-level isolation: An undefined secondary metric does not invalidate independent scalar comparisons (e.g., `accuracy` comparison remains fully valid even if `macro_f1` is unavailable on baseline).

### E. Nadeau and Bengio (2003) Corrected Resampled t-test
For paired differences $d_i = \text{candidate}_i - \text{baseline}_i$ across $m$ folds with sample variance $s_d^2 = \frac{1}{m-1} \sum_{i=1}^m (d_i - \bar{d})^2$, MLForge adjusts the standard error using the test-to-train sample size ratio:
$$SE_{\text{corr}} = \sqrt{\left(\frac{1}{m} + \frac{n_{\text{test}}}{n_{\text{train}}}\right) s_d^2}$$

The test statistic and two-sided $p$-value are:
$$t = \frac{\bar{d}}{SE_{\text{corr}}}, \quad df = m - 1, \quad p = 2 \cdot \left(1 - F_{t, df}(|t|)\right)$$

**Reference:** Nadeau, C., & Bengio, Y. (2003). *Inference for the Generalization Error*. Machine Learning, 52(3), 239–281.

**Assumptions and Protocol Boundaries:**
1. **Protocol Applicability:** Parameterized for single $K$-fold cross-validation (`stratified_k_fold`, `k_fold`) where validation sets partition the dataset without replacement and training sets overlap by $\frac{K-2}{K-1}$. Unsupported designs (e.g. rolling-window or time-series splits) return `UNAVAILABLE`.
2. **Normality:** Assumes paired differences $d_i$ are approximately normally distributed.
3. **Conservative Power on 5-Fold CV:** For single 5-fold cross-validation ($m=5$), $df = 4$ yields conservative statistical power ($t_{\text{crit}} \approx 2.776$ at $\alpha = 0.05$).
4. **Zero Variance:**
   - If $\bar{d} = 0$ and $s_d^2 = 0$: Returns $t = 0.0, p = 1.0, \text{statistically\_significant} = \text{False}$ with an explicit warning note.
   - If $\bar{d} \ne 0$ and $s_d^2 = 0$: Returns `UNAVAILABLE` because standard error evaluates to zero.
5. **Invalid Inputs:** If $n_{\text{test}} \le 0$, $n_{\text{train}} \le 0$, or any intermediate value evaluates to NaN or infinity, returns `UNAVAILABLE`.

### F. Wilcoxon Signed-Rank Test (Supplementary Diagnostic)
MLForge provides the Wilcoxon signed-rank test as a non-parametric diagnostic evaluating median paired differences:
- **Zero Differences:** Handled via `zero_method="wilcox"` (discarded from ranking). If all differences are zero, returns statistic $= 0.0, p = 1.0$.
- **Tied Differences:** Handled via average ranks.
- **Small Sample & Power Floor:** For an exact two-sided Wilcoxon signed-rank test at $\alpha = 0.05$, a sample size $n \le 5$ **cannot mathematically achieve $p < 0.05$** under any circumstances (minimum attainable $p$-value at $n=5$ is $2 \cdot (1/2)^5 = 0.0625$). If non-zero sample size $n < 6$, the engine emits a prominent statistical power warning.
- **Critical Limitation:** Standard cross-validation training set overlap violates the i.i.d. assumption. Wilcoxon does **not** adjust for fold correlation.

### G. Critical Statistical Caveats
1. **Non-Significance is Not Equivalence:** A non-significant result ($p \ge 0.05$) indicates insufficient evidence to reject the null hypothesis; it does **not** prove that two models perform identically or that they are equivalent.
2. **Statistical Significance is Not Practical Importance:** A statistically significant difference ($p < 0.001$) on a large sample may represent a trivial accuracy delta (e.g. $+0.001$), which may be practically meaningless.
3. **Statistical Tests Do Not Dictate Acceptance:** Statistical comparison provides evidence of difference, but automated deployment decisions (Milestone 2.5) require multi-criteria acceptance policies incorporating thresholds, uncertainty bounds, and domain constraints.

---

## 6. Distinguishing Model Evaluation from Model Selection

- **Model Evaluation (Phase 1 & 2 Focus):** Provides an unbiased estimate of the generalization performance of a fixed model pipeline on unseen data.
- **Model Selection & Hyperparameter Tuning:** Searching for optimal hyperparameters on the same cross-validation splits used to report final performance introduces *optimization bias* (peeking leakage).
- **Nested Cross-Validation Requirement:** If hyperparameter search is requested in future phases, the engine will strictly mandate **nested cross-validation** (inner loop for parameter selection, outer loop for unbiased evaluation) or an independent, untouchable hold-out test set.

---

## 7. Acceptance Policy & Evidence Verification

Deployment decisions are evaluated against an explicit, versioned Acceptance Policy (`AcceptancePolicy`) executed by `evaluate_acceptance_policy`.

### A. Three-State Decision Logic
The evaluation engine returns exactly one of three mutually exclusive outcomes (`AcceptanceOutcome`):

- **`PASS`**:
  - Every required (`BLOCKER`) criterion is conclusively satisfied by observed evaluation evidence.
  - Required evidence is present, valid, finite, and well-defined.
  - No unresolved condition prevents a defensible positive conclusion.
  - Advisory (`ADVISORY`) criterion violations produce warnings but do not block a `PASS`.
- **`FAIL`**:
  - At least one required (`BLOCKER`) criterion is conclusively violated using valid evaluation evidence.
  - Under default failure precedence (`fail_precedence=True`), a conclusive failure produces `FAIL` even if other independent criteria are missing or inconclusive.
- **`INCONCLUSIVE`**:
  - Required evidence is missing, undefined (`is_undefined=True`), non-finite (NaN or Inf), or incompatible.
  - Required statistical comparison or baseline contrast is unavailable.
  - Evaluation compatibility or sample-size floor constraints prevent a defensible determination.
  - No conclusive required criterion failure takes precedence.

> **Integrity Rule:** An `INCONCLUSIVE` verdict must never be silently converted to `PASS` or `FAIL`. It requires the operator to gather additional data, refine features, resolve evaluation compatibility, or adjust policy thresholds.

### B. Precedence and Edge Cases
1. **Configurable Precedence:**
   - **`fail_precedence=True` (Default):** If any blocker criterion is `VIOLATED`, the decision is `FAIL`. Conclusive failure takes precedence over missing evidence or inconclusive secondary checks.
   - **`fail_precedence=False`:** If any blocker criterion is `INCONCLUSIVE`, the decision is `INCONCLUSIVE`, even if another blocker criterion is `VIOLATED`.
2. **Advisory Rule Isolation:** Criteria configured with `severity="ADVISORY"` cannot alter the final decision. Violated advisory rules populate `advisory_warnings` with human-readable diagnostic messages.
3. **Statistical Non-Significance is Not Failure:** A non-significant p-value ($p \ge \alpha$) simply denotes insufficient evidence to reject $H_0$. It does not imply that the candidate is worse than baseline or that the models are equivalent. A policy must explicitly define whether statistical significance is a required blocker (`statistically_significant` or `statistically_greater_than`).
4. **Missing or Non-Finite Evidence:** `None`, NaN, positive/negative infinity, or division-by-zero flags are never coerced to zero. They evaluate strictly to `INCONCLUSIVE` with structured reason codes.

### C. Rule Types and Comparison Operators
The policy supports six rule categories evaluated deterministically against underlying evidence:
- **`candidate_metric`**: Evaluates aggregate candidate metrics (e.g. `mean`, `min`, `max`, `std_dev`). Supports operators: `>=`, `<=`, `>`, `<`, `==`, `!=`.
- **`calibration`**: Evaluates calibration diagnostics (e.g. `brier_score`, `ece`, `log_loss`). Requires explicit numerical threshold. Supports operators: `<=`, `<`, `>=`, `>`, `==`, `!=`.
- **`baseline_contrast`**: Evaluates signed performance delta ($\Delta = \text{candidate} - \text{baseline}$).
  - Supported scalar delta operators: `>=`, `>`, `<=`, `<`, `==`, `!=` (where threshold is a signed delta; for higher-is-better, improvement is $\Delta \ge \text{threshold} > 0$; for lower-is-better, improvement is $\Delta \le \text{threshold} < 0$).
  - Supported confidence-interval operator: `ci_disjoint_superior` (verifies that candidate's 95% CI does not overlap and is strictly superior to baseline's 95% CI: `cand_ci_lower > base_ci_upper` for higher-is-better, or `cand_ci_upper < base_ci_lower` for lower-is-better).
- **`statistical_test`**: Evaluates statistical comparison hypothesis test evidence. Supports operators: `statistically_greater_than` ($\Delta > 0$ and $p < \alpha$), `statistically_less_than` ($\Delta < 0$ and $p < \alpha$), `statistically_significant` ($p < \alpha$).
- **`data_quality`**: Evaluates evaluation stability and completeness (`max_undefined_folds`, `max_fold_std`). Requires explicit numerical threshold.
- **`robustness` (Milestone 3.3)**: Evaluates controlled stress-testing resilience and degradation boundaries. Supports aggregate AURDC (`mean_aurdc`), curve-level AURDC (`<type>_aurdc`, `aurdc:<type>`), maximum severity retention (`<type>_max_retention`), maximum severity performance drop (`<type>_max_drop_pct`), and severity-specific metrics (`<type>:<severity>:drop_pct`, `<type>:<severity>:retention`). Supports operators: `<=`, `<`, `>=`, `>`, `==`, `!=`. If candidate robustness evidence, curves, or points are missing, non-finite, or unusable, the criterion evaluates strictly to `INCONCLUSIVE` without false passes.

### D. Cryptographic Decision Signature
Every `AcceptanceDecision` includes a deterministic SHA-256 `decision_signature` computed over canonical JSON representation of:
- Policy ID and version.
- Dataset checksum (`checksum_sha256`).
- Candidate and baseline cross-validation partition fingerprints.
- Final acceptance outcome (`PASS`, `FAIL`, `INCONCLUSIVE`).
- Canonical list of individual criterion evaluations sorted by `rule_id` (including rule ID, metric, operator, threshold, observed value, criterion outcome, and reason code).

> **Identity vs Evidence Integrity:** The ephemeral `decision_id` (which may contain a random UUID for run traceability) is intentionally excluded from the canonical signature payload. This guarantees that identical policy configuration and identical evaluation evidence produce bit-for-bit identical signatures. A SHA-256 digest certifies reproducibility and evidence integrity; it does not authenticate the external truth of the original raw dataset.

---

## 8. Local Artifact Serialization & Integrity Verification

To preserve auditable, reproducible records of evaluation evidence and acceptance decisions, MLForge provides local artifact serialization (`mlforge/application/artifacts.py`).

### A. Storage Layout
Artifacts are stored in run-oriented directories under `artifacts/runs/<run_id>/`:
```
artifacts/runs/<run_id>/
├── evaluation.json     # Serialized EvaluationResult (folds, metrics, diagnostics, robustness curves)
├── policy.json         # Serialized AcceptancePolicy (rules, thresholds, version)
├── decision.json       # Serialized AcceptanceDecision (verdict, criteria, warnings)
└── manifest.json       # ArtifactManifest (schema version, timestamps, SHA-256 checksums, has_robustness)
```

### B. Serialization Guarantees
1. **Local-First & Zero Dependencies:** Stored strictly on the local filesystem without databases, cloud services, or external APIs.
2. **Run-ID Path Confinement:** Run identifiers are validated to reject absolute paths, null bytes, empty strings, path separators (`/`, `\\`, `:`), and directory traversal sequences (`..`). Target directories are verified to reside strictly within the resolved artifact root.
3. **Atomic Writes:** All JSON files are written to a temporary file (`.tmp_<pid>`) in the destination directory, flushed with `os.fsync`, and atomically replaced via `os.replace`. This prevents corrupt or partial artifacts on crash or write failure.
4. **Strict Finite Numbers:** Serialization uses `allow_nan=False` and recursive float validation. Any NaN, positive infinity, or negative infinity raises `ArtifactSerializationError`.
5. **Preservation of `None` and Undefined Reasons:** Missing values and undefined diagnostic explanations are serialized faithfully as JSON `null` and string fields without numerical coercion.
6. **No Data Leakage:** Raw tabular datasets, credentials, and environment variables are strictly excluded from serialized artifacts.
7. **Overwrite Protection:** Attempts to overwrite an existing artifact directory with a different `run_id` raise `ArtifactError`.

### C. Integrity and Versioning
- **Schema Version:** Current artifact format adheres to schema `1.0.0`. Loading artifacts with incompatible major versions raises `ArtifactLoadError`.
- **Byte-Level SHA-256 Verification:** `manifest.json` records SHA-256 checksums for each file. During `load_run_artifacts()`, every file's byte contents are verified against its recorded digest. Tampered or modified files trigger immediate integrity validation failure.
- **Scope of Checksums:** Digest verification certifies byte-level storage fidelity; it does not claim dataset authenticity or model optimality.

---

## 9. Controlled Robustness & Stress Testing Semantics (Phase 3)

A high validation score under ideal split conditions does not certify that a model is reliable in real-world deployment. MLForge evaluates model robustness under controlled, reproducible feature corruptions without model refitting.

### A. Core Robustness Dimensions (Milestone 3.1)

1. **Gaussian Noise Injection (`gaussian_noise`):**
   - **Perturbation Model:** Additive zero-mean Gaussian noise $\mathcal{N}(0, (\text{severity} \cdot \sigma_j)^2)$ added to continuous numerical features, where $\sigma_j$ is the feature standard deviation estimated from the unperturbed data.
   - **Valid Input Conditions:** Continuous numerical features with non-zero variance ($\sigma_j > 0$). Constant or categorical columns are excluded from noise injection with explicit warnings.
   - **Evaluation Procedure:** The candidate estimator / fold pipeline (already fitted strictly on unperturbed training partitions) is evaluated against validation sets perturbed across severity levels $\sigma \ge 0.0$ (e.g. $0.05, 0.10, 0.25$).
   - **Baseline:** Unperturbed validation score ($\text{severity} = 0.0$).
   - **Metrics & Evidence:** Perturbed metric score, retained ratio, and performance drop percentage. For higher-is-better metrics, drop is $(\text{clean} - \text{perturbed}) / \text{clean} \times 100\%$; for lower-is-better metrics, drop is $(\text{perturbed} - \text{clean}) / \text{clean} \times 100\%$ and retained ratio is $\text{clean} / \text{perturbed}$.
   - **Handling of Undefined Evidence:** If the perturbed score or clean score is undefined (e.g. zero predicted positives for precision) or zero baseline, retention ratios are marked `None` with diagnostic reason codes (`METRIC_UNDEFINED`, `ZERO_BASELINE_SCORE`).

2. **Missingness Injection (`missingness_injection`):**
   - **Perturbation Model:** Missing Completely At Random (MCAR) feature masking where each value has probability $p = \text{severity}$ of being replaced with `NaN` (numerical) or `None` (categorical).
   - **Valid Input Conditions:** All feature columns. The evaluated pipeline must include an imputation step (e.g., `SimpleImputer`) to support missing values during prediction.
   - **Evaluation Procedure:** Evaluates performance degradation across missingness fractions $p \in [0.0, 1.0]$ without retraining.
   - **Baseline:** Unperturbed validation score ($p = 0.0$).
   - **Limitations:** MCAR models uniform sensor/transmission failure; it does not model non-random missingness (MNAR) or target-correlated missingness.

3. **Category Dropout / Corruption (`category_dropout`):**
   - **Perturbation Model:** Categorical values are randomly replaced with an out-of-vocabulary token (`'__UNSEEN__'`) with probability $p = \text{severity} \in [0.0, 1.0]$. Supports both object/string columns and native `pd.CategoricalDtype` columns without dtype mutation. If the unseen token collides with existing categories, it is automatically adapted to a genuinely unseen token (e.g. `'__UNSEEN___1'`) with an advisory warning.
   - **Valid Input Conditions:** Categorical / string features. The evaluated pipeline encoder must handle unknown categories gracefully (e.g., `handle_unknown='ignore'`).
   - **Evaluation Procedure:** Evaluates encoder resilience against categorical drift and unexpected category levels.
   - **Baseline:** Unperturbed validation score ($p = 0.0$).

4. **Multi-Feature Compound Perturbations (`compound`) (Milestone 3.2):**
   - **Perturbation Model:** Simultaneous multi-modal corruption applying Gaussian noise, category corruption, and missingness injection across feature types within a single evaluation step.
   - **Canonical Deterministic Execution Order:**
     1. `GAUSSIAN_NOISE`: Continuous numeric features are corrupted with zero-mean Gaussian noise scaled to feature standard deviation.
     2. `CATEGORY_DROPOUT`: Categorical features are injected with out-of-vocabulary tokens (`'__UNSEEN__'`).
     3. `MISSINGNESS_INJECTION`: Eligible features are randomly masked with null/NaN values.
     *Rationale:* Corrupting continuous values and categories prior to missingness masking accurately mirrors real-world telemetry degradation, where sensors and upstream pipelines suffer noise and categorical drift before experiencing telemetry loss.
   - **Step Seed Independence:** Each step receives an independent random seed deterministically derived via `(random_state + step_idx * 10007) & 0xFFFFFFFF`, eliminating inter-step pseudo-random correlations.
   - **Validation & Guards:** Nested `compound` steps are explicitly prohibited and rejected with `ValueError` to prevent arbitrary recursion or ambiguous degradation curves. Input DataFrames are strictly protected with defensive deep copies prior to multi-step corruption.

### B. Standard Stress Testing Profiles & Suite (Milestone 3.2)
- **Standard Severity Grid:** $\sigma \in [0.05, 0.10, 0.25, 0.50]$, testing mild ($5\%$), moderate ($10\%$), severe ($25\%$), and extreme ($50\%$) data corruption.
- **Pre-Packaged Profiles:**
  - `create_noise_stress_profile`: Configures numeric Gaussian noise degradation across standard or custom severity levels.
  - `create_missingness_stress_profile`: Configures feature missingness degradation across standard or custom severity levels.
  - `create_category_dropout_stress_profile`: Configures categorical drift degradation across standard or custom severity levels.
  - `create_compound_stress_profile`: Configures multi-modal compound stress testing across standard or custom severity levels.
- **Suite Runner:** `get_standard_stress_suite` bundles standard profiles for applicable column subsets. `evaluate_stress_profile` evaluates a fitted estimator against a profile or suite, calculating clean baseline, per-step degradation, retention metrics, and aggregate AURDC.

### C. Area Under Robustness Degradation Curve (AURDC) & Metric Retention Semantics
- **Definition:** Normalized trapezoidal integration of the retention metric curve $R(s)$ over evaluated severity range $[s_0, s_k]$:
  $$\text{AURDC} = \frac{1}{s_k - s_0} \int_{s_0}^{s_k} R(s)\, ds = \frac{1}{s_k - s_0} \sum_{i=1}^{k} \frac{R(s_{i-1}) + R(s_i)}{2} (s_i - s_{i-1})$$
  where $R(s)$ is the observed `retained_ratio` at severity $s$.
- **Severity Ordering & Quadrature Preconditions:** Severity levels are explicitly sorted in ascending order prior to numerical quadrature to guarantee positive interval widths ($s_i > s_{i-1}$). Duplicate severity levels are strictly rejected (evaluating to `None` with warning) to prevent multivalued ambiguity.
- **Directional Retention Semantics:**
  - *Higher-is-better metrics* (e.g., accuracy, macro F1, ROC AUC):
    $$R(s) = \frac{\text{score}_{\text{perturbed}}}{\text{score}_{\text{clean}}}, \quad \text{drop\_pct} = \frac{\text{score}_{\text{clean}} - \text{score}_{\text{perturbed}}}{\text{score}_{\text{clean}}} \times 100\%$$
  - *Lower-is-better metrics* (e.g., log loss, Brier score):
    $$R(s) = \frac{\text{score}_{\text{clean}}}{\text{score}_{\text{perturbed}}}, \quad \text{drop\_pct} = \frac{\text{score}_{\text{perturbed}} - \text{score}_{\text{clean}}}{\text{score}_{\text{clean}}} \times 100\%$$
    Positive $\text{drop\_pct}$ uniformly indicates performance degradation across both metric directions. If clean loss is 0.0 or negative, retention is flagged as unusable (`ZERO_BASELINE_SCORE` / `NEGATIVE_SCORE_LOSS`). If perturbed loss is 0.0, the relative drop is $-100\%$ but the retention ratio is undefined (`ZERO_PERTURBED_LOSS`).
- **Interpretation:** For standard models degrading monotonically, $\text{AURDC} \in [0.0, 1.0]$. An AURDC of $1.0$ indicates perfect robustness (zero performance loss across all severities). $\text{AURDC} > 1.0$ occurs if perturbation counter-intuitively improves performance.
- **NumPy Compatibility:** Integrates via `np.trapezoid` (NumPy 2.x) falling back to `np.trapz` (NumPy 1.x) defensively.
- **Single Point Degradation:** For single-point curves ($k = 0$), AURDC is defined directly as the retained ratio at that severity level.

### D. Invariant Guarantees & Curve Integrity
- **Non-Mutation Invariant:** The original input DataFrame must never be mutated in-place. Defensive copying (`df.copy(deep=True)`) is strictly enforced before any perturbation.
- **Cross-Process Reproducibility Invariant:** Fixed random seeds combined with SHA-256 canonical column seed derivation generate bit-for-bit identical perturbed datasets and degradation metrics across separate Python processes, independent of Python hash randomization (`PYTHONHASHSEED`).
- **No Refitting Invariant:** Robustness stress-testing evaluates the resilience of the *existing fitted model*. Transformers and estimators are never refitted on perturbed validation partitions, eliminating data leakage.
- **Survivorship Bias Defense:** In degradation curves, if any requested severity level is unusable, non-finite, duplicate, or encounters estimator failure, the aggregate Area Under Degradation Curve is strictly undefined (`None`) with an explicit warning. Partial curves are never interpolated to prevent catastrophic failures from masquerading as high retention.
