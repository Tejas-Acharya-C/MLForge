# MLForge — Scientific Evaluation Methodology

**Document Version:** 1.0.0  
**Phase:** 1 (Foundation & Architecture)  
**Status:** Approved Specification  

---

## 1. Core Principles of Rigorous Evaluation

In machine learning engineering, an evaluation is only as trustworthy as the safeguards preventing information contamination, statistical misunderstanding, and false certainty. MLForge enforces five foundational principles:

1. **Strict Data Leakage Prevention:** No transformation, imputation, scaling, or feature encoding may observe any sample from a validation fold or held-out test split.
2. **Mandatory Baseline Contrast:** A complex model cannot be judged in isolation. Every candidate must be evaluated in direct comparison with a deterministic or naive baseline (`DummyClassifier`).
3. **Preservation of Fold Variance:** Aggregate scalar metrics (e.g., mean accuracy = 0.84) conceal critical instabilities. Full fold-level metric distributions, worst-case folds, and variance must be recorded and presented.
4. **Statistical Rigor Over P-Hacking:** Cross-validation folds are non-independent. Standard Student's t-tests underestimate variance. Statistical claims must use appropriate corrections (e.g., Nadeau-Bengio corrected resampled t-test) or explicitly document violations.
5. **Three-State Acceptance Integrity:** Evaluation outcomes are categorized as `Pass`, `Fail`, or `Inconclusive`. Unstable, borderline, or under-powered evaluations must be flagged as `Inconclusive`.

---

## 2. Phase 2 Classification Increment

Phase 2 will implement the first testable classification increment using two contrasting estimators on tabular datasets:

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
| **Macro Precision** | $\frac{1}{\|C\|} \sum_{c \in C} \frac{TP_c}{TP_c + FP_c}$ | Zero division handled: when $TP_c + FP_c = 0$, set to `0.0` with documented warning. |
| **Macro Recall** | $\frac{1}{\|C\|} \sum_{c \in C} \frac{TP_c}{TP_c + FN_c}$ | Zero division handled: when $TP_c + FN_c = 0$, set to `0.0` with documented warning. |
| **Macro F1** | $\frac{1}{\|C\|} \sum_{c \in C} 2 \cdot \frac{\text{Prec}_c \cdot \text{Rec}_c}{\text{Prec}_c + \text{Rec}_c}$ | Harmonic mean; zero if both precision and recall are zero. |
| **Confusion Matrix** | Contingency table of true vs predicted labels | Fixed class indexing across all folds even if a fold validation set lacks a class. |

### B. Probability-Based Metrics
When the estimator exposes `.predict_proba()`:

1. **Log Loss (Cross-Entropy Loss):**
   $$\mathcal{L}_{\log} = - \frac{1}{N} \sum_{i=1}^N \sum_{c \in C} y_{ic} \ln(p_{ic})$$
   *Safeguard:* Probabilities clipped to $[\epsilon, 1 - \epsilon]$ where $\epsilon = 10^{-15}$ to prevent $\ln(0)$ infinity errors.
2. **Brier Score (Multi-class / Binary):**
   $$\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{c \in C} (p_{ic} - y_{ic})^2$$
   Measures calibration and refinement simultaneously.
3. **ROC-AUC (Receiver Operating Characteristic):**
   - Binary: Standard area under True Positive Rate vs False Positive Rate.
   - Multiclass: One-vs-Rest (OvR) macro-averaged ROC-AUC.
   *Safeguard:* If a validation fold contains only 1 unique class (e.g. extreme imbalance), ROC-AUC is undefined for that fold. The engine explicitly flags `fold_metric_undefined` rather than crashing or inventing 0.5.
4. **PR-AUC (Precision-Recall AUC / Average Precision):**
   Crucial for imbalanced tabular targets where ROC-AUC gives an overly optimistic assessment.
5. **Expected Calibration Error (ECE):**
   Partitions predictions into $M$ confidence bins (default $M=10$) and computes weighted absolute difference between confidence and empirical accuracy:
   $$\text{ECE} = \sum_{m=1}^M \frac{\|B_m\|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$

---

## 5. Statistical Comparison Methodology

### A. The Flaw of Standard Paired t-tests on CV Folds
In standard $K$-fold cross-validation, the training sets of any two folds overlap by $\frac{K-2}{K-1}$ (e.g., 75% overlap in 5-fold CV). This violates the independence assumption of the Student's paired $t$-test, artificially inflating the test statistic and causing an elevated Type I error rate (false discovery of statistical superiority).

### B. Corrected Statistical Testing
MLForge employs two statistically sound comparison methods:

1. **Nadeau and Bengio (2003) Corrected Resampled t-test:**
   Adjusts the sample variance by accounting for the proportion of test vs train data:
   $$t_{\text{corrected}} = \frac{\bar{d}}{\sqrt{\left(\frac{1}{K} + \frac{n_{\text{test}}}{n_{\text{train}}}\right) S^2_d}}$$
   where $\bar{d}$ is the mean metric difference between candidate and baseline across folds, and $S^2_d$ is the variance of fold differences.
2. **Wilcoxon Signed-Rank Test (Non-parametric Alternative):**
   Used when normality of metric differences cannot be assumed across repeated cross-validation folds.
3. **Explicit Uncertainty Reporting:**
   Every reported aggregate metric must present:
   - Sample mean ($\bar{x}$) and sample median
   - Standard deviation ($s$) and Interquartile Range (IQR)
   - 95% Confidence Interval ($[\text{CI}_{\text{low}}, \text{CI}_{\text{high}}]$)
   - Minimum and maximum fold values

---

## 6. Distinguishing Model Evaluation from Model Selection

- **Model Evaluation (Phase 1 & 2 Focus):** Provides an unbiased estimate of the generalization performance of a fixed model pipeline on unseen data.
- **Model Selection & Hyperparameter Tuning:** Searching for optimal hyperparameters on the same cross-validation splits used to report final performance introduces *optimization bias* (peeking leakage).
- **Nested Cross-Validation Requirement:** If hyperparameter search is requested in future phases, the engine will strictly mandate **nested cross-validation** (inner loop for parameter selection, outer loop for unbiased evaluation) or an independent, untouchable hold-out test set.

---

## 7. Acceptance Policy & Evidence Verification

Deployment decisions are evaluated against an explicit, versioned Acceptance Policy.

### A. Three-State Decision Logic
- **`Pass`:** The candidate model satisfies all required threshold criteria (e.g. Macro F1 $\ge 0.75$, Balanced Accuracy $\ge 0.70$, statistically significant improvement over `DummyClassifier` with $p < 0.05$, and $\text{ECE} \le 0.15$).
- **`Fail`:** The candidate model explicitly violates one or more hard requirements (e.g. fails to outperform `DummyClassifier`, or worst-case fold accuracy drops below safety floor).
- **`Inconclusive`:** The candidate model shows promising mean metrics, but:
  - The 95% confidence interval overlaps with the baseline, OR
  - Fold variance is too high ($s > \sigma_{\max}$), OR
  - An undefined metric condition occurred on one or more folds, OR
  - The sample size was insufficient to reach statistical power.

> **Integrity Rule:** An `Inconclusive` verdict must never be silently converted to `Pass` or `Fail`. It requires the user to gather more data, refine features, or modify evaluation bounds.
