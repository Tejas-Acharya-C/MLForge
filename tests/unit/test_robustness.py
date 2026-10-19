"""Unit Tests for MLForge Robustness & Stress Testing Engine (Milestone 3.1).

Validates feature perturbation transformers (Gaussian noise, missingness injection,
category dropout), retention metrics, degradation curves, input immutability,
seed determinism, and edge-case handling.
"""

import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from mlforge.core.contracts import (
    MetricDirection,
    PerturbationConfig,
    PerturbationType,
    RobustnessCurve,
    RobustnessEvaluationResult,
    RobustnessPerturbationResult,
)
from mlforge.engine.pipeline import build_fold_pipeline
from mlforge.engine.robustness import (
    STANDARD_SEVERITY_GRID,
    apply_category_dropout,
    apply_compound_perturbation,
    apply_gaussian_noise,
    apply_missingness_injection,
    calculate_aurdc,
    calculate_retention_metrics,
    create_category_dropout_stress_profile,
    create_compound_stress_profile,
    create_missingness_stress_profile,
    create_noise_stress_profile,
    evaluate_model_robustness,
    evaluate_perturbation_step,
    evaluate_robustness_curve,
    evaluate_stress_profile,
    get_standard_stress_suite,
    perturb_dataset,
)


@pytest.fixture
def sample_features_df():
    """Generates a small controlled tabular DataFrame for perturbation testing."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "num_1": [10.0, 20.0, 30.0, 40.0, 50.0] * 20,  # std ~ 14.21
        "num_2": [100.0, 200.0, 300.0, 400.0, 500.0] * 20,
        "cat_1": ["A", "B", "C", "A", "B"] * 20,
        "cat_2": ["X", "Y", "X", "Y", "X"] * 20,
    })


# ============================================================================
# 1. Input Immutability & Defensive Copying Tests
# ============================================================================

def test_original_dataframe_never_mutated_by_gaussian_noise(sample_features_df):
    """Gaussian noise must not modify the caller's original DataFrame in-place."""
    df_copy = sample_features_df.copy(deep=True)
    df_pert, _ = apply_gaussian_noise(sample_features_df, severity=0.25, random_state=42)

    pd.testing.assert_frame_equal(sample_features_df, df_copy)
    assert not np.array_equal(df_pert["num_1"].to_numpy(), sample_features_df["num_1"].to_numpy())


def test_original_dataframe_never_mutated_by_missingness(sample_features_df):
    """Missingness injection must not modify the caller's original DataFrame in-place."""
    df_copy = sample_features_df.copy(deep=True)
    df_pert, _ = apply_missingness_injection(sample_features_df, severity=0.30, random_state=42)

    pd.testing.assert_frame_equal(sample_features_df, df_copy)
    assert sample_features_df["num_1"].isna().sum() == 0
    assert df_pert["num_1"].isna().sum() > 0


def test_original_dataframe_never_mutated_by_category_dropout(sample_features_df):
    """Category dropout must not modify the caller's original DataFrame in-place."""
    df_copy = sample_features_df.copy(deep=True)
    df_pert, _ = apply_category_dropout(sample_features_df, severity=0.25, random_state=42)

    pd.testing.assert_frame_equal(sample_features_df, df_copy)
    assert "__UNSEEN__" not in sample_features_df["cat_1"].values
    assert "__UNSEEN__" in df_pert["cat_1"].values


# ============================================================================
# 2. Determinism & Seed Reproducibility Tests
# ============================================================================

def test_reproducible_gaussian_noise_under_identical_seed(sample_features_df):
    """Two independent Gaussian noise runs with identical random_state produce bit-for-bit identical outputs."""
    df_pert_1, _ = apply_gaussian_noise(sample_features_df, severity=0.15, random_state=123)
    df_pert_2, _ = apply_gaussian_noise(sample_features_df, severity=0.15, random_state=123)

    pd.testing.assert_frame_equal(df_pert_1, df_pert_2)


def test_different_seeds_yield_different_gaussian_noise(sample_features_df):
    """Different seeds produce distinct noise perturbations."""
    df_pert_1, _ = apply_gaussian_noise(sample_features_df, severity=0.15, random_state=123)
    df_pert_2, _ = apply_gaussian_noise(sample_features_df, severity=0.15, random_state=999)

    assert not np.allclose(df_pert_1["num_1"].to_numpy(), df_pert_2["num_1"].to_numpy())


def test_reproducible_missingness_under_identical_seed(sample_features_df):
    """Identical seeds produce identical missingness masks."""
    df_pert_1, _ = apply_missingness_injection(sample_features_df, severity=0.20, random_state=42)
    df_pert_2, _ = apply_missingness_injection(sample_features_df, severity=0.20, random_state=42)

    pd.testing.assert_frame_equal(df_pert_1, df_pert_2)


def test_reproducible_category_dropout_under_identical_seed(sample_features_df):
    """Identical seeds produce identical categorical replacements."""
    df_pert_1, _ = apply_category_dropout(sample_features_df, severity=0.20, random_state=42)
    df_pert_2, _ = apply_category_dropout(sample_features_df, severity=0.20, random_state=42)

    pd.testing.assert_frame_equal(df_pert_1, df_pert_2)


# ============================================================================
# 3. Gaussian Noise Properties & Validation Tests
# ============================================================================

def test_gaussian_noise_zero_severity_is_identity(sample_features_df):
    """severity=0.0 must return identical values without noise."""
    df_pert, warnings = apply_gaussian_noise(sample_features_df, severity=0.0)
    pd.testing.assert_frame_equal(df_pert, sample_features_df)
    assert len(warnings) == 0


def test_gaussian_noise_preserves_existing_nans():
    """Existing NaN entries in features must remain NaN and not receive additive noise."""
    df = pd.DataFrame({"num": [10.0, np.nan, 30.0, np.nan, 50.0] * 10})
    df_pert, _ = apply_gaussian_noise(df, severity=0.20, random_state=42)

    assert df_pert["num"].isna().sum() == 20
    np.testing.assert_array_equal(df["num"].isna(), df_pert["num"].isna())


def test_gaussian_noise_skips_zero_variance_columns():
    """Constant columns with zero variance must be skipped with an advisory warning."""
    df = pd.DataFrame({
        "varying": [1.0, 2.0, 3.0, 4.0, 5.0],
        "constant": [42.0, 42.0, 42.0, 42.0, 42.0],
    })
    df_pert, warnings = apply_gaussian_noise(df, severity=0.10, random_state=42)

    assert any("zero variance" in w for w in warnings)
    np.testing.assert_array_equal(df_pert["constant"], df["constant"])
    assert not np.array_equal(df_pert["varying"], df["varying"])


def test_gaussian_noise_rejects_non_numeric_target_columns():
    """Explicitly requesting Gaussian noise on a string column must raise ValueError."""
    df = pd.DataFrame({"num": [1.0, 2.0], "cat": ["A", "B"]})
    with pytest.raises(ValueError, match="can only be applied to numeric columns"):
        apply_gaussian_noise(df, columns=["cat"], severity=0.10)


def test_gaussian_noise_rejects_invalid_severity(sample_features_df):
    """Negative, NaN, or non-finite severity must raise ValueError."""
    with pytest.raises(ValueError, match="finite non-negative"):
        apply_gaussian_noise(sample_features_df, severity=-0.1)

    with pytest.raises(ValueError, match="finite non-negative"):
        apply_gaussian_noise(sample_features_df, severity=float("nan"))


# ============================================================================
# 4. Missingness Injection Properties & Validation Tests
# ============================================================================

def test_missingness_zero_severity_is_identity(sample_features_df):
    """severity=0.0 injects zero missing values."""
    df_pert, warnings = apply_missingness_injection(sample_features_df, severity=0.0)
    pd.testing.assert_frame_equal(df_pert, sample_features_df)


def test_missingness_injection_approximate_rate():
    """severity=0.30 should replace approximately 30% of entries with null."""
    n = 1000
    df = pd.DataFrame({
        "num": [10.0] * n,
        "cat": ["A"] * n,
    })
    df_pert, _ = apply_missingness_injection(df, severity=0.30, random_state=42)

    num_nulls = df_pert["num"].isna().sum()
    cat_nulls = df_pert["cat"].isna().sum()

    # 30% of 1000 = 300, allowing +/- 40 samples tolerance (binomial std ~14.5)
    assert 250 <= num_nulls <= 350
    assert 250 <= cat_nulls <= 350


def test_missingness_rejects_severity_outside_bounds(sample_features_df):
    """Severity < 0.0 or > 1.0 must raise ValueError."""
    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
        apply_missingness_injection(sample_features_df, severity=-0.05)

    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
        apply_missingness_injection(sample_features_df, severity=1.05)


# ============================================================================
# 5. Category Dropout Properties & Validation Tests
# ============================================================================

def test_category_dropout_zero_severity_is_identity(sample_features_df):
    """severity=0.0 injects no unseen tokens."""
    df_pert, warnings = apply_category_dropout(sample_features_df, severity=0.0)
    pd.testing.assert_frame_equal(df_pert, sample_features_df)


def test_category_dropout_preserves_existing_nulls():
    """Existing null/None entries in categorical features are not overwritten by unseen token."""
    df = pd.DataFrame({"cat": ["A", None, "B", None, "C"] * 20})
    df_pert, _ = apply_category_dropout(df, severity=0.50, random_state=42)

    assert df_pert["cat"].isna().sum() == 40  # Null positions unchanged
    assert "__UNSEEN__" in df_pert["cat"].values


def test_category_dropout_rejects_numeric_columns():
    """Explicitly requesting category dropout on a numeric column raises ValueError."""
    df = pd.DataFrame({"num": [1.0, 2.0], "cat": ["A", "B"]})
    with pytest.raises(ValueError, match="cannot be applied to numeric columns"):
        apply_category_dropout(df, columns=["num"], severity=0.10)


# ============================================================================
# 6. Retention & Performance Drop Metric Calculations
# ============================================================================

def test_retention_metrics_calculation_hand_checked():
    """Hand-check relative drop: clean=0.80, perturbed=0.60 -> ratio=0.75, drop=25%."""
    ratio, drop_pct, usable, reason = calculate_retention_metrics(
        clean_score=0.80,
        perturbed_score=0.60,
    )

    assert usable is True
    assert reason is None
    assert ratio == pytest.approx(0.75, abs=1e-5)
    assert drop_pct == pytest.approx(25.0, abs=1e-5)


def test_retention_metrics_performance_improvement_negative_drop():
    """If perturbed performance is higher, drop percentage is negative (gain)."""
    ratio, drop_pct, usable, _ = calculate_retention_metrics(
        clean_score=0.80,
        perturbed_score=0.88,
    )

    assert usable is True
    assert ratio == pytest.approx(1.10, abs=1e-5)
    assert drop_pct == pytest.approx(-10.0, abs=1e-5)


def test_retention_metrics_zero_clean_score_unusable():
    """Clean score == 0.0 prevents division-by-zero, flagged as unusable without crashing."""
    ratio, drop_pct, usable, reason = calculate_retention_metrics(
        clean_score=0.0,
        perturbed_score=0.5,
    )

    assert usable is False
    assert ratio is None
    assert drop_pct is None
    assert "ZERO_BASELINE_SCORE" in reason


def test_retention_metrics_undefined_scores_unusable():
    """None or NaN scores yield explicit unusable status."""
    ratio, drop_pct, usable, reason = calculate_retention_metrics(
        clean_score=None,
        perturbed_score=0.75,
    )
    assert usable is False
    assert "METRIC_UNDEFINED" in reason

    ratio, drop_pct, usable, reason = calculate_retention_metrics(
        clean_score=0.80,
        perturbed_score=float("nan"),
    )
    assert usable is False
    assert "NON_FINITE_METRIC" in reason


# ============================================================================
# 7. Robustness Evaluation Step & Curve Tests with Estimator
# ============================================================================

@pytest.fixture
def trained_pipeline():
    """Trains a simple pipeline with imputer, scaler, and onehot encoder on synthetic data."""
    X_train = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 5,
        "cat": ["tier_a", "tier_b", "tier_a", "tier_b", "tier_a"] * 10,
    })
    y_train = pd.Series([0, 1, 0, 1, 0] * 10)

    pipe = build_fold_pipeline(
        model_type="logistic_regression",
        numeric_features=["num"],
        categorical_features=["cat"],
        random_state=42,
    )
    pipe.fit(X_train, y_train)
    return pipe


def test_evaluate_perturbation_step_gaussian_noise(trained_pipeline):
    """Evaluate fitted pipeline on validation set under Gaussian noise without refitting."""
    X_val = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 2,
        "cat": ["tier_a", "tier_b", "tier_a", "tier_b", "tier_a"] * 4,
    })
    y_val = pd.Series([0, 1, 0, 1, 0] * 4)

    cfg = PerturbationConfig(
        perturbation_type=PerturbationType.GAUSSIAN_NOISE,
        severity_levels=[0.10],
        random_state=42,
    )

    res = evaluate_perturbation_step(
        estimator=trained_pipeline,
        X_val=X_val,
        y_val=y_val,
        config=cfg,
        severity=0.10,
    )

    assert res.perturbation_type == PerturbationType.GAUSSIAN_NOISE
    assert res.severity_level == 0.10
    assert res.is_usable is True
    assert res.clean_score is not None
    assert res.perturbed_score is not None
    assert res.retained_ratio is not None


def test_evaluate_robustness_curve_monotonic_severity(trained_pipeline):
    """Evaluate multi-point degradation curve under incremental missingness."""
    X_val = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 5,
        "cat": ["tier_a", "tier_b", "tier_a", "tier_b", "tier_a"] * 10,
    })
    y_val = pd.Series([0, 1, 0, 1, 0] * 10)

    cfg = PerturbationConfig(
        perturbation_type=PerturbationType.MISSINGNESS_INJECTION,
        severity_levels=[0.0, 0.10, 0.30, 0.60],
        random_state=42,
    )

    curve = evaluate_robustness_curve(
        estimator=trained_pipeline,
        X_val=X_val,
        y_val=y_val,
        config=cfg,
    )

    assert curve.perturbation_type == PerturbationType.MISSINGNESS_INJECTION
    assert len(curve.points) == 4
    assert curve.points[0].severity_level == 0.0
    assert curve.points[0].retained_ratio == pytest.approx(1.0, abs=1e-5)
    assert curve.area_under_degradation_curve is not None
    assert 0.0 <= curve.area_under_degradation_curve <= 1.5


def test_evaluate_model_robustness_multi_config(trained_pipeline):
    """Evaluate comprehensive robustness across noise, missingness, and category dropout."""
    X_val = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 5,
        "cat": ["tier_a", "tier_b", "tier_a", "tier_b", "tier_a"] * 10,
    })
    y_val = pd.Series([0, 1, 0, 1, 0] * 10)

    configs = [
        PerturbationConfig(perturbation_type=PerturbationType.GAUSSIAN_NOISE, severity_levels=[0.10, 0.25]),
        PerturbationConfig(perturbation_type=PerturbationType.MISSINGNESS_INJECTION, severity_levels=[0.10, 0.25]),
        PerturbationConfig(perturbation_type=PerturbationType.CATEGORY_DROPOUT, severity_levels=[0.10, 0.25]),
    ]

    robust_res = evaluate_model_robustness(
        estimator=trained_pipeline,
        X_val=X_val,
        y_val=y_val,
        configs=configs,
    )

    assert robust_res.is_usable is True
    assert len(robust_res.curves) == 3
    assert "gaussian_noise_max_retention" in robust_res.summary_retention
    assert "missingness_injection_max_retention" in robust_res.summary_retention
    assert "category_dropout_max_retention" in robust_res.summary_retention


def test_pipeline_failure_under_perturbation_handled_gracefully():
    """An estimator that crashes on unhandled NaNs is caught and reported as unusable without crashing."""
    from sklearn.linear_model import LogisticRegression

    # Raw estimator without SimpleImputer cannot handle NaNs
    X_clean = pd.DataFrame({"feat": [1.0, 2.0, 3.0, 4.0, 5.0] * 4})
    y_clean = pd.Series([0, 1, 0, 1, 0] * 4)

    raw_estimator = LogisticRegression()
    raw_estimator.fit(X_clean, y_clean)

    cfg = PerturbationConfig(
        perturbation_type=PerturbationType.MISSINGNESS_INJECTION,
        severity_levels=[0.50],
    )

    res = evaluate_perturbation_step(
        estimator=raw_estimator,
        X_val=X_clean,
        y_val=y_clean,
        config=cfg,
        severity=0.50,
    )

    assert res.is_usable is False
    assert res.unusable_reason is not None
    assert "failed under perturbation" in res.unusable_reason


# ============================================================================
# 8. Cross-Process Reproducibility Regression Tests (AUD-3.1-01)
# ============================================================================

def test_cross_process_perturbation_seed_determinism():
    """Verify that perturbations are bit-for-bit identical across separate Python processes."""
    script = (
        "import pandas as pd; "
        "from mlforge.engine.robustness import apply_gaussian_noise; "
        "df = pd.DataFrame({'feat': [10.0, 20.0, 30.0, 40.0, 50.0] * 5}); "
        "df_pert, _ = apply_gaussian_noise(df, severity=0.20, random_state=42); "
        "print(','.join(f'{x:.6f}' for x in df_pert['feat']))"
    )

    proc1 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    proc2 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)

    assert proc1.stdout.strip() == proc2.stdout.strip()
    assert len(proc1.stdout.strip().split(",")) == 25


# ============================================================================
# 9. Native pandas.CategoricalDtype Support (AUD-3.1-03)
# ============================================================================

def test_category_dropout_native_categorical_dtype():
    """Native pandas.CategoricalDtype column receives unseen token without TypeError."""
    df = pd.DataFrame({
        "cat_col": pd.Categorical(["alpha", "beta", "gamma", "alpha", "beta"] * 10),
        "unselected_cat": pd.Categorical(["x", "y", "z", "x", "y"] * 10),
        "num_col": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
    })
    df_copy = df.copy(deep=True)

    df_pert, warnings = apply_category_dropout(
        df,
        columns=["cat_col"],
        severity=0.30,
        unseen_token="__UNSEEN__",
        random_state=42,
    )

    # Input immutability
    pd.testing.assert_frame_equal(df, df_copy)

    # Unselected columns untouched
    pd.testing.assert_series_equal(df_pert["unselected_cat"], df["unselected_cat"])
    pd.testing.assert_series_equal(df_pert["num_col"], df["num_col"])

    # Selected column perturbed with unseen category
    assert "__UNSEEN__" in df_pert["cat_col"].values
    assert "__UNSEEN__" in df_pert["cat_col"].cat.categories
    assert isinstance(df_pert["cat_col"].dtype, pd.CategoricalDtype)


def test_category_dropout_native_categorical_preserves_existing_nulls():
    """Null values in native Categorical series are preserved and not overwritten."""
    df = pd.DataFrame({
        "cat_col": pd.Categorical(["alpha", None, "beta", None, "gamma"] * 10),
    })
    df_pert, _ = apply_category_dropout(df, severity=0.50, random_state=42)

    assert df_pert["cat_col"].isna().sum() == 20
    np.testing.assert_array_equal(df["cat_col"].isna(), df_pert["cat_col"].isna())


def test_category_dropout_collision_adapts_to_genuinely_unseen_token():
    """If __UNSEEN__ already exists in categories, code adapts to genuinely unseen token."""
    df = pd.DataFrame({
        "cat_col": pd.Categorical(["alpha", "__UNSEEN__", "beta"] * 10),
    })
    df_pert, warnings = apply_category_dropout(
        df, severity=0.50, unseen_token="__UNSEEN__", random_state=42
    )

    assert any("adapted to genuinely unseen token" in w for w in warnings)
    assert "__UNSEEN___1" in df_pert["cat_col"].cat.categories


# ============================================================================
# 10. Metric Directionality & Multi-Metric Dispatch (AUD-3.1-02)
# ============================================================================

def test_evaluate_perturbation_step_unsupported_metric_raises_value_error(trained_pipeline):
    """Unsupported metric names must raise ValueError rather than silently returning accuracy."""
    X_val = pd.DataFrame({"num": [1.0, 2.0] * 5, "cat": ["tier_a", "tier_b"] * 5})
    y_val = pd.Series([0, 1] * 5)
    cfg = PerturbationConfig(perturbation_type=PerturbationType.GAUSSIAN_NOISE, severity_levels=[0.10])

    with pytest.raises(ValueError, match="Unsupported metric 'unsupported_metric'"):
        evaluate_perturbation_step(
            estimator=trained_pipeline,
            X_val=X_val,
            y_val=y_val,
            config=cfg,
            severity=0.10,
            metric_name="unsupported_metric",
        )


def test_evaluate_perturbation_step_higher_is_better_macro_f1(trained_pipeline):
    """Supported higher-is-better metric (macro_f1) correctly dispatched and computed."""
    X_val = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0, 5.0] * 4,
        "cat": ["tier_a", "tier_b", "tier_a", "tier_b", "tier_a"] * 4,
    })
    y_val = pd.Series([0, 1, 0, 1, 0] * 4)
    cfg = PerturbationConfig(perturbation_type=PerturbationType.GAUSSIAN_NOISE, severity_levels=[0.10])

    res = evaluate_perturbation_step(
        estimator=trained_pipeline,
        X_val=X_val,
        y_val=y_val,
        config=cfg,
        severity=0.10,
        metric_name="macro_f1",
    )

    assert res.metric_name == "macro_f1"
    assert res.is_usable is True
    assert res.clean_score is not None
    assert res.perturbed_score is not None
    assert res.retained_ratio is not None


def test_evaluate_perturbation_step_lower_is_better_log_loss(trained_pipeline):
    """Supported lower-is-better metric (log_loss) honors loss degradation semantics."""
    X_val = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0, 5.0] * 4,
        "cat": ["tier_a", "tier_b", "tier_a", "tier_b", "tier_a"] * 4,
    })
    y_val = pd.Series([0, 1, 0, 1, 0] * 4)
    cfg = PerturbationConfig(perturbation_type=PerturbationType.GAUSSIAN_NOISE, severity_levels=[0.25])

    res = evaluate_perturbation_step(
        estimator=trained_pipeline,
        X_val=X_val,
        y_val=y_val,
        config=cfg,
        severity=0.25,
        metric_name="log_loss",
    )

    assert res.metric_name == "log_loss"
    assert res.is_usable is True
    assert res.clean_score is not None
    assert res.perturbed_score is not None
    if res.perturbed_score > res.clean_score:
        assert res.performance_drop_percentage > 0.0
        assert res.retained_ratio < 1.0


def test_retention_metrics_lower_is_better_hand_checked():
    """Hand-check lower-is-better: clean=0.20, perturbed=0.40 -> drop=100%, retained_ratio=0.50."""
    ratio, drop_pct, usable, reason = calculate_retention_metrics(
        clean_score=0.20,
        perturbed_score=0.40,
        metric_name="log_loss",
        direction=MetricDirection.LOWER_IS_BETTER,
    )
    assert usable is True
    assert drop_pct == pytest.approx(100.0, abs=1e-5)
    assert ratio == pytest.approx(0.50, abs=1e-5)


def test_retention_metrics_lower_is_better_zero_clean_score():
    """Zero clean score for loss metric cannot compute relative drop percentage."""
    ratio, drop_pct, usable, reason = calculate_retention_metrics(
        clean_score=0.0,
        perturbed_score=0.40,
        metric_name="log_loss",
        direction=MetricDirection.LOWER_IS_BETTER,
    )
    assert usable is False
    assert ratio is None
    assert drop_pct is None
    assert "ZERO_BASELINE_SCORE" in reason


def test_retention_metrics_lower_is_better_zero_perturbed_score():
    """Zero perturbed loss gives 100% loss drop but undefined retention ratio (division by zero)."""
    ratio, drop_pct, usable, reason = calculate_retention_metrics(
        clean_score=0.20,
        perturbed_score=0.0,
        metric_name="log_loss",
        direction=MetricDirection.LOWER_IS_BETTER,
    )
    assert usable is False
    assert ratio is None
    assert drop_pct == pytest.approx(-100.0, abs=1e-5)
    assert "ZERO_PERTURBED_LOSS" in reason


def test_retention_metrics_lower_is_better_improvement_and_worsening():
    """Perturbed performance improving (lower loss) yields negative drop and ratio > 1; worsening yields positive drop and ratio < 1."""
    # Improvement: loss decreases from 0.40 to 0.20
    r_imp, drop_imp, usable_imp, _ = calculate_retention_metrics(
        clean_score=0.40,
        perturbed_score=0.20,
        metric_name="brier_score",
        direction=MetricDirection.LOWER_IS_BETTER,
    )
    assert usable_imp is True
    assert drop_imp == pytest.approx(-50.0, abs=1e-5)
    assert r_imp == pytest.approx(2.0, abs=1e-5)

    # Degradation: loss increases from 0.40 to 0.80
    r_deg, drop_deg, usable_deg, _ = calculate_retention_metrics(
        clean_score=0.40,
        perturbed_score=0.80,
        metric_name="brier_score",
        direction=MetricDirection.LOWER_IS_BETTER,
    )
    assert usable_deg is True
    assert drop_deg == pytest.approx(100.0, abs=1e-5)
    assert r_deg == pytest.approx(0.50, abs=1e-5)


def test_retention_metrics_lower_is_better_negative_scores_unusable():
    """Negative values for loss metrics (log_loss, brier_score) are mathematically invalid."""
    r1, d1, u1, re1 = calculate_retention_metrics(
        clean_score=-0.05,
        perturbed_score=0.20,
        metric_name="log_loss",
        direction=MetricDirection.LOWER_IS_BETTER,
    )
    assert u1 is False
    assert "NEGATIVE_SCORE_LOSS" in re1

    r2, d2, u2, re2 = calculate_retention_metrics(
        clean_score=0.20,
        perturbed_score=-0.05,
        metric_name="log_loss",
        direction=MetricDirection.LOWER_IS_BETTER,
    )
    assert u2 is False
    assert "NEGATIVE_SCORE_LOSS" in re2


def test_retention_metrics_lower_is_better_extreme_near_zero():
    """Extreme near-zero positive values compute finite metrics without underflow crash."""
    r_near, d_near, u_near, _ = calculate_retention_metrics(
        clean_score=1e-12,
        perturbed_score=0.5,
        metric_name="log_loss",
        direction=MetricDirection.LOWER_IS_BETTER,
    )
    assert u_near is True
    assert r_near == pytest.approx(0.0, abs=1e-6)
    assert d_near > 1e13


def test_retention_metrics_non_finite_inputs_unusable():
    """NaN, Inf, and None scores are strictly flagged as unusable."""
    r_nan, _, u_nan, re_nan = calculate_retention_metrics(
        clean_score=float("nan"),
        perturbed_score=0.5,
        metric_name="log_loss",
        direction=MetricDirection.LOWER_IS_BETTER,
    )
    assert u_nan is False
    assert "NON_FINITE_METRIC" in re_nan

    r_inf, _, u_inf, re_inf = calculate_retention_metrics(
        clean_score=0.5,
        perturbed_score=float("inf"),
        metric_name="log_loss",
        direction=MetricDirection.LOWER_IS_BETTER,
    )
    assert u_inf is False
    assert "NON_FINITE_METRIC" in re_inf

    r_none, _, u_none, re_none = calculate_retention_metrics(
        clean_score=None,
        perturbed_score=0.5,
        metric_name="log_loss",
        direction=MetricDirection.LOWER_IS_BETTER,
    )
    assert u_none is False
    assert "METRIC_UNDEFINED" in re_none


def test_retention_metrics_negative_clean_score_unusable():
    """Negative clean baseline score must be flagged as unusable without crashing."""
    ratio, drop_pct, usable, reason = calculate_retention_metrics(
        clean_score=-0.10,
        perturbed_score=-0.20,
        metric_name="accuracy",
    )
    assert usable is False
    assert "NEGATIVE_BASELINE_SCORE" in reason


# ============================================================================
# 11. Robustness Curve Safety Under Incomplete Evaluability (AUD-3.1-04)
# ============================================================================

def test_robustness_curve_auc_undefined_when_intermediate_point_fails():
    """If an intermediate or terminal point in the severity sweep fails, AUC is None with warning."""
    from sklearn.linear_model import LogisticRegression

    raw_estimator = LogisticRegression()
    X_train = pd.DataFrame({"feat": [1.0, 2.0, 3.0, 4.0, 5.0] * 5})
    y_train = pd.Series([0, 1, 0, 1, 0] * 5)
    raw_estimator.fit(X_train, y_train)

    cfg = PerturbationConfig(
        perturbation_type=PerturbationType.MISSINGNESS_INJECTION,
        severity_levels=[0.0, 0.20, 0.50],
    )

    curve = evaluate_robustness_curve(
        estimator=raw_estimator,
        X_val=X_train,
        y_val=y_train,
        config=cfg,
    )

    assert curve.points[0].is_usable is True
    assert curve.points[1].is_usable is False
    assert curve.points[2].is_usable is False
    assert curve.area_under_degradation_curve is None
    assert any("Area under degradation curve is undefined" in w for w in curve.warnings)


def test_robustness_curve_single_point():
    """Single-point severity curve evaluates AUC as single point retention ratio."""
    pt = RobustnessPerturbationResult(
        perturbation_type=PerturbationType.GAUSSIAN_NOISE,
        severity_level=0.10,
        metric_name="accuracy",
        clean_score=0.80,
        perturbed_score=0.72,
        retained_ratio=0.90,
        performance_drop_percentage=10.0,
        is_usable=True,
    )
    curve = RobustnessCurve(
        perturbation_type=PerturbationType.GAUSSIAN_NOISE,
        metric_name="accuracy",
        clean_score=0.80,
        points=[pt],
        area_under_degradation_curve=0.90,
    )
    assert curve.area_under_degradation_curve == 0.90


# ============================================================================
# 12. PerturbationConfig Early Validation (AUD-3.1-06)
# ============================================================================

def test_perturbation_config_rejects_empty_severity_levels():
    """Empty severity levels list raises ValidationError."""
    with pytest.raises(Exception):
        PerturbationConfig(perturbation_type=PerturbationType.GAUSSIAN_NOISE, severity_levels=[])


def test_perturbation_config_rejects_unsorted_severity_levels():
    """Unsorted severity levels raise ValidationError."""
    with pytest.raises(Exception):
        PerturbationConfig(
            perturbation_type=PerturbationType.GAUSSIAN_NOISE,
            severity_levels=[0.25, 0.10],
        )


def test_perturbation_config_rejects_duplicate_severity_levels():
    """Duplicate severity levels raise ValidationError."""
    with pytest.raises(Exception):
        PerturbationConfig(
            perturbation_type=PerturbationType.GAUSSIAN_NOISE,
            severity_levels=[0.10, 0.10, 0.25],
        )


def test_perturbation_config_rejects_out_of_bounds_severities():
    """Missingness severity > 1.0 or < 0.0 raises ValidationError."""
    with pytest.raises(Exception):
        PerturbationConfig(
            perturbation_type=PerturbationType.MISSINGNESS_INJECTION,
            severity_levels=[-0.10, 0.50],
        )
    with pytest.raises(Exception):
        PerturbationConfig(
            perturbation_type=PerturbationType.MISSINGNESS_INJECTION,
            severity_levels=[0.50, 1.20],
        )


# ============================================================================
# 13. Compound Perturbation Transformer Tests (Milestone 3.2)
# ============================================================================

def test_compound_perturbation_preserves_original_dataframe(sample_features_df):
    """Compound perturbation must strictly preserve original DataFrame in-place."""
    df_copy = sample_features_df.copy(deep=True)
    df_pert, _ = apply_compound_perturbation(sample_features_df, severity=0.20, random_state=42)

    pd.testing.assert_frame_equal(sample_features_df, df_copy)
    assert not np.array_equal(df_pert["num_1"].to_numpy(), sample_features_df["num_1"].to_numpy())
    assert df_pert["num_1"].isna().sum() > 0
    assert "__UNSEEN__" in df_pert["cat_1"].values


def test_compound_perturbation_deterministic_reproducibility(sample_features_df):
    """Two independent compound runs with identical seeds produce bit-for-bit identical frames."""
    df_pert_1, _ = apply_compound_perturbation(sample_features_df, severity=0.25, random_state=123)
    df_pert_2, _ = apply_compound_perturbation(sample_features_df, severity=0.25, random_state=123)

    pd.testing.assert_frame_equal(df_pert_1, df_pert_2)


def test_compound_perturbation_distinct_seeds_yield_distinct_results(sample_features_df):
    """Different random seeds produce distinct compound corruption results."""
    df_pert_1, _ = apply_compound_perturbation(sample_features_df, severity=0.25, random_state=111)
    df_pert_2, _ = apply_compound_perturbation(sample_features_df, severity=0.25, random_state=222)

    # Values must differ
    valid_mask = ~df_pert_1["num_1"].isna() & ~df_pert_2["num_1"].isna()
    assert not np.allclose(df_pert_1.loc[valid_mask, "num_1"], df_pert_2.loc[valid_mask, "num_1"])


def test_compound_perturbation_cross_process_determinism():
    """Compound perturbation produces bit-for-bit identical outputs across separate Python processes."""
    script = (
        "import pandas as pd; "
        "from mlforge.engine.robustness import apply_compound_perturbation; "
        "df = pd.DataFrame({'num': [10.0, 20.0, 30.0, 40.0] * 5, 'cat': ['A', 'B', 'A', 'B'] * 5}); "
        "df_pert, _ = apply_compound_perturbation(df, severity=0.20, random_state=42); "
        "print(','.join(str(x) for x in df_pert['cat']))"
    )

    proc1 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    proc2 = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)

    assert proc1.stdout.strip() == proc2.stdout.strip()


def test_compound_perturbation_zero_severity_is_identity(sample_features_df):
    """severity=0.0 must return identical data values without corruption."""
    df_pert, warnings = apply_compound_perturbation(sample_features_df, severity=0.0)
    pd.testing.assert_frame_equal(df_pert, sample_features_df)
    assert len(warnings) == 0


def test_compound_perturbation_all_numeric_dataframe():
    """Compound perturbation works on datasets lacking categorical columns with advisory warning."""
    df = pd.DataFrame({"num_1": [1.0, 2.0, 3.0, 4.0, 5.0] * 10, "num_2": [10.0, 20.0, 30.0, 40.0, 50.0] * 10})
    df_pert, warnings = apply_compound_perturbation(df, severity=0.20, random_state=42)

    assert any("category_dropout" in w for w in warnings)
    assert df_pert["num_1"].isna().sum() > 0


def test_compound_perturbation_all_categorical_dataframe():
    """Compound perturbation works on datasets lacking numeric columns with advisory warning."""
    df = pd.DataFrame({"cat_1": ["A", "B", "C", "D"] * 10, "cat_2": ["X", "Y", "Z", "W"] * 10})
    df_pert, warnings = apply_compound_perturbation(df, severity=0.20, random_state=42)

    assert any("gaussian_noise" in w for w in warnings)
    assert "__UNSEEN__" in df_pert["cat_1"].values
    assert df_pert["cat_1"].isna().sum() > 0


def test_compound_perturbation_custom_steps():
    """Compound perturbation executes caller-specified steps in exact requested order."""
    df = pd.DataFrame({
        "num": [10.0, 20.0, 30.0, 40.0] * 5,
        "cat": ["tier1", "tier2", "tier1", "tier2"] * 5,
    })

    # Only apply noise on num and dropout on cat (no missingness)
    custom_steps = [
        PerturbationConfig(perturbation_type=PerturbationType.GAUSSIAN_NOISE, severity_levels=[0.10]),
        PerturbationConfig(perturbation_type=PerturbationType.CATEGORY_DROPOUT, severity_levels=[0.10]),
    ]
    df_pert, _ = apply_compound_perturbation(df, steps=custom_steps, severity=0.20)

    # Missingness was excluded, so no nulls should be introduced
    assert df_pert["num"].isna().sum() == 0
    assert df_pert["cat"].isna().sum() == 0
    assert "__UNSEEN__" in df_pert["cat"].values


def test_compound_perturbation_rejects_nested_compound():
    """Nested compound steps are strictly rejected with ValueError."""
    nested = PerturbationConfig(perturbation_type=PerturbationType.COMPOUND, severity_levels=[0.10])
    with pytest.raises(Exception):
        apply_compound_perturbation(
            pd.DataFrame({"a": [1, 2]}),
            steps=[nested],
        )


def test_compound_perturbation_rejects_invalid_inputs():
    """Empty dataframe or invalid severity values raise ValueError."""
    with pytest.raises(ValueError, match="Cannot perturb an empty"):
        apply_compound_perturbation(pd.DataFrame(), severity=0.10)

    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
        apply_compound_perturbation(pd.DataFrame({"a": [1.0, 2.0]}), severity=1.5)


# ============================================================================
# 14. Area Under Robustness Degradation Curve (AURDC) Tests (Milestone 3.2)
# ============================================================================

def test_calculate_aurdc_monotonic_curve_hand_checked():
    """Hand-check AURDC: sevs=[0.0, 0.5, 1.0], retentions=[1.0, 0.8, 0.4].
    Trapezoid 1: width 0.5, avg (1.0 + 0.8)/2 = 0.9 -> area = 0.45.
    Trapezoid 2: width 0.5, avg (0.8 + 0.4)/2 = 0.6 -> area = 0.30.
    Total integral = 0.75. Normalized AURDC = 0.75 / 1.0 = 0.75.
    """
    pts = [
        RobustnessPerturbationResult(
            perturbation_type=PerturbationType.GAUSSIAN_NOISE,
            severity_level=0.0,
            metric_name="accuracy",
            retained_ratio=1.0,
            is_usable=True,
        ),
        RobustnessPerturbationResult(
            perturbation_type=PerturbationType.GAUSSIAN_NOISE,
            severity_level=0.5,
            metric_name="accuracy",
            retained_ratio=0.8,
            is_usable=True,
        ),
        RobustnessPerturbationResult(
            perturbation_type=PerturbationType.GAUSSIAN_NOISE,
            severity_level=1.0,
            metric_name="accuracy",
            retained_ratio=0.4,
            is_usable=True,
        ),
    ]

    aurdc, warnings = calculate_aurdc(pts)
    assert len(warnings) == 0
    assert aurdc == pytest.approx(0.75, abs=1e-5)


def test_calculate_aurdc_non_uniform_severity_grid():
    """AURDC handles non-uniform severity grid spacing correctly."""
    pts = [
        RobustnessPerturbationResult(
            perturbation_type=PerturbationType.GAUSSIAN_NOISE,
            severity_level=0.05,
            metric_name="accuracy",
            retained_ratio=1.0,
            is_usable=True,
        ),
        RobustnessPerturbationResult(
            perturbation_type=PerturbationType.GAUSSIAN_NOISE,
            severity_level=0.10,
            metric_name="accuracy",
            retained_ratio=0.90,
            is_usable=True,
        ),
        RobustnessPerturbationResult(
            perturbation_type=PerturbationType.GAUSSIAN_NOISE,
            severity_level=0.50,
            metric_name="accuracy",
            retained_ratio=0.50,
            is_usable=True,
        ),
    ]
    # Interval 1: [0.05, 0.10], width 0.05, avg 0.95 -> 0.0475
    # Interval 2: [0.10, 0.50], width 0.40, avg 0.70 -> 0.2800
    # Integral = 0.3275. Domain = 0.50 - 0.05 = 0.45.
    # Normalized = 0.3275 / 0.45 = 0.727778
    aurdc, warnings = calculate_aurdc(pts)
    assert len(warnings) == 0
    assert aurdc == pytest.approx(0.727778, abs=1e-5)


def test_calculate_aurdc_single_point():
    """Single point returns point retention value as degenerate AURDC."""
    pt = RobustnessPerturbationResult(
        perturbation_type=PerturbationType.GAUSSIAN_NOISE,
        severity_level=0.10,
        metric_name="accuracy",
        retained_ratio=0.88,
        is_usable=True,
    )
    aurdc, warnings = calculate_aurdc([pt])
    assert len(warnings) == 0
    assert aurdc == pytest.approx(0.88, abs=1e-5)


def test_calculate_aurdc_empty_points_returns_none():
    """Empty points list returns None with advisory warning."""
    aurdc, warnings = calculate_aurdc([])
    assert aurdc is None
    assert any("empty points list" in w for w in warnings)


def test_calculate_aurdc_unusable_point_causes_none_with_warning():
    """If any point in the sweep is unusable, AURDC is None to prevent survivorship bias."""
    pts = [
        RobustnessPerturbationResult(
            perturbation_type=PerturbationType.GAUSSIAN_NOISE,
            severity_level=0.05,
            metric_name="accuracy",
            retained_ratio=0.95,
            is_usable=True,
        ),
        RobustnessPerturbationResult(
            perturbation_type=PerturbationType.GAUSSIAN_NOISE,
            severity_level=0.10,
            metric_name="accuracy",
            retained_ratio=None,
            is_usable=False,
            unusable_reason="Pipeline crashed",
        ),
    ]
    aurdc, warnings = calculate_aurdc(pts)
    assert aurdc is None
    assert any("Area under degradation curve is undefined" in w for w in warnings)


def test_calculate_aurdc_unsorted_points_handled_correctly():
    """Verify calculate_aurdc sorts points by severity before integration to prevent negative delta intervals."""
    # Points presented in jumbled order: [0.5, 1.0, 0.0]
    pts = [
        RobustnessPerturbationResult(
            perturbation_type=PerturbationType.GAUSSIAN_NOISE,
            severity_level=0.5,
            metric_name="accuracy",
            retained_ratio=0.8,
            is_usable=True,
        ),
        RobustnessPerturbationResult(
            perturbation_type=PerturbationType.GAUSSIAN_NOISE,
            severity_level=1.0,
            metric_name="accuracy",
            retained_ratio=0.4,
            is_usable=True,
        ),
        RobustnessPerturbationResult(
            perturbation_type=PerturbationType.GAUSSIAN_NOISE,
            severity_level=0.0,
            metric_name="accuracy",
            retained_ratio=1.0,
            is_usable=True,
        ),
    ]
    aurdc, warnings = calculate_aurdc(pts)
    assert len(warnings) == 0
    # Must match monotonic curve test: 0.75
    assert aurdc == pytest.approx(0.75, abs=1e-5)


def test_calculate_aurdc_duplicate_severities_returns_none():
    """Duplicate severity levels indicate ambiguous data; returns None with explanatory warning."""
    pts = [
        RobustnessPerturbationResult(
            perturbation_type=PerturbationType.GAUSSIAN_NOISE,
            severity_level=0.10,
            metric_name="accuracy",
            retained_ratio=0.90,
            is_usable=True,
        ),
        RobustnessPerturbationResult(
            perturbation_type=PerturbationType.GAUSSIAN_NOISE,
            severity_level=0.10,
            metric_name="accuracy",
            retained_ratio=0.85,
            is_usable=True,
        ),
        RobustnessPerturbationResult(
            perturbation_type=PerturbationType.GAUSSIAN_NOISE,
            severity_level=0.50,
            metric_name="accuracy",
            retained_ratio=0.60,
            is_usable=True,
        ),
    ]
    aurdc, warnings = calculate_aurdc(pts)
    assert aurdc is None
    assert any("duplicate severity levels" in w for w in warnings)


def test_calculate_aurdc_non_finite_values_returns_none():
    """Points with NaN or Inf retention ratios or severities are rejected as unusable."""
    pts_nan = [
        RobustnessPerturbationResult(
            perturbation_type=PerturbationType.GAUSSIAN_NOISE,
            severity_level=0.0,
            metric_name="accuracy",
            retained_ratio=1.0,
            is_usable=True,
        ),
        RobustnessPerturbationResult(
            perturbation_type=PerturbationType.GAUSSIAN_NOISE,
            severity_level=0.5,
            metric_name="accuracy",
            retained_ratio=float("nan"),
            is_usable=True,
        ),
    ]
    aurdc_nan, w_nan = calculate_aurdc(pts_nan)
    assert aurdc_nan is None
    assert any("Area under degradation curve is undefined" in w for w in w_nan)

    pts_inf = [
        RobustnessPerturbationResult(
            perturbation_type=PerturbationType.GAUSSIAN_NOISE,
            severity_level=0.0,
            metric_name="accuracy",
            retained_ratio=1.0,
            is_usable=True,
        ),
        RobustnessPerturbationResult(
            perturbation_type=PerturbationType.GAUSSIAN_NOISE,
            severity_level=0.5,
            metric_name="accuracy",
            retained_ratio=float("inf"),
            is_usable=True,
        ),
    ]
    aurdc_inf, w_inf = calculate_aurdc(pts_inf)
    assert aurdc_inf is None
    assert any("Area under degradation curve is undefined" in w for w in w_inf)


def test_robustness_curve_aurdc_property_alias():
    """RobustnessCurve.aurdc property returns area_under_degradation_curve identically."""
    curve = RobustnessCurve(
        perturbation_type=PerturbationType.GAUSSIAN_NOISE,
        metric_name="accuracy",
        clean_score=0.80,
        points=[],
        area_under_degradation_curve=0.825,
    )
    assert curve.aurdc == 0.825


# ============================================================================
# 15. Pre-Packaged Stress Profiles & Suite Tests (Milestone 3.2)
# ============================================================================

def test_standard_severity_grid_contents():
    """Verify standard severity grid contents match roadmap specifications."""
    assert STANDARD_SEVERITY_GRID == [0.05, 0.10, 0.25, 0.50]


def test_create_noise_stress_profile_defaults_and_overrides():
    """Noise stress profile uses standard grid by default and allows column targeting."""
    prof = create_noise_stress_profile(target_columns=["feat_a"], random_state=99)
    assert prof.perturbation_type == PerturbationType.GAUSSIAN_NOISE
    assert prof.severity_levels == [0.05, 0.10, 0.25, 0.50]
    assert prof.target_columns == ["feat_a"]
    assert prof.random_state == 99


def test_create_missingness_stress_profile_defaults_and_overrides():
    """Missingness stress profile uses standard grid by default."""
    prof = create_missingness_stress_profile()
    assert prof.perturbation_type == PerturbationType.MISSINGNESS_INJECTION
    assert prof.severity_levels == [0.05, 0.10, 0.25, 0.50]


def test_create_category_dropout_stress_profile_defaults_and_overrides():
    """Category dropout stress profile uses standard grid by default."""
    prof = create_category_dropout_stress_profile()
    assert prof.perturbation_type == PerturbationType.CATEGORY_DROPOUT
    assert prof.severity_levels == [0.05, 0.10, 0.25, 0.50]


def test_create_compound_stress_profile_defaults_and_overrides():
    """Compound stress profile uses standard grid and compound type."""
    prof = create_compound_stress_profile()
    assert prof.perturbation_type == PerturbationType.COMPOUND
    assert prof.severity_levels == [0.05, 0.10, 0.25, 0.50]


def test_get_standard_stress_suite_returns_all_profiles():
    """Standard stress suite contains noise, missingness, category dropout, and compound profiles."""
    suite = get_standard_stress_suite()
    assert len(suite) == 4
    types = [cfg.perturbation_type for cfg in suite]
    assert PerturbationType.GAUSSIAN_NOISE in types
    assert PerturbationType.MISSINGNESS_INJECTION in types
    assert PerturbationType.CATEGORY_DROPOUT in types
    assert PerturbationType.COMPOUND in types


def test_evaluate_stress_profile_runs_full_standard_suite(trained_pipeline):
    """evaluate_stress_profile executes all 4 standard profiles and computes AURDCs."""
    X_val = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 5,
        "cat": ["tier_a", "tier_b", "tier_a", "tier_b", "tier_a"] * 10,
    })
    y_val = pd.Series([0, 1, 0, 1, 0] * 10)

    res = evaluate_stress_profile(
        estimator=trained_pipeline,
        X_val=X_val,
        y_val=y_val,
        metric_name="accuracy",
    )

    assert res.is_usable is True
    assert len(res.curves) == 4
    assert "gaussian_noise_aurdc" in res.summary_retention
    assert "missingness_injection_aurdc" in res.summary_retention
    assert "category_dropout_aurdc" in res.summary_retention
    assert "compound_aurdc" in res.summary_retention
    assert "mean_aurdc" in res.summary_retention
    assert 0.0 <= res.summary_retention["mean_aurdc"] <= 1.5


def test_evaluate_stress_profile_single_profile(trained_pipeline):
    """evaluate_stress_profile accepts a single profile and executes it cleanly."""
    X_val = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0, 5.0] * 5,
        "cat": ["tier_a", "tier_b", "tier_a", "tier_b", "tier_a"] * 5,
    })
    y_val = pd.Series([0, 1, 0, 1, 0] * 5)
    prof = create_compound_stress_profile(severity_levels=[0.10, 0.25])

    res = evaluate_stress_profile(
        estimator=trained_pipeline,
        X_val=X_val,
        y_val=y_val,
        profile=prof,
    )

    assert res.is_usable is True
    assert len(res.curves) == 1
    assert res.curves[0].perturbation_type == PerturbationType.COMPOUND
    assert "compound_aurdc" in res.summary_retention
