"""Unit and integration tests for classification evaluation orchestrator.

Tests baseline vs candidate execution, mixed features with missingness,
multiclass evaluation, and end-to-end determinism.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from mlforge.engine.evaluator import evaluate_classification


@pytest.fixture
def synthetic_binary_df():
    """Generates a synthetic binary classification dataset with separable patterns."""
    X_raw, y_raw = make_classification(
        n_samples=100,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_classes=2,
        weights=[0.6, 0.4],
        random_state=42,
    )
    df = pd.DataFrame(X_raw, columns=[f"feat_{i}" for i in range(6)])
    df["target"] = y_raw
    return df


def test_end_to_end_evaluation_structure(synthetic_binary_df):
    """Verifies complete evaluation run comparing DummyClassifier vs LogisticRegression."""
    result = evaluate_classification(
        df=synthetic_binary_df,
        target_column="target",
        num_folds=5,
        random_state=42,
    )

    # 1. Dataset Metadata
    assert result.dataset_metadata.num_rows == 100
    assert result.dataset_metadata.num_features == 6
    assert result.dataset_metadata.class_distribution == {"0": 60, "1": 40}

    # 2. Reproducibility Info
    assert result.reproducibility.random_seed == 42
    assert result.reproducibility.num_folds == 5
    assert "scikit-learn" in result.reproducibility.package_versions

    # 3. Baseline vs Candidate Model Separation
    baseline = result.baseline_result
    candidate = result.candidate_result

    assert baseline.model_name == "Baseline: DummyClassifier"
    assert candidate.model_name == "Candidate: LogisticRegression"
    assert baseline.model_provenance == "final_cv_fold"
    assert baseline.fitted_fold_index == 4
    assert candidate.model_provenance == "final_cv_fold"
    assert candidate.fitted_fold_index == 4
    assert len(baseline.fold_results) == 5
    assert len(candidate.fold_results) == 5

    # 4. LogisticRegression should easily outperform Dummy on separable data
    cand_acc = candidate.metric_summaries["accuracy"].mean
    base_acc = baseline.metric_summaries["accuracy"].mean
    cand_bal_acc = candidate.metric_summaries["balanced_accuracy"].mean
    base_bal_acc = baseline.metric_summaries["balanced_accuracy"].mean

    assert cand_acc is not None and base_acc is not None
    assert cand_acc > base_acc
    assert cand_bal_acc is not None and base_bal_acc is not None
    assert cand_bal_acc > base_bal_acc

    # Candidate has valid F1; Dummy has undefined F1 because it never predicted the minority class
    assert candidate.metric_summaries["macro_f1"].mean is not None
    assert baseline.metric_summaries["macro_f1"].undefined_fold_count == 5

    # 5. Aggregate Confusion Matrix Conservation
    # Total count in aggregate CM must equal total sample count across folds (100)
    base_cm_sum = sum(sum(row) for row in baseline.aggregate_confusion_matrix)
    cand_cm_sum = sum(sum(row) for row in candidate.aggregate_confusion_matrix)
    assert base_cm_sum == 100
    assert cand_cm_sum == 100


def test_mixed_features_with_missing_values():
    """Verifies evaluation succeeds on mixed numeric & categorical features with missing values."""
    df = pd.DataFrame({
        "num_clean": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 3,
        "num_with_nan": [10.0, np.nan, 30.0, np.nan, 50.0, 60.0, 70.0, np.nan, 90.0, 100.0] * 3,
        "cat_clean": ["dept_a", "dept_b", "dept_c", "dept_a", "dept_b"] * 6,
        "cat_with_nan": ["tier_1", None, "tier_2", "tier_1", None] * 6,
        "churn": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 3,
    })

    result = evaluate_classification(
        df=df,
        target_column="churn",
        num_folds=3,
        random_state=42,
    )

    assert result.dataset_metadata.numeric_feature_count == 2
    assert result.dataset_metadata.categorical_feature_count == 2
    assert result.dataset_metadata.missing_value_counts["num_with_nan"] == 9
    assert result.dataset_metadata.missing_value_counts["cat_with_nan"] == 12

    # Fold execution completed without imputation or encoding error
    assert len(result.candidate_result.fold_results) == 3
    for fr in result.candidate_result.fold_results:
        assert fr.metrics["accuracy"].value is not None


def test_multiclass_evaluation():
    """Verifies evaluation preserves multiclass labels and 3x3 confusion matrix."""
    df = pd.DataFrame({
        "x1": [1, 2, 3, 4, 5, 6, 7, 8, 9] * 4,
        "x2": [10, 20, 30, 40, 50, 60, 70, 80, 90] * 4,
        "category": ["low", "medium", "high"] * 12,
    })

    result = evaluate_classification(
        df=df,
        target_column="category",
        num_folds=3,
        random_state=42,
    )

    candidate = result.candidate_result
    assert len(candidate.aggregate_confusion_matrix) == 3
    assert len(candidate.aggregate_confusion_matrix[0]) == 3
    assert result.dataset_metadata.class_labels == ["high", "low", "medium"]


def test_end_to_end_determinism(synthetic_binary_df):
    """Verifies that evaluating the exact same dataset with the same seed yields bit-for-bit identical metrics."""
    res1 = evaluate_classification(synthetic_binary_df, target_column="target", num_folds=4, random_state=42)
    res2 = evaluate_classification(synthetic_binary_df, target_column="target", num_folds=4, random_state=42)

    cand1 = res1.candidate_result.metric_summaries
    cand2 = res2.candidate_result.metric_summaries

    for metric_name in cand1:
        assert cand1[metric_name].mean == cand2[metric_name].mean
        assert cand1[metric_name].std == cand2[metric_name].std
        assert cand1[metric_name].confidence_interval_95 == cand2[metric_name].confidence_interval_95
