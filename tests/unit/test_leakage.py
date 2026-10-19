"""Unit tests for leakage-safe preprocessing isolation.

Covers test cases TC-LEAK-01 through TC-LEAK-04.
Specifically validates that scalers, imputers, and encoders fit strictly on fold training data
and are never contaminated by validation samples.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mlforge.engine.evaluator import evaluate_classification
from mlforge.engine.pipeline import build_fold_pipeline


def test_tc_leak_02_scaler_isolation_with_extreme_validation_values():
    """TC-LEAK-02: Scaler fitted statistics strictly match training fold, ignoring extreme validation values.

    If leakage occurred, the scaler mean would be contaminated by the extreme 10,000.0 values.
    """
    # Training samples: values centered at 10.0
    X_train = pd.DataFrame({"num_feat": [9.0, 10.0, 11.0, 10.0, 10.0]})
    y_train = pd.Series([0, 1, 0, 1, 0])

    # Validation samples: extreme values that would distort mean/std if leaked
    X_val = pd.DataFrame({"num_feat": [10000.0, 20000.0]})
    y_val = pd.Series([0, 1])

    pipeline = build_fold_pipeline(
        model_type="logistic_regression",
        numeric_features=["num_feat"],
        categorical_features=[],
        random_state=42,
    )

    # Fit strictly on training fold
    pipeline.fit(X_train, y_train)

    # Extract fitted scaler from ColumnTransformer
    preprocessor = pipeline.named_steps["preprocessor"]
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]

    # Training mean is exactly 10.0
    assert scaler.mean_[0] == pytest.approx(10.0, abs=1e-5)

    # If full dataset had been leaked (train + val), mean would be (50 + 30000) / 7 = ~4292.8
    assert scaler.mean_[0] < 50.0

    # Predict on validation data with extreme values
    preds = pipeline.predict(X_val)
    assert len(preds) == 2


def test_tc_leak_01_imputer_isolation_with_validation_values():
    """TC-LEAK-01: SimpleImputer parameter matches training partition only."""
    # Training partition median is 20.0
    X_train = pd.DataFrame({"age": [10.0, 20.0, 30.0, np.nan]})
    y_train = pd.Series([0, 1, 0, 1])

    # Validation partition with missing value and extreme observed values
    X_val = pd.DataFrame({"age": [9999.0, np.nan]})

    pipeline = build_fold_pipeline(
        model_type="logistic_regression",
        numeric_features=["age"],
        categorical_features=[],
        random_state=42,
    )

    pipeline.fit(X_train, y_train)

    preprocessor = pipeline.named_steps["preprocessor"]
    imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]

    # Imputer statistics must equal training median (20.0)
    assert imputer.statistics_[0] == pytest.approx(20.0, abs=1e-5)


def test_tc_leak_03_unseen_category_handling_without_leakage():
    """TC-LEAK-03: OneHotEncoder categories match training only; unseen validation categories handled safely."""
    X_train = pd.DataFrame({
        "color": ["red", "blue", "red", "blue"],
        "num": [1.0, 2.0, 3.0, 4.0],
    })
    y_train = pd.Series([0, 1, 0, 1])

    # Validation set contains novel category "green" never seen in training
    X_val = pd.DataFrame({
        "color": ["green", "red"],
        "num": [5.0, 6.0],
    })

    pipeline = build_fold_pipeline(
        model_type="logistic_regression",
        numeric_features=["num"],
        categorical_features=["color"],
        random_state=42,
    )

    pipeline.fit(X_train, y_train)

    encoder = pipeline.named_steps["preprocessor"].named_transformers_["categorical"].named_steps["onehot"]
    learned_categories = list(encoder.categories_[0])

    # Encoder only learned training categories
    assert learned_categories == ["blue", "red"]
    assert "green" not in learned_categories

    # Transforms validation set with unseen "green" without crashing (encodes as zeros)
    preds = pipeline.predict(X_val)
    assert len(preds) == 2


def test_tc_leak_04_zero_global_fitting():
    """TC-LEAK-04: Confirms evaluation runner fits pipelines on fold train slices only, never on full X."""
    n_total = 20
    df = pd.DataFrame({
        "feat_num": range(n_total),
        "feat_cat": ["A", "B"] * 10,
        "target": [0, 1] * 10,
    })

    fitted_shapes = []

    # Intercept build_fold_pipeline to wrap fit method and record input shapes
    original_builder = build_fold_pipeline

    def spy_builder(*args, **kwargs):
        pipe = original_builder(*args, **kwargs)
        original_fit = pipe.fit

        def spy_fit(X, y):
            fitted_shapes.append(len(X))
            return original_fit(X, y)

        pipe.fit = spy_fit
        return pipe

    with patch("mlforge.engine.evaluator.build_fold_pipeline", side_effect=spy_builder):
        result = evaluate_classification(df, target_column="target", num_folds=4, random_state=42)

    # 4 folds * 2 models (dummy + logistic) = 8 fit calls
    assert len(fitted_shapes) == 8

    # Each fold training set is 15 rows (3/4 of 20). None should ever be 20 (n_total).
    for shape in fitted_shapes:
        assert shape == 15
        assert shape != n_total
