"""Unit tests for MLForge dataset contracts and validation.

Covers test cases TC-DS-01 through TC-DS-04, TC-SPLIT-01, and edge cases.
"""

import numpy as np
import pandas as pd
import pytest

from mlforge.core.dataset import validate_and_extract_dataset
from mlforge.core.exceptions import (
    DatasetValidationError,
    InsufficientClassInstancesError,
)


def test_tc_ds_01_valid_binary_dataset():
    """TC-DS-01: Valid binary dataset ingestion and metadata calculation."""
    df = pd.DataFrame({
        "age": [25, 45, 35, 50, 23, 37],
        "city": ["NY", "SF", "NY", "LA", "SF", "LA"],
        "income": [50000.0, 120000.0, 75000.0, 90000.0, 48000.0, 82000.0],
        "target": [0, 1, 0, 1, 0, 1],
    })

    X, y, metadata = validate_and_extract_dataset(df, target_column="target")

    assert X.shape == (6, 3)
    assert len(y) == 6
    assert metadata.num_rows == 6
    assert metadata.num_features == 3
    assert metadata.numeric_feature_count == 2
    assert metadata.categorical_feature_count == 1
    assert metadata.target_name == "target"
    assert metadata.class_distribution == {"0": 3, "1": 3}
    assert metadata.class_labels == ["0", "1"]
    assert metadata.imbalance_ratio == 1.0
    assert len(metadata.checksum_sha256) == 64


def test_tc_ds_01_valid_multiclass_dataset():
    """TC-DS-01: Valid multiclass dataset with 3 classes."""
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "feature_2": [10, 20, 30, 40, 50, 60],
        "label": ["setosa", "versicolor", "virginica", "setosa", "versicolor", "virginica"],
    })

    X, y, metadata = validate_and_extract_dataset(df, target_column="label")

    assert X.shape == (6, 2)
    assert metadata.class_labels == ["setosa", "versicolor", "virginica"]
    assert metadata.class_distribution["setosa"] == 2


def test_tc_ds_02_missing_target_column():
    """TC-DS-02: Target column missing from dataframe."""
    df = pd.DataFrame({
        "feature_a": [1, 2, 3],
        "feature_b": [4, 5, 6],
    })

    with pytest.raises(DatasetValidationError, match="Target column 'nonexistent' not found"):
        validate_and_extract_dataset(df, target_column="nonexistent")


def test_tc_ds_03_constant_feature_warning():
    """TC-DS-03: Feature column with constant value triggers warning without dropping."""
    df = pd.DataFrame({
        "constant_col": [1, 1, 1, 1],
        "varying_col": [10, 20, 30, 40],
        "target": [0, 1, 0, 1],
    })

    X, y, metadata = validate_and_extract_dataset(df, target_column="target")

    assert "constant_col" in metadata.constant_columns
    assert any("constant or zero variance" in w for w in metadata.warnings)
    assert "constant_col" in X.columns  # Preserved in features for fold pipeline to handle


def test_tc_ds_04_non_dataframe_input():
    """TC-DS-04: Non-dataframe inputs must be rejected."""
    with pytest.raises(DatasetValidationError, match="Expected input dataset to be a pandas DataFrame"):
        validate_and_extract_dataset([[1, 2], [3, 4]], target_column="target")


def test_empty_dataset_rejected():
    """Empty dataframe must raise validation error."""
    df = pd.DataFrame()
    with pytest.raises(DatasetValidationError, match="Dataset is empty"):
        validate_and_extract_dataset(df, target_column="target")


def test_no_usable_features_rejected():
    """Dataset with only target column has no usable features."""
    df = pd.DataFrame({"target": [0, 1, 0, 1]})
    with pytest.raises(DatasetValidationError, match="no feature columns"):
        validate_and_extract_dataset(df, target_column="target")


def test_duplicate_column_names_rejected():
    """Duplicate column names must raise explicit validation error."""
    df = pd.DataFrame([[1, 2, 0], [3, 4, 1]], columns=["feat", "feat", "target"])
    with pytest.raises(DatasetValidationError, match="duplicate column names"):
        validate_and_extract_dataset(df, target_column="target")


def test_target_with_missing_values_rejected():
    """Missing values in target column must be rejected."""
    df = pd.DataFrame({
        "feature": [1, 2, 3, 4],
        "target": [0, 1, np.nan, 1],
    })
    with pytest.raises(DatasetValidationError, match="contains 1 missing value"):
        validate_and_extract_dataset(df, target_column="target")


def test_target_single_class_rejected():
    """Target with only one distinct class must be rejected."""
    df = pd.DataFrame({
        "feature": [1, 2, 3, 4],
        "target": ["approved", "approved", "approved", "approved"],
    })
    with pytest.raises(DatasetValidationError, match="contains only 1 class"):
        validate_and_extract_dataset(df, target_column="target")


def test_infinite_numeric_values_rejected():
    """Infinite numeric values in features must be rejected."""
    df = pd.DataFrame({
        "feature_clean": [1.0, 2.0, 3.0],
        "feature_inf": [10.0, np.inf, 30.0],
        "target": [0, 1, 0],
    })
    with pytest.raises(DatasetValidationError, match="contains 1 infinite numeric value"):
        validate_and_extract_dataset(df, target_column="target")


def test_tc_split_01_insufficient_class_instances():
    """TC-SPLIT-01: Class with fewer samples than k_folds raises InsufficientClassInstancesError."""
    # Class '1' has only 3 samples, but k_folds = 5
    df = pd.DataFrame({
        "feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "target": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    })

    with pytest.raises(InsufficientClassInstancesError) as exc_info:
        validate_and_extract_dataset(df, target_column="target", k_folds=5)

    assert exc_info.value.class_label == "1"
    assert exc_info.value.count == 3
    assert exc_info.value.required_folds == 5
    assert "fewer than the configured 5 cross-validation folds" in str(exc_info.value)


def test_tc_split_03_singleton_class():
    """TC-SPLIT-03: Extreme multiclass dataset with a singleton class caught before fold generation."""
    # Multiclass dataset where class 'rare' has only 1 sample, while other classes satisfy k_folds=5
    df = pd.DataFrame({
        "feat1": [1.0] * 11,
        "feat2": [2.0] * 11,
        "target": [
            "class_a", "class_a", "class_a", "class_a", "class_a",
            "class_b", "class_b", "class_b", "class_b", "class_b",
            "rare",
        ],
    })

    with pytest.raises(InsufficientClassInstancesError) as exc_info:
        validate_and_extract_dataset(df, target_column="target", k_folds=5)

    assert exc_info.value.class_label == "rare"
    assert exc_info.value.count == 1
    assert exc_info.value.required_folds == 5
    assert "fewer than the configured 5 cross-validation folds" in str(exc_info.value)



def test_caller_dataframe_not_mutated():
    """Original input dataframe must not be mutated in any way."""
    original_df = pd.DataFrame({
        "feature": [10, 20, 30, 40],
        "target": [0, 1, 0, 1],
    })
    df_copy = original_df.copy(deep=True)

    X, y, metadata = validate_and_extract_dataset(original_df, target_column="target")

    pd.testing.assert_frame_equal(original_df, df_copy)
    assert list(original_df.columns) == ["feature", "target"]


def test_missing_feature_values_allowed_and_tracked():
    """Missing values in features are permitted (to be handled in fold pipeline) and counted in metadata."""
    df = pd.DataFrame({
        "numeric_missing": [1.0, np.nan, 3.0, 4.0],
        "cat_missing": ["A", "B", None, "A"],
        "target": [0, 1, 0, 1],
    })

    X, y, metadata = validate_and_extract_dataset(df, target_column="target")

    assert metadata.missing_value_counts["numeric_missing"] == 1
    assert metadata.missing_value_counts["cat_missing"] == 1
    assert metadata.missing_value_counts["target"] == 0
    assert X.shape == (4, 2)
