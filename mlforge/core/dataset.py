"""MLForge Dataset Contracts and Validation.

Provides rigorous tabular dataset validation, feature/target extraction,
and metadata generation for classification evaluation.
"""

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from mlforge.core.contracts import DatasetMetadata
from mlforge.core.exceptions import (
    DatasetValidationError,
    InsufficientClassInstancesError,
)


def validate_and_extract_dataset(
    df: Any,
    target_column: str,
    k_folds: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series, DatasetMetadata]:
    """Validate a tabular classification dataset and extract features, target, and metadata.

    Parameters
    ----------
    df : Any
        The input tabular dataset, expected to be a pandas DataFrame.
    target_column : str
        The name of the classification target column.
    k_folds : Optional[int]
        If provided, validates that every class count is >= k_folds for stratified splitting.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, DatasetMetadata]
        - X: Feature DataFrame (unmutated copy, aligned index).
        - y: Target Series (aligned index).
        - metadata: Structured metadata and integrity warnings.

    Raises
    ------
    DatasetValidationError
        If the dataset is non-tabular, empty, has duplicate columns, missing target column,
        missing target values, fewer than two classes, zero features, or infinite values.
    InsufficientClassInstancesError
        If k_folds is specified and any class count is strictly less than k_folds.
    """
    # 1. Type validation
    if not isinstance(df, pd.DataFrame):
        raise DatasetValidationError(
            f"Expected input dataset to be a pandas DataFrame, but received {type(df).__name__}."
        )

    # 2. Dimensions check
    if df.empty or len(df) == 0:
        raise DatasetValidationError("Dataset is empty. At least one row of data is required.")

    # 3. Duplicate column names check
    if len(df.columns) != len(set(df.columns)):
        duplicates = [col for col in df.columns if list(df.columns).count(col) > 1]
        raise DatasetValidationError(
            f"Dataset contains duplicate column names: {sorted(list(set(duplicates)))}. "
            "Column names must be unique."
        )

    # 4. Target column presence check
    if not isinstance(target_column, str) or not target_column.strip():
        raise DatasetValidationError("Target column name must be a non-empty string.")

    if target_column not in df.columns:
        raise DatasetValidationError(
            f"Target column '{target_column}' not found in dataset columns: {list(df.columns)}."
        )

    # 5. Extract target and features defensively without mutating caller's DataFrame
    y = df[target_column].copy()
    X = df.drop(columns=[target_column]).copy()

    # 6. Feature count check
    if X.shape[1] == 0:
        raise DatasetValidationError(
            f"Dataset contains no feature columns after removing target '{target_column}'."
        )

    # 7. Target missing values check
    target_null_count = int(y.isna().sum())
    if target_null_count > 0:
        raise DatasetValidationError(
            f"Target column '{target_column}' contains {target_null_count} missing value(s). "
            "Classification targets must not contain null/NaN values."
        )

    # 8. Target class count check
    unique_classes = y.unique()
    if len(unique_classes) < 2:
        raise DatasetValidationError(
            f"Target column '{target_column}' contains only {len(unique_classes)} class "
            f"({unique_classes[0] if len(unique_classes) == 1 else 'none'}). "
            "Classification requires at least two distinct classes."
        )

    # 9. Infinite numeric values check in features
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        col_series = X[col]
        if np.isinf(col_series).any():
            inf_count = int(np.isinf(col_series).sum())
            raise DatasetValidationError(
                f"Feature column '{col}' contains {inf_count} infinite numeric value(s). "
                "Infinite values are unsupported; replace them with valid finite numbers or NaN."
            )

    # 10. Class counts vs CV fold requirements check
    class_counts: Dict[str, int] = {str(k): int(v) for k, v in y.value_counts().items()}
    if k_folds is not None:
        if k_folds < 2:
            raise DatasetValidationError(f"k_folds must be >= 2, got {k_folds}.")
        for label, count in class_counts.items():
            if count < k_folds:
                raise InsufficientClassInstancesError(
                    class_label=label,
                    count=count,
                    required_folds=k_folds,
                )

    # 11. Inspect categorical features and constant columns
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    warnings: List[str] = []

    constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if constant_cols:
        warnings.append(
            f"Feature column(s) {constant_cols} exhibit constant or zero variance (<= 1 unique value)."
        )

    # Missing value counts across all columns (including target)
    missing_value_counts = {str(col): int(df[col].isna().sum()) for col in df.columns}

    # Class distribution and imbalance ratio calculation
    sorted_labels = sorted(class_counts.keys())
    counts_list = list(class_counts.values())
    max_count = max(counts_list)
    min_count = min(counts_list)
    imbalance_ratio = round(float(max_count / min_count), 4)

    if imbalance_ratio >= 5.0:
        warnings.append(
            f"Dataset displays significant class imbalance (ratio {imbalance_ratio:.2f}:1). "
            "Macro-averaged and balanced metrics should be prioritized over overall accuracy."
        )

    # 12. SHA-256 Checksum calculation over values
    try:
        # Generate deterministic hash using pandas object hashing
        hashed_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        checksum = hashlib.sha256(hashed_bytes).hexdigest()
    except Exception:
        # Fallback to string representation hashing
        checksum = hashlib.sha256(df.to_csv(index=True).encode("utf-8")).hexdigest()

    metadata = DatasetMetadata(
        num_rows=int(X.shape[0]),
        num_features=int(X.shape[1]),
        feature_names=list(X.columns),
        feature_types={str(k): str(v) for k, v in X.dtypes.items()},
        numeric_feature_count=len(numeric_cols),
        categorical_feature_count=len(categorical_cols),
        target_name=str(target_column),
        target_type=str(y.dtype),
        class_distribution=class_counts,
        class_labels=sorted_labels,
        imbalance_ratio=imbalance_ratio,
        missing_value_counts=missing_value_counts,
        constant_columns=constant_cols,
        checksum_sha256=checksum,
        warnings=warnings,
    )

    return X, y, metadata
