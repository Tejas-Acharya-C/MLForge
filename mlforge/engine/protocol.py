"""MLForge Evaluation Protocol and Splitting.

Provides deterministic, reproducible cross-validation fold generation
with strict stratification and class-count verification.
"""

import hashlib
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from mlforge.core.exceptions import ConfigurationError, InsufficientClassInstancesError


def compute_partition_fingerprint(train_indices: np.ndarray, val_indices: np.ndarray) -> str:
    """Generate a deterministic cryptographic fingerprint of fold partition indices."""
    sorted_train = np.sort(np.asarray(train_indices, dtype=np.int64))
    sorted_val = np.sort(np.asarray(val_indices, dtype=np.int64))
    h = hashlib.sha256()
    h.update(sorted_train.tobytes())
    h.update(b"::")
    h.update(sorted_val.tobytes())
    return h.hexdigest()[:16]


class FoldSplit:
    """Represents a single cross-validation fold with train and validation index arrays."""

    def __init__(self, fold_index: int, train_indices: np.ndarray, val_indices: np.ndarray):
        self.fold_index = fold_index
        self.train_indices = train_indices
        self.val_indices = val_indices

    @property
    def train_count(self) -> int:
        return len(self.train_indices)

    @property
    def val_count(self) -> int:
        return len(self.val_indices)

    @property
    def partition_fingerprint(self) -> str:
        return compute_partition_fingerprint(self.train_indices, self.val_indices)


def generate_stratified_folds(
    X: pd.DataFrame,
    y: pd.Series,
    num_folds: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> List[FoldSplit]:
    """Generate deterministic stratified cross-validation folds.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target classification labels.
    num_folds : int, default=5
        Number of cross-validation folds. Must be >= 2.
    shuffle : bool, default=True
        Whether to shuffle data before splitting into folds.
    random_state : int, default=42
        Deterministic random seed for reproducibility.

    Returns
    -------
    List[FoldSplit]
        Ordered list of FoldSplit objects containing train and validation indices.

    Raises
    ------
    ConfigurationError
        If num_folds is < 2.
    InsufficientClassInstancesError
        If any class in y has fewer than num_folds samples.
    """
    if num_folds < 2:
        raise ConfigurationError(f"num_folds must be >= 2, got {num_folds}.")

    # Verify class counts
    class_counts = y.value_counts()
    for label, count in class_counts.items():
        if count < num_folds:
            raise InsufficientClassInstancesError(
                class_label=str(label),
                count=int(count),
                required_folds=num_folds,
            )

    skf = StratifiedKFold(n_splits=num_folds, shuffle=shuffle, random_state=random_state)
    folds: List[FoldSplit] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        folds.append(
            FoldSplit(
                fold_index=fold_idx,
                train_indices=train_idx,
                val_indices=val_idx,
            )
        )

    return folds
