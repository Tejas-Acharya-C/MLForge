"""Unit tests for deterministic cross-validation protocol and splitting.

Covers test cases TC-SPLIT-01, TC-SPLIT-02, TC-REP-01, TC-REP-02.
"""

import numpy as np
import pandas as pd
import pytest

from mlforge.core.exceptions import ConfigurationError, InsufficientClassInstancesError
from mlforge.engine.protocol import generate_stratified_folds


def test_tc_split_02_stratification_preservation():
    """TC-SPLIT-02: Each fold preserves class ratio approximately proportional to dataset."""
    # 80/20 class ratio: 80 class '0', 20 class '1' = 100 rows
    np.random.seed(42)
    y = pd.Series([0] * 80 + [1] * 20)
    X = pd.DataFrame({"feat": np.random.randn(100)})

    folds = generate_stratified_folds(X, y, num_folds=5, shuffle=True, random_state=42)

    assert len(folds) == 5
    for fold in folds:
        val_y = y.iloc[fold.val_indices]
        # 20 samples in each validation fold (16 of class 0, 4 of class 1)
        assert len(val_y) == 20
        assert (val_y == 0).sum() == 16
        assert (val_y == 1).sum() == 4


def test_every_sample_appears_in_exactly_one_validation_fold():
    """Standard K-fold partition invariant: all samples partitioned across validation folds."""
    n_samples = 60
    y = pd.Series([0] * 30 + [1] * 30)
    X = pd.DataFrame({"feat": range(n_samples)})

    folds = generate_stratified_folds(X, y, num_folds=5, shuffle=True, random_state=42)

    seen_indices = []
    for fold in folds:
        seen_indices.extend(fold.val_indices.tolist())

    # Every sample appears exactly once
    assert sorted(seen_indices) == list(range(n_samples))


def test_tc_rep_01_identical_seeds_yield_identical_folds():
    """TC-REP-01: Identical seeds produce bit-for-bit identical fold splits."""
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    X = pd.DataFrame({"feat": range(10)})

    folds_run1 = generate_stratified_folds(X, y, num_folds=2, shuffle=True, random_state=42)
    folds_run2 = generate_stratified_folds(X, y, num_folds=2, shuffle=True, random_state=42)

    for f1, f2 in zip(folds_run1, folds_run2):
        np.testing.assert_array_equal(f1.train_indices, f2.train_indices)
        np.testing.assert_array_equal(f1.val_indices, f2.val_indices)


def test_tc_rep_02_different_seeds_yield_different_splits():
    """TC-REP-02: Different seeds yield different split orderings."""
    y = pd.Series([0, 1] * 25)
    X = pd.DataFrame({"feat": range(50)})

    folds_seed42 = generate_stratified_folds(X, y, num_folds=5, shuffle=True, random_state=42)
    folds_seed99 = generate_stratified_folds(X, y, num_folds=5, shuffle=True, random_state=99)

    # At least one validation fold must have different indices
    assert not all(
        np.array_equal(f1.val_indices, f2.val_indices)
        for f1, f2 in zip(folds_seed42, folds_seed99)
    )


def test_invalid_num_folds_raises_configuration_error():
    """num_folds < 2 must raise ConfigurationError."""
    y = pd.Series([0, 1, 0, 1])
    X = pd.DataFrame({"feat": [1, 2, 3, 4]})

    with pytest.raises(ConfigurationError, match="num_folds must be >= 2"):
        generate_stratified_folds(X, y, num_folds=1)
