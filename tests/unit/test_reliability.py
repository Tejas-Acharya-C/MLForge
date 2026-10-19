"""Unit Tests for Probability Calibration and Reliability Engine (Milestone 2.4)."""

import math
import numpy as np
import pytest

from mlforge.engine.reliability import compute_calibration_curve


def test_hand_calculated_ece_and_brier():
    """Verify ECE, MCE, and Brier score against exact hand-calculated reference values.

    Reference scenario:
    N = 4, true labels = [1, 1, 0, 0], predicted probabilities = [0.8, 0.7, 0.2, 0.3].
    num_bins = 2 (bins: [0.0, 0.5) and [0.5, 1.0]).

    Bin 0: [0.2, 0.3], count=2, mean_p = 0.25, obs_rate = 0.0 / 2 = 0.0, diff = 0.25.
    Bin 1: [0.8, 0.7], count=2, mean_p = 0.75, obs_rate = 2.0 / 2 = 1.0, diff = 0.25.

    Expected ECE = (2/4)*0.25 + (2/4)*0.25 = 0.25.
    Expected MCE = max(0.25, 0.25) = 0.25.
    Expected Brier = 1/4 * ((0.8-1)^2 + (0.7-1)^2 + (0.2-0)^2 + (0.3-0)^2)
                   = 1/4 * (0.04 + 0.09 + 0.04 + 0.09) = 0.26 / 4 = 0.065.
    """
    y_true = [1, 1, 0, 0]
    y_prob = [0.8, 0.7, 0.2, 0.3]

    res = compute_calibration_curve(
        y_true=y_true,
        y_prob=y_prob,
        positive_label=1,
        num_bins=2,
        strategy="uniform",
    )

    assert res.total_samples == 4
    assert res.num_bins == 2
    assert res.is_usable is True
    assert res.expected_calibration_error == 0.25
    assert res.max_calibration_error == 0.25
    assert res.brier_score == 0.065

    # Check bins
    b0 = res.bins[0]
    assert b0.bin_index == 0
    assert b0.sample_count == 2
    assert b0.mean_predicted_prob == 0.25
    assert b0.observed_positive_rate == 0.0
    assert b0.is_empty is False

    b1 = res.bins[1]
    assert b1.bin_index == 1
    assert b1.sample_count == 2
    assert b1.mean_predicted_prob == 0.75
    assert b1.observed_positive_rate == 1.0
    assert b1.is_empty is False


def test_perfectly_calibrated_probabilities():
    """A perfectly calibrated set of predictions should yield ECE == 0.0."""
    # 50 samples in bin around 0.2 with 10 positives (obs_rate = 0.2)
    # 50 samples in bin around 0.8 with 40 positives (obs_rate = 0.8)
    y_true = [1] * 10 + [0] * 40 + [1] * 40 + [0] * 10
    y_prob = [0.2] * 50 + [0.8] * 50

    res = compute_calibration_curve(y_true=y_true, y_prob=y_prob, num_bins=10, strategy="uniform")

    assert res.expected_calibration_error == 0.0
    assert res.max_calibration_error == 0.0


def test_severely_miscalibrated_probabilities():
    """An overconfident model predicting 0.95 on all-negative data should yield ECE == 0.95."""
    y_true = [0] * 100
    y_prob = [0.95] * 100

    res = compute_calibration_curve(y_true=y_true, y_prob=y_prob, positive_label=1, num_bins=10)

    assert res.expected_calibration_error == 0.95
    assert res.max_calibration_error == 0.95
    assert len(res.warnings) > 0  # Warns about single class partition


def test_boundary_probabilities_0_and_1():
    """Probabilities exactly equal to 0.0 and 1.0 must fall into bin 0 and bin M-1 respectively."""
    y_true = [0, 1]
    y_prob = [0.0, 1.0]

    res = compute_calibration_curve(y_true=y_true, y_prob=y_prob, num_bins=5, strategy="uniform")

    assert res.bins[0].sample_count == 1
    assert res.bins[0].mean_predicted_prob == 0.0
    assert res.bins[0].observed_positive_rate == 0.0

    assert res.bins[4].sample_count == 1
    assert res.bins[4].mean_predicted_prob == 1.0
    assert res.bins[4].observed_positive_rate == 1.0

    assert res.expected_calibration_error == 0.0
    assert res.brier_score == 0.0


def test_empty_bins_handling():
    """Bins with zero samples must be safely handled as empty without distorting ECE."""
    # All samples concentrated strictly in [0.45, 0.55], so 8 out of 10 bins are empty
    y_true = [1, 0, 1, 0]
    y_prob = [0.5, 0.5, 0.5, 0.5]

    res = compute_calibration_curve(y_true=y_true, y_prob=y_prob, num_bins=10, strategy="uniform")

    empty_bins = [b for b in res.bins if b.is_empty]
    assert len(empty_bins) == 9  # Only bin covering 0.5 is non-empty
    for eb in empty_bins:
        assert eb.sample_count == 0
        assert eb.mean_predicted_prob is None
        assert eb.observed_positive_rate is None

    assert res.expected_calibration_error == 0.0  # mean_p = 0.5, obs_rate = 0.5, diff = 0.0


def test_invalid_and_non_finite_inputs():
    """Ensure invalid, NaN, infinite, or mismatched inputs raise explicit ValueError."""
    # Empty inputs
    with pytest.raises(ValueError, match="Evaluation sample size must be at least 1"):
        compute_calibration_curve([], [])

    # None inputs
    with pytest.raises(ValueError, match="must not be None"):
        compute_calibration_curve(None, [0.5])

    # Mismatched lengths
    with pytest.raises(ValueError, match="Mismatched lengths"):
        compute_calibration_curve([0, 1], [0.5])

    # NaN in probabilities
    with pytest.raises(ValueError, match="Non-finite values"):
        compute_calibration_curve([0, 1], [0.5, np.nan])

    # Inf in probabilities
    with pytest.raises(ValueError, match="Non-finite values"):
        compute_calibration_curve([0, 1], [0.5, np.inf])

    # Out of bounds probabilities
    with pytest.raises(ValueError, match="Predicted probabilities must fall strictly in"):
        compute_calibration_curve([0, 1], [-0.1, 0.5])

    with pytest.raises(ValueError, match="Predicted probabilities must fall strictly in"):
        compute_calibration_curve([0, 1], [0.5, 1.1])

    # More than 2 classes
    with pytest.raises(ValueError, match="Calibration analysis requires binary outcomes"):
        compute_calibration_curve([0, 1, 2], [0.1, 0.5, 0.9])

    # Invalid num_bins
    with pytest.raises(ValueError, match="num_bins must be an integer >= 2"):
        compute_calibration_curve([0, 1], [0.2, 0.8], num_bins=1)

    # Invalid strategy
    with pytest.raises(ValueError, match="Unsupported binning strategy"):
        compute_calibration_curve([0, 1], [0.2, 0.8], strategy="invalid_strategy")


def test_2d_probability_array_handling():
    """Scikit-learn predict_proba 2D output ([N, 2]) should be parsed seamlessly."""
    y_true = [0, 1, 0, 1]
    y_prob_2d = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
        [0.9, 0.1],
        [0.1, 0.9],
    ])

    res = compute_calibration_curve(y_true=y_true, y_prob=y_prob_2d, num_bins=5)
    assert res.total_samples == 4
    assert res.is_usable is True


def test_quantile_binning_strategy():
    """Quantile binning should partition observations into approximately equal frequency bins."""
    rng = np.random.RandomState(42)
    y_prob = rng.uniform(0.0, 1.0, 100)
    y_true = (y_prob > 0.5).astype(int)

    res = compute_calibration_curve(
        y_true=y_true,
        y_prob=y_prob,
        num_bins=5,
        strategy="quantile",
    )

    assert res.bin_strategy == "quantile"
    assert res.is_usable is True
    # Non-empty bins should each have roughly equal sample support (~20 samples)
    for b in res.bins:
        assert b.sample_count >= 15


def test_tc_cal_03_dummy_model_calibration():
    """TC-CAL-03: DummyClassifier(strategy='prior') emits flat probability equal to prior, landing in a single bin."""
    from sklearn.dummy import DummyClassifier

    # 100 samples: 70 ones, 30 zeros -> prior is 0.70
    X = np.zeros((100, 2))
    y = np.array([1] * 70 + [0] * 30)

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X, y)

    probs = dummy.predict_proba(X)
    # Binary probabilities: column 1 is probability of class 1, which is exactly 0.70
    assert np.allclose(probs[:, 1], 0.70)

    res = compute_calibration_curve(
        y_true=y,
        y_prob=probs,
        positive_label=1,
        num_bins=10,
        strategy="uniform",
    )

    assert res.is_usable is True
    assert res.total_samples == 100

    # Exactly 1 non-empty bin (covering [0.7, 0.8)) containing all 100 samples
    non_empty_bins = [b for b in res.bins if not b.is_empty]
    assert len(non_empty_bins) == 1

    target_bin = non_empty_bins[0]
    assert target_bin.sample_count == 100
    assert target_bin.mean_predicted_prob == pytest.approx(0.70, abs=1e-5)
    assert target_bin.observed_positive_rate == pytest.approx(0.70, abs=1e-5)

    # Empty bins must be 9
    empty_bins = [b for b in res.bins if b.is_empty]
    assert len(empty_bins) == 9

    # Perfectly calibrated relative to the marginal prior, ECE approaches 0.0
    assert res.expected_calibration_error == pytest.approx(0.0, abs=1e-5)
