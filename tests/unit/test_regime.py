"""Unit tests for regime-switching detectors.

Tests MarkovRegimeDetector, HMMRegimeDetector, GARCHRegimeDetector, and
label_signal_state against synthetic data with known regime properties.

All tests use fixed random seeds for determinism.
"""

import numpy as np
import pytest

from src.detectors.regime import (
    GARCHRegimeDetector,
    HMMRegimeDetector,
    MarkovRegimeDetector,
    label_signal_state,
)
from src.utils.synthetic import (
    generate_regime_switching,
    generate_variance_shift,
)


# ---------------------------------------------------------------------------
# MarkovRegimeDetector
# ---------------------------------------------------------------------------

class TestMarkovRegimeDetector:
    """Tests for MarkovRegimeDetector."""

    def test_markov_detects_two_regimes(self):
        """Fit on 2-regime switching data and verify 2 distinct labels."""
        series, meta = generate_regime_switching(
            n=1000,
            n_regimes=2,
            regime_means=[0.05, -0.05],
            regime_stds=[0.01, 0.01],
            transition_matrix=[[0.98, 0.02], [0.02, 0.98]],
            seed=42,
        )
        detector = MarkovRegimeDetector(n_regimes=2)
        detector.fit(series)
        regimes = detector.get_regimes()
        distinct_labels = set(regimes)
        assert len(distinct_labels) == 2, (
            f"Expected 2 distinct regime labels, got {len(distinct_labels)}: {distinct_labels}"
        )

    def test_markov_regime_means_approximate(self):
        """Verify detected regime means are within 50% of true means."""
        true_means = [0.05, -0.05]
        series, meta = generate_regime_switching(
            n=2000,
            n_regimes=2,
            regime_means=true_means,
            regime_stds=[0.01, 0.01],
            transition_matrix=[[0.98, 0.02], [0.02, 0.98]],
            seed=123,
        )
        detector = MarkovRegimeDetector(n_regimes=2)
        detector.fit(series)
        stats = detector.get_regime_statistics()

        # Collect detected means and sort them
        detected_means = sorted([stats[k]["mean"] for k in stats])
        sorted_true = sorted(true_means)

        for detected, true_val in zip(detected_means, sorted_true):
            # Within 50% of the true mean
            assert abs(detected - true_val) <= 0.5 * abs(true_val), (
                f"Detected mean {detected:.4f} not within 50% of true mean {true_val:.4f}"
            )

    def test_markov_transition_matrix_shape(self):
        """Verify transition matrix shape is (n_regimes, n_regimes)."""
        series, meta = generate_regime_switching(
            n=1000,
            n_regimes=2,
            regime_means=[0.05, -0.05],
            regime_stds=[0.01, 0.01],
            transition_matrix=[[0.98, 0.02], [0.02, 0.98]],
            seed=42,
        )
        detector = MarkovRegimeDetector(n_regimes=2)
        detector.fit(series)
        trans = detector.get_transition_matrix()
        # statsmodels may return (k, k, 1) for single-variable models;
        # squeeze to (k, k) for the shape check
        effective_shape = trans.squeeze().shape
        assert effective_shape == (2, 2), f"Expected shape (2, 2), got {effective_shape}"

    def test_markov_no_nan_input(self):
        """Series with NaN raises ValueError."""
        series = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        detector = MarkovRegimeDetector(n_regimes=2)
        with pytest.raises(ValueError, match="NaN"):
            detector.fit(series)


# ---------------------------------------------------------------------------
# HMMRegimeDetector
# ---------------------------------------------------------------------------

class TestHMMRegimeDetector:
    """Tests for HMMRegimeDetector."""

    def test_hmm_detects_two_regimes(self):
        """Fit on 2-regime switching data and verify 2 distinct labels."""
        series, meta = generate_regime_switching(
            n=1000,
            n_regimes=2,
            regime_means=[0.05, -0.05],
            regime_stds=[0.01, 0.01],
            transition_matrix=[[0.98, 0.02], [0.02, 0.98]],
            seed=42,
        )
        detector = HMMRegimeDetector(n_regimes=2)
        detector.fit(series)
        regimes = detector.get_regimes()
        distinct_labels = set(regimes)
        assert len(distinct_labels) == 2, (
            f"Expected 2 distinct regime labels, got {len(distinct_labels)}: {distinct_labels}"
        )

    def test_hmm_select_n_regimes(self):
        """Generate 3-regime data, verify select_n_regimes returns 2 or 3."""
        series, meta = generate_regime_switching(
            n=2000,
            n_regimes=3,
            regime_means=[0.08, 0.0, -0.08],
            regime_stds=[0.01, 0.01, 0.01],
            transition_matrix=[
                [0.96, 0.02, 0.02],
                [0.02, 0.96, 0.02],
                [0.02, 0.02, 0.96],
            ],
            seed=77,
        )
        detector = HMMRegimeDetector(n_regimes=2)
        best_k = detector.select_n_regimes(series, max_regimes=4)
        assert best_k in (2, 3), (
            f"Expected select_n_regimes to return 2 or 3, got {best_k}"
        )


# ---------------------------------------------------------------------------
# GARCHRegimeDetector
# ---------------------------------------------------------------------------

class TestGARCHRegimeDetector:
    """Tests for GARCHRegimeDetector."""

    def test_garch_detects_vol_shift(self):
        """Generate variance shift data, verify breakpoint near true break."""
        true_break = 500
        series, meta = generate_variance_shift(
            n=1000,
            break_points=[true_break],
            stds=[0.01, 0.05],
            mean=0.0,
            seed=42,
        )
        detector = GARCHRegimeDetector(vol_model="GARCH", p=1, q=1)
        detector.fit(series, penalty_value=3.0)
        breakpoints = detector.get_vol_breakpoints()

        assert len(breakpoints) >= 1, "Expected at least one breakpoint detected"
        # At least one breakpoint should be within 100 observations of the true break
        distances = [abs(int(bp) - true_break) for bp in breakpoints]
        min_distance = min(distances)
        assert min_distance <= 100, (
            f"Closest breakpoint is {min_distance} obs from true break at {true_break}; "
            f"detected breakpoints: {breakpoints.tolist()}"
        )

    def test_garch_conditional_volatility_shape(self):
        """Verify conditional volatility output length matches input."""
        series, meta = generate_variance_shift(
            n=500,
            break_points=[250],
            stds=[0.01, 0.03],
            mean=0.0,
            seed=42,
        )
        detector = GARCHRegimeDetector(vol_model="GARCH", p=1, q=1)
        detector.fit(series)
        cond_vol = detector.get_conditional_volatility()
        assert len(cond_vol) == len(series), (
            f"Expected conditional volatility length {len(series)}, got {len(cond_vol)}"
        )


# ---------------------------------------------------------------------------
# label_signal_state
# ---------------------------------------------------------------------------

class TestLabelSignalState:
    """Tests for the label_signal_state function."""

    def test_label_two_regimes(self):
        """Highest mean labeled alpha_generating, lowest labeled decayed."""
        means = {0: 0.01, 1: -0.003}
        labels = label_signal_state(means)
        assert labels[0] == "alpha_generating"
        assert labels[1] == "decayed"

    def test_label_three_regimes(self):
        """Middle regime labeled neutral."""
        means = {0: 0.01, 1: -0.003, 2: 0.002}
        labels = label_signal_state(means)
        # Sorted by mean: 1 (-0.003) < 2 (0.002) < 0 (0.01)
        assert labels[1] == "decayed"
        assert labels[2] == "neutral"
        assert labels[0] == "alpha_generating"
