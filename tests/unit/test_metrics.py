"""Unit tests for evaluation metrics.

Tests out-of-sample R-squared, Clark-West, signal half-life, rolling Sharpe,
decay onset detection, and Giacomini-White test from src/evaluation/metrics.py.

All tests use fixed random seeds for determinism.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    clark_west_test,
    detect_decay_onset,
    giacomini_white_test,
    oos_r_squared,
    rolling_sharpe,
    signal_half_life,
)
from src.utils.synthetic import generate_factor_with_decay, generate_ou_process


# ===================================================================
# 1. oos_r_squared
# ===================================================================

class TestOosRSquared:

    def test_perfect_forecast(self):
        rng = np.random.default_rng(42)
        actual = rng.normal(0.001, 0.01, size=200)
        forecast = actual.copy()  # perfect forecast

        r2 = oos_r_squared(actual, forecast, forecast_benchmark=np.zeros(200))

        assert r2 == pytest.approx(1.0), (
            f"Perfect forecast should yield R²=1.0, got {r2}"
        )

    def test_mean_forecast(self):
        rng = np.random.default_rng(42)
        actual = rng.normal(0.0, 0.01, size=500)
        mean_val = np.mean(actual)
        forecast = np.full_like(actual, mean_val)
        benchmark = np.full_like(actual, mean_val)

        r2 = oos_r_squared(actual, forecast, forecast_benchmark=benchmark)

        # When forecast == benchmark, R² = 1 - SSE_model/SSE_bench = 1 - 1 = 0
        assert abs(r2) < 1e-10, (
            f"Mean forecast vs mean benchmark should give R²≈0, got {r2}"
        )

    def test_worse_than_mean(self):
        rng = np.random.default_rng(42)
        actual = rng.normal(0.0, 0.01, size=200)
        mean_val = np.mean(actual)
        # A deliberately bad forecast: opposite sign with large magnitude
        bad_forecast = -actual * 2.0
        benchmark = np.full_like(actual, mean_val)

        r2 = oos_r_squared(actual, bad_forecast, forecast_benchmark=benchmark)

        assert r2 < 0, f"Bad forecast should yield R²<0, got {r2}"


# ===================================================================
# 2. clark_west_test
# ===================================================================

class TestClarkWestTest:

    def test_with_known_better_model(self):
        rng = np.random.default_rng(42)
        actual = rng.normal(0.001, 0.01, size=500)
        # Restricted model: constant zero forecast
        forecast_restricted = np.zeros(500)
        # Unrestricted model: actual + small noise (genuinely better)
        forecast_unrestricted = actual + rng.normal(0, 0.002, size=500)

        result = clark_west_test(actual, forecast_restricted, forecast_unrestricted)

        assert result["reject"] is True, (
            f"Should reject null when unrestricted is better "
            f"(stat={result['statistic']:.3f}, p={result['p_value']:.4f})"
        )

    def test_with_equal_models(self):
        rng = np.random.default_rng(42)
        actual = rng.normal(0.0, 0.01, size=500)
        # Both models make the same (zero) forecast
        forecast = np.zeros(500)

        result = clark_west_test(actual, forecast, forecast)

        assert result["reject"] is False, (
            f"Should not reject when both forecasts are identical "
            f"(stat={result['statistic']}, p={result['p_value']})"
        )


# ===================================================================
# 3. signal_half_life
# ===================================================================

class TestSignalHalfLife:

    def test_ou_process(self):
        theta = 0.05
        true_half_life = np.log(2) / theta  # ~13.86

        series, meta = generate_ou_process(
            n=5000, theta=theta, mu=0.0, sigma=0.01, x0=0.05, seed=42
        )

        result = signal_half_life(series)
        estimated_hl = result["half_life"]

        # Allow 30% tolerance
        assert estimated_hl != np.inf, "Half-life should be finite for O-U process"
        assert abs(estimated_hl - true_half_life) / true_half_life < 0.30, (
            f"Estimated half-life {estimated_hl:.2f} not within 30% of "
            f"true {true_half_life:.2f}"
        )

    def test_random_walk_infinite(self):
        rng = np.random.default_rng(42)
        # Pure random walk: x_t = x_{t-1} + noise
        noise = rng.normal(0, 0.01, size=1000)
        rw = np.cumsum(noise)

        result = signal_half_life(rw)

        # A random walk has slope ~1 in the AR(1) regression, so the
        # estimated half-life should be inf (slope >= 1) or very large.
        # In finite samples the OLS slope can be slightly below 1, yielding
        # a large but finite half-life.
        assert result["half_life"] == np.inf or result["half_life"] > 50, (
            f"Random walk should have inf or large half-life, "
            f"got {result['half_life']:.2f}"
        )


# ===================================================================
# 4. rolling_sharpe
# ===================================================================

class TestRollingSharpe:

    def test_positive_returns(self):
        # Constant positive returns with tiny noise
        rng = np.random.default_rng(42)
        returns = 0.001 + rng.normal(0, 0.0001, size=300)

        sharpe = rolling_sharpe(returns, window=252, annualization=252)
        # After burn-in, all values should be positive
        valid = sharpe.dropna() if isinstance(sharpe, pd.Series) else sharpe[~np.isnan(sharpe)]

        assert len(valid) > 0, "Should have non-NaN values after burn-in"
        assert np.all(np.array(valid) > 0), (
            "Consistent positive returns should yield positive rolling Sharpe"
        )

    def test_output_length(self):
        rng = np.random.default_rng(42)
        n = 500
        returns = rng.normal(0.0, 0.01, size=n)

        sharpe = rolling_sharpe(returns, window=252, annualization=252)

        assert len(sharpe) == n, (
            f"Output length {len(sharpe)} should match input length {n}"
        )


# ===================================================================
# 5. detect_decay_onset
# ===================================================================

class TestDetectDecayOnset:

    def test_detects_decay_in_factor_lifecycle(self):
        # Use a strong signal so the rolling Sharpe clearly declines.
        series, meta = generate_factor_with_decay(
            alpha_period=800,
            decay_period=400,
            dead_period=400,
            alpha_mean=0.002,
            decay_end_mean=0.0,
            dead_mean=-0.001,
            std=0.005,
            seed=42,
        )

        # Compute rolling Sharpe as the signal quality metric
        sharpe = rolling_sharpe(
            pd.Series(series), window=60, annualization=252
        )

        # Use the threshold method: flag when the Sharpe ratio drops
        # below 2.0 for 5+ consecutive periods.  During the alpha phase
        # the Sharpe is high (~6); during/after decay it drops well below 2.
        result = detect_decay_onset(sharpe, method="threshold", threshold=2.0)

        true_decay_start = meta["break_points"][0]  # 800

        assert result["decay_onset_index"] is not None, (
            "Should detect decay onset in factor lifecycle"
        )
        detected = result["decay_onset_index"]
        # The detected onset should be in the decay or dead phase,
        # after the alpha period ends.  Allow generous tolerance because
        # the rolling window introduces lag.
        assert true_decay_start - 60 <= detected <= true_decay_start + 400, (
            f"Decay onset {detected} should be near true decay start "
            f"{true_decay_start} (tolerance accounts for rolling window lag)"
        )


# ===================================================================
# 6. giacomini_white_test
# ===================================================================

class TestGiacominiWhiteTest:

    def test_unconditional_equal_loss(self):
        rng = np.random.default_rng(42)
        n = 500
        # Two models with identical loss distributions
        loss = rng.normal(0, 0.01, size=n) ** 2
        instruments = np.ones(n)  # constant instrument (ignored for unconditional)

        result = giacomini_white_test(
            loss, loss, instruments, test_type="unconditional"
        )

        # Equal losses => should NOT reject
        assert result["p_value"] > 0.05 or np.isnan(result["statistic"]), (
            f"Equal losses should not reject null "
            f"(stat={result['statistic']}, p={result['p_value']:.4f})"
        )
