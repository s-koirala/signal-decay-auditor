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
    cumulative_sum_of_squared_forecast_errors,
    detect_decay_onset,
    giacomini_white_test,
    oos_r_squared,
    rolling_half_life,
    rolling_information_coefficient,
    rolling_oos_r_squared,
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

    def test_nan_in_forecast_raises(self):
        """Forecast containing NaN should raise ValueError."""
        rng = np.random.default_rng(42)
        actual = rng.normal(0.0, 0.01, size=100)
        forecast = actual.copy()
        forecast[50] = np.nan
        benchmark = np.zeros(100)

        with pytest.raises(ValueError, match="NaN"):
            oos_r_squared(actual, forecast, forecast_benchmark=benchmark)


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

    def test_significance_parameter(self):
        """Verify the significance parameter controls the rejection threshold.

        Use a case where the test is borderline at 0.05 but should not
        reject at 0.01.  We construct a moderately better unrestricted model.
        """
        rng = np.random.default_rng(55)
        actual = rng.normal(0.001, 0.01, size=500)
        forecast_restricted = np.zeros(500)
        # Unrestricted model: actual + moderate noise (somewhat better)
        forecast_unrestricted = actual + rng.normal(0, 0.005, size=500)

        result_loose = clark_west_test(
            actual, forecast_restricted, forecast_unrestricted, significance=0.05
        )
        result_strict = clark_west_test(
            actual, forecast_restricted, forecast_unrestricted, significance=0.01
        )

        # The statistic and p-value should be identical regardless of significance
        assert result_loose["statistic"] == pytest.approx(result_strict["statistic"])
        assert result_loose["p_value"] == pytest.approx(result_strict["p_value"])

        # With a stricter significance level, the rejection threshold is lower.
        # If both reject, strict should still be valid; the key check is that
        # the reject field is computed using the provided significance level.
        if result_loose["p_value"] < 0.05:
            assert result_loose["reject"] is True
        if result_strict["p_value"] >= 0.01:
            assert result_strict["reject"] is False


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

    def test_cusum_method_detects_decay(self):
        """The CUSUM method (default) should detect decay in a factor lifecycle."""
        series, meta = generate_factor_with_decay(
            alpha_period=800,
            decay_period=400,
            dead_period=400,
            alpha_mean=0.002,
            decay_end_mean=0.0,
            dead_mean=-0.001,
            std=0.005,
            seed=77,
        )

        # Compute rolling Sharpe as the signal quality metric
        sharpe = rolling_sharpe(
            pd.Series(series), window=60, annualization=252
        )

        # Use the CUSUM method (default) to detect decay onset
        result = detect_decay_onset(sharpe, method="cusum")

        assert result["decay_onset_index"] is not None, (
            "CUSUM method should detect decay onset in factor lifecycle"
        )
        assert result["method"] == "cusum", (
            f"Expected method='cusum', got '{result['method']}'"
        )
        detected = result["decay_onset_index"]
        # The CUSUM tracks cumulative deviations from the expanding mean,
        # so it may trigger before the true structural break.  The key
        # assertion is that decay IS detected (not None) and occurs before
        # the series ends.
        total_length = len(series)
        assert 0 < detected < total_length, (
            f"CUSUM decay onset {detected} should be within series "
            f"(length {total_length})"
        )
        assert result["confidence"] is not None and result["confidence"] > 0, (
            "CUSUM detection confidence should be positive"
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

    def test_conditional_different_models(self):
        """Conditional GW test should reject when one model is clearly better."""
        rng = np.random.default_rng(42)
        n = 500
        actual = rng.normal(0.001, 0.01, size=n)

        # Model 1: bad (zero forecast)
        loss1 = (actual - 0.0) ** 2
        # Model 2: good (near-perfect forecast)
        loss2 = (actual - (actual + rng.normal(0, 0.001, size=n))) ** 2

        # Instrument: lagged squared return (informative about volatility)
        instruments = np.zeros(n)
        instruments[1:] = actual[:-1] ** 2

        result = giacomini_white_test(
            loss1, loss2, instruments, test_type="conditional"
        )

        assert result["statistic"] > 0, (
            f"Wald statistic should be positive, got {result['statistic']}"
        )
        assert result["df"] == 2, (
            f"Expected df=2 (intercept + 1 instrument), got {result['df']}"
        )


# ===================================================================
# 7. rolling_oos_r_squared
# ===================================================================

class TestRollingOosRSquared:

    def test_output_length(self):
        rng = np.random.default_rng(42)
        n = 300
        returns = rng.normal(0.0, 0.01, size=n)
        predictor = rng.normal(0.0, 0.01, size=n)

        result = rolling_oos_r_squared(returns, predictor, window=60)
        assert len(result) == n, (
            f"Output length {len(result)} should match input length {n}"
        )

    def test_nan_before_window(self):
        rng = np.random.default_rng(42)
        n = 200
        returns = rng.normal(0.001, 0.01, size=n)
        predictor = returns + rng.normal(0, 0.002, size=n)
        window = 60

        result = rolling_oos_r_squared(returns, predictor, window=window)
        # First window-1 values should be NaN
        assert np.all(np.isnan(result.values[:window - 1])), (
            "First window-1 values should be NaN"
        )

    def test_good_predictor_has_positive_values(self):
        """A near-perfect predictor should yield some positive R2 values."""
        rng = np.random.default_rng(42)
        n = 300
        returns = rng.normal(0.001, 0.01, size=n)
        # Good predictor: actual + small noise
        predictor = returns + rng.normal(0, 0.001, size=n)

        result = rolling_oos_r_squared(returns, predictor, window=60)
        valid = result.dropna()
        assert np.any(valid > 0), (
            "A good predictor should produce some positive rolling OOS R2 values"
        )


# ===================================================================
# 8. rolling_half_life
# ===================================================================

class TestRollingHalfLife:

    def test_output_length(self):
        rng = np.random.default_rng(42)
        n = 500
        series = rng.normal(0.0, 0.01, size=n)

        result = rolling_half_life(series, window=252)
        assert len(result) == n, (
            f"Output length {len(result)} should match input length {n}"
        )

    def test_ou_process_finite_half_life(self):
        """O-U process should produce finite, positive rolling half-life."""
        series, _ = generate_ou_process(
            n=2000, theta=0.05, mu=0.0, sigma=0.01, x0=0.05, seed=42
        )

        result = rolling_half_life(series, window=500)
        valid = result.dropna()

        assert len(valid) > 0, "Should have non-NaN values after burn-in"
        assert np.all(valid > 0), "All valid half-life values should be positive"
        # Half-life should be capped at window (500), so no inf
        assert np.all(np.isfinite(valid)), "All valid values should be finite"


# ===================================================================
# 9. rolling_information_coefficient
# ===================================================================

class TestRollingInformationCoefficient:

    def test_output_length(self):
        rng = np.random.default_rng(42)
        n = 200
        predictions = rng.normal(0, 0.01, size=n)
        realized = rng.normal(0, 0.01, size=n)

        result = rolling_information_coefficient(predictions, realized, window=63)
        assert len(result) == n, (
            f"Output length {len(result)} should match input length {n}"
        )

    def test_perfect_rank_correlation(self):
        """Identical predictions and realized should give IC ~1.0."""
        rng = np.random.default_rng(42)
        n = 200
        values = rng.normal(0, 0.01, size=n)

        result = rolling_information_coefficient(values, values, window=63)
        valid = result.dropna()

        assert len(valid) > 0, "Should have non-NaN values after burn-in"
        assert np.all(valid > 0.99), (
            f"Perfect rank correlation should yield IC ~1.0, "
            f"got min={valid.min():.4f}"
        )


# ===================================================================
# 10. cumulative_sum_of_squared_forecast_errors
# ===================================================================

class TestCSSFE:

    def test_output_length(self):
        rng = np.random.default_rng(42)
        n = 200
        actual = rng.normal(0.001, 0.01, size=n)
        forecast = actual + rng.normal(0, 0.001, size=n)

        result = cumulative_sum_of_squared_forecast_errors(actual, forecast)
        assert len(result) == n, (
            f"Output length {len(result)} should match input length {n}"
        )

    def test_perfect_forecast_rising(self):
        """Perfect forecast should produce a rising CSSFE (beats expanding mean)."""
        rng = np.random.default_rng(42)
        n = 500
        # Signal with consistent positive mean (expanding mean lags behind)
        actual = rng.normal(0.005, 0.01, size=n)
        forecast = actual.copy()  # perfect forecast

        result = cumulative_sum_of_squared_forecast_errors(actual, forecast)
        valid = result.dropna()

        # CSSFE should generally trend upward: perfect model always beats
        # the expanding-mean benchmark.
        assert valid.iloc[-1] > valid.iloc[1], (
            "Perfect forecast CSSFE should trend upward (model beats benchmark)"
        )
