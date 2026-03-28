"""Forecast evaluation and signal quality metrics for detecting alpha decay.

Implements out-of-sample predictive performance measures, statistical tests
for model comparison, and rolling signal diagnostics used to identify
structural breaks in factor return predictability.

References:
    - Campbell & Thompson (2008) — out-of-sample R²
    - Clark & West (2007) — MSPE-adjusted nested model comparison
    - Giacomini & White (2006) — conditional predictive ability
    - Diebold & Mariano (1995) — predictive accuracy testing
    - Welch & Goyal (2008) — equity premium predictability
    - Grinold & Kahn (2000) — information coefficient
    - Sharpe (1994) — Sharpe ratio
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_array(x: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """Convert pandas Series or numpy array to a 1-D numpy array."""
    if isinstance(x, pd.Series):
        return x.values.astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _to_series(
    values: np.ndarray,
    index: Optional[pd.Index] = None,
    name: Optional[str] = None,
) -> pd.Series:
    """Wrap an array as a pandas Series, preserving an optional index."""
    return pd.Series(values, index=index, name=name)


# ---------------------------------------------------------------------------
# 1. Out-of-sample R²
# ---------------------------------------------------------------------------

def oos_r_squared(
    actual: Union[np.ndarray, pd.Series],
    forecast_model: Union[np.ndarray, pd.Series],
    forecast_benchmark: Optional[Union[np.ndarray, pd.Series]] = None,
) -> float:
    """Compute out-of-sample R² per Campbell & Thompson (2008).

    Measures whether a forecasting model improves upon a benchmark in
    terms of mean squared prediction error:

        R²_OOS = 1 - sum((actual - forecast_model)²)
                       / sum((actual - forecast_benchmark)²)

    A positive R²_OOS indicates that the model's forecasts have lower
    cumulative squared error than the benchmark.

    Parameters:
        actual: Realized return series, length T.
        forecast_model: Model forecasts aligned to ``actual``, length T.
        forecast_benchmark: Benchmark forecasts aligned to ``actual``,
            length T. If *None*, the expanding (cumulative) historical
            mean of ``actual`` is used — equivalent to assuming the
            benchmark is the prevailing-mean forecast at each point.

    Returns:
        float: R²_OOS value. Positive means the model outperforms the
        benchmark; negative means it underperforms.

    References:
        Campbell, J. Y. & Thompson, S. B. (2008). Predicting excess
        stock returns out of sample: Can anything beat the historical
        average? *Review of Financial Studies*, 21(4), 1509–1531.
    """
    actual_arr = _to_array(actual)
    model_arr = _to_array(forecast_model)

    if forecast_benchmark is not None:
        bench_arr = _to_array(forecast_benchmark)
    else:
        # Expanding historical mean: at each t, the mean of actual[0:t].
        # Shift by 1 so the benchmark at time t uses data up to t-1.
        cumsum = np.nancumsum(actual_arr)
        counts = np.arange(1, len(actual_arr) + 1, dtype=np.float64)
        expanding_mean = cumsum / counts
        # Shift: benchmark at t is the mean of actual[0..t-1].
        bench_arr = np.empty_like(actual_arr)
        bench_arr[0] = np.nan  # No history available at time 0.
        bench_arr[1:] = expanding_mean[:-1]

    sse_model = np.nansum((actual_arr - model_arr) ** 2)
    sse_bench = np.nansum((actual_arr - bench_arr) ** 2)

    if sse_bench == 0.0:
        logger.warning(
            "Benchmark SSE is zero; R²_OOS is undefined. Returning NaN."
        )
        return np.nan

    return float(1.0 - sse_model / sse_bench)


# ---------------------------------------------------------------------------
# 2. Rolling out-of-sample R²
# ---------------------------------------------------------------------------

def rolling_oos_r_squared(
    returns: Union[np.ndarray, pd.Series],
    predictor: Union[np.ndarray, pd.Series],
    window: int = 60,
    min_periods: int = 30,
) -> pd.Series:
    """Compute R²_OOS on a rolling basis.

    At each time *t*, uses data in ``[t - window + 1 : t + 1]`` to
    evaluate the model's out-of-sample predictive power relative to the
    prevailing historical mean benchmark within that window.

    Parameters:
        returns: Realized return series.
        predictor: Model forecast series aligned to ``returns``.
        window: Number of observations in each evaluation window.
            Default 60 corresponds to approximately 3 months of daily
            data — sufficient for stable OOS evaluation per Welch &
            Goyal (2008).
        min_periods: Minimum number of non-NaN observations required
            in the window to produce a value. Default 30.

    Returns:
        pandas.Series: Rolling R²_OOS values indexed to match ``returns``.

    References:
        Welch, I. & Goyal, A. (2008). A comprehensive look at the
        empirical performance of equity premium prediction. *Review of
        Financial Studies*, 21(4), 1455–1508.
    """
    if isinstance(returns, pd.Series):
        index = returns.index
    elif isinstance(predictor, pd.Series):
        index = predictor.index
    else:
        index = pd.RangeIndex(len(returns))

    ret_arr = _to_array(returns)
    pred_arr = _to_array(predictor)
    n = len(ret_arr)

    result = np.full(n, np.nan)

    for t in range(window - 1, n):
        start = t - window + 1
        r_win = ret_arr[start: t + 1]
        p_win = pred_arr[start: t + 1]

        valid = ~(np.isnan(r_win) | np.isnan(p_win))
        if valid.sum() < min_periods:
            continue

        r_valid = r_win[valid]
        p_valid = p_win[valid]

        # Benchmark: expanding mean within the window (shifted).
        cumsum_r = np.cumsum(r_valid)
        counts = np.arange(1, len(r_valid) + 1, dtype=np.float64)
        bench = np.empty_like(r_valid)
        bench[0] = np.nan
        bench[1:] = cumsum_r[:-1] / counts[:-1]

        # Skip the first observation in the window (no benchmark).
        r_eval = r_valid[1:]
        p_eval = p_valid[1:]
        b_eval = bench[1:]

        sse_model = np.nansum((r_eval - p_eval) ** 2)
        sse_bench = np.nansum((r_eval - b_eval) ** 2)

        if sse_bench == 0.0:
            continue

        result[t] = 1.0 - sse_model / sse_bench

    return _to_series(result, index=index, name="rolling_oos_r2")


# ---------------------------------------------------------------------------
# 3. Clark-West MSPE-adjusted test
# ---------------------------------------------------------------------------

def clark_west_test(
    actual: Union[np.ndarray, pd.Series],
    forecast_restricted: Union[np.ndarray, pd.Series],
    forecast_unrestricted: Union[np.ndarray, pd.Series],
) -> Dict[str, object]:
    """MSPE-adjusted test for nested model comparison.

    Tests whether an unrestricted (larger) forecasting model produces
    significantly lower mean squared prediction error than a restricted
    (nested) model, after adjusting for the noise introduced by
    estimating additional parameters.

    The adjustment accounts for the fact that under the null hypothesis
    of equal predictive ability the unrestricted model's MSPE is biased
    upward, making the standard Diebold-Mariano test undersized.

    Procedure (Clark & West, 2007):
        e1_t = actual_t - forecast_restricted_t   (restricted error)
        e2_t = actual_t - forecast_unrestricted_t  (unrestricted error)
        f_t  = e1_t² - [e2_t² - (forecast_restricted_t - forecast_unrestricted_t)²]

    The test statistic is the t-statistic from regressing f_t on a
    constant (equivalently, t-stat for mean(f_t) != 0).

    Parameters:
        actual: Realized return series, length T.
        forecast_restricted: Forecasts from the restricted (nested)
            model, length T.
        forecast_unrestricted: Forecasts from the unrestricted (larger)
            model, length T.

    Returns:
        dict with keys:
            - ``"statistic"``: float, the Clark-West t-statistic.
            - ``"p_value"``: float, one-sided p-value (H_a: unrestricted
              model has lower MSPE after adjustment).
            - ``"reject"``: bool, True if p_value < 0.05.

    References:
        Clark, T. E. & West, K. D. (2007). Approximately normal tests
        for equal predictive accuracy in nested models. *Journal of
        Econometrics*, 138(1), 291–311.
    """
    y = _to_array(actual)
    y1 = _to_array(forecast_restricted)
    y2 = _to_array(forecast_unrestricted)

    e1 = y - y1  # restricted errors
    e2 = y - y2  # unrestricted errors

    # MSPE-adjusted series
    f_t = e1 ** 2 - (e2 ** 2 - (y1 - y2) ** 2)

    # Drop NaN observations
    valid = ~np.isnan(f_t)
    f_valid = f_t[valid]
    n = len(f_valid)

    if n < 2:
        logger.warning("Fewer than 2 valid observations for Clark-West test.")
        return {"statistic": np.nan, "p_value": np.nan, "reject": False}

    # t-statistic for mean(f_t) = 0
    mean_f = np.mean(f_valid)
    se_f = np.std(f_valid, ddof=1) / np.sqrt(n)

    if se_f == 0.0:
        logger.warning("Zero standard error in Clark-West f_t series.")
        return {"statistic": np.nan, "p_value": np.nan, "reject": False}

    t_stat = mean_f / se_f

    # One-sided test: H_a is that the unrestricted model is better
    # (positive mean f_t), so p_value = P(T > t_stat) under t(n-1).
    p_value = float(1.0 - stats.t.cdf(t_stat, df=n - 1))

    significance_level = 0.05

    return {
        "statistic": float(t_stat),
        "p_value": p_value,
        "reject": bool(p_value < significance_level),
    }


# ---------------------------------------------------------------------------
# 4. Giacomini-White test
# ---------------------------------------------------------------------------

def giacomini_white_test(
    loss1: Union[np.ndarray, pd.Series],
    loss2: Union[np.ndarray, pd.Series],
    instruments: Union[np.ndarray, pd.DataFrame, pd.Series],
    test_type: str = "unconditional",
) -> Dict[str, object]:
    """Test for equal (conditional) predictive ability.

    Compares two models via their loss differentials. The unconditional
    variant reduces to a Diebold-Mariano style test; the conditional
    variant regresses loss differentials on instruments and performs a
    Wald test of joint significance.

    Parameters:
        loss1: Loss series from model 1 (e.g., squared forecast errors),
            length T.
        loss2: Loss series from model 2, length T.
        instruments: Conditioning variables used in the conditional test.
            For the unconditional test, these are ignored beyond
            determining the sample. Shape (T, q) or (T,) for a single
            instrument (e.g., lagged volatility, regime indicators).
        test_type: ``"unconditional"`` for a standard Diebold-Mariano
            style test on the mean loss differential; ``"conditional"``
            for a Wald test on the coefficients from regressing the loss
            differential on ``instruments``.

    Returns:
        dict with keys:
            - ``"statistic"``: float, the test statistic (t-stat for
              unconditional; Wald chi² for conditional).
            - ``"p_value"``: float, two-sided p-value.
            - ``"df"``: int, degrees of freedom (1 for unconditional;
              number of instruments for conditional).

    References:
        Giacomini, R. & White, H. (2006). Tests of conditional
        predictive ability. *Econometrica*, 74(6), 1545–1578.

        Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive
        accuracy. *Journal of Business & Economic Statistics*, 13(3),
        253–263.
    """
    l1 = _to_array(loss1)
    l2 = _to_array(loss2)
    d_t = l1 - l2  # loss differential

    # Convert instruments to 2-D array
    if isinstance(instruments, pd.DataFrame):
        Z = instruments.values.astype(np.float64)
    elif isinstance(instruments, pd.Series):
        Z = instruments.values.astype(np.float64).reshape(-1, 1)
    else:
        Z = np.asarray(instruments, dtype=np.float64)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

    # Align lengths and remove NaN rows
    min_len = min(len(d_t), len(Z))
    d_t = d_t[:min_len]
    Z = Z[:min_len]

    valid = ~np.isnan(d_t) & ~np.any(np.isnan(Z), axis=1)
    d_valid = d_t[valid]
    Z_valid = Z[valid]
    n = len(d_valid)

    if test_type == "unconditional":
        # Diebold-Mariano style: t-test on mean loss differential.
        if n < 2:
            return {"statistic": np.nan, "p_value": np.nan, "df": 1}

        mean_d = np.mean(d_valid)
        se_d = np.std(d_valid, ddof=1) / np.sqrt(n)

        if se_d == 0.0:
            return {"statistic": np.nan, "p_value": np.nan, "df": 1}

        t_stat = mean_d / se_d
        p_value = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n - 1)))

        return {
            "statistic": float(t_stat),
            "p_value": p_value,
            "df": 1,
        }

    elif test_type == "conditional":
        # Regress d_t on instruments (with intercept), Wald test on all
        # coefficients including the intercept.
        q = Z_valid.shape[1]
        # Design matrix with intercept
        X = np.column_stack([np.ones(n), Z_valid])
        k = X.shape[1]  # intercept + q instruments

        if n <= k:
            logger.warning(
                "Insufficient observations (%d) for %d regressors in "
                "Giacomini-White conditional test.",
                n,
                k,
            )
            return {"statistic": np.nan, "p_value": np.nan, "df": k}

        # OLS: beta = (X'X)^{-1} X' d
        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            logger.warning("Singular X'X matrix in Giacomini-White test.")
            return {"statistic": np.nan, "p_value": np.nan, "df": k}

        beta = XtX_inv @ (X.T @ d_valid)
        residuals = d_valid - X @ beta
        sigma2 = np.sum(residuals ** 2) / (n - k)

        # Variance-covariance of beta
        var_beta = sigma2 * XtX_inv

        # Wald statistic: beta' * Var(beta)^{-1} * beta ~ chi²(k)
        try:
            var_beta_inv = np.linalg.inv(var_beta)
        except np.linalg.LinAlgError:
            logger.warning(
                "Singular variance matrix in Giacomini-White Wald test."
            )
            return {"statistic": np.nan, "p_value": np.nan, "df": k}

        wald_stat = float(beta @ var_beta_inv @ beta)
        p_value = float(1.0 - stats.chi2.cdf(wald_stat, df=k))

        return {
            "statistic": wald_stat,
            "p_value": p_value,
            "df": k,
        }

    else:
        raise ValueError(
            f"test_type must be 'unconditional' or 'conditional', "
            f"got '{test_type}'."
        )


# ---------------------------------------------------------------------------
# 5. Signal half-life (Ornstein-Uhlenbeck)
# ---------------------------------------------------------------------------

def signal_half_life(
    returns: Union[np.ndarray, pd.Series],
) -> Dict[str, object]:
    """Estimate the half-life of mean reversion from an Ornstein-Uhlenbeck process.

    Models the return series as a discrete-time AR(1) process:

        r_t = phi * r_{t-1} + epsilon_t

    Under the continuous-time Ornstein-Uhlenbeck interpretation:
        - phi is the AR(1) slope coefficient.
        - The mean-reversion speed is theta = -ln(phi).
        - The half-life (time for a deviation to decay by 50%) is
          h = ln(2) / theta.

    If the OLS slope phi is >= 1 (unit root / explosive) or <= 0
    (negative autocorrelation inconsistent with O-U dynamics), the
    process does not exhibit the expected mean reversion and
    ``np.inf`` is returned for the half-life.

    Parameters:
        returns: Time series of returns or signal values, length T.

    Returns:
        dict with keys:
            - ``"half_life"``: float, estimated half-life in periods
              (np.inf if no mean reversion detected).
            - ``"theta"``: float, mean-reversion speed parameter
              (np.nan if no mean reversion).
            - ``"r_squared"``: float, R² of the AR(1) regression.
            - ``"slope"``: float, OLS slope (phi).
            - ``"se_slope"``: float, standard error of the OLS slope.

    Notes:
        The Ornstein-Uhlenbeck process is:
            dX_t = -theta * (X_t - mu) * dt + sigma * dW_t

        Discretizing with interval dt=1 yields:
            X_t = exp(-theta) * X_{t-1} + mu*(1 - exp(-theta)) + noise

        so phi = exp(-theta) and theta = -ln(phi).

    References:
        Uhlenbeck, G. E. & Ornstein, L. S. (1930). On the theory of
        the Brownian motion. *Physical Review*, 36(5), 823–841.
    """
    r = _to_array(returns)

    # Remove NaN for regression
    valid_mask = ~np.isnan(r[:-1]) & ~np.isnan(r[1:])
    y = r[1:][valid_mask]
    x = r[:-1][valid_mask]

    n = len(y)
    if n < 3:
        logger.warning(
            "Fewer than 3 valid observation pairs for half-life estimation."
        )
        return {
            "half_life": np.inf,
            "theta": np.nan,
            "r_squared": np.nan,
            "slope": np.nan,
            "se_slope": np.nan,
        }

    # OLS: y = slope * x + intercept
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xx = np.sum((x - x_mean) ** 2)

    if ss_xx == 0.0:
        return {
            "half_life": np.inf,
            "theta": np.nan,
            "r_squared": np.nan,
            "slope": np.nan,
            "se_slope": np.nan,
        }

    slope = np.sum((x - x_mean) * (y - y_mean)) / ss_xx
    intercept = y_mean - slope * x_mean

    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan

    # Standard error of slope
    sigma2 = ss_res / (n - 2)
    se_slope = np.sqrt(sigma2 / ss_xx)

    # Mean-reversion check
    if slope >= 1.0 or slope <= 0.0:
        logger.info(
            "OLS slope = %.4f: no O-U mean reversion detected "
            "(slope must be in (0, 1)).",
            slope,
        )
        return {
            "half_life": np.inf,
            "theta": np.nan,
            "r_squared": float(r_squared),
            "slope": float(slope),
            "se_slope": float(se_slope),
        }

    theta = -np.log(slope)
    half_life = np.log(2.0) / theta

    return {
        "half_life": float(half_life),
        "theta": float(theta),
        "r_squared": float(r_squared),
        "slope": float(slope),
        "se_slope": float(se_slope),
    }


# ---------------------------------------------------------------------------
# 6. Rolling half-life
# ---------------------------------------------------------------------------

def rolling_half_life(
    returns: Union[np.ndarray, pd.Series],
    window: int = 252,
) -> pd.Series:
    """Compute signal half-life on a rolling basis.

    Applies :func:`signal_half_life` over a rolling window of the
    return series. Infinite half-life values are capped at ``window``
    to avoid unbounded values in downstream plots and analyses.

    Parameters:
        returns: Time series of returns or signal values.
        window: Rolling window size in periods. Default 252
            corresponds to approximately one year of daily trading
            data.

    Returns:
        pandas.Series: Rolling half-life values. Infinite values are
        capped at ``window``.
    """
    r = _to_array(returns)
    index = returns.index if isinstance(returns, pd.Series) else pd.RangeIndex(len(r))
    n = len(r)

    result = np.full(n, np.nan)

    for t in range(window - 1, n):
        start = t - window + 1
        window_data = r[start: t + 1]
        hl_result = signal_half_life(window_data)
        hl = hl_result["half_life"]
        # Cap at window to avoid inf
        result[t] = min(hl, window)

    return _to_series(result, index=index, name="rolling_half_life")


# ---------------------------------------------------------------------------
# 7. Rolling Sharpe ratio
# ---------------------------------------------------------------------------

def rolling_sharpe(
    returns: Union[np.ndarray, pd.Series],
    window: int = 252,
    annualization: int = 252,
) -> pd.Series:
    """Compute rolling annualized Sharpe ratio.

    Sharpe = mean(r) / std(r) * sqrt(annualization)

    computed over a rolling window of ``window`` observations.

    Parameters:
        returns: Excess return series (already net of risk-free rate).
        window: Rolling window size in periods. Default 252
            corresponds to approximately one year of daily trading
            data — the standard horizon for evaluating annual
            risk-adjusted performance (Sharpe, 1994).
        annualization: Annualization factor. Default 252 for daily
            data. Uses sqrt-of-time scaling (assumes i.i.d. returns
            as a first-order approximation; see Lo, 2002).

    Returns:
        pandas.Series: Rolling annualized Sharpe ratio values.

    References:
        Sharpe, W. F. (1994). The Sharpe ratio. *Journal of Portfolio
        Management*, 21(1), 49–58.

        Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial
        Analysts Journal*, 58(4), 36–52.
    """
    if isinstance(returns, pd.Series):
        rolling_mean = returns.rolling(window=window, min_periods=window).mean()
        rolling_std = returns.rolling(
            window=window, min_periods=window
        ).std(ddof=1)
        sharpe = (rolling_mean / rolling_std) * np.sqrt(annualization)
        sharpe.name = "rolling_sharpe"
        return sharpe

    # numpy array path
    r = _to_array(returns)
    n = len(r)
    result = np.full(n, np.nan)

    for t in range(window - 1, n):
        start = t - window + 1
        win = r[start: t + 1]
        valid = win[~np.isnan(win)]
        if len(valid) < window:
            continue
        mu = np.mean(valid)
        sigma = np.std(valid, ddof=1)
        if sigma == 0.0:
            result[t] = np.nan
        else:
            result[t] = (mu / sigma) * np.sqrt(annualization)

    return _to_series(result, index=pd.RangeIndex(n), name="rolling_sharpe")


# ---------------------------------------------------------------------------
# 8. Rolling information coefficient
# ---------------------------------------------------------------------------

def rolling_information_coefficient(
    predictions: Union[np.ndarray, pd.Series],
    realized: Union[np.ndarray, pd.Series],
    window: int = 63,
) -> pd.Series:
    """Compute rolling Spearman rank correlation (information coefficient).

    The IC measures the rank correlation between predictions and
    realized returns — the canonical measure of factor predictive
    power (Grinold & Kahn, 2000).

    Parameters:
        predictions: Predicted return or signal series.
        realized: Realized return series aligned to ``predictions``.
        window: Rolling window size in periods. Default 63 corresponds
            to approximately one calendar quarter of daily trading
            data (~63 trading days), a standard evaluation horizon that
            balances noise reduction with responsiveness to regime
            shifts.

    Returns:
        pandas.Series: Rolling Spearman rank IC values.

    References:
        Grinold, R. C. & Kahn, R. N. (2000). *Active Portfolio
        Management* (2nd ed.). McGraw-Hill.
    """
    if isinstance(predictions, pd.Series):
        index = predictions.index
    elif isinstance(realized, pd.Series):
        index = realized.index
    else:
        index = pd.RangeIndex(len(predictions))

    pred_arr = _to_array(predictions)
    real_arr = _to_array(realized)
    n = len(pred_arr)

    result = np.full(n, np.nan)

    for t in range(window - 1, n):
        start = t - window + 1
        p_win = pred_arr[start: t + 1]
        r_win = real_arr[start: t + 1]

        valid = ~(np.isnan(p_win) | np.isnan(r_win))
        if valid.sum() < window:
            continue

        corr, _ = stats.spearmanr(p_win[valid], r_win[valid])
        result[t] = corr

    return _to_series(result, index=index, name="rolling_ic")


# ---------------------------------------------------------------------------
# 9. Cumulative Sum of Squared Forecast Errors (CSSFE)
# ---------------------------------------------------------------------------

def cumulative_sum_of_squared_forecast_errors(
    actual: Union[np.ndarray, pd.Series],
    forecast: Union[np.ndarray, pd.Series],
) -> pd.Series:
    """Compute the cumulative sum of squared forecast errors (CSSFE).

    CSSFE_t = sum_{s=1}^{t} [ e²_bench,s - e²_model,s ]

    where e_bench = actual - benchmark (expanding historical mean) and
    e_model = actual - forecast.

    Interpretation:
        - Rising CSSFE: the model is outperforming the benchmark
          (accumulating gains in predictive accuracy).
        - Falling CSSFE: the model is underperforming the benchmark
          (signal decay / loss of predictive power).

    Parameters:
        actual: Realized return series, length T.
        forecast: Model forecast series aligned to ``actual``, length T.

    Returns:
        pandas.Series: Cumulative CSSFE values.

    References:
        Welch, I. & Goyal, A. (2008). A comprehensive look at the
        empirical performance of equity premium prediction. *Review of
        Financial Studies*, 21(4), 1455–1508.
    """
    if isinstance(actual, pd.Series):
        index = actual.index
    elif isinstance(forecast, pd.Series):
        index = forecast.index
    else:
        index = pd.RangeIndex(len(actual))

    actual_arr = _to_array(actual)
    forecast_arr = _to_array(forecast)

    # Benchmark: expanding historical mean (shifted by 1)
    cumsum = np.nancumsum(actual_arr)
    counts = np.arange(1, len(actual_arr) + 1, dtype=np.float64)
    expanding_mean = cumsum / counts

    bench_forecast = np.empty_like(actual_arr)
    bench_forecast[0] = np.nan  # No benchmark at t=0
    bench_forecast[1:] = expanding_mean[:-1]

    e_bench_sq = (actual_arr - bench_forecast) ** 2
    e_model_sq = (actual_arr - forecast_arr) ** 2

    # Difference: positive when model beats benchmark
    diff = e_bench_sq - e_model_sq

    cssfe = np.nancumsum(diff)
    # Set first value to NaN since benchmark is undefined at t=0
    cssfe[0] = np.nan

    return _to_series(cssfe, index=index, name="cssfe")


# ---------------------------------------------------------------------------
# 10. Decay onset detection
# ---------------------------------------------------------------------------

def detect_decay_onset(
    metric_series: Union[np.ndarray, pd.Series],
    method: str = "cusum",
    threshold: Optional[float] = None,
) -> Dict[str, object]:
    """Detect when signal quality decay begins in a rolling metric series.

    Meta-function that takes a rolling signal quality metric (e.g.,
    rolling Sharpe ratio, rolling IC, rolling R²_OOS) and identifies
    the onset of a sustained decline.

    Parameters:
        metric_series: Rolling signal quality metric values. May
            contain NaN at the beginning due to burn-in.
        method: Detection method.
            - ``"cusum"``: Apply CUSUM (cumulative sum of deviations
              from the mean) to the metric series. Detects the first
              significant downward shift as the point where the CUSUM
              exceeds a critical threshold (set at 2 standard
              deviations of the metric series if ``threshold`` is
              None). Based on Page (1954).
            - ``"threshold"``: Flag when the metric drops below
              ``threshold`` for 5 or more consecutive periods.
        threshold: Sensitivity parameter.
            - For ``method="cusum"``: CUSUM critical value. If None,
              defaults to 2 * std(metric_series) — a standard choice
              per Page (1954) and Hinkley (1971).
            - For ``method="threshold"``: The minimum acceptable metric
              value. Required when ``method="threshold"``.

    Returns:
        dict with keys:
            - ``"decay_onset_index"``: int or None, positional index of
              the first detected decay point.
            - ``"decay_onset_date"``: Timestamp or None, date of the
              onset if the input has a DatetimeIndex.
            - ``"method"``: str, the detection method used.
            - ``"confidence"``: float or None, a measure of detection
              confidence (for CUSUM: the CUSUM value at onset divided
              by the critical threshold; for threshold: fraction of
              subsequent periods below threshold).

    References:
        Page, E. S. (1954). Continuous inspection schemes.
        *Biometrika*, 41(1/2), 100–115.

        Hinkley, D. V. (1971). Inference about the change-point from
        cumulative sum tests. *Biometrika*, 58(3), 509–523.
    """
    if isinstance(metric_series, pd.Series):
        index = metric_series.index
        values = metric_series.values.astype(np.float64)
    else:
        index = None
        values = np.asarray(metric_series, dtype=np.float64)

    # Strip leading NaN (burn-in)
    valid_mask = ~np.isnan(values)
    if not np.any(valid_mask):
        return {
            "decay_onset_index": None,
            "decay_onset_date": None,
            "method": method,
            "confidence": None,
        }

    first_valid = int(np.argmax(valid_mask))
    valid_values = values[first_valid:]
    # For indexing back into original array
    offset = first_valid

    no_decay = {
        "decay_onset_index": None,
        "decay_onset_date": None,
        "method": method,
        "confidence": None,
    }

    if method == "cusum":
        # CUSUM for downward shift detection.
        # S_t = sum_{s=1}^{t} (mu - x_s)  (positive when values drop)
        clean = valid_values[~np.isnan(valid_values)]
        if len(clean) < 2:
            return no_decay

        mu = np.nanmean(valid_values)
        sigma = np.nanstd(valid_values, ddof=1)

        if sigma == 0.0:
            return no_decay

        # Critical value: 2 * sigma per Page (1954) / Hinkley (1971)
        critical = threshold if threshold is not None else 2.0 * sigma

        # Cumulative sum of deviations (positive = values below mean)
        cusum = np.nancumsum(mu - valid_values)

        # Find first time CUSUM exceeds the critical value
        exceedance = np.where(cusum > critical)[0]

        if len(exceedance) == 0:
            return no_decay

        onset_local = int(exceedance[0])
        onset_global = onset_local + offset
        cusum_at_onset = cusum[onset_local]
        confidence = float(cusum_at_onset / critical) if critical > 0 else None

        onset_date = None
        if index is not None and isinstance(index, pd.DatetimeIndex):
            onset_date = index[onset_global]

        return {
            "decay_onset_index": onset_global,
            "decay_onset_date": onset_date,
            "method": method,
            "confidence": confidence,
        }

    elif method == "threshold":
        if threshold is None:
            raise ValueError(
                "threshold must be specified when method='threshold'."
            )

        # Consecutive periods below threshold: require 5 in a row
        consecutive_required = 5
        below = valid_values < threshold
        count = 0

        for i in range(len(valid_values)):
            if np.isnan(valid_values[i]):
                count = 0
                continue
            if below[i]:
                count += 1
                if count >= consecutive_required:
                    # Onset is the first of the consecutive run
                    onset_local = i - consecutive_required + 1
                    onset_global = onset_local + offset

                    # Confidence: fraction of remaining periods below threshold
                    remaining = valid_values[onset_local:]
                    remaining_valid = remaining[~np.isnan(remaining)]
                    if len(remaining_valid) > 0:
                        confidence = float(
                            np.sum(remaining_valid < threshold)
                            / len(remaining_valid)
                        )
                    else:
                        confidence = None

                    onset_date = None
                    if index is not None and isinstance(index, pd.DatetimeIndex):
                        onset_date = index[onset_global]

                    return {
                        "decay_onset_index": onset_global,
                        "decay_onset_date": onset_date,
                        "method": method,
                        "confidence": confidence,
                    }
            else:
                count = 0

        return no_decay

    else:
        raise ValueError(
            f"method must be 'cusum' or 'threshold', got '{method}'."
        )
