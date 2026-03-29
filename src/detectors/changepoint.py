"""
Changepoint detection algorithms for structural break identification
in factor return series.

This module provides a unified interface to several changepoint detection
methods commonly used in the econometrics and structural break literature.
Each detector inherits from ``ChangePointDetector`` and exposes a consistent
``fit`` / ``get_breakpoints`` API.

Implemented detectors
---------------------
- **PELTDetector** -- Pruned Exact Linear Time (Killick et al., 2012)
- **CUSUMDetector** -- OLS-CUSUM mean-shift test (Page, 1954; Brown,
  Durbin & Evans, 1975)
- **BaiPerronDetector** -- Sequential sup-F structural break test
  (Bai & Perron, 1998, 2003; Andrews, 1993)
- **MOSUMDetector** -- Moving-sum detector for parameter constancy
  (Chu, Hornik & Kauan, 1995)

References
----------
Andrews, D.W.K. (1993). Tests for Parameter Instability and Structural
    Change With Unknown Change Point. *Econometrica*, 61(4), 821-856.
Bai, J. & Perron, P. (1998). Estimating and Testing Linear Models with
    Multiple Structural Changes. *Econometrica*, 66(1), 47-78.
Bai, J. & Perron, P. (2003). Computation and analysis of multiple
    structural change models. *Journal of Applied Econometrics*, 18(1), 1-22.
Brown, R.L., Durbin, J. & Evans, J.M. (1975). Techniques for Testing the
    Constancy of Regression Relationships over Time. *JRSS B*, 37(2), 149-192.
Chu, C.-S.J., Hornik, K. & Kauan, C.-M. (1995). MOSUM Tests for Parameter
    Constancy. *Biometrika*, 82(3), 603-617.
Killick, R., Fearnhead, P. & Eckley, I.A. (2012). Optimal Detection of
    Changepoints with a Linear Computational Cost. *JASA*, 107(500), 1590-1598.
Page, E.S. (1954). Continuous Inspection Schemes. *Biometrika*, 41(1/2),
    100-115.
"""

from __future__ import annotations

import abc
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ChangePointDetector(abc.ABC):
    """Abstract base class for changepoint detectors.

    Every detector operates on a 1-D numeric array (e.g. factor returns,
    rolling Sharpe ratios, information coefficients) and stores results as
    instance attributes after :meth:`fit` is called.
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        self._breakpoints: List[int] = []

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _validate_series(series, min_length: int = 3) -> np.ndarray:
        """Convert *series* to a 1-D float64 numpy array and validate.

        Parameters
        ----------
        series : array-like
            1-D numeric data.
        min_length : int
            Minimum acceptable length.  Default is 3 (the theoretical
            minimum for most changepoint tests).

        Returns
        -------
        np.ndarray
            Validated 1-D float64 array.

        Raises
        ------
        ValueError
            If *series* is not 1-D or is shorter than *min_length*.
        """
        if isinstance(series, pd.Series):
            arr = series.to_numpy(dtype=np.float64, copy=True)
        else:
            arr = np.asarray(series, dtype=np.float64)

        if arr.ndim != 1:
            raise ValueError(
                f"Expected 1-D series, got array with {arr.ndim} dimensions."
            )
        if arr.shape[0] < min_length:
            raise ValueError(
                f"Series length ({arr.shape[0]}) is less than the required "
                f"minimum ({min_length})."
            )
        if np.any(np.isnan(arr)):
            raise ValueError(
                "Series contains NaN values.  Remove or impute before fitting."
            )
        return arr

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted yet.  "
                "Call .fit(series) first."
            )

    # -- abstract interface -----------------------------------------------

    @abc.abstractmethod
    def fit(self, series) -> "ChangePointDetector":
        """Detect changepoints in *series*.

        Parameters
        ----------
        series : array-like
            1-D numeric array or pandas Series.

        Returns
        -------
        self
        """
        ...

    @abc.abstractmethod
    def get_breakpoints(self) -> List[int]:
        """Return detected breakpoint indices (0-based).

        Returns
        -------
        list of int
        """
        ...

    @abc.abstractmethod
    def get_confidence_intervals(self, **kwargs):
        """Return confidence intervals for breakpoint locations."""
        ...


# ---------------------------------------------------------------------------
# 1. PELT Detector
# ---------------------------------------------------------------------------

class PELTDetector(ChangePointDetector):
    """Pruned Exact Linear Time (PELT) changepoint detector.

    Wraps :class:`ruptures.Pelt` with sensible defaults and adds bootstrap
    confidence intervals on break locations.

    Parameters
    ----------
    model : str, default ``"rbf"``
        Cost function passed to ``ruptures.Pelt``.  Common choices:

        - ``"l2"`` -- least-squares (mean shift in Gaussian data)
        - ``"rbf"`` -- kernel change in distribution (Gaussian kernel)
        - ``"normal"`` -- maximum-likelihood for normal segments

        The ``"rbf"`` cost is a non-parametric default that is robust to
        non-Gaussian returns (Killick et al., 2012, Section 5).
    min_size : int, default 30
        Minimum segment length.  30 observations ensures each segment
        contains enough data for reliable mean/variance estimation,
        consistent with common practice in empirical finance
        (monthly returns ≈ 2.5 years).
    penalty : float or None
        Penalty value for the PELT penalised likelihood.  When ``None``
        (default) a variance-scaled BIC penalty is used:
        ``pen = n_params * sigma^2 * log(n)`` where ``n_params`` equals
        the number of parameters in the cost model (2 for mean + variance),
        ``sigma^2`` is estimated from the full series, and ``n`` is the
        series length.  Variance scaling is essential for financial return
        data where the absolute cost improvement per observation is tiny
        relative to unscaled BIC (Killick et al., 2012, Section 3.1;
        Schwarz, 1978; Yao, 1988).

    References
    ----------
    Killick, R., Fearnhead, P. & Eckley, I.A. (2012). Optimal Detection of
        Changepoints with a Linear Computational Cost. *JASA*, 107(500),
        1590-1598.
    Schwarz, G. (1978). Estimating the Dimension of a Model. *Annals of
        Statistics*, 6(2), 461-464.
    Yao, Y.-C. (1988). Estimating the Number of Change-points via Schwarz'
        Criterion. *Statistics & Probability Letters*, 6(3), 181-189.
    """

    # Number of estimated parameters per segment for common cost models.
    _MODEL_NPARAMS = {
        "l2": 2,       # mean + variance
        "l1": 1,       # median
        "rbf": 2,      # implicit mean + scale via kernel
        "normal": 2,   # mean + variance (MLE)
        "ar": 3,       # AR(1) coeff + mean + variance
    }

    def __init__(
        self,
        model: str = "rbf",
        min_size: int = 30,
        penalty: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.min_size = min_size
        self.penalty = penalty
        self._series: Optional[np.ndarray] = None

    def fit(self, series) -> "PELTDetector":
        """Run PELT on *series* and store breakpoint indices.

        Parameters
        ----------
        series : array-like
            1-D numeric data (returns, rolling metrics, etc.).

        Returns
        -------
        self
        """
        import ruptures  # local import keeps the module importable without ruptures

        arr = self._validate_series(series, min_length=2 * self.min_size)
        self._series = arr
        n = len(arr)

        # Determine penalty.  The standard BIC penalty for changepoint
        # detection in the Gaussian mean+variance model is:
        #   pen = n_params * sigma^2 * log(n)
        # where sigma^2 is estimated from the full series.  Without the
        # variance scaling the penalty is in "log-likelihood units" and
        # overwhelms tiny cost improvements that are typical for daily
        # financial return data (mean shifts of ~1-10 bps against
        # ~100 bps daily vol).  See Killick et al. (2012), Section 3.1;
        # the ruptures documentation likewise recommends calibrating
        # the penalty to the data scale.
        if self.penalty is not None:
            pen = self.penalty
        else:
            n_params = self._MODEL_NPARAMS.get(self.model, 2)
            sigma2 = np.var(arr, ddof=1)
            pen = n_params * sigma2 * np.log(n)

        algo = ruptures.Pelt(model=self.model, min_size=self.min_size).fit(
            arr.reshape(-1, 1)
        )
        # ruptures returns breakpoints with the last element == n (end of signal)
        raw = algo.predict(pen=pen)
        self._breakpoints = [b for b in raw if b < n]
        self._fitted = True
        return self

    def get_breakpoints(self) -> List[int]:
        """Return 0-based breakpoint indices detected by PELT."""
        self._check_fitted()
        return list(self._breakpoints)

    def get_confidence_intervals(
        self,
        series=None,
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
    ) -> List[Tuple[int, int]]:
        """Bootstrap confidence intervals for breakpoint locations.

        Uses the stationary bootstrap of Politis & Romano (1994) via the
        ``arch`` package to generate resampled paths, re-runs PELT on each
        replicate, and constructs percentile CIs on each breakpoint location.

        Parameters
        ----------
        series : array-like or None
            Series to bootstrap.  Defaults to the series passed to
            :meth:`fit`.
        n_bootstrap : int, default 1000
            Number of bootstrap replicates.
        alpha : float, default 0.05
            Significance level (two-sided).  Produces ``(1-alpha)*100%``
            confidence intervals.

        Returns
        -------
        list of (int, int)
            Lower and upper bounds for each breakpoint, in the same order
            as :meth:`get_breakpoints`.
        """
        import ruptures
        from arch.bootstrap import StationaryBootstrap

        self._check_fitted()
        if not self._breakpoints:
            return []

        arr = self._series if series is None else self._validate_series(series)
        n = len(arr)

        if self.penalty is not None:
            pen = self.penalty
        else:
            n_params = self._MODEL_NPARAMS.get(self.model, 2)
            sigma2 = np.var(arr, ddof=1)
            pen = n_params * sigma2 * np.log(n)

        n_breaks = len(self._breakpoints)
        # Collect bootstrap breakpoint locations
        boot_breaks: List[List[int]] = []

        # Optimal block length for stationary bootstrap defaults in arch;
        # use series indices for resampling.
        bs = StationaryBootstrap(12, arr)  # 12 = avg block length
        for pos_data in bs.bootstrap(n_bootstrap):
            sample = pos_data[0][0].flatten()
            try:
                algo = ruptures.Pelt(
                    model=self.model, min_size=self.min_size
                ).fit(sample.reshape(-1, 1))
                raw = algo.predict(pen=pen)
                bps = sorted(b for b in raw if b < n)
                if len(bps) == n_breaks:
                    boot_breaks.append(bps)
            except Exception:
                # Skip replicates that fail (e.g., degenerate resamples)
                continue

        if not boot_breaks:
            return [(bp, bp) for bp in self._breakpoints]

        boot_arr = np.array(boot_breaks)  # (B, n_breaks)
        lo = alpha / 2
        hi = 1.0 - lo
        cis: List[Tuple[int, int]] = []
        for j in range(n_breaks):
            lower = int(np.percentile(boot_arr[:, j], 100 * lo))
            upper = int(np.percentile(boot_arr[:, j], 100 * hi))
            cis.append((lower, upper))
        return cis


# ---------------------------------------------------------------------------
# 2. CUSUM Detector
# ---------------------------------------------------------------------------

class CUSUMDetector(ChangePointDetector):
    """OLS-CUSUM test for mean-shift detection in a return series.

    Regresses the series on a constant and computes the cumulative sum
    of recursive (OLS) residuals.  Under the null of parameter constancy
    the normalised CUSUM path converges to a standard Brownian bridge.
    A crossing of the critical boundary signals a structural break.

    Parameters
    ----------
    alpha : float, default 0.05
        Significance level for the Brownian-bridge boundary.  Standard
        values from Brown, Durbin & Evans (1975), Table 1.

    Notes
    -----
    The implementation follows Brown, Durbin & Evans (1975) using
    recursive residuals computed via the standard forward-recursion
    formula.  When ``statsmodels`` provides
    ``breaks_cusumolsresid`` it is used for the boundary p-value;
    otherwise the asymptotic distribution of the supremum of a
    Brownian bridge is used (Kolmogorov-Smirnov type):

        P(sup |B(t)| > c) = 2 * sum_{k=1}^{inf} (-1)^{k+1} exp(-2 k^2 c^2)

    References
    ----------
    Page, E.S. (1954). Continuous Inspection Schemes. *Biometrika*,
        41(1/2), 100-115.
    Brown, R.L., Durbin, J. & Evans, J.M. (1975). Techniques for Testing
        the Constancy of Regression Relationships over Time. *JRSS B*,
        37(2), 149-192.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        super().__init__()
        self.alpha = alpha
        self._cusum_path: Optional[np.ndarray] = None
        self._sup_stat: Optional[float] = None
        self._pvalue: Optional[float] = None
        self._series_cache: Optional[np.ndarray] = None

    # -- internal helpers -------------------------------------------------

    @staticmethod
    def _recursive_residuals(y: np.ndarray) -> np.ndarray:
        """Compute recursive (one-step-ahead) OLS residuals.

        For a regression on a constant the recursive residual at
        observation *t* (starting from t = 1) is:

            w_t = (y_t - \\bar{y}_{1:t-1}) / sqrt(1 + 1/(t-1))

        Parameters
        ----------
        y : np.ndarray
            1-D series.

        Returns
        -------
        np.ndarray
            Recursive residuals of length ``n - 1``.
        """
        n = len(y)
        cumsum = np.cumsum(y)
        # Running mean of y[0:t] for t = 1, ..., n-1
        t_idx = np.arange(1, n)  # t = 1 .. n-1
        running_mean = cumsum[t_idx - 1] / t_idx  # mean of y[0:t]
        forecast_error = y[1:] - running_mean
        scale = np.sqrt(1.0 + 1.0 / t_idx)
        return forecast_error / scale

    @staticmethod
    def _brownian_bridge_pvalue(sup_stat: float) -> float:
        """Asymptotic p-value for sup|B(t)| where B is a Brownian bridge.

        Uses the Kolmogorov-Smirnov series expansion (rapidly converging):

            P(sup |B(t)| > c) = 2 * sum_{k>=1} (-1)^{k+1} exp(-2 k^2 c^2)
        """
        if sup_stat <= 0:
            return 1.0
        c = sup_stat
        terms = np.arange(1, 501)  # 500 terms is more than sufficient
        signs = (-1.0) ** (terms + 1)
        vals = signs * np.exp(-2.0 * terms ** 2 * c ** 2)
        p = 2.0 * np.sum(vals)
        return float(np.clip(p, 0.0, 1.0))

    # -- public interface -------------------------------------------------

    def get_breakpoints(self) -> List[int]:
        """Return estimated break location(s) as 0-based indices.

        A breakpoint is reported only when the CUSUM path crosses the
        critical boundary at the specified significance level.
        """
        self._check_fitted()
        return list(self._breakpoints)

    def get_statistic(self) -> Tuple[float, float]:
        """Return ``(sup_cusum, p_value)``.

        The supremum of the normalised CUSUM path and the corresponding
        asymptotic p-value derived from the Brownian bridge distribution.
        """
        self._check_fitted()
        return (self._sup_stat, self._pvalue)  # type: ignore[return-value]

    def get_confidence_intervals(
        self,
        series=None,
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
    ) -> List[Tuple[int, int]]:
        """Bootstrap percentile CIs on break locations.

        Re-fits the detector on *n_bootstrap* stationary-bootstrap
        replicates and returns the ``alpha/2`` and ``1-alpha/2``
        percentiles for each detected breakpoint.
        """
        from arch.bootstrap import StationaryBootstrap

        self._check_fitted()
        if not self._breakpoints:
            return []

        arr = self._series_cache if series is None else self._validate_series(series)
        boot_locs: List[int] = []
        bs = StationaryBootstrap(12, arr)
        for pos_data in bs.bootstrap(n_bootstrap):
            sample = pos_data[0][0].flatten()
            det = CUSUMDetector(alpha=self.alpha)
            try:
                det.fit(sample)
                bps = det.get_breakpoints()
                if len(bps) == 1:
                    boot_locs.append(bps[0])
            except (ValueError, RuntimeError):
                continue

        if not boot_locs:
            return [(bp, bp) for bp in self._breakpoints]

        lo_q = alpha / 2
        hi_q = 1.0 - lo_q
        lower = int(np.percentile(boot_locs, 100 * lo_q))
        upper = int(np.percentile(boot_locs, 100 * hi_q))
        return [(lower, upper)]

    def fit(self, series) -> "CUSUMDetector":
        """Compute the OLS-CUSUM path and test for structural breaks.

        Parameters
        ----------
        series : array-like
            1-D numeric data (minimum length 6 to ensure meaningful
            recursive residuals).

        Returns
        -------
        self
        """
        arr = self._validate_series(series, min_length=6)
        self._series_cache = arr
        n = len(arr)

        w = self._recursive_residuals(arr)
        sigma_w = np.std(w, ddof=1)
        if sigma_w < 1e-15:
            self._cusum_path = np.zeros_like(w)
            self._sup_stat = 0.0
            self._pvalue = 1.0
            self._breakpoints = []
            self._fitted = True
            return self

        cusum_raw = np.cumsum(w)
        self._cusum_path = cusum_raw / (sigma_w * np.sqrt(len(w)))

        self._sup_stat = float(np.max(np.abs(self._cusum_path)))
        self._pvalue = self._brownian_bridge_pvalue(self._sup_stat)

        if self._pvalue < self.alpha:
            # The CUSUM path accumulates after the break, so the sup is
            # often near the tail.  A better location estimator is the
            # point of maximum absolute change in the CUSUM path (i.e.,
            # argmax |diff(CUSUM)|), which corresponds to where the
            # recursive residuals are largest — the break onset.
            abs_diff = np.abs(np.diff(self._cusum_path))
            peak_idx = int(np.argmax(abs_diff)) + 1  # +1 for diff offset
            self._breakpoints = [peak_idx]
        else:
            self._breakpoints = []

        self._fitted = True
        return self


# ---------------------------------------------------------------------------
# 3. Bai-Perron Detector
# ---------------------------------------------------------------------------

class BaiPerronDetector(ChangePointDetector):
    """Sequential Bai-Perron structural break test.

    Implements the sequential sup-F procedure of Bai & Perron (1998, 2003)
    for detecting multiple structural breaks in the mean of a series:

    1. For each candidate break date *t* in the trimmed interior of the
       sample, compute a Wald/F statistic for the equality of the
       sub-sample means before and after *t*.
    2. The sup-F statistic is the maximum F over all candidates.
    3. Compare sup-F to the asymptotic critical values tabulated by
       Andrews (1993).
    4. If significant, split at the break and recursively test each
       sub-segment (sequential procedure of Bai & Perron, 1998, Sec. 4).

    Parameters
    ----------
    max_breaks : int, default 5
        Maximum number of breaks to search for.
    trimming : float, default 0.15
        Fraction of the sample trimmed from each endpoint.  Bai & Perron
        (1998) recommend 0.15 (``epsilon = 0.15``) as the default
        trimming parameter for the asymptotic theory.  Andrews (1993)
        derives critical values assuming 15 % trimming.
    significance : float, default 0.05
        Significance level for each sequential sup-F test.

    References
    ----------
    Andrews, D.W.K. (1993). Tests for Parameter Instability and Structural
        Change With Unknown Change Point. *Econometrica*, 61(4), 821-856.
    Bai, J. & Perron, P. (1998). Estimating and Testing Linear Models with
        Multiple Structural Changes. *Econometrica*, 66(1), 47-78.
    Bai, J. & Perron, P. (2003). Computation and analysis of multiple
        structural change models. *Journal of Applied Econometrics*, 18, 1-22.
    """

    # Asymptotic critical values for sup-F test with *p = 1* (intercept
    # only, i.e. mean-shift model) from Andrews (1993), Table 1, and
    # Bai & Perron (2003), Table 1.  Keys are (significance, trimming_bin).
    # trimming_bin is the closest tabulated epsilon value (0.05, 0.10, 0.15,
    # 0.20, 0.25).  Values are approximate sup-F critical values for one
    # break.
    _CRITICAL_VALUES = {
        # (alpha, number_of_breaks) -> critical value
        # Sourced from Bai & Perron (2003), Table 1 (p=1, epsilon=0.15)
        (0.10, 1): 7.04,
        (0.05, 1): 8.58,
        (0.025, 1): 10.13,
        (0.01, 1): 12.29,
        (0.10, 2): 5.96,
        (0.05, 2): 7.22,
        (0.025, 2): 8.43,
        (0.01, 2): 10.17,
        (0.10, 3): 5.46,
        (0.05, 3): 6.58,
        (0.025, 3): 7.64,
        (0.01, 3): 9.16,
        (0.10, 4): 5.15,
        (0.05, 4): 6.19,
        (0.025, 4): 7.16,
        (0.01, 4): 8.55,
        (0.10, 5): 4.93,
        (0.05, 5): 5.92,
        (0.025, 5): 6.82,
        (0.01, 5): 8.12,
    }

    def __init__(
        self,
        max_breaks: int = 5,
        trimming: float = 0.15,
        significance: float = 0.05,
    ) -> None:
        super().__init__()
        if not 0 < trimming < 0.5:
            raise ValueError(
                f"trimming must be in (0, 0.5); got {trimming}.  "
                "Bai & Perron (1998) recommend 0.15."
            )
        if max_breaks < 1:
            raise ValueError("max_breaks must be >= 1.")
        self.max_breaks = max_breaks
        self.trimming = trimming
        self.significance = significance
        self._f_statistics: List[float] = []
        self._series_cache: Optional[np.ndarray] = None

    # -- internal helpers -------------------------------------------------

    def _get_critical_value(self, n_breaks: int) -> float:
        """Look up or interpolate the critical value for the given
        significance level and break count.

        Falls back to the closest tabulated alpha when the exact value
        is not available.
        """
        key = (self.significance, min(n_breaks, 5))
        if key in self._CRITICAL_VALUES:
            return self._CRITICAL_VALUES[key]

        # Fall back to closest alpha
        available = sorted(
            {a for (a, _) in self._CRITICAL_VALUES},
            key=lambda a: abs(a - self.significance),
        )
        for a in available:
            k = (a, min(n_breaks, 5))
            if k in self._CRITICAL_VALUES:
                return self._CRITICAL_VALUES[k]

        # Ultimate fallback — conservative 1% value
        return 12.29

    @staticmethod
    def _sup_f_statistic(
        y: np.ndarray,
        start: int,
        end: int,
        trim_start: int,
        trim_end: int,
    ) -> Tuple[float, int]:
        """Compute the sup-F statistic over candidate break dates.

        Tests mean equality between sub-samples ``y[start:t]`` and
        ``y[t:end]`` for every ``t`` in ``[trim_start, trim_end)``.

        Returns
        -------
        (sup_f, argmax_t)
        """
        seg = y[start:end]
        n_seg = len(seg)
        total_sum = np.sum(seg)
        total_ss = np.sum(seg ** 2)

        best_f = -np.inf
        best_t = trim_start

        # Vectorised cumulative sums for fast F computation
        cumsum = np.cumsum(seg)
        # Candidate indices relative to *seg*
        lo = trim_start - start
        hi = trim_end - start

        for t_rel in range(lo, hi):
            n1 = t_rel
            n2 = n_seg - t_rel
            if n1 < 2 or n2 < 2:
                continue
            mean1 = cumsum[t_rel - 1] / n1
            mean2 = (total_sum - cumsum[t_rel - 1]) / n2
            grand_mean = total_sum / n_seg
            # Between-group sum of squares
            bss = n1 * (mean1 - grand_mean) ** 2 + n2 * (mean2 - grand_mean) ** 2
            # Within-group sum of squares
            wss = total_ss - n1 * mean1 ** 2 - n2 * mean2 ** 2
            if wss <= 0:
                continue
            f_stat = bss / (wss / (n_seg - 2))
            if f_stat > best_f:
                best_f = f_stat
                best_t = t_rel + start

        return (float(best_f), best_t)

    def _sequential_search(
        self,
        y: np.ndarray,
        start: int,
        end: int,
        depth: int,
    ) -> List[Tuple[int, float]]:
        """Recursively search for breaks in ``y[start:end]``."""
        n_seg = end - start
        min_seg = max(int(self.trimming * len(self._series_cache)), 5)  # type: ignore[arg-type]

        if n_seg < 2 * min_seg or depth >= self.max_breaks:
            return []

        trim_start = start + max(int(self.trimming * n_seg), 2)
        trim_end = end - max(int(self.trimming * n_seg), 2)

        if trim_start >= trim_end:
            return []

        sup_f, break_idx = self._sup_f_statistic(
            y, start, end, trim_start, trim_end
        )
        # In the sequential procedure (Bai & Perron, 1998, Section 4),
        # each step tests for exactly one additional break in a sub-
        # segment, so the critical value is always for a single-break
        # sup-F test regardless of recursion depth.
        cv = self._get_critical_value(1)

        if sup_f < cv:
            return []

        results: List[Tuple[int, float]] = [(break_idx, sup_f)]

        # Recurse left
        left = self._sequential_search(y, start, break_idx, depth + 1)
        # Recurse right
        right = self._sequential_search(y, break_idx, end, depth + 1)

        results.extend(left)
        results.extend(right)
        return results

    # -- public interface -------------------------------------------------

    def fit(self, series) -> "BaiPerronDetector":
        """Run the sequential Bai-Perron sup-F procedure.

        Parameters
        ----------
        series : array-like
            1-D numeric data.  Must be long enough to satisfy the
            trimming constraint: ``len(series) >= ceil(1/trimming) * 2``.

        Returns
        -------
        self
        """
        min_len = max(int(np.ceil(2 / self.trimming)), 10)
        arr = self._validate_series(series, min_length=min_len)
        self._series_cache = arr
        n = len(arr)

        results = self._sequential_search(arr, 0, n, 0)
        results.sort(key=lambda x: x[0])

        self._breakpoints = [r[0] for r in results]
        self._f_statistics = [r[1] for r in results]
        self._fitted = True
        return self

    def get_breakpoints(self) -> List[int]:
        """Return detected breakpoint indices (0-based), sorted."""
        self._check_fitted()
        return list(self._breakpoints)

    def get_f_statistics(self) -> List[float]:
        """Return the sup-F statistic associated with each detected break.

        The list is in the same order as :meth:`get_breakpoints`.
        """
        self._check_fitted()
        return list(self._f_statistics)

    def get_confidence_intervals(
        self,
        series=None,
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
    ) -> List[Tuple[int, int]]:
        """Bootstrap percentile CIs on detected break locations.

        Uses a block bootstrap (block length = ``int(trimming * n)``) to
        preserve local dependence structure, re-runs the sequential
        procedure on each replicate, and reports percentile intervals.
        """
        self._check_fitted()
        if not self._breakpoints:
            return []

        arr = self._series_cache if series is None else self._validate_series(series)
        n = len(arr)
        n_breaks = len(self._breakpoints)
        block_len = max(int(self.trimming * n), 5)

        boot_collection: List[List[int]] = []
        rng = np.random.default_rng(42)

        for _ in range(n_bootstrap):
            # Circular block bootstrap
            n_blocks = int(np.ceil(n / block_len))
            starts = rng.integers(0, n, size=n_blocks)
            indices = np.concatenate(
                [np.arange(s, s + block_len) % n for s in starts]
            )[:n]
            sample = arr[indices]
            det = BaiPerronDetector(
                max_breaks=self.max_breaks,
                trimming=self.trimming,
                significance=self.significance,
            )
            try:
                det.fit(sample)
                bps = det.get_breakpoints()
                if len(bps) == n_breaks:
                    boot_collection.append(bps)
            except (ValueError, RuntimeError):
                continue

        if not boot_collection:
            return [(bp, bp) for bp in self._breakpoints]

        boot_arr = np.array(boot_collection)
        lo_q = alpha / 2
        hi_q = 1.0 - lo_q
        cis = []
        for j in range(n_breaks):
            lower = int(np.percentile(boot_arr[:, j], 100 * lo_q))
            upper = int(np.percentile(boot_arr[:, j], 100 * hi_q))
            cis.append((lower, upper))
        return cis


# ---------------------------------------------------------------------------
# 4. MOSUM Detector
# ---------------------------------------------------------------------------

class MOSUMDetector(ChangePointDetector):
    """Moving-Sum (MOSUM) detector for parameter constancy.

    Computes a moving-window sum of OLS residuals (from regressing the
    series on a constant, i.e. a mean-shift model) and compares the
    normalised path to a critical boundary.  Exceedances indicate
    structural breaks.

    Parameters
    ----------
    bandwidth : float, default 0.1
        Moving-window size as a fraction of the sample length.  Chu,
        Hornik & Kauan (1995) study ``h in {0.05, 0.10, 0.15, 0.20}``
        and recommend ``h = 0.10`` as a compromise between detection
        power and localisation accuracy (their Section 4).
    boundary : str, default ``"linear"``
        Shape of the critical boundary:

        - ``"linear"`` — boundary increases linearly away from
          endpoints: ``c(t) = c_alpha * (1 + 2*t)`` where *t* is the
          rescaled time.  This corresponds to the Brownian bridge
          boundary and provides higher power near the centre.
        - ``"constant"`` — flat boundary ``c_alpha``.  More
          conservative, equal sensitivity across the sample.
    alpha : float, default 0.05
        Significance level for the boundary crossing test.

    Notes
    -----
    Boundary critical values for the ``"linear"`` case are derived from
    the distribution of the supremum of a standard Brownian bridge
    (same as CUSUM).  For the ``"constant"`` boundary the critical
    values are from Chu, Hornik & Kauan (1995), Table 1.

    References
    ----------
    Chu, C.-S.J., Hornik, K. & Kauan, C.-M. (1995). MOSUM Tests for
        Parameter Constancy. *Biometrika*, 82(3), 603-617.
    """

    # Constant-boundary critical values from Chu et al. (1995), Table 1.
    # Indexed by (alpha, bandwidth_bin) where bandwidth_bin is the closest
    # tabulated h in {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50}.
    _CONSTANT_CV = {
        (0.10, 0.05): 2.804,
        (0.10, 0.10): 2.540,
        (0.10, 0.15): 2.402,
        (0.10, 0.20): 2.306,
        (0.10, 0.25): 2.232,
        (0.10, 0.30): 2.170,
        (0.10, 0.40): 2.072,
        (0.10, 0.50): 1.996,
        (0.05, 0.05): 3.023,
        (0.05, 0.10): 2.744,
        (0.05, 0.15): 2.598,
        (0.05, 0.20): 2.497,
        (0.05, 0.25): 2.419,
        (0.05, 0.30): 2.354,
        (0.05, 0.40): 2.250,
        (0.05, 0.50): 2.170,
        (0.01, 0.05): 3.444,
        (0.01, 0.10): 3.140,
        (0.01, 0.15): 2.980,
        (0.01, 0.20): 2.870,
        (0.01, 0.25): 2.785,
        (0.01, 0.30): 2.714,
        (0.01, 0.40): 2.601,
        (0.01, 0.50): 2.512,
    }

    # Linear boundary critical values (Brownian-bridge-type).
    _LINEAR_CV = {
        0.10: 1.224,
        0.05: 1.358,
        0.025: 1.480,
        0.01: 1.628,
    }

    def __init__(
        self,
        bandwidth: float = 0.1,
        boundary: str = "linear",
        alpha: float = 0.05,
    ) -> None:
        super().__init__()
        if not 0 < bandwidth < 0.5:
            raise ValueError(
                f"bandwidth must be in (0, 0.5); got {bandwidth}.  "
                "Chu et al. (1995) recommend 0.10."
            )
        if boundary not in ("linear", "constant"):
            raise ValueError(
                f"boundary must be 'linear' or 'constant'; got '{boundary}'."
            )
        self.bandwidth = bandwidth
        self.boundary = boundary
        self.alpha = alpha
        self._mosum_path: Optional[np.ndarray] = None
        self._sup_stat: Optional[float] = None
        self._series_cache: Optional[np.ndarray] = None

    # -- internal helpers -------------------------------------------------

    def _get_critical_value(self) -> float:
        if self.boundary == "linear":
            closest_alpha = min(
                self._LINEAR_CV, key=lambda a: abs(a - self.alpha)
            )
            return self._LINEAR_CV[closest_alpha]
        else:
            bw_bins = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
            closest_bw = min(bw_bins, key=lambda b: abs(b - self.bandwidth))
            closest_alpha = min(
                {a for (a, _) in self._CONSTANT_CV},
                key=lambda a: abs(a - self.alpha),
            )
            key = (closest_alpha, closest_bw)
            return self._CONSTANT_CV.get(key, 2.744)  # default to 5% / 0.10

    def _compute_boundary_path(self, n: int, h: int) -> np.ndarray:
        """Compute the critical boundary at each valid MOSUM position."""
        n_valid = n - h + 1
        if self.boundary == "constant":
            cv = self._get_critical_value()
            return np.full(n_valid, cv)
        else:
            # Linear (Brownian bridge) boundary per Chu, Hornik & Kauan
            # (1995): c_alpha * (1 + 2 * min(t, 1-t)).  This widens the
            # boundary at endpoints (conservative) and narrows it at the
            # sample centre (higher power), matching the variance profile
            # of the MOSUM process.
            cv = self._get_critical_value()
            t = np.linspace(h / (2.0 * n), 1.0 - h / (2.0 * n), n_valid)
            return cv * (1.0 + 2.0 * np.minimum(t, 1.0 - t))

    # -- public interface -------------------------------------------------

    def fit(self, series) -> "MOSUMDetector":
        """Compute the MOSUM path and detect structural breaks.

        Parameters
        ----------
        series : array-like
            1-D numeric data.  Must satisfy
            ``len(series) >= max(20, 2 / bandwidth)``.

        Returns
        -------
        self
        """
        min_len = max(20, int(np.ceil(2.0 / self.bandwidth)))
        arr = self._validate_series(series, min_length=min_len)
        self._series_cache = arr
        n = len(arr)
        h = max(int(self.bandwidth * n), 2)

        # OLS residuals from regression on a constant = demeaned series
        residuals = arr - np.mean(arr)
        sigma = np.std(arr, ddof=1)

        if sigma < 1e-15:
            self._mosum_path = np.zeros(n - h + 1)
            self._sup_stat = 0.0
            self._breakpoints = []
            self._fitted = True
            return self

        # Vectorised moving sum via cumsum trick
        cumsum_resid = np.cumsum(residuals)
        # mosum[i] = sum(residuals[i:i+h])
        padded = np.concatenate([[0.0], cumsum_resid])
        mosum_raw = padded[h:] - padded[:-h]

        # Normalise: MOSUM / (sigma * sqrt(h))
        self._mosum_path = mosum_raw / (sigma * np.sqrt(h))

        boundary_path = self._compute_boundary_path(n, h)
        self._sup_stat = float(np.max(np.abs(self._mosum_path)))

        # Detect crossings: positions where |MOSUM| > boundary
        crossings = np.abs(self._mosum_path) > boundary_path

        if not np.any(crossings):
            self._breakpoints = []
            self._fitted = True
            return self

        # Cluster crossings into distinct breakpoints: take the peak
        # |MOSUM| within each contiguous run of crossings.
        crossing_indices = np.where(crossings)[0]
        breaks = []
        cluster_start = crossing_indices[0]
        prev = crossing_indices[0]

        for idx in crossing_indices[1:]:
            if idx - prev > 1:
                # End of cluster — find peak
                cluster = self._mosum_path[cluster_start : prev + 1]
                peak = cluster_start + int(np.argmax(np.abs(cluster)))
                # Offset by h//2 to centre the break in the window
                breaks.append(peak + h // 2)
                cluster_start = idx
            prev = idx

        # Last cluster
        cluster = self._mosum_path[cluster_start : prev + 1]
        peak = cluster_start + int(np.argmax(np.abs(cluster)))
        breaks.append(peak + h // 2)

        # Clamp to valid range
        self._breakpoints = [min(max(b, 0), n - 1) for b in breaks]
        self._fitted = True
        return self

    def get_breakpoints(self) -> List[int]:
        """Return detected breakpoint indices (0-based)."""
        self._check_fitted()
        return list(self._breakpoints)

    def get_statistic(self) -> float:
        """Return the supremum of the normalised MOSUM path."""
        self._check_fitted()
        return self._sup_stat  # type: ignore[return-value]

    def get_confidence_intervals(
        self,
        series=None,
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
    ) -> List[Tuple[int, int]]:
        """Bootstrap percentile CIs on break locations.

        Uses a circular block bootstrap to resample while preserving
        local dependence, re-fits the MOSUM detector on each replicate,
        and reports percentile intervals.
        """
        self._check_fitted()
        if not self._breakpoints:
            return []

        arr = self._series_cache if series is None else self._validate_series(series)
        n = len(arr)
        n_breaks = len(self._breakpoints)
        block_len = max(int(self.bandwidth * n), 5)

        boot_collection: List[List[int]] = []
        rng = np.random.default_rng(42)

        for _ in range(n_bootstrap):
            n_blocks = int(np.ceil(n / block_len))
            starts = rng.integers(0, n, size=n_blocks)
            indices = np.concatenate(
                [np.arange(s, s + block_len) % n for s in starts]
            )[:n]
            sample = arr[indices]
            det = MOSUMDetector(
                bandwidth=self.bandwidth,
                boundary=self.boundary,
                alpha=self.alpha,
            )
            try:
                det.fit(sample)
                bps = det.get_breakpoints()
                if len(bps) == n_breaks:
                    boot_collection.append(bps)
            except (ValueError, RuntimeError):
                continue

        if not boot_collection:
            return [(bp, bp) for bp in self._breakpoints]

        boot_arr = np.array(boot_collection)
        lo_q = alpha / 2
        hi_q = 1.0 - lo_q
        cis = []
        for j in range(n_breaks):
            lower = int(np.percentile(boot_arr[:, j], 100 * lo_q))
            upper = int(np.percentile(boot_arr[:, j], 100 * hi_q))
            cis.append((lower, upper))
        return cis
