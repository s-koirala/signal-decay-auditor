"""
Regime-switching models for detecting changes in factor return dynamics.

This module implements three complementary approaches to regime detection:

1. MarkovRegimeDetector — Markov-switching regression (Hamilton, 1989)
2. HMMRegimeDetector — Gaussian Hidden Markov Models (Hamilton, 1989; Rabiner, 1989)
3. GARCHRegimeDetector — GARCH-family volatility models with changepoint detection
   (Bollerslev, 1986; Nelson, 1991; Glosten-Jagannathan-Runkle, 1993)

References
----------
- Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle." Econometrica, 57(2), 357-384.
- Rabiner, L.R. (1989). "A Tutorial on Hidden Markov Models and Selected
  Applications in Speech Recognition." Proceedings of the IEEE, 77(2), 257-286.
- Bollerslev, T. (1986). "Generalized Autoregressive Conditional
  Heteroskedasticity." Journal of Econometrics, 31(3), 307-327.
- Nelson, D.B. (1991). "Conditional Heteroskedasticity in Asset Returns:
  A New Approach." Econometrica, 59(2), 347-370.
- Glosten, L.R., Jagannathan, R., & Runkle, D.E. (1993). "On the Relation
  between the Expected Value and the Volatility of the Nominal Excess Return
  on Stocks." Journal of Finance, 48(5), 1779-1801.
"""

import logging
import warnings
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from arch import arch_model
import ruptures

logger = logging.getLogger(__name__)


def _validate_series(series: Union[np.ndarray, pd.Series, pd.DataFrame]) -> np.ndarray:
    """Validate input series: reject NaN values and convert to numpy array.

    Parameters
    ----------
    series : array-like
        Input time series data.

    Returns
    -------
    np.ndarray
        Validated numpy array.

    Raises
    ------
    ValueError
        If series contains NaN values or is empty.
    """
    if isinstance(series, (pd.Series, pd.DataFrame)):
        arr = series.values
    else:
        arr = np.asarray(series)

    if arr.size == 0:
        raise ValueError("Input series is empty.")

    if np.any(np.isnan(arr)):
        raise ValueError(
            "Input series contains NaN values. Regime-switching models require "
            "clean data. Impute or drop NaN values before fitting."
        )

    return arr


class MarkovRegimeDetector:
    """Markov-switching regression detector for factor return regimes.

    Wraps ``statsmodels.tsa.regime_switching.MarkovRegression`` to fit
    Hamilton (1989) regime-switching models with regime-dependent intercept
    and optionally switching variance.

    Parameters
    ----------
    n_regimes : int, default 2
        Number of regimes. Two regimes is the canonical specification from
        Hamilton (1989) and is sufficient for most factor decay applications
        (bull/bear or alpha/no-alpha states). Higher counts risk overfitting
        on short samples.
    switching_variance : bool, default True
        Whether regime-specific variances are estimated. Enabled by default
        because factor return volatility typically differs across regimes
        (e.g., crisis vs. calm periods).

    Notes
    -----
    The intercept (mean) always switches across regimes in
    ``statsmodels.MarkovRegression``. There is no option to disable
    switching of the constant term, so no ``switching_mean`` parameter
    is exposed.

    References
    ----------
    Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of
    Nonstationary Time Series and the Business Cycle." Econometrica, 57(2),
    357-384.

    Examples
    --------
    >>> detector = MarkovRegimeDetector(n_regimes=2)
    >>> detector.fit(returns_series)
    >>> regimes = detector.get_regimes()
    """

    def __init__(
        self,
        n_regimes: int = 2,
        switching_variance: bool = True,
    ):
        self.n_regimes = n_regimes
        self.switching_variance = switching_variance

        self._fitted_model = None
        self._smoothed_probabilities = None
        self._regime_means = None
        self._regime_variances = None
        self._transition_matrix = None

    def fit(self, series: Union[np.ndarray, pd.Series], n_init: int = 10) -> "MarkovRegimeDetector":
        """Fit the Markov-switching model via EM with multiple random starts.

        Runs EM estimation ``n_init`` times with different random
        initializations and selects the fit with the highest log-likelihood
        to mitigate sensitivity to local optima.

        Parameters
        ----------
        series : array-like, 1D
            Factor returns series. Must not contain NaN.
        n_init : int, default 10
            Number of random EM initializations. 10 provides a good
            trade-off between robustness and computation time for typical
            factor return series (250-5000 observations).

        Returns
        -------
        self
            Fitted detector instance.

        Raises
        ------
        ValueError
            If series contains NaN values.
        """
        arr = _validate_series(series)
        if arr.ndim != 1:
            raise ValueError("MarkovRegimeDetector requires a 1D series.")

        # Use pandas Series for statsmodels compatibility
        if isinstance(series, pd.Series):
            endog = series
        else:
            endog = pd.Series(arr)

        best_model = None
        best_llf = -np.inf

        for i in range(n_init):
            try:
                model = MarkovRegression(
                    endog,
                    k_regimes=self.n_regimes,
                    switching_variance=self.switching_variance,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Generate random starting parameters using a local RNG
                    # to avoid mutating global random state
                    rng = np.random.default_rng(i * 42 + 7)
                    start_params = model.start_params.copy()
                    start_params += rng.normal(scale=0.1, size=start_params.shape)
                    fitted = model.fit(disp=False, start_params=start_params)

                if fitted.llf > best_llf:
                    best_llf = fitted.llf
                    best_model = fitted

            except Exception as exc:
                logger.debug(
                    "Markov regression initialization %d/%d failed: %s",
                    i + 1, n_init, exc,
                )
                continue

        if best_model is None:
            raise RuntimeError(
                "All Markov regression initializations failed. Check that the "
                "series has sufficient length and variance for regime detection."
            )

        self._fitted_model = best_model
        self._smoothed_probabilities = best_model.smoothed_marginal_probabilities

        # Extract regime means
        self._regime_means = np.array([
            best_model.params[f"const[{k}]"] if f"const[{k}]" in best_model.params.index
            else best_model.params.iloc[k]
            for k in range(self.n_regimes)
        ])

        # Extract regime variances
        if self.switching_variance:
            self._regime_variances = np.array([
                best_model.params[f"sigma2[{k}]"] if f"sigma2[{k}]" in best_model.params.index
                else best_model.params.iloc[self.n_regimes + k]
                for k in range(self.n_regimes)
            ])
        else:
            sigma2 = best_model.params.get("sigma2", best_model.params.iloc[self.n_regimes])
            self._regime_variances = np.full(self.n_regimes, sigma2)

        # Extract transition matrix from the fitted model
        self._transition_matrix = best_model.regime_transition

        logger.info(
            "MarkovRegimeDetector fitted with %d regimes. Log-likelihood: %.4f",
            self.n_regimes, best_llf,
        )

        return self

    def get_regimes(self) -> np.ndarray:
        """Return the most likely regime at each time step.

        Uses argmax of smoothed probabilities, analogous to Viterbi
        decoding for the most probable state sequence.

        Returns
        -------
        np.ndarray
            Array of regime labels (0, 1, ..., n_regimes-1).
        """
        self._check_fitted()
        probs = self._smoothed_probabilities
        # smoothed_marginal_probabilities is a list of arrays or a DataFrame
        if isinstance(probs, list):
            prob_matrix = np.column_stack(probs)
        elif isinstance(probs, pd.DataFrame):
            prob_matrix = probs.values
        else:
            prob_matrix = np.array(probs)
        return np.argmax(prob_matrix, axis=1)

    def get_smoothed_probabilities(self) -> pd.DataFrame:
        """Return smoothed regime probabilities at each time step.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'regime_0', 'regime_1', etc., containing
            P(S_t = k | Y_1, ..., Y_T) for each regime k.
        """
        self._check_fitted()
        probs = self._smoothed_probabilities
        if isinstance(probs, list):
            prob_matrix = np.column_stack(probs)
        elif isinstance(probs, pd.DataFrame):
            return probs.rename(
                columns={c: f"regime_{c}" for c in probs.columns}
            )
        else:
            prob_matrix = np.array(probs)

        columns = [f"regime_{k}" for k in range(self.n_regimes)]
        return pd.DataFrame(prob_matrix, columns=columns)

    def get_regime_statistics(self) -> Dict[int, Dict[str, float]]:
        """Return summary statistics for each detected regime.

        Returns
        -------
        dict
            Keys are regime indices; values contain 'mean', 'variance',
            and 'expected_duration' (expected number of periods spent in
            the regime, computed as 1 / (1 - p_ii)).
        """
        self._check_fitted()
        trans = self.get_transition_matrix()
        stats = {}
        for k in range(self.n_regimes):
            p_ii = trans[k, k]
            expected_duration = 1.0 / (1.0 - p_ii) if p_ii < 1.0 else np.inf
            stats[k] = {
                "mean": float(np.asarray(self._regime_means[k]).flat[0]),
                "variance": float(np.asarray(self._regime_variances[k]).flat[0]),
                "expected_duration": float(np.asarray(expected_duration).flat[0]),
            }
        return stats

    def get_transition_matrix(self) -> np.ndarray:
        """Return the estimated transition probability matrix.

        Returns
        -------
        np.ndarray
            Shape (n_regimes, n_regimes). Entry (i, j) is P(S_t = j | S_{t-1} = i).
        """
        self._check_fitted()
        trans = self._transition_matrix
        if isinstance(trans, pd.DataFrame):
            return trans.values
        return np.asarray(trans)

    def select_n_regimes(
        self,
        series: Union[np.ndarray, pd.Series],
        max_regimes: int = 4,
    ) -> int:
        """Select optimal number of regimes via BIC.

        Fits models with 2 through ``max_regimes`` regimes and selects the
        specification that minimizes the Bayesian Information Criterion.

        Parameters
        ----------
        series : array-like, 1D
            Factor returns series.
        max_regimes : int, default 4
            Maximum number of regimes to evaluate. Capped at 4 by default
            because financial regime-switching models rarely benefit from
            more than 3-4 states and higher counts lead to estimation
            instability.

        Returns
        -------
        int
            Optimal number of regimes.
        """
        arr = _validate_series(series)
        if arr.ndim != 1:
            raise ValueError("select_n_regimes requires a 1D series.")

        best_bic = np.inf
        best_k = 2

        for k in range(2, max_regimes + 1):
            try:
                detector = MarkovRegimeDetector(
                    n_regimes=k,
                    switching_variance=self.switching_variance,
                )
                detector.fit(series)
                bic = detector._fitted_model.bic
                logger.info("MarkovRegime k=%d BIC=%.4f", k, bic)
                if bic < best_bic:
                    best_bic = bic
                    best_k = k
            except Exception as exc:
                logger.warning(
                    "Failed to fit Markov regime model with k=%d: %s", k, exc
                )
                continue

        logger.info("Optimal n_regimes=%d (BIC=%.4f)", best_k, best_bic)
        return best_k

    def _check_fitted(self) -> None:
        if self._fitted_model is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before accessing results."
            )


class HMMRegimeDetector:
    """Gaussian Hidden Markov Model detector for factor return regimes.

    Wraps ``hmmlearn.GaussianHMM`` for flexible univariate or multivariate
    regime detection using the Baum-Welch (EM) algorithm for parameter
    estimation and the Viterbi algorithm for state decoding.

    Parameters
    ----------
    n_regimes : int, default 2
        Number of hidden states. Two is the minimal specification for
        regime detection and is standard in the Hamilton (1989) framework.
    covariance_type : str, default "full"
        Covariance matrix type: "full" (default, allows correlation between
        features in multivariate case), "diag" (diagonal, independent
        features), or "spherical" (isotropic).
    n_iter : int, default 100
        Maximum number of EM iterations. 100 is typically sufficient for
        convergence on financial time series of moderate length.

    References
    ----------
    Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of
    Nonstationary Time Series and the Business Cycle." Econometrica, 57(2),
    357-384.

    Rabiner, L.R. (1989). "A Tutorial on Hidden Markov Models and Selected
    Applications in Speech Recognition." Proceedings of the IEEE, 77(2),
    257-286.

    Examples
    --------
    >>> detector = HMMRegimeDetector(n_regimes=3, covariance_type="diag")
    >>> detector.fit(returns_series)
    >>> states = detector.get_regimes()
    """

    def __init__(
        self,
        n_regimes: int = 2,
        covariance_type: str = "full",
        n_iter: int = 100,
    ):
        if covariance_type not in ("full", "diag", "spherical"):
            raise ValueError(
                f"covariance_type must be 'full', 'diag', or 'spherical', "
                f"got '{covariance_type}'."
            )
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter

        self._fitted_model: Optional[GaussianHMM] = None
        self._decoded_states: Optional[np.ndarray] = None
        self._state_means: Optional[np.ndarray] = None
        self._state_covariances: Optional[np.ndarray] = None
        self._transition_matrix: Optional[np.ndarray] = None
        self._X: Optional[np.ndarray] = None

    def fit(
        self,
        series: Union[np.ndarray, pd.Series, pd.DataFrame],
        n_init: int = 10,
    ) -> "HMMRegimeDetector":
        """Fit the Gaussian HMM with multiple random initializations.

        Parameters
        ----------
        series : array-like
            1D (univariate) or 2D (multivariate, e.g. returns + volatility).
            Must not contain NaN.
        n_init : int, default 10
            Number of random initializations. The fit with the highest
            log-likelihood score is retained.

        Returns
        -------
        self
            Fitted detector instance.

        Raises
        ------
        ValueError
            If series contains NaN values.
        """
        arr = _validate_series(series)

        # Ensure 2D shape (n_samples, n_features)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        self._X = arr
        best_model = None
        best_score = -np.inf

        for i in range(n_init):
            try:
                model = GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    random_state=i * 42 + 7,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(arr)

                score = model.score(arr)
                if score > best_score:
                    best_score = score
                    best_model = model

            except Exception as exc:
                logger.debug(
                    "HMM initialization %d/%d failed: %s", i + 1, n_init, exc
                )
                continue

        if best_model is None:
            raise RuntimeError(
                "All HMM initializations failed. Check series length and variance."
            )

        self._fitted_model = best_model
        self._decoded_states = best_model.predict(arr)
        self._state_means = best_model.means_
        self._state_covariances = best_model.covars_
        self._transition_matrix = best_model.transmat_

        logger.info(
            "HMMRegimeDetector fitted with %d states. Score: %.4f",
            self.n_regimes, best_score,
        )

        return self

    def get_regimes(self) -> np.ndarray:
        """Return the Viterbi-decoded state sequence.

        Returns
        -------
        np.ndarray
            Array of state labels (0, 1, ..., n_regimes-1).
        """
        self._check_fitted()
        return self._decoded_states.copy()

    def get_state_probabilities(self) -> pd.DataFrame:
        """Return posterior state probabilities at each time step.

        Uses the forward-backward algorithm to compute
        P(S_t = k | Y_1, ..., Y_T).

        Returns
        -------
        pd.DataFrame
            Columns 'state_0', 'state_1', ..., with posterior probabilities.
        """
        self._check_fitted()
        posteriors = self._fitted_model.predict_proba(self._X)
        columns = [f"state_{k}" for k in range(self.n_regimes)]
        return pd.DataFrame(posteriors, columns=columns)

    def get_regime_statistics(self) -> Dict[int, Dict[str, float]]:
        """Return summary statistics for each hidden state.

        Returns
        -------
        dict
            Keys are state indices; values contain 'mean' (scalar for
            univariate, array for multivariate), 'variance', and
            'expected_duration' computed as 1 / (1 - p_ii).
        """
        self._check_fitted()
        stats = {}
        for k in range(self.n_regimes):
            p_ii = self._transition_matrix[k, k]
            expected_duration = 1.0 / (1.0 - p_ii) if p_ii < 1.0 else np.inf

            mean = self._state_means[k]
            cov = self._state_covariances[k]

            # For univariate case, flatten to scalar
            if mean.size == 1:
                mean_val = float(mean.flatten()[0])
                var_val = float(cov.flatten()[0])
            else:
                mean_val = mean.tolist()
                var_val = cov.tolist()

            stats[k] = {
                "mean": mean_val,
                "variance": var_val,
                "expected_duration": float(expected_duration),
            }
        return stats

    def select_n_regimes(
        self,
        series: Union[np.ndarray, pd.Series, pd.DataFrame],
        max_regimes: int = 5,
    ) -> int:
        """Select optimal number of hidden states via BIC.

        BIC = -2 * log_likelihood + n_params * log(n_observations)

        Parameters
        ----------
        series : array-like
            Input time series (1D or 2D).
        max_regimes : int, default 5
            Maximum number of states to evaluate. Default of 5 is generous
            for financial applications; most factor series are well-described
            by 2-3 regimes.

        Returns
        -------
        int
            Optimal number of regimes.
        """
        arr = _validate_series(series)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        n_samples, n_features = arr.shape
        best_bic = np.inf
        best_k = 2

        for k in range(2, max_regimes + 1):
            try:
                detector = HMMRegimeDetector(
                    n_regimes=k,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                )
                detector.fit(series)

                log_likelihood = detector._fitted_model.score(arr)

                # Number of free parameters:
                # - Transition matrix: k*(k-1) (rows sum to 1)
                # - Means: k * n_features
                # - Covariances depend on type
                # - Initial state probs: k-1
                n_params = k * (k - 1) + k * n_features + (k - 1)
                if self.covariance_type == "full":
                    n_params += k * n_features * (n_features + 1) // 2
                elif self.covariance_type == "diag":
                    n_params += k * n_features
                elif self.covariance_type == "spherical":
                    n_params += k

                bic = -2.0 * log_likelihood + n_params * np.log(n_samples)
                logger.info("HMM k=%d BIC=%.4f", k, bic)

                if bic < best_bic:
                    best_bic = bic
                    best_k = k

            except Exception as exc:
                logger.warning("Failed to fit HMM with k=%d: %s", k, exc)
                continue

        logger.info("Optimal n_regimes=%d (BIC=%.4f)", best_k, best_bic)
        return best_k

    def _check_fitted(self) -> None:
        if self._fitted_model is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before accessing results."
            )


class GARCHRegimeDetector:
    """GARCH-family volatility regime detector.

    Fits a GARCH-type model to extract the conditional volatility series,
    then applies the PELT (Pruned Exact Linear Time) changepoint algorithm
    to detect structural breaks in volatility, identifying distinct
    volatility regimes.

    Parameters
    ----------
    vol_model : str, default "GARCH"
        Volatility model specification:
        - "GARCH": Standard GARCH(p,q) of Bollerslev (1986).
        - "EGARCH": Exponential GARCH of Nelson (1991). Captures asymmetric
          volatility response (leverage effect).
        - "GJR-GARCH": GJR-GARCH of Glosten, Jagannathan & Runkle (1993).
          Threshold model for asymmetric shocks.
    p : int, default 1
        Lag order for the volatility (GARCH) term. p=1 is standard and
        sufficient for most daily/monthly factor returns.
    q : int, default 1
        Lag order for the squared residual (ARCH) term. q=1 is standard.
    dist : str, default "normal"
        Error distribution: "normal" (Gaussian), "t" (Student-t for fat
        tails), or "skewt" (skewed Student-t for asymmetric fat tails).

    References
    ----------
    Bollerslev, T. (1986). "Generalized Autoregressive Conditional
    Heteroskedasticity." Journal of Econometrics, 31(3), 307-327.

    Nelson, D.B. (1991). "Conditional Heteroskedasticity in Asset Returns:
    A New Approach." Econometrica, 59(2), 347-370.

    Glosten, L.R., Jagannathan, R., & Runkle, D.E. (1993). "On the Relation
    between the Expected Value and the Volatility of the Nominal Excess Return
    on Stocks." Journal of Finance, 48(5), 1779-1801.

    Examples
    --------
    >>> detector = GARCHRegimeDetector(vol_model="GJR-GARCH", dist="t")
    >>> detector.fit(returns_series)
    >>> vol_regimes = detector.get_volatility_regimes()
    """

    # Map user-facing names to arch package vol parameter
    _VOL_MODEL_MAP = {
        "GARCH": "GARCH",
        "EGARCH": "EGARCH",
        "GJR-GARCH": "GARCH",  # Uses GJR via the o parameter
    }

    def __init__(
        self,
        vol_model: str = "GARCH",
        p: int = 1,
        q: int = 1,
        dist: str = "normal",
    ):
        if vol_model not in ("GARCH", "EGARCH", "GJR-GARCH"):
            raise ValueError(
                f"vol_model must be 'GARCH', 'EGARCH', or 'GJR-GARCH', "
                f"got '{vol_model}'."
            )
        if dist not in ("normal", "t", "skewt"):
            raise ValueError(
                f"dist must be 'normal', 't', or 'skewt', got '{dist}'."
            )

        self.vol_model = vol_model
        self.p = p
        self.q = q
        self.dist = dist

        self._fitted_garch = None
        self._conditional_volatility: Optional[pd.Series] = None
        self._breakpoints: Optional[np.ndarray] = None
        self._regime_labels: Optional[np.ndarray] = None

    def fit(
        self,
        series: Union[np.ndarray, pd.Series],
        penalty_value: float = 3.0,
    ) -> "GARCHRegimeDetector":
        """Fit the GARCH model and detect volatility regime breakpoints.

        Parameters
        ----------
        series : array-like, 1D
            Factor returns series. Must not contain NaN.
        penalty_value : float, default 3.0
            PELT penalty parameter (multiplied by log(n)). Controls
            sensitivity of changepoint detection. Default of 3.0 provides
            a BIC-like penalty that balances false positives and detection
            power for typical financial time series.

        Returns
        -------
        self
            Fitted detector instance.

        Raises
        ------
        ValueError
            If series contains NaN values.
        """
        arr = _validate_series(series)
        if arr.ndim != 1:
            raise ValueError("GARCHRegimeDetector requires a 1D series.")

        # Scale returns to percentage if necessary (arch convention)
        if isinstance(series, pd.Series):
            endog = series
        else:
            endog = pd.Series(arr)

        # Build the GARCH model
        if self.vol_model == "GJR-GARCH":
            model = arch_model(
                endog,
                mean="Constant",
                vol="GARCH",
                p=self.p,
                o=1,  # GJR asymmetric term
                q=self.q,
                dist=self.dist,
            )
        elif self.vol_model == "EGARCH":
            model = arch_model(
                endog,
                mean="Constant",
                vol="EGARCH",
                p=self.p,
                q=self.q,
                dist=self.dist,
            )
        else:
            model = arch_model(
                endog,
                mean="Constant",
                vol="GARCH",
                p=self.p,
                q=self.q,
                dist=self.dist,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fitted_garch = model.fit(disp="off")

        self._conditional_volatility = self._fitted_garch.conditional_volatility

        # Apply PELT changepoint detection on conditional volatility
        vol_values = self._conditional_volatility.values
        n = len(vol_values)
        penalty = penalty_value * np.log(n)

        algo = ruptures.Pelt(model="rbf", min_size=max(2, n // 50)).fit(
            vol_values.reshape(-1, 1)
        )
        breakpoints = algo.predict(pen=penalty)

        # ruptures returns breakpoints including the last index (n)
        # Remove the terminal index
        self._breakpoints = np.array(
            [bp for bp in breakpoints if bp < n], dtype=int
        )

        # Label segments by volatility level
        self._assign_volatility_regimes(vol_values)

        logger.info(
            "GARCHRegimeDetector fitted (%s). Found %d volatility breakpoints.",
            self.vol_model, len(self._breakpoints),
        )

        return self

    def _assign_volatility_regimes(self, vol_values: np.ndarray) -> None:
        """Assign low/medium/high labels to volatility segments."""
        n = len(vol_values)
        # Build segment boundaries
        boundaries = [0] + sorted(self._breakpoints.tolist()) + [n]
        segment_means = []
        segment_indices = []

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            segment_means.append(np.mean(vol_values[start:end]))
            segment_indices.append((start, end))

        # Rank segments by mean volatility
        n_segments = len(segment_means)
        if n_segments == 1:
            # No breakpoints: single regime
            self._regime_labels = np.zeros(n, dtype=int)
            return

        sorted_indices = np.argsort(segment_means)
        rank_map = np.zeros(n_segments, dtype=int)

        if n_segments == 2:
            # Two segments: low (0) and high (1)
            rank_map[sorted_indices[0]] = 0  # low
            rank_map[sorted_indices[1]] = 2  # high
        else:
            # Divide into terciles: low=0, medium=1, high=2
            tercile_size = n_segments / 3.0
            for rank_pos, seg_idx in enumerate(sorted_indices):
                if rank_pos < tercile_size:
                    rank_map[seg_idx] = 0  # low
                elif rank_pos < 2 * tercile_size:
                    rank_map[seg_idx] = 1  # medium
                else:
                    rank_map[seg_idx] = 2  # high

        labels = np.zeros(n, dtype=int)
        for seg_i, (start, end) in enumerate(segment_indices):
            labels[start:end] = rank_map[seg_i]

        self._regime_labels = labels

    def get_volatility_regimes(self) -> np.ndarray:
        """Return array of volatility regime labels.

        Labels: 0 = low volatility, 1 = medium volatility, 2 = high volatility.

        Returns
        -------
        np.ndarray
            Integer array with regime labels for each time period.
        """
        self._check_fitted()
        return self._regime_labels.copy()

    def get_conditional_volatility(self) -> pd.Series:
        """Return the conditional volatility series from the GARCH model.

        Returns
        -------
        pd.Series
            Conditional volatility (sigma_t) at each time step.
        """
        self._check_fitted()
        return self._conditional_volatility.copy()

    def get_vol_breakpoints(self) -> np.ndarray:
        """Return indices where volatility regime shifts are detected.

        Returns
        -------
        np.ndarray
            Array of integer indices marking regime transition points.
        """
        self._check_fitted()
        return self._breakpoints.copy()

    def _check_fitted(self) -> None:
        if self._fitted_garch is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before accessing results."
            )


def label_signal_state(
    regime_means: Dict[int, float],
    regime_labels: Optional[Dict[int, str]] = None,
) -> Dict[int, str]:
    """Label regimes by signal quality based on their mean returns.

    Given the mean return for each regime from any detector, assigns
    interpretive labels:

    - ``"alpha_generating"`` — regime with the highest mean return,
      indicating the factor signal is producing excess returns.
    - ``"neutral"`` — intermediate regimes (if more than 2 regimes).
    - ``"decayed"`` — regime with the lowest (or most negative) mean
      return, indicating the signal has been arbitraged away or reversed.

    Parameters
    ----------
    regime_means : dict
        Mapping from regime ID (int) to mean return (float). E.g.,
        ``{0: 0.005, 1: -0.002}``.
    regime_labels : dict, optional
        Custom label overrides. If provided, these take precedence over
        the automatic labeling for the specified regime IDs.

    Returns
    -------
    dict
        Mapping from regime ID to label string.

    Examples
    --------
    >>> means = {0: 0.01, 1: -0.003, 2: 0.002}
    >>> label_signal_state(means)
    {0: 'alpha_generating', 1: 'decayed', 2: 'neutral'}
    """
    if not regime_means:
        raise ValueError("regime_means must be a non-empty dict.")

    # Sort regimes by mean return
    sorted_regimes = sorted(regime_means.items(), key=lambda x: x[1])
    n = len(sorted_regimes)

    labels = {}

    if n == 1:
        regime_id = sorted_regimes[0][0]
        mean_val = sorted_regimes[0][1]
        labels[regime_id] = "alpha_generating" if mean_val > 0 else "decayed"
    elif n == 2:
        labels[sorted_regimes[0][0]] = "decayed"
        labels[sorted_regimes[1][0]] = "alpha_generating"
    else:
        # Lowest mean -> decayed, highest mean -> alpha_generating, rest -> neutral
        labels[sorted_regimes[0][0]] = "decayed"
        labels[sorted_regimes[-1][0]] = "alpha_generating"
        for i in range(1, n - 1):
            labels[sorted_regimes[i][0]] = "neutral"

    # Apply custom overrides
    if regime_labels is not None:
        labels.update(regime_labels)

    return labels
