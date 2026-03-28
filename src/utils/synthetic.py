"""
Synthetic time series generators for testing changepoint detectors.

This module provides functions that generate time series with known structural
properties (breaks, regime switches, decay patterns) so that detection
algorithms can be validated against ground truth.

All generators return a (series, metadata) tuple where metadata captures the
true data-generating parameters for downstream evaluation.
"""

from __future__ import annotations

import numpy as np


def generate_mean_shift(
    n: int = 1000,
    break_points: list[int] | None = None,
    means: list[float] | None = None,
    std: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Generate a series with abrupt shifts in mean at known break points.

    Models the scenario where a trading signal's expected return changes
    discretely — for example, after a factor is published or a regime shift
    occurs in the underlying market.

    Parameters
    ----------
    n : int
        Length of the series.
    break_points : list[int] or None
        Indices where the mean changes. Defaults to a single break at n//2.
    means : list[float] or None
        Mean value for each segment (len = len(break_points) + 1).
        Defaults to [0.0005, -0.0002], simulating daily returns going from
        positive alpha to near-zero.
    std : float
        Standard deviation (constant across segments).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    series : np.ndarray
        Generated time series of length n.
    metadata : dict
        Ground-truth parameters including break_points and means.
    """
    if break_points is None:
        break_points = [n // 2]
    if means is None:
        means = [0.0005, -0.0002]

    rng = np.random.default_rng(seed)
    series = np.empty(n)

    # Build segment boundaries: [0, bp1, bp2, ..., n]
    boundaries = [0] + list(break_points) + [n]
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        series[start:end] = rng.normal(means[i], std, size=end - start)

    metadata = {
        "break_points": list(break_points),
        "means": list(means),
        "std": std,
        "seed": seed,
        "n": n,
    }
    return series, metadata


def generate_variance_shift(
    n: int = 1000,
    break_points: list[int] | None = None,
    stds: list[float] | None = None,
    mean: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Generate a series with abrupt shifts in variance (volatility regimes).

    Models transitions between low-volatility and high-volatility environments,
    which is common in financial markets (e.g., calm periods followed by crises).

    Parameters
    ----------
    n : int
        Length of the series.
    break_points : list[int] or None
        Indices where variance changes. Defaults to a single break at n//2.
    stds : list[float] or None
        Standard deviation for each segment. Defaults to [0.01, 0.03]
        (low vol to high vol).
    mean : float
        Constant mean across all segments.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    series : np.ndarray
        Generated time series of length n.
    metadata : dict
        Ground-truth parameters including break_points and stds.
    """
    if break_points is None:
        break_points = [n // 2]
    if stds is None:
        stds = [0.01, 0.03]

    rng = np.random.default_rng(seed)
    series = np.empty(n)

    boundaries = [0] + list(break_points) + [n]
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        series[start:end] = rng.normal(mean, stds[i], size=end - start)

    metadata = {
        "break_points": list(break_points),
        "stds": list(stds),
        "mean": mean,
        "seed": seed,
        "n": n,
    }
    return series, metadata


def generate_gradual_decay(
    n: int = 1000,
    initial_mean: float = 0.001,
    decay_rate: float = 0.003,
    std: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Generate a series with exponentially decaying mean.

    Models the gradual erosion of a trading signal's alpha over time, as
    described by McLean & Pontiff (2016): after publication, factor premia
    decay as more capital chases the same anomaly.

    The true conditional mean at time t is:
        mu(t) = initial_mean * exp(-decay_rate * t)

    Parameters
    ----------
    n : int
        Length of the series.
    initial_mean : float
        Mean return at t=0.
    decay_rate : float
        Exponential decay rate. Larger values produce faster decay.
    std : float
        Constant noise standard deviation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    series : np.ndarray
        Generated time series of length n.
    metadata : dict
        Ground-truth parameters including the true mean function evaluated
        at each time step.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    true_means = initial_mean * np.exp(-decay_rate * t)
    series = rng.normal(true_means, std)

    metadata = {
        "initial_mean": initial_mean,
        "decay_rate": decay_rate,
        "true_means": true_means,
        "std": std,
        "seed": seed,
        "n": n,
    }
    return series, metadata


def generate_regime_switching(
    n: int = 1000,
    n_regimes: int = 2,
    regime_means: list[float] | None = None,
    regime_stds: list[float] | None = None,
    transition_matrix: list[list[float]] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Generate data from a Markov-switching process.

    Simulates a market with discrete hidden states (e.g., bull/bear, or
    risk-on/risk-off/crisis) where transitions follow a Markov chain. This is
    the ground-truth DGP that Hamilton-filter-based detectors attempt to
    recover.

    Parameters
    ----------
    n : int
        Length of the series.
    n_regimes : int
        Number of hidden regimes.
    regime_means : list[float] or None
        Mean for each regime. Defaults to [0.0005, -0.0001] for 2 regimes.
    regime_stds : list[float] or None
        Std for each regime. Defaults to [0.01, 0.02] for 2 regimes.
    transition_matrix : list[list[float]] or None
        Row-stochastic transition matrix P where P[i][j] = Pr(s_{t+1}=j | s_t=i).
        Defaults to [[0.98, 0.02], [0.05, 0.95]] for 2 regimes.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    series : np.ndarray
        Generated time series of length n.
    metadata : dict
        Ground-truth parameters including the true regime sequence.
    """
    if regime_means is None:
        regime_means = [0.0005, -0.0001]
    if regime_stds is None:
        regime_stds = [0.01, 0.02]
    if transition_matrix is None:
        transition_matrix = [[0.98, 0.02], [0.05, 0.95]]

    rng = np.random.default_rng(seed)
    P = np.array(transition_matrix)

    # Simulate regime sequence
    regimes = np.empty(n, dtype=int)
    regimes[0] = 0  # start in regime 0
    for t in range(1, n):
        regimes[t] = rng.choice(n_regimes, p=P[regimes[t - 1]])

    # Draw observations conditional on regime
    series = np.empty(n)
    for t in range(n):
        r = regimes[t]
        series[t] = rng.normal(regime_means[r], regime_stds[r])

    metadata = {
        "n_regimes": n_regimes,
        "regime_means": list(regime_means),
        "regime_stds": list(regime_stds),
        "transition_matrix": P.tolist(),
        "regimes": regimes,
        "seed": seed,
        "n": n,
    }
    return series, metadata


def generate_factor_with_decay(
    n: int = 2000,
    alpha_period: int = 1000,
    decay_period: int = 500,
    dead_period: int = 500,
    alpha_mean: float = 0.0003,
    decay_end_mean: float = 0.0,
    dead_mean: float = -0.0001,
    std: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Simulate a realistic factor lifecycle: alpha, decay, and crowding.

    Models three distinct phases that a quantitative factor typically
    experiences:

    1. **Alpha period** — the signal generates genuine excess returns before
       widespread adoption.
    2. **Decay period** — returns decay linearly as the anomaly becomes known
       and capital flows in.
    3. **Dead period** — the signal is fully arbitraged away (or even
       reversed due to crowding).

    The total series length is alpha_period + decay_period + dead_period; the
    ``n`` parameter is overridden by this sum.

    Parameters
    ----------
    n : int
        Ignored in favor of the sum of period lengths; kept for API
        consistency.
    alpha_period : int
        Number of observations in the alpha phase.
    decay_period : int
        Number of observations in the linear-decay phase.
    dead_period : int
        Number of observations in the dead/crowded phase.
    alpha_mean : float
        Mean return during the alpha phase.
    decay_end_mean : float
        Mean return at the end of the decay phase.
    dead_mean : float
        Mean return during the dead phase.
    std : float
        Constant noise standard deviation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    series : np.ndarray
        Generated time series.
    metadata : dict
        Ground-truth parameters including break points and regime labels.
    """
    total_n = alpha_period + decay_period + dead_period
    rng = np.random.default_rng(seed)

    # Build the true mean trajectory
    true_means = np.empty(total_n)

    # Phase 1: constant alpha
    true_means[:alpha_period] = alpha_mean

    # Phase 2: linear decay from alpha_mean to decay_end_mean
    if decay_period > 0:
        true_means[alpha_period : alpha_period + decay_period] = np.linspace(
            alpha_mean, decay_end_mean, decay_period
        )

    # Phase 3: constant dead mean
    true_means[alpha_period + decay_period :] = dead_mean

    series = rng.normal(true_means, std)

    # Regime labels: 0=alpha, 1=decay, 2=dead
    regime_labels = np.empty(total_n, dtype=int)
    regime_labels[:alpha_period] = 0
    regime_labels[alpha_period : alpha_period + decay_period] = 1
    regime_labels[alpha_period + decay_period :] = 2

    break_points = [alpha_period, alpha_period + decay_period]

    metadata = {
        "break_points": break_points,
        "regime_labels": regime_labels,
        "true_means": true_means,
        "alpha_period": alpha_period,
        "decay_period": decay_period,
        "dead_period": dead_period,
        "alpha_mean": alpha_mean,
        "decay_end_mean": decay_end_mean,
        "dead_mean": dead_mean,
        "std": std,
        "seed": seed,
        "n": total_n,
    }
    return series, metadata


def generate_ou_process(
    n: int = 1000,
    theta: float = 0.05,
    mu: float = 0.0,
    sigma: float = 0.01,
    x0: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Generate a discrete-time Ornstein-Uhlenbeck process.

    The O-U process is the continuous-time analogue of an AR(1) and is widely
    used to model mean-reverting spreads in pairs trading and statistical
    arbitrage. The SDE is:

        dx = theta * (mu - x) * dt + sigma * dW

    This function uses an Euler-Maruyama discretisation with dt = 1.

    The true half-life (time for the expected deviation from mu to halve) is
    log(2) / theta.

    Parameters
    ----------
    n : int
        Length of the series.
    theta : float
        Mean-reversion speed. Larger theta means faster reversion.
    mu : float
        Long-run mean.
    sigma : float
        Volatility of the noise term.
    x0 : float
        Initial value of the process.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    series : np.ndarray
        Generated O-U path of length n.
    metadata : dict
        Ground-truth parameters including theta and the true half-life.
    """
    rng = np.random.default_rng(seed)
    series = np.empty(n)
    series[0] = x0
    dt = 1.0

    noise = rng.normal(0.0, sigma * np.sqrt(dt), size=n - 1)
    for t in range(1, n):
        series[t] = series[t - 1] + theta * (mu - series[t - 1]) * dt + noise[t - 1]

    half_life = np.log(2) / theta

    metadata = {
        "theta": theta,
        "mu": mu,
        "sigma": sigma,
        "x0": x0,
        "half_life": half_life,
        "seed": seed,
        "n": n,
    }
    return series, metadata


def generate_benchmark_suite(seed: int = 42) -> dict[str, tuple[np.ndarray, dict]]:
    """Generate a reproducible suite of synthetic test cases.

    Returns a dictionary of named scenarios covering the major structural
    patterns that changepoint detectors should handle: abrupt mean shifts,
    variance shifts, gradual decay, Markov regime switching, realistic
    factor lifecycles, and mean-reverting processes with different speeds.

    Every case uses a deterministic seed derived from the base ``seed`` so
    the entire suite is fully reproducible.

    Parameters
    ----------
    seed : int
        Base random seed. Individual cases use seed, seed+1, seed+2, ...

    Returns
    -------
    suite : dict[str, tuple[np.ndarray, dict]]
        Mapping from descriptive name to (series, metadata) tuples.
    """
    suite: dict[str, tuple[np.ndarray, dict]] = {}

    suite["single_break_mean"] = generate_mean_shift(
        n=1000, break_points=[500], means=[0.0005, -0.0002], seed=seed
    )

    suite["double_break_mean"] = generate_mean_shift(
        n=1000,
        break_points=[300, 700],
        means=[0.0005, 0.0, -0.0003],
        seed=seed + 1,
    )

    # Null case: pure noise with no structural break
    suite["no_break"] = generate_mean_shift(
        n=1000, break_points=[], means=[0.0], std=0.01, seed=seed + 2
    )

    suite["variance_shift"] = generate_variance_shift(
        n=1000, break_points=[500], stds=[0.01, 0.03], seed=seed + 3
    )

    suite["gradual_decay"] = generate_gradual_decay(
        n=1000, initial_mean=0.001, decay_rate=0.003, seed=seed + 4
    )

    suite["regime_switching_2"] = generate_regime_switching(
        n=1000,
        n_regimes=2,
        regime_means=[0.0005, -0.0001],
        regime_stds=[0.01, 0.02],
        transition_matrix=[[0.98, 0.02], [0.05, 0.95]],
        seed=seed + 5,
    )

    suite["regime_switching_3"] = generate_regime_switching(
        n=1500,
        n_regimes=3,
        regime_means=[0.0008, 0.0, -0.0005],
        regime_stds=[0.008, 0.015, 0.025],
        transition_matrix=[
            [0.96, 0.03, 0.01],
            [0.04, 0.92, 0.04],
            [0.02, 0.05, 0.93],
        ],
        seed=seed + 6,
    )

    suite["factor_lifecycle"] = generate_factor_with_decay(
        alpha_period=1000,
        decay_period=500,
        dead_period=500,
        alpha_mean=0.0003,
        decay_end_mean=0.0,
        dead_mean=-0.0001,
        seed=seed + 7,
    )

    suite["ou_fast"] = generate_ou_process(
        n=1000, theta=0.1, mu=0.0, sigma=0.01, seed=seed + 8
    )

    suite["ou_slow"] = generate_ou_process(
        n=1000, theta=0.01, mu=0.0, sigma=0.01, seed=seed + 9
    )

    return suite
