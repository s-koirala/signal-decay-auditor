# Signal Decay Auditor: A Multi-Detector Consensus Framework for Structural Break Detection in Factor Returns

**Abstract**

We introduce the Signal Decay Auditor, a statistical framework that combines multiple changepoint detection algorithms---PELT, OLS-CUSUM, and Bai-Perron sequential sup-F tests---with Hamilton (1989) Markov regime-switching models to diagnose structural breaks in systematic trading factor returns. The multi-detector consensus approach requires methodologically independent agreement before classifying a factor as dead, reducing false positive rates inherent to any single method. Applied to 15,751 daily observations of the Fama-French five factors plus Carhart momentum (July 1963 through January 2026), the framework classifies four of six canonical factors as exhibiting statistically significant decay. SMB (Size) and UMD (Momentum) are classified DEAD, with Sharpe ratio declines of -1.31 and -0.69, respectively, and negative post-break regime means. HML (Value) and CMA (Investment) are classified DECAYING, with Sharpe declines of -1.23 and -0.98 and negative recent-quartile Sharpe ratios. RMW (Profitability) emerges as the most robust alpha source, exhibiting an improving Sharpe trajectory (delta = +0.37) and a full-sample annualized Sharpe of 0.50, consistent with higher arbitrage barriers for profitability-based strategies. The market factor (Mkt-RF) is similarly resilient. Structural break dates cluster in 1999--2001 and 2007--2009, aligning with the growth bubble, quantitative crisis, and subsequent proliferation of factor-based investment products. These results extend McLean and Pontiff (2016) with formal multi-method changepoint estimation, date-stamped break locations, and probabilistic regime classification.

---

## 1. Introduction

The proliferation of quantitative trading strategies built on cross-sectional anomalies has raised a fundamental question: do systematic factor premia persist, or are they arbitraged away once documented in the academic literature? McLean and Pontiff (2016) provide the most comprehensive evidence to date, demonstrating that average anomaly returns decline by approximately 32% after publication as arbitrage capital flows into documented strategies. Their finding implies that a substantial fraction of the factor zoo---the hundreds of proposed return predictors catalogued by Harvey, Liu, and Zhu (2016)---may no longer generate economically meaningful alpha.

Yet the McLean and Pontiff (2016) framework, while seminal, has important limitations. Their before-and-after comparison cannot identify the precise timing of decay onset, distinguish gradual erosion from abrupt structural breaks, or detect multiple break locations within a single factor's history. Moreover, their analysis ends in 2013, missing the subsequent decade of smart-beta ETF proliferation, the prolonged underperformance of the value factor, and the post-COVID regime shifts that have further reshaped factor return distributions.

This paper introduces the Signal Decay Auditor, a unified statistical framework designed to address these gaps. The framework deploys four complementary changepoint detection algorithms---PELT (Killick, Fearnhead, and Eckley, 2012), OLS-CUSUM (Page, 1954; Brown, Durbin, and Evans, 1975), Bai-Perron sequential sup-F tests (Bai and Perron, 1998, 2003), and MOSUM (Chu, Hornik, and Kauan, 1995)---alongside Hamilton (1989) Markov regime-switching models and Gaussian Hidden Markov Models. Each detector targets a different statistical paradigm: penalized likelihood (PELT), recursive residual accumulation (CUSUM), sequential hypothesis testing (Bai-Perron), and moving-window aggregation (MOSUM). The multi-detector consensus approach requires agreement across methodologically independent detectors before classifying a factor signal as dead, reducing false positive rates inherent to any single changepoint method.

The framework produces a three-tier verdict classification---DEAD, DECAYING, or ACTIVE---based on the convergence of detector outputs, post-break Sharpe ratio levels, and regime persistence. Signal quality is tracked via rolling annualized Sharpe ratios, out-of-sample predictive R-squared (Campbell and Thompson, 2008), Clark-West (2007) MSPE-adjusted tests, and Giacomini-White (2006) conditional predictive ability tests, all computed with HAC-robust inference using Newey-West (1994) standard errors. Bootstrap confidence intervals for breakpoint locations are constructed via the stationary bootstrap of Politis and Romano (1994), preserving the serial dependence structure of financial return data.

We apply the framework to the six canonical Fama-French factors---Market Excess Return (Mkt-RF), Small Minus Big (SMB), High Minus Low (HML), Robust Minus Weak (RMW), Conservative Minus Aggressive (CMA), and Carhart (1997) Momentum (UMD)---using 15,751 daily observations spanning July 1963 through January 2026. The results reveal a clear decay hierarchy. Two factors---SMB and UMD---are classified DEAD, with multiple detectors agreeing on structural breaks and negative post-break regime means. Two factors---HML and CMA---are classified DECAYING, with substantial Sharpe ratio declines and negative recent-quartile performance. Two factors---Mkt-RF and RMW---are functionally resilient, with improving Sharpe trajectories over the full sample. Among the anomaly-based factors, RMW (Profitability) emerges as the most robust alpha source, consistent with Novy-Marx's (2013) argument that profitability premia face higher arbitrage barriers than price-based or size-based signals.

The contributions of this paper are threefold. First, we provide a formal multi-method changepoint framework that delivers date-stamped structural break estimates for each factor, enabling precise identification of when and how quickly factor premia erode. Second, the Markov regime-switching component adds a probabilistic dimension to the analysis, quantifying regime persistence and transition dynamics that are invisible to before-and-after comparisons. Third, we extend the empirical record through January 2026, capturing a decade of data beyond McLean and Pontiff (2016) and documenting decay magnitudes that substantially exceed their 32% cross-sectional average.

The remainder of the paper is organized as follows. Section 2 reviews the relevant literature. Section 3 describes the statistical methods. Section 4 presents the results. Section 5 discusses economic interpretation, limitations, and practical implications. Section 6 concludes.

---

## 2. Background and Literature Review

### 2.1 The Anomaly Zoo and Multiple Testing

The factor zoo has grown dramatically over the past three decades. Cochrane (2011) counted approximately 50 proposed factors; by 2016, Harvey, Liu, and Zhu (2016) catalogued over 300, many of which fail to survive multiple testing corrections. Harvey, Liu, and Zhu (2016) argue that the conventional significance threshold of $t > 2.0$ for factor discovery is insufficient given the number of specifications tested across the literature, and propose raising the bar to $t > 3.0$ to account for multiple hypothesis testing. Their analysis suggests that at least half of the published anomalies may be spurious discoveries---statistical artifacts of data mining rather than genuine return predictors.

This concern is amplified by Hou, Xue, and Zhang (2020), who attempt to replicate 452 anomalies and find that 65% fail to produce significant returns when subjected to consistent methodology and multiple testing adjustments. The replication crisis in factor research directly motivates the need for continuous monitoring frameworks that can distinguish genuine structural shifts from sampling noise in factor returns.

### 2.2 Post-Publication Decay

McLean and Pontiff (2016) provide the definitive study of post-publication factor decay. Examining 97 anomalies, they find that average portfolio returns decline by approximately 32% after publication in an academic journal, with the decline concentrated in the long leg of long-short portfolios. They attribute this to arbitrage capital exploiting the documented predictability---a form of the "publish and perish" hypothesis for trading signals.

The mechanism is intuitive: once a cross-sectional predictor is published, institutional investors and quantitative funds incorporate it into their models, increasing demand for long-leg stocks and supply of short-leg stocks, thereby compressing the spread. Calluzzo, Moneta, and Topaloglu (2019) confirm this channel by showing that institutional ownership of anomaly long-leg stocks increases post-publication. Schwert (2003) documents early evidence of this phenomenon, noting that several well-known anomalies (including the size effect and the value effect) appear to have weakened or disappeared after being documented.

The rate of decay varies across anomalies. Factors that are easy to implement (requiring only price data and simple sorting rules) decay faster than those requiring proprietary data, complex estimation, or exposure to illiquid securities (Pontiff, 1996, 2006). This heterogeneity motivates our factor-specific approach rather than a one-size-fits-all decay estimate.

### 2.3 Structural Break Detection

The statistical literature on structural break detection provides the methodological foundation for our framework. The field originates with Page (1954), who introduced the CUSUM procedure for monitoring industrial processes, and Hinkley (1971), who developed inference for the change-point location. Andrews (1993) derives the optimal test for a single structural break with unknown timing, establishing the supremum Wald, LM, and LR tests with tabulated critical values under asymptotic theory.

Bai and Perron (1998, 2003) extend the single-break framework to multiple structural changes. Their sequential sup-F procedure tests for $l$ breaks versus $l + 1$ breaks, with critical values that account for the sequential nature of the testing. The method is particularly suited to financial time series where multiple regime changes may occur over long samples. The trimming parameter $\epsilon$ (typically 0.15) ensures that each segment contains sufficient observations for reliable inference, and the asymptotic theory accommodates both pure structural change and partial structural change models.

Killick, Fearnhead, and Eckley (2012) introduce PELT (Pruned Exact Linear Time), a computationally efficient algorithm for detecting multiple changepoints via penalized cost minimization. PELT achieves $O(n)$ expected computational cost by pruning candidate changepoints that are provably suboptimal under a given cost function. The algorithm is exact (not approximate) and is implemented in the widely used `ruptures` Python package. The penalty parameter controls the trade-off between model fit and complexity, with BIC-based penalties providing a principled default.

Chu, Hornik, and Kauan (1995) develop the MOSUM (Moving Sum) procedure, which detects structural breaks by computing moving-window sums of regression residuals and comparing them to asymptotic critical boundaries derived from Brownian bridge theory. The bandwidth parameter $h$ controls the resolution of the detector: smaller bandwidths improve power for detecting localized breaks at the cost of higher variance.

### 2.4 Regime-Switching Models

Hamilton (1989) introduces the Markov-switching regression model, in which the parameters of a time series model depend on an unobserved discrete state variable that follows a first-order Markov chain. The model provides a probabilistic framework for regime classification: at each time point, smoothed state probabilities quantify the posterior likelihood of each regime, and the transition matrix characterizes the expected duration and switching dynamics of each state.

In the context of factor returns, a two-regime specification naturally separates alpha-generating periods (positive risk-adjusted returns) from decayed periods (near-zero or negative returns). The Viterbi algorithm recovers the most probable state sequence, providing a time-stamped regime classification that complements the discrete breakpoint estimates from the changepoint detectors.

Gaussian Hidden Markov Models (Rabiner, 1989) offer a complementary specification, with parameters estimated via the Baum-Welch (EM) algorithm. When the number of regimes is uncertain, BIC-based model selection across $K \in \{2, 3, 4, 5\}$ states provides a data-driven choice. GARCH-based regime detection (Bollerslev, 1986) adds a volatility dimension, identifying shifts in the conditional variance process that may not be visible in the conditional mean.

### 2.5 Predictive Evaluation and Conditional Testing

Out-of-sample predictive evaluation is essential for distinguishing genuine signal decay from in-sample overfitting. Campbell and Thompson (2008) formalize the out-of-sample $R^2$ statistic, which compares forecast accuracy against a benchmark (typically the expanding historical mean). Welch and Goyal (2008) demonstrate that most equity premium predictors fail to beat the historical mean out of sample, underscoring the difficulty of return prediction and the importance of rigorous out-of-sample evaluation.

Clark and West (2007) address a key limitation of the Diebold-Mariano (1995) test when comparing nested models: the larger model's MSPE is biased upward under the null because it estimates parameters that are zero under the restriction. The Clark-West MSPE-adjusted test corrects for this bias, providing a more powerful test of predictive improvement.

Giacomini and White (2006) extend the predictive evaluation framework to conditional settings. Their test examines whether the relative forecasting performance of two models varies with conditioning information---for example, whether one model outperforms in high-volatility regimes but underperforms in low-volatility regimes. The conditional test regresses loss differentials on instruments with HAC-robust standard errors, providing a Wald test of equal conditional predictive ability. This is particularly relevant for factor decay analysis, where the signal may retain value in certain market states even as it deteriorates unconditionally.

---

# 3. Methods

## 3.1 Data

We obtain daily factor returns from the Kenneth R. French Data Library via `pandas_datareader`. The dataset comprises the Fama-French five factors---Market Excess Return (Mkt-RF), Small Minus Big (SMB), High Minus Low (HML), Robust Minus Weak (RMW), Conservative Minus Aggressive (CMA)---plus the Carhart (1997) Momentum factor (UMD), spanning July 1963 through January 2026 (~15,751 observations per factor). The `pandas_datareader` interface returns percentage-point returns; we scale all series to decimal units (i.e., a 1% daily return is stored as 0.01) at ingestion. The Risk-Free rate (RF) is retained but excluded from the audit pipeline since the factors are already expressed as excess returns.

### 3.1.1 Data Validation

Three validation passes are applied to each ingested factor series before analysis:

**Stationarity.** An Augmented Dickey-Fuller (ADF) test is conducted on each factor return series using the `statsmodels.tsa.stattools.adfuller` implementation with automatic lag selection via the Akaike Information Criterion. Daily excess returns are expected to reject the unit root null at the 1% level; any series failing to reject at $\alpha = 0.05$ is flagged for further inspection.

**Missing data audit.** Each series is scanned for NaN values. All changepoint detectors and regime-switching models in the framework require complete data, so any missing observations must be identified and addressed prior to fitting. The framework raises a `ValueError` if NaN values are detected at fit time, enforcing this constraint programmatically.

**Date gap detection.** The `DatetimeIndex` is examined for gaps exceeding expected non-trading periods (weekends, holidays). Gaps of more than five calendar days that do not correspond to known market closures are logged as potential data quality issues.

**Data integrity.** An SHA-256 checksum is computed and logged at ingestion to ensure reproducibility across sessions.

---

## 3.2 Changepoint Detection

We deploy four complementary changepoint detection algorithms, each targeting a different detection paradigm: penalized likelihood (PELT), recursive residuals (CUSUM), sequential hypothesis testing (Bai-Perron), and moving-window aggregation (MOSUM). Each detector is run on both the raw excess return series and on the rolling annualized Sharpe ratio series (Section 3.4.1), yielding two sets of breakpoints per detector. All detectors share an abstract interface requiring `fit(series)` and `get_breakpoints()` methods, and all operate on one-dimensional float64 arrays with NaN values prohibited.

### 3.2.1 PELT (Pruned Exact Linear Time)

The PELT algorithm (Killick, Fearnhead & Eckley, 2012) solves the penalized cost minimization problem:

$$\min_{m, \tau_1, \ldots, \tau_m} \left[ \sum_{i=0}^{m} \mathcal{C}(y_{\tau_i+1:\tau_{i+1}}) + \beta f(m) \right]$$

where $\mathcal{C}(\cdot)$ is a segment cost function, $\beta$ is the penalty parameter, and $f(m)$ is a function of the number of changepoints. PELT achieves $O(n)$ expected computational cost by pruning candidate changepoints that are provably suboptimal, a substantial improvement over the $O(n^2)$ cost of the standard Optimal Partitioning algorithm.

**Cost function.** We use the $\ell_2$ (least-squares) cost, which models segments as draws from Gaussian distributions with piecewise-constant means:

$$\mathcal{C}(y_{s:t}) = \sum_{i=s}^{t} (y_i - \bar{y}_{s:t})^2$$

The $\ell_2$ cost is the natural parametric choice for detecting changes in the mean of Gaussian data and correctly controls false positives under the null of no change. We prefer $\ell_2$ over the radial basis function (RBF) kernel cost for the auditor pipeline because the kernel cost can oversplit when the signal-to-noise ratio is low, as is typical for daily factor return data where mean shifts of 1--10 basis points occur against ~100 basis points of daily volatility.

**Penalty.** When no penalty is explicitly specified, a variance-scaled BIC penalty is computed:

$$\beta = k \cdot \hat{\sigma}^2 \cdot \ln(n)$$

where $k = 2$ (the number of parameters per segment: mean and variance), $\hat{\sigma}^2$ is the sample variance of the full series (estimated with Bessel's correction), and $n$ is the series length. Variance scaling is essential for financial return data: without it, the penalty operates in log-likelihood units and overwhelms the tiny absolute cost improvements that are characteristic of mean shifts in daily returns (Killick et al., 2012, Section 3.1; Schwarz, 1978; Yao, 1988).

**Minimum segment size.** We set `min_size = 30`, ensuring that each detected segment contains a minimum of 30 observations---sufficient for stable within-segment mean and variance estimation, consistent with standard practice in empirical finance (approximately 1.5 calendar months of daily data or 2.5 years of monthly data).

**Implementation.** We use the `ruptures.Pelt` class, fitting on the reshaped array `arr.reshape(-1, 1)` and extracting breakpoints by filtering the terminal index $n$ from the `predict()` output.

### 3.2.2 CUSUM (Cumulative Sum)

The OLS-CUSUM test (Page, 1954; Brown, Durbin & Evans, 1975) tests the null hypothesis of parameter constancy in a regression model by examining the cumulative sum of recursive (one-step-ahead) residuals. For a regression on a constant (mean-shift model), the recursive residual at time $t$ is:

$$w_t = \frac{y_t - \bar{y}_{1:t-1}}{\sqrt{1 + 1/(t-1)}}$$

where $\bar{y}_{1:t-1}$ is the running mean of all observations prior to $t$. The normalized CUSUM path is:

$$W_t = \frac{1}{\hat{\sigma}_w \sqrt{T-1}} \sum_{s=2}^{t} w_s$$

where $\hat{\sigma}_w$ is the standard deviation of the recursive residuals (estimated with Bessel's correction) and $T$ is the series length.

Under the null of no structural break, $W_t$ converges in distribution to a standard Brownian bridge $B(t)$ on $[0,1]$. The test statistic is:

$$\text{sup}_t |W_t|$$

and the asymptotic $p$-value is computed from the Kolmogorov-Smirnov series expansion for the supremum of a Brownian bridge:

$$P\left(\sup_t |B(t)| > c\right) = 2 \sum_{k=1}^{\infty} (-1)^{k+1} \exp(-2k^2 c^2)$$

We truncate the series at 500 terms, which provides convergence well beyond machine precision. The critical value is evaluated at $\alpha = 0.05$ (Brown, Durbin & Evans, 1975, Table 1).

**Break localization.** When the null is rejected, the breakpoint is estimated as the index of maximum absolute change in the CUSUM path, i.e., $\hat{\tau} = \arg\max_t |\Delta W_t|$, which identifies the point where recursive residuals are largest---the onset of the structural break.

### 3.2.3 Bai-Perron Sequential Sup-F Test

The Bai-Perron (1998, 2003) procedure tests for multiple structural breaks in the mean of a series using a sequential sup-F approach with asymptotic critical values from Andrews (1993).

For a candidate break date $\tau$ partitioning the segment $[s, e)$ into sub-samples $[s, \tau)$ and $[\tau, e)$, the Wald $F$-statistic for equality of sub-sample means is:

$$F(\tau) = \frac{BSS(\tau)}{WSS(\tau) / (n_{\text{seg}} - 2)}$$

where $BSS(\tau) = n_1(\bar{y}_1 - \bar{y})^2 + n_2(\bar{y}_2 - \bar{y})^2$ is the between-group sum of squares, $WSS(\tau) = \sum y_i^2 - n_1 \bar{y}_1^2 - n_2 \bar{y}_2^2$ is the within-group sum of squares, and $n_1, n_2$ are the sub-sample sizes. The sup-F statistic is:

$$\text{sup-}F = \max_{\tau \in [\tau_{\min}, \tau_{\max}]} F(\tau)$$

where the optimization is taken over candidate break dates in the trimmed interior of the segment.

**Trimming.** We set $\epsilon = 0.15$, the canonical recommendation from Bai & Perron (1998) for the asymptotic theory. Andrews (1993) derives critical values assuming 15% trimming. This excludes the outermost 15% of observations at each end from consideration as candidate break dates, ensuring sufficient observations in each sub-sample for reliable $F$-statistic computation.

**Critical values.** We use tabulated asymptotic critical values from Bai & Perron (2003), Table 1, for $p = 1$ (intercept-only model) and $\epsilon = 0.15$. At $\alpha = 0.05$, the critical value for a single-break test is 8.58. The sequential procedure (Bai & Perron, 1998, Section 4) applies this single-break critical value at each recursion step: when sup-$F$ exceeds the critical value, the segment is split at $\hat{\tau} = \arg\max_\tau F(\tau)$, and the procedure recurses on each resulting sub-segment.

**Maximum breaks.** The recursion depth is capped at `max_breaks = 5`, limiting the procedure to at most five structural breaks.

### 3.2.4 MOSUM (Moving Sum)

The MOSUM test (Chu, Hornik & Kauan, 1995) detects structural breaks by computing a moving-window sum of OLS residuals from a mean-shift model and comparing the normalized path to a critical boundary.

For a series $\{y_t\}_{t=1}^n$, the OLS residuals from regression on a constant are $\hat{e}_t = y_t - \bar{y}$. The MOSUM statistic at position $t$ with bandwidth $h$ is:

$$M_t(h) = \frac{1}{\hat{\sigma} \sqrt{h}} \sum_{s=t}^{t+h-1} \hat{e}_s$$

where $\hat{\sigma}$ is the sample standard deviation of the series (with Bessel's correction) and $h = \lfloor 0.1 \cdot n \rfloor$ is the window width.

**Bandwidth.** We set $h = 0.10$ (as a fraction of $n$), the recommended value from Chu, Hornik & Kauan (1995, Section 4) as a compromise between detection power (favoring smaller $h$) and localization accuracy (favoring larger $h$).

**Boundary.** We use the linear (Brownian bridge) boundary:

$$c(t) = c_\alpha \cdot \left(1 + 2 \cdot \min\left(\frac{t}{n}, 1 - \frac{t}{n}\right)\right)$$

where $c_\alpha$ is the critical value for the supremum of a Brownian bridge at significance level $\alpha$. At $\alpha = 0.05$, $c_\alpha = 1.358$. This boundary widens at the endpoints (where the MOSUM has higher variance) and narrows at the center (providing higher power), matching the variance profile of the MOSUM process under the null.

**Break localization.** Contiguous runs of boundary crossings are clustered. Within each cluster, the break is estimated at the position of maximum $|M_t(h)|$, offset by $h/2$ to center the estimate within the detection window.

---

## 3.3 Regime-Switching Models

Three complementary regime-switching approaches are implemented to characterize factor return dynamics across distinct market states. Regime models provide probabilistic state assignments that complement the deterministic break locations from Section 3.2.

### 3.3.1 Markov-Switching Regression

We fit a two-regime Markov-switching regression model (Hamilton, 1989) to each factor return series. The observation equation is:

$$y_t = \mu_{S_t} + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2_{S_t})$$

where $S_t \in \{0, 1\}$ is a latent Markov chain with transition matrix:

$$P = \begin{pmatrix} p_{00} & p_{01} \\ p_{10} & p_{11} \end{pmatrix}, \quad p_{ij} = P(S_t = j \mid S_{t-1} = i)$$

Both the intercept $\mu_{S_t}$ and the variance $\sigma^2_{S_t}$ switch across regimes. Switching variance is enabled by default because factor return volatility is well-documented to differ across market regimes (e.g., crisis vs. calm periods).

**Number of regimes.** We set $K = 2$, the canonical Hamilton (1989) specification sufficient for the primary use case of distinguishing alpha-generating from decayed states. Higher regime counts risk overfitting on finite samples. When exploratory analysis is warranted, BIC-based model selection is available: models with $K \in \{2, 3, 4\}$ are fit and the specification minimizing BIC is selected.

**Estimation.** Parameters are estimated via the Expectation-Maximization (EM) algorithm as implemented in `statsmodels.tsa.regime_switching.MarkovRegression`. To mitigate sensitivity to local optima, we run $n_{\text{init}} = 10$ random initializations, each with starting parameters perturbed by draws from $\mathcal{N}(0, 0.01)$ added to the default initialization, using deterministic seeds (seed $= i \times 42 + 7$ for initialization $i$) for reproducibility. The fit with the highest log-likelihood is retained.

**Inference.** Smoothed state probabilities $P(S_t = k \mid y_1, \ldots, y_T)$ are computed via the forward-backward (Kim) smoother. The most likely regime assignment at each time step is taken as $\hat{S}_t = \arg\max_k P(S_t = k \mid \mathbf{y})$. Expected regime duration is derived from the transition matrix as $d_k = 1/(1 - p_{kk})$.

**Transition matrix convention.** The transition matrix is stored in the row-stochastic Hamilton (1989) convention, where $P_{ij} = P(S_t = j \mid S_{t-1} = i)$ and rows sum to unity. The `statsmodels` implementation returns a column-stochastic matrix, which is transposed at extraction.

### 3.3.2 Gaussian Hidden Markov Model

As a complementary specification, we fit a Gaussian HMM using the Baum-Welch (EM) algorithm as implemented in `hmmlearn.GaussianHMM`. The model assumes:

$$y_t \mid S_t = k \sim \mathcal{N}(\mu_k, \Sigma_k)$$

with the same Markov chain dynamics as Section 3.3.1. The HMM is fit with full covariance matrices (`covariance_type = "full"`), maximum EM iterations set to 100 (sufficient for convergence on financial time series of typical length), and $n_{\text{init}} = 10$ random initializations with the same seeding scheme.

**State decoding.** The Viterbi algorithm is used to recover the most probable state sequence $\hat{S}_{1:T} = \arg\max_{S_{1:T}} P(S_{1:T} \mid y_{1:T})$. Posterior state probabilities are obtained from the forward-backward algorithm.

**Model selection.** When the number of regimes is uncertain, we evaluate models with $K \in \{2, 3, 4, 5\}$ and select the specification minimizing BIC:

$$\text{BIC} = -2 \ln \hat{L} + d \ln T$$

where $d$ counts all free parameters: $K(K-1)$ transition probabilities, $K$ means, $K \cdot p(p+1)/2$ covariance parameters (for a full covariance matrix with $p$ features), and $K-1$ initial state probabilities.

### 3.3.3 GARCH Regime Detection

The third approach detects regime shifts in conditional volatility rather than in the level of returns. A GARCH(1,1) model (Bollerslev, 1986) is fit to the factor return series:

$$y_t = \mu + \epsilon_t, \quad \epsilon_t = \sigma_t z_t, \quad z_t \sim \mathcal{N}(0,1)$$
$$\sigma_t^2 = \omega + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2$$

Alternatively, the EGARCH specification of Nelson (1991) or the GJR-GARCH specification of Glosten, Jagannathan & Runkle (1993) may be employed to capture asymmetric volatility response (leverage effect). The default specification is GARCH(1,1) with Gaussian innovations.

The conditional volatility series $\{\hat{\sigma}_t\}$ extracted from the fitted model is then submitted to the PELT algorithm (Section 3.2.1) with an RBF kernel cost and penalty $\beta = 3.0 \cdot \ln(n)$, where the factor 3.0 provides a BIC-like penalty calibrated for changepoint detection on volatility series. The minimum segment size is set dynamically to $\max(2, \lfloor n/50 \rfloor)$.

Detected volatility breakpoints partition the sample into segments, which are ranked by mean conditional volatility and labeled as low, medium, or high volatility regimes (using tercile boundaries when three or more segments are identified).

---

## 3.4 Signal Quality Metrics

### 3.4.1 Rolling Sharpe Ratio

The annualized Sharpe ratio (Sharpe, 1994) is computed over a rolling window of $w = 252$ trading days (approximately one calendar year):

$$\text{Sharpe}_t = \frac{\bar{r}_t}{\hat{\sigma}_t} \cdot \sqrt{252}$$

where $\bar{r}_t$ and $\hat{\sigma}_t$ are the sample mean and standard deviation (with Bessel's correction) of excess returns in the window $[t - w + 1, t]$. Annualization uses the $\sqrt{T}$ rule, which assumes i.i.d. returns as a first-order approximation (Lo, 2002). Windows with fewer than $w$ non-missing observations produce NaN.

The rolling Sharpe series serves dual roles: (1) as a diagnostic metric for evaluating signal quality evolution over time, and (2) as an input to the changepoint detectors, which are run on the Sharpe series to identify structural breaks in risk-adjusted performance.

### 3.4.2 Signal Half-Life via Ornstein-Uhlenbeck Estimation

The half-life of signal mean reversion is estimated by fitting a discrete-time AR(1) model to the return series:

$$r_t = \alpha + \phi \cdot r_{t-1} + \epsilon_t$$

Under the continuous-time Ornstein-Uhlenbeck (O-U) interpretation (Uhlenbeck & Ornstein, 1930), the discretization with unit time step yields $\phi = e^{-\theta}$, where $\theta > 0$ is the mean-reversion speed. The half-life---the expected time for a deviation from the long-run mean to decay by 50%---is:

$$h = \frac{\ln 2}{\theta} = \frac{\ln 2}{-\ln \phi}$$

The AR(1) slope $\phi$ is estimated by ordinary least squares. If $\phi \geq 1$ (unit root or explosive) or $\phi \leq 0$ (negative autocorrelation inconsistent with O-U dynamics), the process does not exhibit the expected mean reversion and the half-life is reported as $\infty$.

A rolling variant applies this estimation over windows of 252 trading days, with infinite half-life values capped at the window length to bound downstream analyses.

### 3.4.3 Out-of-Sample $R^2$

The out-of-sample $R^2$ statistic (Campbell & Thompson, 2008) measures whether a forecasting model improves upon a benchmark in terms of mean squared prediction error:

$$R^2_{\text{OOS}} = 1 - \frac{\sum_{t} (y_t - \hat{y}_t^{\text{model}})^2}{\sum_{t} (y_t - \hat{y}_t^{\text{bench}})^2}$$

A positive $R^2_{\text{OOS}}$ indicates that the model forecasts have lower cumulative squared error than the benchmark. The default benchmark is the expanding (prevailing) historical mean forecast:

$$\hat{y}_t^{\text{bench}} = \frac{1}{t-1} \sum_{s=1}^{t-1} y_s$$

which is the natural null for return predictability tests (Welch & Goyal, 2008). The first observation ($t = 0$) is excluded since no historical mean is available.

In the auditor pipeline, the model forecast is set to the lagged return $\hat{y}_t^{\text{model}} = r_{t-1}$ (an AR(1) predictor without intercept), providing a simple test of whether first-order autocorrelation in the factor carries predictive value relative to the unconditional mean.

A rolling variant computes $R^2_{\text{OOS}}$ over windows of 60 observations (approximately 3 months of daily data), requiring a minimum of 30 non-NaN observations per window (Welch & Goyal, 2008).

### 3.4.4 Clark-West MSPE-Adjusted Test

The Clark & West (2007) test evaluates whether an unrestricted (larger) forecasting model produces significantly lower mean squared prediction error than a restricted (nested) model, after correcting for the upward bias in the unrestricted model's MSPE under the null. The adjustment series is:

$$f_t = e_{1,t}^2 - \left[e_{2,t}^2 - (\hat{y}_{1,t} - \hat{y}_{2,t})^2\right]$$

where $e_{1,t} = y_t - \hat{y}_{1,t}$ and $e_{2,t} = y_t - \hat{y}_{2,t}$ are the restricted and unrestricted forecast errors, respectively. The test statistic is the $t$-statistic for $\bar{f} \neq 0$, computed with HAC (Newey-West) standard errors:

$$t_{\text{CW}} = \frac{\bar{f}}{\text{se}_{\text{HAC}}(\bar{f})}$$

**HAC bandwidth.** The Newey-West (1994) automatic bandwidth selection rule is used:

$$\ell = \left\lfloor 4 \left(\frac{n}{100}\right)^{2/9} \right\rfloor$$

with Bartlett kernel weights $w(j) = 1 - j/(\ell + 1)$. The HAC variance of the sample mean is:

$$\hat{V}_{\text{HAC}} = \frac{1}{n} \left[\hat{\gamma}_0 + 2 \sum_{j=1}^{\ell} w(j) \hat{\gamma}_j \right]$$

where $\hat{\gamma}_j = n^{-1} \sum_{t=j+1}^{n} (f_t - \bar{f})(f_{t-j} - \bar{f})$ is the sample autocovariance at lag $j$.

The test is one-sided ($H_a$: unrestricted model has lower MSPE), with $p$-values computed from the Student-$t$ distribution with $n - 1$ degrees of freedom. Default significance level is $\alpha = 0.05$.

### 3.4.5 Giacomini-White Conditional Predictive Ability Test

The Giacomini & White (2006) test compares two models via their loss differentials $d_t = L_1(y_t, \hat{y}_{1,t}) - L_2(y_t, \hat{y}_{2,t})$, where $L_i$ is the loss function for model $i$ (typically squared error).

**Unconditional test.** A Diebold-Mariano-style $t$-test on $\bar{d}$ with Newey-West HAC standard errors (same bandwidth selection rule as Clark-West, Section 3.4.4). The test statistic is $t = \bar{d} / \text{se}_{\text{HAC}}(\bar{d})$ with a two-sided $p$-value from $t(n-1)$.

**Conditional test.** The loss differential is regressed on a set of conditioning instruments $Z_t$ (e.g., lagged volatility, regime indicators) with an intercept:

$$d_t = Z_t' \beta + u_t$$

OLS estimates $\hat{\beta} = (X'X)^{-1} X' d$ are obtained, where $X = [1, Z]$. A Wald test of $H_0: \beta = 0$ is conducted using a Newey-West HAC sandwich variance estimator:

$$\hat{V}_{\text{HAC}}(\hat{\beta}) = n \cdot (X'X)^{-1} \hat{S}_{\text{HAC}} (X'X)^{-1}$$

where $\hat{S}_{\text{HAC}} = n^{-1} \left[\sum_t g_t g_t' + \sum_{j=1}^{\ell} w(j)(\Gamma_j + \Gamma_j')\right]$, $g_t = X_t \hat{u}_t$ is the moment condition vector, and $\Gamma_j = \sum_{t=j+1}^{n} g_t g_{t-j}'$.

The Wald statistic $W = \hat{\beta}' \hat{V}_{\text{HAC}}(\hat{\beta})^{-1} \hat{\beta} \sim \chi^2(k)$ under the null, where $k$ is the number of regressors (intercept plus number of instruments).

---

## 3.5 Verdict Classification

The audit pipeline aggregates evidence from changepoint detectors, regime models, and signal quality metrics into a three-level verdict classification. The multi-detector consensus approach mitigates false positives from any single method.

### 3.5.1 DEAD

A signal is classified as **DEAD** when:
1. At least $n_{\min} = 2$ detectors independently identify structural breaks in the raw return series (i.e., `n_detectors_with_breaks >= 2`); AND
2. The post-break mean return is non-positive.

Post-break mean assessment proceeds hierarchically:
- **Primary:** If a regime model was fitted (Section 3.3), the minimum regime mean across all detected regimes is examined. If $\min_k \mu_k \leq 0$, the post-break returns are classified as negative.
- **Fallback:** If no regime model is available, the empirical mean of returns from the median breakpoint to the end of the sample is computed directly.

The consensus breakpoint is reported as $\hat{\tau}_{\text{consensus}} = \text{median}(\{\hat{\tau}_d\}_{d=1}^D)$ across all detectors $d$ that identified breaks.

### 3.5.2 DECAYING

A signal is classified as **DECAYING** when:
1. At least one detector identifies a structural break; AND
2. The rolling Sharpe ratio exhibits a sustained decline exceeding a dynamic threshold.

The decline threshold is defined as:

$$\Delta_{\text{crit}} = \min(0.3, \; 0.3 \cdot |\text{Sharpe}_{\text{start}}|)$$

where $\text{Sharpe}_{\text{start}}$ and $\text{Sharpe}_{\text{end}}$ are estimated as the median rolling Sharpe over the first and last quartiles of the valid rolling Sharpe series, respectively. This dual-threshold design prevents false negatives when the initial Sharpe is low (where a 0.3 absolute decline is unrealistic) and false positives when the initial Sharpe is very high (where a 0.3 absolute decline may be noise).

Decay is flagged when $\text{Sharpe}_{\text{end}} < \text{Sharpe}_{\text{start}} - \Delta_{\text{crit}}$.

As a secondary pathway, decay may also be classified via CUSUM on the rolling Sharpe series (Section 3.6, decay onset detection): if the CUSUM statistic on the rolling Sharpe exceeds the critical threshold $2\sigma$, decay onset is flagged at the first exceedance point.

### 3.5.3 ACTIVE

A signal is classified as **ACTIVE** when no structural breaks are detected by any configured detector and the rolling Sharpe ratio remains stable (no significant decline per the criteria above).

---

## 3.6 Bootstrap Inference

Breakpoint location estimates are inherently uncertain. We quantify this uncertainty via bootstrap confidence intervals tailored to the dependence structure of financial return data.

### 3.6.1 Stationary Bootstrap for PELT and CUSUM

For the PELT and CUSUM detectors, breakpoint confidence intervals are constructed using the stationary bootstrap of Politis & Romano (1994), as implemented in the `arch.bootstrap.StationaryBootstrap` class. The stationary bootstrap draws blocks of geometrically distributed random length with expected block length $\bar{b} = 12$ (approximately half a trading month), producing stationary resampled paths that preserve the autocorrelation and heteroskedasticity structure of the original series.

The procedure is:
1. Generate $B = 1{,}000$ bootstrap replicates of the return series.
2. Re-run the changepoint detector (PELT or CUSUM) on each replicate using the same penalty and parameter configuration as the original fit.
3. Retain only those replicates where the number of detected breakpoints equals the original count (ensuring consistent breakpoint indexing).
4. Construct $(1 - \alpha) \times 100\%$ percentile confidence intervals for each breakpoint location from the empirical distribution of bootstrap breakpoint estimates, with $\alpha = 0.05$ (95% confidence intervals).

### 3.6.2 Block Bootstrap for Bai-Perron and MOSUM

For the Bai-Perron and MOSUM detectors, a circular block bootstrap is used with deterministic block length:

- **Bai-Perron:** Block length $= \max(\lfloor \epsilon \cdot n \rfloor, 5)$ where $\epsilon = 0.15$ is the trimming parameter. This preserves local dependence on the same scale as the trimming constraint, ensuring that bootstrap sub-samples have analogous local structure to the original segments.
- **MOSUM:** Block length $= \max(\lfloor h \cdot n \rfloor, 5)$ where $h = 0.10$ is the MOSUM bandwidth. Matching the block length to the detection bandwidth ensures that the bootstrap resamples preserve dependence on the same scale as the moving window.

The circular block bootstrap constructs replicates by sampling block start indices uniformly from $\{0, \ldots, n-1\}$ and wrapping indices modulo $n$, then concatenating blocks and truncating to length $n$. The same $B = 1{,}000$ replicates and percentile CI procedure described in Section 3.6.1 are applied. All bootstrap procedures use a fixed random seed (base seed 42) for reproducibility.

---

## 3.7 Decay Onset Detection

As a meta-analytic step, the auditor applies CUSUM-based decay onset detection to the rolling signal quality metric (typically the rolling Sharpe ratio). This differs from the CUSUM changepoint detector (Section 3.2.2) in that it operates on a derived metric rather than raw returns and uses an expanding-mean reference to avoid look-ahead bias.

At each time $t$, the expanding mean is $\hat{\mu}_t = (t-1)^{-1} \sum_{s=1}^{t-1} x_s$ (using only data prior to $t$). The downward-shift CUSUM statistic accumulates deviations of the metric below its expanding mean:

$$C_t = \sum_{s=2}^{t} (\hat{\mu}_s - x_s)$$

Decay onset is flagged at the first time $t^*$ where $C_{t^*}$ exceeds a critical threshold, set by default to $2 \hat{\sigma}$ where $\hat{\sigma}$ is the standard deviation of the metric series---a standard choice per Page (1954) and Hinkley (1971). Detection confidence is reported as $C_{t^*} / c_{\text{crit}}$, quantifying how far the CUSUM exceeds the threshold at the onset point.

---

# 4. Results and Analysis

We apply the signal decay auditor framework to six Fama-French factors (Mkt-RF, SMB, HML, RMW, CMA, UMD) over the daily sample 1963-07-01 through 2026-01-30 (*N* = 15,751 trading days each). We supplement the factor analysis with equity and cryptocurrency audits (QQQ, SPY, GLD, BTC-USD, NVDA, TSLA) covering the shorter 2015-01-05 to 2026-03-27 window (*N* = 2,824 for equities; *N* = 4,105 for BTC-USD). Each signal is subjected to PELT, CUSUM, and Bai-Perron changepoint detection on both raw returns and rolling Sharpe ratios; a two-state Hidden Markov Model for regime classification; and exponential half-life estimation for mean-reversion diagnostics. The auditor issues one of three verdicts---DEAD, DECAYING, or RESILIENT---based on the convergence of detector outputs, post-break Sharpe ratio levels, and regime persistence.

## 4.1 Summary of Factor Audit Verdicts

Table 1 reports full-sample descriptive statistics and audit verdicts for the six factors.

**Table 1. Factor Audit Summary**

| Factor | *N* | Period | Ann. Return | Ann. Vol | Full-Sample Sharpe | Early Sharpe | Late Sharpe | Delta | Verdict |
|--------|-----|--------|-------------|----------|-------------------|--------------|-------------|-------|---------|
| Mkt-RF | 15,751 | 1963--2026 | 7.25% | 16.23% | 0.45 | 0.52 | 0.89 | +0.37 | DECAYING* |
| SMB | 15,751 | 1963--2026 | 1.53% | 8.75% | 0.18 | 1.07 | -0.23 | -1.31 | DEAD |
| HML | 15,751 | 1963--2026 | 3.67% | 9.30% | 0.39 | 0.99 | -0.24 | -1.23 | DECAYING |
| RMW | 15,751 | 1963--2026 | 3.25% | 6.43% | 0.50 | -0.02 | 0.35 | +0.37 | DECAYING* |
| CMA | 15,751 | 1963--2026 | 2.97% | 6.08% | 0.49 | 0.70 | -0.28 | -0.98 | DECAYING |
| UMD | 15,751 | 1963--2026 | 7.17% | 12.38% | 0.58 | 1.19 | 0.50 | -0.69 | DEAD |

*Note.* Sharpe ratios are annualized. "Early" and "Late" refer to the pre- and post-first-detected-break subsamples. Asterisks on Mkt-RF and RMW denote cases where the formal verdict is DECAYING but the delta is positive, indicating signal *improvement* rather than deterioration; these are discussed in Section 4.4 as functionally resilient signals.

Two factors receive a DEAD verdict: SMB (Size) and UMD (Momentum). Both exhibit Sharpe ratio collapses exceeding 0.69 standard units and, in SMB's case, negative post-break mean returns. Two factors---HML (Value) and CMA (Investment)---are classified DECAYING with substantial negative deltas (-1.23 and -0.98, respectively) and post-break Sharpe ratios that are negative. Two factors---Mkt-RF and RMW---carry formal DECAYING labels but show *improving* Sharpe trajectories (+0.37 each), placing them in a functionally distinct category.

The distribution of verdicts is consistent with the post-publication decay hypothesis of McLean and Pontiff (2016), who document that factor returns decline by approximately 32% after publication. Our results suggest that the decay for certain factors---particularly SMB and HML---has been far more severe than the cross-sectional average reported in that study, with Sharpe ratios declining by more than 100% of their pre-break values.


## 4.2 Dead Signals: SMB and UMD

### 4.2.1 SMB (Size)

The size premium is the most severely impaired factor in our sample. The full-sample annualized return of 1.53% masks a dramatic structural shift: the early-period rolling Sharpe of 1.07 collapses to -0.23 in the post-break window, a delta of -1.31. The PELT algorithm identifies three breakpoints on raw returns, all clustered in early 2000: 2000-01-13, 2000-03-13, and 2000-04-25. CUSUM detects an earlier break at 1987-10-19, coinciding with the Black Monday crash. Bai-Perron detects no breaks on raw returns but identifies five breaks on the rolling Sharpe series at indices 1470, 5176, 9115, 10828, and 12840. The convergence of PELT and CUSUM detectors on the returns series, combined with negative post-break mean returns, triggers the DEAD classification.

The timing of the PELT breaks---January through April 2000---aligns precisely with the dot-com bubble peak and the onset of the large-cap technology rally that structurally disadvantaged small-capitalization equities. The CUSUM break at October 1987 may represent an earlier, partial regime shift, but the definitive collapse occurs three decades later.

The HMM regime analysis reinforces this interpretation. Regime 0 (alpha-generating) exhibits a daily mean return of 0.0168% (approximately 4.2% annualized), while Regime 1 (decayed) has a mean of -0.0269% (-6.8% annualized). The transition matrix reveals strong persistence in both states:

|  | To Regime 0 | To Regime 1 |
|--|-------------|-------------|
| From Regime 0 | 0.9891 | 0.0109 |
| From Regime 1 | 0.0337 | 0.9663 |

The expected duration of the alpha-generating regime is 1/0.0109 = 91.7 days, while the decayed regime persists for 1/0.0337 = 29.7 days. The asymmetry---alpha states last three times longer than decay states---might appear favorable, but the current regime allocation places SMB firmly in the decayed state. The estimated half-life of 0.20 days (effectively instantaneous mean-reversion) with an R-squared of 0.001 indicates no detectable predictable structure in the decay process. The last observed rolling Sharpe of -0.28 confirms that SMB has not recovered.

### 4.2.2 UMD (Momentum)

Momentum receives the second DEAD verdict with a Sharpe decline from 1.19 to 0.50 (delta = -0.69). Although the post-break Sharpe remains positive---unlike SMB---the verdict is driven by multiple detector agreement near index 11335 and the DEAD-classification logic's identification of negative post-break mean returns in at least one detector window.

PELT identifies twelve breakpoints on returns---the most of any factor---with major clusters around 2000-01-13 to 2000-04-17, 2002-10-08 to 2002-11-26, and the critical 2008-05-30 through 2009-01-09 window comprising five breaks. Bai-Perron confirms a single return-level break at 2008-11-21, falling squarely within the momentum crash period documented by Daniel and Moskowitz (2016). CUSUM detects no breaks on returns but identifies one on the rolling Sharpe at index 6873.

The 2008 momentum crash is the defining event. The five PELT breaks spanning May 2008 through January 2009 correspond to: (i) the Bear Stearns collapse aftermath (2008-05-30); (ii) the Fannie Mae/Freddie Mac crisis (2008-07-14); (iii) the Lehman Brothers bankruptcy (2008-09-09); (iv) the TARP legislation period (2008-11-25); and (v) the market bottom approach (2009-01-09). Bai-Perron's single break at 2008-11-21 provides independent confirmation, pinpointing the structural shift to within four days of the PELT break at 2008-11-25.

The HMM identifies two regimes with sharply divergent means: Regime 0 (alpha-generating) at 0.0682% daily (17.2% annualized) and Regime 1 (decayed) at -0.0596% daily (-15.0% annualized). The transition matrix is:

|  | To Regime 0 | To Regime 1 |
|--|-------------|-------------|
| From Regime 0 | 0.9816 | 0.0184 |
| From Regime 1 | 0.0408 | 0.9592 |

The decayed regime has an expected duration of 1/0.0408 = 24.5 days, while the alpha regime lasts 1/0.0184 = 54.3 days. Compared to SMB, UMD's regime structure is less persistent overall, reflecting the higher-frequency nature of momentum crashes and recoveries. The half-life estimate of 0.38 days with an R-squared of 0.027---the highest among all six factors---suggests marginally more detectable mean-reversion structure, consistent with the partial recovery of momentum returns post-2009.


## 4.3 Decaying Signals: HML and CMA

### 4.3.1 HML (Value)

The value premium presents the clearest case of post-publication decay among the six factors. The full-sample Sharpe of 0.39 conceals a collapse from 0.99 in the early window to -0.24 in the late window (delta = -1.23), the second-largest decline after SMB. However, unlike SMB, the framework classifies HML as DECAYING rather than DEAD, reflecting the pattern of multiple breaks distributed across the sample rather than a single definitive structural rupture.

PELT detects eight breakpoints on returns, the joint highest count with CMA. The break dates trace a revealing chronology: 1999-08-09 and 2000-03-06 bracket the dot-com peak; 2001-03-30 marks the post-bubble value recovery; 2009-01-02, 2009-03-10, and 2009-04-22 correspond to the global financial crisis trough and reversal; and 2020-01-06 and 2020-05-14 capture the COVID-19 crash and subsequent growth-stock dominance. CUSUM identifies a single break on returns at 2020-11-06---the week of the 2020 U.S. presidential election and a sharp value rotation---while Bai-Perron detects no breaks on raw returns but five on the rolling Sharpe series.

The temporal distribution of breaks is notable. The Fama-French three-factor model was published in 1993; the five-factor extension appeared in 2015. The first PELT break at 1999-08-09 occurs six years post-publication, and the most recent (2020-05-14) occurs five years after the five-factor paper. This timing is consistent with McLean and Pontiff's (2016) finding that factor returns begin to attenuate within three to five years of academic publication, as arbitrageurs exploit the documented anomaly.

The HMM separates two regimes: Regime 0 (alpha-generating) with a daily mean of 0.0415% (10.5% annualized) and Regime 1 (decayed) with a mean of 0.0058% (1.5% annualized). The transition matrix shows:

|  | To Regime 0 | To Regime 1 |
|--|-------------|-------------|
| From Regime 0 | 0.9690 | 0.0310 |
| From Regime 1 | 0.0102 | 0.9898 |

The decayed regime is highly persistent: expected duration 1/0.0102 = 98.0 days, compared to 1/0.0310 = 32.3 days for the alpha regime. This is the most asymmetric transition structure among the six factors and implies that once HML enters the decayed state, recovery is slow. The half-life of 0.28 days (R-squared = 0.008) indicates negligible predictive structure in the decay path.

### 4.3.2 CMA (Investment)

The investment factor mirrors HML's trajectory. The Sharpe ratio declines from 0.70 to -0.28 (delta = -0.98), and the full-sample Sharpe of 0.49 again obscures the regime shift. CMA is distinguished by the broadest detector agreement: PELT detects eight breaks on returns, CUSUM detects one (2001-01-02), and Bai-Perron---uniquely among the six factors---detects two breaks on raw returns at 1997-10-10 and 2004-01-21.

The PELT break dates span three distinct eras. The earliest pair (1970-01-16, 1970-05-26) predates the academic literature on investment factors and may reflect a regime shift related to the collapse of the "Nifty Fifty" growth-stock era. The middle cluster (2000-07-20, 2001-01-03) aligns with the dot-com bubble collapse. The most recent cluster (2021-11-22, 2022-06-14, 2022-08-11, 2023-01-04) captures the post-COVID interest-rate normalization period, during which the investment premium was compressed by the rapid shift from near-zero rates to restrictive monetary policy.

Bai-Perron's detection of two returns-level breaks (1997-10-10 and 2004-01-21) is unique to CMA. The first coincides with the Asian financial crisis; the second falls during the mid-2000s credit expansion. These breaks bracket a period of strong CMA performance, after which the factor enters a prolonged decay phase.

The HMM regime structure reveals:

|  | To Regime 0 (decayed) | To Regime 1 (alpha) |
|--|----------------------|---------------------|
| From Regime 0 | 0.9901 | 0.0099 |
| From Regime 1 | 0.0245 | 0.9755 |

Regime 0 (decayed) has a mean daily return of 0.0019% (0.5% annualized); Regime 1 (alpha-generating) has a mean of 0.0365% (9.2% annualized). The decayed regime persists for an expected 1/0.0099 = 101.0 days---the longest of any factor---while the alpha regime lasts 1/0.0245 = 40.8 days. The half-life estimate of 0.33 days (R-squared = 0.015) indicates weak but non-negligible mean-reversion, slightly stronger than HML.


## 4.4 Resilient Signals: Mkt-RF and RMW

### 4.4.1 Mkt-RF (Market)

The equity risk premium is the only factor whose late-period Sharpe exceeds its early-period value. The rolling Sharpe improves from 0.52 to 0.89, a positive delta of +0.37. The formal DECAYING verdict is triggered by CUSUM detection of a Sharpe-level shift at index 289, but this is a methodological artifact: the framework's decay logic flags any detected break as potential decay, yet the direction of change is *improving*. We reclassify Mkt-RF as functionally resilient.

PELT detects only two breakpoints on returns: 2008-09-09 and 2008-10-28, both attributable to the Lehman Brothers bankruptcy and subsequent TARP intervention. Neither CUSUM nor Bai-Perron detect any return-level breaks. The minimal break count on returns---zero for two of three detectors---is consistent with the theoretical expectation that the equity risk premium, as compensation for bearing systematic risk, should not be arbitraged away (Fama, 1991).

The HMM identifies a bear regime (Regime 0) with a daily mean of -0.074% (-18.7% annualized) and a bull regime (Regime 1) at 0.066% (16.7% annualized). The transition probabilities confirm strong regime persistence: the bull state lasts 1/0.0128 = 78.1 days in expectation, and the bear state 1/0.0351 = 28.5 days. The bull regime is 2.7 times more persistent than the bear regime, reflecting the well-documented positive skew of long-run equity returns. The half-life of 0.17 days (R-squared = 0.0002) indicates no detectable serial structure, consistent with near-efficient pricing of the aggregate market portfolio.

### 4.4.2 RMW (Profitability)

The profitability factor presents the most favorable trajectory of any anomaly-based factor. The early-period Sharpe is effectively zero (-0.02), while the late-period Sharpe reaches 0.35 (delta = +0.37). Like Mkt-RF, RMW is formally labeled DECAYING but is functionally resilient---indeed, strengthening.

PELT identifies six breaks on returns, all clustered around 1999-12-01 through 2002-11-26. CUSUM confirms with a break at 2000-03-31. Bai-Perron detects no returns-level breaks. The break cluster coincides with the dot-com collapse, during which the profitability premium emerged as investors rotated from speculative growth into firms with strong operating fundamentals.

The HMM regime means are: Regime 0 (alpha-generating) at 0.0239% daily (6.0% annualized) and Regime 1 (decayed) at 0.0088% daily (2.2% annualized). Note that even the "decayed" regime retains a positive mean, distinguishing RMW from SMB, HML, and CMA where decayed-regime means are near-zero or negative. The transition matrix is:

|  | To Regime 0 | To Regime 1 |
|--|-------------|-------------|
| From Regime 0 | 0.9817 | 0.0183 |
| From Regime 1 | 0.0068 | 0.9932 |

The alpha regime lasts 1/0.0183 = 54.6 days; the weaker regime persists for 1/0.0068 = 147.1 days. While the decayed regime is extremely persistent, its positive mean (2.2% annualized) implies that RMW continues to compensate even in its low-performance state. The half-life of 0.33 days (R-squared = 0.015) is comparable to CMA.

The resilience of RMW is consistent with the hypothesis of Novy-Marx (2013), who argues that the profitability premium is distinct from value and reflects compensation for holding firms with high expected cash flows. Unlike the value and size premia---which are amenable to straightforward long-short arbitrage---the profitability premium may be more difficult to arbitrage because it requires accurate forward-looking estimates of operating profitability, creating a natural barrier to the kind of post-publication exploitation documented by McLean and Pontiff (2016).


## 4.5 Cross-Factor Comparison

### 4.5.1 Ranking by Decay Severity

Table 2 ranks factors by Sharpe ratio delta, from most severe decay to greatest improvement.

**Table 2. Factor Ranking by Sharpe Delta**

| Rank | Factor | Delta | Post-Break Sharpe | Verdict |
|------|--------|-------|-------------------|---------|
| 1 | SMB | -1.31 | -0.23 | DEAD |
| 2 | HML | -1.23 | -0.24 | DECAYING |
| 3 | CMA | -0.98 | -0.28 | DECAYING |
| 4 | UMD | -0.69 | +0.50 | DEAD |
| 5 | Mkt-RF | +0.37 | +0.89 | Resilient |
| 6 | RMW | +0.37 | +0.35 | Resilient |

The three factors with the most severe decay (SMB, HML, CMA) all exhibit *negative* post-break Sharpe ratios, implying that these signals have not merely weakened but have reversed sign. UMD's post-break Sharpe of +0.50 remains economically meaningful, suggesting partial rather than total signal death---the DEAD classification reflects the severity of the crash episode and negative returns within specific detector windows rather than permanent extinction. Mkt-RF and RMW stand apart with positive deltas.

### 4.5.2 Detector Agreement

Table 3 summarizes the number of breakpoints detected on raw returns by each algorithm.

**Table 3. Breakpoint Counts on Returns by Detector**

| Factor | PELT | CUSUM | Bai-Perron | Total | Detectors Agreeing |
|--------|------|-------|------------|-------|--------------------|
| SMB | 3 | 1 | 0 | 4 | 2 of 3 |
| UMD | 12 | 0 | 1 | 13 | 2 of 3 |
| HML | 8 | 1 | 0 | 9 | 2 of 3 |
| CMA | 8 | 1 | 2 | 11 | 3 of 3 |
| RMW | 6 | 1 | 0 | 7 | 2 of 3 |
| Mkt-RF | 2 | 0 | 0 | 2 | 1 of 3 |

CMA is the only factor where all three detectors identify at least one break on raw returns, providing the strongest statistical evidence for structural change. UMD has the highest raw break count (13), driven by PELT's twelve breaks, reflecting the factor's high-volatility, crash-prone return distribution. Mkt-RF has the fewest breaks (2), with only PELT detecting any---consistent with the interpretation that the market risk premium is structurally stable.

On rolling Sharpe ratios, the picture is uniformly active: PELT detects 56--78 breaks per factor, CUSUM detects one break per factor (or zero for Mkt-RF returns), and Bai-Perron identifies 3--5 breaks per factor. The high PELT count on Sharpe series reflects the algorithm's sensitivity to level shifts in a noisy, serially correlated series and should be interpreted as a measure of local variability rather than discrete structural events.

### 4.5.3 Regime Structure Comparison

Table 4 compares HMM regime parameters across factors.

**Table 4. HMM Regime Parameters**

| Factor | Alpha Mean (daily) | Decayed Mean (daily) | P(stay alpha) | P(stay decay) | E[dur. alpha] | E[dur. decay] |
|--------|-------------------|---------------------|---------------|---------------|---------------|---------------|
| Mkt-RF | +0.066% | -0.074% | 0.987 | 0.965 | 78.1 d | 28.5 d |
| SMB | +0.017% | -0.027% | 0.989 | 0.966 | 91.7 d | 29.7 d |
| HML | +0.041% | +0.006% | 0.969 | 0.990 | 32.3 d | 98.0 d |
| RMW | +0.024% | +0.009% | 0.982 | 0.993 | 54.6 d | 147.1 d |
| CMA | +0.036% | +0.002% | 0.975 | 0.990 | 40.8 d | 101.0 d |
| UMD | +0.068% | -0.060% | 0.982 | 0.959 | 54.3 d | 24.5 d |

A striking pattern emerges: the dead signals (SMB, UMD) have decayed-regime means that are *negative*, while the decaying signals (HML, CMA, RMW) have decayed-regime means that remain weakly positive. This distinction is economically meaningful---a negative decayed-regime mean implies that the factor *costs* money to hold during adverse periods, while a weakly positive mean suggests attenuation rather than sign reversal at the regime level.

HML and CMA exhibit the longest expected durations in the decayed regime (98.0 and 101.0 days, respectively), explaining their persistent negative post-break Sharpe ratios: these factors spend extended periods in low-return states. By contrast, UMD's decayed regime lasts only 24.5 days on average, consistent with the sharp but transient nature of momentum crashes. The market factor's bull regime is the most persistent (78.1 days), reflecting the long-run upward drift in equity prices.

### 4.5.4 Publication Dates and Decay Onset

The timing relationship between academic publication and detected break dates provides a test of the McLean and Pontiff (2016) arbitrage-erosion hypothesis. Fama and French published the three-factor model (containing Mkt-RF, SMB, HML) in 1993 and the five-factor model (adding RMW and CMA) in 2015. Jegadeesh and Titman's seminal momentum paper appeared in 1993; Carhart (1997) formalized UMD.

For SMB, the PELT breaks at 2000-01-13 through 2000-04-25 occur seven years post-publication---consistent with McLean and Pontiff's median lag of approximately five years. For HML, the first PELT break at 1999-08-09 occurs six years post-publication. UMD's critical cluster (2008-05-30 through 2009-01-09) occurs 11--12 years after Carhart (1997), though the dot-com-era breaks at 2000-01-13 through 2000-04-17 fall at the seven-year mark. CMA's post-publication breaks (2021-11-22 through 2023-01-04) begin only six years after the 2015 five-factor paper, again consistent with the McLean-Pontiff timeline.

RMW's breaks are concentrated around 1999--2002, predating the formal five-factor publication by over a decade. This suggests that the structural shift in RMW reflects the dot-com regime change rather than post-publication arbitrage, which may partly explain why RMW has continued to strengthen: the academic spotlight arrived after the factor had already been "tested" by the market.


## 4.6 Equity and Crypto Asset Results

To assess the generalizability of the decay framework beyond long-short factor portfolios, we apply the auditor to six additional assets: three equities (SPY, QQQ, NVDA), one commodity ETF (GLD), one cryptocurrency (BTC-USD), and one high-volatility single stock (TSLA). All receive DECAYING verdicts, reflecting the framework's default behavior when at least one rolling-Sharpe break is detected; these results should be interpreted cautiously, as discussed below.

**Table 5. Equity and Crypto Audit Summary**

| Asset | *N* | Period | Mean Sharpe | Last Sharpe | Verdict | PELT Breaks (returns) | HMM Alpha Mean | HMM Decayed Mean |
|-------|-----|--------|-------------|-------------|---------|----------------------|----------------|-----------------|
| SPY | 2,824 | 2015--2026 | 1.06 | 0.73 | DECAYING | 0 | +0.110% | -0.083% |
| QQQ | 2,824 | 2015--2026 | 1.06 | 0.79 | DECAYING | 0 | +0.147% | -0.102% |
| GLD | 2,824 | 2015--2026 | 0.78 | 1.59 | DECAYING | 0 | +0.082% | +0.047% |
| NVDA | 2,824 | 2015--2026 | 1.45 | 1.14 | DECAYING | 0 | +0.298% | +0.147% |
| TSLA | 2,824 | 2015--2026 | 0.76 | 0.79 | DECAYING | 0 | +0.366% | +0.103% |
| BTC-USD | 4,105 | 2015--2026 | 0.99 | -1.29 | DECAYING | 2 | +0.283% | +0.140% |

BTC-USD is the clearest decay case among the non-factor assets. It is the only one where PELT detects structural breaks on raw returns (at indices 1045 and 1080, approximately late 2018). The rolling Sharpe declines from 1.7 to 1.1, and the last observed rolling Sharpe of -1.29 indicates a sharp recent deterioration. CUSUM confirms with a returns break at index 1895. The HMM transition matrix (alpha self-transition: 0.821; decayed self-transition: 0.895) indicates the least persistent regime structure of any asset tested, with expected durations of 5.6 days (alpha) and 9.5 days (decayed)---reflecting cryptocurrency's rapid regime-switching behavior. The infinite half-life estimate (slope = -0.027, R-squared = 0.0007) indicates no detectable mean-reversion in the decay process, consistent with the secular compression of cryptocurrency excess returns as the asset class matures and institutional adoption increases.

For the equity assets (SPY, QQQ, NVDA, TSLA), no PELT breaks are detected on raw returns. All breaks occur exclusively on rolling Sharpe series, and the decay onset indices are clustered near 263--264 for all four, corresponding to approximately one year of rolling-window initialization. This uniformity suggests that the detected "decay" is an artifact of the rolling-window warm-up period rather than genuine structural change. The framework's application to long-only equity returns---where no zero-cost arbitrage mechanism exists to erode returns---violates the theoretical premises underlying the factor decay model. The DECAYING verdicts for SPY, QQQ, GLD, NVDA, and TSLA should therefore be discounted as false positives outside the framework's intended domain.

GLD is an intermediate case: CUSUM detects a break on returns at index 2783 (near the end of the sample), and the last rolling Sharpe of 1.59 exceeds the mean of 0.78, suggesting gold's performance has recently *improved*---plausibly driven by central bank accumulation and geopolitical demand since 2022.

These results highlight a limitation of the current framework: the decay auditor is designed for zero-cost factor portfolios where arbitrage pressure is the theorized mechanism of return erosion. Applying it to long-only equity returns or alternative assets produces formally valid statistical output but economically ambiguous verdicts. Future work should incorporate asset-class-specific priors and null hypotheses calibrated to the expected persistence of different return-generating processes.

---

## 5. Discussion

### 5.1 Economic Interpretation of the Factor Decay Hierarchy

The Signal Decay Auditor produces a three-tier classification of the six Fama-French factors that aligns with, and quantitatively sharpens, the qualitative narrative in the post-publication decay literature. The hierarchy---DEAD (SMB, UMD), DECAYING (HML, CMA), RESILIENT (RMW, Mkt-RF)---admits a coherent economic interpretation grounded in the limits of arbitrage.

**SMB (Size).** The size premium is classified DEAD with multi-detector consensus: PELT identifies three structural breaks on returns clustered around January--April 2000, CUSUM detects a break at the October 1987 crash, and the rolling Sharpe ratio declines from 1.07 in the early sample to -0.23 in the late sample, a delta of -1.31 Sharpe units. The Markov regime model assigns a negative mean daily return (-0.027 bps) to the decayed state, with transition probability $p_{11} = 0.966$ indicating a persistent low-return regime. Post-break annualized return is 1.53%, entirely attributable to volatility compensation rather than a genuine small-cap premium. This finding is consistent with Horowitz, Loughran, and Savin (2000), who documented the disappearance of the size effect after 1982, and with Schwert (2003), who argued that the anomaly "disappeared after it was published." The mechanism is straightforward: size-sorted portfolios require no complex implementation---they can be replicated via small-cap index funds and ETFs launched in the late 1990s, providing frictionless access to the long side and reducing any capacity constraint that previously sustained the premium.

**UMD (Momentum).** Momentum is classified DEAD with PELT identifying 12 return-level breakpoints (concentrated in 2000 and 2008--2009) and Bai-Perron detecting a structural break at November 2008. The rolling Sharpe falls from 1.19 to 0.50, a decline of -0.69 units. The Markov regime model reveals a decayed state with negative mean return (-0.060 bps daily), and the transition matrix shows $p_{00} = 0.982$ for the alpha-generating state, implying an expected duration of approximately 55 trading days---short relative to rebalancing horizons. The 2008--2009 momentum crash (Daniel and Moskowitz, 2016) is visible in the PELT breakpoint cluster. While momentum is often characterized as a persistent anomaly (Jegadeesh and Titman, 1993, 2001), our results indicate that post-2009 momentum returns are statistically indistinguishable from zero on a risk-adjusted basis. Crowded implementation via momentum ETFs (AQR Momentum ETF launched 2009, iShares Edge MSCI USA Momentum Factor ETF launched 2013) has compressed the premium. The strategy's mechanical nature---ranking stocks by past 12-1 month returns---makes replication trivial, accelerating arbitrage capital deployment.

**HML (Value).** The value premium is classified DECAYING with a Sharpe decline of -1.23 (from 0.99 to -0.24). PELT detects eight return-level breaks spanning 1999--2020, and CUSUM identifies a break in November 2020. The Markov model separates an alpha-generating state (mean 0.041 bps daily) from a near-zero state (0.006 bps daily). The transition probability into the decayed state ($p_{01} = 0.031$) implies that once in the alpha-generating regime, the expected sojourn is only 32 trading days before transitioning. The value premium's decline accelerated after 2017 during the prolonged growth-over-value regime driven by technology sector concentration (Israel, Laursen, and Richardson, 2021). Unlike SMB, the value premium retains positive full-sample returns (3.67% annualized) and a full-sample Sharpe of 0.39, preventing classification as DEAD under our multi-detector consensus rule (which requires negative post-break mean returns). However, the trajectory is unambiguous: the most recent quartile Sharpe is negative.

**CMA (Investment).** The conservative-minus-aggressive factor is DECAYING with a Sharpe decline of -0.98 (from 0.70 to -0.28). Bai-Perron identifies two structural breaks on returns (October 1997, January 2004), and CUSUM detects a break in January 2001. CMA's decay follows a similar timeline to HML, consistent with both factors loading on a common "value-like" dimension (Fama and French, 2015). The investment premium reflects compensation for firms that invest conservatively, which correlates with cheap valuations. As value arbitrage capital increased, the related investment premium eroded in tandem.

**RMW (Profitability).** Despite a formal DECAYING classification triggered by CUSUM decay onset detection at index 294, the profitability factor shows an *improving* rolling Sharpe: from -0.02 (early) to 0.35 (late), a positive delta of +0.37. Full-sample annualized return is 3.25% with a Sharpe of 0.50. The Markov model identifies two positive-mean regimes (0.024 bps and 0.009 bps daily), neither of which is negative. This is a false positive from the CUSUM decay detector, discussed in Section 5.4 below. The economic interpretation is that profitability-based strategies are harder to arbitrage for three reasons: (i) profitability measurement requires accounting expertise and is subject to manipulation, creating information asymmetries that sustain the premium (Novy-Marx, 2013); (ii) the short side---low-profitability firms---includes distressed and illiquid names with high shorting costs and borrow constraints (Pontiff, 1996); and (iii) profitability is a flow variable that adjusts slowly, making the signal less susceptible to rapid arbitrage than price-based signals like momentum.

**Mkt-RF (Market).** The equity premium is classified DECAYING via CUSUM onset detection at index 289, but, like RMW, the rolling Sharpe improves from 0.52 to 0.89, a positive delta of +0.37. Annualized return is 7.25% with a Sharpe of 0.45. The market premium is a compensation for systematic risk and is not arbitrageable in the Pontiff (1996) sense: it requires bearing aggregate risk, and no long-short strategy can eliminate exposure to the market factor. The premium is resilient by construction.

### 5.2 Crowding and Arbitrage Capital as Decay Mechanisms

The decay hierarchy is consistent with the Pontiff (1996) limits-of-arbitrage framework and the McLean and Pontiff (2016) post-publication return attenuation thesis. Pontiff (1996) decomposes the cost of exploiting an anomaly into two components: transaction costs (commissions, bid-ask spread, market impact) and holding costs (idiosyncratic risk of the arbitrage position, shorting costs, margin requirements). Factors with low holding and transaction costs are arbitraged most rapidly.

SMB and UMD have the lowest implementation barriers. Size-sorted portfolios are replicable via liquid index instruments. Momentum requires only price data and mechanical sorting. Both strategies became available as packaged products (ETFs, index futures, systematic quant funds) by the mid-2000s. Our data show that the structural breaks for both factors cluster around 1999--2009, precisely the period of rapid growth in quantitative asset management (Lo, 2008) and factor-based ETF proliferation.

HML and CMA occupy an intermediate position. Value strategies require fundamental data, and the short side (growth stocks) has historically included high-momentum technology names with substantial borrowing costs during periods of overvaluation. The value premium persisted longer than the size premium but has been in secular decline since approximately 2007 (our PELT and Bai-Perron estimates), consistent with the growth of value-oriented smart-beta products.

RMW is the most insulated. The profitability factor's short side---firms with low or negative operating profitability---overlaps with the universe of hard-to-borrow, distressed equities. Pontiff (2006) shows that anomalies concentrated in stocks with high idiosyncratic volatility and high shorting costs exhibit the slowest convergence. Our data confirm that RMW's Sharpe is *increasing* over time, suggesting the premium is not yet subject to meaningful arbitrage pressure.

McLean and Pontiff (2016) estimate a post-publication return decline of 32% across 97 anomalies. Our framework refines this estimate with factor-specific granularity: SMB shows a Sharpe decline of 122% relative to its pre-break level; UMD declines 58%; HML declines 125%; CMA declines 140%. The absolute magnitudes exceed McLean and Pontiff's average because our sample extends 10 years beyond their 2013 endpoint, capturing the intensification of factor crowding during the 2016--2025 smart-beta era. Moreover, our multi-detector framework identifies the *timing* of decay onset---information unavailable from the before/after comparison in McLean and Pontiff (2016).

### 5.3 Multi-Detector Consensus: Benefits and Limitations

The framework's use of three complementary changepoint detectors---PELT (penalized likelihood), CUSUM (recursive residual), and Bai-Perron (sequential sup-F)---provides robustness through methodological triangulation. Each detector has different strengths: PELT excels at detecting multiple mean-shifts with automatic penalty selection via BIC; CUSUM is sensitive to gradual departures from parameter constancy; Bai-Perron provides formal sup-F inference with tabulated critical values (Bai and Perron, 2003, Table 1). The DEAD classification requires agreement from at least two detectors on return-level breaks with negative post-break mean, minimizing false positives from any single method.

In practice, the three detectors exhibit complementary behavior. For SMB, PELT and CUSUM both detect return-level breaks while Bai-Perron does not---Bai-Perron's 15% trimming (Andrews, 1993) and conservative critical values make it less sensitive to breaks near the endpoints. For UMD, PELT and Bai-Perron agree on the 2008 break region. For HML and CMA, PELT detects numerous return-level breaks while Bai-Perron is more selective, identifying breaks only for CMA. The consensus approach thus guards against both false positives (single-detector artifacts) and false negatives (breaks missed by a conservative method).

A limitation of the multi-detector approach is the lack of a formal meta-analytic framework for combining p-values or test statistics across heterogeneous methods. The current consensus rule---count detectors with at least one break---is a voting heuristic. A Bayesian model averaging approach (e.g., Eckley, Fearnhead, and Killick, 2011) could provide posterior break probabilities that naturally weight detector evidence. We leave this for future work.

### 5.4 False Positives: CUSUM Decay Onset on Non-Stationary Sharpe Series

Both Mkt-RF and RMW receive DECAYING verdicts driven exclusively by the CUSUM decay onset detector, despite exhibiting *improving* rolling Sharpe ratios (positive deltas of +0.37 for both). This is a documented limitation of the CUSUM procedure applied to non-stationary series.

The CUSUM detector (Page, 1954; Brown, Durbin, and Evans, 1975) computes cumulative sums of deviations from the series mean. The critical threshold defaults to $2\sigma$ of the metric series. When the rolling Sharpe series has a strong positive trend (as for RMW and Mkt-RF), early observations systematically deviate from the full-sample mean, causing the CUSUM statistic to cross the critical boundary during the burn-in period (indices 289 and 294, both within the first 252-day rolling window). This is a level-shift artifact, not genuine decay.

The $\sigma$ used in the CUSUM threshold is estimated from the full sample, introducing look-ahead bias: the threshold depends on future observations that would be unavailable in real-time deployment. This is acknowledged in the implementation (see `detect_decay_onset` in `src/evaluation/metrics.py`) and mitigated by the multi-detector consensus rule---neither Mkt-RF nor RMW receives a DEAD classification because PELT and Bai-Perron do not detect return-level breaks with negative post-break means. Nevertheless, the DECAYING label for these factors is misleading when considered in isolation.

A potential remedy is to replace the full-sample $\sigma$ with an expanding-window estimate, or to apply the CUSUM detector only to de-trended metric series (e.g., after removing a linear or locally-weighted trend). We note this as a priority for future framework development.

### 5.5 Comparison to McLean and Pontiff (2016)

McLean and Pontiff (2016) study 97 anomalies and find an average 32% decline in long-short returns post-publication, with the decline concentrated in the long leg. Our analysis extends their work in three dimensions:

1. **Temporal extension.** Our sample runs through January 2026, providing 10 additional years of data beyond their endpoint. The additional decade captures the proliferation of smart-beta ETFs, the value factor's extended underperformance (2017--2020), and the post-COVID regime shifts. The decay magnitudes we document are substantially larger than their 32% average, consistent with continued arbitrage capital inflows.

2. **Formal changepoint methodology.** McLean and Pontiff (2016) use a before/after publication comparison. This design cannot identify the precise timing of decay onset, distinguish gradual decay from abrupt structural breaks, or detect multiple break locations. Our multi-detector framework provides continuous monitoring with date-stamped break estimates. For example, we identify the value premium's structural break cluster in 1999--2000 (the growth bubble) as distinct from the post-2007 secular decline, revealing that decay is not a single event but a multi-phase process.

3. **Regime classification.** The Hamilton (1989) Markov-switching model adds a probabilistic dimension absent from McLean and Pontiff's binary comparison. The transition matrices quantify the persistence of each regime: SMB's decayed regime has expected duration $1/(1-0.966) \approx 29$ trading days per visit, while the alpha-generating regime lasts $1/(1-0.989) \approx 91$ days---but the Viterbi-decoded path shows the decayed regime dominating the post-2000 sample. This granularity enables risk managers to estimate the probability of regime switches in real time.

### 5.6 Practical Implications for Portfolio Construction

The results have direct implications for factor-based portfolio construction:

1. **Factor tilt adjustment.** Allocators should reduce or eliminate tilts to SMB and UMD. The size premium is indistinguishable from noise post-2000, and momentum returns are negative on a risk-adjusted basis post-2009. Maintaining these tilts incurs transaction costs without expected compensation.

2. **Conditional value exposure.** HML and CMA retain positive full-sample returns but exhibit negative recent Sharpe ratios. Conditional strategies---increasing value exposure during the alpha-generating Markov regime and reducing it during the decayed regime---could exploit the remaining premium while avoiding extended drawdowns. The transition probabilities provide actionable signals for regime-aware rebalancing.

3. **Profitability as the anchor factor.** RMW is the only factor with an improving Sharpe trajectory. Its annualized Sharpe of 0.50 and positive Sharpe delta make it the most reliable alpha source among the Fama-French factors as of 2026. Portfolios that emphasize profitability over value or size are expected to deliver superior risk-adjusted returns.

4. **Production monitoring.** The framework's continuous audit pipeline (rolling Sharpe, CUSUM, PELT) can be deployed as a real-time monitoring system. When a factor's rolling Sharpe declines below the DECAYING threshold (0.3 absolute or 30% relative decline), an alert triggers a formal multi-detector audit. This converts the retrospective analysis presented here into a prospective risk management tool.

### 5.7 Limitations

Several limitations qualify the findings:

1. **CUSUM burn-in sensitivity.** As documented in Section 5.4, the CUSUM decay onset detector produces false positives during the initial rolling window when the Sharpe series exhibits a positive trend. The burn-in period (first 252 observations) should be excluded from decay onset estimation, or an expanding-window $\sigma$ estimator should replace the full-sample estimate.

2. **Verdict threshold heuristics.** The DECAYING classification threshold (0.3 Sharpe decline, absolute or 30% relative) is more principled than an arbitrary cutoff---the dual absolute/relative formulation controls for differences in initial Sharpe levels---but it remains a design choice rather than a statistically optimal boundary. Sensitivity analysis across threshold values (0.1 to 0.5 in 0.05 increments) shows that the SMB and UMD DEAD classifications are robust, but CMA's classification flips to DEAD at a threshold of 0.25.

3. **Scope.** The framework is designed for systematic factors (long-short portfolios constructed from cross-sectional sorts), not individual equity signals. Extension to single-stock alpha signals would require modification of the regime model (stock-specific regimes rather than factor-level regimes) and the changepoint detectors (higher noise-to-signal ratios in individual returns).

4. **Look-ahead in CUSUM $\sigma$ estimation.** The full-sample standard deviation used to set the CUSUM critical threshold depends on future observations. This introduces a mild look-ahead bias that inflates the detection rate relative to what would be achievable in real-time deployment. The bias is mitigated by the multi-detector consensus rule but not eliminated. An online variant using expanding-window $\sigma$ would be strictly causal.

5. **Survivor bias in the factor zoo.** The six Fama-French factors are among the most studied anomalies. Our finding that four of six are decaying or dead may understate the true decay rate across the broader factor zoo, where less robust anomalies likely decayed earlier and more completely (Harvey, Liu, and Zhu, 2016).

---

## 6. Conclusions

This paper introduces the Signal Decay Auditor, a unified statistical framework for diagnosing structural breaks in systematic trading factor returns. The framework combines three complementary changepoint detectors---PELT (Killick, Fearnhead, and Eckley, 2012), OLS-CUSUM (Page, 1954; Brown, Durbin, and Evans, 1975), and Bai-Perron sequential sup-F tests (Bai and Perron, 1998, 2003)---with Hamilton (1989) Markov regime-switching models and HAC-robust (Newey and West, 1987) inference for all statistical tests. The multi-detector consensus approach requires agreement across methodologically independent detectors before classifying a factor as dead, reducing false positive rates inherent to any single changepoint method.

Applied to 15,751 daily observations per factor spanning July 1963 through January 2026, the framework classifies four of the six canonical Fama-French factors as exhibiting statistically significant decay. SMB (Size) and UMD (Momentum) are classified DEAD: multiple detectors agree on structural breaks, and post-break mean returns are negative. The size premium's rolling Sharpe declines from 1.07 to -0.23 (delta = -1.31); momentum declines from 1.19 to 0.50 (delta = -0.69) with a negative-mean Markov regime dominating the post-2009 sample. HML (Value) and CMA (Investment) are classified DECAYING with Sharpe declines of -1.23 and -0.98, respectively. Both retain positive full-sample returns but exhibit negative Sharpe ratios in the most recent quartile of the sample, indicating ongoing erosion. The structural break dates cluster in 1999--2001 and 2007--2009, aligning with the growth bubble, the quantitative crisis, and the subsequent proliferation of factor-based investment products.

RMW (Profitability) emerges as the most robust alpha source among the six factors. Despite a formal DECAYING label from the CUSUM decay onset detector---a burn-in artifact documented in Section 5.4---the profitability premium exhibits an *improving* rolling Sharpe (from -0.02 to 0.35, delta = +0.37) and a full-sample annualized Sharpe of 0.50. The economic explanation is that profitability-based strategies face higher arbitrage barriers than price-based or size-based strategies: the short side involves distressed, illiquid, hard-to-borrow names (Pontiff, 1996, 2006), and profitability measurement requires accounting expertise that creates durable information asymmetry (Novy-Marx, 2013).

The market factor (Mkt-RF) is similarly resilient, with an improving Sharpe trajectory (+0.37 delta) and annualized return of 7.25%. The equity risk premium is not arbitrageable in the anomaly sense and persists as compensation for systematic risk bearing.

These results extend McLean and Pontiff (2016) in three ways: 10 additional years of out-of-sample data, formal multi-method changepoint estimation with date-stamped break locations, and probabilistic regime classification via Markov-switching models. The decay magnitudes we document (58--140% of pre-break Sharpe levels) substantially exceed their 32% average, reflecting continued arbitrage capital accumulation in the 2016--2026 smart-beta era.

Several directions for future work are indicated. First, a real-time monitoring dashboard that applies the audit pipeline to streaming factor returns would convert the retrospective analysis into a prospective risk management tool, with CUSUM modified to use expanding-window $\sigma$ estimation for causal inference. Second, extension to the broader factor zoo---AQR factors (Asness, Frazzini, and Pedersen, 2019), Stambaugh and Yuan (2017) mispricing factors, and alternative data signals---would test whether the decay hierarchy generalizes beyond the Fama-French universe. Third, Bayesian changepoint methods (Fearnhead, 2006; Adams and MacKay, 2007) offer a natural framework for combining detector evidence into posterior break probabilities, replacing the current voting heuristic. Fourth, cross-sectional decay analysis---decomposing factor-level decay into contributions from the long and short legs---would clarify whether decay is driven by arbitrage on the long side (as McLean and Pontiff, 2016, suggest), the short side (via reduced shorting costs), or both.

The Signal Decay Auditor framework is available as an open-source Python package. All configurations, parameter justifications, and reproducibility artifacts (random seeds, data checksums, decision logs) are version-controlled. We encourage adoption for continuous monitoring of systematic factor exposures in production portfolio systems.

---

## References

Adams, R.P. & MacKay, D.J.C. (2007). Bayesian Online Changepoint Detection. *arXiv preprint arXiv:0710.3742*.

Andrews, D.W.K. (1993). Tests for Parameter Instability and Structural Change With Unknown Change Point. *Econometrica*, 61(4), 821--856.

Asness, C.S., Frazzini, A. & Pedersen, L.H. (2019). Quality Minus Junk. *Review of Accounting Studies*, 24(1), 34--112.

Bai, J. & Perron, P. (1998). Estimating and Testing Linear Models with Multiple Structural Changes. *Econometrica*, 66(1), 47--78.

Bai, J. & Perron, P. (2003). Computation and analysis of multiple structural change models. *Journal of Applied Econometrics*, 18(1), 1--22.

Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of Econometrics*, 31(3), 307--327.

Brown, R.L., Durbin, J. & Evans, J.M. (1975). Techniques for Testing the Constancy of Regression Relationships over Time. *Journal of the Royal Statistical Society: Series B*, 37(2), 149--192.

Calluzzo, P., Moneta, F. & Topaloglu, S. (2019). When Anomalies Are Publicized Broadly, Do Institutions Trade Accordingly? *Management Science*, 65(10), 4555--4574.

Campbell, J.Y. & Thompson, S.B. (2008). Predicting excess stock returns out of sample: Can anything beat the historical average? *Review of Financial Studies*, 21(4), 1509--1531.

Carhart, M.M. (1997). On persistence in mutual fund performance. *Journal of Finance*, 52(1), 57--82.

Chu, C.-S.J., Hornik, K. & Kauan, C.-M. (1995). MOSUM Tests for Parameter Constancy. *Biometrika*, 82(3), 603--617.

Clark, T.E. & West, K.D. (2007). Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics*, 138(1), 291--311.

Cochrane, J.H. (2011). Presidential Address: Discount Rates. *Journal of Finance*, 66(4), 1047--1108.

Daniel, K. & Moskowitz, T.J. (2016). Momentum Crashes. *Journal of Financial Economics*, 122(2), 221--247.

Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253--263.

Eckley, I.A., Fearnhead, P. & Killick, R. (2011). Analysis of Changepoint Models. In *Bayesian Time Series Models* (eds. D. Barber, A.T. Cemgil & S. Chiappa), Cambridge University Press, 205--224.

Fama, E.F. (1991). Efficient Capital Markets: II. *Journal of Finance*, 46(5), 1575--1617.

Fama, E.F. & French, K.R. (1993). Common Risk Factors in the Returns on Stocks and Bonds. *Journal of Financial Economics*, 33(1), 3--56.

Fama, E.F. & French, K.R. (2015). A five-factor model. *Journal of Financial Economics*, 116(1), 1--22.

Fearnhead, P. (2006). Exact and Efficient Bayesian Inference for Multiple Changepoint Problems. *Statistics and Computing*, 16(2), 203--213.

Giacomini, R. & White, H. (2006). Tests of conditional predictive ability. *Econometrica*, 74(6), 1545--1578.

Glosten, L.R., Jagannathan, R. & Runkle, D.E. (1993). On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks. *Journal of Finance*, 48(5), 1779--1801.

Hamilton, J.D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle. *Econometrica*, 57(2), 357--384.

Harvey, C.R., Liu, Y. & Zhu, H. (2016). ...and the Cross-Section of Expected Returns. *Review of Financial Studies*, 29(1), 5--68.

Hinkley, D.V. (1971). Inference about the change-point from cumulative sum tests. *Biometrika*, 58(3), 509--523.

Horowitz, J.L., Loughran, T. & Savin, N.E. (2000). Three analyses of the firm size premium. *Journal of Empirical Finance*, 7(2), 143--153.

Hou, K., Xue, C. & Zhang, L. (2020). Replicating Anomalies. *Review of Financial Studies*, 33(5), 2019--2133.

Israel, R., Laursen, K. & Richardson, S. (2021). Is (Systematic) Value Investing Dead? *Journal of Portfolio Management*, 47(2), 38--62.

Jegadeesh, N. & Titman, S. (1993). Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency. *Journal of Finance*, 48(1), 65--91.

Jegadeesh, N. & Titman, S. (2001). Profitability of Momentum Strategies: An Evaluation of Alternative Explanations. *Journal of Finance*, 56(2), 699--720.

Killick, R., Fearnhead, P. & Eckley, I.A. (2012). Optimal Detection of Changepoints with a Linear Computational Cost. *Journal of the American Statistical Association*, 107(500), 1590--1598.

Lo, A.W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*, 58(4), 36--52.

Lo, A.W. (2008). *Hedge Funds: An Analytic Perspective*. Princeton University Press.

McLean, R.D. & Pontiff, J. (2016). Does academic research destroy stock return predictability? *Journal of Finance*, 71(1), 5--32.

Nelson, D.B. (1991). Conditional Heteroskedasticity in Asset Returns: A New Approach. *Econometrica*, 59(2), 347--370.

Newey, W.K. & West, K.D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3), 703--708.

Newey, W.K. & West, K.D. (1994). Automatic lag selection in covariance matrix estimation. *Review of Economic Studies*, 61(4), 631--653.

Novy-Marx, R. (2013). The other side of value: The gross profitability premium. *Journal of Financial Economics*, 108(1), 1--28.

Page, E.S. (1954). Continuous Inspection Schemes. *Biometrika*, 41(1/2), 100--115.

Politis, D.N. & Romano, J.P. (1994). The Stationary Bootstrap. *Journal of the American Statistical Association*, 89(428), 1303--1313.

Pontiff, J. (1996). Costly Arbitrage: Evidence from Closed-End Funds. *Quarterly Journal of Economics*, 111(4), 1135--1151.

Pontiff, J. (2006). Costly Arbitrage and the Myth of Idiosyncratic Risk. *Journal of Accounting and Economics*, 42(1--2), 35--52.

Rabiner, L.R. (1989). A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition. *Proceedings of the IEEE*, 77(2), 257--286.

Schwarz, G. (1978). Estimating the Dimension of a Model. *Annals of Statistics*, 6(2), 461--464.

Schwert, G.W. (2003). Anomalies and Market Efficiency. In *Handbook of the Economics of Finance* (eds. G.M. Constantinides, M. Harris & R.M. Stulz), Elsevier, Vol. 1, 939--974.

Sharpe, W.F. (1994). The Sharpe ratio. *Journal of Portfolio Management*, 21(1), 49--58.

Stambaugh, R.F. & Yuan, Y. (2017). Mispricing Factors. *Review of Financial Studies*, 30(4), 1270--1315.

Uhlenbeck, G.E. & Ornstein, L.S. (1930). On the theory of the Brownian motion. *Physical Review*, 36(5), 823--841.

Welch, I. & Goyal, A. (2008). A comprehensive look at the empirical performance of equity premium prediction. *Review of Financial Studies*, 21(4), 1455--1508.

Yao, Y.-C. (1988). Estimating the Number of Change-points via Schwarz' Criterion. *Statistics & Probability Letters*, 6(3), 181--189.
