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

## References

Andrews, D.W.K. (1993). Tests for Parameter Instability and Structural Change With Unknown Change Point. *Econometrica*, 61(4), 821--856.

Bai, J. & Perron, P. (1998). Estimating and Testing Linear Models with Multiple Structural Changes. *Econometrica*, 66(1), 47--78.

Bai, J. & Perron, P. (2003). Computation and analysis of multiple structural change models. *Journal of Applied Econometrics*, 18(1), 1--22.

Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of Econometrics*, 31(3), 307--327.

Brown, R.L., Durbin, J. & Evans, J.M. (1975). Techniques for Testing the Constancy of Regression Relationships over Time. *Journal of the Royal Statistical Society: Series B*, 37(2), 149--192.

Campbell, J.Y. & Thompson, S.B. (2008). Predicting excess stock returns out of sample: Can anything beat the historical average? *Review of Financial Studies*, 21(4), 1509--1531.

Carhart, M.M. (1997). On persistence in mutual fund performance. *Journal of Finance*, 52(1), 57--82.

Chu, C.-S.J., Hornik, K. & Kauan, C.-M. (1995). MOSUM Tests for Parameter Constancy. *Biometrika*, 82(3), 603--617.

Clark, T.E. & West, K.D. (2007). Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics*, 138(1), 291--311.

Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253--263.

Fama, E.F. & French, K.R. (2015). A five-factor model. *Journal of Financial Economics*, 116(1), 1--22.

Giacomini, R. & White, H. (2006). Tests of conditional predictive ability. *Econometrica*, 74(6), 1545--1578.

Glosten, L.R., Jagannathan, R. & Runkle, D.E. (1993). On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks. *Journal of Finance*, 48(5), 1779--1801.

Hamilton, J.D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle. *Econometrica*, 57(2), 357--384.

Hinkley, D.V. (1971). Inference about the change-point from cumulative sum tests. *Biometrika*, 58(3), 509--523.

Killick, R., Fearnhead, P. & Eckley, I.A. (2012). Optimal Detection of Changepoints with a Linear Computational Cost. *Journal of the American Statistical Association*, 107(500), 1590--1598.

Lo, A.W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*, 58(4), 36--52.

McLean, R.D. & Pontiff, J. (2016). Does academic research destroy stock return predictability? *Journal of Finance*, 71(1), 5--32.

Nelson, D.B. (1991). Conditional Heteroskedasticity in Asset Returns: A New Approach. *Econometrica*, 59(2), 347--370.

Newey, W.K. & West, K.D. (1994). Automatic lag selection in covariance matrix estimation. *Review of Economic Studies*, 61(4), 631--653.

Page, E.S. (1954). Continuous Inspection Schemes. *Biometrika*, 41(1/2), 100--115.

Politis, D.N. & Romano, J.P. (1994). The Stationary Bootstrap. *Journal of the American Statistical Association*, 89(428), 1303--1313.

Rabiner, L.R. (1989). A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition. *Proceedings of the IEEE*, 77(2), 257--286.

Schwarz, G. (1978). Estimating the Dimension of a Model. *Annals of Statistics*, 6(2), 461--464.

Sharpe, W.F. (1994). The Sharpe ratio. *Journal of Portfolio Management*, 21(1), 49--58.

Uhlenbeck, G.E. & Ornstein, L.S. (1930). On the theory of the Brownian motion. *Physical Review*, 36(5), 823--841.

Welch, I. & Goyal, A. (2008). A comprehensive look at the empirical performance of equity premium prediction. *Review of Financial Studies*, 21(4), 1455--1508.

Yao, Y.-C. (1988). Estimating the Number of Change-points via Schwarz' Criterion. *Statistics & Probability Letters*, 6(3), 181--189.
