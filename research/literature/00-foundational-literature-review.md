# Signal Decay Auditor — Foundational Literature Review

*Compiled 2026-03-28*

---

## 1. Changepoint Detection Theory

### Page (1954) — CUSUM
**Citation:** Page, E.S. (1954). "Continuous Inspection Schemes." *Biometrika*, 41(1/2), 100–115.
**Contribution:** Introduced the Cumulative Sum (CUSUM) control chart for sequential monitoring. Accumulates deviations from a target value and signals change when the cumulative sum exceeds a threshold. Foundation for all subsequent changepoint methods.
**Statistics:** CUSUM statistic, sequential probability ratio framework.
**Python:** `ruptures`, `statsmodels.stats.diagnostic.breaks_cusumolsresid`.

### Bai & Perron (1998, 2003) — Multiple Structural Breaks
**Citation:** Bai, J. & Perron, P. (1998). "Estimating and Testing Linear Models with Multiple Structural Changes." *Econometrica*, 66(1), 47–78. Companion: *Journal of Applied Econometrics*, 18(1), 1–22 (2003).
**Contribution:** Comprehensive framework for estimating and testing multiple structural breaks at unknown dates in linear regression. Asymptotic distributions for break-date estimators. Sequential testing procedure (sup-F tests).
**Statistics:** Sup-F test for l vs. 0 breaks; sequential sup-F(l+1|l); BIC/LWZ for number of breaks; dynamic programming for global SSR minimization; break-date confidence intervals.
**Python:** No canonical package. `ruptures` provides DP/PELT backends (approximate). Full procedure (sequential testing, sup-F critical values) needs to be built. R `strucchange` is gold standard.

### Andrews (1993) — Optimal Tests for Single Unknown Break
**Citation:** Andrews, D.W.K. (1993). "Tests for Parameter Instability and Structural Change with Unknown Change Point." *Econometrica*, 61(4), 821–856.
**Contribution:** Derived asymptotic distribution of sup-Wald, sup-LM, sup-LR statistics for testing a single structural break at unknown date. Provided critical value tables.
**Statistics:** Sup-Wald, sup-LM, sup-LR statistics; trimming proportion.
**Python:** Needs to be built. Test statistics computable from `statsmodels` OLS.

### Chu, Hornik & Kauan (1995) — MOSUM
**Citation:** Chu, C.-S.J., Hornik, K. & Kauan, C.-M. (1995). "MOSUM Tests for Parameter Constancy." *Biometrika*, 82(3), 603–617.
**Contribution:** Moving Sum procedures for monitoring parameter constancy. Better power than CUSUM for localized breaks. Detects multiple breaks in single pass.
**Statistics:** OLS-MOSUM statistic; recursive MOSUM; bandwidth selection.
**Python:** Approximate in `ruptures` (window methods). Direct implementation needed for exact version.

### Killick, Fearnhead & Eckley (2012) — PELT
**Citation:** Killick, R., Fearnhead, P. & Eckley, I.A. (2012). "Optimal Detection of Changepoints with a Linear Computational Cost." *JASA*, 107(500), 1590–1598.
**Contribution:** Pruned Exact Linear Time algorithm. O(n) expected cost for exact penalized cost optimization. Pruning discards provably suboptimal candidates.
**Statistics:** PELT algorithm; pruning condition; applicable to any sub-additive cost function.
**Python:** `ruptures.Pelt` — excellent implementation.

---

## 2. Regime-Switching Models

### Hamilton (1989) — Markov-Switching
**Citation:** Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2), 357–384.
**Contribution:** Markov-switching AR model with unobserved regime Markov chain. Hamilton filter for regime probability inference.
**Statistics:** Markov-switching AR; Hamilton filter; EM/MLE estimation; smoothed regime probabilities.
**Python:** `statsmodels.tsa.regime_switching.MarkovRegression`, `MarkovAutoregression`.

### Ang & Bekaert (2002) — Regime Switches in Interest Rates
**Citation:** Ang, A. & Bekaert, G. (2002). "Regime Switches in Interest Rates." *JBES*, 20(2), 163–182.
**Contribution:** Two-regime model for short rates dramatically improves fit. Ignoring regimes biases risk premia.
**Python:** `statsmodels` Markov-switching machinery.

### Guidolin & Timmermann (2007) — Multivariate Regime Switching
**Citation:** Guidolin, M. & Timmermann, A. (2007). "Asset Allocation under Multivariate Regime Switching." *JEDC*, 31(11), 3503–3544.
**Contribution:** Four-regime model for stock/bond returns with regime-dependent correlations yields significant OOS utility gains.
**Python:** Needs custom build (extend `statsmodels` univariate to multivariate).

---

## 3. Factor Decay / Publication Effect

### McLean & Pontiff (2016) — Publication Destroys Predictability
**Citation:** McLean, R.D. & Pontiff, J. (2016). "Does Academic Research Destroy Stock Return Predictability?" *JoF*, 71(1), 5–32.
**Contribution:** 97 predictors: 26% lower OOS, 58% lower post-publication. Decomposition into overfitting vs. investor learning. Higher liquidity → faster decay.
**Python:** Standard portfolio sorts + OLS. No specialized package needed.

### Harvey, Liu & Zhu (2016) — Multiple Testing
**Citation:** Harvey, C.R., Liu, Y. & Zhu, C. (2016). "...and the Cross-Section of Expected Returns." *RFS*, 29(1), 5–68.
**Contribution:** t > 3.0 threshold after multiple testing. 300+ factors catalogued. Bonferroni/Holm/BHY adjustments.
**Python:** `statsmodels.stats.multitest`.

### Chordia, Subrahmanyam & Tong (2014) — Anomaly Attenuation
**Citation:** Chordia, T., Subrahmanyam, A. & Tong, Q. (2014). "Have Capital Market Anomalies Attenuated?" *JFQA*, 49(4), 1139–1168.
**Contribution:** Anomalies attenuated post-2003. Consistent with increased hedge fund activity and lower costs.

### Schwert (2003) — Anomalies and Market Efficiency
**Citation:** Schwert, G.W. (2003). "Anomalies and Market Efficiency." *Handbook of Economics of Finance*, Vol. 1B, 939–974.
**Contribution:** Several anomalies disappeared post-publication. Consistent with market efficiency.

---

## 4. Forecast Evaluation

### Giacomini & White (2006) — Conditional Predictive Ability
**Citation:** Giacomini, R. & White, H. (2006). "Tests of Conditional Predictive Ability." *Econometrica*, 74(6), 1545–1578.
**Contribution:** Tests that condition on current information set rather than averaging over all states. Model A better on average but Model B better in specific regimes.
**Statistics:** CPA test; regression of loss differentials on instruments; Wald statistic.
**Python:** Needs to be built. OLS + Wald test with `statsmodels`.

### Clark & West (2007) — MSPE-Adjusted Test
**Citation:** Clark, T.E. & West, M.W. (2007). "Approximately Normal Tests for Equal Predictive Accuracy in Nested Models." *JoE*, 138(1), 291–311.
**Contribution:** Correction for upward bias in unrestricted model MSPE under the null (nested models).
**Statistics:** MSPE-adjusted (CW) statistic; one-sided normal test.
**Python:** Trivial to implement (~5 lines).

### Campbell & Thompson (2008) — OOS R²
**Citation:** Campbell, J.Y. & Thompson, S.B. (2008). "Predicting Excess Stock Returns Out of Sample." *RFS*, 21(4), 1509–1531.
**Contribution:** Sign-restricted forecasts improve OOS R². Established R²_OOS as standard metric.
**Statistics:** R²_OOS = 1 - MSPE_model / MSPE_mean.
**Python:** Trivial.

---

## 5. Bootstrap Methods for Changepoints

### Hansen (2000) — Fixed-Regressor Bootstrap
**Citation:** Hansen, B.E. (2000). "Testing for Structural Change in Conditional Models." *JoE*, 97(1), 93–115.
**Contribution:** Bootstrap p-values for sup-type statistics robust to heteroskedasticity and non-normality.
**Statistics:** Fixed-regressor wild bootstrap; heteroskedasticity-robust versions.
**Python:** Needs to be built. `arch` provides bootstrap primitives.

### Antoch & Huskova (2001) — Permutation Tests
**Citation:** Antoch, J. & Huskova, M. (2001). "Permutation Tests in Change Point Analysis." *SPL*, 53(1), 37–46.
**Contribution:** Exact, distribution-free permutation tests for changepoints. Correct size in small samples.
**Statistics:** Permutation CUSUM; Monte Carlo permutation p-values.
**Python:** Adaptable from `scipy.stats.permutation_test` (SciPy 1.8+).

---

## Implementation Priority

| Priority | Method | Package | Build Effort |
|----------|--------|---------|-------------|
| 1 | PELT changepoint detection | `ruptures` | Ready |
| 2 | Markov-switching regime detection | `statsmodels` | Ready |
| 3 | Bai-Perron sequential testing | Custom | Medium |
| 4 | Clark-West MSPE-adjusted test | Custom | Trivial |
| 5 | Giacomini-White CPA test | Custom | Low |
| 6 | OOS R² tracking | Custom | Trivial |
| 7 | Hansen fixed-regressor bootstrap | Custom + `arch` | Medium |
| 8 | Andrews sup-Wald/LM/LR | Custom | Medium |
| 9 | Permutation changepoint tests | `scipy` | Low |
| 10 | O-U half-life estimation | Custom | Trivial |
