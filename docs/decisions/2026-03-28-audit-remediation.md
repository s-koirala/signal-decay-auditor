# Audit Remediation Log — 2026-03-28

## Scope
Full code audit of all source modules in signal-decay-auditor: changepoint detectors, regime models, evaluation metrics, orchestration pipeline, data loader, and synthetic generators.

## Methodology
Four parallel audit agents reviewed each module against cited literature, CLAUDE.md directives, and statistical best practices. Findings classified by severity: CRITICAL, HIGH, MEDIUM, LOW. Remediation applied iteratively with regression testing at each stage. Residual verification audit confirmed all fixes.

## Baseline
- 36 unit tests, all passing
- Python 3.9.13, pytest 7.4.3

## Findings Summary
- **4 CRITICAL** — all remediated
- **10 HIGH** — all remediated (7 code fixes, 3 accepted as design constraints with documentation)
- **19 MEDIUM** — key items remediated
- **27 LOW** — cataloged for future work

## CRITICAL Remediations

### C1. Clark-West test used OLS standard errors instead of HAC
**File:** `src/evaluation/metrics.py`, `clark_west_test()`
**Issue:** Clark & West (2007) require Newey-West HAC standard errors because the MSPE-adjusted series inherits serial correlation from overlapping forecast errors. The original implementation used i.i.d. standard errors, producing anti-conservative inference (inflated Type I error).
**Fix:** Implemented Bartlett-kernel HAC variance estimator with Newey-West (1994) automatic bandwidth: `floor(4*(n/100)^(2/9))`.

### C2. Giacomini-White unconditional test used OLS standard errors instead of HAC
**File:** `src/evaluation/metrics.py`, `giacomini_white_test()`
**Issue:** Same as C1. Giacomini & White (2006) explicitly require HAC for loss differentials.
**Fix:** Same Newey-West HAC implementation applied. Also updated the conditional branch to use the Newey-West (1994) bandwidth formula instead of the previous `n^(1/3)` heuristic.

### C3. CUSUM decay onset expanding mean included current observation
**File:** `src/evaluation/metrics.py`, `detect_decay_onset()`
**Issue:** The expanding mean at time t included x_t itself, dampening all deviations toward zero and reducing detection power.
**Fix:** Shifted expanding mean by one period: `expanding_mu[t] = mean(x[0:t-1])`. First observation produces zero deviation.

### C4. MarkovRegimeDetector transition matrix convention mismatch
**File:** `src/detectors/regime.py`, `MarkovRegimeDetector`
**Issue:** `statsmodels.MarkovRegression.regime_transition` returns a column-stochastic matrix (columns sum to 1). The docstring claimed row-stochastic (Hamilton 1989 convention, rows sum to 1). Off-diagonal probabilities were transposed for any downstream consumer.
**Fix:** Transpose `regime_transition` in `fit()` before storing. Updated `get_transition_matrix()` to return the already-transposed matrix.

## HIGH Remediations

### H1. MOSUM linear boundary formula inverted
**File:** `src/detectors/changepoint.py`, `MOSUMDetector`
**Issue:** Used `abs(t - 0.5)` which widens boundary at center and narrows at endpoints — the inverse of Chu, Hornik & Kauan (1995).
**Fix:** Changed to `np.minimum(t, 1.0 - t)` per CHK (1995).

### H2. Bai-Perron sequential procedure used depth-dependent critical values
**File:** `src/detectors/changepoint.py`, `BaiPerronDetector`
**Issue:** `_get_critical_value(depth + 1)` passed recursion depth, using lower (more liberal) critical values at deeper levels. The sequential procedure tests one break per step; always uses single-break CVs.
**Fix:** Changed to `_get_critical_value(1)` per Bai & Perron (1998, Section 4).

### H3. GARCH 2-segment labeling non-contiguous
**File:** `src/detectors/regime.py`, `GARCHRegimeDetector`
**Issue:** Two-segment case assigned labels {0, 2}, skipping 1.
**Fix:** Changed to {0, 1} for contiguous labels.

### H4. Silent EM initialization failures
**File:** `src/detectors/regime.py`
**Issue:** No warning when >50% of EM random starts fail.
**Fix:** Added success counters to both MarkovRegimeDetector and HMMRegimeDetector; log warning when `n_success < n_init // 2`.

### H5. oos_r_squared asymmetric t=0 treatment
**File:** `src/evaluation/metrics.py`
**Issue:** Model penalized at t=0 (NaN benchmark contributes 0 via nansum, but model contributes its full squared error).
**Fix:** Added valid_mask ensuring model and benchmark SSE are computed over identical observation sets.

### H6. signal_half_life docstring mismatch
**File:** `src/evaluation/metrics.py`
**Issue:** Docstring wrote `r_t = phi * r_{t-1} + epsilon_t` but implementation fits OLS with intercept.
**Fix:** Updated docstring to `r_t = alpha + phi * r_{t-1} + epsilon_t`.

### H7. data_loader date index fragile for monthly PeriodIndex
**File:** `src/factors/data_loader.py`
**Issue:** `pd.to_datetime(df.index.astype(str))` converts monthly periods to first-of-month.
**Fix:** Uses `to_timestamp(how='end')` for PeriodIndex inputs.

## MEDIUM Remediations (Selected)

- **PELT BIC penalty**: Changed `np.var(arr)` to `np.var(arr, ddof=1)` (unbiased variance).
- **Attribute initialization**: Added `_series_cache = None` in CUSUMDetector `__init__`.
- **clark_west_test**: Added configurable `significance` parameter (default 0.05).
- **detect_decay_onset**: Added configurable `consecutive_periods` parameter (default 5).
- **auditor.py DEAD verdict**: Falls back to empirical post-break mean when regime model unavailable.
- **auditor.py Sharpe threshold**: Changed from absolute 0.3 to `min(0.3, 0.3 * |sharpe_start|)`.
- **Dead code removal**: Removed unused `self.cache` dict and `_spearman_corr` helper from data_loader.py.

## Test Coverage Expansion
- **Before:** 36 unit tests
- **After:** 47 unit tests (+11)
- New tests cover: constant series, short series validation, CUSUM decay detection, NaN input handling, significance parameter, transition matrix row-sum verification, contiguous GARCH labels, single-regime labeling.

## Post-Remediation Status
- 47/47 unit tests passing
- Residual verification audit: 21/22 VERIFIED, 1 PARTIAL (stale docstring, subsequently fixed), 0 FAILED
- Zero regressions detected

## References
- Newey, W. K. & West, K. D. (1994). Automatic Lag Selection in Covariance Matrix Estimation. *Review of Economic Studies*, 61(4), 631-653.
- Clark, T. E. & West, K. D. (2007). *Journal of Econometrics*, 138(1), 291-311.
- Giacomini, R. & White, H. (2006). *Econometrica*, 74(6), 1545-1578.
- Chu, C.-S.J., Hornik, K. & Kauan, C.-M. (1995). *Biometrika*, 82(3), 603-617.
- Bai, J. & Perron, P. (1998). *Econometrica*, 66(1), 47-78.
- Hamilton, J.D. (1989). *Econometrica*, 57(2), 357-384.
- Killick, R., Fearnhead, P. & Eckley, I.A. (2012). *JASA*, 107(500), 1590-1598.
- Campbell, J. Y. & Thompson, S. B. (2008). *Review of Financial Studies*, 21(4), 1509-1531.
- Page, E. S. (1954). *Biometrika*, 41(1/2), 100-115.

---

## Round 2 — Test Coverage Expansion and Additional Fixes (2026-03-29)

### Test Coverage (continued)
- **Before round 2:** 47 unit tests + 0 non-slow integration tests
- **After round 2:** 57 unit tests + 7 synthetic integration tests = 64 total (non-slow)

### New Unit Tests Added
- `TestRollingOosRSquared` (3 tests): output length, NaN before window, positive R2 for good predictor
- `TestRollingHalfLife` (2 tests): output length, O-U process finite half-life
- `TestRollingInformationCoefficient` (2 tests): output length, perfect rank correlation
- `TestCSSFE` (2 tests): output length, perfect forecast rising
- `TestGiacominiWhiteTest::test_conditional_different_models`: conditional variant with HAC Wald test

### New Synthetic Integration Tests
Created `tests/integration/test_synthetic_audit.py` — network-independent, runs on every CI build:
- **TestFactorLifecycleAudit**: decay detection, non-ACTIVE verdict, changepoint near true break
- **TestNoBreakAudit**: no breakpoints on returns, well-formed report
- **TestStrongMeanShiftAudit**: >=2 detectors fire, break near true location

### Additional Fix: PELT Cost Model in Auditor Pipeline
Changed `SignalDecayAuditor._make_detector("pelt")` from default `model="rbf"` to `model="l2"`. The `l2` (least-squares) cost is the natural parametric choice for mean-shift detection in Gaussian data and correctly controls false positives under the null. The `rbf` kernel cost oversplits when SNR is low (57 false breakpoints on pure noise in testing).
