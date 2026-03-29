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
