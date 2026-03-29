# Signal Decay Audit Report: NVDA

## Overview
- **Signal**: NVDA
- **Observations**: 2824
- **Date range**: 2015-01-05 00:00:00 to 2026-03-27 00:00:00
- **Verdict**: DECAYING: Decay onset detected at index 263 via CUSUM on rolling Sharpe.

## Metrics
- **rolling_sharpe**: mean=1.4485, last=1.1428, min=-0.9915, max=3.6426
- **half_life**: half_life=inf, theta=nan, r_squared=0.0054, slope=-0.0738, se_slope=0.0188
- **oos_r_squared**: not computed

## Changepoint Detection
### Pelt
- Breakpoints on returns: none detected
- Breakpoints on rolling Sharpe: [346, 646, 846, 956, 1226, 1351, 1831, 1926, 2026, 2101, 2176, 2531]

### Cusum
- Breakpoints on returns: none detected
- Breakpoints on rolling Sharpe: [1022]

### Bai Perron
- Breakpoints on returns: none detected
- Breakpoints on rolling Sharpe: [850, 1229, 1838, 2112]

## Regime Detection
- Regime 0 (alpha_generating): mean=0.002975
- Regime 1 (decayed): mean=0.001470

Transition matrix:
```
  0.9685  0.0315
  0.0765  0.9235
```

## Decay Detection
- **Decay detected**: Yes
- **Onset index**: 263
