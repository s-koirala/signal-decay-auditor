# Signal Decay Audit Report: GLD

## Overview
- **Signal**: GLD
- **Observations**: 2824
- **Date range**: 2015-01-05 00:00:00 to 2026-03-27 00:00:00
- **Verdict**: DECAYING: Decay onset detected at index 264 via CUSUM on rolling Sharpe.

## Metrics
- **rolling_sharpe**: mean=0.7807, last=1.5866, min=-1.2323, max=3.3325
- **half_life**: half_life=0.1375, theta=5.0396, r_squared=0.0000, slope=0.0065, se_slope=0.0189
- **oos_r_squared**: not computed

## Changepoint Detection
### Pelt
- Breakpoints on returns: none detected
- Breakpoints on rolling Sharpe: [281, 526, 721, 856, 1121, 1511, 1626, 1791, 1846, 2091, 2326, 2416]

### Cusum
- Breakpoints on returns: [2783]
- Breakpoints on rolling Sharpe: [2781]

### Bai Perron
- Breakpoints on returns: none detected
- Breakpoints on rolling Sharpe: [889, 1120, 1524, 2095, 2330]

## Regime Detection
- Regime 0 (decayed): mean=0.000468
- Regime 1 (alpha_generating): mean=0.000823

Transition matrix:
```
  0.9892  0.0108
  0.0854  0.9146
```

## Decay Detection
- **Decay detected**: Yes
- **Onset index**: 264
