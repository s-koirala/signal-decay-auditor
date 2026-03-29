# Signal Decay Audit Report: SPY

## Overview
- **Signal**: SPY
- **Observations**: 2824
- **Date range**: 2015-01-05 00:00:00 to 2026-03-27 00:00:00
- **Verdict**: DECAYING: Decay onset detected at index 264 via CUSUM on rolling Sharpe.

## Metrics
- **rolling_sharpe**: mean=1.0608, last=0.7288, min=-0.7906, max=3.5977
- **half_life**: half_life=inf, theta=nan, r_squared=0.0147, slope=-0.1212, se_slope=0.0187
- **oos_r_squared**: not computed

## Changepoint Detection
### Pelt
- Breakpoints on returns: none detected
- Breakpoints on rolling Sharpe: [391, 506, 691, 776, 951, 1241, 1291, 1556, 1771, 1841, 2101, 2251, 2551]

### Cusum
- Breakpoints on returns: none detected
- Breakpoints on rolling Sharpe: [767]

### Bai Perron
- Breakpoints on returns: none detected
- Breakpoints on rolling Sharpe: [487, 806, 1556, 1830, 2247]

## Regime Detection
- Regime 0 (decayed): mean=-0.000825
- Regime 1 (alpha_generating): mean=0.001102

Transition matrix:
```
  0.9584  0.0416
  0.0183  0.9817
```

## Decay Detection
- **Decay detected**: Yes
- **Onset index**: 264
