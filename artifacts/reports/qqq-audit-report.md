# Signal Decay Audit Report: QQQ

## Overview
- **Signal**: QQQ
- **Observations**: 2824
- **Date range**: 2015-01-05 00:00:00 to 2026-03-27 00:00:00
- **Verdict**: DECAYING: Decay onset detected at index 264 via CUSUM on rolling Sharpe.

## Metrics
- **rolling_sharpe**: mean=1.0594, last=0.7897, min=-1.1747, max=3.1493
- **half_life**: half_life=inf, theta=nan, r_squared=0.0117, slope=-0.1081, se_slope=0.0187
- **oos_r_squared**: not computed

## Changepoint Detection
### Pelt
- Breakpoints on returns: none detected
- Breakpoints on rolling Sharpe: [401, 511, 711, 781, 956, 1226, 1301, 1351, 1556, 1771, 1836, 2101, 2226, 2396, 2551]

### Cusum
- Breakpoints on returns: none detected
- Breakpoints on rolling Sharpe: [1972]

### Bai Perron
- Breakpoints on returns: none detected
- Breakpoints on rolling Sharpe: [506, 809, 1226, 1772, 2103]

## Regime Detection
- Regime 0 (alpha_generating): mean=0.001465
- Regime 1 (decayed): mean=-0.001015

Transition matrix:
```
  0.9853  0.0147
  0.0342  0.9658
```

## Decay Detection
- **Decay detected**: Yes
- **Onset index**: 264
