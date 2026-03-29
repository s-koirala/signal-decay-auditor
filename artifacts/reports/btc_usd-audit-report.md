# Signal Decay Audit Report: BTC-USD

## Overview
- **Signal**: BTC-USD
- **Observations**: 4105
- **Date range**: 2015-01-02 00:00:00 to 2026-03-29 00:00:00
- **Verdict**: DECAYING: Pelt detected structural break at index 1045. Rolling Sharpe declined from 1.7 to 1.1.

## Metrics
- **rolling_sharpe**: mean=0.9908, last=-1.2897, min=-1.8244, max=4.0529
- **half_life**: half_life=inf, theta=nan, r_squared=0.0007, slope=-0.0273, se_slope=0.0156
- **oos_r_squared**: not computed

## Changepoint Detection
### Pelt
- Breakpoints on returns: [1045, 1080]
- Breakpoints on rolling Sharpe: [331, 851, 1126, 1256, 1316, 1566, 1631, 1801, 1881, 1976, 2146, 2331, 2441, 2681, 2936, 2996, 3326, 3471, 3861, 4036]

### Cusum
- Breakpoints on returns: [1895]
- Breakpoints on rolling Sharpe: [1471]

### Bai Perron
- Breakpoints on returns: none detected
- Breakpoints on rolling Sharpe: [1222, 1654, 2441, 2971]

## Regime Detection
- Regime 0 (alpha_generating): mean=0.002825
- Regime 1 (decayed): mean=0.001400

Transition matrix:
```
  0.8207  0.1793
  0.1053  0.8947
```

## Decay Detection
- **Decay detected**: Yes
- **Onset index**: 1492
