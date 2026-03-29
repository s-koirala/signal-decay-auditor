# Signal Decay Audit Report: TSLA

## Overview
- **Signal**: TSLA
- **Observations**: 2824
- **Date range**: 2015-01-05 00:00:00 to 2026-03-27 00:00:00
- **Verdict**: DECAYING: Decay onset detected at index 272 via CUSUM on rolling Sharpe.

## Metrics
- **rolling_sharpe**: mean=0.7554, last=0.7870, min=-1.5564, max=3.3419
- **half_life**: half_life=inf, theta=nan, r_squared=0.0000, slope=-0.0070, se_slope=0.0188
- **oos_r_squared**: not computed

## Changepoint Detection
### Pelt
- Breakpoints on returns: none detected
- Breakpoints on rolling Sharpe: [511, 591, 751, 811, 1251, 1351, 1601, 1746, 1956, 2106, 2236, 2281, 2466]

### Cusum
- Breakpoints on returns: [2580]
- Breakpoints on rolling Sharpe: [1423]

### Bai Perron
- Breakpoints on returns: none detected
- Breakpoints on rolling Sharpe: [812, 1250, 1742, 2468]

## Regime Detection
- Regime 0 (alpha_generating): mean=0.003660
- Regime 1 (decayed): mean=0.001027

Transition matrix:
```
  0.8758  0.1242
  0.0503  0.9497
```

## Decay Detection
- **Decay detected**: Yes
- **Onset index**: 272
