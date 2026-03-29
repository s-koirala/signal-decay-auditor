# Signal Decay Audit Report: HML (Value)

## Overview
- **Signal**: HML (Value)
- **Observations**: 15751
- **Date range**: 1963-07-01 00:00:00 to 2026-01-30 00:00:00
- **Verdict**: DECAYING: Pelt detected structural break at index 9090. Rolling Sharpe declined from 1.0 to -0.2.

## Metrics
- **rolling_sharpe**: mean=0.5489, last=0.7606, min=-3.9491, max=5.6068
- **half_life**: half_life=0.2850, theta=2.4323, r_squared=0.0077, slope=0.0878, se_slope=0.0079
- **oos_r_squared**: not computed

## Changepoint Detection
### Pelt
- Breakpoints on returns: [9090, 9235, 9505, 11455, 11500, 11530, 14225, 14315]
- Breakpoints on rolling Sharpe: [371, 466, 616, 861, 1166, 1321, 1521, 1681, 1941, 2081, 2291, 2336, 2421, 2846, 2956, 3151, 3456, 3531, 3701, 3971, 4086, 4266, 4391, 4501, 4566, 4811, 4901, 5096, 5166, 5296, 5416, 5716, 5786, 6051, 6171, 6396, 6501, 6611, 6761, 6826, 7201, 7406, 7501, 7701, 8456, 8711, 8916, 9346, 9416, 9661, 9971, 10201, 11031, 11081, 11341, 11716, 11856, 12021, 12161, 12386, 12701, 13006, 13236, 13431, 13541, 13686, 14516, 15026, 15281]

### Cusum
- Breakpoints on returns: [14438]
- Breakpoints on rolling Sharpe: [6798]

### Bai Perron
- Breakpoints on returns: none detected
- Breakpoints on rolling Sharpe: [2402, 6562, 9416, 11048, 14518]

## Regime Detection
- Regime 0 (alpha_generating): mean=0.000415
- Regime 1 (decayed): mean=0.000058

Transition matrix:
```
  0.9690  0.0310
  0.0102  0.9898
```

## Decay Detection
- **Decay detected**: Yes
- **Onset index**: 371
