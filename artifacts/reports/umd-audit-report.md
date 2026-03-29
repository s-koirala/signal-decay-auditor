# Signal Decay Audit Report: UMD (Momentum)

## Overview
- **Signal**: UMD (Momentum)
- **Observations**: 15751
- **Date range**: 1963-07-01 00:00:00 to 2026-01-30 00:00:00
- **Verdict**: DEAD: Multiple detectors agree on break near index 11335. Post-break mean return is negative. Post-break rolling Sharpe: 0.50.

## Metrics
- **rolling_sharpe**: mean=1.0251, last=0.6064, min=-3.0007, max=6.2345
- **half_life**: half_life=0.3841, theta=1.8045, r_squared=0.0271, slope=0.1645, se_slope=0.0079
- **oos_r_squared**: not computed

## Changepoint Detection
### Pelt
- Breakpoints on returns: [9200, 9235, 9265, 9885, 9920, 11305, 11335, 11375, 11430, 11460, 11500, 11545]
- Breakpoints on rolling Sharpe: [556, 606, 796, 886, 1066, 1521, 1771, 1871, 2026, 2136, 2371, 2501, 2661, 2876, 3026, 3141, 3446, 3486, 3636, 3831, 3991, 4091, 4391, 4566, 4661, 4756, 4896, 5026, 5116, 5281, 5346, 5551, 5666, 5731, 5826, 5931, 6106, 6451, 6531, 6816, 6896, 6996, 7456, 7536, 7646, 7726, 7801, 7896, 8006, 8321, 8786, 9011, 9141, 9401, 9771, 9901, 10026, 10296, 10481, 10796, 10866, 11116, 11351, 11521, 11631, 11781, 11871, 12216, 13081, 13256, 13356, 13616, 13686, 13926, 14446, 14766, 14996, 15246]

### Cusum
- Breakpoints on returns: none detected
- Breakpoints on rolling Sharpe: [6873]

### Bai Perron
- Breakpoints on returns: [11428]
- Breakpoints on rolling Sharpe: [3454, 5022, 6455, 10007, 11876]

## Regime Detection
- Regime 0 (alpha_generating): mean=0.000682
- Regime 1 (decayed): mean=-0.000596

Transition matrix:
```
  0.9816  0.0184
  0.0408  0.9592
```

## Decay Detection
- **Decay detected**: Yes
- **Onset index**: 279
