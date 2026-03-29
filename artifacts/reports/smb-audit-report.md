# Signal Decay Audit Report: SMB (Size)

## Overview
- **Signal**: SMB (Size)
- **Observations**: 15751
- **Date range**: 1963-07-01 00:00:00 to 2026-01-30 00:00:00
- **Verdict**: DEAD: Multiple detectors agree on break near index 9220. Post-break mean return is negative. Post-break rolling Sharpe: -0.23.

## Metrics
- **rolling_sharpe**: mean=0.2845, last=-0.2772, min=-8.0970, max=7.2557
- **half_life**: half_life=0.2010, theta=3.4492, r_squared=0.0010, slope=0.0318, se_slope=0.0080
- **oos_r_squared**: not computed

## Changepoint Detection
### Pelt
- Breakpoints on returns: [9200, 9240, 9270]
- Breakpoints on rolling Sharpe: [396, 591, 816, 1006, 1071, 1161, 1391, 1476, 1591, 1871, 1931, 2156, 2286, 2386, 2506, 2641, 2971, 3356, 3526, 3836, 4191, 4446, 4536, 4686, 4906, 5176, 5791, 6201, 6531, 6831, 6956, 7101, 7236, 7531, 7871, 8841, 9061, 9206, 9261, 9516, 9836, 10096, 10321, 10831, 11146, 11336, 11686, 12136, 12571, 12801, 13401, 13611, 14041, 14321, 14476, 14711]

### Cusum
- Breakpoints on returns: [6107]
- Breakpoints on rolling Sharpe: [2446]

### Bai Perron
- Breakpoints on returns: none detected
- Breakpoints on rolling Sharpe: [1470, 5176, 9115, 10828, 12840]

## Regime Detection
- Regime 0 (alpha_generating): mean=0.000168
- Regime 1 (decayed): mean=-0.000269

Transition matrix:
```
  0.9891  0.0109
  0.0337  0.9663
```

## Decay Detection
- **Decay detected**: Yes
- **Onset index**: 1821
