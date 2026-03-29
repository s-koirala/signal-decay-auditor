# Signal Decay Audit Report: CMA (Investment)

## Overview
- **Signal**: CMA (Investment)
- **Observations**: 15751
- **Date range**: 1963-07-01 00:00:00 to 2026-01-30 00:00:00
- **Verdict**: DECAYING: Pelt detected structural break at index 1620. Rolling Sharpe declined from 0.7 to -0.3.

## Metrics
- **rolling_sharpe**: mean=0.5064, last=-0.0224, min=-3.2567, max=6.1321
- **half_life**: half_life=0.3306, theta=2.0965, r_squared=0.0151, slope=0.1229, se_slope=0.0079
- **oos_r_squared**: not computed

## Changepoint Detection
### Pelt
- Breakpoints on returns: [1620, 1710, 9330, 9445, 14700, 14840, 14880, 14980]
- Breakpoints on rolling Sharpe: [426, 536, 796, 896, 1071, 1151, 1321, 1516, 1676, 1926, 2081, 2431, 2611, 2746, 2901, 3016, 3156, 3316, 3406, 4061, 4271, 4556, 4611, 4906, 5091, 5156, 5291, 5406, 5821, 5891, 6071, 6716, 6811, 6896, 7066, 7186, 7246, 7586, 7701, 8016, 8151, 8916, 9401, 10001, 10111, 10286, 10361, 10446, 10726, 10851, 11051, 11121, 11371, 11746, 12086, 12311, 12401, 12646, 12721, 13021, 13241, 13511, 13676, 13911, 14186, 14511, 14726, 15001, 15186, 15386]

### Cusum
- Breakpoints on returns: [9444]
- Breakpoints on rolling Sharpe: [2782]

### Bai Perron
- Breakpoints on returns: [8631, 10208]
- Breakpoints on rolling Sharpe: [2439, 6715, 10359, 11747, 12718]

## Regime Detection
- Regime 0 (decayed): mean=0.000019
- Regime 1 (alpha_generating): mean=0.000365

Transition matrix:
```
  0.9901  0.0099
  0.0245  0.9755
```

## Decay Detection
- **Decay detected**: Yes
- **Onset index**: 448
