# Signal Decay Audit Report: Mkt-RF (Market)

## Overview
- **Signal**: Mkt-RF (Market)
- **Observations**: 15751
- **Date range**: 1963-07-01 00:00:00 to 2026-01-30 00:00:00
- **Verdict**: DECAYING: Decay onset detected at index 289 via CUSUM on rolling Sharpe.

## Metrics
- **rolling_sharpe**: mean=0.6460, last=0.6358, min=-3.6284, max=4.0270
- **half_life**: half_life=0.1653, theta=4.1922, r_squared=0.0002, slope=0.0151, se_slope=0.0080
- **oos_r_squared**: not computed

## Changepoint Detection
### Pelt
- Breakpoints on returns: [11375, 11410]
- Breakpoints on rolling Sharpe: [486, 671, 786, 921, 1026, 1151, 1471, 1581, 1776, 1871, 1941, 2046, 2321, 2386, 2421, 2586, 2751, 2891, 2961, 3036, 3396, 3551, 3706, 4311, 4516, 4556, 4676, 4811, 4921, 5091, 5176, 5416, 5621, 5826, 6106, 6361, 6501, 6676, 6811, 6931, 7076, 7196, 7736, 7986, 8046, 8296, 8561, 8641, 8826, 9411, 10051, 10186, 10301, 11201, 11646, 11736, 11801, 12131, 12361, 12551, 12901, 13121, 13376, 13471, 13656, 13746, 13916, 14196, 14261, 14521, 14731, 14806, 15066, 15221, 15516]

### Cusum
- Breakpoints on returns: none detected
- Breakpoints on rolling Sharpe: [1709]

### Bai Perron
- Breakpoints on returns: none detected
- Breakpoints on rolling Sharpe: [4887, 8833, 11649]

## Regime Detection
- Regime 0 (decayed): mean=-0.000741
- Regime 1 (alpha_generating): mean=0.000664

Transition matrix:
```
  0.9649  0.0351
  0.0128  0.9872
```

## Decay Detection
- **Decay detected**: Yes
- **Onset index**: 289
