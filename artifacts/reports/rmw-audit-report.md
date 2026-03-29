# Signal Decay Audit Report: RMW (Profitability)

## Overview
- **Signal**: RMW (Profitability)
- **Observations**: 15751
- **Date range**: 1963-07-01 00:00:00 to 2026-01-30 00:00:00
- **Verdict**: DECAYING: Decay onset detected at index 294 via CUSUM on rolling Sharpe.

## Metrics
- **rolling_sharpe**: mean=0.6423, last=-0.7512, min=-4.4670, max=6.0819
- **half_life**: half_life=0.3324, theta=2.0852, r_squared=0.0154, slope=0.1243, se_slope=0.0079
- **oos_r_squared**: not computed

## Changepoint Detection
### Pelt
- Breakpoints on returns: [9170, 9235, 9265, 9310, 9890, 9920]
- Breakpoints on rolling Sharpe: [431, 626, 886, 1071, 1136, 1301, 1506, 1536, 1591, 1646, 1731, 1956, 2061, 2321, 2466, 2551, 2606, 2696, 2971, 3051, 3161, 3536, 3701, 3976, 4271, 4546, 5166, 5236, 5356, 5531, 5816, 5981, 6551, 6811, 6896, 7266, 7341, 7411, 7491, 7571, 7676, 7746, 7781, 8016, 8321, 8426, 8596, 8746, 8946, 9151, 9256, 9411, 9471, 9951, 10026, 10086, 10276, 10326, 10586, 10836, 11206, 11566, 12086, 12371, 12546, 12656, 12781, 13241, 13386, 14696, 14856, 15386, 15596]

### Cusum
- Breakpoints on returns: [9254]
- Breakpoints on rolling Sharpe: [5457]

### Bai Perron
- Breakpoints on returns: none detected
- Breakpoints on rolling Sharpe: [2493, 5180, 8749, 12360]

## Regime Detection
- Regime 0 (alpha_generating): mean=0.000239
- Regime 1 (decayed): mean=0.000088

Transition matrix:
```
  0.9817  0.0183
  0.0068  0.9932
```

## Decay Detection
- **Decay detected**: Yes
- **Onset index**: 294
