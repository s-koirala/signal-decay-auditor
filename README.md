# Signal Decay Auditor

Statistical framework for detecting structural breaks in systematic trading factor returns. Determines whether alpha signals have been arbitraged away versus temporarily suppressed.

## Problem Statement

Every systematic trading signal decays. Most funds discover their alpha is gone from drawdowns, not from measurement. This framework provides real-time, statistically rigorous detection of when factor returns undergo structural breaks — enabling proactive signal management rather than reactive loss recognition.

## Methods

| Method | Purpose | Reference |
|--------|---------|-----------|
| CUSUM / MOSUM | Sequential changepoint detection | Page (1954), Chu et al. (1996) |
| Bai-Perron | Multiple structural break estimation | Bai & Perron (1998, 2003) |
| Giacomini-White | Conditional predictive ability testing | Giacomini & White (2006) |
| HMM regime detection | Latent state classification | Hamilton (1989) |
| Rolling predictive R² | Out-of-sample signal strength tracking | Campbell & Thompson (2008) |
| Bootstrap inference | Uncertainty quantification on break dates/magnitudes | Hansen (2000) |

## Data Sources

- **Ken French Data Library** — Fama-French factor returns (daily, monthly)
- **AQR Data Sets** — Momentum, value, carry, defensive factors
- **FRED** — Macro regime indicators
- **Personal brokerage / crypto APIs** — Live signal testing (24/7 for crypto)

## Installation

```bash
cd signal-decay-auditor
pip install -r requirements.txt
```

## Project Status

Phase 0: Environment setup and literature review.
