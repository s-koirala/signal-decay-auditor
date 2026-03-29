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

## Usage

### CLI

```bash
# Audit a single Fama-French factor
python -m src audit --factor HML

# Audit all FF3 factors
python -m src audit --all

# Audit custom CSV data
python -m src audit --csv data/raw/my_returns.csv --col signal_returns

# With options
python -m src audit --factor HML --start 2000-01-01 --end 2023-12-31 --verbose
```

### Python API

```python
from src.auditor import SignalDecayAuditor
from src.factors.data_loader import FactorDataLoader

# Load data
loader = FactorDataLoader()
ff3 = loader.load_fama_french(start="2000-01-01")

# Run audit
auditor = SignalDecayAuditor()
report = auditor.audit(ff3["HML"], name="HML (Value)")

print(report.summary)
# => "DEAD: Multiple detectors agree on break near index 4521. ..."

# Generate markdown report
auditor.generate_report(report, output_path="artifacts/reports/hml-audit-report.md")
```

### Verdict Classification

| Verdict | Criteria |
| ------- | -------- |
| **DEAD** | Multiple detectors agree on structural break + negative post-break returns |
| **DECAYING** | Break detected + sustained rolling Sharpe decline, or CUSUM decay onset |
| **ACTIVE** | No structural breaks detected, rolling Sharpe stable |

## Configuration

Default parameters are in `configs/default.yaml`. All values are empirically justified with citations to the originating literature. See `docs/decisions/` for parameter selection rationale.

## Testing

```bash
# Unit + synthetic integration tests (no network required)
pytest tests/ -k "not slow"

# Full suite including Fama-French live data
pytest tests/ -v
```

## Project Status

Phase 1: Core pipeline operational. Changepoint detectors (PELT, CUSUM, Bai-Perron, MOSUM), regime models (Markov, HMM, GARCH), and evaluation metrics (rolling Sharpe, half-life, OOS R-squared, Giacomini-White, Clark-West) implemented and tested against synthetic ground truth and live Fama-French factor data.
