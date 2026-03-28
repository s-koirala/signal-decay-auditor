# Signal Decay Auditor — CLAUDE.md

## Project Purpose
Statistical framework for detecting structural breaks in systematic trading factor returns.
Determines whether alpha signals have been arbitraged away vs. temporarily suppressed.

## Domain
Quantitative finance, time series econometrics, changepoint detection, statistical arbitrage.

## Core Methods
- CUSUM / MOSUM changepoint detection on rolling Sharpe ratios
- Bai-Perron multiple structural break tests
- Out-of-sample predictive R-squared tracking
- Giacomini-White conditional predictive ability tests
- Hidden Markov Models for regime classification
- Bootstrap inference for uncertainty quantification

## Directory Structure
```
signal-decay-auditor/
├── CLAUDE.md                  # This file — project directives for Claude agents
├── README.md                  # Project overview and usage
├── requirements.txt           # Pinned Python dependencies
├── configs/                   # YAML/JSON configuration files
│   └── default.yaml           # Default parameters (all empirically justified)
├── src/
│   ├── detectors/             # Changepoint detection algorithms
│   ├── factors/               # Factor data ingestion and construction
│   ├── evaluation/            # Signal quality metrics and scoring
│   └── utils/                 # Shared utilities (stats, IO, plotting)
├── data/
│   ├── raw/                   # Untouched source data (Ken French, AQR, etc.)
│   ├── processed/             # Cleaned, aligned factor returns
│   └── cache/                 # Intermediate computations
├── research/
│   ├── literature/            # Paper summaries, BibTeX references
│   ├── methods/               # Method validation notes
│   └── benchmarks/            # Known analytical solutions for verification
├── docs/
│   ├── methodology/           # Statistical methodology documentation
│   ├── architecture/          # System design docs
│   └── decisions/             # Decision logs with citations
├── tests/
│   ├── unit/                  # Unit tests for individual components
│   └── integration/           # End-to-end pipeline tests
├── notebooks/                 # Exploratory analysis (numbered, dated)
└── artifacts/
    ├── models/                # Fitted model objects
    ├── reports/               # Generated audit reports
    └── logs/                  # Execution logs with timestamps
```

## Execution Protocol
1. Read this file first on every session.
2. Check `research/literature/` for existing method validation before implementing.
3. All parameter choices require empirical justification — no magic numbers.
4. Cross-check implementations against cited papers.
5. Unit test every detector against known synthetic changepoints.
6. Log methodology decisions in `docs/decisions/` with ISO 8601 timestamps.

## Evidence Hierarchy
1. Peer-reviewed econometrics/statistics literature
2. Package documentation (ruptures, statsmodels, arch)
3. Professional standards (backtesting best practices: Bailey et al., Lopez de Prado)
4. CrossValidated, StackOverflow, GitHub issues
5. Reproduce — do not paraphrase

## Parameter Selection
- Zero arbitrary thresholds
- Justify via: grid search, cross-validation, information criteria (AIC/BIC/WAIC), bootstrap CIs
- Document rationale in `docs/decisions/`

## Verification
- Synthetic data with known break locations for detector validation
- Compare against R packages (strucchange, bcp) as reference implementations
- Numerical results checked against analytical solutions where available
- Flag any divergence from canonical method

## Key References (to be expanded in research/literature/)
- Bai & Perron (1998, 2003) — multiple structural breaks
- Page (1954), Hinkley (1971) — CUSUM foundations
- Giacomini & White (2006) — conditional predictive ability
- Andrews (1993) — optimal changepoint tests
- Hamilton (1989) — regime-switching models
- McLean & Pontiff (2016) — factor decay post-publication
- Harvey, Liu & Zhu (2016) — multiple testing in factor discovery

## Agent Workflow
- Reference `~/Documents/Agent Directives/prompts/` for prompt templates
- Use `statistical-modeling/statistical-analysis-directive.md` for method implementation
- Use `operations/audit-qa-qc-qi-directive.md` for verification passes
- Use `validation/` templates for test case design

## Naming Conventions
- Files: kebab-case (e.g., `bai-perron-detector.py`)
- Classes: PascalCase
- Functions/variables: snake_case
- Dates: ISO 8601 (YYYY-MM-DD)
- Notebooks: `NNN-description.ipynb` (e.g., `001-exploratory-factor-returns.ipynb`)

## Reproducibility
- Random seeds documented in configs
- Data checksums logged on ingestion
- Environment pinned in requirements.txt
- Git-tracked parameter rationale
