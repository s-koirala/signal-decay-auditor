# Signal Decay Auditor — Open-Source Landscape

*Compiled 2026-03-28 via live web search*

---

## Core Dependencies (Tier 1)

### ruptures — Changepoint Detection
- **URL:** https://github.com/deepcharles/ruptures
- **Stars:** ~2,000 | **Last update:** Sep 2025 (v1.1.10)
- **Algorithms:** PELT, BinSeg, BottomUp, Window, DynP
- **Cost functions:** L1, L2, RBF, linear, normal, AR, custom
- **Role:** Primary offline changepoint detector. Feed rolling signal metrics into ruptures to detect structural breaks.

### statsmodels — Econometric Tests + Regime Switching
- **URL:** https://github.com/statsmodels/statsmodels
- **Stars:** ~12,000+ | **Maintenance:** Very active
- **Structural breaks:** CUSUM (breaks_cusumolsresid), RecursiveLS, Hansen test
- **Regime switching:** MarkovRegression, MarkovAutoregression
- **Role:** Formal econometric structural break tests + production-grade Markov-switching.

### arch — Volatility + Bootstrap
- **URL:** https://github.com/bashtage/arch
- **Stars:** ~1,300+ | **Maintenance:** Active (Kevin Sheppard)
- **Key:** GARCH family, Zivot-Andrews (structural break in unit root), stationary/circular/block bootstrap
- **Role:** Volatility regime detection, bootstrap inference for non-iid returns.

### hmmlearn — Hidden Markov Models
- **URL:** https://github.com/hmmlearn/hmmlearn
- **Stars:** ~3,400 | **Maintenance:** Active
- **Key:** GaussianHMM, GMMHMM, Viterbi decoding, Baum-Welch EM
- **Role:** Model signal effectiveness as hidden regime (working / decaying / dead).

### alphalens (archived) / alphalens-reloaded
- **URL:** https://github.com/quantopian/alphalens (original, 4,200 stars)
- **URL:** https://github.com/stefan-jansen/alphalens-reloaded (maintained fork)
- **Key:** IC analysis, IC decay, quantile returns, turnover, forward-return alpha decay
- **Role:** Built-in alpha decay measurement. IC over time directly quantifies signal decay.

---

## Data Access (Tier 1)

### pandas-datareader — Fama-French + FRED
- **URL:** https://github.com/pydata/pandas-datareader
- **Stars:** ~2,800 | **Maintenance:** Low (Fama-French reader still works)
- **Role:** Direct access to Ken French factor returns.

### fredapi — Macro Regime Context
- **URL:** https://github.com/mortada/fredapi
- **Stars:** ~700 | **Key:** FRED API wrapper. Requires free API key.
- **Role:** Macro indicators for regime context (yield curve, credit spreads, VIX).

### yfinance — Price Data
- **URL:** https://github.com/ranaroussi/yfinance
- **Stars:** ~13,000 | **Maintenance:** Active
- **Role:** Quick equity/ETF/index data for custom factor construction.

---

## Signal Evaluation (Tier 2)

### quantstats
- **URL:** https://github.com/ranaroussi/quantstats
- **Stars:** ~5,000 | Rolling Sharpe, Sortino, drawdown, tearsheets.

### empyrical-reloaded
- **Fork of:** quantopian/empyrical
- **Key:** `stability_of_timeseries` metric — R² of cumulative log returns vs. time.

### pyfolio-reloaded
- **Fork of:** quantopian/pyfolio
- **Key:** OOS vs. IS tearsheet, rolling factor exposures, Bayesian alpha posterior.

### linearmodels — Fama-MacBeth
- **URL:** https://github.com/bashtage/linearmodels
- **Stars:** ~900 | **Key:** Fama-MacBeth regressions for cross-sectional factor evaluation.

---

## Bayesian Changepoint Detection (Tier 2)

### bayesian_changepoint_detection
- **URL:** https://github.com/hildensia/bayesian_changepoint_detection
- **Stars:** ~759 | PyTorch-based. Online + offline BOCPD. GPU-accelerated.

### bocd (gwgundersen)
- **URL:** https://github.com/gwgundersen/bocd
- **Stars:** ~108 | Clean minimal BOCPD (Adams & MacKay 2007).

### changepoynt
- **URL:** https://github.com/Lucew/changepoynt
- **Stars:** ~35 | SST, density-ratio methods. JIT-compiled, sklearn-compatible.

### promised-ai/changepoint
- **URL:** https://github.com/promised-ai/changepoint
- **Stars:** ~35 | Rust + Python bindings. BOCPD + GP changepoint.

### Rbeast
- **URL:** https://github.com/zhaokg/Rbeast
- **Stars:** ~355 | Bayesian decomposition: trend + abrupt changes + seasonality.

---

## Reference / Educational (Tier 3)

### machine-learning-for-trading
- **URL:** https://github.com/stefan-jansen/machine-learning-for-trading
- **Stars:** ~16,700 | Ch4: alpha factor research + decay analysis.

### alphatools
- **URL:** https://github.com/marketneutral/alphatools
- **Stars:** ~456 | `decay_linear`, alpha formula parsing from "101 Formulaic Alphas."

### awesome-systematic-trading
- **URL:** https://github.com/wangzhe3224/awesome-systematic-trading
- **Stars:** ~3,700 | Curated list of systematic trading resources.

---

## Claude Agent Workflow Repos

### everything-claude-code (affaan-m)
- **URL:** https://github.com/affaan-m/everything-claude-code
- **Stars:** ~114,000 | 28 agents, 119 skills, 60 commands. Anthropic hackathon winner.
- **Role:** Agent orchestration patterns, memory persistence, multi-agent workflows.

### awesome-claude-code (hesreallyhim)
- **URL:** https://github.com/hesreallyhim/awesome-claude-code
- **Stars:** ~33,600 | Curated index of 75+ Claude Code repos.

### claude-code-best-practice (shanraisshan)
- **URL:** https://github.com/shanraisshan/claude-code-best-practice
- **Stars:** ~23,300 | Reference CLAUDE.md patterns, hooks, commands.

### awesome-claude-code-subagents (VoltAgent)
- **URL:** https://github.com/VoltAgent/awesome-claude-code-subagents
- **Stars:** ~15,500 | 100+ specialized subagents.

### claude-code-hooks-mastery (disler)
- **URL:** https://github.com/disler/claude-code-hooks-mastery
- **Stars:** ~3,400 | All 13 hook events with implementation examples.

### claude-code-system-prompts (Piebald-AI)
- **URL:** https://github.com/Piebald-AI/claude-code-system-prompts
- **Stars:** ~6,900 | Full system prompt extraction.

### claude-md-templates (abhishekray07)
- **URL:** https://github.com/abhishekray07/claude-md-templates
- **Stars:** ~109 | Three-tier CLAUDE.md architecture (global/project/local).

---

## Key Gap

**No existing "Signal Decay Auditor" exists on GitHub.** The building blocks exist (ruptures, alphalens, hmmlearn, statsmodels) but nobody has assembled them into a unified signal decay monitoring framework. This is greenfield.
