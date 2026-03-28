# Claude Agent Workflow — Signal Decay Auditor

## Purpose
Guardrails and checklists for Claude agents working on this project.

---

## Pre-Implementation Checklist

Before writing any statistical method:

1. [ ] Check `research/literature/` for existing paper summary
2. [ ] Identify the canonical reference (paper, textbook chapter)
3. [ ] Confirm the test statistic formula against the source
4. [ ] Identify existing Python implementation (if any) and its limitations
5. [ ] Define synthetic test case with known analytical solution
6. [ ] Confirm parameter selection strategy (no magic numbers)

## Implementation Protocol

1. Write the detector/test with explicit docstring citing the paper
2. Include formula for the test statistic in docstring (LaTeX or plain text)
3. All tunable parameters must have:
   - A default justified by the literature (with citation)
   - OR a data-driven selection mechanism (cross-validation, IC, grid search)
4. Log parameter selection rationale in `docs/decisions/`

## Verification Protocol

1. Run against synthetic data with known break locations
2. Compare output against:
   - R reference implementation (if available via rpy2 or manual comparison)
   - Analytical solution (if closed-form exists)
   - Published table/figure from the original paper
3. Bootstrap confidence intervals on break date estimates
4. Document any divergence from canonical method in `docs/decisions/`

## Testing Requirements

- **Unit tests** (`tests/unit/`): Each detector tested against >=3 synthetic scenarios:
  1. No break (null case — verify correct size)
  2. Single break at known location
  3. Multiple breaks at known locations
- **Integration tests** (`tests/integration/`): Full pipeline from data ingestion to report generation

## Data Integrity

- Raw data never modified — transformations produce new files in `data/processed/`
- Log checksums on data ingestion (SHA-256)
- Document data source, access date, and any known issues
- Fama-French data: verify against Ken French website totals

## Evidence Templates

### Decision Log Entry (`docs/decisions/YYYY-MM-DD-topic.md`)
```
# Decision: [topic]
Date: YYYY-MM-DD

## Context
[What problem required a decision]

## Options Considered
1. [Option A] — [pros/cons]
2. [Option B] — [pros/cons]

## Decision
[What was chosen and why]

## Evidence
- [Citation 1]
- [Citation 2]
- [Empirical result if applicable]

## Consequences
[What this implies for the rest of the system]
```

### Method Validation Entry (`research/methods/method-name.md`)
```
# Method: [name]
Reference: [full citation]

## Test Statistic
[Formula]

## Assumptions
[List explicitly]

## Implementation
- Package: [name] or custom
- Key parameters: [list with justification]

## Verification
- Synthetic test: [description and result]
- Reference comparison: [R package / analytical solution]
- Discrepancies: [none / describe]
```

## Agent Routing

| Task | Directive to Load |
|------|-------------------|
| New statistical method | `~/Documents/Agent Directives/prompts/statistical-modeling/statistical-analysis-directive.md` |
| Code review / audit | `~/Documents/Agent Directives/prompts/operations/audit-qa-qc-qi-directive.md` |
| Multi-agent orchestration | `~/Documents/Agent Directives/prompts/templates/multi-agent-template.md` |
| New directive creation | `~/Documents/Agent Directives/prompts/templates/base-template.md` |

## Key Repos for Reference

- **everything-claude-code** (https://github.com/affaan-m/everything-claude-code) — agent orchestration patterns
- **claude-code-best-practice** (https://github.com/shanraisshan/claude-code-best-practice) — CLAUDE.md and hook patterns
- **claude-code-hooks-mastery** (https://github.com/disler/claude-code-hooks-mastery) — hook event implementations
- **claude-md-templates** (https://github.com/abhishekray07/claude-md-templates) — three-tier CLAUDE.md structure
