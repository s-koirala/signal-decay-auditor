"""Integration test: Fama-French factor audit via SignalDecayAuditor.

Downloads Fama-French 3-factor daily data from Ken French's website,
extracts HML (value), SMB (size), and Mkt-RF (market) factor returns,
then runs the full SignalDecayAuditor pipeline on each.

Marked as slow — skip with ``pytest -m "not slow"``.

References:
    - Fama & French (1993) — 3-factor model
    - McLean & Pontiff (2016) — post-publication factor decay
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.factors.data_loader import FactorDataLoader
from src.auditor import SignalDecayAuditor, AuditReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.slow,
]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "reports"


@pytest.fixture(scope="module")
def ff3_data() -> pd.DataFrame:
    """Load Fama-French 3-factor daily data.

    Skips the entire module if the download fails (network error,
    service unavailable, etc.).
    """
    loader = FactorDataLoader()
    try:
        df = loader.load_fama_french(
            dataset="F-F_Research_Data_Factors_daily",
            start="1963-07-01",
        )
    except Exception as exc:
        pytest.skip(f"Could not download Fama-French data: {exc}")

    # Basic sanity checks on the downloaded data
    assert len(df) > 10_000, f"Expected >10k rows, got {len(df)}"
    for col in ["Mkt-RF", "SMB", "HML", "RF"]:
        assert col in df.columns, f"Missing expected column: {col}"

    # Confirm decimal scaling: daily returns should rarely exceed 20%
    for col in ["Mkt-RF", "SMB", "HML"]:
        assert df[col].abs().max() < 0.50, (
            f"{col} max abs return {df[col].abs().max():.4f} looks too large "
            f"for decimal-scaled daily returns — possible percentage issue"
        )

    return df


@pytest.fixture(scope="module")
def auditor() -> SignalDecayAuditor:
    """Create a SignalDecayAuditor with default configuration."""
    return SignalDecayAuditor(
        detectors=["pelt", "cusum", "bai_perron"],
        regime_model="markov",
        metrics=["rolling_sharpe", "half_life", "oos_r_squared"],
        rolling_window=252,
        min_segment_size=30,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_audit_and_print(
    auditor: SignalDecayAuditor,
    series: pd.Series,
    name: str,
) -> AuditReport:
    """Run audit, print summary, return report."""
    report = auditor.audit(series.dropna(), name=name)

    # Print summary to stdout for visibility in pytest -v output
    print(f"\n{'='*60}")
    print(f"  AUDIT RESULTS: {name}")
    print(f"{'='*60}")
    print(f"  Observations : {report.n_observations}")
    if report.date_range:
        print(f"  Date range   : {report.date_range[0].date()} to {report.date_range[1].date()}")
    print(f"  Verdict      : {report.summary}")
    print(f"  Decay detected: {report.decay_detected}")
    if report.decay_onset is not None:
        print(f"  Decay onset  : index {report.decay_onset}")

    # Changepoint summary
    for det_name, det_result in report.changepoint_results.items():
        bp_ret = det_result.get("on_returns", [])
        bp_sh = det_result.get("on_sharpe", [])
        print(f"  {det_name:12s} | returns breaks: {len(bp_ret):2d} | sharpe breaks: {len(bp_sh):2d}")

    # Regime summary
    if report.regime_results:
        rm = report.regime_results
        means = rm.get("regime_means", {})
        labels = rm.get("regime_labels", {})
        for rid in sorted(means.keys()):
            print(f"  Regime {rid} ({labels.get(rid, '?'):8s}): mean={means[rid]:.6f}")

    # Key metrics
    hl = report.metrics.get("half_life")
    if isinstance(hl, dict):
        print(f"  Half-life    : {hl.get('half_life', 'N/A')}")
    oos = report.metrics.get("oos_r_squared")
    if oos is not None:
        print(f"  OOS R-squared: {oos:.6f}")

    print(f"{'='*60}\n")
    return report


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHMLAudit:
    """Audit the HML (value) factor — expected to show decay post-2000."""

    def test_hml_audit_runs(self, ff3_data: pd.DataFrame, auditor: SignalDecayAuditor):
        """Full audit pipeline completes on HML without errors."""
        hml = ff3_data["HML"]
        report = _run_audit_and_print(auditor, hml, "HML (Value)")

        # The report must be well-formed
        assert isinstance(report, AuditReport)
        assert report.name == "HML (Value)"
        assert report.n_observations > 10_000
        assert report.summary  # non-empty verdict string

    def test_hml_report_generation(self, ff3_data: pd.DataFrame, auditor: SignalDecayAuditor):
        """Generate and save markdown report for HML."""
        hml = ff3_data["HML"]
        report = auditor.audit(hml.dropna(), name="HML (Value)")

        output_path = str(ARTIFACTS_DIR / "hml-audit-report.md")
        md_text = auditor.generate_report(report, output_path=output_path)

        assert Path(output_path).exists(), "Report file was not written"
        assert len(md_text) > 200, "Report seems too short"
        assert "HML" in md_text
        assert "## Changepoint Detection" in md_text
        assert "## Decay Detection" in md_text

        print(f"HML report saved to: {output_path}")


class TestSMBAudit:
    """Audit the SMB (size) factor for comparison."""

    def test_smb_audit_runs(self, ff3_data: pd.DataFrame, auditor: SignalDecayAuditor):
        """Full audit pipeline completes on SMB without errors."""
        smb = ff3_data["SMB"]
        report = _run_audit_and_print(auditor, smb, "SMB (Size)")

        assert isinstance(report, AuditReport)
        assert report.name == "SMB (Size)"
        assert report.n_observations > 10_000
        assert report.summary


class TestMktRFAudit:
    """Audit the Mkt-RF (market excess return) factor for comparison."""

    def test_mktrf_audit_runs(self, ff3_data: pd.DataFrame, auditor: SignalDecayAuditor):
        """Full audit pipeline completes on Mkt-RF without errors."""
        mktrf = ff3_data["Mkt-RF"]
        report = _run_audit_and_print(auditor, mktrf, "Mkt-RF (Market)")

        assert isinstance(report, AuditReport)
        assert report.name == "Mkt-RF (Market)"
        assert report.n_observations > 10_000
        assert report.summary


class TestCrossFactorComparison:
    """Compare audit results across all three factors."""

    def test_all_factors_audited(self, ff3_data: pd.DataFrame, auditor: SignalDecayAuditor):
        """Run all three factors and print a comparative summary."""
        factors = {
            "HML (Value)": ff3_data["HML"].dropna(),
            "SMB (Size)": ff3_data["SMB"].dropna(),
            "Mkt-RF (Market)": ff3_data["Mkt-RF"].dropna(),
        }

        reports = auditor.audit_multiple(factors)

        print(f"\n{'='*70}")
        print("  CROSS-FACTOR COMPARISON")
        print(f"{'='*70}")
        print(f"  {'Factor':<20s} {'Verdict':<50s}")
        print(f"  {'-'*18:<20s} {'-'*48:<50s}")

        for name, report in reports.items():
            verdict_short = report.summary.split(":")[0] if ":" in report.summary else report.summary
            print(f"  {name:<20s} {report.summary}")

        print(f"{'='*70}\n")

        # All three should produce valid reports
        assert len(reports) == 3
        for name, report in reports.items():
            assert report.n_observations > 10_000, f"{name} has too few observations"
            assert report.summary, f"{name} has empty verdict"
