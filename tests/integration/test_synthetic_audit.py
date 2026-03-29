"""Integration test: SignalDecayAuditor on synthetic data with known ground truth.

Validates that the full pipeline correctly detects (or does not detect) structural
breaks in synthetic series where the true break locations and regime structure are
known a priori.

Does NOT require network access.  Runs on every CI build.
"""

import numpy as np
import pytest

from src.auditor import AuditReport, SignalDecayAuditor
from src.utils.synthetic import generate_factor_with_decay, generate_mean_shift


# ---------------------------------------------------------------------------
# Shared auditor fixture (fast configuration for synthetic data)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def auditor() -> SignalDecayAuditor:
    return SignalDecayAuditor(
        detectors=["pelt", "cusum", "bai_perron"],
        regime_model=None,  # skip regime for speed
        metrics=["rolling_sharpe", "half_life", "oos_r_squared"],
        rolling_window=60,
        min_segment_size=30,
    )


# ===================================================================
# 1. Factor lifecycle: alpha -> decay -> dead
# ===================================================================

class TestFactorLifecycleAudit:
    """Audit a synthetic factor with known decay onset."""

    @pytest.fixture(scope="class")
    def lifecycle_report(self, auditor: SignalDecayAuditor):
        series, meta = generate_factor_with_decay(
            alpha_period=1000,
            decay_period=500,
            dead_period=500,
            alpha_mean=0.002,
            decay_end_mean=0.0,
            dead_mean=-0.001,
            std=0.005,
            seed=42,
        )
        return auditor.audit(series, name="synthetic_lifecycle"), meta

    def test_detects_decay(self, lifecycle_report):
        report, _ = lifecycle_report
        assert report.decay_detected is True, (
            "Auditor should detect decay in a factor lifecycle series"
        )

    def test_verdict_not_active(self, lifecycle_report):
        report, _ = lifecycle_report
        assert not report.summary.startswith("ACTIVE"), (
            f"Verdict should not be ACTIVE for a decaying factor, "
            f"got: {report.summary}"
        )

    def test_changepoint_near_true_break(self, lifecycle_report):
        report, meta = lifecycle_report
        true_break = meta["break_points"][0]  # 1000

        all_breaks = []
        for det_result in report.changepoint_results.values():
            all_breaks.extend(det_result.get("on_returns", []))

        assert len(all_breaks) > 0, (
            "At least one detector should find a breakpoint"
        )
        closest = min(all_breaks, key=lambda b: abs(b - true_break))
        assert abs(closest - true_break) <= 200, (
            f"Closest break {closest} is >200 from true break {true_break}. "
            f"All detected: {all_breaks}"
        )


# ===================================================================
# 2. Pure noise: no structural break
# ===================================================================

class TestNoBreakAudit:
    """Audit pure noise — no structural breaks on returns."""

    @pytest.fixture(scope="class")
    def noise_report(self, auditor: SignalDecayAuditor):
        series, meta = generate_mean_shift(
            n=2000,
            break_points=[],
            means=[0.0],
            std=0.01,
            seed=99,
        )
        return auditor.audit(series, name="pure_noise"), meta

    def test_no_breakpoints_on_returns(self, noise_report):
        """No detector should find structural breaks on pure noise returns."""
        report, _ = noise_report
        for det_name, det_result in report.changepoint_results.items():
            bp_returns = det_result.get("on_returns", [])
            assert len(bp_returns) == 0, (
                f"{det_name} found {len(bp_returns)} breakpoints on pure noise: "
                f"{bp_returns}"
            )

    def test_report_well_formed(self, noise_report):
        """Report should be structurally valid."""
        report, _ = noise_report
        assert isinstance(report, AuditReport)
        assert report.n_observations == 2000
        assert report.summary != ""


# ===================================================================
# 3. Strong mean shift
# ===================================================================

class TestStrongMeanShiftAudit:
    """Audit a strong mean shift — most detectors should find it."""

    @pytest.fixture(scope="class")
    def shift_report(self, auditor: SignalDecayAuditor):
        series, meta = generate_mean_shift(
            n=2000,
            break_points=[1000],
            means=[0.005, -0.005],
            std=0.01,
            seed=42,
        )
        return auditor.audit(series, name="strong_mean_shift"), meta

    def test_breakpoints_detected(self, shift_report):
        report, _ = shift_report
        n_detectors_with_breaks = sum(
            1 for det_result in report.changepoint_results.values()
            if len(det_result.get("on_returns", [])) > 0
        )
        assert n_detectors_with_breaks >= 2, (
            f"At least 2/3 detectors should find the strong break, "
            f"only {n_detectors_with_breaks} did"
        )

    def test_break_near_true_location(self, shift_report):
        report, meta = shift_report
        true_break = meta["break_points"][0]

        all_breaks = []
        for det_result in report.changepoint_results.values():
            all_breaks.extend(det_result.get("on_returns", []))

        assert len(all_breaks) > 0, "Should detect at least one break"
        closest = min(all_breaks, key=lambda b: abs(b - true_break))
        assert abs(closest - true_break) <= 50, (
            f"Closest break {closest} too far from true {true_break}. "
            f"All: {all_breaks}"
        )
