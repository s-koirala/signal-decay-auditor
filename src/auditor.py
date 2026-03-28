"""Main orchestration module for the Signal Decay Auditor.

Ties together changepoint detectors, regime models, and evaluation metrics
into a single audit pipeline that produces structured diagnostic reports
for factor return series.

Usage::

    from src.auditor import SignalDecayAuditor

    auditor = SignalDecayAuditor()
    report = auditor.audit(returns_series, name="HML")
    print(report.summary)
    auditor.generate_report(report, output_path="artifacts/reports/hml_audit.md")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.detectors.changepoint import (
    BaiPerronDetector,
    CUSUMDetector,
    MOSUMDetector,
    PELTDetector,
)
from src.detectors.regime import (
    GARCHRegimeDetector,
    HMMRegimeDetector,
    MarkovRegimeDetector,
    label_signal_state,
)
from src.evaluation.metrics import (
    detect_decay_onset,
    oos_r_squared,
    rolling_half_life,
    rolling_sharpe,
    signal_half_life,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass: AuditReport
# ---------------------------------------------------------------------------

@dataclass
class AuditReport:
    """Structured output from a signal decay audit.

    Attributes
    ----------
    name : str
        Human-readable label for the audited signal.
    n_observations : int
        Total number of observations in the input series.
    date_range : tuple or None
        ``(start, end)`` timestamps if the input has a DatetimeIndex,
        otherwise None.
    changepoint_results : dict
        Mapping from detector name to a dict with keys:
        ``"breakpoints"`` (list of int indices detected on raw returns),
        ``"on_returns"`` (same as breakpoints, for explicit clarity),
        ``"on_sharpe"`` (breakpoints detected on rolling Sharpe series).
    regime_results : dict or None
        If a regime model was fitted, contains:
        ``"regime_means"``, ``"regime_labels"``, ``"transition_matrix"``,
        ``"smoothed_probs"``.  None if no regime model was configured.
    metrics : dict
        Mapping from metric name to its computed value (scalar) or
        Series (for rolling metrics).
    decay_detected : bool
        True if the audit pipeline detected evidence of signal decay.
    decay_onset : int or None
        Positional index of the estimated decay onset, or None if no
        decay was detected.
    summary : str
        One-line plain-language verdict (see :func:`_verdict`).
    """

    name: str
    n_observations: int
    date_range: Optional[Tuple] = None
    changepoint_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    regime_results: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    decay_detected: bool = False
    decay_onset: Optional[int] = None
    summary: str = ""


# ---------------------------------------------------------------------------
# Detector registry
# ---------------------------------------------------------------------------

_DETECTOR_MAP = {
    "pelt": PELTDetector,
    "cusum": CUSUMDetector,
    "bai_perron": BaiPerronDetector,
    "mosum": MOSUMDetector,
}

_REGIME_MAP = {
    "markov": MarkovRegimeDetector,
    "hmm": HMMRegimeDetector,
    "garch": GARCHRegimeDetector,
}


# ---------------------------------------------------------------------------
# Helper: verdict generation
# ---------------------------------------------------------------------------

def _verdict(report: AuditReport) -> str:
    """Generate a plain-language summary from an AuditReport.

    Classification logic:

    * **DEAD** -- Multiple detectors agree on a structural break, and
      the post-break mean return is negative or near zero.
    * **DECAYING** -- At least one detector found a break and rolling
      Sharpe shows a sustained decline.
    * **ACTIVE** -- No structural breaks detected and rolling Sharpe
      remains stable.

    Parameters
    ----------
    report : AuditReport
        A completed audit report (all fields populated).

    Returns
    -------
    str
        A one-line verdict string.
    """
    # Collect all breakpoints across detectors (on returns)
    all_breaks: List[int] = []
    for det_name, det_result in report.changepoint_results.items():
        all_breaks.extend(det_result.get("on_returns", []))

    n_detectors_with_breaks = sum(
        1 for det_result in report.changepoint_results.values()
        if len(det_result.get("on_returns", [])) > 0
    )

    # Extract rolling Sharpe for trend analysis
    rolling_sharpe_series = report.metrics.get("rolling_sharpe")
    sharpe_start = np.nan
    sharpe_end = np.nan
    if rolling_sharpe_series is not None:
        if isinstance(rolling_sharpe_series, pd.Series):
            valid = rolling_sharpe_series.dropna()
        else:
            valid_mask = ~np.isnan(rolling_sharpe_series)
            valid = rolling_sharpe_series[valid_mask] if hasattr(rolling_sharpe_series, '__getitem__') else np.array([])
            valid = pd.Series(valid)

        if len(valid) >= 2:
            # Use first/last quartile medians for robust start/end estimates
            q = max(1, len(valid) // 4)
            sharpe_start = float(np.nanmedian(valid.iloc[:q]))
            sharpe_end = float(np.nanmedian(valid.iloc[-q:]))

    # Determine if post-break returns are negative
    post_break_negative = False
    if all_breaks:
        # Use the median break location as reference
        median_break = int(np.median(all_breaks))
        if median_break < report.n_observations:
            # Attempt to infer post-break mean from regime results
            if report.regime_results is not None:
                regime_means = report.regime_results.get("regime_means", {})
                if regime_means:
                    min_mean = min(regime_means.values()) if isinstance(regime_means, dict) else np.nan
                    post_break_negative = min_mean <= 0.0

    # Classification
    if n_detectors_with_breaks >= 2 and post_break_negative:
        # Multiple detectors agree + negative post-break returns
        consensus_break = int(np.median(all_breaks))
        sharpe_info = ""
        if not np.isnan(sharpe_end):
            sharpe_info = f" Post-break rolling Sharpe: {sharpe_end:.2f}."
        return (
            f"DEAD: Multiple detectors agree on break near index {consensus_break}. "
            f"Post-break mean return is negative.{sharpe_info}"
        )

    if n_detectors_with_breaks >= 1 and not np.isnan(sharpe_start) and not np.isnan(sharpe_end):
        if sharpe_end < sharpe_start - 0.3:
            # Meaningful Sharpe decline
            det_names = [
                name for name, res in report.changepoint_results.items()
                if len(res.get("on_returns", [])) > 0
            ]
            primary_det = det_names[0] if det_names else "detector"
            primary_breaks = report.changepoint_results.get(
                primary_det, {}
            ).get("on_returns", [])
            break_loc = primary_breaks[0] if primary_breaks else "unknown"
            return (
                f"DECAYING: {primary_det.replace('_', ' ').title()} detected structural break "
                f"at index {break_loc}. Rolling Sharpe declined from "
                f"{sharpe_start:.1f} to {sharpe_end:.1f}."
            )

    if report.decay_detected and report.decay_onset is not None:
        return (
            f"DECAYING: Decay onset detected at index {report.decay_onset} "
            f"via CUSUM on rolling Sharpe."
        )

    # Active
    sharpe_info = ""
    if not np.isnan(sharpe_end):
        sharpe_info = f" Rolling Sharpe stable at {sharpe_end:.1f}."
    return f"ACTIVE: No structural breaks detected.{sharpe_info}"


# ---------------------------------------------------------------------------
# Main class: SignalDecayAuditor
# ---------------------------------------------------------------------------

class SignalDecayAuditor:
    """Orchestrates the full signal decay audit pipeline.

    Parameters
    ----------
    detectors : list of str, default ``["pelt", "cusum", "bai_perron"]``
        Names of changepoint detectors to run.  Valid names:
        ``"pelt"``, ``"cusum"``, ``"bai_perron"``, ``"mosum"``.
    regime_model : str or None, default ``"markov"``
        Regime detector to use: ``"markov"``, ``"hmm"``, or ``None``
        to skip regime detection.
    metrics : list of str, default ``["rolling_sharpe", "half_life", "oos_r_squared"]``
        Metric names to compute during the audit.
    rolling_window : int, default 252
        Window size for rolling metric computations.  252 trading days
        corresponds to approximately one calendar year.
    min_segment_size : int, default 30
        Minimum segment size passed to changepoint detectors.  30
        observations ensures stable within-segment estimation.
    """

    def __init__(
        self,
        detectors: Optional[List[str]] = None,
        regime_model: Optional[str] = "markov",
        metrics: Optional[List[str]] = None,
        rolling_window: int = 252,
        min_segment_size: int = 30,
    ) -> None:
        self.detector_names = detectors or ["pelt", "cusum", "bai_perron"]
        self.regime_model = regime_model
        self.metric_names = metrics or ["rolling_sharpe", "half_life", "oos_r_squared"]
        self.rolling_window = rolling_window
        self.min_segment_size = min_segment_size

        # Validate detector names
        for name in self.detector_names:
            if name not in _DETECTOR_MAP:
                raise ValueError(
                    f"Unknown detector '{name}'. "
                    f"Valid options: {sorted(_DETECTOR_MAP.keys())}"
                )

        # Validate regime model
        if self.regime_model is not None and self.regime_model not in _REGIME_MAP:
            raise ValueError(
                f"Unknown regime model '{self.regime_model}'. "
                f"Valid options: {sorted(_REGIME_MAP.keys())} or None"
            )

    # ------------------------------------------------------------------
    # Core audit
    # ------------------------------------------------------------------

    def audit(
        self,
        returns: Union[np.ndarray, pd.Series],
        name: str = "signal",
    ) -> AuditReport:
        """Run the full audit pipeline on a single return series.

        Steps:
            1. Compute rolling performance metrics (Sharpe, half-life).
            2. Run all configured changepoint detectors on raw returns
               AND on the rolling Sharpe series.
            3. Run regime detection if configured.
            4. Compute decay onset detection via CUSUM on rolling Sharpe.
            5. Aggregate results into a structured ``AuditReport``.

        Parameters
        ----------
        returns : array-like, 1D
            Factor return series.  May be a numpy array or pandas Series.
            Must not contain NaN values.
        name : str, default ``"signal"``
            Human-readable label for the signal being audited.

        Returns
        -------
        AuditReport
            Structured audit results.
        """
        logger.info("Starting audit for '%s'", name)

        # --- Coerce input ---
        if isinstance(returns, pd.Series):
            returns_arr = returns.values.astype(np.float64)
            has_dt_index = isinstance(returns.index, pd.DatetimeIndex)
            date_range = (
                (returns.index.min(), returns.index.max())
                if has_dt_index and len(returns) > 0
                else None
            )
        else:
            returns_arr = np.asarray(returns, dtype=np.float64)
            has_dt_index = False
            date_range = None

        n = len(returns_arr)
        logger.info("Series '%s': %d observations", name, n)

        # --- Step 1: Compute metrics ---
        computed_metrics: Dict[str, Any] = {}

        if "rolling_sharpe" in self.metric_names:
            try:
                rs = rolling_sharpe(returns, window=self.rolling_window)
                computed_metrics["rolling_sharpe"] = rs
                logger.info("Computed rolling Sharpe for '%s'", name)
            except Exception as exc:
                logger.warning(
                    "Failed to compute rolling Sharpe for '%s': %s", name, exc
                )
                computed_metrics["rolling_sharpe"] = None

        if "half_life" in self.metric_names:
            try:
                hl = signal_half_life(returns_arr)
                computed_metrics["half_life"] = hl
                logger.info(
                    "Signal half-life for '%s': %.1f periods",
                    name,
                    hl.get("half_life", np.inf),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to compute half-life for '%s': %s", name, exc
                )
                computed_metrics["half_life"] = None

        if "rolling_half_life" in self.metric_names:
            try:
                rhl = rolling_half_life(returns, window=self.rolling_window)
                computed_metrics["rolling_half_life"] = rhl
            except Exception as exc:
                logger.warning(
                    "Failed to compute rolling half-life for '%s': %s",
                    name, exc,
                )
                computed_metrics["rolling_half_life"] = None

        if "oos_r_squared" in self.metric_names:
            try:
                # Use lagged returns as a naive predictor (AR(1) benchmark)
                if n > 1:
                    predictor = np.empty_like(returns_arr)
                    predictor[0] = np.nan
                    predictor[1:] = returns_arr[:-1]
                    oos_r2 = oos_r_squared(returns_arr, predictor)
                    computed_metrics["oos_r_squared"] = oos_r2
                    logger.info(
                        "OOS R-squared for '%s': %.4f", name, oos_r2
                    )
                else:
                    computed_metrics["oos_r_squared"] = np.nan
            except Exception as exc:
                logger.warning(
                    "Failed to compute OOS R-squared for '%s': %s", name, exc
                )
                computed_metrics["oos_r_squared"] = None

        # --- Step 2: Changepoint detection ---
        changepoint_results: Dict[str, Dict[str, Any]] = {}

        # Prepare rolling Sharpe for changepoint detection (drop NaN)
        sharpe_series = computed_metrics.get("rolling_sharpe")
        sharpe_clean: Optional[np.ndarray] = None
        sharpe_offset: int = 0
        if sharpe_series is not None:
            if isinstance(sharpe_series, pd.Series):
                valid_mask = ~sharpe_series.isna()
                sharpe_clean = sharpe_series.dropna().values.astype(np.float64)
                sharpe_offset = int(valid_mask.values.argmax()) if valid_mask.any() else 0
            else:
                valid_mask = ~np.isnan(sharpe_series)
                sharpe_clean = sharpe_series[valid_mask].astype(np.float64)
                sharpe_offset = int(np.argmax(valid_mask)) if valid_mask.any() else 0

        for det_name in self.detector_names:
            logger.info("Running %s detector on '%s'", det_name, name)
            det_result: Dict[str, Any] = {
                "breakpoints": [],
                "on_returns": [],
                "on_sharpe": [],
            }

            # Detect on raw returns
            try:
                detector = self._make_detector(det_name)
                detector.fit(returns_arr)
                bp_returns = detector.get_breakpoints()
                det_result["breakpoints"] = bp_returns
                det_result["on_returns"] = bp_returns
                logger.info(
                    "%s found %d breakpoints on returns for '%s': %s",
                    det_name, len(bp_returns), name, bp_returns,
                )
            except Exception as exc:
                logger.warning(
                    "%s failed on returns for '%s': %s", det_name, name, exc
                )

            # Detect on rolling Sharpe
            if sharpe_clean is not None and len(sharpe_clean) >= self.min_segment_size * 2:
                try:
                    detector_sharpe = self._make_detector(det_name)
                    detector_sharpe.fit(sharpe_clean)
                    bp_sharpe_local = detector_sharpe.get_breakpoints()
                    # Map back to original index space
                    bp_sharpe = [bp + sharpe_offset for bp in bp_sharpe_local]
                    det_result["on_sharpe"] = bp_sharpe
                    logger.info(
                        "%s found %d breakpoints on rolling Sharpe for '%s': %s",
                        det_name, len(bp_sharpe), name, bp_sharpe,
                    )
                except Exception as exc:
                    logger.warning(
                        "%s failed on rolling Sharpe for '%s': %s",
                        det_name, name, exc,
                    )

            changepoint_results[det_name] = det_result

        # --- Step 3: Regime detection ---
        regime_results: Optional[Dict[str, Any]] = None

        if self.regime_model is not None:
            logger.info(
                "Running %s regime detection on '%s'",
                self.regime_model, name,
            )
            try:
                regime_detector = _REGIME_MAP[self.regime_model]()
                regime_detector.fit(returns_arr)

                regimes = regime_detector.get_regimes()

                # Extract regime means
                regime_stats = regime_detector.get_regime_statistics()
                regime_means = {
                    k: v["mean"] for k, v in regime_stats.items()
                }

                # Label regimes
                regime_labels = label_signal_state(regime_means)

                # Transition matrix
                if hasattr(regime_detector, "get_transition_matrix"):
                    trans_matrix = regime_detector.get_transition_matrix()
                else:
                    trans_matrix = None

                # Smoothed probabilities
                if hasattr(regime_detector, "get_smoothed_probabilities"):
                    smoothed_probs = regime_detector.get_smoothed_probabilities()
                elif hasattr(regime_detector, "get_state_probabilities"):
                    smoothed_probs = regime_detector.get_state_probabilities()
                else:
                    smoothed_probs = None

                regime_results = {
                    "regime_means": regime_means,
                    "regime_labels": regime_labels,
                    "transition_matrix": trans_matrix,
                    "smoothed_probs": smoothed_probs,
                }
                logger.info(
                    "Regime detection for '%s': %d regimes, means=%s",
                    name, len(regime_means), regime_means,
                )
            except Exception as exc:
                logger.warning(
                    "Regime detection failed for '%s': %s", name, exc
                )

        # --- Step 4: Decay onset detection ---
        decay_detected = False
        decay_onset: Optional[int] = None

        if sharpe_series is not None:
            try:
                decay_result = detect_decay_onset(sharpe_series, method="cusum")
                onset_idx = decay_result.get("decay_onset_index")
                if onset_idx is not None:
                    decay_detected = True
                    decay_onset = int(onset_idx)
                    logger.info(
                        "Decay onset detected for '%s' at index %d",
                        name, decay_onset,
                    )
                else:
                    logger.info(
                        "No decay onset detected for '%s'", name
                    )
            except Exception as exc:
                logger.warning(
                    "Decay onset detection failed for '%s': %s", name, exc
                )

        # --- Step 5: Assemble report ---
        report = AuditReport(
            name=name,
            n_observations=n,
            date_range=date_range,
            changepoint_results=changepoint_results,
            regime_results=regime_results,
            metrics=computed_metrics,
            decay_detected=decay_detected,
            decay_onset=decay_onset,
            summary="",
        )

        report.summary = _verdict(report)
        logger.info("Audit complete for '%s': %s", name, report.summary)

        return report

    # ------------------------------------------------------------------
    # Batch audit
    # ------------------------------------------------------------------

    def audit_multiple(
        self,
        returns_dict: Dict[str, Union[np.ndarray, pd.Series]],
    ) -> Dict[str, AuditReport]:
        """Audit multiple signals and return a dict of reports.

        Parameters
        ----------
        returns_dict : dict
            Mapping from signal name to return series.

        Returns
        -------
        dict
            Mapping from signal name to ``AuditReport``.
        """
        reports: Dict[str, AuditReport] = {}
        for signal_name, returns in returns_dict.items():
            try:
                reports[signal_name] = self.audit(returns, name=signal_name)
            except Exception as exc:
                logger.error(
                    "Audit failed for '%s': %s", signal_name, exc,
                )
        return reports

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        audit_result: AuditReport,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate a text/markdown summary of audit findings.

        Parameters
        ----------
        audit_result : AuditReport
            Completed audit result from :meth:`audit`.
        output_path : str or None
            If provided, writes the report to this file path.

        Returns
        -------
        str
            The formatted report text.
        """
        lines: List[str] = []
        lines.append(f"# Signal Decay Audit Report: {audit_result.name}")
        lines.append("")

        # Overview
        lines.append("## Overview")
        lines.append(f"- **Signal**: {audit_result.name}")
        lines.append(f"- **Observations**: {audit_result.n_observations}")
        if audit_result.date_range is not None:
            start, end = audit_result.date_range
            lines.append(f"- **Date range**: {start} to {end}")
        lines.append(f"- **Verdict**: {audit_result.summary}")
        lines.append("")

        # Metrics
        lines.append("## Metrics")
        for metric_name, metric_val in audit_result.metrics.items():
            if isinstance(metric_val, (pd.Series, np.ndarray)):
                # Summarize rolling metrics
                if isinstance(metric_val, pd.Series):
                    vals = metric_val.dropna()
                else:
                    vals = metric_val[~np.isnan(metric_val)]
                if len(vals) > 0:
                    lines.append(
                        f"- **{metric_name}**: "
                        f"mean={np.mean(vals):.4f}, "
                        f"last={vals.iloc[-1] if isinstance(vals, pd.Series) else vals[-1]:.4f}, "
                        f"min={np.min(vals):.4f}, "
                        f"max={np.max(vals):.4f}"
                    )
                else:
                    lines.append(f"- **{metric_name}**: no valid values")
            elif isinstance(metric_val, dict):
                # Dict metrics like half_life
                summary_parts = []
                for k, v in metric_val.items():
                    if isinstance(v, float):
                        summary_parts.append(f"{k}={v:.4f}")
                    else:
                        summary_parts.append(f"{k}={v}")
                lines.append(f"- **{metric_name}**: {', '.join(summary_parts)}")
            elif metric_val is not None:
                lines.append(f"- **{metric_name}**: {metric_val:.4f}")
            else:
                lines.append(f"- **{metric_name}**: not computed")
        lines.append("")

        # Changepoint results
        lines.append("## Changepoint Detection")
        for det_name, det_result in audit_result.changepoint_results.items():
            bp_returns = det_result.get("on_returns", [])
            bp_sharpe = det_result.get("on_sharpe", [])
            lines.append(f"### {det_name.replace('_', ' ').title()}")
            if bp_returns:
                lines.append(f"- Breakpoints on returns: {bp_returns}")
            else:
                lines.append("- Breakpoints on returns: none detected")
            if bp_sharpe:
                lines.append(f"- Breakpoints on rolling Sharpe: {bp_sharpe}")
            else:
                lines.append("- Breakpoints on rolling Sharpe: none detected")
            lines.append("")

        # Regime detection
        if audit_result.regime_results is not None:
            lines.append("## Regime Detection")
            rm = audit_result.regime_results
            regime_means = rm.get("regime_means", {})
            regime_labels = rm.get("regime_labels", {})
            for regime_id in sorted(regime_means.keys()):
                mean_val = regime_means[regime_id]
                label = regime_labels.get(regime_id, "unknown")
                lines.append(
                    f"- Regime {regime_id} ({label}): mean={mean_val:.6f}"
                )
            trans = rm.get("transition_matrix")
            if trans is not None:
                lines.append("")
                lines.append("Transition matrix:")
                lines.append("```")
                if isinstance(trans, np.ndarray):
                    for row in trans:
                        lines.append(
                            "  " + "  ".join(
                                f"{float(v):.4f}" for v in np.atleast_1d(row)
                            )
                        )
                elif isinstance(trans, pd.DataFrame):
                    lines.append(str(trans))
                lines.append("```")
            lines.append("")

        # Decay detection
        lines.append("## Decay Detection")
        if audit_result.decay_detected:
            lines.append(f"- **Decay detected**: Yes")
            lines.append(f"- **Onset index**: {audit_result.decay_onset}")
        else:
            lines.append("- **Decay detected**: No")
        lines.append("")

        report_text = "\n".join(lines)

        if output_path is not None:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(report_text, encoding="utf-8")
            logger.info("Report written to %s", output_path)

        return report_text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_detector(self, name: str):
        """Instantiate a changepoint detector by name.

        Parameters
        ----------
        name : str
            Detector name (must be a key in ``_DETECTOR_MAP``).

        Returns
        -------
        ChangePointDetector
            An uninitialized detector instance with configured
            ``min_segment_size``.
        """
        cls = _DETECTOR_MAP[name]

        # Each detector has slightly different constructor signatures;
        # pass min_size / min_segment_size where supported.
        if name == "pelt":
            return cls(min_size=self.min_segment_size)
        elif name == "cusum":
            # CUSUMDetector takes alpha only; no min_size parameter.
            return cls()
        elif name == "bai_perron":
            # BaiPerronDetector uses trimming (fraction), not absolute
            # min_segment_size.  Default trimming=0.15 is appropriate.
            return cls()
        elif name == "mosum":
            return cls()
        else:
            return cls()
