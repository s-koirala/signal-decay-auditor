"""CLI entrypoint for the signal-decay-auditor.

Usage::

    python -m src audit --factor HML
    python -m src audit --factor HML SMB MKT --start 2000-01-01 --end 2023-12-31
    python -m src audit --csv data/custom.csv --col returns
    python -m src audit --all --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.auditor import SignalDecayAuditor
from src.factors.data_loader import FactorDataLoader

# Factor name mapping: CLI shorthand -> Fama-French column name
FACTOR_ALIAS = {
    "HML": "HML",
    "SMB": "SMB",
    "MKT": "Mkt-RF",
    "Mkt-RF": "Mkt-RF",
}

FF3_FACTORS = ["HML", "SMB", "MKT"]

logger = logging.getLogger("src")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def _resolve_factor_name(name: str) -> str:
    """Map a CLI factor name to its Fama-French column name."""
    upper = name.upper()
    if upper in FACTOR_ALIAS:
        return FACTOR_ALIAS[upper]
    # Pass through as-is for non-standard names
    return name


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Signal Decay Auditor — detect structural breaks in factor returns.",
    )
    sub = parser.add_subparsers(dest="command")

    audit = sub.add_parser("audit", help="Run signal decay audit on factor return series.")

    # Input source (mutually exclusive)
    source = audit.add_argument_group("data source")
    source.add_argument(
        "--factor",
        nargs="+",
        metavar="NAME",
        help="Fama-French factor name(s): HML, SMB, MKT.",
    )
    source.add_argument(
        "--all",
        action="store_true",
        dest="all_factors",
        help="Audit all FF3 factors (HML, SMB, Mkt-RF).",
    )
    source.add_argument(
        "--csv",
        metavar="PATH",
        help="Path to a CSV file with return data.",
    )
    source.add_argument(
        "--col",
        metavar="COLUMN",
        help="Column name in CSV to audit (required with --csv).",
    )

    # Date range
    audit.add_argument("--start", metavar="DATE", help="Start date (ISO 8601).")
    audit.add_argument("--end", metavar="DATE", help="End date (ISO 8601).")

    # Output
    audit.add_argument(
        "--output",
        metavar="DIR",
        default="artifacts/reports",
        help="Output directory for markdown reports (default: artifacts/reports/).",
    )

    # Configuration file
    audit.add_argument(
        "--config",
        metavar="PATH",
        help="Path to a YAML config file. CLI arguments override config values.",
    )

    # Detector / model overrides
    audit.add_argument(
        "--detectors",
        nargs="+",
        metavar="DET",
        help="Override detector list (e.g. pelt cusum bai_perron mosum).",
    )
    audit.add_argument(
        "--regime-model",
        metavar="MODEL",
        choices=["markov", "hmm", "none"],
        help="Regime model: markov, hmm, or none.",
    )
    audit.add_argument(
        "--window",
        type=int,
        metavar="N",
        help="Rolling window size (default: 252).",
    )

    # Verbosity
    audit.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser


def _run_audit(args: argparse.Namespace) -> int:
    """Execute the audit command. Returns exit code."""
    _configure_logging(args.verbose)

    # --- Validate input source ---
    has_factor = args.factor is not None
    has_csv = args.csv is not None

    if not has_factor and not args.all_factors and not has_csv:
        print("Error: specify --factor, --all, or --csv.", file=sys.stderr)
        return 1

    if has_csv and not args.col:
        print("Error: --col is required when using --csv.", file=sys.stderr)
        return 1

    # --- Build auditor ---
    auditor_kwargs = {}
    if args.detectors:
        auditor_kwargs["detectors"] = args.detectors
    if args.regime_model is not None:
        regime = None if args.regime_model == "none" else args.regime_model
        auditor_kwargs["regime_model"] = regime
    if args.window:
        auditor_kwargs["rolling_window"] = args.window

    if args.config:
        auditor = SignalDecayAuditor.from_yaml(args.config)
        # CLI arguments override config file values
        if args.detectors:
            auditor.detector_names = args.detectors
        if args.regime_model is not None:
            auditor.regime_model = None if args.regime_model == "none" else args.regime_model
        if args.window:
            auditor.rolling_window = args.window
    else:
        auditor = SignalDecayAuditor(**auditor_kwargs)
    loader = FactorDataLoader()
    output_dir = Path(args.output)

    # --- Collect return series to audit ---
    series_to_audit: dict = {}  # name -> pd.Series

    if has_csv:
        df = loader.load_custom_returns(args.csv, return_cols=[args.col])
        series_to_audit[args.col] = df[args.col].dropna()

    else:
        factor_names = FF3_FACTORS if args.all_factors else args.factor
        ff_col_names = [_resolve_factor_name(f) for f in factor_names]

        df = loader.load_fama_french(start=args.start, end=args.end)

        for cli_name, col_name in zip(factor_names, ff_col_names):
            if col_name not in df.columns:
                print(f"Error: column '{col_name}' not found in FF dataset. "
                      f"Available: {list(df.columns)}", file=sys.stderr)
                return 1
            series_to_audit[col_name] = df[col_name].dropna()

    # --- Run audits ---
    reports = auditor.audit_multiple(series_to_audit)

    # --- Output ---
    for name, report in reports.items():
        # Print verdict to stdout
        print(f"[{name}] {report.summary}")

        # Save markdown report
        safe_name = name.lower().replace("-", "_").replace(" ", "_")
        report_path = output_dir / f"{safe_name}_audit.md"
        auditor.generate_report(report, output_path=str(report_path))
        print(f"  Report saved: {report_path}")

    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "audit":
        return _run_audit(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
