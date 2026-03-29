"""Factor return data ingestion from multiple sources.

Handles loading, validation, and basic transformation of factor return
time series from Fama-French (via pandas_datareader), FRED (via fredapi),
and custom CSV files. All returned DataFrames use DatetimeIndex and
decimal-scaled returns.

References:
    - Fama & French (1993, 2015) — 3-factor and 5-factor models
    - Carhart (1997) — momentum factor
    - McLean & Pontiff (2016) — factor decay post-publication
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


class FactorDataLoader:
    """Loads and validates factor return data from multiple sources.

    All date parameters accept anything parseable by ``pd.Timestamp``.
    Returned DataFrames always have a ``DatetimeIndex`` named ``"date"``.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Fama-French factor loaders
    # ------------------------------------------------------------------

    def load_fama_french(
        self,
        dataset: str = "F-F_Research_Data_Factors_daily",
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
    ) -> pd.DataFrame:
        """Load Fama-French 3-factor daily returns via pandas_datareader.

        Parameters:
            dataset: Name of the Fama-French dataset on Ken French's data
                library. Default is the daily 3-factor file.
            start: Inclusive start date. If None, loads from the earliest
                available date.
            end: Inclusive end date. If None, loads through the latest
                available date.

        Returns:
            DataFrame with columns [Mkt-RF, SMB, HML, RF] in decimal
            units (i.e., 0.01 not 1.0 for a 1 % return). Index is a
            DatetimeIndex named ``"date"``.
        """
        import pandas_datareader.data as web

        start = pd.Timestamp(start) if start is not None else None
        end = pd.Timestamp(end) if end is not None else None

        raw = web.DataReader(dataset, "famafrench", start=start, end=end)
        # DataReader returns a dict; index 0 is the main table.
        df: pd.DataFrame = raw[0]

        # Convert percentage points to decimals.
        df = df / 100.0

        # pandas_datareader returns a PeriodIndex.  For monthly datasets
        # (e.g., "F-F_Research_Data_Factors"), convert to end-of-period
        # Timestamps to preserve the correct date semantics.  For daily
        # datasets the PeriodIndex is already day-resolution.
        if hasattr(df.index, 'to_timestamp'):
            df.index = df.index.to_timestamp(how='end')
        else:
            df.index = pd.to_datetime(df.index.astype(str))
        df.index.name = "date"

        self._log_checksum(df, label=dataset)
        self._report_missing(df, label=dataset)

        logger.info(
            "Loaded Fama-French dataset '%s': %d rows, %s to %s",
            dataset,
            len(df),
            df.index.min().date(),
            df.index.max().date(),
        )
        return df

    def load_fama_french_five(
        self,
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
    ) -> pd.DataFrame:
        """Load the Fama-French 5-factor daily returns.

        Adds RMW (robust-minus-weak profitability) and CMA
        (conservative-minus-aggressive investment) to the standard
        3-factor set.

        Parameters:
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            DataFrame with columns [Mkt-RF, SMB, HML, RMW, CMA, RF]
            in decimal units with DatetimeIndex.
        """
        return self.load_fama_french(
            dataset="F-F_Research_Data_5_Factors_2x3_daily",
            start=start,
            end=end,
        )

    def load_momentum(
        self,
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
    ) -> pd.DataFrame:
        """Load the Fama-French momentum (UMD / WML) factor.

        Parameters:
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            DataFrame with the momentum factor column in decimal units
            with DatetimeIndex.
        """
        return self.load_fama_french(
            dataset="F-F_Momentum_Factor_daily",
            start=start,
            end=end,
        )

    # ------------------------------------------------------------------
    # FRED macro series
    # ------------------------------------------------------------------

    def load_fred_series(
        self,
        series_ids: List[str],
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
    ) -> pd.DataFrame:
        """Load one or more FRED macro series.

        Requires the ``FRED_API_KEY`` environment variable to be set.

        Parameters:
            series_ids: List of FRED series identifiers, e.g.
                ``["DGS10", "BAMLH0A0HYM2"]``.
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            DataFrame with one column per series and a DatetimeIndex
            named ``"date"``.

        Raises:
            EnvironmentError: If ``FRED_API_KEY`` is not set.
        """
        from fredapi import Fred

        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "FRED_API_KEY environment variable is not set. "
                "Obtain a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )

        start = pd.Timestamp(start) if start is not None else None
        end = pd.Timestamp(end) if end is not None else None

        fred = Fred(api_key=api_key)
        frames: Dict[str, pd.Series] = {}
        for sid in series_ids:
            series = fred.get_series(
                sid,
                observation_start=start,
                observation_end=end,
            )
            frames[sid] = series

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"

        self._log_checksum(df, label="FRED:" + ",".join(series_ids))
        self._report_missing(df, label="FRED")

        logger.info(
            "Loaded %d FRED series: %s (%d rows)",
            len(series_ids),
            ", ".join(series_ids),
            len(df),
        )
        return df

    # ------------------------------------------------------------------
    # Custom CSV
    # ------------------------------------------------------------------

    def load_custom_returns(
        self,
        filepath: str,
        date_col: str = "date",
        return_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load a CSV of custom factor returns.

        Parameters:
            filepath: Path to the CSV file.
            date_col: Name of the column containing dates. Parsed with
                ``pd.to_datetime``.
            return_cols: Subset of columns to keep. If None, all columns
                except ``date_col`` are retained.

        Returns:
            DataFrame with selected return columns and a DatetimeIndex
            named ``"date"``.
        """
        df = pd.read_csv(filepath, parse_dates=[date_col])
        df = df.rename(columns={date_col: "date"}).set_index("date").sort_index()

        if return_cols is not None:
            missing = set(return_cols) - set(df.columns)
            if missing:
                raise ValueError(
                    f"Requested columns not found in CSV: {sorted(missing)}"
                )
            df = df[return_cols]

        self._log_checksum(df, label=filepath)
        self._report_missing(df, label=filepath)

        logger.info("Loaded custom returns from '%s': %d rows, %d cols", filepath, len(df), len(df.columns))
        return df

    # ------------------------------------------------------------------
    # Rolling analytics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_rolling_sharpe(
        returns: pd.Series,
        window: int = 252,
        annualization: int = 252,
    ) -> pd.Series:
        """Compute rolling annualized Sharpe ratio.

        Parameters:
            returns: Daily excess return series (already in excess of
                the risk-free rate).
            window: Lookback window in trading days. Default 252
                corresponds to approximately one calendar year of US
                equity trading days, the standard horizon for evaluating
                annual risk-adjusted performance (Sharpe, 1994).
            annualization: Factor to annualize the Sharpe ratio. Default
                252 for daily data (sqrt-of-time scaling assumes i.i.d.
                returns — a first-order approximation; see Lo, 2002).

        Returns:
            Series of rolling annualized Sharpe ratios aligned to the
            end of each window.
        """
        rolling_mean = returns.rolling(window=window, min_periods=window).mean()
        rolling_std = returns.rolling(window=window, min_periods=window).std(ddof=1)

        sharpe = (rolling_mean / rolling_std) * np.sqrt(annualization)
        sharpe.name = "rolling_sharpe"
        return sharpe

    @staticmethod
    def compute_rolling_ic(
        factor_values: pd.Series,
        forward_returns: pd.Series,
        window: int = 63,
    ) -> pd.Series:
        """Compute rolling Spearman rank information coefficient.

        The IC is the rank correlation between contemporaneous factor
        exposures and subsequent realised returns — the canonical
        measure of factor predictive power (Grinold & Kahn, 2000).

        Parameters:
            factor_values: Cross-sectional factor scores aligned by date.
            forward_returns: Subsequent realised returns aligned to the
                same dates as ``factor_values``.
            window: Lookback window in trading days. Default 63 is
                approximately one calendar quarter (≈ 63 trading days),
                a standard evaluation horizon that balances noise
                reduction with responsiveness to regime shifts.

        Returns:
            Series of rolling Spearman rank IC values.
        """
        aligned = pd.DataFrame({
            "factor": factor_values,
            "forward": forward_returns,
        }).dropna()

        ic_values = []
        dates = aligned.index
        for i in range(len(dates)):
            if i < window - 1:
                ic_values.append(np.nan)
            else:
                sub = aligned.iloc[i - window + 1 : i + 1]
                corr, _ = stats.spearmanr(sub["factor"], sub["forward"])
                ic_values.append(corr)

        result = pd.Series(ic_values, index=dates, name="rolling_ic")
        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_data(df: pd.DataFrame) -> Dict[str, object]:
        """Run diagnostic checks on a factor return DataFrame.

        Checks performed:
            1. **NaN audit** — per-column and total missing percentage.
            2. **Stationarity** — Augmented Dickey-Fuller test per column
               (Dickey & Fuller, 1979). Reports test statistic, p-value,
               and a boolean indicating rejection of the unit-root null
               at the 5 % level.
            3. **Date gaps** — identifies calendar-day gaps exceeding 3
               days (accounts for weekends) in the DatetimeIndex, which
               may indicate missing trading days.

        Parameters:
            df: DataFrame with DatetimeIndex and numeric columns.

        Returns:
            Dict with keys:
                - ``"nan_counts"``: per-column NaN counts (dict).
                - ``"nan_pct"``: per-column NaN percentage (dict).
                - ``"total_nan_pct"``: overall NaN percentage (float).
                - ``"adf_tests"``: per-column ADF results (dict of dicts
                  with keys ``"statistic"``, ``"pvalue"``,
                  ``"is_stationary"``).
                - ``"date_gaps"``: list of ``(gap_start, gap_end,
                  gap_days)`` tuples for gaps > 3 calendar days.
                - ``"row_count"``: number of rows (int).
                - ``"date_range"``: ``(first_date, last_date)`` tuple.
        """
        diagnostics: Dict[str, object] = {}

        # --- NaN audit ---
        nan_counts = df.isna().sum().to_dict()
        nan_pct = (df.isna().mean() * 100.0).to_dict()
        total_nan_pct = float(df.isna().values.mean() * 100.0)

        diagnostics["nan_counts"] = nan_counts
        diagnostics["nan_pct"] = nan_pct
        diagnostics["total_nan_pct"] = total_nan_pct

        logger.info(
            "NaN audit: %.2f%% overall missing (%s)",
            total_nan_pct,
            ", ".join(f"{c}: {nan_counts[c]}" for c in df.columns if nan_counts[c] > 0)
            or "none",
        )

        # --- ADF stationarity tests ---
        adf_results: Dict[str, Dict[str, object]] = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) < 20:
                adf_results[col] = {
                    "statistic": np.nan,
                    "pvalue": np.nan,
                    "is_stationary": None,
                }
                continue
            stat, pvalue, _, _, _, _ = adfuller(series, autolag="AIC")
            adf_results[col] = {
                "statistic": float(stat),
                "pvalue": float(pvalue),
                "is_stationary": bool(pvalue < 0.05),
            }
        diagnostics["adf_tests"] = adf_results

        # --- Date gaps ---
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
            sorted_idx = df.index.sort_values()
            deltas = sorted_idx[1:] - sorted_idx[:-1]
            gap_threshold = pd.Timedelta(days=3)
            gaps = []
            for i, delta in enumerate(deltas):
                if delta > gap_threshold:
                    gaps.append((
                        sorted_idx[i],
                        sorted_idx[i + 1],
                        int(delta.days),
                    ))
            diagnostics["date_gaps"] = gaps
            if gaps:
                logger.warning(
                    "Found %d date gaps exceeding 3 calendar days", len(gaps)
                )
        else:
            diagnostics["date_gaps"] = []

        diagnostics["row_count"] = len(df)
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
            diagnostics["date_range"] = (df.index.min(), df.index.max())
        else:
            diagnostics["date_range"] = (None, None)

        return diagnostics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_checksum(df: pd.DataFrame, label: str = "") -> str:
        """Compute and log SHA-256 checksum of the DataFrame's CSV bytes.

        Parameters:
            df: DataFrame to checksum.
            label: Human-readable source label for the log message.

        Returns:
            Hex-encoded SHA-256 digest string.
        """
        csv_bytes = df.to_csv().encode("utf-8")
        digest = hashlib.sha256(csv_bytes).hexdigest()
        logger.info("Data checksum [%s]: sha256=%s", label, digest)
        return digest

    @staticmethod
    def _report_missing(df: pd.DataFrame, label: str = "") -> None:
        """Log per-column missing value counts without filling.

        Parameters:
            df: DataFrame to inspect.
            label: Human-readable source label for the log message.
        """
        nan_counts = df.isna().sum()
        total_missing = int(nan_counts.sum())
        if total_missing > 0:
            detail = ", ".join(
                f"{col}: {int(count)}"
                for col, count in nan_counts.items()
                if count > 0
            )
            logger.warning(
                "Missing values in '%s': %d total (%s). "
                "Values are NOT filled — handle explicitly before analysis.",
                label,
                total_missing,
                detail,
            )
        else:
            logger.info("No missing values in '%s'.", label)
