"""
Figure: VaR and balance sheet levels (NO standardization)

Purpose
-------
Construct a diagnostic figure (Figure-5-style, following Adrian & Shin,
but using LEVELS rather than standardized values) for 2014Q1–2025Q3.

The figure displays:
- Total Assets (level)
- Book Common Equity (level)
- Liabilities (level) = Assets - Equity (accounting identity)
- Value-at-Risk (99%) level, harmonized using the Bank of America scaling factor

Important: NO standardization
------------------------------
All series are shown in levels.
We do not normalize by initial values, means, GDP, or balance sheet size.
As a result:
- The figure reflects long-run growth and scale effects.
- Variables are plotted on two y-axes:
    • Left axis: balance sheet quantities
    • Right axis: VaR levels

Aggregation (bank → sector)
---------------------------
For each quarter, we compute a value-weighted average across banks:
- Weights = lagged total assets (assets in t−1).
- Motivation:
    • Larger banks receive more weight.
    • Using lagged assets avoids mechanical simultaneity between shocks
      to assets and the aggregation weights.

Input files
-----------
1) data/processed/merged_quarterly_balanced.csv
   Quarterly balance sheet data at the bank level
2) output/data/merged_with_var_99_dual_methods.csv
   Quarterly VaR data, including 99% harmonized VaR (BoA factor)

Output
------
output/figures/figure_levels_var_and_liabilities.png
"""

from __future__ import annotations
from pathlib import Path

import os
import numpy as np
import pandas as pd


# -----------------------------
# File paths (project structure)
# -----------------------------
BASE_FILE = Path("data/processed/merged_quarterly_balanced.csv")
VAR_FILE = Path("output/data/merged_with_var_99_dual_methods.csv")


# -----------------------------
# Fixed column definitions (thesis version)
# -----------------------------
ASSETS_COL = "financial_summary_Total Assets"
EQUITY_COL = "financial_summary_Common Equity - Total"
VAR99_COL = "var_99_boa_factor"


# -----------------------------
# Sample period
# -----------------------------
SAMPLE_START = pd.Timestamp("2014-03-31")  # 2014Q1 quarter-end
SAMPLE_END = pd.Timestamp("2025-09-30")    # 2025Q3 quarter-end


def normalize_bank_id(bank_id: str) -> str:
    """
    Normalize bank identifiers to a single canonical format.

    Motivation:
    Different data sources may refer to the same bank using different names
    (e.g. "JPM", "J.P. Morgan", "JPMorgan Chase").

    This function standardizes identifiers so that merges across datasets
    are reliable and reproducible.
    """
    bank_id = str(bank_id).lower().strip()
    normalized = bank_id.replace(" ", "").replace("_", "").replace("-", "").replace(".", "")

    if "bankofamerica" in normalized or normalized == "bofa":
        return "bank_of_america"
    if "citigroup" in normalized or "citibank" in normalized or normalized == "citi":
        return "citibank"
    if "wellsfargo" in normalized or normalized == "wells":
        return "wells_fargo"
    if "jpmorgan" in normalized or normalized == "jpm":
        return "jpmorgan_chase"
    if "goldmansachs" in normalized or "goldman" in normalized:
        return "goldmansachs"
    if "morganstanley" in normalized:
        return "morganstanley"
    if "fifththird" in normalized:
        return "fifththird"
    if "keycorp" in normalized or "keybank" in normalized:
        return "keycorp"
    if "pnc" in normalized:
        return "pnc"
    if "usbancorp" in normalized or "usbank" in normalized:
        return "usbancorp"
    if "regionsfinancial" in normalized or "regions" in normalized:
        return "regionsfinancial"

    return bank_id.replace(" ", "_").replace("-", "_").replace(".", "")


def _require_columns(df: pd.DataFrame, cols: list[str], *, where: str) -> None:
    """
    Defensive check: ensure required columns exist before proceeding.

    This produces clearer error messages than allowing pandas to fail later.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s) in {where}: " + ", ".join(missing))


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Identify the first column present among a list of candidate names.

    Used to accommodate minor naming differences across data sources
    (e.g. 'bank', 'entity', 'ticker').
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_datetime(series: pd.Series) -> pd.Series:
    """
    Convert date or quarter information into pandas datetime.

    Supports:
    - pandas Period (quarters)
    - datetime columns
    - strings such as '2014Q1'
    - standard date strings

    All quarters are converted to quarter-end dates.
    """
    if isinstance(series.dtype, pd.PeriodDtype):
        return series.dt.to_timestamp(how="end").dt.normalize()
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    s = series.astype(str).str.strip()
    s_q = s.str.upper().str.replace(r"[\s\-_]", "", regex=True)
    is_q = s_q.str.match(r"^\d{4}Q[1-4]$")

    if float(is_q.mean()) >= 0.5:
        out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        if bool(is_q.any()):
            out.loc[is_q] = (
                pd.PeriodIndex(s_q.loc[is_q], freq="Q")
                .to_timestamp(how="end")
                .normalize()
                .to_numpy()
            )
        if bool((~is_q).any()):
            out.loc[~is_q] = pd.to_datetime(s.loc[~is_q], errors="coerce")
        return out

    return pd.to_datetime(s, errors="coerce")


def _to_numeric(series: pd.Series) -> pd.Series:
    """
    Convert a series to numeric values.

    Handles common formatting issues:
    - empty strings or textual NA values
    - parentheses for negative numbers
    - thousands separators
    - stray non-numeric characters
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan, "N/A": np.nan})
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace(r"[^\d\.\-eE+]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def load_df_from_project_outputs(
    base_file: Path = BASE_FILE,
    var_file: Path = VAR_FILE,
) -> pd.DataFrame:
    """
    Load, clean, and merge balance sheet data with 99% VaR data.

    Returns a bank-quarter panel containing:
    - bank_id
    - quarter (quarter-end date)
    - assets
    - equity
    - var99 (99% VaR level)

    The merge is an inner join on (bank_id, quarter),
    and the sample is restricted to 2014Q1–2025Q3.
    """

    if not base_file.exists():
        raise FileNotFoundError(f"Missing file: {base_file}")
    if not var_file.exists():
        raise FileNotFoundError(f"Missing file: {var_file}")

    base = pd.read_csv(base_file)
    var = pd.read_csv(var_file)

    # Identify and standardize entity and time columns
    base_entity = _find_column(base, ["bank_id", "bank", "entity", "ticker"])
    var_entity = _find_column(var, ["bank_id", "bank", "entity", "ticker"])
    base_time = _find_column(base, ["period_end_date", "quarter", "date", "time"])
    var_time = _find_column(var, ["period_end_date", "quarter", "date", "time"])

    base = base.rename(columns={base_entity: "bank_id", base_time: "period_end_date"})
    var = var.rename(columns={var_entity: "bank_id", var_time: "period_end_date"})

    base["bank_id"] = base["bank_id"].apply(normalize_bank_id)
    var["bank_id"] = var["bank_id"].apply(normalize_bank_id)

    base["period_end_date"] = _to_datetime(base["period_end_date"])
    var["period_end_date"] = _to_datetime(var["period_end_date"])

    var[VAR99_COL] = _to_numeric(var[VAR99_COL])

    merged = base.merge(
        var[["bank_id", "period_end_date", VAR99_COL]].rename(columns={VAR99_COL: "var_99_level"}),
        on=["bank_id", "period_end_date"],
        how="inner",
    )

    merged = merged[(merged["period_end_date"] >= SAMPLE_START) &
                    (merged["period_end_date"] <= SAMPLE_END)]

    df = pd.DataFrame({
        "bank_id": merged["bank_id"],
        "quarter": merged["period_end_date"],
        "assets": _to_numeric(merged[ASSETS_COL]),
        "equity": _to_numeric(merged[EQUITY_COL]),
        "var99": _to_numeric(merged["var_99_level"]),
    })

    df = df.dropna()
    df = df[(df["assets"] > 0) & (df["equity"] > 0) & (df["var99"] > 0)]
    df = df.sort_values(["bank_id", "quarter"]).reset_index(drop=True)

    return df
