"""
Replicate Adrian & Shin Figure 5-style plot (risk + balance sheet adjustment),
for 2014Q1–2025Q3, using:
- Unit VaR = VaR / Assets
- VaR/E   = VaR / Book Equity
- Leverage = Assets / Book Equity

Key changes vs your prior script:
1) Uses ratios in LEVELS (not log-ratios).
2) Standardizes relative to PRE-PERIOD (default: 2014Q1–2019Q4).
3) Value-weighted average with LAGGED assets as weights (no fallback to simple mean).
4) Hard-codes equity column (you can change EQUITY_COL if necessary).
5) Filters sample to 2014Q1–2025Q3.

Inputs:
- output/merged_quarterly_balanced.csv
- output/merged_with_var_95_approx.csv

"""

from __future__ import annotations

from pathlib import Path

import os
import numpy as np
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
# Base file contains balance sheet data (assets, equity, etc.)
BASE_FILE = Path("output/merged_quarterly_balanced.csv")

# VaR file contains Value-at-Risk estimates at 95% confidence level
VAR_FILE = Path("output/merged_with_var_95_approx.csv")

# -----------------------------
# Bank ID harmonization
# -----------------------------
# Standardize bank identifiers across different data sources
# Maps alternative names to canonical identifiers
BANK_MAP = {
    "bankofamerica": "bank_of_america",
    "citigroup": "citibank",
    "wellsfargo": "wells_fargo",
}

# -----------------------------
# Plot / sample settings
# -----------------------------
# Full sample to plot: 2014Q1–2025Q3
# These are quarter-end dates (e.g., 2014Q1 ends March 31)
SAMPLE_START = pd.Timestamp("2014-03-31")  # 2014Q1 end
SAMPLE_END = pd.Timestamp("2025-09-30")    # 2025Q3 end

# Pre-period for standardization (caption: "pre-crisis")
# This serves as the "normal" baseline period for z-score transformation
# Adrian & Shin use a calm period before major financial stress
PRE_START = pd.Timestamp("2014-03-31")
PRE_END = pd.Timestamp("2019-12-31")

# -----------------------------
# Column choices
# -----------------------------
# IMPORTANT: Set this to the column that represents "total book equity" in your dataset.
# If this column doesn't exist, the script will stop with a clear error message.
EQUITY_COL = "total_shareholders_equity"  # <-- change if necessary

# Fallback if EQUITY_COL is missing/empty: choose the column with best coverage.
# Listed in preference order - script will pick the first available with data
EQUITY_CANDIDATES = [
    EQUITY_COL,
    "shareholders_equity_common",
    "common_equity_total",
    "common_equity_attributable_to_parent_shareholders",
    "shareholders_equity_attributable_to_parent_shareholders_total",
    "tangible_total_equity",
    "book_value_excluding_other_equity",
    "total_shareholders_equity_including_minority_interest_hybrid_debt",
]

# Assets candidates (script chooses the one with most coverage, but you can also hard-code)
ASSET_CANDIDATES = ["total_assets_2", "total_assets", "assets"]

# VaR columns in var-file (95% confidence level Value-at-Risk)
VAR95_CANDIDATES = ["var_95_level", "var_95", "var_95_approx"]


def _require_columns(df: pd.DataFrame, cols: list[str], *, where: str) -> None:
    """
    Check that required columns exist in dataframe.
    Raises KeyError with clear message if any are missing.
    
    Args:
        df: DataFrame to check
        cols: List of required column names
        where: Description of which file/stage (for error message)
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s) in {where}: " + ", ".join(missing))


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Find the first existing column from a list of candidates.
    
    Args:
        df: DataFrame to search
        candidates: List of column names to try
        
    Returns:
        First matching column name, or None if none exist
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _best_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Find the column with the most non-missing values from a list of candidates.
    This ensures we use the most complete data source available.
    
    Args:
        df: DataFrame to search
        candidates: List of column names to consider
        
    Returns:
        Column name with most non-missing values, or None if all are empty/missing
    """
    present = [c for c in candidates if c in df.columns]
    if not present:
        return None
    # Count non-missing values for each candidate
    counts = [(int(df[c].notna().sum()), c) for c in present]
    counts.sort(reverse=True)
    best_count, best_col = counts[0]
    if best_count == 0:
        return None
    return best_col


def _to_datetime(series: pd.Series) -> pd.Series:
    """
    Convert series to datetime, handling both Period and datetime types.
    
    Period types (e.g., "2014Q1") are converted to quarter-end timestamps.
    
    Args:
        series: Series to convert
        
    Returns:
        Series with datetime64 dtype
    """
    if isinstance(series.dtype, pd.PeriodDtype):
        return series.dt.to_timestamp()
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce")


def _to_numeric(series: pd.Series) -> pd.Series:
    """
    Robustly convert series to numeric, handling common data issues:
    - String representations of numbers
    - Parentheses for negative numbers: "(1000)" -> -1000
    - Thousands separators: "1,000" -> 1000
    - Various missing value representations
    
    Args:
        series: Series to convert
        
    Returns:
        Series with float64 dtype
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    
    # Convert to string and clean
    s = series.astype(str).str.strip()
    
    # Replace various missing value representations with NaN
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan, "N/A": np.nan, "na": np.nan})
    
    # Handle accounting format: (1000) means -1000
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    
    # Remove thousands separators
    s = s.str.replace(",", "", regex=False)
    
    # Remove any remaining non-numeric characters except digits, decimal point, minus sign, and scientific notation
    s = s.str.replace(r"[^\d\.\-eE+]", "", regex=True)
    
    return pd.to_numeric(s, errors="coerce")


def load_df_from_project_outputs(
    base_file: Path = BASE_FILE,
    var_file: Path = VAR_FILE,
) -> pd.DataFrame:
    """
    Load and merge balance sheet data with VaR estimates.
    
    This function:
    1. Loads base balance sheet data and VaR data from separate files
    2. Harmonizes bank identifiers and date columns
    3. Merges on bank_id and quarter
    4. Selects appropriate columns for assets, equity, and VaR
    5. Filters to the analysis period (2014Q1-2025Q3)
    6. Cleans data (removes missing/non-positive values)
    
    Returns:
        DataFrame with columns: bank_id, quarter, assets, equity, var95
        
    Raises:
        FileNotFoundError: If input files don't exist
        KeyError: If required columns are missing
        ValueError: If no valid data remains after cleaning
    """
    # Check that input files exist
    if not base_file.exists():
        raise FileNotFoundError(f"Missing file: {base_file}")
    if not var_file.exists():
        raise FileNotFoundError(f"Missing file: {var_file}")

    # Load raw data
    base = pd.read_csv(base_file)
    var = pd.read_csv(var_file)

    # -----------------------------
    # Identify and standardize entity (bank) column
    # Different data sources may use different column names for bank identifier
    # -----------------------------
    base_entity = _find_column(base, ["bank_id", "bank", "entity", "ticker"])
    if base_entity is None:
        raise KeyError("Could not identify entity column in base (expected bank_id/bank/entity/ticker)")
    if base_entity != "bank_id":
        base = base.rename(columns={base_entity: "bank_id"})

    var_entity = _find_column(var, ["bank_id", "bank", "entity", "ticker"])
    if var_entity is None:
        raise KeyError("Could not identify entity column in var (expected bank_id/bank/entity/ticker)")
    if var_entity != "bank_id":
        var = var.rename(columns={var_entity: "bank_id"})

    # -----------------------------
    # Identify and standardize time column
    # Different data sources may use different column names for date/quarter
    # -----------------------------
    base_time = _find_column(base, ["period_end_date", "quarter", "date", "time"])
    if base_time is None:
        raise KeyError("Could not identify time column in base (expected period_end_date/quarter/date/time)")
    if base_time != "period_end_date":
        base = base.rename(columns={base_time: "period_end_date"})

    var_time = _find_column(var, ["period_end_date", "quarter", "date", "time"])
    if var_time is None:
        raise KeyError("Could not identify time column in var (expected period_end_date/quarter/date/time)")
    if var_time != "period_end_date":
        var = var.rename(columns={var_time: "period_end_date"})

    # -----------------------------
    # Clean bank identifiers
    # Standardize to lowercase and apply harmonization mapping
    # -----------------------------
    base["bank_id"] = base["bank_id"].astype(str).str.strip().str.lower().replace(BANK_MAP)
    var["bank_id"] = var["bank_id"].astype(str).str.strip().str.lower().replace(BANK_MAP)

    # -----------------------------
    # Convert dates to datetime format
    # -----------------------------
    base["period_end_date"] = _to_datetime(base["period_end_date"])
    var["period_end_date"] = _to_datetime(var["period_end_date"])

    # Drop rows with missing bank_id or date (can't merge without these)
    base = base.dropna(subset=["bank_id", "period_end_date"])
    var = var.dropna(subset=["bank_id", "period_end_date"])

    # -----------------------------
    # Choose VaR column from available candidates
    # -----------------------------
    var95_col = _find_column(var, VAR95_CANDIDATES)
    if var95_col is None:
        raise KeyError(f"Missing VaR column in var file. Expected one of: {VAR95_CANDIDATES}")

    # Convert VaR to numeric
    var[var95_col] = _to_numeric(var[var95_col])

    # -----------------------------
    # Merge base data with VaR data
    # Inner join ensures we only keep quarters where both balance sheet and VaR exist
    # -----------------------------
    merged = base.merge(
        var[["bank_id", "period_end_date", var95_col]].rename(columns={var95_col: "var_95_level"}),
        on=["bank_id", "period_end_date"],
        how="inner",
    )

    # -----------------------------
    # Filter to analysis period (2014Q1–2025Q3)
    # -----------------------------
    merged = merged[(merged["period_end_date"] >= SAMPLE_START) & (merged["period_end_date"] <= SAMPLE_END)].copy()
    if merged.empty:
        raise ValueError("No observations after merging + filtering to 2014Q1–2025Q3.")

    # -----------------------------
    # Choose best available columns for assets and equity
    # "Best" means most non-missing observations
    # -----------------------------
    assets_col = _best_column(merged, ASSET_CANDIDATES)
    if assets_col is None:
        raise KeyError(f"Missing assets data (expected one of {ASSET_CANDIDATES} with non-missing values).")

    # For equity: first try the specified EQUITY_COL, then fall back to candidates
    equity_col = None
    if EQUITY_COL in merged.columns and int(merged[EQUITY_COL].notna().sum()) > 0:
        equity_col = EQUITY_COL
    else:
        equity_col = _best_column(merged, EQUITY_CANDIDATES)
    if equity_col is None:
        raise KeyError(
            f"No usable equity column found. Tried EQUITY_COL='{EQUITY_COL}' and candidates={EQUITY_CANDIDATES}."
        )

    # -----------------------------
    # Convert selected columns to numeric
    # -----------------------------
    assets = _to_numeric(merged[assets_col])
    equity = _to_numeric(merged[equity_col])
    var95 = _to_numeric(merged["var_95_level"])

    # -----------------------------
    # Create clean output dataframe with standardized column names
    # -----------------------------
    df = pd.DataFrame(
        {
            "bank_id": merged["bank_id"].astype(str),
            "quarter": merged["period_end_date"],
            "assets": assets,
            "equity": equity,
            "var95": var95,
        }
    )

    # -----------------------------
    # Data cleaning: remove missing and non-positive values
    # Financial ratios require positive denominators and numerators
    # -----------------------------
    n0 = len(df)
    df = df.dropna(subset=["bank_id", "quarter", "assets", "equity", "var95"])
    n_dropna = len(df)
    df = df[(df["assets"] > 0) & (df["equity"] > 0) & (df["var95"] > 0)].copy()
    n_pos = len(df)
    
    # Sort by bank and time for lagging operations later
    df = df.sort_values(["bank_id", "quarter"]).reset_index(drop=True)

    # Check that we have valid data remaining
    if df.empty:
        raise ValueError(
            "No valid rows after cleaning (need positive assets/equity/var). "
            f"Rows: start={n0}, after_dropna={n_dropna}, after_positive_filter={n_pos}. "
            f"Parsed columns: assets='{assets_col}', equity='{equity_col}', var='{var95_col}'."
        )

    return df


def plot_adrian_shin_figure_5(df: pd.DataFrame):
    """
    Create Adrian & Shin Figure 5 replication plot.
    
    This function:
    1. Calculates three risk indicators: Unit VaR, Leverage, VaR/Equity
    2. Aggregates across banks using value-weighted averages (lagged assets as weights)
    3. Standardizes all series relative to pre-crisis period (z-scores)
    4. Plots all three series on same scale to show co-movement
    
    The resulting plot shows how risk and leverage evolved together,
    a key finding in Adrian & Shin (2014) about procyclical leverage.
    
    Args:
        df: DataFrame with columns: bank_id, quarter, assets, equity, var95
        
    Returns:
        matplotlib Figure object
        
    Raises:
        KeyError: If required columns are missing
        ValueError: If no valid observations after aggregation
    """
    # Verify required columns exist
    _require_columns(df, ["bank_id", "quarter", "assets", "equity", "var95"], where="df")

    # Set matplotlib config directory to avoid permission issues
    if "MPLCONFIGDIR" not in os.environ:
        mpl_dir = Path("output") / ".mplconfig"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    # Work with a copy to avoid modifying original data
    d = df.copy()
    d["quarter"] = _to_datetime(d["quarter"])
    d = d.sort_values(["bank_id", "quarter"]).reset_index(drop=True)

    # -----------------------------
    # Calculate three risk indicators (ratios in levels, not logs)
    # These are the key metrics from Adrian & Shin (2014)
    # -----------------------------
    
    # Unit VaR: Risk per unit of assets (normalized risk measure)
    # Shows how much risk the bank takes relative to its size
    d["unit_var"] = d["var95"] / d["assets"]
    
    # VaR/Equity: Risk relative to equity capital (capital buffer indicator)
    # High values mean risk is large relative to the cushion available to absorb losses
    d["var_to_equity"] = d["var95"] / d["equity"]
    
    # Leverage: Total gearing ratio (assets/equity)
    # Higher leverage means more debt financing relative to equity
    d["leverage"] = d["assets"] / d["equity"]

    # -----------------------------
    # Create lagged assets for value-weighting
    # Using t-1 assets as weights avoids mechanical correlation between
    # weights and period t ratios (Adrian & Shin recommendation)
    # -----------------------------
    d["assets_lag"] = d.groupby("bank_id", sort=False)["assets"].shift(1)

    # -----------------------------
    # Define weighted mean function
    # Value-weighted average: each bank weighted by its (lagged) size
    # This gives more influence to larger banks, as in Adrian & Shin
    # -----------------------------
    def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
        """
        Calculate weighted average, handling missing values.
        
        Args:
            values: Series of values to average
            weights: Series of weights (must be positive and finite)
            
        Returns:
            Weighted mean, or NaN if no valid observations
        """
        # Only use observations where both value and weight are valid and weight > 0
        ok = values.notna() & weights.notna() & np.isfinite(weights) & (weights > 0)
        if int(ok.sum()) == 0:
            return np.nan
        return float(np.average(values[ok].to_numpy(), weights=weights[ok].to_numpy()))

    # -----------------------------
    # Aggregate function for groupby
    # Computes value-weighted mean for each risk indicator
    # -----------------------------
    def _agg(grp: pd.DataFrame) -> pd.Series:
        """
        Aggregate a quarter's data across all banks.
        
        Args:
            grp: DataFrame for a single quarter containing all banks
            
        Returns:
            Series with weighted means for each risk indicator
        """
        w = grp["assets_lag"]
        return pd.Series(
            {
                "unit_var": _weighted_mean(grp["unit_var"], w),
                "leverage": _weighted_mean(grp["leverage"], w),
                "var_to_equity": _weighted_mean(grp["var_to_equity"], w),
            }
        )

    # -----------------------------
    # Aggregate across banks for each quarter
    # Result: one observation per quarter (value-weighted average across banks)
    # -----------------------------
    g = d.groupby("quarter", as_index=False).apply(_agg, include_groups=False).sort_values("quarter")
    g = g.dropna(subset=["unit_var", "leverage", "var_to_equity"]).reset_index(drop=True)

    if g.empty:
        raise ValueError("No valid observations after aggregation (check weights / lagging).")

    # -----------------------------
    # Standardize relative to pre-crisis period
    # This transforms each series to z-scores (mean=0, std=1 in pre-period)
    # Y-axis then shows "number of standard deviations from normal state"
    # Makes it easy to compare different indicators on same plot
    # -----------------------------
    pre_mask = (g["quarter"] >= PRE_START) & (g["quarter"] <= PRE_END)
    pre = g.loc[pre_mask].copy()
    
    if pre.empty:
        raise ValueError(
            "Pre-period is empty after aggregation. "
            "Adjust PRE_START/PRE_END so it overlaps your data."
        )

    # Apply z-score transformation to each indicator
    for col in ["unit_var", "leverage", "var_to_equity"]:
        mu = float(pre[col].mean())  # Mean in pre-period (baseline "normal" level)
        sd = float(pre[col].std())    # Standard deviation in pre-period
        
        if not np.isfinite(sd) or sd == 0.0:
            raise ValueError(f"Cannot standardize '{col}' (std is zero/missing in pre-period).")
        
        # Z-score: (x - mean) / std
        g[col] = (g[col] - mu) / sd

    # -----------------------------
    # Create the plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10.5, 5.6))

    # Gray background band marks ±2 standard deviations ("normal" volatility range)
    # Values outside this band represent unusual market conditions
    ax.axhspan(-2, 2, color="0.85", zorder=0)
    
    # Black horizontal line at zero represents pre-period average
    # Series above/below this line are above/below their historical norm
    ax.axhline(0, color="black", linewidth=1.0, zorder=1)

    # Plot three risk indicators
    # Dashed lines for Unit VaR and Leverage (more volatile, secondary indicators)
    ax.plot(g["quarter"], g["unit_var"], color="tab:blue", linestyle="--", linewidth=2.5, label="Unit VaR", zorder=2)
    ax.plot(g["quarter"], g["leverage"], color="tab:green", linestyle="--", linewidth=2.5, label="Leverage", zorder=2)
    
    # Solid line for VaR/E (primary indicator of risk relative to capital buffer)
    ax.plot(g["quarter"], g["var_to_equity"], color="tab:red", linestyle="-", linewidth=2.5, label="VaR/E", zorder=3)

    # Labels and legend
    ax.set_title("Risk and balance sheet adjustment")
    ax.set_ylabel("Pre-Period Standard Deviations")
    ax.legend(loc="upper left", frameon=False)

    # Format x-axis: show June and December of each year
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[6, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

    # Set y-limits to match Adrian & Shin Figure 5 style
    # Adjust if your data has different range
    ax.set_ylim(-15, 10)

    # Mirror y-axis on right side (cosmetic, matches journal figure style)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(ax.get_yticks())
    ax2.set_ylabel("")

    fig.tight_layout()
    return fig


def main() -> None:
    """
    Main execution function.
    
    1. Loads and merges data from CSV files
    2. Creates Adrian & Shin Figure 5 replication
    3. Displays the plot
    """
    # Load and prepare data
    df = load_df_from_project_outputs()
    
    # Create plot
    fig = plot_adrian_shin_figure_5(df)
    
    # Display plot
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()