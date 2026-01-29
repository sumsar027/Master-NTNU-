"""
Replicate Adrian & Shin Figure 5-style plot (risk + balance sheet adjustment),
for 2014Q1-2025Q3, using:
- Unit VaR = VaR / Assets
- VaR/E   = VaR / Book Equity
- Leverage = Assets / Book Equity

Uses harmonized 99% VaR with BoA factor (*2) for banks without reported 99% VaR.

Key choices in this script:
1) Uses ratios in LEVELS (not log-ratios).
2) Standardizes relative to PRE-PERIOD (default: 2014Q1-2019Q4).
3) Value-weighted average with LAGGED assets as weights.
4) Filters sample to 2014Q1-2025Q3.
5) Uses 99% VaR (instead of 95%).
6) Prioritizes var_99_boa_factor (uses reported 99% + factor *2 conversion).
7) Standardize per bank first, then aggregate across banks.

Inputs:
- data/processed/merged_quarterly_balanced.csv   (balance sheet panel)
- output/data/merged_with_var_99_dual_methods.csv (VaR panel, incl. 99% harmonization)

Output:
- output/figures/figure5_var99_boa.png
"""

from __future__ import annotations
from pathlib import Path

import os
import numpy as np
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
# BASE_FILE: balance sheet panel (assets, equity, etc.) by bank and quarter
# VAR_FILE: VaR panel by bank and quarter (includes harmonized 99% VaR)
BASE_FILE = Path("data/processed/merged_quarterly_balanced.csv")
VAR_FILE = Path("output/data/merged_with_var_99_dual_methods.csv")

# -----------------------------
# Bank ID harmonization
# -----------------------------
def normalize_bank_id(bank_id: str) -> str:
    """
    Normalize bank identifiers to a consistent canonical format.

    Why this exists:
    - The two input files may refer to the same bank using different spellings.
    - We need consistent bank_id values so that merging on (bank_id, period_end_date) works correctly.

    Examples:
    - "Bank of America", "bankofamerica", "bank_of_america" -> "bank_of_america"
    - "JPMorgan Chase", "jpmorgan", "jpmorganchase" -> "jpmorgan_chase"
    """
    bank_id = str(bank_id).lower().strip()
    normalized = bank_id.replace(" ", "").replace("_", "").replace("-", "").replace(".", "")

    if "bankofamerica" in normalized or "bofa" == normalized:
        return "bank_of_america"
    elif "citigroup" in normalized or "citibank" in normalized or "citi" == normalized:
        return "citibank"
    elif "wellsfargo" in normalized or normalized == "wells":
        return "wells_fargo"
    elif "jpmorgan" in normalized or "jpm" == normalized:
        return "jpmorgan_chase"
    elif "goldmansachs" in normalized or "goldman" in normalized:
        return "goldmansachs"
    elif "morganstanley" in normalized:
        return "morganstanley"
    elif "fifththird" in normalized:
        return "fifththird"
    elif "keycorp" in normalized or "keybank" in normalized:
        return "keycorp"
    elif "pnc" in normalized:
        return "pnc"
    elif "usbancorp" in normalized or "usbank" in normalized:
        return "usbancorp"
    elif "regionsfinancial" in normalized or "regions" in normalized:
        return "regionsfinancial"
    else:
        # Fallback: make a safe identifier (spaces/punctuation removed)
        return bank_id.replace(" ", "_").replace("-", "_").replace(".", "")

# -----------------------------
# Plot / sample settings
# -----------------------------
# Analysis window (what we keep after merging)
SAMPLE_START = pd.Timestamp("2014-03-31")  # 2014Q1 quarter-end
SAMPLE_END = pd.Timestamp("2025-09-30")    # 2025Q3 quarter-end

# Pre-period used for standardization (z-scores)
# Interpretation: series are expressed in "pre-period standard deviations"
PRE_START = pd.Timestamp("2014-03-31")
PRE_END = pd.Timestamp("2019-12-31")

# -----------------------------
# Column choices
# -----------------------------
# Preferred equity column name (if present and populated)
EQUITY_COL = "common_equity_total"

# Candidate columns: the script auto-selects what exists with the most data
EQUITY_CANDIDATES = [
    EQUITY_COL,
    "shareholders_equity_common",
    "common_equity_total",
    "common_equity_attributable_to_parent_shareholders",
    "shareholders_equity_attributable_to_parent_shareholders_total",
    "tangible_total_equity",
    "book_value_excluding_other_equity",
    "total_shareholders_equity_including_minority_interest_hybrid_debt",
    # Seen in merged_quarterly_balanced.csv exports
    "financial_summary_Common Equity - Total",
    "financial_summary_Tangible Total Equity",
    "balance_Common Equity - Total",
    "balance_Common Equity Attributable to Parent Shareholders",
    "balance_Shareholders Equity - Common",
    "balance_Book Value excluding Other Equity",
    "balance_Tangible Total Equity",
    "balance_Total Shareholders' Equity - including Minority Interest & Hybrid Debt",
]

ASSET_CANDIDATES = [
    "total_assets_2",
    "total_assets",
    "assets",
    # Seen in merged_quarterly_balanced.csv exports
    "financial_summary_Total Assets",
    "balance_Total Assets",
]

# Uses var_99_boa_factor, which equals reported 99% VaR when available and otherwise uses a ×2 conversion
VAR99_CANDIDATES = ["var_99_boa_factor"]



def _require_columns(df: pd.DataFrame, cols: list[str], *, where: str) -> None:
    """Hard check: raise an error if required columns are missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s) in {where}: " + ", ".join(missing))


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Return the first candidate column that exists in the dataframe.
    Used when we have a preferred ordering.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _best_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Return the candidate column with the most non-missing values.
    Used when multiple possible column names exist and we want the most complete series.
    """
    present = [c for c in candidates if c in df.columns]
    if not present:
        return None
    counts = [(int(df[c].notna().sum()), c) for c in present]
    counts.sort(reverse=True)
    best_count, best_col = counts[0]
    if best_count == 0:
        return None
    return best_col


def _to_datetime(series: pd.Series) -> pd.Series:
    """
    Convert various date/quarter formats into a proper datetime series.

    Important:
    - If dates are represented as "YYYYQn", we convert to QUARTER-END (e.g., 2014Q1 -> 2014-03-31)
      so that quarter labels match quarter-end dates across files.
    """
    if isinstance(series.dtype, pd.PeriodDtype):
        # Ensure quarter-end dating for Period columns
        return series.dt.to_timestamp(how="end").dt.normalize()
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    # Handle common "YYYYQn" quarter strings
    s = series.astype(str).str.strip()
    s_q = s.str.upper().str.replace(r"[\s\-_]", "", regex=True)
    is_q = s_q.str.match(r"^\d{4}Q[1-4]$")

    # If the column is mostly quarter strings, treat it as quarter identifiers
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

    # Fallback: try generic parsing
    return pd.to_datetime(s, errors="coerce")


def _to_numeric(series: pd.Series) -> pd.Series:
    """
    Convert a column to numeric robustly:
    - removes commas
    - converts "(123)" to -123
    - strips non-numeric characters
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan, "N/A": np.nan, "na": np.nan})
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace(r"[^\d\.\-eE+]", "", regex=True)

    return pd.to_numeric(s, errors="coerce")


def load_df_from_project_outputs(
    base_file: Path = BASE_FILE,
    var_file: Path = VAR_FILE,
) -> pd.DataFrame:
    """
    Load and merge balance sheet data with 99% VaR data.

    High-level steps:
    1) Read base and VaR CSV files
    2) Identify bank and date columns in each file (they may differ by export)
    3) Normalize bank identifiers and date formats
    4) Merge on (bank_id, period_end_date)
    5) Filter to analysis window
    6) Select best available assets/equity columns
    7) Return clean dataframe with: bank_id, quarter, assets, equity, var99
    """
    if not base_file.exists():
        raise FileNotFoundError(f"Missing file: {base_file}")
    if not var_file.exists():
        raise FileNotFoundError(f"Missing file: {var_file}")

    base = pd.read_csv(base_file)
    var = pd.read_csv(var_file)

    # Identify entity (bank) column in each file and rename to a common name: "bank_id"
    base_entity = _find_column(base, ["bank_id", "bank", "entity", "ticker"])
    if base_entity is None:
        raise KeyError("Could not identify entity column in base")
    if base_entity != "bank_id":
        base = base.rename(columns={base_entity: "bank_id"})

    var_entity = _find_column(var, ["bank_id", "bank", "entity", "ticker"])
    if var_entity is None:
        raise KeyError("Could not identify entity column in var")
    if var_entity != "bank_id":
        var = var.rename(columns={var_entity: "bank_id"})

    # Identify time column in each file and rename to a common name: "period_end_date"
    base_time = _find_column(base, ["period_end_date", "quarter", "date", "time"])
    if base_time is None:
        raise KeyError("Could not identify time column in base")
    if base_time != "period_end_date":
        base = base.rename(columns={base_time: "period_end_date"})

    var_time = _find_column(var, ["period_end_date", "quarter", "date", "time"])
    if var_time is None:
        raise KeyError("Could not identify time column in var")
    if var_time != "period_end_date":
        var = var.rename(columns={var_time: "period_end_date"})

    # Normalize bank IDs so the same bank uses the same label in both datasets
    base["bank_id"] = base["bank_id"].apply(normalize_bank_id)
    var["bank_id"] = var["bank_id"].apply(normalize_bank_id)

    # Normalize dates so quarters align (e.g., 2014Q1 -> 2014-03-31)
    base["period_end_date"] = _to_datetime(base["period_end_date"])
    var["period_end_date"] = _to_datetime(var["period_end_date"])

    # Drop rows that cannot be matched because bank/date is missing
    base = base.dropna(subset=["bank_id", "period_end_date"])
    var = var.dropna(subset=["bank_id", "period_end_date"])

    # Select which VaR column to use (prefers harmonized 99% series with BoA-factor)
    var99_col = _find_column(var, VAR99_CANDIDATES)
    if var99_col is None:
        raise KeyError(f"Missing VaR column in var file. Expected one of: {VAR99_CANDIDATES}")

    print(f"Using VaR column: {var99_col}")
    var[var99_col] = _to_numeric(var[var99_col])

    # Merge datasets: keep only (bank, quarter) pairs that exist in BOTH files
    merged = base.merge(
        var[["bank_id", "period_end_date", var99_col]].rename(columns={var99_col: "var_99_level"}),
        on=["bank_id", "period_end_date"],
        how="inner",
    )

    # Keep only the analysis window we want to plot
    merged = merged[(merged["period_end_date"] >= SAMPLE_START) & (merged["period_end_date"] <= SAMPLE_END)].copy()
    if merged.empty:
        raise ValueError("No observations after merging + filtering to 2014Q1–2025Q3.")

    # Choose the most complete assets and equity columns available
    assets_col = _best_column(merged, ASSET_CANDIDATES)
    if assets_col is None:
        raise KeyError(f"Missing assets data (expected one of {ASSET_CANDIDATES}).")

    if EQUITY_COL in merged.columns and int(merged[EQUITY_COL].notna().sum()) > 0:
        equity_col = EQUITY_COL
    else:
        equity_col = _best_column(merged, EQUITY_CANDIDATES)
    if equity_col is None:
        raise KeyError("No usable equity column found.")

    # Convert key numeric columns to floats
    assets = _to_numeric(merged[assets_col])
    equity = _to_numeric(merged[equity_col])
    var99 = _to_numeric(merged["var_99_level"])

    # Build a clean analysis dataframe with consistent column names
    df = pd.DataFrame(
        {
            "bank_id": merged["bank_id"].astype(str),
            "quarter": merged["period_end_date"],
            "assets": assets,
            "equity": equity,
            "var99": var99,
        }
    )

    # Drop missing and non-positive values (ratios require positive denominators)
    n0 = len(df)
    df = df.dropna(subset=["bank_id", "quarter", "assets", "equity", "var99"])
    n_dropna = len(df)
    df = df[(df["assets"] > 0) & (df["equity"] > 0) & (df["var99"] > 0)].copy()
    n_pos = len(df)

    df = df.sort_values(["bank_id", "quarter"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"No valid rows after cleaning. "
            f"Rows: start={n0}, after_dropna={n_dropna}, after_positive_filter={n_pos}."
        )

    print(f"Loaded {len(df)} observations for {df['bank_id'].nunique()} banks")
    return df


def plot_adrian_shin_figure_5(df: pd.DataFrame):
    """
    Create a Figure 5-style plot: risk measures and balance-sheet adjustment over time.

    Conceptual steps:
    1) Compute ratios for each bank and quarter:
       - Unit VaR    = VaR / Assets
       - VaR / Equity = VaR / Equity
       - Leverage    = Assets / Equity

    2) Standardize within each bank using the PRE-period:
       - For each bank, compute mean and std in PRE-period
       - Express all observations as z-scores (pre-period standard deviations)

    3) Aggregate across banks each quarter:
       - Compute value-weighted averages of the standardized series
       - Weights are lagged assets (previous quarter), so large banks get larger weight

    4) Plot the three aggregate series and save as a PNG.
    """
    _require_columns(df, ["bank_id", "quarter", "assets", "equity", "var99"], where="df")

    # Set a writable Matplotlib config directory (avoids issues on some systems)
    if "MPLCONFIGDIR" not in os.environ:
        mpl_dir = Path("output") / ".mplconfig"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    d = df.copy()
    d["quarter"] = _to_datetime(d["quarter"])
    d = d.sort_values(["bank_id", "quarter"]).reset_index(drop=True)

    # Compute ratios (per bank, per quarter)
    # Unit VaR: risk per unit of assets (size-adjusted VaR)
    d["unit_var"] = d["var99"] / d["assets"]
    # VaR/E: VaR relative to book equity (risk relative to equity base)
    d["var_to_equity"] = d["var99"] / d["equity"]
    # Leverage: assets per unit of book equity
    d["leverage"] = d["assets"] / d["equity"]

    # Lagged assets are used as weights (value-weighting with last quarter's size)
    d["assets_lag"] = d.groupby("bank_id", sort=False)["assets"].shift(1)

    # ------------------------------------------------------------
    # Step 1: Standardize within each bank relative to the PRE-period
    # ------------------------------------------------------------
    pre_mask_bank = (d["quarter"] >= PRE_START) & (d["quarter"] <= PRE_END)
    if int(pre_mask_bank.sum()) == 0:
        raise ValueError("Pre-period is empty on bank-level data (before aggregation).")

    cols_to_std = ["unit_var", "leverage", "var_to_equity"]

    def _zscore_within_bank(grp: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize each bank's series relative to its own pre-period distribution.
        Output values are z-scores: (x - pre_mean) / pre_std.
        """
        out = grp.copy()
        pre = out.loc[pre_mask_bank.loc[out.index], cols_to_std]

        # If a bank has no pre-period observations, mark its series as NaN (it drops out later)
        if pre.empty:
            for c in cols_to_std:
                out[c] = np.nan
            return out

        for c in cols_to_std:
            mu = float(pre[c].mean())
            sd = float(pre[c].std())
            if (not np.isfinite(mu)) or (not np.isfinite(sd)) or sd == 0.0:
                out[c] = np.nan
            else:
                out[c] = (out[c] - mu) / sd
        return out

    d = d.groupby("bank_id", group_keys=False).apply(_zscore_within_bank)

    # ------------------------------------------------------------
    # Step 2: Aggregate standardized series across banks (value-weighted)
    # ------------------------------------------------------------
    def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
        """Compute a weighted mean, ignoring missing/invalid weights and values."""
        ok = values.notna() & weights.notna() & np.isfinite(weights) & (weights > 0)
        if int(ok.sum()) == 0:
            return np.nan
        return float(np.average(values[ok].to_numpy(), weights=weights[ok].to_numpy()))

    def _agg(grp: pd.DataFrame) -> pd.Series:
        """For a given quarter, compute value-weighted averages of the standardized series."""
        w = grp["assets_lag"]
        return pd.Series(
            {
                "unit_var": _weighted_mean(grp["unit_var"], w),
                "leverage": _weighted_mean(grp["leverage"], w),
                "var_to_equity": _weighted_mean(grp["var_to_equity"], w),
            }
        )

    g = (
        d.groupby("quarter", as_index=False)
         .apply(_agg, include_groups=False)
         .sort_values("quarter")
         .dropna(subset=["unit_var", "leverage", "var_to_equity"])
         .reset_index(drop=True)
    )

    if g.empty:
        raise ValueError("No valid observations after aggregation of standardized series.")

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 5.6))

    # Grey band for ±2 standard deviations and a zero line
    ax.axhspan(-2, 2, color="0.85", zorder=0)
    ax.axhline(0, color="black", linewidth=1.0, zorder=1)

    # Three aggregate series
    ax.plot(g["quarter"], g["unit_var"], color="tab:blue", linestyle="--", linewidth=2.5, label="Unit VaR (99%)", zorder=2)
    ax.plot(g["quarter"], g["leverage"], color="tab:green", linestyle="--", linewidth=2.5, label="Leverage", zorder=2)
    ax.plot(g["quarter"], g["var_to_equity"], color="tab:red", linestyle="-", linewidth=2.5, label="VaR/E (99%)", zorder=3)

    ax.set_title("Risk and balance sheet adjustment (99% VaR, BoA factor)")
    ax.set_ylabel("Pre-Period Standard Deviations")
    ax.legend(loc="upper left", frameon=False)

    # X-axis formatting (tick labels at Jun/Dec; rotate for readability)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[6, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

    ax.set_ylim(-15, 10)

    # Mirror axis on the right (visual symmetry only)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(ax.get_yticks())
    ax2.set_ylabel("")

    fig.tight_layout()
    return fig


def main() -> None:
    """Run the full pipeline: load data, create figure, save to disk, and display."""
    df = load_df_from_project_outputs()
    fig = plot_adrian_shin_figure_5(df)

    out_dir = Path("output/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "figure5_var99_boa.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path.resolve()}")

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
