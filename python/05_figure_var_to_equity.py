"""
Replicate Adrian & Shin Figure 5-style plot (risk + balance sheet adjustment),
for 2014Q1–2025Q3, using:
- Unit VaR = VaR / Assets
- VaR/E   = VaR / Book Equity
- Leverage = Assets / Book Equity

Uses harmonized 99% VaR with BoA factor (×2) for banks without reported 99% VaR.

Key changes vs original:
1) Uses ratios in LEVELS (not log-ratios).
2) Standardizes relative to PRE-PERIOD (default: 2014Q1–2019Q4).
3) Value-weighted average with LAGGED assets as weights.
4) Filters sample to 2014Q1–2025Q3.
5) Uses 99% VaR (instead of 95%).
6) Prioritizes var_99_boa_factor (uses reported 99% + factor ×2 conversion).

Inputs:
- output/merged_quarterly_balanced.csv
- output/merged_with_var_99_dual_methods.csv

"""

from __future__ import annotations

from pathlib import Path

import os
import numpy as np
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
BASE_FILE = Path("data/processed/merged_quarterly_balanced.csv")
VAR_FILE = Path("output/merged_with_var_99_dual_methods.csv")

# -----------------------------
# Bank ID harmonization
# -----------------------------
BANK_MAP = {
    "bankofamerica": "bank_of_america",
    "citigroup": "citibank",
    "wellsfargo": "wells_fargo",
}

# -----------------------------
# Plot / sample settings
# -----------------------------
SAMPLE_START = pd.Timestamp("2014-03-31")  # 2014Q1 end
SAMPLE_END = pd.Timestamp("2025-09-30")    # 2025Q3 end

# Pre-period for standardization
PRE_START = pd.Timestamp("2014-03-31")
PRE_END = pd.Timestamp("2019-12-31")

# -----------------------------
# Column choices
# -----------------------------
EQUITY_COL = "total_shareholders_equity"

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

ASSET_CANDIDATES = ["total_assets_2", "total_assets", "assets"]

# Prioritize var_99_boa_factor (uses reported 99% + factor ×2 conversion)
VAR99_CANDIDATES = ["var_99_boa_factor", "var_99_x2", "var_99_gauss", "var_99_harmonized", "var_99_level", "var_99", "var_99_approx"]


def _require_columns(df: pd.DataFrame, cols: list[str], *, where: str) -> None:
    """Check that required columns exist in dataframe."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s) in {where}: " + ", ".join(missing))


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first existing column from a list of candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _best_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the column with the most non-missing values from candidates."""
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
    """Convert series to datetime, handling both Period and datetime types."""
    if isinstance(series.dtype, pd.PeriodDtype):
        return series.dt.to_timestamp()
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce")


def _to_numeric(series: pd.Series) -> pd.Series:
    """Robustly convert series to numeric."""
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
    """Load and merge balance sheet data with 99% VaR estimates."""
    
    if not base_file.exists():
        raise FileNotFoundError(f"Missing file: {base_file}")
    if not var_file.exists():
        raise FileNotFoundError(f"Missing file: {var_file}")

    base = pd.read_csv(base_file)
    var = pd.read_csv(var_file)

    # Identify entity column
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

    # Identify time column
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

    # Clean bank identifiers
    base["bank_id"] = base["bank_id"].astype(str).str.strip().str.lower().replace(BANK_MAP)
    var["bank_id"] = var["bank_id"].astype(str).str.strip().str.lower().replace(BANK_MAP)

    # Convert dates
    base["period_end_date"] = _to_datetime(base["period_end_date"])
    var["period_end_date"] = _to_datetime(var["period_end_date"])

    base = base.dropna(subset=["bank_id", "period_end_date"])
    var = var.dropna(subset=["bank_id", "period_end_date"])

    # Choose 99% VaR column (prioritizes var_99_boa_factor)
    var99_col = _find_column(var, VAR99_CANDIDATES)
    if var99_col is None:
        raise KeyError(f"Missing VaR column in var file. Expected one of: {VAR99_CANDIDATES}")

    print(f"Using VaR column: {var99_col}")
    var[var99_col] = _to_numeric(var[var99_col])

    # Merge
    merged = base.merge(
        var[["bank_id", "period_end_date", var99_col]].rename(columns={var99_col: "var_99_level"}),
        on=["bank_id", "period_end_date"],
        how="inner",
    )

    # Filter to analysis period
    merged = merged[(merged["period_end_date"] >= SAMPLE_START) & (merged["period_end_date"] <= SAMPLE_END)].copy()
    if merged.empty:
        raise ValueError("No observations after merging + filtering to 2014Q1–2025Q3.")

    # Choose best columns for assets and equity
    assets_col = _best_column(merged, ASSET_CANDIDATES)
    if assets_col is None:
        raise KeyError(f"Missing assets data (expected one of {ASSET_CANDIDATES}).")

    equity_col = None
    if EQUITY_COL in merged.columns and int(merged[EQUITY_COL].notna().sum()) > 0:
        equity_col = EQUITY_COL
    else:
        equity_col = _best_column(merged, EQUITY_CANDIDATES)
    if equity_col is None:
        raise KeyError(f"No usable equity column found.")

    # Convert to numeric
    assets = _to_numeric(merged[assets_col])
    equity = _to_numeric(merged[equity_col])
    var99 = _to_numeric(merged["var_99_level"])

    # Create clean output
    df = pd.DataFrame(
        {
            "bank_id": merged["bank_id"].astype(str),
            "quarter": merged["period_end_date"],
            "assets": assets,
            "equity": equity,
            "var99": var99,
        }
    )

    # Clean data
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
    """Create Adrian & Shin Figure 5 replication plot using 99% VaR."""
    
    _require_columns(df, ["bank_id", "quarter", "assets", "equity", "var99"], where="df")

    if "MPLCONFIGDIR" not in os.environ:
        mpl_dir = Path("output") / ".mplconfig"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    d = df.copy()
    d["quarter"] = _to_datetime(d["quarter"])
    d = d.sort_values(["bank_id", "quarter"]).reset_index(drop=True)

    # Calculate three risk indicators using 99% VaR
    d["unit_var"] = d["var99"] / d["assets"]
    d["var_to_equity"] = d["var99"] / d["equity"]
    d["leverage"] = d["assets"] / d["equity"]

    # Create lagged assets for value-weighting
    d["assets_lag"] = d.groupby("bank_id", sort=False)["assets"].shift(1)

    def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
        ok = values.notna() & weights.notna() & np.isfinite(weights) & (weights > 0)
        if int(ok.sum()) == 0:
            return np.nan
        return float(np.average(values[ok].to_numpy(), weights=weights[ok].to_numpy()))

    def _agg(grp: pd.DataFrame) -> pd.Series:
        w = grp["assets_lag"]
        return pd.Series(
            {
                "unit_var": _weighted_mean(grp["unit_var"], w),
                "leverage": _weighted_mean(grp["leverage"], w),
                "var_to_equity": _weighted_mean(grp["var_to_equity"], w),
            }
        )

    # Aggregate across banks
    g = d.groupby("quarter", as_index=False).apply(_agg, include_groups=False).sort_values("quarter")
    g = g.dropna(subset=["unit_var", "leverage", "var_to_equity"]).reset_index(drop=True)

    if g.empty:
        raise ValueError("No valid observations after aggregation.")

    # Standardize relative to pre-period
    pre_mask = (g["quarter"] >= PRE_START) & (g["quarter"] <= PRE_END)
    pre = g.loc[pre_mask].copy()
    
    if pre.empty:
        raise ValueError("Pre-period is empty after aggregation.")

    for col in ["unit_var", "leverage", "var_to_equity"]:
        mu = float(pre[col].mean())
        sd = float(pre[col].std())
        
        if not np.isfinite(sd) or sd == 0.0:
            raise ValueError(f"Cannot standardize '{col}' (std is zero/missing in pre-period).")
        
        g[col] = (g[col] - mu) / sd

    # Create plot
    fig, ax = plt.subplots(figsize=(10.5, 5.6))

    ax.axhspan(-2, 2, color="0.85", zorder=0)
    ax.axhline(0, color="black", linewidth=1.0, zorder=1)

    ax.plot(g["quarter"], g["unit_var"], color="tab:blue", linestyle="--", linewidth=2.5, label="Unit VaR (99%)", zorder=2)
    ax.plot(g["quarter"], g["leverage"], color="tab:green", linestyle="--", linewidth=2.5, label="Leverage", zorder=2)
    ax.plot(g["quarter"], g["var_to_equity"], color="tab:red", linestyle="-", linewidth=2.5, label="VaR/E (99%)", zorder=3)

    ax.set_title("Risk and balance sheet adjustment (99% VaR, BoA factor)")
    ax.set_ylabel("Pre-Period Standard Deviations")
    ax.legend(loc="upper left", frameon=False)

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[6, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

    ax.set_ylim(-15, 10)

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(ax.get_yticks())
    ax2.set_ylabel("")

    fig.tight_layout()
    return fig


def main() -> None:
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
