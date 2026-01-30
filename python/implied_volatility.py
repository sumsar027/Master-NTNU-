"""
Replicate Adrian & Shin Figure 4-style plot: implied volatility vs Unit VaR.

Objective
---------
Produce a quarterly time-series figure comparing:
1) Market-based risk proxy: implied volatility (VIX)
2) Balance-sheet-based risk proxy: Unit VaR = VaR(99%) / Total Assets

Both series are standardized relative to a common PRE period (2014Q1–2019Q4):
    z_t = (x_t - mean_pre) / std_pre

This matches the logic of plotting "deviations from normal times" using a
pre-crisis baseline.

Inputs
------
- Daily VIX from FRED CSV (e.g., data/raw/VIXCLS.csv)
- Bank-quarter panel + VaR from our pipeline (load_df_from_project_outputs)

Method
------
A) Unit VaR (bank panel):
   1) unit_var_{i,t} = var99_{i,t} / assets_{i,t}
   2) Standardize within bank using that bank's PRE-period mean and std
   3) Aggregate across banks each quarter using value weights (lagged assets)

B) VIX (market series):
   1) Convert daily VIX to quarterly mean
   2) Standardize using the same PRE period

C) Merge on quarter-end dates and plot both standardized series with a ±2 band.

Output
------
output/figures/implied_vol_vs_unit_var.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Import sample boundaries and the bank-panel loader from the existing figure script
try:
    from pipeline.figure_var_to_equity_ratio import (
        PRE_END, PRE_START, SAMPLE_END, SAMPLE_START, load_df_from_project_outputs
    )
except ModuleNotFoundError:
    from figure_var_to_equity_ratio import (
        PRE_END, PRE_START, SAMPLE_END, SAMPLE_START, load_df_from_project_outputs
    )


# ============================================================================
# VIX: load daily data, convert to quarterly mean, standardize to PRE
# ============================================================================

def load_vix_data(vix_filepath: Path) -> pd.DataFrame:
    """
    Load daily VIX data from a FRED-style CSV.

    The function is robust to minor differences in column names across downloads:
    - identifies the date column by searching for 'date' in the column name
    - identifies the VIX column by checking common names (VIXCLS, VIX)
    """
    raw = pd.read_csv(vix_filepath)

    # Identify date column (fallback: first column)
    date_col = next((c for c in raw.columns if "date" in c.lower()), raw.columns[0])

    # Identify VIX value column (fallback: first non-date column)
    candidates = ["VIXCLS", "vixcls", "VIX", "vix"]
    value_col = next((c for c in candidates if c in raw.columns),
                     [c for c in raw.columns if c != date_col][0])

    vix = raw[[date_col, value_col]].copy()
    vix.columns = ["date", "vix"]

    vix["date"] = pd.to_datetime(vix["date"], errors="coerce")
    vix["vix"] = pd.to_numeric(vix["vix"], errors="coerce")

    vix = vix.dropna().sort_values("date").reset_index(drop=True)
    return vix


def convert_vix_to_quarterly_mean(vix_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily VIX to quarterly frequency using the within-quarter mean.

    Critical detail:
    - we map each observation to the QUARTER-END date and normalize time to 00:00:00
      to ensure exact matching when merging with bank panel quarters.
    """
    df = vix_daily.copy()
    df["quarter"] = (
        df["date"]
        .dt.to_period("Q")
        .dt.to_timestamp(how="end")
        .dt.normalize()
    )
    quarterly = df.groupby("quarter", as_index=False)["vix"].mean()
    quarterly = quarterly.sort_values("quarter").reset_index(drop=True)
    return quarterly


def standardize_to_pre_period(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Standardize df[value_col] to PRE period moments:
        z = (x - mean_pre) / std_pre

    If std_pre is zero or missing, we set the series to NaN to avoid silent infinities.
    """
    pre_mask = (df["quarter"] >= PRE_START) & (df["quarter"] <= PRE_END)
    pre_values = df.loc[pre_mask, value_col]

    mean_pre = pre_values.mean()
    std_pre = pre_values.std()

    if pd.isna(std_pre) or std_pre == 0:
        df[value_col] = np.nan
        return df

    df[value_col] = (df[value_col] - mean_pre) / std_pre
    return df


# ============================================================================
# Unit VaR: compute per bank, standardize within bank, value-weighted aggregate
# ============================================================================

def compute_unit_var_aggregate() -> pd.DataFrame:
    """
    Compute the sector-level (value-weighted) standardized Unit VaR series.

    Value weights:
    - lagged assets (assets_{t-1}), so weights are predetermined relative to period t
    """
    df = load_df_from_project_outputs()

    # Ensure quarter is a clean datetime quarter-end (no time component)
    df["quarter"] = pd.to_datetime(df["quarter"], errors="coerce").dt.normalize()

    # Restrict to the analysis sample
    df = df[(df["quarter"] >= SAMPLE_START) & (df["quarter"] <= SAMPLE_END)].copy()

    # Unit VaR at the bank level
    df["unit_var"] = df["var99"] / df["assets"]

    # Lagged assets for value-weighting
    df = df.sort_values(["bank_id", "quarter"])
    df["assets_lag"] = df.groupby("bank_id")["assets"].shift(1)

    def standardize_bank(g: pd.DataFrame) -> pd.DataFrame:
        """Standardize unit_var within one bank using that bank's PRE-period moments."""
        pre_mask = (g["quarter"] >= PRE_START) & (g["quarter"] <= PRE_END)
        pre_values = g.loc[pre_mask, "unit_var"]

        mean_pre = pre_values.mean()
        std_pre = pre_values.std()

        if pd.isna(std_pre) or std_pre == 0:
            g["unit_var"] = np.nan
            return g

        g["unit_var"] = (g["unit_var"] - mean_pre) / std_pre
        return g

    df = df.groupby("bank_id", group_keys=False).apply(standardize_bank, include_groups=False)

    def weighted_mean(grp: pd.DataFrame) -> float:
        """Value-weighted mean within a quarter, using lagged assets as weights."""
        valid = grp["unit_var"].notna() & grp["assets_lag"].notna() & (grp["assets_lag"] > 0)
        if valid.sum() == 0:
            return np.nan
        return float(np.average(grp.loc[valid, "unit_var"], weights=grp.loc[valid, "assets_lag"]))

    agg = df.groupby("quarter").apply(weighted_mean, include_groups=False).reset_index()
    agg.columns = ["quarter", "unit_var"]
    agg = agg.dropna().sort_values("quarter").reset_index(drop=True)
    return agg


# ============================================================================
# Create plot: merge VIX + Unit VaR and plot both standardized series
# ============================================================================

def create_plot(vix_filepath: Path, output_filepath: Path):
    # --- VIX pipeline ---
    vix_daily = load_vix_data(vix_filepath)
    vix_q = convert_vix_to_quarterly_mean(vix_daily)

    # Restrict to sample period and standardize to PRE
    vix_q = vix_q[(vix_q["quarter"] >= SAMPLE_START) & (vix_q["quarter"] <= SAMPLE_END)].copy()
    vix_q = standardize_to_pre_period(vix_q, "vix").rename(columns={"vix": "implied_vol"})

    # --- Unit VaR pipeline ---
    unit_var = compute_unit_var_aggregate()

    # --- Merge on quarter-end dates ---
    merged = unit_var.merge(vix_q, on="quarter", how="inner").sort_values("quarter").reset_index(drop=True)

    # Hard guards: fail loudly if the merge is empty or if a series became entirely NaN
    if merged.empty:
        raise ValueError("Merge produced 0 rows. Check that both series use identical quarter-end datetimes.")
    if merged["implied_vol"].isna().all() or merged["unit_var"].isna().all():
        raise ValueError("A plotted series is entirely NaN. Check PRE-period standardization and data coverage.")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10.5, 5.6))

    ax.axhspan(-2, 2, color="0.85", zorder=0, label="±2 std")
    ax.axhline(0, color="black", linewidth=1.0, zorder=1)

    ax.plot(merged["quarter"], merged["implied_vol"],
            linewidth=2.5, label="Implied Vol (VIX)", zorder=3)
    ax.plot(merged["quarter"], merged["unit_var"],
            linestyle="--", linewidth=2.5, label="Unit VaR (99%)", zorder=2)

    ax.set_title("Risk measures: implied volatility vs unit VaR")
    ax.set_ylabel("Pre-period standard deviations")
    ax.legend(loc="upper left", frameon=False)

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[6, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

    # Cosmetic twin axis (optional)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(ax.get_yticks())
    ax2.set_ylabel("")

    fig.tight_layout()

    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_filepath, dpi=300, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    VIX_FILE = Path("data/raw/VIXCLS (1).csv")
    OUTPUT_FILE = Path("output/figures/implied_vol_vs_unit_var.png")
    create_plot(VIX_FILE, OUTPUT_FILE)
    # plt.show()  # For interactive inspection
