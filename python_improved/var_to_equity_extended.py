"""
Figure: VaR/E levels (NO standardization) — sector aggregate

What this script does (in plain English)
----------------------------------------
1) Read balance sheet data (assets and equity) from the balance sheet panel file.
2) Read VaR(99%) data from the VaR file.
3) Make bank names and dates compatible across the two files.
4) Merge the two datasets on (bank, quarter-end date).
5) Compute VaR/E for each bank-quarter:  VaR(99%) / Common Equity.
6) Aggregate to a "sector" series each quarter using value-weights:
     weight = lagged total assets (assets from the previous quarter).
7) Plot the aggregated VaR/E series in LEVELS (no standardization).

Input files
-----------
- data/processed/balance_sheet_panel_balanced.csv
    must include: bank_id, period_end_date, total_assets, common_equity_total
- output/data/var_99.csv
    must include: bank, date, var_99_boa_factor

Output
------
- output/figures/figure_var_to_equity_levels.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# -----------------------------
# File locations (project paths)
# -----------------------------
BASE_FILE = Path("data/processed/balance_sheet_panel_balanced.csv")
VAR_FILE  = Path("output/data/var_99.csv")
OUTPUT_FILE = Path("output/figures/figure_var_to_equity_levels.png")


# -----------------------------
# Sample window (quarter-end dates)
# -----------------------------
SAMPLE_START = pd.Timestamp("2014-03-31")  # 2014Q1
SAMPLE_END   = pd.Timestamp("2025-09-30")  # 2025Q3


# VaR column to use (already harmonized to 99% in your pipeline)
VAR99_COL = "var_99_boa_factor"


def norm_bank(x: str) -> str:
    """
    Make bank identifiers comparable across files.

    Example: "Bank of America", "bank_of_america", "BoA" → becomes a simplified key.
    Rule: lowercase and remove all characters except letters and numbers.
    """
    return "".join(ch for ch in str(x).lower().strip() if ch.isalnum())


def main():
    # 1) Read the two input files
    base = pd.read_csv(BASE_FILE)
    var = pd.read_csv(VAR_FILE)

    # 2) Keep only the columns we actually need
    base = base[["bank_id", "period_end_date", "total_assets", "common_equity_total"]].copy()
    var  = var[["bank", "date", VAR99_COL]].copy()

    # 3) Convert date columns to real dates
    base["period_end_date"] = pd.to_datetime(base["period_end_date"], errors="coerce")
    var["date"] = pd.to_datetime(var["date"], errors="coerce")

    # 4) Convert numeric columns to numbers
    base["total_assets"] = pd.to_numeric(base["total_assets"], errors="coerce")
    base["common_equity_total"] = pd.to_numeric(base["common_equity_total"], errors="coerce")
    var[VAR99_COL] = pd.to_numeric(var[VAR99_COL], errors="coerce")

    # 5) Create a common bank key for merging (because the two files use different bank labels)
    base["bank_key"] = base["bank_id"].apply(norm_bank)
    var["bank_key"] = var["bank"].apply(norm_bank)

    # 6) Make the VaR date column name match the balance sheet date column name
    var = var.rename(columns={"date": "period_end_date"})

    # 7) Merge VaR and balance sheet on (bank_key, quarter-end date)
    df = base.merge(
        var[["bank_key", "period_end_date", VAR99_COL]],
        on=["bank_key", "period_end_date"],
        how="inner",
    )

    # 8) Restrict to the sample period (2014Q1–2025Q3)
    df = df[(df["period_end_date"] >= SAMPLE_START) & (df["period_end_date"] <= SAMPLE_END)].copy()

    # 9) Basic cleaning: drop missing values and require strictly positive levels
    df = df.dropna(subset=["total_assets", "common_equity_total", VAR99_COL])
    df = df[(df["total_assets"] > 0) & (df["common_equity_total"] > 0) & (df[VAR99_COL] > 0)].copy()

    # 10) Sort within each bank and compute lagged assets (assets from previous quarter)
    #     These lagged assets will be the value-weights in the aggregation.
    df = df.sort_values(["bank_key", "period_end_date"]).copy()
    df["assets_lag"] = df.groupby("bank_key")["total_assets"].shift(1)

    # 11) Compute VaR/E for each bank-quarter (LEVELS, no standardization)
    df["var_to_equity"] = df[VAR99_COL] / df["common_equity_total"]

    # 12) Drop first quarter per bank (because assets_lag is missing there)
    #     Also remove any non-finite or non-positive VaR/E values.
    df = df.dropna(subset=["assets_lag", "var_to_equity"])
    df = df[(df["assets_lag"] > 0) & np.isfinite(df["var_to_equity"]) & (df["var_to_equity"] > 0)].copy()

    # 13) Aggregate to a single sector series per quarter (value-weighted average)
    #     weights = lagged assets
    def wavg(x, w):
        return float(np.average(x.to_numpy(), weights=w.to_numpy()))

    g = (
        df.groupby("period_end_date", as_index=False)
          .apply(lambda grp: pd.Series({
              "var_to_equity": wavg(grp["var_to_equity"], grp["assets_lag"])
          }), include_groups=False)
          .sort_values("period_end_date")
          .reset_index(drop=True)
    )

    # 14) Plot the aggregated VaR/E time series
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.plot(g["period_end_date"], g["var_to_equity"], linewidth=2.4, label="VaR/E (99%)")

    ax.set_ylabel("VaR/E (levels)")
    ax.set_title("VaR/E (99%) — Levels (No Standardization)")
    ax.grid(True, linestyle="--", alpha=0.6)

    # x-axis formatting (show ticks at Jun and Dec)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[6, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

    ax.legend(loc="upper left", frameon=False)

    # 15) Save output figure
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
