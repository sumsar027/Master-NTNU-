"""
Figure 5-style plot of risk and balance sheet adjustment, 2014Q1 - 2025Q3.

Base series (in levels before optional differencing):
- Unit VaR  = VaR(99%) / Assets
- VaR/E     = VaR(99%) / Book common equity
- Leverage  = Assets / Book common equity
- Equity    = Book common equity

Key choices:
1) Uses ratios in levels as baseline, in the spirit of Adrian & Shin Figure 5.
2) Standardizes relative to PRE-PERIOD (2014Q1 - 2019Q4).
3) Value-weighted average across banks using LAGGED assets as weights.
4) Filters sample to 2014Q1 - 2025Q3.
5) Uses 99% VaR from var_99_boa_factor.
6) Standardizes within each bank first, then aggregates across banks.
7) Can optionally use first differences for VaR/E and/or Leverage.

Inputs:
- data/processed/balance_sheet_panel_balanced.csv
- output/data/var_99.csv

Output:
- output/figures/var_to_equity/figure5_var99_boa_<suffix>.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# -----------------------------
# Paths (project-root based)
# -----------------------------
ROOT = Path(__file__).resolve().parents[2]
BASE_FILE = ROOT / "data" / "processed" / "balance_sheet_panel_balanced.csv"
VAR_FILE = ROOT / "output" / "data" / "var_99.csv"


# -----------------------------
# Sample settings
# -----------------------------
SAMPLE_START = pd.Timestamp("2014-03-31")  # 2014Q1
SAMPLE_END = pd.Timestamp("2025-09-30")    # 2025Q3

PRE_START = pd.Timestamp("2014-03-31")
PRE_END = pd.Timestamp("2019-12-31")       # 2019Q4


# -----------------------------
# Bank filter
# -----------------------------
BANK_IDS = ["bny", "statestreet"]
# custody banks: ["bny", "statestreet"]
# commercial banks: ["bankofamerica", "citigroup", "jpmorgan", "wellsfargo"]
# market makers: ["goldmansachs", "morganstanley"]
# all banks: None


# -----------------------------
# Toggles
# -----------------------------
USE_CHANGE_VAR_TO_EQUITY = False
USE_CHANGE_LEVERAGE = True


def out_file() -> Path:
    return ROOT / "output" / "figures" / "var_to_equity" / "figure5_var99_boa.png"


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Value-weighted mean with basic NA/valid-weight filtering."""
    ok = values.notna() & weights.notna() & np.isfinite(weights) & (weights > 0)
    if ok.sum() == 0:
        return np.nan
    return float(np.average(values[ok].to_numpy(), weights=weights[ok].to_numpy()))


def main() -> None:
    outfile = out_file()
    outfile.parent.mkdir(parents=True, exist_ok=True)

    # --- Load balance sheet panel ---
    base = pd.read_csv(BASE_FILE)
    base = base[["bank_id", "period_end_date", "total_assets", "common_equity_total"]].copy()
    base["bank_id"] = base["bank_id"].astype(str).str.lower().str.strip()
    base["period_end_date"] = pd.to_datetime(base["period_end_date"])

    # --- Load VaR (99%) ---
    var = pd.read_csv(VAR_FILE)
    var = var[["bank", "date", "var_99_boa_factor"]].copy()
    var = var.rename(columns={"bank": "bank_id", "date": "period_end_date", "var_99_boa_factor": "var99"})
    var["bank_id"] = var["bank_id"].astype(str).str.lower().str.strip()
    var["period_end_date"] = pd.to_datetime(var["period_end_date"])

    if BANK_IDS:
        base = base[base["bank_id"].isin(BANK_IDS)].copy()
        var = var[var["bank_id"].isin(BANK_IDS)].copy()

    # Ensure numeric
    base["total_assets"] = pd.to_numeric(base["total_assets"], errors="coerce")
    base["common_equity_total"] = pd.to_numeric(base["common_equity_total"], errors="coerce")
    var["var99"] = pd.to_numeric(var["var99"], errors="coerce")

    # --- Merge on (bank_id, date) ---
    d = base.merge(var, on=["bank_id", "period_end_date"], how="inner")

    print("Banks in VaR file:", var["bank_id"].nunique(), sorted(var["bank_id"].unique()))
    print("Banks in balance file:", base["bank_id"].nunique(), sorted(base["bank_id"].unique()))
    print("Banks after merge:", d["bank_id"].nunique(), sorted(d["bank_id"].unique()))

    missing_in_balance = set(var["bank_id"]) - set(base["bank_id"])
    print("VaR banks missing in balance:", sorted(missing_in_balance))

    print("Obs after merge:", len(d))
    print("Date range after merge:", d["period_end_date"].min(), "to", d["period_end_date"].max())

    # --- Filter sample window ---
    d = d[(d["period_end_date"] >= SAMPLE_START) & (d["period_end_date"] <= SAMPLE_END)].copy()

    # Drop invalid rows
    d = d.dropna(subset=["total_assets", "common_equity_total", "var99"])
    d = d[(d["total_assets"] > 0) & (d["common_equity_total"] > 0) & (d["var99"] > 0)].copy()

    # Sort for lagging/differencing
    d = d.sort_values(["bank_id", "period_end_date"]).reset_index(drop=True)

    # --- Construct ratios (levels) ---
    d["unit_var"] = d["var99"] / d["total_assets"]
    d["var_to_equity"] = d["var99"] / d["common_equity_total"]
    d["leverage"] = d["total_assets"] / d["common_equity_total"]
    d["equity"] = d["common_equity_total"]

    # Lagged assets for value weights
    d["assets_weight"] = d.groupby("bank_id")["total_assets"].shift(1)
    if d["bank_id"].nunique() == 1:
        d["assets_weight"] = d["total_assets"]

    # --- Optional first differences ---
    if USE_CHANGE_VAR_TO_EQUITY:
        d["var_to_equity"] = d.groupby("bank_id")["var_to_equity"].diff()

    if USE_CHANGE_LEVERAGE:
        d["leverage"] = d.groupby("bank_id")["leverage"].diff()

    # --- Standardize within each bank relative to pre-period ---
    cols_to_std = ["unit_var", "leverage", "var_to_equity", "equity"]

    def zscore_with_preperiod(grp: pd.DataFrame) -> pd.DataFrame:
        out = grp.copy()
        pre = out[(out["period_end_date"] >= PRE_START) & (out["period_end_date"] <= PRE_END)]

        for c in cols_to_std:
            mu = pre[c].mean()
            sd = pre[c].std()
            out[c] = (out[c] - mu) / sd if (pd.notna(mu) and pd.notna(sd) and sd != 0) else np.nan

        return out

    d = d.groupby("bank_id", group_keys=False).apply(zscore_with_preperiod, include_groups=False)

    # --- Aggregate across banks each quarter (value-weighted by lagged assets) ---
    g = (
        d.groupby("period_end_date", as_index=False)
        .apply(
            lambda grp: pd.Series({
                "unit_var": weighted_mean(grp["unit_var"], grp["assets_weight"]),
                "leverage": weighted_mean(grp["leverage"], grp["assets_weight"]),
                "var_to_equity": weighted_mean(grp["var_to_equity"], grp["assets_weight"]),
                "equity": weighted_mean(grp["equity"], grp["assets_weight"]),
            }),
            include_groups=False
        )
        .sort_values("period_end_date")
        .dropna(subset=["unit_var", "leverage", "var_to_equity"])
        .reset_index(drop=True)
    )

    if g.empty:
        raise ValueError("No valid observations after aggregation (check pre-period coverage and weights).")

    # --- Labels ---
    lev_label = "ΔLeverage" if USE_CHANGE_LEVERAGE else "Leverage"
    vare_label = "ΔVaR/E (99%)" if USE_CHANGE_VAR_TO_EQUITY else "VaR/E (99%)"





     # --- Plot ---
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.axhspan(-2, 2, color="0.94", zorder=0)
    ax.axhline(0, color="black", linewidth=1.0, zorder=1)

    ax.plot(g["period_end_date"], g["unit_var"],
        linestyle="--", linewidth=2.2, label="Unit VaR (99%)", c="#050505"
    )
    ax.plot(
        g["period_end_date"], g["leverage"],
        linestyle="--", linewidth=2.2, label=lev_label, c="#8a8a8a"
    )
    ax.plot(
        g["period_end_date"], g["var_to_equity"],
        linestyle="-", linewidth=3.2, label=vare_label, c="red"
    )

    # Clean main title
    ax.set_title("Risk and balance sheet adjustment")
    ax.set_ylabel("Pre-Period Standard Deviations")
    ax.legend(loc="upper right", frameon=False)

    # Small info box in top-left
    bank_text = "All banks" if not BANK_IDS else ", ".join(BANK_IDS)

    notes = []
    if USE_CHANGE_VAR_TO_EQUITY:
        notes.append("VaR/E in first differences")
    if USE_CHANGE_LEVERAGE:
        notes.append("Leverage in first differences")

    info_text = f"Bank: {bank_text}"
    if notes:
        info_text += "\n" + "; ".join(notes)

    ax.text(
        0.02, 0.98, info_text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9)
    )

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    ax.set_ylim(-3, 10)
    ax.set_xlim(g["period_end_date"].min(), g["period_end_date"].max())
    fig.tight_layout()

    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outfile}")

if __name__ == "__main__":
    main()