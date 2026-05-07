"""
Plot a Figure 5-style comparison of risk and balance sheet adjustment.

Base series (in levels before optional differencing):
- Unit VaR  = VaR(99%) / Assets
- VaR/E     = VaR(99%) / Book common equity
- Leverage  = Assets / Book common equity

Key choices:
1) Uses ratios in levels as baseline, in the spirit of Adrian & Shin Figure 5.
2) Standardizes relative to PRE-PERIOD (2014Q1 - 2019Q4).
3) Value-weighted average across banks using lagged assets as weights.
4) Filters sample to 2014Q1 - 2025Q3.
5) Uses 99% VaR from var_99_gaussian.
6) Standardizes within each bank first, then aggregates across banks.
7) Can optionally use first differences for VaR/E and/or Leverage.

Inputs:
- data/processed/panel.csv

Output:
- output/figures/var_to_equity/figure5_var99_market.png
- output/figures/var_to_equity/figure5_var99_commercial.png
- output/figures/var_to_equity/figure5_var99_custody.png
"""

from pathlib import Path
import os
import tempfile
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "fontconfig"))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Project paths.
ROOT = Path(__file__).resolve().parents[2]
PANEL_FILE = ROOT / "data" / "processed" / "panel.csv"


# Sample settings. The pre-period is the reference window used to standardize
# each bank's series before aggregating across banks.
SAMPLE_START = pd.Timestamp("2014-03-31")  # 2014Q1
SAMPLE_END = pd.Timestamp("2025-09-30")    # 2025Q3

PRE_START = pd.Timestamp("2014-03-31")
PRE_END = pd.Timestamp("2019-12-31")       # 2019Q4


# Bank groups used in the descriptive figures.
Custody_banks = ["bny", "statestreet"]
Commercial_banks = ["bankofamerica", "citigroup", "jpmorgan", "wellsfargo"]
Market_banks = ["goldmansachs", "morganstanley"]



# Plot options. These are left as switches so the same script can reproduce the
# level-based figure or alternative first-difference versions if needed.
USE_CHANGE_VAR_TO_EQUITY = False
USE_CHANGE_LEVERAGE = False


def out_file(group_name: str) -> Path:
    return ROOT / "output" / "figures" / "var_to_equity" / f"figure5_var99_{group_name}"


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Compute a value-weighted mean after filtering invalid observations."""
    ok = values.notna() & weights.notna() & np.isfinite(weights) & (weights > 0)
    if ok.sum() == 0:
        return np.nan
    return float(np.average(values[ok].to_numpy(), weights=weights[ok].to_numpy()))


def zscore_with_reference(grp: pd.DataFrame, cols_to_std: list[str]) -> pd.DataFrame:
    """Standardize each series relative to a bank-specific reference period."""
    out = grp.copy()
    ref = out[(out["period_end_date"] >= PRE_START) & (out["period_end_date"] <= PRE_END)]

    for c in cols_to_std:
        mu = ref[c].mean()
        sd = ref[c].std()
        out[c + "_z"] = (out[c] - mu) / sd if pd.notna(sd) and sd != 0 else np.nan

    return out


def plot_group(panel: pd.DataFrame, bank_ids: list[str], group_name: str) -> None:
    outfile = out_file(group_name)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    d = panel[panel["bank_id"].isin(bank_ids)].copy()

    print("Banks in panel file:", panel["bank_id"].nunique(), sorted(panel["bank_id"].unique()))
    print("Banks after merge:", d["bank_id"].nunique(), sorted(d["bank_id"].unique()))
    print("Obs in group:", len(d))
    print("Date range in group:", d["period_end_date"].min(), "to", d["period_end_date"].max())

    # Restrict the sample to the analysis window.
    d = d[(d["period_end_date"] >= SAMPLE_START) & (d["period_end_date"] <= SAMPLE_END)].copy()

    # Remove observations with missing or non-positive values.
    d = d.dropna(subset=["total_assets", "total_equity", "var99"])
    d = d[(d["total_assets"] > 0) & (d["total_equity"] > 0) & (d["var99"] > 0)].copy()

    # Sort the panel before computing lags and first differences.
    d = d.sort_values(["bank_id", "period_end_date"]).reset_index(drop=True)

    # Construct the baseline ratios in levels.
    d["unit_var"] = d["var99"] / d["total_assets"]
    d["var_to_equity"] = d["var99"] / d["total_equity"]
    d["leverage"] = d["total_assets"] / d["total_equity"]

    # Use lagged assets as aggregation weights.
    d["assets_weight"] = d.groupby("bank_id")["total_assets"].shift(1)
    if d["bank_id"].nunique() == 1:
        d["assets_weight"] = d["total_assets"]

    # Optionally convert the selected ratios to first differences.
    if USE_CHANGE_VAR_TO_EQUITY:
        d["var_to_equity"] = d.groupby("bank_id")["var_to_equity"].diff()

    if USE_CHANGE_LEVERAGE:
        d["leverage"] = d.groupby("bank_id")["leverage"].diff()

    # Standardize within each bank relative to the pre-period. This means each
    # plotted value is measured in standard deviations from that bank's own
    # 2014-2019 average.
    cols_to_std = ["unit_var", "var_to_equity", "leverage"]

    d = (
        d.groupby("bank_id", group_keys=False)
        .apply(lambda grp: zscore_with_reference(grp, cols_to_std), include_groups=False)
        .reset_index(drop=True)
    )

    # Aggregate bank-level standardized series to quarter-level averages. Lagged
    # assets are used as weights so larger banks receive more weight.
    g = (
        d.groupby("period_end_date")
        .apply(
            lambda grp: pd.Series(
                {
                    "unit_var_z": weighted_mean(grp["unit_var_z"], grp["assets_weight"]),
                    "leverage_z": weighted_mean(grp["leverage_z"], grp["assets_weight"]),
                    "var_to_equity_z": weighted_mean(grp["var_to_equity_z"], grp["assets_weight"]),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .sort_values("period_end_date")
        .dropna(subset=["unit_var_z", "leverage_z", "var_to_equity_z"])
        .reset_index(drop=True)
    )

    if g.empty:
        raise ValueError("No valid observations after aggregation (check pre-period coverage and weights).")

    # Build legend labels that reflect the current plotting options.
    lev_label = "ΔLeverage" if USE_CHANGE_LEVERAGE else "Leverage"
    vare_label = "ΔVaR/E (99%)" if USE_CHANGE_VAR_TO_EQUITY else "VaR/E (99%)"

    # Create the final figure.
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.axhspan(-2, 2, color="0.94", zorder=0)
    ax.axhline(0, color="black", linewidth=1.0, zorder=1)

    ax.plot(
        g["period_end_date"],
        g["unit_var_z"],
        linestyle="--",
        linewidth=2.2,
        label="Unit VaR (99%)",
        c="#050505",
    )
    ax.plot(
        g["period_end_date"],
        g["leverage_z"],
        linestyle="--",
        linewidth=2.2,
        label=lev_label,
        c="#8a8a8a",
    )
    ax.plot(
        g["period_end_date"],
        g["var_to_equity_z"],
        linestyle="-",
        linewidth=3.2,
        label=vare_label,
        c="red",
    )

    ax.set_title("Unit VaR, Leverage, and VaR-to-Equity")
    ax.set_ylabel("Pre-Period Standard Deviations")
    ax.legend(loc="upper right", frameon=False)

    # Add a small annotation box with the bank selection and plot settings.
    bank_text = ", ".join(bank_ids)

    notes = []
    if USE_CHANGE_VAR_TO_EQUITY:
        notes.append("VaR/E in first differences")
    if USE_CHANGE_LEVERAGE:
        notes.append("Leverage in first differences")

    info_text = f"Bank: {bank_text}"
    if notes:
        info_text += "\n" + "; ".join(notes)

    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.9),
    )

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    y = g[["unit_var_z", "leverage_z", "var_to_equity_z"]].to_numpy(dtype=float)
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    y_pad = max(0.5, 0.08 * (y_max - y_min))
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlim(g["period_end_date"].min(), g["period_end_date"].max())
    fig.tight_layout()

    png_outfile = outfile.with_suffix(".png")
    pdf_outfile = outfile.with_suffix(".pdf")
    fig.savefig(png_outfile, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_outfile}")
    print(f"Saved: {pdf_outfile}")


def main() -> None:
    # Load and rename only the columns needed for the figure.
    panel = pd.read_csv(PANEL_FILE)
    panel = panel[["bank", "date", "total_assets", "total_equity", "total_var"]].copy()
    panel = panel.rename(
        columns={
            "bank": "bank_id",
            "date": "period_end_date",
            "total_var": "var99",
        }
    )
    panel["bank_id"] = panel["bank_id"].astype(str).str.lower().str.strip()
    panel["period_end_date"] = pd.to_datetime(panel["period_end_date"])
    panel["total_assets"] = pd.to_numeric(panel["total_assets"], errors="coerce")
    panel["total_equity"] = pd.to_numeric(panel["total_equity"], errors="coerce")
    panel["var99"] = pd.to_numeric(panel["var99"], errors="coerce")

    # Create one figure for each bank group.
    plot_group(panel, Market_banks, "market")
    plot_group(panel, Commercial_banks, "commercial")
    plot_group(panel, Custody_banks, "custody")


if __name__ == "__main__":
    main()
