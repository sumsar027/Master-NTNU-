"""
Adrian–Shin Figure 4-style plot: Implied volatility (VIX) vs Unit VaR (99%)

- Unit VaR = VaR(99%) / Total Assets
- VIX is converted from daily to quarterly mean
- Both series are standardized to PRE period (2014Q1–2019Q4):
    z_t = (x_t - mean_pre) / std_pre
- Unit VaR is standardized within bank first, then aggregated (value-weighted by lagged assets)

Inputs:
- data/raw/VIXCLS.csv  (or similar FRED download)
- data/processed/balance_sheet_panel_balanced.csv
- output/data/var_99.csv

Output:
- output/figures/implied_vol_vs_unit_var.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# -----------------------------
# Sample windows
# -----------------------------
SAMPLE_START = pd.Timestamp("2014-03-31")  # 2014Q1
SAMPLE_END   = pd.Timestamp("2025-09-30")  # 2025Q3
PRE_START    = pd.Timestamp("2014-03-31")  # 2014Q1
PRE_END      = pd.Timestamp("2019-12-31")  # 2019Q4


# -----------------------------
# Helpers
# -----------------------------
def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    ok = values.notna() & weights.notna() & np.isfinite(weights) & (weights > 0)
    if ok.sum() == 0:
        return np.nan
    return float(np.average(values[ok].to_numpy(), weights=weights[ok].to_numpy()))


def standardize_to_pre(df: pd.DataFrame, col: str) -> pd.DataFrame:
    pre = df[(df["quarter"] >= PRE_START) & (df["quarter"] <= PRE_END)][col]
    mu, sd = pre.mean(), pre.std()
    if pd.isna(sd) or sd == 0:
        df[col] = np.nan
    else:
        df[col] = (df[col] - mu) / sd
    return df


# -----------------------------
# Load VIX (daily -> quarterly mean)
# -----------------------------
def load_vix_quarterly(vix_file: Path) -> pd.DataFrame:
    raw = pd.read_csv(vix_file)

    # Find date column (fallback: first col)
    date_col = next((c for c in raw.columns if "date" in c.lower()), raw.columns[0])

    # Find value column (common FRED name: VIXCLS)
    candidates = ["VIXCLS", "vixcls", "VIX", "vix"]
    value_col = next((c for c in candidates if c in raw.columns),
                     [c for c in raw.columns if c != date_col][0])

    vix = raw[[date_col, value_col]].copy()
    vix.columns = ["date", "vix"]
    vix["date"] = pd.to_datetime(vix["date"], errors="coerce")
    vix["vix"] = pd.to_numeric(vix["vix"], errors="coerce")
    vix = vix.dropna().sort_values("date").reset_index(drop=True)

    # Map each day to quarter-end and take within-quarter mean
    vix["quarter"] = vix["date"].dt.to_period("Q").dt.to_timestamp(how="end").dt.normalize()
    vix_q = vix.groupby("quarter", as_index=False)["vix"].mean().sort_values("quarter").reset_index(drop=True)

    # Filter + standardize
    vix_q = vix_q[(vix_q["quarter"] >= SAMPLE_START) & (vix_q["quarter"] <= SAMPLE_END)].copy()
    vix_q = standardize_to_pre(vix_q, "vix").rename(columns={"vix": "implied_vol"})
    return vix_q


# -----------------------------
# Load bank panel (balance + VaR) and build Unit VaR aggregate
# -----------------------------
def load_bank_panel(root: Path) -> pd.DataFrame:
    base_file = root / "data" / "processed" / "balance_sheet_panel_balanced.csv"
    var_file  = root / "output" / "data" / "var_99.csv"

    base = pd.read_csv(base_file)
    base = base[["bank_id", "period_end_date", "total_assets", "common_equity_total"]].copy()
    base["bank_id"] = base["bank_id"].astype(str).str.lower().str.strip()
    base["period_end_date"] = pd.to_datetime(base["period_end_date"])
    base["total_assets"] = pd.to_numeric(base["total_assets"], errors="coerce")
    base["common_equity_total"] = pd.to_numeric(base["common_equity_total"], errors="coerce")

    var = pd.read_csv(var_file)
    var = var[["bank", "date", "var_99_boa_factor"]].copy()
    var = var.rename(columns={"bank": "bank_id", "date": "period_end_date", "var_99_boa_factor": "var99"})
    var["bank_id"] = var["bank_id"].astype(str).str.lower().str.strip()
    var["period_end_date"] = pd.to_datetime(var["period_end_date"])
    var["var99"] = pd.to_numeric(var["var99"], errors="coerce")

    d = base.merge(var, on=["bank_id", "period_end_date"], how="inner")
    d = d.dropna(subset=["total_assets", "common_equity_total", "var99"])
    d = d[(d["total_assets"] > 0) & (d["common_equity_total"] > 0) & (d["var99"] > 0)].copy()

    d = d.rename(columns={
        "period_end_date": "quarter",
        "total_assets": "assets",
        "common_equity_total": "equity",
    })

    d = d[(d["quarter"] >= SAMPLE_START) & (d["quarter"] <= SAMPLE_END)].copy()
    return d[["bank_id", "quarter", "assets", "equity", "var99"]].sort_values(["bank_id", "quarter"]).reset_index(drop=True)


def compute_unit_var_aggregate(panel: pd.DataFrame) -> pd.DataFrame:
    d = panel.copy()
    d["quarter"] = pd.to_datetime(d["quarter"]).dt.normalize()

    # Unit VaR (bank-quarter)
    d["unit_var"] = d["var99"] / d["assets"]

    # Lagged assets for value weights
    d = d.sort_values(["bank_id", "quarter"])
    d["assets_lag"] = d.groupby("bank_id")["assets"].shift(1)

    # Standardize within bank to PRE period
    def std_bank(g: pd.DataFrame) -> pd.DataFrame:
        pre = g[(g["quarter"] >= PRE_START) & (g["quarter"] <= PRE_END)]["unit_var"]
        mu, sd = pre.mean(), pre.std()
        if pd.isna(sd) or sd == 0:
            g["unit_var"] = np.nan
        else:
            g["unit_var"] = (g["unit_var"] - mu) / sd
        return g

    d = d.groupby("bank_id", group_keys=False).apply(std_bank, include_groups=False)

    # Value-weighted aggregate each quarter
    agg = (
        d.groupby("quarter", as_index=False)
        .apply(lambda grp: pd.Series({
            "unit_var": weighted_mean(grp["unit_var"], grp["assets_lag"])
        }), include_groups=False)
        .dropna()
        .sort_values("quarter")
        .reset_index(drop=True)
    )
    return agg


# -----------------------------
# Plot
# -----------------------------
def make_plot(vix_q: pd.DataFrame, unit_var_q: pd.DataFrame, out_file: Path) -> None:
    merged = unit_var_q.merge(vix_q, on="quarter", how="inner").sort_values("quarter").reset_index(drop=True)
    if merged.empty:
        raise ValueError("Merge produced 0 rows. Check quarter-end alignment in VIX vs bank panel.")

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.axhspan(-2, 2, color="0.85", zorder=0)
    ax.axhline(0, color="black", linewidth=1.0, zorder=1)

    ax.plot(merged["quarter"], merged["implied_vol"], linewidth=2.5, label="Implied Vol (VIX)", zorder=3)
    ax.plot(merged["quarter"], merged["unit_var"], linestyle="--", linewidth=2.5, label="Unit VaR (99%)", zorder=2)

    ax.set_title("Risk measures: implied volatility vs unit VaR")
    ax.set_ylabel("Pre-period standard deviations")
    ax.legend(loc="upper left", frameon=False)

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[6, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def main() -> None:
    ROOT = Path(__file__).resolve().parents[2]

    # Update this if your filename is different:
    VIX_FILE = ROOT / "data" / "raw" / "VIXCLS (1).csv"
    OUT_FILE = ROOT / "output" / "figures" / "implied_vol_vs_unit_var.png"

    vix_q = load_vix_quarterly(VIX_FILE)
    panel = load_bank_panel(ROOT)
    unit_var_q = compute_unit_var_aggregate(panel)

    make_plot(vix_q, unit_var_q, OUT_FILE)


if __name__ == "__main__":
    main()
