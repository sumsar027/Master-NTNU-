"""
Replicate Adrian & Shin Figure 5-style plot (risk + balance sheet adjustment),
for 2014Q1–2025Q3, using:
- Unit VaR  = VaR / Assets
- VaR/E     = VaR / Book common equity
- Leverage  = Assets / Book common equity

Uses harmonized 99% VaR with BoA factor (×2) for banks without reported 99% VaR.

Key choices:
1) Uses ratios in LEVELS (not log-ratios).
2) Standardizes relative to PRE-PERIOD (default: 2014Q1–2019Q4).
3) Value-weighted average with LAGGED assets as weights.
4) Filters sample to 2014Q1–2025Q3.
5) Uses 99% VaR.
6) Uses var_99_boa_factor (reported 99% VaR when available; otherwise ×2 conversion).
7) Standardize per bank first, then aggregate across banks.

Inputs:
- data/processed/merged_quarterly_balanced.csv
- output/data/merged_with_var_99_dual_methods.csv

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
BASE_FILE = Path("data/processed/merged_quarterly_balanced.csv")
VAR_FILE = Path("output/data/merged_with_var_99_dual_methods.csv")

# -----------------------------
# Fixed column definitions (thesis version)
# -----------------------------
ASSETS_COL = "financial_summary_Total Assets"
EQUITY_COL = "financial_summary_Common Equity - Total"
VAR99_COL = "var_99_boa_factor"

# -----------------------------
# Sample settings
# -----------------------------
SAMPLE_START = pd.Timestamp("2014-03-31")  # 2014Q1 quarter-end
SAMPLE_END = pd.Timestamp("2025-09-30")    # 2025Q3 quarter-end

PRE_START = pd.Timestamp("2014-03-31")
PRE_END = pd.Timestamp("2019-12-31")


def normalize_bank_id(bank_id: str) -> str:
    """Normalize bank identifiers to a consistent canonical format."""
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

# make sure required columns exist
def _require_columns(df: pd.DataFrame, cols: list[str], *, where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s) in {where}: " + ", ".join(missing))

# find first matching column from candidates
def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_datetime(series: pd.Series) -> pd.Series:
    """
    Convert date/quarter formats into datetime. If values look like "YYYYQn",
    convert explicitly to quarter-end (e.g. 2014Q1 -> 2014-03-31).
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

# make numeric, handling common formats (commas, parentheses for negatives, NAs)
def _to_numeric(series: pd.Series) -> pd.Series:
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
    """Load and merge balance sheet data with 99% VaR data (fixed column definitions)."""

    # Load files and check existence
    if not base_file.exists():
        raise FileNotFoundError(f"Missing file: {base_file}")
    if not var_file.exists():
        raise FileNotFoundError(f"Missing file: {var_file}")

    base = pd.read_csv(base_file)
    var = pd.read_csv(var_file)

    # Identify entity + time columns and rename to common names
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

    # Normalize IDs + dates
    base["bank_id"] = base["bank_id"].apply(normalize_bank_id)
    var["bank_id"] = var["bank_id"].apply(normalize_bank_id)

    base["period_end_date"] = _to_datetime(base["period_end_date"])
    var["period_end_date"] = _to_datetime(var["period_end_date"])

    base = base.dropna(subset=["bank_id", "period_end_date"])
    var = var.dropna(subset=["bank_id", "period_end_date"])

    # Require VaR column
    if VAR99_COL not in var.columns:
        raise KeyError(f"Required VaR column not found: {VAR99_COL}")

    print(f"Using VaR column: {VAR99_COL}")
    var[VAR99_COL] = _to_numeric(var[VAR99_COL])

    # Merge on (bank_id, period_end_date)
    merged = base.merge(
        var[["bank_id", "period_end_date", VAR99_COL]].rename(columns={VAR99_COL: "var_99_level"}),
        on=["bank_id", "period_end_date"],
        how="inner",
    )

    # Filter sample
    merged = merged[(merged["period_end_date"] >= SAMPLE_START) & (merged["period_end_date"] <= SAMPLE_END)].copy()
    if merged.empty:
        raise ValueError("No observations after merging + filtering to 2014Q1–2025Q3.")

    # Select balance sheet columns (prefer thesis names, fall back to snake_case outputs)
    assets_col = _find_column(merged, [ASSETS_COL, "total_assets_2", "total_assets", "assets"])
    if assets_col is None:
        raise KeyError(f"Required assets column not found: {ASSETS_COL}")

    equity_col = _find_column(
        merged,
        [
            EQUITY_COL,
            "common_equity_total",
            "common_equity_attributable_to_parent_shareholders",
            "total_shareholders_equity",
            "total_equity",
        ],
    )
    if equity_col is None:
        raise KeyError(f"Required equity column not found: {EQUITY_COL}")

    print("\n=== COLUMN SELECTION ===")
    print("Assets column used:", assets_col)
    print("Equity column used:", equity_col)
    print("=======================\n")

    # Small robustness: if both total_assets and total_assets_2 exist, treat total_assets_2 as fallback
    if assets_col == "total_assets" and "total_assets_2" in merged.columns:
        assets = _to_numeric(merged["total_assets"]).fillna(_to_numeric(merged["total_assets_2"]))
    else:
        assets = _to_numeric(merged[assets_col])

    equity = _to_numeric(merged[equity_col])
    var99 = _to_numeric(merged["var_99_level"])

    df = pd.DataFrame(
        {
            "bank_id": merged["bank_id"].astype(str),
            "quarter": merged["period_end_date"],
            "assets": assets,
            "equity": equity,
            "var99": var99,
        }
    )

    # Clean
    n0 = len(df)
    df = df.dropna(subset=["bank_id", "quarter", "assets", "equity", "var99"])
    n_dropna = len(df)
    df = df[(df["assets"] > 0) & (df["equity"] > 0) & (df["var99"] > 0)].copy()
    n_pos = len(df)

    df = df.sort_values(["bank_id", "quarter"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"No valid rows after cleaning. Rows: start={n0}, after_dropna={n_dropna}, after_positive_filter={n_pos}."
        )

    print(f"Loaded {len(df)} observations for {df['bank_id'].nunique()} banks")
    return df


def plot_adrian_shin_figure_5(df: pd.DataFrame):
    """Create Figure 5-style plot using fixed 99% VaR (BoA factor) and fixed balance sheet definitions."""

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

    # Ratios per bank-quarter
    d["unit_var"] = d["var99"] / d["assets"]
    d["var_to_equity"] = d["var99"] / d["equity"]
    d["leverage"] = d["assets"] / d["equity"]

    # Lagged assets for value-weighting
    d["assets_lag"] = d.groupby("bank_id", sort=False)["assets"].shift(1)

    # --- Standardize per bank relative to PRE-period ---
    pre_mask_bank = (d["quarter"] >= PRE_START) & (d["quarter"] <= PRE_END)
    if int(pre_mask_bank.sum()) == 0:
        raise ValueError("Pre-period is empty on bank-level data (before aggregation).")

    cols_to_std = ["unit_var", "leverage", "var_to_equity", "equity"]

    def _zscore_within_bank(grp: pd.DataFrame) -> pd.DataFrame:
        out = grp.copy()
        pre = out.loc[pre_mask_bank.loc[out.index], cols_to_std]

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

    d = d.groupby("bank_id", group_keys=False).apply(_zscore_within_bank, include_groups=False)

    # --- Aggregate across banks (value-weighted by lagged assets) ---
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
                "equity": _weighted_mean(grp["equity"], w),
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

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.axhspan(-2, 2, color="0.85", zorder=0)
    ax.axhline(0, color="black", linewidth=1.0, zorder=1)

    ax.plot(g["quarter"], g["unit_var"], color="tab:blue", linestyle="--", linewidth=2.5, label="Unit VaR (99%)", zorder=2)
    ax.plot(g["quarter"], g["leverage"], color="tab:green", linestyle="--", linewidth=2.5, label="Leverage", zorder=2)
    ax.plot(g["quarter"], g["var_to_equity"], color="tab:red", linestyle="-", linewidth=2.5, label="VaR/E (99%)", zorder=3)
    ax.plot(g["quarter"], g["equity"], color="tab:purple", linestyle=":", linewidth=2.5, label="Equity (standardized)", zorder=2)

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
