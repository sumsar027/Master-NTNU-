"""
Figure: VaR and balance sheet levels (NO standardization)

Creates a Figure 5-style diagnostic plot for 2014Q1–2025Q3 using LEVELS:
- VaR (99%) level (harmonized via BoA factor inside var_99_boa_factor)
- Total Assets (level)
- Book common equity (level)
- Liabilities (level) = Assets - Equity

Aggregation:
- Value-weighted average across banks each quarter
- Weights: lagged assets (previous quarter)

Inputs:
- data/processed/merged_quarterly_balanced.csv
- output/data/merged_with_var_99_dual_methods.csv

Output:
- output/figures/figure_levels_var_and_liabilities.png
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


def _require_columns(df: pd.DataFrame, cols: list[str], *, where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing column(s) in {where}: " + ", ".join(missing))


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
    """Load and merge balance sheet data with 99% VaR data (fixed definitions)."""

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

    # Require fixed balance sheet columns
    if ASSETS_COL not in merged.columns:
        raise KeyError(f"Required assets column not found: {ASSETS_COL}")
    if EQUITY_COL not in merged.columns:
        raise KeyError(f"Required equity column not found: {EQUITY_COL}")

    print("\n=== COLUMN SELECTION (FIXED) ===")
    print("Assets column used:", ASSETS_COL)
    print("Equity column used:", EQUITY_COL)
    print("===============================\n")

    assets = _to_numeric(merged[ASSETS_COL])
    equity = _to_numeric(merged[EQUITY_COL])
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
    df = df.dropna(subset=["bank_id", "quarter", "assets", "equity", "var99"])
    df = df[(df["assets"] > 0) & (df["equity"] > 0) & (df["var99"] > 0)].copy()
    df = df.sort_values(["bank_id", "quarter"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows after cleaning.")

    print(f"Loaded {len(df)} observations for {df['bank_id'].nunique()} banks")
    return df


def plot_levels_var_and_liabilities(df: pd.DataFrame):
    """Plot value-weighted LEVELS: Assets, Liabilities, Equity (left axis) and VaR (right axis)."""

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

    # Construct liabilities as A - E (book identity)
    d["liabilities"] = d["assets"] - d["equity"]

    # Lagged assets for value-weighting
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
                "assets": _weighted_mean(grp["assets"], w),
                "liabilities": _weighted_mean(grp["liabilities"], w),
                "equity": _weighted_mean(grp["equity"], w),
                "var99": _weighted_mean(grp["var99"], w),
            }
        )

    g = (
        d.groupby("quarter", as_index=False)
        .apply(_agg, include_groups=False)
        .sort_values("quarter")
        .dropna(subset=["assets", "liabilities", "equity", "var99"])
        .reset_index(drop=True)
    )

    if g.empty:
        raise ValueError("No valid observations after aggregation.")

    fig, ax = plt.subplots(figsize=(10.5, 5.6))

    # Left axis: balance sheet levels
    ax.plot(g["quarter"], g["assets"], linestyle="-", linewidth=2.0, label="Total Assets (VW, level)")
    ax.plot(g["quarter"], g["liabilities"], linestyle="--", linewidth=2.0, label="Liabilities = Assets − Equity (VW, level)")
    ax.plot(g["quarter"], g["equity"], linestyle=":", linewidth=2.5, label="Book Common Equity (VW, level)")

    ax.set_title("VaR and balance sheet levels (value-weighted, no standardization)")
    ax.set_ylabel("Balance sheet levels (left axis)")
    ax.legend(loc="upper left", frameon=False)

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[6, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

    # Right axis: VaR level
    ax2 = ax.twinx()
    ax2.plot(g["quarter"], g["var99"], linestyle="-.", linewidth=2.2, label="VaR (99%, VW, level)")
    ax2.set_ylabel("VaR level (right axis)")

    # Combine legends (left + right)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)

    fig.tight_layout()
    return fig


def main() -> None:
    df = load_df_from_project_outputs()
    fig = plot_levels_var_and_liabilities(df)

    out_dir = Path("output/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "figure_levels_var_and_liabilities.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path.resolve()}")

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
