"""
Replicate Adrian & Shin Figure 4-style plot: Implied volatility vs Unit VaR.

Inputs:
- VIX CSV from FRED (e.g. data/raw/VIXCLS (1).csv). Can be daily/monthly/quarterly.
- Bank panel + VaR merge already used by figure_var_to_equity_ratio.py:
  - data/processed/merged_quarterly_balanced.csv
  - output/data/merged_with_var_99_dual_methods.csv

Method:
1) Compute Unit VaR per bank = VaR(99%) / Assets
2) Standardize within bank relative to PRE period (default: 2014Q1–2019Q4)
3) Aggregate across banks using value-weighted mean (weights = lagged assets)
4) Convert VIX to quarterly frequency and standardize relative to the same PRE period
5) Plot both standardized series with ±2 band around zero
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import os
import tempfile

import numpy as np
import pandas as pd


# Reuse the same dates + loader as the Unit VaR figure.
#
# When this file is run as a script (e.g. `python src/pipeline/implied_volatility.py`),
# Python puts `src/pipeline` on sys.path, so we can import siblings directly.
# When run as a module (e.g. `PYTHONPATH=src python -m pipeline.implied_volatility`),
# we want the package import.
try:  # pragma: no cover
    from pipeline.figure_var_to_equity_ratio import (
        PRE_END,
        PRE_START,
        SAMPLE_END,
        SAMPLE_START,
        load_df_from_project_outputs,
    )
except ModuleNotFoundError:  # pragma: no cover
    from figure_var_to_equity_ratio import (  # type: ignore
        PRE_END,
        PRE_START,
        SAMPLE_END,
        SAMPLE_START,
        load_df_from_project_outputs,
    )
except Exception as e:  # pragma: no cover
    raise ImportError("Could not import from figure_var_to_equity_ratio.") from e


DEFAULT_VIX_FILE = Path("data/raw/VIXCLS (1).csv")
DEFAULT_OUT_FILE = Path("output/figures/implied_vol_vs_unit_var.png")


def _ensure_matplotlib_cache_dir() -> None:
    # Matplotlib sometimes tries to create config/cache dirs.
    # Prefer a project-local dir if MPLCONFIGDIR isn't set.
    if "MPLCONFIGDIR" in os.environ:
        return

    candidates: list[Path] = [
        Path("output") / ".mplconfig",
    ]
    for p in candidates:
        try:
            p.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(p)
            return
        except OSError:
            continue

    # Fallback to OS temp if available/writable
    try:
        base_tmp = tempfile.gettempdir()
        p = Path(base_tmp) / "mplconfig"
        p.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(p)
    except Exception:
        # If we can't set a writable dir, matplotlib may fail at import time.
        return


def _find_date_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "date" in str(c).lower():
            return c
    return str(df.columns[0])


def _find_value_col(df: pd.DataFrame, date_col: str) -> str:
    # Common FRED names
    for c in ["VIXCLS", "vixcls", "vix", "VIX"]:
        if c in df.columns:
            return c
    # Fallback: first non-date column
    for c in df.columns:
        if c != date_col:
            return str(c)
    raise ValueError("Could not identify value column in VIX CSV.")


def _to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    s = series.astype(str).str.strip()
    s = s.replace({".": np.nan, "": np.nan, "NA": np.nan, "N/A": np.nan, "nan": np.nan, "None": np.nan})
    s = s.str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def load_vix_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing VIX file: {path}")
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"VIX file is empty: {path}")

    date_col = _find_date_col(raw)
    value_col = _find_value_col(raw, date_col=date_col)

    vix = raw[[date_col, value_col]].rename(columns={date_col: "date", value_col: "vix"}).copy()
    vix["date"] = pd.to_datetime(vix["date"], errors="coerce")
    vix["vix"] = _to_numeric(vix["vix"])
    vix = vix.dropna(subset=["date", "vix"]).sort_values("date").reset_index(drop=True)
    if vix.empty:
        raise ValueError(f"No usable rows in VIX file after parsing: {path}")
    return vix


def vix_to_quarterly(vix: pd.DataFrame) -> pd.DataFrame:
    d = vix.copy()
    d["quarter"] = d["date"].dt.to_period("Q").dt.to_timestamp(how="end").dt.normalize()
    q = d.groupby("quarter", as_index=False)["vix"].mean().sort_values("quarter").reset_index(drop=True)
    return q


def standardize_to_pre_period(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    d = df.copy()
    pre = d[(d["quarter"] >= PRE_START) & (d["quarter"] <= PRE_END)][value_col]
    if pre.empty:
        raise ValueError(f"Pre-period is empty for {value_col}. Check dates in your VIX file.")
    mu = float(pre.mean())
    sd = float(pre.std())
    if (not np.isfinite(mu)) or (not np.isfinite(sd)) or sd == 0.0:
        raise ValueError(f"Cannot standardize {value_col}: mean/std not finite (mu={mu}, sd={sd}).")
    d[value_col] = (d[value_col] - mu) / sd
    return d


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    ok = values.notna() & weights.notna() & np.isfinite(weights) & (weights > 0)
    if int(ok.sum()) == 0:
        return np.nan
    return float(np.average(values[ok].to_numpy(), weights=weights[ok].to_numpy()))


def compute_unit_var_agg() -> pd.DataFrame:
    df = load_df_from_project_outputs()
    d = df.copy()
    d["quarter"] = pd.to_datetime(d["quarter"], errors="coerce")
    d = d.dropna(subset=["bank_id", "quarter", "assets", "var99"]).sort_values(["bank_id", "quarter"]).reset_index(drop=True)

    d = d[(d["quarter"] >= SAMPLE_START) & (d["quarter"] <= SAMPLE_END)].copy()
    if d.empty:
        raise ValueError("Unit VaR input is empty after filtering to sample period.")

    d["unit_var"] = d["var99"] / d["assets"]
    d["assets_lag"] = d.groupby("bank_id", sort=False)["assets"].shift(1)

    pre_mask = (d["quarter"] >= PRE_START) & (d["quarter"] <= PRE_END)
    if int(pre_mask.sum()) == 0:
        raise ValueError("Pre-period is empty on bank-level data (before aggregation).")

    def _zscore_bank(grp: pd.DataFrame) -> pd.DataFrame:
        out = grp.copy()
        pre = out.loc[pre_mask.loc[out.index], "unit_var"]
        if pre.empty:
            out["unit_var"] = np.nan
            return out
        mu = float(pre.mean())
        sd = float(pre.std())
        if (not np.isfinite(mu)) or (not np.isfinite(sd)) or sd == 0.0:
            out["unit_var"] = np.nan
        else:
            out["unit_var"] = (out["unit_var"] - mu) / sd
        return out

    d = d.groupby("bank_id", group_keys=False).apply(_zscore_bank)

    g = (
        d.groupby("quarter", as_index=False)
        .apply(lambda grp: pd.Series({"unit_var": _weighted_mean(grp["unit_var"], grp["assets_lag"])}), include_groups=False)
        .sort_values("quarter")
        .dropna(subset=["unit_var"])
        .reset_index(drop=True)
    )
    if g.empty:
        raise ValueError("No valid observations after aggregating standardized Unit VaR.")
    return g


@dataclass(frozen=True)
class PlotData:
    quarter: pd.Series
    implied_vol: pd.Series
    unit_var: pd.Series


def build_plot_data(vix_file: Path) -> PlotData:
    vix = load_vix_csv(vix_file)
    vix_q = vix_to_quarterly(vix)
    vix_q = vix_q[(vix_q["quarter"] >= SAMPLE_START) & (vix_q["quarter"] <= SAMPLE_END)].copy()
    vix_q = standardize_to_pre_period(vix_q, "vix").rename(columns={"vix": "implied_vol"})

    unit = compute_unit_var_agg()

    merged = unit.merge(vix_q[["quarter", "implied_vol"]], on="quarter", how="inner").sort_values("quarter").reset_index(drop=True)
    if merged.empty:
        raise ValueError("No overlapping quarters between Unit VaR and VIX series after filtering.")

    return PlotData(
        quarter=merged["quarter"],
        implied_vol=merged["implied_vol"],
        unit_var=merged["unit_var"],
    )


def plot_implied_vol_vs_unit_var(plot_data: PlotData):
    _ensure_matplotlib_cache_dir()
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.axhspan(-2, 2, color="0.85", zorder=0)
    ax.axhline(0, color="black", linewidth=1.0, zorder=1)

    ax.plot(plot_data.quarter, plot_data.implied_vol, color="tab:green", linewidth=2.5, label="Implied Vol (VIX)", zorder=3)
    ax.plot(plot_data.quarter, plot_data.unit_var, color="tab:blue", linestyle="--", linewidth=2.5, label="Unit VaR (99%)", zorder=2)

    ax.set_title("Risk measures: implied volatility vs unit VaR")
    ax.set_ylabel("Pre-Period Standard Deviations")
    ax.legend(loc="upper left", frameon=False)

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[6, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(ax.get_yticks())
    ax2.set_ylabel("")

    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vix", type=Path, default=DEFAULT_VIX_FILE, help="Path to VIX CSV (FRED download).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE, help="Output PNG path.")
    parser.add_argument("--show", action="store_true", help="Show interactive window.")
    args = parser.parse_args()

    plot_data = build_plot_data(args.vix)
    fig = plot_implied_vol_vs_unit_var(plot_data)

    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=300, bbox_inches="tight")
        print(f"Saved: {args.out.resolve()}")
    finally:
        if args.show:
            import matplotlib.pyplot as plt

            plt.show()


if __name__ == "__main__":
    main()
