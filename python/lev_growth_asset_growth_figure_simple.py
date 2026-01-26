"""
Simple Adrian–Shin (2010) Figure 2.4–style scatter plot:

- x-axis: Total Asset Growth (quarterly, percent)
- y-axis: Leverage Growth (quarterly, percent)

Default input (this repo):
  output/data/merged_quarterly_balanced.csv

Assumed columns (override via flags if needed):
  bank = bank
  date = period_end_date
  assets = total_assets_2 (fallback: total_assets, assets)
  equity = common_equity_total

Run:
  /opt/anaconda3/bin/python src/pipeline/lev_growth_asset_growth_figure_simple.py
"""

from __future__ import annotations

from pathlib import Path
import argparse
import os
import re
import sys
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True

DEFAULT_INPUT = ROOT / "output" / "data" / "merged_quarterly_balanced.csv"
DEFAULT_OUTPUT = ROOT / "output" / "figures" / "figure_2_4_replica.png"


def _prep_matplotlib() -> None:
    # Robust defaults: no GUI needed + try to keep cache/temp inside repo.
    os.environ.setdefault("MPLBACKEND", "Agg")
    if not os.environ.get("MPLCONFIGDIR"):
        for candidate in [ROOT / "output" / ".mplconfig", ROOT / ".mplconfig"]:
            try:
                candidate.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
            if os.access(candidate, os.W_OK):
                os.environ["MPLCONFIGDIR"] = str(candidate)
                break
        else:
            try:
                os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="mplconfig-")
            except OSError:
                pass

    if not os.environ.get("TMPDIR"):
        for candidate in [ROOT / "output" / ".tmp", ROOT / ".tmp"]:
            try:
                candidate.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
            if os.access(candidate, os.W_OK):
                os.environ["TMPDIR"] = str(candidate)
                break


_prep_matplotlib()
import matplotlib.pyplot as plt


def _read_panel(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _to_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s, errors="coerce")


def _to_quarter(s: pd.Series) -> pd.Series:
    if isinstance(s.dtype, pd.PeriodDtype):
        return s.asfreq("Q")
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.to_period("Q")
    return pd.to_datetime(s, errors="coerce").dt.to_period("Q")


def _norm(x: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())


def main() -> int:
    p = argparse.ArgumentParser(description="Simple Adrian–Shin Figure 2.4-style scatter plot.")
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--banks", default="", help="Comma-separated bank IDs to keep (default: all).")
    p.add_argument("--bank-col", default="bank")
    p.add_argument("--date-col", default="period_end_date")
    p.add_argument("--assets-col", default="", help="Default: total_assets_2 if present else total_assets else assets.")
    p.add_argument("--equity-col", default="common_equity_total")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = _read_panel(in_path)

    bank_col = args.bank_col
    date_col = args.date_col
    equity_col = args.equity_col
    if args.assets_col:
        assets_col = args.assets_col
    elif "total_assets_2" in df.columns:
        assets_col = "total_assets_2"
    elif "total_assets" in df.columns:
        assets_col = "total_assets"
    else:
        assets_col = "assets"

    missing = [c for c in [bank_col, date_col, assets_col, equity_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {in_path}: {', '.join(missing)}")

    df = df[[bank_col, date_col, assets_col, equity_col]].copy()
    df[assets_col] = _to_numeric(df[assets_col])
    df[equity_col] = _to_numeric(df[equity_col])
    df = df.dropna(subset=[bank_col, date_col, assets_col, equity_col])
    df = df[(df[assets_col] > 0) & (df[equity_col] > 0)]

    if args.banks.strip():
        keep = {_norm(x) for x in args.banks.split(",") if x.strip()}
        df = df[df[bank_col].map(_norm).isin(keep)].copy()

    df["_quarter"] = _to_quarter(df[date_col])
    df = df.dropna(subset=["_quarter"])
    df = df.sort_values([bank_col, "_quarter"], kind="mergesort")

    df["_lev"] = df[assets_col] / df[equity_col]
    df["asset_growth"] = 100.0 * np.log(df[assets_col]).groupby(df[bank_col]).diff()
    df["equity_growth"] = 100.0 * np.log(df[equity_col]).groupby(df[bank_col]).diff()
    df["leverage_growth"] = 100.0 * np.log(df["_lev"]).groupby(df[bank_col]).diff()
    df = df.dropna(subset=["asset_growth", "equity_growth", "leverage_growth"])

    n_banks = int(df[bank_col].nunique())
    n_quarters = int(df["_quarter"].nunique())
    n_obs = int(len(df))
    banks_used = sorted(df[bank_col].astype(str).unique().tolist())

    identity_err = df["leverage_growth"] - (df["asset_growth"] - df["equity_growth"])
    print(f"Using {n_banks} banks, {n_quarters} quarters, {n_obs} observations.")
    print("Banks used:", ", ".join(banks_used))
    print(f"Columns: bank={bank_col!r}, date={date_col!r}, assets={assets_col!r}, equity={equity_col!r}.")
    print(f"Identity check max |error|: {float(identity_err.abs().max()):.6g}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(
        df["leverage_growth"].to_numpy(),
        df["asset_growth"].to_numpy(),
        s=28,
        c="#f3e21b",
        edgecolors="black",
        linewidths=0.5,
        alpha=0.9,
    )
    ax.set_xlabel("Leverage Growth (percent quarterly)")
    ax.set_ylabel("Total Asset Growth (percent quarterly)")
    ax.set_title("Total Asset Growth vs Leverage Growth (Balanced Panel)")
    ax.set_xlim(-10, 20)
    ax.set_ylim(-10, 20)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    if args.show:
        plt.show()
    plt.close(fig)
    print(f"Saved figure to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
