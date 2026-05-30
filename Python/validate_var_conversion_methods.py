"""
Validate alternative VaR conversion methods using Bank of America data.

The script compares two ways of converting VaR(95%) to VaR(99%) against the
observed Bank of America VaR(99%) series in the raw Excel workbook.

Purpose in the thesis:
- Bank of America reports both 95% and 99% VaR.
- This makes it possible to check how well different conversion rules reproduce
  an actually observed 99% VaR series.
- The output figure is used as a methodological robustness check.
"""

from pathlib import Path
import os
import tempfile
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "fontconfig"))

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})

# The Gaussian method rescales 95% VaR to 99% VaR using normal-distribution
# quantiles. The BoA factor is a simple empirical benchmark used only here.
GAUSSIAN_RATIO = 2.326348 / 1.644854  # ≈ 1.414319
BOA_FACTOR = 2.0

# Input workbook and output figure.
INPUT_FILE = Path("data/raw/VaR_python.xlsx")
SHEET_NAME = "new"

OUTPUT_DIR = Path("output/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG = OUTPUT_DIR / "boa_validation_comprehensive_excel.png"


def r2_score_simple(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared between actual and predicted VaR."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def rmse_simple(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate root mean squared error between actual and predicted VaR."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def norm_bank(x: str) -> str:
    return "".join(ch for ch in str(x).lower().strip() if ch.isalnum())


def main():
    raw = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME, header=None)

    bank_row = raw.iloc[1].ffill()
    level_row = raw.iloc[2]

    cols = []
    for b, lvl in zip(bank_row, level_row):
        b_txt = norm_bank(b) if pd.notna(b) else ""
        lvl_txt = str(lvl).strip() if pd.notna(lvl) else ""
        cols.append(f"{b_txt}_{lvl_txt}")

    data = raw.iloc[3:].copy()
    data.columns = cols

    if "year_Level" not in data.columns:
        raise KeyError("Missing 'year_Level' column. Excel layout changed.")

    boa_95_col = "bankofamerica_0.95"
    boa_99_col = "bankofamerica_0.99"

    if boa_95_col not in data.columns or boa_99_col not in data.columns:
        raise KeyError(
            "Missing Bank of America 0.95/0.99 columns. "
            "Expected bankofamerica_0.95 and bankofamerica_0.99."
        )

    df = data[["year_Level", boa_95_col, boa_99_col]].copy()
    df.columns = ["period_end_date", "var_95", "var_99"]

    df["period_end_date"] = pd.to_datetime(df["period_end_date"], errors="coerce")
    df["var_95"] = pd.to_numeric(df["var_95"], errors="coerce")
    df["var_99"] = pd.to_numeric(df["var_99"], errors="coerce")

    df = df.dropna(subset=["period_end_date", "var_95", "var_99"])
    df = df[(df["var_95"] > 0) & (df["var_99"] > 0)].copy()
    df = df.sort_values("period_end_date").reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No valid Bank of America observations with both var_95 and var_99.")

    df["predicted_gaussian"] = df["var_95"] * GAUSSIAN_RATIO
    df["predicted_boa"] = df["var_95"] * BOA_FACTOR

    df["error_gaussian_pct"] = 100 * (
        df["predicted_gaussian"] - df["var_99"]
    ) / df["var_99"]

    df["error_boa_pct"] = 100 * (
        df["predicted_boa"] - df["var_99"]
    ) / df["var_99"]

    r2_g = r2_score_simple(df["var_99"].to_numpy(), df["predicted_gaussian"].to_numpy())
    rmse_g = rmse_simple(df["var_99"].to_numpy(), df["predicted_gaussian"].to_numpy())
    r2_b = r2_score_simple(df["var_99"].to_numpy(), df["predicted_boa"].to_numpy())
    rmse_b = rmse_simple(df["var_99"].to_numpy(), df["predicted_boa"].to_numpy())

    fig = plt.figure(figsize=(11, 8.2), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.42, wspace=0.30)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.10)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(
        df["period_end_date"],
        df["var_99"],
        marker="o",
        linewidth=2.8,
        label="Actual VaR 99%",
        c="#000000",
    )
    ax1.plot(
        df["period_end_date"],
        df["predicted_gaussian"],
        marker="s",
        linestyle="--",
        linewidth=2.4,
        label=f"Gaussian (* {GAUSSIAN_RATIO:.4f})",
        c="#D62728",
    )
    ax1.plot(
        df["period_end_date"],
        df["predicted_boa"],
        marker="^",
        linestyle="--",
        linewidth=2.4,
        label=f"BoA factor (* {BOA_FACTOR:.1f})",
        c="#888888",
    )

    ax1.set_title("Bank of America: VaR 99% — Actual vs Predicted", fontsize=17)
    ax1.set_ylabel("VaR level", fontsize=16)
    ax1.legend(frameon=False, fontsize=13)
    ax1.grid(True, linestyle="--", alpha=0.4)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(df["var_99"], df["predicted_gaussian"], s=80, alpha=0.75, c="black")

    min_v = float(np.min([df["var_99"].min(), df["predicted_gaussian"].min()]))
    max_v = float(np.max([df["var_99"].max(), df["predicted_gaussian"].max()]))

    ax2.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1.2)
    ax2.set_title("Gaussian method", fontsize=16)
    ax2.set_xlabel("Actual VaR 99%", fontsize=16)
    ax2.set_ylabel("Predicted VaR 99%", fontsize=16)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.text(
        0.05,
        0.95,
        f"R² = {r2_g:.3f}\nRMSE = {rmse_g:.2f}",
        transform=ax2.transAxes,
        va="top",
        fontsize=13,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(df["var_99"], df["predicted_boa"], s=80, alpha=0.75, c="black")

    min_v = float(np.min([df["var_99"].min(), df["predicted_boa"].min()]))
    max_v = float(np.max([df["var_99"].max(), df["predicted_boa"].max()]))

    ax3.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1.2)
    ax3.set_title("BoA factor method", fontsize=16)
    ax3.set_xlabel("Actual VaR 99%", fontsize=16)
    ax3.set_ylabel("Predicted VaR 99%", fontsize=16)
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.text(
        0.05,
        0.95,
        f"R² = {r2_b:.3f}\nRMSE = {rmse_b:.2f}",
        transform=ax3.transAxes,
        va="top",
        fontsize=13,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_FIG}")
    print(f"Observations used: {len(df)}")
    print(f"Gaussian: R2={r2_g:.3f}, RMSE={rmse_g:.2f}")
    print(f"BoA factor: R2={r2_b:.3f}, RMSE={rmse_b:.2f}")


if __name__ == "__main__":
    main()