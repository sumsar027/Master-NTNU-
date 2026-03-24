"""
Validate VaR conversion methods on Bank of America (Excel version).

What this script does (plain English)
-------------------------------------
1) Read the VaR Excel sheet (layout: bank names on row 2, levels on row 3).
2) Extract Bank of America VaR at 0.95 and 0.99.
3) Predict VaR(0.99) from VaR(0.95) using:
     - Gaussian ratio (z0.99 / z0.95)
     - BoA empirical factor (2.0)
4) Compare predictions to the actual VaR(0.99):
     - Time series
     - Scatter vs 45-degree line
     - Error distribution
5) Save one figure.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Conversion factors
# -----------------------------
# Gaussian ratio = z_0.99 / z_0.95
GAUSSIAN_RATIO = 2.326348 / 1.644854  # ≈ 1.414319
BOA_FACTOR = 2.0

# -----------------------------
# Paths
# -----------------------------
INPUT_FILE = Path("data/raw/VaR_python.xlsx")
SHEET_NAME = "new"

OUTPUT_DIR = Path("output/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG = OUTPUT_DIR / "boa_validation_comprehensive_excel.png"

# -----------------------------
# Helper: simple R2 and RMSE (no sklearn needed)
# -----------------------------
def r2_score_simple(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

def rmse_simple(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def norm_bank(x: str) -> str:
    # lower-case and keep only letters/numbers (to match the earlier logic)
    return "".join(ch for ch in str(x).lower().strip() if ch.isalnum())


def main():
    # 1) Read Excel WITHOUT headers (we build headers ourselves)
    raw = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME, header=None)

    # Excel layout (0-based indexing):
    # row 1 = bank names
    # row 2 = levels (0.95 / 0.99)
    # row 3+ = data
    bank_row = raw.iloc[1].ffill()
    level_row = raw.iloc[2]

    # Build column names like: "bankofamerica_0.95"
    cols = []
    for b, lvl in zip(bank_row, level_row):
        b_txt = norm_bank(b) if pd.notna(b) else ""
        lvl_txt = str(lvl).strip() if pd.notna(lvl) else ""
        cols.append(f"{b_txt}_{lvl_txt}")

    data = raw.iloc[3:].copy()
    data.columns = cols

    # 2) Extract BOA series
    # Date column is typically "year_Level" from this sheet layout
    if "year_Level" not in data.columns:
        raise KeyError("Missing 'year_Level' column. Excel layout changed.")

    boa_95_col = "bankofamerica_0.95"
    boa_99_col = "bankofamerica_0.99"
    if boa_95_col not in data.columns or boa_99_col not in data.columns:
        raise KeyError("Missing Bank of America 0.95/0.99 columns. Expected bankofamerica_0.95 and bankofamerica_0.99.")

    df = data[["year_Level", boa_95_col, boa_99_col]].copy()
    df.columns = ["period_end_date", "var_95", "var_99"]

    # Types
    df["period_end_date"] = pd.to_datetime(df["period_end_date"], errors="coerce")
    df["var_95"] = pd.to_numeric(df["var_95"], errors="coerce")
    df["var_99"] = pd.to_numeric(df["var_99"], errors="coerce")

    # Clean
    df = df.dropna(subset=["period_end_date", "var_95", "var_99"])
    df = df[(df["var_95"] > 0) & (df["var_99"] > 0)].copy()
    df = df.sort_values("period_end_date").reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No valid Bank of America observations with both var_95 and var_99.")

    # 3) Predictions
    df["predicted_gaussian"] = df["var_95"] * GAUSSIAN_RATIO
    df["predicted_boa"] = df["var_95"] * BOA_FACTOR

    # Errors (percentage, relative to actual 99%)
    df["error_gaussian_pct"] = 100 * (df["predicted_gaussian"] - df["var_99"]) / df["var_99"]
    df["error_boa_pct"] = 100 * (df["predicted_boa"] - df["var_99"]) / df["var_99"]

    # Metrics
    r2_g = r2_score_simple(df["var_99"].to_numpy(), df["predicted_gaussian"].to_numpy())
    rmse_g = rmse_simple(df["var_99"].to_numpy(), df["predicted_gaussian"].to_numpy())
    r2_b = r2_score_simple(df["var_99"].to_numpy(), df["predicted_boa"].to_numpy())
    rmse_b = rmse_simple(df["var_99"].to_numpy(), df["predicted_boa"].to_numpy())

    # 4) Figure (same idea as before)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # (1) Time series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df["period_end_date"], df["var_99"], marker="o", linewidth=2.5, label="Actual VaR 99%", c="#000000")
    ax1.plot(df["period_end_date"], df["predicted_gaussian"], marker="s", linestyle="--", linewidth=2,
             label=f"Gaussian (* {GAUSSIAN_RATIO:.4f})", c ="#D62728")
    ax1.plot(df["period_end_date"], df["predicted_boa"], marker="^", linestyle="--", linewidth=2,
             label=f"BoA factor (* {BOA_FACTOR:.1f})", c="#888888")
    ax1.set_title("Bank of America: VaR 99% — Actual vs Predicted")
    ax1.set_ylabel("VaR level")
    ax1.legend(frameon=False)
    ax1.grid(True, linestyle="--", alpha=0.4)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    # (2) Scatter – Gaussian
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(df["var_99"], df["predicted_gaussian"], s=70, alpha=0.75, c="black")
    min_v = float(np.min([df["var_99"].min(), df["predicted_gaussian"].min()]))
    max_v = float(np.max([df["var_99"].max(), df["predicted_gaussian"].max()]))
    ax2.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1)
    ax2.set_title("Gaussian method")
    ax2.set_xlabel("Actual VaR 99%")
    ax2.set_ylabel("Predicted VaR 99%")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.text(
        0.05, 0.95, f"R² = {r2_g:.3f}\nRMSE = {rmse_g:.2f}",
        transform=ax2.transAxes, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    # (3) Scatter – BoA factor
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(df["var_99"], df["predicted_boa"], s=70, alpha=0.75, c="black")
    min_v = float(np.min([df["var_99"].min(), df["predicted_boa"].min()]))
    max_v = float(np.max([df["var_99"].max(), df["predicted_boa"].max()]))
    ax3.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1)
    ax3.set_title("BoA factor method")
    ax3.set_xlabel("Actual VaR 99%")
    ax3.set_ylabel("Predicted VaR 99%")
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.text(
        0.05, 0.95, f"R² = {r2_b:.3f}\nRMSE = {rmse_b:.2f}",
        transform=ax3.transAxes, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    # Save
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_FIG}")
    print(f"Observations used: {len(df)}")
    print(f"Gaussian: R2={r2_g:.3f}, RMSE={rmse_g:.2f}")
    print(f"BoA factor: R2={r2_b:.3f}, RMSE={rmse_b:.2f}")


if __name__ == "__main__":
    main()
