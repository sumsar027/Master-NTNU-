"""
Adrian–Shin (2010) Figure 2.4–style scatter plot
x-axis: Leverage Growth (quarterly, log change × 100)
y-axis: Total Asset Growth (quarterly, log change × 100)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------------------------
# Plot options
# -------------------------------------------------------------------
N_OUTLIERS_TO_LABEL = 10  # max labels (after threshold); set 0 to disable
OUTLIER_SCORE_THRESHOLD = 4.0  # higher => fewer labels; set 0 to disable thresholding
SHOW_PERPENDICULAR_LINE = False
X_COL = "leverage_growth"
Y_COL = "asset_growth"
START_QUARTER = "2014Q1"
END_QUARTER = "2025Q3"

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "processed" / "merged_quarterly_balanced.csv"
OUTPUT_FILE = ROOT / "output" / "figures" / "figure_8_replica.png"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
df = pd.read_csv(INPUT_FILE)

df = df[
    ["bank", "period_end_date", "total_assets_2", "common_equity_total"]
].copy()

df["total_assets_2"] = pd.to_numeric(df["total_assets_2"], errors="coerce")
df["common_equity_total"] = pd.to_numeric(df["common_equity_total"], errors="coerce")

df = df.dropna()
df = df[(df["total_assets_2"] > 0) & (df["common_equity_total"] > 0)]

df["quarter"] = pd.to_datetime(df["period_end_date"]).dt.to_period("Q")
df = df.sort_values(["bank", "quarter"])

# -------------------------------------------------------------------
# Construct leverage and log growth rates
# -------------------------------------------------------------------
df["leverage"] = df["total_assets_2"] / df["common_equity_total"]

df["asset_growth"] = (
    100 * np.log(df["total_assets_2"]).groupby(df["bank"]).diff()
)

df["leverage_growth"] = (
    100 * np.log(df["leverage"]).groupby(df["bank"]).diff()
)

df = df.dropna(subset=[X_COL, Y_COL])

# -------------------------------------------------------------------
# Restrict sample window (inclusive)
# -------------------------------------------------------------------
start_q = pd.Period(START_QUARTER, freq="Q")
end_q = pd.Period(END_QUARTER, freq="Q")
df = df[(df["quarter"] >= start_q) & (df["quarter"] <= end_q)].copy()

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
print(f"Number of banks: {df['bank'].nunique()}")
print(f"Number of quarters: {df['quarter'].nunique()}")
print(f"Total observations: {len(df)}")
if len(df) > 0:
    print(f"Quarter window: {df['quarter'].min()} to {df['quarter'].max()}")

# -------------------------------------------------------------------
# Outliers (label quarters)
# -------------------------------------------------------------------
def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(values)
    return 0.6745 * (values - med) / mad

zx = robust_z(df[X_COL].to_numpy())
zy = robust_z(df[Y_COL].to_numpy())
df["outlier_score"] = np.sqrt(zx**2 + zy**2)

q_start = df["quarter"].dt.start_time
df["quarter_label"] = [f"Q{d.quarter} {d.year}" for d in q_start]

quarter_extreme_idx = df.groupby("quarter")["outlier_score"].idxmax()
df_quarter_extremes = df.loc[quarter_extreme_idx].copy()

outlier_candidates = df_quarter_extremes
if OUTLIER_SCORE_THRESHOLD and OUTLIER_SCORE_THRESHOLD > 0:
    outlier_candidates = outlier_candidates[
        outlier_candidates["outlier_score"] >= OUTLIER_SCORE_THRESHOLD
    ]

outliers = (
    outlier_candidates.nlargest(N_OUTLIERS_TO_LABEL, "outlier_score")
    if N_OUTLIERS_TO_LABEL
    else df.iloc[0:0]
)

if len(outliers) > 0:
    labeled = ", ".join(
        f"{r.quarter_label} (score={r.outlier_score:.2f})"
        for r in outliers.sort_values("outlier_score", ascending=False).itertuples()
    )
    print(f"Labeled outliers: {labeled}")

# -------------------------------------------------------------------
# Plot (scatter, like the reference)
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))

ax.scatter(
    df[X_COL],       # x-axis
    df[Y_COL],       # y-axis
    s=28,
    color="blue",
    edgecolors="none",
    alpha=0.9
)

if len(outliers) > 0:
    ax.scatter(
        outliers[X_COL],
        outliers[Y_COL],
        s=46,
        color="red",
        edgecolors="black",
        linewidths=0.6,
        zorder=5,
    )
    for _, row in outliers.iterrows():
        ax.annotate(
            row["quarter_label"],
            (row[X_COL], row[Y_COL]),
            textcoords="offset points",
            xytext=(7, 7),
            fontsize=9,
            color="red",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
            zorder=6,
        )

ax.set_xlabel("Leverage Growth (log change × 100, quarterly)")
ax.set_ylabel("Total Asset Growth (log change × 100, quarterly)")
ax.set_title("Leverage Growth vs Asset Growth (Balanced Panel)")

ax.grid(True, linestyle="--", alpha=0.6)

# -------------------------------------------------------------------
# 45° line and perpendicular line
# -------------------------------------------------------------------
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
low = min(xmin, ymin) # can change to a fixed value or dynamic limits based on data
high = max(xmax, ymax)

ax.set_xlim(low, high) # can change to (low, high) for dynamic limits
ax.set_ylim(low, high) # can change to (low, high) for dynamic limits
ax.set_aspect("equal", adjustable="box")

# 45° line: y = x
ax.axline((0, 0), slope=1, linewidth=1)

if SHOW_PERPENDICULAR_LINE:
    ax.axline((0, 0), slope=-1, linestyle="--", linewidth=1)

# -------------------------------------------------------------------
# Save
# -------------------------------------------------------------------
fig.tight_layout()
fig.savefig(OUTPUT_FILE, dpi=300)
plt.show()
plt.close(fig)

print(f"Figure saved to: {OUTPUT_FILE}")
