"""
Simple Adrian - Shin (2010) Figure 2.4 - style scatter plot:

- x-axis: Total Asset Growth (quarterly, percent)
- y-axis: Leverage Growth (quarterly, percent)

Default input (this repo):
  output/data/merged_quarterly_balanced.csv

Assumed columns (override via flags if needed):
  bank = bank
  date = period_end_date
  assets = total_assets (fallback: total_assets_2, assets)
  equity = common_equity_total (fallback: tangible_total_equity)

Run:
  /opt/anaconda3/bin/python src/pipeline/lev_growth_asset_growth_figure_simple.py
"""

"""
Adrian-Shin (2010) Figure 2.4 replica: Scatter plot of leverage growth vs asset growth

This script:
1. Reads quarterly bank panel data
2. Calculates quarterly growth rates for assets and leverage
3. Creates a scatter plot showing the relationship between them
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


# Set up file paths
ROOT = Path(__file__).resolve().parents[2]  # Go up 2 folders from this script
INPUT_FILE = ROOT / "data" / "processed" / "merged_quarterly_balanced.csv"
OUTPUT_FILE = ROOT / "output" / "figures" / "figure_2_4_replica.png"

# Make sure output folder exists
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Read the data
df = pd.read_csv(INPUT_FILE)

# --- KEEP ONLY GROUP 1 BANKS ---
GROUP_1 = [
    "Bank of America",
    "Citigroup",
    "Goldman Sachs",
    "JPMorgan Chase",
    "Morgan Stanley",
    "Wells Fargo",
]

df = df[df["bank"].isin(GROUP_1)].copy()
#remove from GROUP_1 to here if you want all 11 banks

assets_primary = _find_column(df, ["total_assets"])
assets_fallback = _find_column(df, ["total_assets_2"])
equity_col = _find_column(df, ["common_equity_total"])

if assets_primary is None and assets_fallback is None:
    raise ValueError(
        "Could not find an assets column. Tried: total_assets or total_assets_2"
    )
if equity_col is None:
    raise ValueError(
        "Could not find an equity column. Tried: common_equity_total."
    )

cols = ["bank", "period_end_date", equity_col]
if assets_primary is not None:
    cols.append(assets_primary)
if assets_fallback is not None and assets_fallback not in cols:
    cols.append(assets_fallback)
df = df[cols].copy()

# Convert to numeric (handles any text/errors in data)
equity = pd.to_numeric(df[equity_col], errors="coerce")
assets = None
if assets_primary is not None:
    assets = pd.to_numeric(df[assets_primary], errors="coerce")
if assets_fallback is not None:
    assets_fb = pd.to_numeric(df[assets_fallback], errors="coerce")
    assets = assets_fb if assets is None else assets.fillna(assets_fb)

df["assets"] = assets
df["equity"] = equity

# Remove rows with missing data or negative values
df = df.dropna()
df = df[(df["assets"] > 0) & (df["equity"] > 0)]

# Convert dates to quarterly periods
df["quarter"] = pd.to_datetime(df["period_end_date"]).dt.to_period("Q")

# Sort by bank and time (important for calculating growth!)
df = df.sort_values(["bank", "quarter"])

# Calculate leverage (assets / equity)
df["leverage"] = df["assets"] / df["equity"]

# Calculate quarterly growth rates using log differences
# Formula: growth = 100 * log(value_t / value_t-1) = 100 * [log(value_t) - log(value_t-1)]
df["asset_growth"] = 100 * np.log(df["assets"]).groupby(df["bank"]).diff()
df["leverage_growth"] = 100 * np.log(df["leverage"]).groupby(df["bank"]).diff()

# Remove the first observation for each bank (no growth rate available)
df = df.dropna()

# Print summary statistics
print(f"Number of banks: {df['bank'].nunique()}")
print(f"Number of quarters: {df['quarter'].nunique()}")
print(f"Total observations: {len(df)}")

# Create the scatter plot
fig, ax = plt.subplots(figsize=(9, 5.5))

ax.scatter(      
    df["asset_growth"],  # x-axis: Total Asset Growth
    df["leverage_growth"], # y-axis: Leverage Growth
    s=28,                        # point size
    c="#f3e21b",                 # yellow color
    edgecolors="black",          # black outline
    linewidths=0.5,
    alpha=0.9                    # slight transparency
)

# Labels and formatting
ax.set_xlabel("Total Leverage Growth (quarterly, %)", fontsize=12)
ax.set_ylabel("Total Asset Growth (quarterly, %)", fontsize=12)
ax.set_title("Asset Growth vs Leverage Growth (Balanced Panel)")
ax.set_xlim(-10, 20)
ax.set_ylim(-10, 20)
ax.grid(True, linestyle="--", alpha=0.6)

# Save the figure
fig.tight_layout()
fig.savefig(OUTPUT_FILE, dpi=300)
plt.show()  # Display it
plt.close(fig)

print(f"Figure saved to: {OUTPUT_FILE}")
