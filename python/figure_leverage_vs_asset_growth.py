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

# Set up file paths
ROOT = Path(__file__).resolve().parents[2]  # Go up 2 folders from this script
INPUT_FILE = ROOT / "data" / "processed" / "merged_quarterly_balanced.csv"
OUTPUT_FILE = ROOT / "output" / "figures" / "figure_2_4_replica.png"

# Make sure output folder exists
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Read the data
df = pd.read_csv(INPUT_FILE)

# Select and clean the columns we need
df = df[["bank", "period_end_date", "total_assets_2", "common_equity_total"]].copy()

# Convert to numeric (handles any text/errors in data)
df["total_assets_2"] = pd.to_numeric(df["total_assets_2"], errors="coerce")
df["common_equity_total"] = pd.to_numeric(df["common_equity_total"], errors="coerce")

# Remove rows with missing data or negative values
df = df.dropna()
df = df[(df["total_assets_2"] > 0) & (df["common_equity_total"] > 0)]

# Convert dates to quarterly periods
df["quarter"] = pd.to_datetime(df["period_end_date"]).dt.to_period("Q")

# Sort by bank and time (important for calculating growth!)
df = df.sort_values(["bank", "quarter"])

# Calculate leverage (assets / equity)
df["leverage"] = df["total_assets_2"] / df["common_equity_total"]

# Calculate quarterly growth rates using log differences
# Formula: growth = 100 * ln(value_t / value_t-1) = 100 * [ln(value_t) - ln(value_t-1)]
df["asset_growth"] = 100 * np.log(df["total_assets_2"]).groupby(df["bank"]).diff()
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
    df["leverage_growth"],      # x-axis
    df["asset_growth"],          # y-axis
    s=28,                        # point size
    c="#f3e21b",                 # yellow color
    edgecolors="black",          # black outline
    linewidths=0.5,
    alpha=0.9                    # slight transparency
)

# Labels and formatting
ax.set_xlabel("Leverage Growth (percent quarterly)")
ax.set_ylabel("Total Asset Growth (percent quarterly)")
ax.set_title("Total Asset Growth vs Leverage Growth (Balanced Panel)")
ax.set_xlim(-10, 20)
ax.set_ylim(-10, 20)
ax.grid(True, linestyle="--", alpha=0.6)

# Save the figure
fig.tight_layout()
fig.savefig(OUTPUT_FILE, dpi=300)
plt.show()  # Display it
plt.close(fig)

print(f"Figure saved to: {OUTPUT_FILE}")