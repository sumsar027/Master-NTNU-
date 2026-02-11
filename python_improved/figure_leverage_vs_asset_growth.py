"""
Adrian - Shin (2010) Figure 2.4 Replica

This script reproduces the core idea of Figure 2.4 in Adrian & Shin (2010):
the relationship between quarterly leverage growth and asset growth.

What the script does:
1. Reads the cleaned balance sheet panel.
2. Computes leverage = total assets / common equity.
3. Computes quarterly log growth rates (in percent) for:
   - Total assets
   - Leverage
4. Plots leverage growth (x-axis) against asset growth (y-axis).

Input file:
    data/processed/balance_sheet_panel_balanced.csv

Required columns:
    bank_id
    period_end_date
    total_assets
    common_equity_total

Output:
    output/figures/figure_2_4_replica.png
"""


# Import the libraries we need
# numpy: math operations
# pandas: working with data tables
# matplotlib: plotting graphs
# pathlib: handling file paths
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Define where the project folder is located
ROOT = Path(__file__).resolve().parents[2]

# Define input data file (our cleaned balance sheet panel)
INPUT_FILE = ROOT / "data" / "processed" / "balance_sheet_panel_balanced.csv"

# Define where we want to save the figure
OUTPUT_FILE = ROOT / "output" / "figures" / "figure_2_4_replica.png"

# Make sure the output folder exists (create it if not)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# Read the balance sheet data into a dataframe
df = pd.read_csv(INPUT_FILE)

# Keep only the columns we actually need
df = df[["bank_id", "period_end_date", "total_assets", "common_equity_total"]].copy()

# Convert the date column into proper datetime format
df["period_end_date"] = pd.to_datetime(df["period_end_date"])

# Sort by bank and time (important before calculating growth)
df = df.sort_values(["bank_id", "period_end_date"])


# Remove observations with zero or negative values
# (logarithms cannot be taken of zero or negative numbers)
df = df[(df["total_assets"] > 0) & (df["common_equity_total"] > 0)]


# Calculate leverage = assets / equity
df["leverage"] = df["total_assets"] / df["common_equity_total"]


# Calculate quarterly growth rates using log differences
# 100 * (log(x_t) - log(x_{t-1})) ≈ percentage growth
# We calculate this separately for each bank
df["asset_growth"] = 100 * np.log(df["total_assets"]).groupby(df["bank_id"]).diff()
df["leverage_growth"] = 100 * np.log(df["leverage"]).groupby(df["bank_id"]).diff()

# Remove the first observation for each bank (no previous quarter to compare with)
df = df.dropna(subset=["asset_growth", "leverage_growth"])


# Create the figure and axis
fig, ax = plt.subplots(figsize=(9, 5.5))

# Scatter plot:
# x-axis = leverage growth
# y-axis = asset growth
ax.scatter(
    df["leverage_growth"],
    df["asset_growth"],
    c="#f3e21b",          # yellow color for points
    edgecolors="black",
    s=28,
    linewidths=0.5,
    alpha=0.9
)

# Axis labels
ax.set_xlabel("Leverage Growth (quarterly, %)")
ax.set_ylabel("Asset Growth (quarterly, %)")

# Fix axis limits to match Adrian-Shin style
ax.set_xlim(-15, 20)
ax.set_ylim(-15, 20)

# Title
ax.set_title("Adrian-Shin (2010) Figure 2.4 style")

# Add grid
ax.grid(True)

# Adjust layout and save figure
fig.tight_layout()
fig.savefig(OUTPUT_FILE, dpi=300)

# Close the figure to free memory
plt.close(fig)

print(f"Saved: {OUTPUT_FILE}")


