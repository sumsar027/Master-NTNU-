"""
Bank of America VaR Level Check (99% vs 95%)

Goal:
Compute the ratio VaR(99%) / VaR(95%) for Bank of America over time,
based on the Excel layout where:
- Row 2 contains bank names
- Row 3 contains VaR level (0.95 / 0.99)
- Data starts from row 4
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = Path("data/raw/VaR_python.xlsx")  # change if your file is elsewhere
SHEET_NAME = "new"

OUTPUT_DIR_csv = Path("output/tables")
OUTPUT_DIR_csv.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR_png = Path("output/figures")
OUTPUT_DIR_png.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Load the sheet WITHOUT headers
# (we will build headers from row 2 and row 3)
# -----------------------------
raw = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME, header=None)

# Row indices (0-based):
# row 1 = bank names
# row 2 = VaR levels (0.95/0.99)
# row 3+ = data
bank_row = raw.iloc[1].copy()
level_row = raw.iloc[2].copy()

# Some columns have bank name only once and the next column is blank.
# Fill blank bank names using the previous bank name (so 0.95/0.99 stay under the right bank).
bank_row = bank_row.ffill()

# Build column labels like: "bankofamerica_0.99"
cols = []
for b, lvl in zip(bank_row, level_row):
    b = str(b).strip().lower() if pd.notna(b) else ""
    lvl = str(lvl).strip() if pd.notna(lvl) else ""
    cols.append(f"{b}_{lvl}")

# Data part starts from row 3
data = raw.iloc[3:].copy()
data.columns = cols

# -----------------------------
# Extract Bank of America 0.95 and 0.99 series + date column
# -----------------------------
# Date column is in the first column labeled "year_Level" (from the sheet structure)
# We rename it to "date" for clarity.
if "year_Level" not in data.columns:
    raise KeyError("Could not find the date column 'year_Level'. The Excel layout may have changed again.")

boa_95_col = "bankofamerica_0.95"
boa_99_col = "bankofamerica_0.99"

if boa_95_col not in data.columns or boa_99_col not in data.columns:
    raise KeyError(
        "Could not find Bank of America columns for 0.95 and 0.99. "
        "Expected columns: 'bankofamerica_0.95' and 'bankofamerica_0.99'."
    )

boa = data[["year_Level", boa_95_col, boa_99_col]].copy()
boa.columns = ["date", "var_95", "var_99"]

# Convert types
boa["date"] = pd.to_datetime(boa["date"], errors="coerce")
boa["var_95"] = pd.to_numeric(boa["var_95"], errors="coerce")
boa["var_99"] = pd.to_numeric(boa["var_99"], errors="coerce")

# Drop missing + require positive values
boa = boa.dropna(subset=["date", "var_95", "var_99"])
boa = boa[(boa["var_95"] > 0) & (boa["var_99"] > 0)].copy()

# Ratio
boa["ratio"] = boa["var_99"] / boa["var_95"]

# Sort by date
boa = boa.sort_values("date").reset_index(drop=True)

# -----------------------------
# Print summary stats
# -----------------------------
print("=" * 60)
print("Bank of America: VaR 99% / VaR 95% Ratio")
print("=" * 60)

print(f"\nObservations: {len(boa)}")
print(f"Mean:   {boa['ratio'].mean():.4f}")
print(f"Median: {boa['ratio'].median():.4f}")
print(f"Std:    {boa['ratio'].std():.4f}")
print(f"Min:    {boa['ratio'].min():.4f}")
print(f"Max:    {boa['ratio'].max():.4f}")

print("\nFirst 10 rows:")
print(boa.head(10).to_string(index=False))

# -----------------------------
# Plot ratio over time
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(boa["date"], boa["ratio"], marker="o", linewidth=1.5)

median_val = boa["ratio"].median()
plt.axhline(median_val, color="red", linestyle="--", linewidth=2, label=f"Median = {median_val:.4f}")

plt.xlabel("Date")
plt.ylabel("VaR 99% / VaR 95%")
plt.title("Bank of America: VaR 99% / VaR 95% Ratio Over Time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()



# Save CSV
csv_path = OUTPUT_DIR_csv / "bank_of_america_var_ratio.csv"
boa.to_csv(csv_path, index=False)
print(f"CSV saved: {csv_path}")

#save png
png_path = OUTPUT_DIR_png / "bank_of_america_var_ratio.png"
plt.savefig(png_path, dpi=300, bbox_inches="tight")
print(f"PNG saved: {png_path}")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)
