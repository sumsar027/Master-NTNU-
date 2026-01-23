"""
Bank of America VaR Level Check (99% vs 95%) — what this script does

Purpose
-------
This script is a *sanity check* for converting 95% VaR to 99% VaR. It looks for
Bank of America observations where both 95% and 99% VaR are reported, then
computes the ratio (VaR99 / VaR95) over time.

Inputs
------
- data/raw/VaR_python.xlsx
  The Excel file may contain multiple sheets. The script automatically selects
  the first sheet whose column names contain both "95" and "99".

Core steps
----------
1) Locate the Excel file by searching common locations.
2) List available Excel sheets and auto-select a sheet that contains both "95" and "99"
   in its column names.
3) Identify the VaR 95% column and VaR 99% column (by keywords in column names).
4) Identify a time column (date/quarter/period-like).
5) If the data is in "long format" (has an entity/name/ticker column), filter rows to
   Bank of America. Otherwise assume "wide format" and use the selected VaR columns directly.
6) Keep only rows where both VaR95 and VaR99 are present.
7) Compute the ratio: ratio = VaR99 / VaR95
8) Print summary statistics and the first 10 observations.
9) Save:
   - A PNG plot of the ratio over time
   - A CSV with time, VaR95, VaR99, ratio

Outputs
-------
- output/bank_of_america_var_ratio.png
- output/bank_of_america_var_ratio.csv

Notes / interpretation
----------------------
- A stable ratio suggests a stable scaling relationship between 95% and 99% VaR for BoA.
- This is not a cross-bank validation; it is specifically for Bank of America and only
  for periods where both levels are available.
"""

import pandas as pd
import os
from pathlib import Path


def find_excel_file(filename: str) -> Path:
    """
    Find the Excel file by checking multiple common locations.
    Returns the first match found.
    """
    candidates = [
        Path(filename),
        Path(__file__).parent / filename,
        Path(__file__).parent / "output" / filename,
        Path("output") / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"Could not find '{filename}' in any standard location")


# Output directory (next to script); keep as-is to avoid changing behavior
output_dir = Path("output/figures")
output_dir.mkdir(parents=True, exist_ok=True)

output_dir.mkdir(exist_ok=True)

# Load Excel file
excel_path = find_excel_file("data/raw/VaR_python.xlsx")
print(f"Reading Excel file: {excel_path}")
print("=" * 80)

# Read all sheets and display names
excel_file = pd.ExcelFile(excel_path)
print("\nAvailable sheets:")
for i, sheet in enumerate(excel_file.sheet_names, 1):
    print(f"  {i}. {sheet}")

# Find sheet containing both 95 and 99 in column names
selected_sheet = None
for sheet_name in excel_file.sheet_names:
    df_temp = pd.read_excel(excel_file, sheet_name=sheet_name)
    columns_text = " ".join(str(col).lower() for col in df_temp.columns)

    if "95" in columns_text and "99" in columns_text:
        selected_sheet = sheet_name
        print(f"\nSelected sheet: '{selected_sheet}' (contains 95 and 99)")
        break

if not selected_sheet:
    print("ERROR: No sheet found containing both 95 and 99 in column names")
    raise SystemExit(1)

# Load the selected sheet
df = pd.read_excel(excel_file, sheet_name=selected_sheet)
print(f"Rows in sheet: {len(df)}")
print(f"Columns: {list(df.columns)}\n")

# Identify VaR columns (looking for 95 and 99)
var95_col = None
var99_col = None

for col in df.columns:
    col_lower = str(col).lower()
    if "95" in col_lower and ("var" in col_lower or "bank" in col_lower):
        var95_col = col
    if "99" in col_lower and ("var" in col_lower or "bank" in col_lower):
        var99_col = col

print("Identified columns:")
print(f"  VaR 95%: {var95_col}")
print(f"  VaR 99%: {var99_col}")

if not var95_col or not var99_col:
    print("ERROR: Could not identify VaR columns")
    raise SystemExit(1)

# Identify time column
time_col = None
time_keywords = ["date", "quarter", "period", "report", "time", "year"]

for col in df.columns:
    col_lower = str(col).lower()
    if any(keyword in col_lower for keyword in time_keywords):
        time_col = col
        break

print(f"  Time column: {time_col}\n")

if not time_col:
    print("ERROR: Could not identify time column")
    raise SystemExit(1)

# Check if there's a bank/entity column (long format data)
bank_col = None
bank_keywords = ["entity", "name", "ticker", "institution", "company"]

for col in df.columns:
    col_lower = str(col).lower()
    if "95" not in col_lower and "99" not in col_lower:
        if any(keyword in col_lower for keyword in bank_keywords):
            bank_col = col
            break

# Filter to Bank of America if we have a bank column (long format)
if bank_col:
    print("Filtering to Bank of America (long format data)...")
    df_boa = df[
        df[bank_col].astype(str).str.lower().str.contains("bank.*america", regex=True, na=False)
    ].copy()
    print(f"Rows for Bank of America: {len(df_boa)}")
else:
    print("Using wide format (Bank of America columns directly)...")
    df_boa = df[[time_col, var95_col, var99_col]].copy()

# Keep only rows where both VaR95 and VaR99 are present
df_boa = df_boa.dropna(subset=[var95_col, var99_col])
print(f"Rows with both VaR 95% and VaR 99%: {len(df_boa)}\n")

if len(df_boa) == 0:
    print("=" * 80)
    print("Cannot perform analysis: no overlapping VaR observations found.")
    print("=" * 80)
    raise SystemExit(0)

# Calculate VaR99/VaR95 ratio
df_boa["ratio"] = df_boa[var99_col] / df_boa[var95_col]

# Sort by time
df_boa = df_boa.sort_values(by=time_col)

# Statistics
median_ratio = df_boa["ratio"].median()
mean_ratio = df_boa["ratio"].mean()
std_ratio = df_boa["ratio"].std()
min_ratio = df_boa["ratio"].min()
max_ratio = df_boa["ratio"].max()

print("VaR99/VaR95 Ratio Statistics:")
print(f"  Observations: {len(df_boa)}")
print(f"  Mean: {mean_ratio:.4f}")
print(f"  Median: {median_ratio:.4f}")
print(f"  Std Dev: {std_ratio:.4f}")
print(f"  Min: {min_ratio:.4f}")
print(f"  Max: {max_ratio:.4f}\n")

# First 10 observations
print("First 10 observations:")
print("-" * 80)
output_df = df_boa[[time_col, var95_col, var99_col, "ratio"]].head(10)
for _, row in output_df.iterrows():
    print(f"{row[time_col]}\t{row[var95_col]:.2f}\t{row[var99_col]:.2f}\t{row['ratio']:.4f}")
print()

# Plot
try:
    mpl_config_dir = output_dir / ".mplconfig"
    mpl_config_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(df_boa[time_col], df_boa["ratio"], marker="o", linestyle="-", linewidth=1.5)
    plt.axhline(
        y=median_ratio,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Median = {median_ratio:.4f}",
    )
    plt.xlabel("Time")
    plt.ylabel("VaR 99% / VaR 95%")
    plt.title("Bank of America: VaR 99% / VaR 95% Ratio Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    png_path = output_dir / "bank_of_america_var_ratio.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved: {png_path}")

except Exception as e:
    print(f"Could not create plot: {e}")

# Save CSV
csv_df = df_boa[[time_col, var95_col, var99_col, "ratio"]].copy()
csv_df.columns = ["time", "var_95", "var_99", "ratio"]
csv_path = output_dir / "bank_of_america_var_ratio.csv"

try:
    csv_df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")
except Exception as e:
    print(f"Could not save CSV: {e}")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)
