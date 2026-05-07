"""
Build harmonized 99% VaR from the raw Excel source.

Banks report VaR at different confidence levels. This script puts all reported
VaR series on a common 99% basis so they can be merged into the panel dataset.
Reported 99% VaR is kept as-is; reported 95% VaR is scaled to 99% using the
normal-distribution quantile ratio.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm

# Input workbook with manually collected VaR data and output CSV used later by
# the panel-building script.
INPUT_FILE = "data/raw/VaR_python.xlsx"
OUTPUT_FILE = "output/data/var_99.csv"

# Gaussian scaling factor from the standard normal distribution.
z_095 = norm.ppf(0.95)  # 95% quantile
z_099 = norm.ppf(0.99)  # 99% quantile
GAUSSIAN_FACTOR = z_099 / z_095

def load_var_data(file_path):
    """Load raw VaR data from the Excel layout used in the source workbook."""
    # Read the worksheet without headers because metadata is stored in rows.
    df_raw = pd.read_excel(file_path, header=None)
    
    # Row 1 (index 1) contains bank names.
    # Row 2 (index 2) contains confidence levels.
    # Row 3 onward contains the time series observations.
    
    headers = df_raw.iloc[1].tolist()  # Bank names
    levels = df_raw.iloc[2].tolist()   # Confidence levels
    
    # Data start on row 3 (index 3).
    df = df_raw.iloc[3:].reset_index(drop=True).copy()
    df.columns = range(len(df.columns))
    
    # Parse dates from the first column.
    df[0] = pd.to_datetime(df[0], errors='coerce')
    df = df.dropna(subset=[0]).reset_index(drop=True)  # Keep only valid observation rows.
    
    # Assemble a cleaned table with explicit bank/level column names. This
    # converts the Excel layout into a normal table that is easier to process.
    result = pd.DataFrame()
    result['year'] = df[0]
    
    # Parse each bank column and attach its confidence level to the name.
    for i in range(1, len(headers)):
        if pd.isna(headers[i]):
            continue
            
        bank_name = str(headers[i]).strip()
        level_str = str(levels[i]).strip().replace('%', '').replace(' ', '')
        
        # Keep only the confidence levels used in the harmonization step.
        if '99' in level_str:
            level = '99'
        elif '95' in level_str:
            level = '95'
        else:
            continue
        
        # Example output names: "goldmansachs_95" or "citigroup_99".
        col_name = f"{bank_name}_{level}"
        result[col_name] = pd.to_numeric(df[i], errors='coerce')
    
    return result


def identify_var_columns(df):
    """Map VaR columns to their confidence level and bank identifier."""
    var_cols = {}
    
    for col in df.columns:
        if col == 'year':
            continue
            
        col_str = str(col).lower()
        
        # Infer the reported confidence level from the column name.
        if '95' in col_str:
            var_cols[col] = ('95', col.replace('_95', '').replace('95', ''))
        elif '99' in col_str:
            var_cols[col] = ('99', col.replace('_99', '').replace('99', ''))
    
    return var_cols


def harmonize_var(df):
    """Convert all available VaR observations to comparable 99% measures."""
    var_cols = identify_var_columns(df)
    
    results = []
    
    for idx, row in df.iterrows():
        date = row['year']
        
        # Group the available VaR observations by bank for each date.
        banks = {}
        for col, (level, bank) in var_cols.items():
            bank = bank.strip('_').strip()
            if bank not in banks:
                banks[bank] = {}
            banks[bank][level] = row[col]
        
        # Build one harmonized 99% VaR series for each bank-date observation.
        for bank, values in banks.items():
            var_95 = values.get('95', np.nan)
            var_99 = values.get('99', np.nan)
            
            # Use the reported 99% value when available; otherwise scale 95% VaR.
            if pd.notna(var_99):
                var_99_gaussian = var_99
            elif pd.notna(var_95):
                var_99_gaussian = var_95 * GAUSSIAN_FACTOR
            else:
                var_99_gaussian = np.nan

            results.append({
                'bank': bank,
                'date': date,
                'var_99_gaussian': var_99_gaussian
            })
    
    return pd.DataFrame(results)


def main():
    """Run the VaR harmonization pipeline and write the output file."""
    print("VaR Harmonization - Convert to 99%")
    print("=" * 60)
    
    # Load raw VaR data.
    print(f"\nLoading data from {INPUT_FILE}...")
    df = load_var_data(INPUT_FILE)
    print(f"Loaded {len(df)} rows")
    
    # Convert all series to a comparable 99% confidence level.
    print("\nConverting all VaR to 99% using Gaussian approximation:")
    print(f"  z_0.95 = {z_095:.6f}")
    print(f"  z_0.99 = {z_099:.6f}")
    print(f"  Conversion factor = {GAUSSIAN_FACTOR:.6f}")
    
    result = harmonize_var(df)
    
    # Report a short summary of the harmonized panel.
    print("\nSummary:")
    print(f"  Total observations: {len(result)}")
    print(f"  Banks: {result['bank'].nunique()}")
    print(f"  Date range: {result['date'].min()} to {result['date'].max()}")
    
    # Save the harmonized dataset.
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")
    print(f"Output columns: bank, date, var_99_gaussian")
    
    return result


if __name__ == "__main__":
    result = main()
