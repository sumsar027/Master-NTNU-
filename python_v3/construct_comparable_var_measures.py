"""Simplified VaR approximation script.

Converts all VaR values to 99% confidence level using two methods:
1. Gaussian approximation: multiply 95% VaR by (z_0.99 / z_0.95)
2. Bank of America empirical factor: multiply 95% VaR by 2.0
"""

import pandas as pd
import numpy as np
from scipy.stats import norm

# Configuration
INPUT_FILE = "data/raw/VaR_python.xlsx"
OUTPUT_FILE = "output/data/var_99.csv"

# Calculate Gaussian conversion factor from standard normal distribution
z_095 = norm.ppf(0.95)  # 95% quantile
z_099 = norm.ppf(0.99)  # 99% quantile
GAUSSIAN_FACTOR = z_099 / z_095

# Bank of America empirical factor
BOA_FACTOR = 2.0


def load_var_data(file_path):
    """Load VaR data from Excel file with level row format."""
    # Read raw data without header
    df_raw = pd.read_excel(file_path, header=None)
    
    # Row 1 (index 1): column headers (year, goldmansachs, morganstanley, etc.)
    # Row 2 (index 2): levels (95%, 95%, 95%, 99%, etc.)
    # Row 3+ (index 3+): data
    
    headers = df_raw.iloc[1].tolist()  # Bank names
    levels = df_raw.iloc[2].tolist()   # Confidence levels
    
    # Data starts from row 3 (index 3)
    df = df_raw.iloc[3:].reset_index(drop=True).copy()
    df.columns = range(len(df.columns))
    
    # Parse dates (first column)
    df[0] = pd.to_datetime(df[0], errors='coerce')
    df = df.dropna(subset=[0]).reset_index(drop=True)  # Remove rows without dates
    
    # Build result with proper column names
    result = pd.DataFrame()
    result['year'] = df[0]
    
    # Process each bank column
    for i in range(1, len(headers)):
        if pd.isna(headers[i]):
            continue
            
        bank_name = str(headers[i]).strip()
        level_str = str(levels[i]).strip().replace('%', '').replace(' ', '')
        
        # Determine level (95 or 99)
        if '99' in level_str:
            level = '99'
        elif '95' in level_str:
            level = '95'
        else:
            continue
        
        # Create column name like "goldmansachs_95" or "citigroup_99"
        col_name = f"{bank_name}_{level}"
        result[col_name] = pd.to_numeric(df[i], errors='coerce')
    
    return result


def identify_var_columns(df):
    """Identify which columns contain VaR data and their confidence levels."""
    var_cols = {}
    
    for col in df.columns:
        if col == 'year':
            continue
            
        col_str = str(col).lower()
        
        # Check if column ends with 95 or 99
        if '95' in col_str:
            var_cols[col] = ('95', col.replace('_95', '').replace('95', ''))
        elif '99' in col_str:
            var_cols[col] = ('99', col.replace('_99', '').replace('99', ''))
    
    return var_cols


def harmonize_var(df):
    """Convert all VaR values to 99% using both methods."""
    var_cols = identify_var_columns(df)
    
    results = []
    
    for idx, row in df.iterrows():
        date = row['year']
        
        # Group columns by bank
        banks = {}
        for col, (level, bank) in var_cols.items():
            bank = bank.strip('_').strip()
            if bank not in banks:
                banks[bank] = {}
            banks[bank][level] = row[col]
        
        # Create 99% VaR for each bank using both methods
        for bank, values in banks.items():
            var_95 = values.get('95', np.nan)
            var_99 = values.get('99', np.nan)
            
            # Method 1: Gaussian approximation
            if pd.notna(var_99):
                var_99_gaussian = var_99
            elif pd.notna(var_95):
                var_99_gaussian = var_95 * GAUSSIAN_FACTOR
            else:
                var_99_gaussian = np.nan
            
            # Method 2: BoA empirical factor
            if pd.notna(var_99):
                var_99_boa = var_99
            elif pd.notna(var_95):
                var_99_boa = var_95 * BOA_FACTOR
            else:
                var_99_boa = np.nan
            
            results.append({
                'bank': bank,
                'date': date,
                'var_99_gaussian': var_99_gaussian,
                'var_99_boa_factor': var_99_boa
            })
    
    return pd.DataFrame(results)


def main():
    """Main execution."""
    print("VaR Harmonization - Convert to 99%")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from {INPUT_FILE}...")
    df = load_var_data(INPUT_FILE)
    print(f"Loaded {len(df)} rows")
    
    # Harmonize
    print("\nConverting all VaR to 99% using two methods:")
    print(f"  1. Gaussian approximation:")
    print(f"     z_0.95 = {z_095:.6f}")
    print(f"     z_0.99 = {z_099:.6f}")
    print(f"     Conversion factor = {GAUSSIAN_FACTOR:.6f}")
    print(f"  2. Bank of America empirical factor: {BOA_FACTOR}")
    
    result = harmonize_var(df)
    
    # Summary
    print("\nSummary:")
    print(f"  Total observations: {len(result)}")
    print(f"  Banks: {result['bank'].nunique()}")
    print(f"  Date range: {result['date'].min()} to {result['date'].max()}")
    
    # Save
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")
    print(f"Output columns: bank, date, var_99_gaussian, var_99_boa_factor")
    
    return result


if __name__ == "__main__":
    result = main()