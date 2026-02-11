"""
Balance Sheet Panel Construction Script

Builds quarterly balance sheet panel data from Refinitiv Excel exports.
Designed to be easily extensible - just add bank names to the BANKS list.

Author: Created for master thesis analysis
Date: 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import Dict, List, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input/Output paths
DATA_DIR = Path("data/raw/Balansesheet")
OUTPUT_DIR = Path("data/processed")

# Bank list - expand this list to add more banks
BANKS = [
    "bankofamerica",
    "jpmorgan",
    "citigroup",
    "wellsfargo",
    "goldmansachs",
    "morganstanley",
    "keycorp"    
]

# Sheet name to extract 
SHEET_NAME = "Balance Sheet"

# Output files
OUTPUT_BALANCED = OUTPUT_DIR / "balance_sheet_panel_balanced.csv"
OUTPUT_UNBALANCED = OUTPUT_DIR / "balance_sheet_panel_unbalanced.csv"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_column_name(name: str) -> str:
    """
    Clean and standardize column names.
    
    Examples:
        'Cash & Short-Term Deposits Due from Banks - Total' 
        -> 'cash_short_term_deposits_due_from_banks_total'
    """
    if pd.isna(name) or not isinstance(name, str):
        return ""
    
    # Convert to lowercase
    name = name.lower()
    
    # Replace common separators with spaces
    name = re.sub(r'[&/\-,()]+', ' ', name)
    
    # Remove special characters except spaces
    name = re.sub(r'[^a-z0-9\s]+', '', name)
    
    # Replace multiple spaces with single space
    name = re.sub(r'\s+', ' ', name)
    
    # Trim and replace spaces with underscores
    name = name.strip().replace(' ', '_')
    
    # Remove trailing/leading underscores
    name = name.strip('_')
    
    return name


def parse_refinitiv_balance_sheet(file_path: Path) -> pd.DataFrame:
    """
    Parse a Refinitiv Balance Sheet Excel export.
    
    The structure is:
    - Row 11 (index 11): Period End Dates
    - Row 17 (index 17): "Field Name" header with date columns
    - Row 18+ (index 18+): Field names and data
    
    Returns a long-format DataFrame with columns:
        - period_end_date: Quarter end date
        - field_name: Balance sheet line item (cleaned)
        - value: Numeric value
    """
    # Read the raw Excel file
    df_raw = pd.read_excel(file_path, sheet_name=SHEET_NAME, header=None)
    
    # Extract period end dates from row 11
    dates_row = df_raw.iloc[11, 1:].copy()  # Skip first column (label)
    dates = pd.to_datetime(dates_row, errors='coerce')
    dates = dates.dropna()
    
    # Find where actual data starts (row 18 is typically "Assets")
    data_start_row = 18
    
    # Extract field names (first column) and data
    field_names = df_raw.iloc[data_start_row:, 0].copy()
    data_values = df_raw.iloc[data_start_row:, 1:len(dates)+1].copy()
    
    # Reset index
    field_names = field_names.reset_index(drop=True)
    data_values = data_values.reset_index(drop=True)
    data_values.columns = range(len(data_values.columns))
    
    # Remove rows where field name is NaN or empty
    valid_rows = field_names.notna() & (field_names != '')
    field_names = field_names[valid_rows].reset_index(drop=True)
    data_values = data_values[valid_rows].reset_index(drop=True)
    
    # Build long format dataframe
    records = []
    for date_idx, date in enumerate(dates):
        if pd.isna(date):
            continue
            
        for row_idx in range(len(field_names)):
            field_name = field_names.iloc[row_idx]
            value = data_values.iloc[row_idx, date_idx]
            
            # Skip if field name is not valid
            if pd.isna(field_name) or field_name == '':
                continue
            
            # Clean field name
            clean_field = clean_column_name(field_name)
            if not clean_field:
                continue
            
            # Convert value to numeric
            numeric_value = pd.to_numeric(value, errors='coerce')
            
            records.append({
                'period_end_date': date,
                'field_name': clean_field,
                'original_field_name': field_name,
                'value': numeric_value
            })
    
    df_long = pd.DataFrame(records)
    
    return df_long


def load_bank_data(bank_id: str) -> pd.DataFrame:
    """
    Load balance sheet data for a single bank.
    
    Returns a long-format DataFrame with bank_id added.
    """
    file_path = DATA_DIR / f"{bank_id}.xlsx"
    
    if not file_path.exists():
        print(f"  File not found: {file_path}")
        return pd.DataFrame()
    
    try:
        df = parse_refinitiv_balance_sheet(file_path)
        df['bank_id'] = bank_id
        
        print(f"  ✓ {bank_id}: {len(df['period_end_date'].unique())} quarters, "
              f"{len(df['field_name'].unique())} fields")
        
        return df
    
    except Exception as e:
        print(f" Error loading {bank_id}: {str(e)}")
        return pd.DataFrame()


def build_panel(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long-format data to wide panel format.
    
    Returns a DataFrame with:
        - bank_id
        - period_end_date
        - [all balance sheet fields as columns]
    """
    # Pivot to wide format
    df_wide = df_long.pivot_table(
        index=['bank_id', 'period_end_date'],
        columns='field_name',
        values='value',
        aggfunc='first'  # In case of duplicates, take first
    ).reset_index()
    
    # Clean up column names
    df_wide.columns.name = None
    
    return df_wide


def create_balanced_panel(df_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Create a balanced panel containing only quarters where all banks have data.
    """
    # Count number of banks per quarter
    banks_per_quarter = (
        df_panel.groupby('period_end_date')['bank_id']
        .nunique()
        .reset_index()
        .rename(columns={'bank_id': 'n_banks'})
    )
    
    # Find quarters where all banks are present
    total_banks = df_panel['bank_id'].nunique()
    complete_quarters = banks_per_quarter[
        banks_per_quarter['n_banks'] == total_banks
    ]['period_end_date']
    
    # Filter to complete quarters
    df_balanced = df_panel[
        df_panel['period_end_date'].isin(complete_quarters)
    ].copy()
    
    return df_balanced


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.
    """
    print("=" * 70)
    print("BALANCE SHEET PANEL CONSTRUCTION")
    print("=" * 70)
    
    print(f"\nBanks to process: {len(BANKS)}")
    for bank in BANKS:
        print(f"  - {bank}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all bank data
    print("\nLoading bank data...")
    all_data = []
    
    for bank_id in BANKS:
        df_bank = load_bank_data(bank_id)
        if not df_bank.empty:
            all_data.append(df_bank)
    
    if not all_data:
        print("\n No data loaded. Exiting.")
        return 1
    
    # Combine all data
    print("\nCombining data...")
    df_combined = pd.concat(all_data, ignore_index=True)
    
    print(f"  Total observations: {len(df_combined):,}")
    print(f"  Banks: {df_combined['bank_id'].nunique()}")
    print(f"  Unique quarters: {df_combined['period_end_date'].nunique()}")
    print(f"  Unique fields: {df_combined['field_name'].nunique()}")
    
    # Build panel
    print("\nBuilding panel format...")
    df_panel = build_panel(df_combined)
    
    print(f"  Panel shape: {df_panel.shape}")
    print(f"  Columns: bank_id, period_end_date + {len(df_panel.columns) - 2} balance sheet fields")
    
    # Create balanced panel
    print("\nCreating balanced panel...")
    df_balanced = create_balanced_panel(df_panel)
    
    print(f"  Balanced panel shape: {df_balanced.shape}")
    print(f"  Quarters in balanced panel: {df_balanced['period_end_date'].nunique()}")
    print(f"  Date range: {df_balanced['period_end_date'].min()} to {df_balanced['period_end_date'].max()}")
    
    # Display summary statistics
    print("\nPanel coverage by bank:")
    coverage = (
        df_panel.groupby('bank_id')['period_end_date']
        .agg(['count', 'min', 'max'])
        .rename(columns={
            'count': 'quarters',
            'min': 'first_quarter',
            'max': 'last_quarter'
        })
    )
    print(coverage)
    
    # Save unbalanced panel
    print(f"\nSaving unbalanced panel to {OUTPUT_UNBALANCED}...")
    df_panel.to_csv(OUTPUT_UNBALANCED, index=False)
    print(f"  ✓ Saved {len(df_panel):,} rows")
    
    # Save balanced panel
    print(f"\nSaving balanced panel to {OUTPUT_BALANCED}...")
    df_balanced.to_csv(OUTPUT_BALANCED, index=False)
    print(f"  ✓ Saved {len(df_balanced):,} rows")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Unbalanced panel: {len(df_panel):,} observations across {df_panel['period_end_date'].nunique()} quarters")
    print(f"Balanced panel: {len(df_balanced):,} observations across {df_balanced['period_end_date'].nunique()} quarters")
    print(f"Balance sheet fields: {len(df_panel.columns) - 2}")
    print("\n✓ Panel construction complete!")
    
    return 0


if __name__ == "__main__":
    exit(main())