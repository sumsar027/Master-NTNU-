"""
Balance Sheet Panel Construction Script

Builds quarterly balance sheet panel data from Refinitiv Excel exports.
Reads two files per bank:
  1. data/raw/Balansesheet/<bank_id>.xlsx          - "Balance Sheet" tab
  2. data/raw/Financial/<fin_name>_financial.xlsx  - "Financial Summary" tab
     (only dividend fields extracted)

Author: Created for master thesis analysis
Date: 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

BS_DIR     = Path("data/raw/Balansesheet")   # e.g. jpmorgan.xlsx
FIN_DIR    = Path("data/raw/Financial")       # e.g. jp_morgan_financial.xlsx
OUTPUT_DIR = Path("data/processed")

# Balance sheet bank_id list (filenames in Balansesheet/)
BANKS = [
    "bankofamerica",
    "jpmorgan",
    "citigroup",
    "wellsfargo",
    "goldmansachs",
    "morganstanley",
    "statestreet",
    "bny",
]

# Mapping: bank_id -> filename stem in data/raw/Financial/
FINANCIAL_FILENAMES = {
    "bankofamerica": "bank_of_america_financial",
    "jpmorgan":      "jp_morgan_financial",
    "citigroup":     "citigroup_financial",
    "wellsfargo":    "wells_fargo_financial",
    "goldmansachs":  "goldman_sachs_financial",
    "morganstanley": "morgan_stanley_financial",
    "statestreet":   "statestreet_financial",
    "bny":           "bny_financial",
}

BALANCE_SHEET_TAB     = "Balance Sheet"
FINANCIAL_SUMMARY_TAB = "Financial Summary"

OUTPUT_BALANCED   = OUTPUT_DIR / "balance_sheet_panel_balanced.csv"
OUTPUT_UNBALANCED = OUTPUT_DIR / "balance_sheet_panel_unbalanced.csv"


# ============================================================================
# HELPERS
# ============================================================================

def clean_column_name(name: str) -> str:
    """Standardize a Refinitiv field name to snake_case."""
    if pd.isna(name) or not isinstance(name, str):
        return ""
    name = name.lower()
    name = re.sub(r'[&/\-,()%]+', ' ', name)
    name = re.sub(r'[^a-z0-9\s]+', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name.strip().replace(' ', '_').strip('_')


# ============================================================================
# PARSERS
# ============================================================================

def _parse_refinitiv_sheet(file_path: Path, sheet_name: str,
                            field_filter: set = None) -> pd.DataFrame:
    """
    Parse any Refinitiv sheet with the standard layout:
      Row 11  - Period End Dates
      Row 18+ - Field name | values...

    field_filter: if given, only keep rows where the cleaned field name
                  contains at least one of the strings in the set.
    Returns long-format DataFrame (period_end_date, field_name, value).
    """
    df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    dates = pd.to_datetime(df_raw.iloc[11, 1:], errors='coerce').dropna()

    field_names = df_raw.iloc[18:, 0].reset_index(drop=True)
    data_values = df_raw.iloc[18:, 1:len(dates) + 1].reset_index(drop=True)
    data_values.columns = range(len(data_values.columns))

    valid = field_names.notna() & (field_names != '')
    field_names = field_names[valid].reset_index(drop=True)
    data_values = data_values[valid].reset_index(drop=True)

    records = []
    for date_idx, date in enumerate(dates):
        if pd.isna(date):
            continue
        for row_idx in range(len(field_names)):
            raw = field_names.iloc[row_idx]
            if pd.isna(raw) or raw == '':
                continue
            clean = clean_column_name(raw)
            if not clean:
                continue
            if field_filter and not any(f in clean for f in field_filter):
                continue
            records.append({
                'period_end_date': date,
                'field_name':      clean,
                'value':           pd.to_numeric(
                    data_values.iloc[row_idx, date_idx], errors='coerce'),
            })

    return pd.DataFrame(records)


def parse_balance_sheet(file_path: Path) -> pd.DataFrame:
    """Parse all fields from the Balance Sheet tab."""
    return _parse_refinitiv_sheet(file_path, BALANCE_SHEET_TAB)


def parse_financial_summary(file_path: Path) -> pd.DataFrame:
    return _parse_refinitiv_sheet(
        file_path, FINANCIAL_SUMMARY_TAB,
        field_filter={"dividend", "return_on_average_total_assets"}
    )

# ============================================================================
# BANK LOADER
# ============================================================================

def load_bank_data(bank_id: str) -> pd.DataFrame:
    """
    Load Balance Sheet + dividend data for one bank.
    Balance sheet: data/raw/Balansesheet/<bank_id>.xlsx
    Dividends:     data/raw/Financial/<fin_stem>.xlsx
    """
    bs_path  = BS_DIR  / f"{bank_id}.xlsx"
    fin_stem = FINANCIAL_FILENAMES.get(bank_id, f"{bank_id}_financial")
    fin_path = FIN_DIR / f"{fin_stem}.xlsx"

    if not bs_path.exists():
        print(f"  X Balance sheet not found: {bs_path}")
        return pd.DataFrame()

    try:
        df_bs = parse_balance_sheet(bs_path)
    except Exception as e:
        print(f"  X [{bank_id}] Balance sheet error: {e}")
        return pd.DataFrame()

    # Dividend data from separate Financial file
    df_fin = pd.DataFrame()
    if fin_path.exists():
        try:
            df_fin = parse_financial_summary(fin_path)
        except Exception as e:
            print(f"  ! [{bank_id}] Financial summary error: {e}")
    else:
        print(f"  ! [{bank_id}] Financial file not found: {fin_path}")

    frames = [df_bs] + ([df_fin] if not df_fin.empty else [])
    df = pd.concat(frames, ignore_index=True)
    df['bank_id'] = bank_id

    div_fields = sorted(df.loc[
        df['field_name'].str.contains('dividend', na=False), 'field_name'].unique())
    print(f"  ok {bank_id}: {df['period_end_date'].nunique()} quarters, "
          f"{df['field_name'].nunique()} fields | dividends: {div_fields or 'none'}")

    return df


# ============================================================================
# PANEL BUILDER
# ============================================================================

def build_panel(df_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format data to wide panel."""
    df_wide = df_long.pivot_table(
        index=['bank_id', 'period_end_date'],
        columns='field_name',
        values='value',
        aggfunc='first'
    ).reset_index()
    df_wide.columns.name = None
    return df_wide


def create_balanced_panel(df_panel: pd.DataFrame) -> pd.DataFrame:
    """Keep only quarters where ALL banks have observations."""
    total_banks = df_panel['bank_id'].nunique()
    complete_quarters = (
        df_panel.groupby('period_end_date')['bank_id']
        .nunique()
        .pipe(lambda s: s[s == total_banks])
        .index
    )
    return df_panel[df_panel['period_end_date'].isin(complete_quarters)].copy()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("BALANCE SHEET PANEL CONSTRUCTION")
    print(f"  Balance sheets : {BS_DIR}")
    print(f"  Financial data : {FIN_DIR}")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {len(BANKS)} banks...")
    all_data = [load_bank_data(b) for b in BANKS]
    all_data = [d for d in all_data if not d.empty]

    if not all_data:
        print("\nX No data loaded. Exiting.")
        return 1

    df_combined = pd.concat(all_data, ignore_index=True)
    print(f"\nCombined: {len(df_combined):,} obs | "
          f"{df_combined['bank_id'].nunique()} banks | "
          f"{df_combined['period_end_date'].nunique()} quarters | "
          f"{df_combined['field_name'].nunique()} fields")

    # Dividend coverage report
    div_rows = df_combined[df_combined['field_name'].str.contains('dividend', na=False)]
    if not div_rows.empty:
        print(f"\nDividend fields: {sorted(div_rows['field_name'].unique())}")
        cov = div_rows.groupby(['field_name', 'bank_id'])['value'].count().unstack(fill_value=0)
        print("\nDividend coverage (quarters per bank):")
        print(cov.to_string())
    else:
        print("\n! No dividend fields found.")

    # Build panels
    df_panel    = build_panel(df_combined)
    df_balanced = create_balanced_panel(df_panel)

    print(f"\nUnbalanced panel : {df_panel.shape}")
    print(f"Balanced panel   : {df_balanced.shape}  "
          f"({df_balanced['period_end_date'].nunique()} quarters, "
          f"{df_balanced['period_end_date'].min().date()} - "
          f"{df_balanced['period_end_date'].max().date()})")

    # Coverage by bank
    print("\nCoverage by bank:")
    print(df_panel.groupby('bank_id')['period_end_date']
          .agg(['count', 'min', 'max'])
          .rename(columns={'count': 'quarters', 'min': 'first', 'max': 'last'}))

    # Save
    df_panel.to_csv(OUTPUT_UNBALANCED, index=False)
    print(f"\nok Saved unbalanced -> {OUTPUT_UNBALANCED}  ({len(df_panel):,} rows)")

    df_balanced.to_csv(OUTPUT_BALANCED, index=False)
    print(f"ok Saved balanced   -> {OUTPUT_BALANCED}  ({len(df_balanced):,} rows)")

    div_cols = [c for c in df_panel.columns if 'dividend' in c]
    print(f"\nDividend columns in output: {div_cols}")
    print("\nok Panel construction complete!")
    return 0


if __name__ == "__main__":
    exit(main())