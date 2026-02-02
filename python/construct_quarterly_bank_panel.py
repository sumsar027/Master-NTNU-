"""
Refinitiv Excel Reader - Direct to Quarterly Datasets 
Reads all Refinitiv Excel files and creates quarterly balanced/unbalanced CSVs directly.

CRITICAL FIX: Removes ALL prefix from column names so they match your existing scripts!
- 'Total Assets' (NOT 'financial_summary_Total Assets')
- 'Common Equity - Total' (NOT 'financial_summary_Common Equity - Total')

Output:
    data/processed/merged_quarterly_balanced.csv
    data/processed/merged_quarterly_unbalanced.csv

Usage:
    python construct_quarterly_bank_panel.py
"""

import pandas as pd
from pathlib import Path
import re


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def clean_column_name(name: object) -> str:
    """Convert Refinitiv field names to stable snake_case."""
    text = str(name).strip().lower()
    text = text.replace("&", " and ")
    text = _NON_ALNUM_RE.sub("_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "col"


def dedupe_names(names: list[str]) -> list[str]:
    """Make a list of names unique by appending _2, _3, ... where needed."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        count = seen.get(name, 0) + 1
        seen[name] = count
        out.append(name if count == 1 else f"{name}_{count}")
    return out


def make_bank_id(bank: object) -> str:
    """Create a merge-friendly bank_id (matches other pipeline scripts)."""
    bank_id = clean_column_name(bank)
    bank_id = bank_id.replace("u_s_", "us_").replace("_u_s_", "_us_")
    return bank_id


def merge_keep_duplicates(left: pd.DataFrame, right: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    """Outer-merge two wide panels, keeping overlapping columns as *_2, *_3, ..."""
    left_cols = set(left.columns)
    rename_map: dict[str, str] = {}
    for col in right.columns:
        if col in on:
            continue
        if col not in left_cols:
            left_cols.add(col)
            continue
        base = col
        i = 2
        while f"{base}_{i}" in left_cols:
            i += 1
        new_col = f"{base}_{i}"
        rename_map[col] = new_col
        left_cols.add(new_col)

    if rename_map:
        right = right.rename(columns=rename_map)
    return left.merge(right, on=on, how="outer")


def extract_bank_name(filepath: Path) -> str:
    """Extract bank name from filename with standardization."""
    name = filepath.stem.lower()
    
    # Remove common suffixes
    for suffix in ['_financial', '_income', '_balance', 'financial', 'income', 'balance', 'sheet', 'statement']:
        name = name.replace(suffix, '')
    name = name.replace('_', '').replace(' ', '').strip()
    
    # Standardize bank names to consistent format
    bank_mapping = {
        'bankofamerica': 'Bank of America',
        'jpmorgan': 'JPMorgan Chase',
        'wellsfargo': 'Wells Fargo',
        'citigroup': 'Citigroup',
        'goldmansachs': 'Goldman Sachs',
        'morganstanley': 'Morgan Stanley',
        'usbancorp': 'U.S. Bancorp',
        'unsbancorp': 'U.S. Bancorp',  # Fix typo in filename
        'pnc': 'PNC Financial',
        'keycorp': 'KeyCorp',
        'regions': 'Regions Financial',
        'regionsfinancial': 'Regions Financial',
        'fifththird': 'Fifth Third Bancorp',
    }
    
    return bank_mapping.get(name, name.title())


def read_refinitiv_file(filepath: Path, sheet_name: str, statement_type: str) -> pd.DataFrame:
    """Read one Refinitiv Excel file and convert to long format."""
    # Read Excel - Refinitiv always has header at row 17
    df = pd.read_excel(filepath, sheet_name=sheet_name, header=17)
    
    # Extract bank name from filename
    bank_name = extract_bank_name(filepath)
    
    # Find date columns (everything except 'Field Name')
    date_cols = [col for col in df.columns if col != 'Field Name']
    
    # Convert to long format
    df_long = df.melt(
        id_vars='Field Name',
        value_vars=date_cols,
        var_name='period_end_date',
        value_name='value'
    )
    
    # Parse dates
    df_long['period_end_date'] = pd.to_datetime(df_long['period_end_date'], dayfirst=True, errors='coerce')
    
    # Add metadata
    df_long['bank'] = bank_name
    df_long['statement_type'] = statement_type
    
    # Rename and select columns
    df_long = df_long.rename(columns={'Field Name': 'field_name'})
    df_long = df_long[['bank', 'statement_type', 'field_name', 'period_end_date', 'value']]
    
    # Remove rows with missing data
    df_long = df_long.dropna(subset=['period_end_date', 'field_name'])
    
    return df_long


def process_folder(folder_path: Path, sheet_name: str, statement_type: str) -> list:
    """Process all Excel files in a folder."""
    all_data = []
    
    if not folder_path.exists():
        print(f"Folder not found: {folder_path}")
        return all_data
    
    # Find all Excel files (exclude temp files)
    excel_files = sorted([f for f in folder_path.glob('*.xlsx') if not f.name.startswith('~$')])
    
    if not excel_files:
        print(f"No Excel files in: {folder_path}")
        return all_data
    
    print(f"\n📂 Processing {folder_path.name}/ ({len(excel_files)} files)")
    
    for filepath in excel_files:
        try:
            print(f"   → {filepath.name}")
            df = read_refinitiv_file(filepath, sheet_name, statement_type)
            bank = df['bank'].iloc[0]
            all_data.append(df)
            print(f"      ✓ {bank}: {len(df):,} rows")
        except Exception as e:
            print(f"      ✗ Error: {e}")
    
    return all_data


def create_quarterly_datasets(df: pd.DataFrame, output_dir: Path):
    """Convert long-format data to quarterly balanced and unbalanced datasets."""
    
    print(f"\n{'=' * 70}")
    print("📊 Creating Quarterly Datasets (NO PREFIX VERSION)")
    print(f"{'=' * 70}")
    
    # Extract quarter from date
    df['quarter'] = df['period_end_date'].dt.to_period('Q')
    
    # Get unique quarters and banks
    all_quarters = sorted(df['quarter'].unique())
    all_banks = sorted(df['bank'].unique())
    
    print(f"\n📅 Total quarters in data: {len(all_quarters)}")
    print(f"🏦 Total banks: {len(all_banks)}")
    
    # ========================================================================
    # 1. CREATE UNBALANCED DATASET (NO PREFIX!)
    # ========================================================================
    
    print(f"\n✅ Creating UNBALANCED dataset (no prefix)...")
    
    unbalanced_dfs = []
    
    for statement_type in df['statement_type'].unique():
        stmt_data = df[df['statement_type'] == statement_type].copy()
        
        # Pivot: rows = (bank, quarter), columns = field_name
        pivot = stmt_data.pivot_table(
            index=['bank', 'quarter'],
            columns='field_name',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        # Clean Refinitiv field names to snake_case (NO PREFIX)
        cleaned = [clean_column_name(c) for c in list(pivot.columns[2:])]
        cleaned = dedupe_names(cleaned)
        pivot.columns = ['bank', 'quarter'] + cleaned
        
        unbalanced_dfs.append(pivot)
    
    # Merge all statement types
    unbalanced = unbalanced_dfs[0]
    for df_stmt in unbalanced_dfs[1:]:
        unbalanced = merge_keep_duplicates(unbalanced, df_stmt, on=['bank', 'quarter'])

    # Add canonical IDs/dates expected by downstream scripts
    if pd.api.types.is_period_dtype(unbalanced["quarter"]):
        quarter_period = unbalanced["quarter"]
    else:
        quarter_period = pd.PeriodIndex(unbalanced["quarter"].astype(str), freq="Q")
    unbalanced["quarter"] = quarter_period.astype(str)
    unbalanced["period_end_date"] = quarter_period.to_timestamp(how="end").normalize()
    unbalanced["bank_id"] = unbalanced["bank"].map(make_bank_id)

    # Order identifier columns first
    id_cols = ["bank", "bank_id", "quarter", "period_end_date"]
    other_cols = [c for c in unbalanced.columns if c not in id_cols]
    unbalanced = unbalanced[id_cols + other_cols]
    
    # Save unbalanced
    output_dir.mkdir(parents=True, exist_ok=True)
    output_unbalanced = output_dir / 'merged_quarterly_unbalanced.csv'
    unbalanced.to_csv(output_unbalanced, index=False)
    
    print(f"   ✓ Saved: {output_unbalanced}")
    print(f"   • Shape: {unbalanced.shape}")
    print(f"   • Quarters: {unbalanced['quarter'].nunique()}")
    
    # ========================================================================
    # 2. CREATE BALANCED DATASET (NO PREFIX!)
    # ========================================================================
    
    print(f"\n✅ Creating BALANCED dataset (no prefix)...")
    
    # Count banks per quarter
    banks_per_quarter = df.groupby('quarter')['bank'].nunique().reset_index()
    banks_per_quarter.columns = ['quarter', 'bank_count']
    
    # Find quarters where ALL banks have data
    n_banks = df['bank'].nunique()
    complete_quarters = banks_per_quarter[
        banks_per_quarter['bank_count'] == n_banks
    ]['quarter'].tolist()
    
    print(f"   • Complete quarters (all {n_banks} banks): {len(complete_quarters)}")
    
    if len(complete_quarters) == 0:
        print("   ⚠ No quarters with all banks present!")
        min_banks = int(n_banks * 0.8)
        complete_quarters = banks_per_quarter[
            banks_per_quarter['bank_count'] >= min_banks
        ]['quarter'].tolist()
        print(f"   • Using quarters with ≥{min_banks} banks: {len(complete_quarters)}")
    
    # Filter to complete quarters
    df_balanced = df[df['quarter'].isin(complete_quarters)].copy()
    
    # Convert to wide format
    balanced_dfs = []
    
    for statement_type in df_balanced['statement_type'].unique():
        stmt_data = df_balanced[df_balanced['statement_type'] == statement_type].copy()
        
        # Pivot
        pivot = stmt_data.pivot_table(
            index=['bank', 'quarter'],
            columns='field_name',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        # Clean Refinitiv field names to snake_case (NO PREFIX)
        cleaned = [clean_column_name(c) for c in list(pivot.columns[2:])]
        cleaned = dedupe_names(cleaned)
        pivot.columns = ['bank', 'quarter'] + cleaned
        
        balanced_dfs.append(pivot)
    
    # Merge all statement types
    balanced = balanced_dfs[0]
    for df_stmt in balanced_dfs[1:]:
        balanced = merge_keep_duplicates(balanced, df_stmt, on=['bank', 'quarter'])

    # Add canonical IDs/dates expected by downstream scripts
    if pd.api.types.is_period_dtype(balanced["quarter"]):
        quarter_period = balanced["quarter"]
    else:
        quarter_period = pd.PeriodIndex(balanced["quarter"].astype(str), freq="Q")
    balanced["quarter"] = quarter_period.astype(str)
    balanced["period_end_date"] = quarter_period.to_timestamp(how="end").normalize()
    balanced["bank_id"] = balanced["bank"].map(make_bank_id)

    # Order identifier columns first
    id_cols = ["bank", "bank_id", "quarter", "period_end_date"]
    other_cols = [c for c in balanced.columns if c not in id_cols]
    balanced = balanced[id_cols + other_cols]
    
    # Save balanced
    output_balanced = output_dir / 'merged_quarterly_balanced.csv'
    balanced.to_csv(output_balanced, index=False)
    
    print(f"   ✓ Saved: {output_balanced}")
    print(f"   • Shape: {balanced.shape}")
    print(f"   • Quarters: {balanced['quarter'].nunique()}")
    print(f"   • Date range: {balanced['quarter'].min()} to {balanced['quarter'].max()}")
    
    # Show sample of column names (to verify no prefix)
    sample_cols = [c for c in balanced.columns if 'asset' in c.lower() or 'equity' in c.lower()][:5]
    print(f"\n   Sample columns (verify no prefix): {sample_cols}")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("✅ SUCCESS! (Column names have NO prefix)")
    print(f"{'=' * 70}")
    print(f"\nOutput files in {output_dir}:")
    print(f"  • merged_quarterly_unbalanced.csv: {unbalanced.shape[0]} rows × {unbalanced.shape[1]} cols")
    print(f"  • merged_quarterly_balanced.csv: {balanced.shape[0]} rows × {balanced.shape[1]} cols")
    print(f"{'=' * 70}")


def main():
    """Main function to process all Refinitiv files."""
    
    print("=" * 70)
    print("Refinitiv Excel to Quarterly CSVs (FIXED - NO PREFIX)")
    print("=" * 70)
    
    # CONFIGURE YOUR FOLDER STRUCTURE HERE
    base_path = Path('data/raw')
    output_dir = Path('data/processed')
    
    folders_to_process = [
        {
            'folder': base_path / 'Balansesheet',
            'sheet': 'Balance Sheet',
            'type': 'balance'
        },
        {
            'folder': base_path / 'Income_statement',
            'sheet': 'Income Statement',
            'type': 'income'
        },
        {
            'folder': base_path / 'Financial',
            'sheet': 'Financial Summary',
            'type': 'financial_summary'
        },
    ]
    
    # Process all folders and combine
    all_data = []
    
    for folder_info in folders_to_process:
        data = process_folder(
            folder_path=folder_info['folder'],
            sheet_name=folder_info['sheet'],
            statement_type=folder_info['type']
        )
        all_data.extend(data)
    
    if not all_data:
        print("\n⚠ No data processed. Check your folder paths!")
        return
    
    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(['bank', 'period_end_date', 'statement_type', 'field_name'])
    combined = combined.reset_index(drop=True)
    
    print(f"\n{'=' * 70}")
    print(f"📊 Combined {len(combined):,} rows from {combined['bank'].nunique()} banks")
    print(f"{'=' * 70}")
    
    # Create quarterly datasets
    create_quarterly_datasets(combined, output_dir)


if __name__ == '__main__':
    main()
