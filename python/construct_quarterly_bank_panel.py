"""
Refinitiv Excel Reader - Direct to Quarterly Datasets
Reads all Refinitiv Excel files and creates quarterly balanced/unbalanced CSVs directly.

Output:
    data/processed/merged_quarterly_balanced.csv
    data/processed/merged_quarterly_unbalanced.csv

Usage:
    python refinitiv_to_quarterly.py
"""

import pandas as pd
from pathlib import Path


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
        print(f"⚠️  Folder not found: {folder_path}")
        return all_data
    
    # Find all Excel files (exclude temp files)
    excel_files = sorted([f for f in folder_path.glob('*.xlsx') if not f.name.startswith('~$')])
    
    if not excel_files:
        print(f"⚠️  No Excel files in: {folder_path}")
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
            print(f"      ❌ Error: {e}")
    
    return all_data


def create_quarterly_datasets(df: pd.DataFrame, output_dir: Path):
    """Convert long-format data to quarterly balanced and unbalanced datasets."""
    
    print(f"\n{'=' * 70}")
    print("📊 Creating Quarterly Datasets")
    print(f"{'=' * 70}")
    
    # Extract quarter from date
    df['quarter'] = df['period_end_date'].dt.to_period('Q')
    
    # Get unique quarters and banks
    all_quarters = sorted(df['quarter'].unique())
    all_banks = sorted(df['bank'].unique())
    
    print(f"\n📅 Total quarters in data: {len(all_quarters)}")
    print(f"🏦 Total banks: {len(all_banks)}")
    
    # ========================================================================
    # 1. CREATE UNBALANCED DATASET
    # ========================================================================
    
    print(f"\n🔄 Creating UNBALANCED dataset...")
    
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
        
        # Add statement type prefix to column names
        pivot.columns = ['bank', 'quarter'] + [
            f"{statement_type}_{col}" for col in pivot.columns[2:]
        ]
        
        unbalanced_dfs.append(pivot)
    
    # Merge all statement types
    unbalanced = unbalanced_dfs[0]
    for df_stmt in unbalanced_dfs[1:]:
        unbalanced = unbalanced.merge(df_stmt, on=['bank', 'quarter'], how='outer')
    
    # Convert quarter to string for CSV
    unbalanced['quarter'] = unbalanced['quarter'].astype(str)
    
    # Save unbalanced
    output_dir.mkdir(parents=True, exist_ok=True)
    output_unbalanced = output_dir / 'merged_quarterly_unbalanced.csv'
    unbalanced.to_csv(output_unbalanced, index=False)
    
    print(f"   ✓ Saved: {output_unbalanced}")
    print(f"   • Shape: {unbalanced.shape}")
    print(f"   • Quarters: {unbalanced['quarter'].nunique()}")
    
    # ========================================================================
    # 2. CREATE BALANCED DATASET
    # ========================================================================
    
    print(f"\n🔄 Creating BALANCED dataset...")
    
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
        print("   ⚠️  No quarters with all banks present!")
        min_banks = int(n_banks * 0.8)
        complete_quarters = banks_per_quarter[
            banks_per_quarter['bank_count'] >= min_banks
        ]['quarter'].tolist()
        print(f"   • Quarters with ≥{min_banks} banks: {len(complete_quarters)}")
    
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
        
        # Add statement type prefix
        pivot.columns = ['bank', 'quarter'] + [
            f"{statement_type}_{col}" for col in pivot.columns[2:]
        ]
        
        balanced_dfs.append(pivot)
    
    # Merge all statement types
    balanced = balanced_dfs[0]
    for df_stmt in balanced_dfs[1:]:
        balanced = balanced.merge(df_stmt, on=['bank', 'quarter'], how='outer')
    
    # Convert quarter to string
    balanced['quarter'] = balanced['quarter'].astype(str)
    
    # Save balanced
    output_balanced = output_dir / 'merged_quarterly_balanced.csv'
    balanced.to_csv(output_balanced, index=False)
    
    print(f"   ✓ Saved: {output_balanced}")
    print(f"   • Shape: {balanced.shape}")
    print(f"   • Quarters: {balanced['quarter'].nunique()}")
    print(f"   • Date range: {balanced['quarter'].min()} to {balanced['quarter'].max()}")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("✅ SUCCESS!")
    print(f"{'=' * 70}")
    print(f"\nOutput files in {output_dir}:")
    print(f"  • merged_quarterly_unbalanced.csv: {unbalanced.shape[0]} rows × {unbalanced.shape[1]} cols")
    print(f"  • merged_quarterly_balanced.csv: {balanced.shape[0]} rows × {balanced.shape[1]} cols")
    print(f"{'=' * 70}")


def main():
    """Main function to process all Refinitiv files."""
    
    print("=" * 70)
    print("📊 Refinitiv Excel to Quarterly CSVs")
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
        print("\n❌ No data processed. Check your folder paths!")
        return
    
    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(['bank', 'period_end_date', 'statement_type', 'field_name'])
    combined = combined.reset_index(drop=True)
    
    print(f"\n{'=' * 70}")
    print(f"✅ Combined {len(combined):,} rows from {combined['bank'].nunique()} banks")
    print(f"{'=' * 70}")
    
    # Create quarterly datasets
    create_quarterly_datasets(combined, output_dir)


if __name__ == '__main__':
    main()