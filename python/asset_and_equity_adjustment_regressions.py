"""
Asset Growth vs Leverage Growth Regressions
Tests if banks grow assets when they increase leverage (procyclical behavior)

Model: Asset Growth = β₀ + β₁·Leverage Growth + ε

We run three versions:
1. Pooled OLS (simple regression)
2. Bank FE (controls for bank-specific differences)
3. Time FE (controls for quarter-specific shocks like financial crisis)
4. Bank + Time FE (controls for both bank differences and time shocks)

Positive β₁ = when leverage increases, assets also increase (procyclical)
"""


import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================

# Input file (merged quarterly balance sheet data)
INPUT_FILE = Path("data/processed/merged_quarterly_balanced.csv")

# Output files
OUTPUT_DATASET = Path("output/data/analysis_dataset.csv")
OUTPUT_RESULTS = Path("output/tables/book_side_regression_results.csv")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_column(df, candidates):
    """Find the first column name that exists in the dataframe"""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def calculate_growth(df, entity_col, value_col, output_col):
    """
    Calculate quarterly growth rate as log difference (in percent)
    
    Growth = 100 * (log(X_t) - log(X_t-1))
    
    This is approximately equal to percentage change.
    Only calculated when both current and previous values are positive.
    """
    # Convert to numeric (handles any text values)
    x = pd.to_numeric(df[value_col], errors='coerce')
    
    # Get previous quarter's value for each bank
    x_lag = x.groupby(df[entity_col], sort=False).shift(1)
    
    # Only calculate when both current and previous values are positive
    valid = x.notna() & x_lag.notna() & (x > 0) & (x_lag > 0)
    
    # Calculate log difference and convert to percent
    df[output_col] = np.where(valid, 100.0 * (np.log(x) - np.log(x_lag)), np.nan)
    
    return df


def add_significance_stars(p_value):
    """Add significance stars based on p-value"""
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.01:
        return "***"  # 1% significance
    if p_value < 0.05:
        return "**"   # 5% significance
    if p_value < 0.10:
        return "*"    # 10% significance
    return ""


# ============================================================================
# LOAD & PREPARE DATA
# ============================================================================

def load_data():
    """Load balance sheet data and identify bank and date columns"""
    
    if not INPUT_FILE.exists():
        print(f"ERROR: Missing file {INPUT_FILE}")
        exit(1)
    
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {INPUT_FILE.name}: {len(df)} rows, {df.shape[1]} columns")
    
    # Find the bank identifier column (different datasets use different names)
    entity_col = find_column(df, ['bank', 'bank_id', 'ticker', 'entity'])
    if entity_col is None:
        print("ERROR: Could not find bank identifier column")
        exit(1)
    
    # Find the date column
    time_col = find_column(df, ['period_end_date', 'date', 'quarter', 'time'])
    if time_col is None:
        print("ERROR: Could not find date column")
        exit(1)
    
    print(f"Using: bank identifier = '{entity_col}', date = '{time_col}'")
    
    # Clean up bank names and convert dates
    df[entity_col] = df[entity_col].astype(str).str.strip()
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Sort by bank and date (important for calculating growth rates)
    df = df.sort_values([entity_col, time_col]).reset_index(drop=True)
    
    return df, entity_col, time_col


def prepare_variables(df, entity_col):
    """
    Create leverage and growth variables
    
    Book Leverage = Total Assets / Equity
    (how many dollars of assets per dollar of equity)
    """
    
    # Find the columns we need (handle different naming conventions)
    assets_col = find_column(df, [ 'total_assets', 'total_assets_2','assets'])
    equity_col = find_column(df, ['common_equity_total', 'total_shareholders_equity', 'total_equity'])
    
    if assets_col is None:
        print("ERROR: Could not find assets column")
        exit(1)
    if equity_col is None:
        print("ERROR: Could not find equity column")
        exit(1)
    
    print(f"Using: assets = '{assets_col}', equity = '{equity_col}'")
    
    # Convert to numeric
    df['assets'] = pd.to_numeric(df[assets_col], errors='coerce')
    df['equity'] = pd.to_numeric(df[equity_col], errors='coerce')
    
    # Calculate book leverage (only when both assets and equity are positive)
    valid = df['assets'].notna() & df['equity'].notna() & (df['assets'] > 0) & (df['equity'] > 0)
    df['book_leverage'] = np.where(valid, df['assets'] / df['equity'], np.nan)
    
    # Calculate growth rates (quarter-to-quarter % change)
    df = calculate_growth(df, entity_col, 'assets', 'asset_growth')
    df = calculate_growth(df, entity_col, 'book_leverage', 'leverage_growth')
    
    # Report how many observations we have
    n_valid = df[['asset_growth', 'leverage_growth']].notna().all(axis=1).sum()
    print(f"Valid growth observations: {n_valid} (first quarter per bank is dropped)")
    
    return df


# ============================================================================
# RUN REGRESSIONS
# ============================================================================

def run_pooled_ols(df, entity_col):
    """
    Model 1: Pooled OLS
    Simple regression without any fixed effects
    Standard errors clustered by bank
    """
    # Keep only complete observations
    reg_data = df[[entity_col, 'asset_growth', 'leverage_growth']].dropna()
    
    if reg_data.empty:
        raise ValueError("No valid observations for regression")
    
    # Prepare variables
    y = reg_data['asset_growth'].values
    X = sm.add_constant(reg_data[['leverage_growth']].values)
    
    # Run regression with clustered standard errors
    model = sm.OLS(y, X)
    result = model.fit(cov_type='cluster', cov_kwds={'groups': reg_data[entity_col]})
    
    return {
        'coef': result.params[1],
        'se': result.bse[1],
        'pvalue': result.pvalues[1],
        'r2': result.rsquared_adj,
        'nobs': int(result.nobs)
    }


def run_panel_fe(df, entity_col, time_col, bank_fe=False, time_fe=False):
    """
    Fixed effects models
    
    bank_fe=True: controls for time-invariant bank characteristics
    time_fe=True: controls for quarter-specific shocks (e.g., financial crisis)
    
    Standard errors clustered by bank
    """
    # Keep only complete observations
    reg_data = df[[entity_col, time_col, 'asset_growth', 'leverage_growth']].dropna()
    
    if reg_data.empty:
        raise ValueError("No valid observations for regression")
    
    # Set up panel structure (bank × quarter)
    reg_data = reg_data.set_index([entity_col, time_col])
    
    # Run fixed effects regression
    model = PanelOLS(
        reg_data['asset_growth'],
        reg_data[['leverage_growth']],
        entity_effects=bank_fe,
        time_effects=time_fe
    )
    result = model.fit(cov_type='clustered', cluster_entity=True)
    
    # For FE models, we prefer "within R²" which shows fit after removing fixed effects
    r2 = getattr(result, 'rsquared_within', result.rsquared_inclusive)
    
    return {
        'coef': result.params['leverage_growth'],
        'se': result.std_errors['leverage_growth'],
        'pvalue': result.pvalues['leverage_growth'],
        'r2': r2,
        'nobs': int(result.nobs)
    }


def run_all_regressions(df, entity_col, time_col):
    """Run all three model specifications"""
    
    results = {}
    
    # Model 1: Pooled OLS (no fixed effects)
    print("\nRunning Model 1: Pooled OLS...")
    results[1] = run_pooled_ols(df, entity_col)
    
     # Model 2: Bank FE (controls for quarter shocks)
    print("\nRunning Model 2: Bank FE...")
    results[2] = run_panel_fe(df, entity_col, time_col, bank_fe=True, time_fe=False)
    
    # Model 3: Time FE (controls for quarter shocks)
    print("Running Model 3: Time FE...")
    results[3] = run_panel_fe(df, entity_col, time_col, bank_fe=False, time_fe=True)
    
    # Model 4: Bank + Time FE (controls for both)
    print("Running Model 4: Bank + Time FE...")
    results[4] = run_panel_fe(df, entity_col, time_col, bank_fe=True, time_fe=True)
    
    return results


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(results):
    """Save regression results to CSV"""
    
    rows = []
    spec_names = {
        1: "Pooled OLS",
        2: "Bank FE",
        3: "Time FE",
        4: "Bank + Time FE"
    }
    
    for model_num, res in results.items():
        # Calculate t-statistic
        t_stat = res['coef'] / res['se'] if res['se'] != 0 else np.nan
        
        rows.append({
            'model': model_num,
            'specification': spec_names[model_num],
            'dependent_variable': 'asset_growth',
            'independent_variable': 'leverage_growth',
            'coef': res['coef'],
            'std_error': res['se'],
            't_stat': t_stat,
            'p_value': res['pvalue'],
            'r2': res['r2'],
            'n_obs': res['nobs']
        })
    
    results_df = pd.DataFrame(rows)
    
    # Save to CSV
    OUTPUT_RESULTS.parent.mkdir(exist_ok=True)
    results_df.to_csv(OUTPUT_RESULTS, index=False)
    
    return results_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main analysis pipeline:
    1. Load balance sheet data
    2. Calculate leverage and growth rates
    3. Run four regression specifications
    4. Display and save results
    """
    
    print("="*80)
    print("ASSET GROWTH vs LEVERAGE GROWTH ANALYSIS")
    print("="*80)
    
    # Step 1-2: Load data and create variables
    df, entity_col, time_col = load_data()
    df = prepare_variables(df, entity_col)
    
    # Save the constructed dataset (useful for checking)
    OUTPUT_DATASET.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_DATASET, index=False)
    print(f"\nSaved analysis dataset to {OUTPUT_DATASET.name}")
    
    # Step 3: Run regressions
    results = run_all_regressions(df, entity_col, time_col)
    
    # Step 4: Display results
    print("\n" + "="*80)
    print("REGRESSION RESULTS")
    print("="*80)
    print("\nDependent variable: Asset Growth (quarterly % change)")
    print("Independent variable: Leverage Growth (quarterly % change)")
    print("\nModels:")
    print("  (1) Pooled OLS - simple regression")
    print("  (2) Bank FE - controls for bank-specific differences")
    print("  (3) Time FE - controls for quarter-specific shocks")
    print("  (4) Bank + Time FE - controls for bank differences and quarter shocks")
    print("\n" + "-"*80)
    
    # Print results table
    for model_num in [1, 2, 3, 4]:
        res = results[model_num]
        coef = res['coef']
        se = res['se']
        t_stat = coef / se if se != 0 else np.nan
        stars = add_significance_stars(res['pvalue'])
        
        print(f"({model_num}) Coefficient: {coef:7.3f}{stars:3s}  SE: {se:6.3f}  t-stat: {t_stat:6.2f}  R²: {res['r2']:.3f}")
    
    #save results to CSV 
    save_results(results)
    print(f"\nSaved regression results to {OUTPUT_RESULTS.name}")   
 
if __name__ == "__main__":
    main()