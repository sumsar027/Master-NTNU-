"""
Panel regressions: Leverage, Assets, Equity on VaR measures
Replicates Adrian & Shin (2014) style analysis

Models:
  1. log(Leverage) ~ log(UnitVaR) + bank FE
  2. log(Assets) ~ log(VaR) + bank FE  
  3. log(Equity) ~ log(VaR) + bank FE
"""

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, PooledOLS
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================

# Harmonize bank names across files
BANK_MAP = {
    "bankofamerica": "bank_of_america",
    "citigroup": "citibank", 
    "wellsfargo": "wells_fargo"
}

TIME_FE = False  # Set True for robustness checks

# ============================================================================
# LOAD & CLEAN DATA
# ============================================================================

def load_data():
    """Load and merge balance sheet + VaR data"""
    
    # Load files
    base = pd.read_csv('output/merged_quarterly_balanced.csv')
    var = pd.read_csv('output/merged_with_var_95_approx.csv')
    
    # Normalize column names
    for df in [base, var]:
        if 'bank' in df.columns and 'bank_id' not in df.columns:
            df.rename(columns={'bank': 'bank_id'}, inplace=True)
    
    # Handle alternative column names in base data
    if 'total_assets_2' in base.columns:
        base['total_assets'] = base['total_assets'].fillna(base['total_assets_2'])
    
    # Harmonize bank IDs
    base['bank_id'] = base['bank_id'].str.strip().replace(BANK_MAP)
    var['bank_id'] = var['bank_id'].str.strip().replace(BANK_MAP)
    
    # Ensure datetime
    base['period_end_date'] = pd.to_datetime(base['period_end_date'])
    var['period_end_date'] = pd.to_datetime(var['period_end_date'])
    
    # Create VaR level (prefer var_95, fallback to var_95_approx)
    if 'var_95' in var.columns:
        var['var_95_level'] = var['var_95'].fillna(var.get('var_95_approx', np.nan))
    else:
        var['var_95_level'] = var['var_95_approx']
    
    # Merge
    df = base.merge(
        var[['bank_id', 'period_end_date', 'var_95_level']], 
        on=['bank_id', 'period_end_date'],
        how='inner'
    )
    
    print(f"Merged data: {len(df)} observations, {df['bank_id'].nunique()} banks")
    
    return df


# ============================================================================
# CONSTRUCT VARIABLES
# ============================================================================

def prepare_variables(df):
    """Create all regression variables"""
    
    df = df.copy()
    
    # Balance sheet ratios
    df['leverage'] = df['total_assets'] / df['common_equity_total']
    df['unit_var'] = df['var_95_level'] / df['total_assets']
    
    # Log transforms (auto-handles negatives/zeros as NaN)
    for var in ['leverage', 'total_assets', 'common_equity_total', 'var_95_level', 'unit_var']:
        df[f'log_{var}'] = np.log(df[var].where(df[var] > 0))
    
    # Use common estimation sample across all models
    required_vars = [
        'bank_id', 'period_end_date',
        'log_leverage', 'log_total_assets', 'log_common_equity_total',
        'log_unit_var', 'log_var_95_level'
    ]
    
    n_before = len(df)
    df = df[required_vars].dropna()
    
    print(f"Estimation sample: {len(df)} obs ({n_before - len(df)} dropped due to missing values)")
    
    return df


# ============================================================================
# ESTIMATION
# ============================================================================

def estimate_model(df, dep_var, indep_var, model_name, estimator='FE'):
    """Run single regression"""
    
    # Prepare panel data
    panel = df.set_index(['bank_id', 'period_end_date'])
    y = panel[[dep_var]]
    X = panel[[indep_var]]
    
    # Estimate
    if estimator == 'FE':
        model = PanelOLS(y, X, entity_effects=True, time_effects=TIME_FE)
        res = model.fit(cov_type='clustered', cluster_entity=True)
        r2 = res.rsquared_within
        const = res.estimated_effects.mean().iloc[0] if hasattr(res, 'estimated_effects') else np.nan
        
    else:  # Pooled
        X_pooled = X.copy()
        X_pooled.insert(0, 'const', 1.0)
        model = PooledOLS(y, X_pooled)
        res = model.fit(cov_type='clustered', cluster_entity=True)
        r2 = res.rsquared
        const = res.params['const']
    
    # Extract results
    return {
        'estimator': estimator,
        'model': model_name,
        'dependent_var': dep_var,
        'independent_var': indep_var,
        'const': const,
        'beta': res.params[indep_var],
        'se': res.std_errors[indep_var],
        't_stat': res.tstats[indep_var],
        'p_value': res.pvalues[indep_var],
        'r_squared': r2,
        'n_obs': int(res.nobs),
        'n_banks': panel.index.get_level_values(0).nunique(),
        'n_periods': panel.index.get_level_values(1).nunique()
    }


def run_all_models(df):
    """Estimate all three specifications"""
    
    specs = [
        ('Model 1: Leverage', 'log_leverage', 'log_unit_var'),
        ('Model 2: Assets', 'log_total_assets', 'log_var_95_level'),
        ('Model 3: Equity', 'log_common_equity_total', 'log_var_95_level')
    ]
    
    results = []
    
    # Fixed effects (main results)
    for name, dep, indep in specs:
        results.append(estimate_model(df, dep, indep, name, 'FE'))
    
    # Pooled (benchmark)
    for name, dep, indep in specs:
        results.append(estimate_model(df, dep, indep, name, 'Pooled'))
    
    return pd.DataFrame(results)


# ============================================================================
# MAIN
# ============================================================================

def main():
    
    # Load and prepare data
    df = load_data()
    df = prepare_variables(df)
    
    # Run regressions
    results = run_all_models(df)
    
    # Display results
    print("\n" + "="*80)
    print("REGRESSION RESULTS")
    print("="*80 + "\n")
    print(results.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    # Save
    Path('output').mkdir(exist_ok=True)
    results.to_csv('output/Model_1_result.csv', index=False)
    print("\nResults saved to output/Model_1_result.csv")


if __name__ == "__main__":
    main()