"""
Panel regressions: Leverage, Assets, Equity on VaR measures
Replicates Adrian & Shin (2014) style analysis

Models:
  1. log(Leverage) ~ log(UnitVaR) + bank FE
  2. log(Assets) ~ log(VaR) + bank FE
  3. log(Equity) ~ log(VaR) + bank FE
  4A. log(Leverage) ~ log(UnitVaR) + LCR_{t-1} + log(UnitVaR)*LCR_{t-1} + bank FE
"""

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from pathlib import Path

from sklearn import base

# ============================================================================
# CONFIG
# ============================================================================

BANK_MAP = {
    "bankofamerica": "bank_of_america",
    "citigroup": "citibank",
    "wellsfargo": "wells_fargo"
}

# ============================================================================
# LOAD & CLEAN DATA
# ============================================================================

def load_data():
    """Load and merge balance sheet + VaR data"""
    
    base = pd.read_csv("data/processed/merged_quarterly_balanced.csv")
    var  = pd.read_csv("output/data/merged_with_var_99_dual_methods.csv")
    
    base["bank_id"] = base["bank"].str.strip().str.lower().str.replace(" ", "_")
    base["period_end_date"] = pd.PeriodIndex(base["quarter"], freq="Q").to_timestamp(how="end").normalize()

    var["period_end_date"] = pd.to_datetime(var["period_end_date"])
    
    print(f"Balance sheet: {len(base)} rows")
    print(f"VaR data: {len(var)} rows")
    
    # Normalize column names
    if 'bank' in base.columns and 'bank_id' not in base.columns:
        base.rename(columns={'bank': 'bank_id'}, inplace=True)
    if 'bank' in var.columns and 'bank_id' not in var.columns:
        var.rename(columns={'bank': 'bank_id'}, inplace=True)
    
    # Harmonize bank IDs using BANK_MAP
    base['bank_id'] = base['bank_id'].str.strip().str.lower().replace(BANK_MAP)
    var['bank_id']  = var['bank_id'].str.strip().str.lower().replace(BANK_MAP)
    
    print(f"\nBalance sheet banks: {sorted(base['bank_id'].unique())}")
    print(f"VaR banks: {sorted(var['bank_id'].unique())}")
    
    # Ensure datetime
    base['period_end_date'] = pd.to_datetime(base['period_end_date'])
    var['period_end_date']  = pd.to_datetime(var['period_end_date'])
    
    # Use var_99_level (reported 99% VaR or Gaussian-converted)
    if 'var_99_level' not in var.columns:
        var['var_99_level'] = var['var_99'].fillna(var['var_99_gaussian'])
    
    # Merge
    df = base.merge(
        var[['bank_id', 'period_end_date', 'var_99_level']],
        on=['bank_id', 'period_end_date'],
        how='inner'
    )
    
    print(f"\nMerged data: {len(df)} observations, {df['bank_id'].nunique()} banks")
    print(f"Banks in final sample: {sorted(df['bank_id'].unique())}")
    
    return df

# ============================================================================
# CONSTRUCT VARIABLES
# ============================================================================

def prepare_variables(df):
    """Create all regression variables"""
    
    df = df.copy()
    
    # Use total_assets (or total_assets_2 as fallback)
    if 'total_assets_2' in df.columns:
        df['total_assets'] = df['total_assets'].fillna(df['total_assets_2'])
    
    # Core variables
    df['leverage']  = df['total_assets'] / df['common_equity_total']
    df['unit_var']  = df['var_99_level'] / df['total_assets']
    
    # Log transformations
    for var in ['leverage', 'total_assets', 'common_equity_total', 
                'var_99_level', 'unit_var']:
        df[f'log_{var}'] = np.log(df[var].where(df[var] > 0))
    
    # LCR variables (for Model 4A only)
    if 'liquidity_coverage_ratio_basel_3' in df.columns:
        df['lcr_ratio'] = pd.to_numeric(
            df['liquidity_coverage_ratio_basel_3'], errors='coerce'
        ) / 100.0
        df = df.sort_values(['bank_id', 'period_end_date'])
        df['lcr_lag1'] = df.groupby('bank_id')['lcr_ratio'].shift(1)
        df['int_unitvar_lcr'] = df['log_unit_var'] * df['lcr_lag1']
    
    print(f"\nEstimation sample prepared: {len(df)} observations")
    
    return df

# ============================================================================
# ESTIMATION
# ============================================================================

def estimate_model(df, dep_var, indep_vars, model_name):
    """Run panel regression with bank fixed effects"""
    
    # Handle single variable as string
    if isinstance(indep_vars, str):
        indep_vars = [indep_vars]
    
    # Drop missing values for this model
    df_clean = df.dropna(subset=[dep_var] + indep_vars)
    
    print(f"\n{'='*70}")
    print(f"{model_name}")
    print(f"{'='*70}")
    print(f"Sample: {len(df_clean)} observations, "
          f"{df_clean['bank_id'].nunique()} banks, "
          f"{df_clean['period_end_date'].nunique()} periods")
    
    # Set up panel structure
    panel = df_clean.set_index(['bank_id', 'period_end_date'])
    y = panel[[dep_var]]
    X = panel[indep_vars]
    
    # Estimate with bank fixed effects and clustered standard errors
    model = PanelOLS(y, X, entity_effects=True)
    res = model.fit(cov_type='clustered', cluster_entity=True)
    
    print(res.summary)
    
    return {
        'model_name': model_name,
        'dep_var': dep_var,
        'result': res,
        'n_obs': int(res.nobs),
        'n_banks': panel.index.get_level_values(0).nunique(),
        'r_squared': res.rsquared_within
    }

# ============================================================================
# SAVE RESULTS - CLEANER FORMAT
# ============================================================================
def save_results(results_list, output_path='output/tables/model_results_model_one.csv'):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    model_rows = []
    for res_dict in results_list:
        res = res_dict['result']

        row = {
            'Model': res_dict['model_name'],
            'N': int(res_dict['n_obs']),
            'N_banks': int(res_dict['n_banks']),
            'R_squared_within': float(res_dict['r_squared']),
        }

        for var in res.params.index:
            row[f"b_{var}"]  = float(res.params[var])
            row[f"se_{var}"] = float(res.std_errors[var])
            row[f"t_{var}"]  = float(res.tstats[var])
            row[f"p_{var}"]  = float(res.pvalues[var])

        model_rows.append(row)

    df_model = pd.DataFrame(model_rows)

    # Reorder columns: first fixed, then sorted others
    first_cols = ['Model', 'N', 'N_banks', 'R_squared_within']
    other_cols = [c for c in df_model.columns if c not in first_cols]
    df_model = df_model[first_cols + sorted(other_cols)]

    # Save to CSV (replace NaN with empty string for clarity)
    df_model.replace({np.nan: ''}).to_csv(output_path, index=False)

    print(f"\n✓ Results saved to: {output_path}\n")

    return df_model


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all models and save results"""
    
    # Load and prepare data
    df = load_data()
    df = prepare_variables(df)
    
    # Store all results
    all_results = []
    
    # MODEL 1: Leverage on Unit VaR
    res1 = estimate_model(
        df, 
        dep_var='log_leverage', 
        indep_vars='log_unit_var',
        model_name='Model_1_Leverage'
    )
    all_results.append(res1)
    
    # MODEL 2: Assets on VaR
    res2 = estimate_model(
        df,
        dep_var='log_total_assets',
        indep_vars='log_var_99_level',
        model_name='Model_2_Assets'
    )
    all_results.append(res2)
    
    # MODEL 3: Equity on VaR
    res3 = estimate_model(
        df,
        dep_var='log_common_equity_total',
        indep_vars='log_var_99_level',
        model_name='Model_3_Equity'
    )
    all_results.append(res3)
    
    # MODEL 4A: Leverage with LCR Interaction
    res4 = estimate_model(
        df,
        dep_var='log_leverage',
        indep_vars=['log_unit_var', 'lcr_lag1', 'int_unitvar_lcr'],
        model_name='Model_4A_LCR'
    )
    all_results.append(res4)
    
    # SAVE RESULTS
    save_results(all_results)
    
    print(f"\n{'='*70}")
    print("ALL MODELS COMPLETED ✓")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()