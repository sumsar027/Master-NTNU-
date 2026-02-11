"""
Samlet fil for alle panel-regresjoner
Kjører alle modeller og lagrer resultater i én CSV-fil

Bruker var_99_gaussian i stedet for var_99_boa_factor

Modeller:
ASSET GROWTH MODELS:
  A1. Asset Growth ~ Leverage Growth (Pooled OLS)
  A2. Asset Growth ~ Leverage Growth (Bank FE)
  A3. Asset Growth ~ Leverage Growth (Time FE)
  A4. Asset Growth ~ Leverage Growth (Bank + Time FE)

LEVERAGE/RISK MODELS:
  1. log(Leverage) ~ log(UnitVaR) + bank FE
  2. log(Assets) ~ log(VaR) + bank FE
  3. log(Equity) ~ log(VaR) + bank FE
  4A. log(Leverage) ~ log(UnitVaR) + LCR_{t-1} + log(UnitVaR)*LCR_{t-1} + bank FE
  4B. log(Leverage) ~ log(UnitVaR) + CET1_{t-1} + log(UnitVaR)*CET1_{t-1} + bank FE
  4C. log(Leverage) ~ log(UnitVaR) + LCR_{t-1} + CET1_{t-1}
                      + log(UnitVaR)*LCR_{t-1} + log(UnitVaR)*CET1_{t-1} + bank FE
  5. log(Leverage) ~ LCR_{t-1} + bank FE

COVID MODEL:
  C1. log(Leverage) ~ log(UnitVaR) + COVID + log(UnitVaR)*COVID + bank FE
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================

# CET1 column name
CET1_COLUMN = "capital_adequacy_core_tier_1"

# COVID dummy window (inclusive)
COVID_START = "2019-03-31"
COVID_END = "2021-12-31"

# Output file
OUTPUT_FILE = Path("output/tables/all_regression_results.csv")

# ============================================================================
# DATA LOADING
# ============================================================================

def load_balance_sheet_data():
    """Load balance sheet data"""
    df = pd.read_csv("data/processed/balance_sheet_panel_balanced.csv")
    
    # Standardize bank_id
    df['bank_id'] = df['bank_id'].str.strip().str.lower()
    
    # Ensure datetime
    df['period_end_date'] = pd.to_datetime(df['period_end_date'])
    
    print(f"Balance sheet: {len(df)} rows, {df['bank_id'].nunique()} banks")
    print(f"Banks: {sorted(df['bank_id'].unique())}")
    
    return df


def load_var_data():
    """Load VaR data - BRUKER var_99_gaussian"""
    df = pd.read_csv("output/data/var_99.csv")
    
    # Rename columns
    df.rename(columns={'bank': 'bank_id', 'date': 'period_end_date'}, inplace=True)
    
    # Standardize bank_id
    df['bank_id'] = df['bank_id'].str.strip().str.lower()
    
    # Ensure datetime
    df['period_end_date'] = pd.to_datetime(df['period_end_date'])
    
    # Use GAUSSIAN VaR (not BoA factor)
    df['var_99_level'] = df['var_99_gaussian']
    
    print(f"\nVaR data: {len(df)} rows, {df['bank_id'].nunique()} banks")
    print(f"Banks: {sorted(df['bank_id'].unique())}")
    
    return df[['bank_id', 'period_end_date', 'var_99_level']]


def merge_data():
    """Merge balance sheet and VaR data"""
    base = load_balance_sheet_data()
    var = load_var_data()
    
    df = base.merge(var, on=['bank_id', 'period_end_date'], how='inner')
    
    print(f"\nMerged data: {len(df)} observations, {df['bank_id'].nunique()} banks")
    print(f"Date range: {df['period_end_date'].min()} to {df['period_end_date'].max()}")
    
    return df


# ============================================================================
# VARIABLE CONSTRUCTION
# ============================================================================

def calculate_growth(df, entity_col, value_col, output_col):
    """Calculate quarterly growth rate as log difference (in percent)"""
    x = pd.to_numeric(df[value_col], errors='coerce')
    x_lag = x.groupby(df[entity_col], sort=False).shift(1)
    valid = x.notna() & x_lag.notna() & (x > 0) & (x_lag > 0)
    df[output_col] = np.where(valid, 100.0 * (np.log(x) - np.log(x_lag)), np.nan)
    return df


def to_decimal_ratio(series: pd.Series) -> pd.Series:
    """Convert percentage to decimal if needed"""
    series = pd.to_numeric(series, errors="coerce")
    median = series.dropna().median()
    if pd.notna(median) and median > 1.0:
        return series / 100.0
    return series


def prepare_all_variables(df):
    """Create all variables needed for all regressions"""
    df = df.copy()
    df = df.sort_values(['bank_id', 'period_end_date']).reset_index(drop=True)
    
    # Find correct column names
    assets_col = 'total_assets'
    equity_col = 'common_equity_total'
    
    # Use total_assets_2 as fallback if needed
    if 'total_assets_2' in df.columns:
        df['total_assets'] = df['total_assets'].fillna(df['total_assets_2'])
    
    # Convert to numeric
    df['assets'] = pd.to_numeric(df[assets_col], errors='coerce')
    df['equity'] = pd.to_numeric(df[equity_col], errors='coerce')
    
    # Book leverage
    valid = df['assets'].notna() & df['equity'].notna() & (df['assets'] > 0) & (df['equity'] > 0)
    df['book_leverage'] = np.where(valid, df['assets'] / df['equity'], np.nan)
    
    # Growth rates (for Asset Growth models)
    df = calculate_growth(df, 'bank_id', 'assets', 'asset_growth')
    df = calculate_growth(df, 'bank_id', 'book_leverage', 'leverage_growth')
    
    # Core variables for Leverage/Risk models
    df['leverage'] = df['total_assets'] / df['common_equity_total']
    df['unit_var'] = df['var_99_level'] / df['total_assets']
    
    # Log transformations
    for var in ['leverage', 'total_assets', 'common_equity_total', 'var_99_level', 'unit_var']:
        df[f'log_{var}'] = np.log(df[var].where(df[var] > 0))
    
    # LCR variables
    if 'liquidity_coverage_ratio_basel_3' in df.columns:
        df['lcr_ratio'] = to_decimal_ratio(df['liquidity_coverage_ratio_basel_3'])
        df['lcr_lag1'] = df.groupby('bank_id')['lcr_ratio'].shift(1)
        df['int_unitvar_lcr'] = df['log_unit_var'] * df['lcr_lag1']
    
    # CET1 variables
    if CET1_COLUMN in df.columns:
        df['cet1_ratio'] = to_decimal_ratio(df[CET1_COLUMN])
        df['cet1_lag1'] = df.groupby('bank_id')['cet1_ratio'].shift(1)
        df['int_unitvar_cet1'] = df['log_unit_var'] * df['cet1_lag1']
    
    # COVID variables
    covid_start = pd.to_datetime(COVID_START)
    covid_end = pd.to_datetime(COVID_END)
    df['covid_dummy'] = (
        (df['period_end_date'] >= covid_start) &
        (df['period_end_date'] <= covid_end)
    ).astype(int)
    df['int_covid_unitvar'] = df['covid_dummy'] * df['log_unit_var']
    
    print(f"\nVariables prepared: {len(df)} observations")
    
    return df


# ============================================================================
# REGRESSION FUNCTIONS
# ============================================================================

def run_pooled_ols(df, dep_var, indep_vars, model_name):
    """Pooled OLS with clustered standard errors"""
    if isinstance(indep_vars, str):
        indep_vars = [indep_vars]
    
    reg_data = df[['bank_id'] + [dep_var] + indep_vars].dropna()
    
    if reg_data.empty:
        return None
    
    y = reg_data[dep_var].values
    X = sm.add_constant(reg_data[indep_vars].values)
    
    model = sm.OLS(y, X)
    result = model.fit(cov_type='cluster', cov_kwds={'groups': reg_data['bank_id']})
    
    # Extract coefficients
    coef_dict = {'model_name': model_name, 'n_obs': int(result.nobs), 
                 'r_squared': result.rsquared_adj}
    
    for i, var in enumerate(['const'] + indep_vars):
        coef_dict[f'b_{var}'] = result.params[i]
        coef_dict[f'se_{var}'] = result.bse[i]
        coef_dict[f't_{var}'] = result.tvalues[i]
        coef_dict[f'p_{var}'] = result.pvalues[i]
    
    return coef_dict


def run_panel_fe(df, dep_var, indep_vars, model_name, bank_fe=False, time_fe=False):
    """Panel regression with fixed effects"""
    if isinstance(indep_vars, str):
        indep_vars = [indep_vars]
    
    reg_data = df[['bank_id', 'period_end_date'] + [dep_var] + indep_vars].dropna()
    
    if reg_data.empty:
        return None
    
    panel = reg_data.set_index(['bank_id', 'period_end_date'])
    y = panel[[dep_var]]
    X = panel[indep_vars]
    
    model = PanelOLS(y, X, entity_effects=bank_fe, time_effects=time_fe)
    result = model.fit(cov_type='clustered', cluster_entity=True)
    
    # Extract coefficients
    r2 = getattr(result, 'rsquared_within', result.rsquared_inclusive)
    coef_dict = {
        'model_name': model_name,
        'n_obs': int(result.nobs),
        'n_banks': panel.index.get_level_values(0).nunique(),
        'r_squared': r2
    }
    
    for var in result.params.index:
        coef_dict[f'b_{var}'] = float(result.params[var])
        coef_dict[f'se_{var}'] = float(result.std_errors[var])
        coef_dict[f't_{var}'] = float(result.tstats[var])
        coef_dict[f'p_{var}'] = float(result.pvalues[var])
    
    return coef_dict


# ============================================================================
# RUN ALL MODELS
# ============================================================================

def run_all_models(df):
    """Run all regression models"""
    results = []
    
    print("\n" + "="*80)
    print("RUNNING ALL REGRESSIONS")
    print("="*80)
    
    # =========================
    # ASSET GROWTH MODELS
    # =========================
    print("\n--- ASSET GROWTH MODELS ---")
    
    res = run_pooled_ols(df, 'asset_growth', 'leverage_growth', 'A1_AssetGrowth_Pooled')
    if res: results.append(res)
    
    res = run_panel_fe(df, 'asset_growth', 'leverage_growth', 'A2_AssetGrowth_BankFE', 
                       bank_fe=True, time_fe=False)
    if res: results.append(res)
    
    res = run_panel_fe(df, 'asset_growth', 'leverage_growth', 'A3_AssetGrowth_TimeFE',
                       bank_fe=False, time_fe=True)
    if res: results.append(res)
    
    res = run_panel_fe(df, 'asset_growth', 'leverage_growth', 'A4_AssetGrowth_BankTimeFE',
                       bank_fe=True, time_fe=True)
    if res: results.append(res)
    
    # =========================
    # LEVERAGE/RISK MODELS
    # =========================
    print("\n--- LEVERAGE/RISK MODELS ---")
    
    res = run_panel_fe(df, 'log_leverage', 'log_unit_var', 'M1_Leverage_UnitVaR',
                       bank_fe=True, time_fe=False)
    if res: results.append(res)
    
    res = run_panel_fe(df, 'log_total_assets', 'log_var_99_level', 'M2_Assets_VaR',
                       bank_fe=True, time_fe=False)
    if res: results.append(res)
    
    res = run_panel_fe(df, 'log_common_equity_total', 'log_var_99_level', 'M3_Equity_VaR',
                       bank_fe=True, time_fe=False)
    if res: results.append(res)
    
    # Model 4A: LCR interaction
    res = run_panel_fe(df, 'log_leverage', 
                       ['log_unit_var', 'lcr_lag1', 'int_unitvar_lcr'],
                       'M4A_Leverage_LCR', bank_fe=True, time_fe=False)
    if res: results.append(res)
    
    # Model 4B: CET1 interaction
    res = run_panel_fe(df, 'log_leverage',
                       ['log_unit_var', 'cet1_lag1', 'int_unitvar_cet1'],
                       'M4B_Leverage_CET1', bank_fe=True, time_fe=False)
    if res: results.append(res)
    
    # Model 4C: LCR + CET1 interactions
    res = run_panel_fe(df, 'log_leverage',
                       ['log_unit_var', 'lcr_lag1', 'cet1_lag1', 
                        'int_unitvar_lcr', 'int_unitvar_cet1'],
                       'M4C_Leverage_LCR_CET1', bank_fe=True, time_fe=False)
    if res: results.append(res)
    
    # Model 5: LCR only
    res = run_panel_fe(df, 'log_leverage', 'lcr_lag1', 'M5_Leverage_LCR_only',
                       bank_fe=True, time_fe=False)
    if res: results.append(res)
    
    # =========================
    # COVID MODEL
    # =========================
    print("\n--- COVID MODEL ---")
    
    res = run_panel_fe(df, 'log_leverage',
                       ['log_unit_var', 'covid_dummy', 'int_covid_unitvar'],
                       'C1_Leverage_COVID', bank_fe=True, time_fe=False)
    if res: results.append(res)
    
    return results


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_all_results(results):
    """Save all results to one CSV file"""
    df_results = pd.DataFrame(results)
    
    # Reorder columns: model info first, then sorted coefficient columns
    info_cols = [c for c in df_results.columns if not c.startswith(('b_', 'se_', 't_', 'p_'))]
    coef_cols = sorted([c for c in df_results.columns if c.startswith(('b_', 'se_', 't_', 'p_'))])
    
    df_results = df_results[info_cols + coef_cols]
    
    # Create output directory
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    df_results.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n{'='*80}")
    print(f"✓ Saved all results to: {OUTPUT_FILE}")
    print(f"  Total models: {len(results)}")
    print(f"{'='*80}\n")
    
    return df_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution"""
    print("="*80)
    print("COMBINED REGRESSION ANALYSIS")
    print("Using var_99_gaussian (NOT BoA factor)")
    print("="*80)
    
    # Load and merge data
    df = merge_data()
    
    # Prepare all variables
    df = prepare_all_variables(df)
    
    # Run all models
    results = run_all_models(df)
    
    # Save results
    df_results = save_all_results(results)
    
    # Display summary
    print("\nSUMMARY OF RESULTS:")
    print("-"*80)
    for _, row in df_results.iterrows():
        print(f"{row['model_name']:30s} N={row['n_obs']:5.0f}  R²={row['r_squared']:6.3f}")
    
    print("\n" + "="*80)
    print("ALL REGRESSIONS COMPLETED ✓")
    print("="*80)


if __name__ == "__main__":
    main()