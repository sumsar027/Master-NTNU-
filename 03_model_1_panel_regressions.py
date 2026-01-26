"""
Panel regressions: Leverage, Assets, Equity on VaR measures
Replicates Adrian & Shin (2014) style analysis

Models:
  1. log(Leverage) ~ log(UnitVaR) + bank FE
  2. log(Assets) ~ log(VaR) + bank FE
  3. log(Equity) ~ log(VaR) + bank FE
  4A. log(Leverage) ~ log(UnitVaR) + LCR_{t-1} + log(UnitVaR)*LCR_{t-1} + bank FE

Uses harmonized 99% VaR from Gaussian conversion.
"""

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, PooledOLS
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================

BANK_MAP = {
    "bankofamerica": "bank_of_america",
    "citigroup": "citibank",
    "wellsfargo": "wells_fargo"
}

TIME_FE = False  # Set True for robustness checks

ASSET_COL = "total_assets"
EQUITY_COL = "common_equity_total"      # evt. "shareholders_equity_common"
LCR_COL   = "liquidity_coverage_ratio_basel_3"

# ============================================================================
# LOAD & CLEAN DATA
# ============================================================================

def load_data():
    """Load and merge balance sheet + VaR data"""

    base = pd.read_csv("output/data/merged_quarterly_balanced.csv")
    var  = pd.read_csv("output/data/merged_with_var_99_dual_methods.csv")

    print(f"Balance sheet: {len(base)} rows, {base['bank_id'].nunique() if 'bank_id' in base.columns else base['bank'].nunique()} banks")
    print(f"VaR data: {len(var)} rows, {var['bank_id'].nunique()} banks")

    # Normalize column names
    for df in [base, var]:
        if 'bank' in df.columns and 'bank_id' not in df.columns:
            df.rename(columns={'bank': 'bank_id'}, inplace=True)

    # Handle alternative column names in base data (holder Modell 1–3 likt som før)
    if 'total_assets_2' in base.columns:
        base['total_assets'] = base['total_assets'].fillna(base['total_assets_2'])

    # Harmonize bank IDs
    base['bank_id'] = base['bank_id'].str.strip().str.lower().replace(BANK_MAP)
    var['bank_id']  = var['bank_id'].str.strip().str.lower().replace(BANK_MAP)

    print(f"\nAfter harmonization:")
    print(f"Balance sheet banks: {sorted(base['bank_id'].unique())}")
    print(f"VaR banks: {sorted(var['bank_id'].unique())}")

    # Ensure datetime
    base['period_end_date'] = pd.to_datetime(base['period_end_date'])
    var['period_end_date']  = pd.to_datetime(var['period_end_date'])

    # Create VaR level using reported 99% when available, otherwise Gaussian-converted 99%
    if 'var_99' in var.columns:
        var['var_99_level'] = var['var_99']
    else:
        var['var_99_level'] = np.nan

    if 'var_99_gaussian' in var.columns:
        var['var_99_level'] = var['var_99_level'].fillna(var['var_99_gaussian'])

    print(f"\nVaR data: {var['var_99_level'].notna().sum()} non-missing values out of {len(var)}")

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
    """Create all regression variables (Models 1–3 sample stays the same as before)"""

    df = df.copy()

    # Core vars (samme som før)
    df['leverage']  = df[ASSET_COL] / df[EQUITY_COL]
    df['unit_var']  = df['var_99_level'] / df[ASSET_COL]

    # Logs (samme som før)
    for var in ['leverage', ASSET_COL, EQUITY_COL, 'var_99_level', 'unit_var']:
        df[f'log_{var}'] = np.log(df[var].where(df[var] > 0))

    # --- NEW (Model 4A only): LCR lag + interaksjon ---
    # Dette endrer IKKE sample til Modell 1–3, fordi vi ikke dropper på disse.
    if LCR_COL in df.columns:
        df[LCR_COL] = pd.to_numeric(df[LCR_COL], errors='coerce')
        df = df.sort_values(['bank_id', 'period_end_date'])
        df['lcr_ratio'] = df[LCR_COL] / 100.0
        df['lcr_lag1'] = df.groupby('bank_id')['lcr_ratio'].shift(1)
        df['int_unitvar_lcr'] = df['log_unit_var'] * df['lcr_lag1']
    else:
        df['lcr_lag1'] = np.nan
        df['int_unitvar_lcr'] = np.nan

    # Use common estimation sample across Models 1–3 (SAMME som før)
    required_base = [
        'bank_id', 'period_end_date',
        'log_leverage', f'log_{ASSET_COL}', f'log_{EQUITY_COL}',
        'log_unit_var', 'log_var_99_level'
    ]

    # behold LCR-kolonnene uten å la dem styre dropna
    keep_cols = required_base + ['lcr_lag1', 'int_unitvar_lcr']

    n_before = len(df)
    df = df[keep_cols]
    df = df.dropna(subset=required_base)  # <- viktig: Modell 1–3 uendret sample
    print(f"Estimation sample (Models 1–3): {len(df)} obs ({n_before - len(df)} dropped due to missing base values)")

    # gi gamle navn tilbake (slik at resten av koden din er uendret)
    df = df.rename(columns={
        f'log_{ASSET_COL}': 'log_total_assets',
        f'log_{EQUITY_COL}': 'log_common_equity_total'
    })

    return df

# ============================================================================
# ESTIMATION
# ============================================================================

def estimate_model(df, dep_var, indep_var, model_name, estimator='FE'):
    """Run single regression (Models 1–3 unchanged)"""

    panel = df.set_index(['bank_id', 'period_end_date'])
    y = panel[[dep_var]]
    X = panel[[indep_var]]

    if estimator == 'FE':
        model = PanelOLS(y, X, entity_effects=True, time_effects=TIME_FE)
        res = model.fit(cov_type='clustered', cluster_entity=True)
        r2 = res.rsquared_within
        const = np.nan
    else:
        X_pooled = X.copy()
        X_pooled.insert(0, 'const', 1.0)
        model = PooledOLS(y, X_pooled)
        res = model.fit(cov_type='clustered', cluster_entity=True)
        r2 = res.rsquared
        const = res.params['const']

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

def estimate_model_4a(df, estimator='FE'):
    """
    Model 4A:
      log(Leverage) ~ log(UnitVaR) + LCR_{t-1} + log(UnitVaR)*LCR_{t-1} + bank FE
    Returnerer en liste med rader (én per koeff) for enkel output.
    """

    dep_var = 'log_leverage'
    indep_vars = ['log_unit_var', 'lcr_lag1', 'int_unitvar_lcr']
    model_name = "Model 4A: Leverage + LCR"

    # kun Model 4A sample (dropper bare på det Model 4A trenger)
    df4 = df.dropna(subset=[dep_var] + indep_vars).copy()

    panel = df4.set_index(['bank_id', 'period_end_date'])
    y = panel[[dep_var]]
    X = panel[indep_vars]

    if estimator == 'FE':
        model = PanelOLS(y, X, entity_effects=True, time_effects=TIME_FE)
        res = model.fit(cov_type='clustered', cluster_entity=True)
        print(res.summary)
        r2 = res.rsquared_within
        const = np.nan
    else:
        X_pooled = X.copy()
        X_pooled.insert(0, 'const', 1.0)
        model = PooledOLS(y, X_pooled)
        res = model.fit(cov_type='clustered', cluster_entity=True)
        print(res.summary)
        r2 = res.rsquared
        const = res.params['const']

    rows = []
    for v in indep_vars:
        rows.append({
            'estimator': estimator,
            'model': model_name,
            'dependent_var': dep_var,
            'independent_var': v,
            'const': const,
            'beta': res.params[v],
            'se': res.std_errors[v],
            't_stat': res.tstats[v],
            'p_value': res.pvalues[v],
            'r_squared': r2,
            'n_obs': int(res.nobs),
            'n_banks': panel.index.get_level_values(0).nunique(),
            'n_periods': panel.index.get_level_values(1).nunique()
        })

    return rows

def run_all_models(df):
    """Estimate all specifications"""

    specs = [
        ('Model 1: Leverage', 'log_leverage', 'log_unit_var'),
        ('Model 2: Assets',   'log_total_assets', 'log_var_99_level'),
        ('Model 3: Equity',   'log_common_equity_total', 'log_var_99_level')
    ]

    results = []

    # Models 1–3 (FE) — uendret
    for name, dep, indep in specs:
        results.append(estimate_model(df, dep, indep, name, 'FE'))

    # Model 4A (FE)
    results.extend(estimate_model_4a(df, estimator='FE'))

    # Models 1–3 (Pooled) — uendret
    for name, dep, indep in specs:
        results.append(estimate_model(df, dep, indep, name, 'Pooled'))

    # Model 4A (Pooled)
    results.extend(estimate_model_4a(df, estimator='Pooled'))

    return pd.DataFrame(results)

# ============================================================================
# MAIN
# ============================================================================

def main():
    df = load_data()
    df = prepare_variables(df)

    results = run_all_models(df)

    print("\n" + "="*80)
    print("REGRESSION RESULTS")
    print("="*80 + "\n")
    print(results.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    Path('output/tables').mkdir(parents=True, exist_ok=True)
    results.to_csv('output/tables/Model_1_result.csv', index=False)
    print("\nResults saved to output/tables/Model_1_result.csv")

if __name__ == "__main__":
    main()
