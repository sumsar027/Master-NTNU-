"""
Robustness Test: Compare Gaussian vs Simple Scaling for VaR 99%

Tests whether using Gaussian scaling (*1.414) vs simple scaling (*2.0) 
affects regression results. If results are similar, this validates the 
choice of Gaussian scaling in the main analysis.

Models tested:
  1. log(Leverage) ~ log(Unit VaR)
  2. log(Assets) ~ log(VaR)  
  3. log(Equity) ~ log(VaR)

Each model estimated with:
  - Bank fixed effects (FE)
  - Pooled OLS
"""

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, PooledOLS
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================

BALANCE_FILE = Path("data/processed/merged_quarterly_balanced.csv")
VAR_FILE = Path("output/data/merged_with_var_99_dual_methods.csv")
OUTPUT_FILE = Path("output/tables/panel_results_var99_gauss_vs_x2.csv")

GAUSS_SCALE = 1.41421356237
SIMPLE_SCALE = 2.0

# Bank name mapping (harmonize differences between files)
BANK_MAP = {
    "bankofamerica": "bank_of_america",
    "citigroup": "citibank",
    "wellsfargo": "wells_fargo"
}

# ============================================================================
# LOAD & PREPARE DATA
# ============================================================================

# Read files
balance = pd.read_csv(BALANCE_FILE)
var_data = pd.read_csv(VAR_FILE)

# Standardize column names
if "bank" in balance.columns and "bank_id" not in balance.columns:
    balance = balance.rename(columns={"bank": "bank_id"})
elif "bank" in balance.columns and "bank_id" in balance.columns:
    balance = balance.drop(columns=["bank"])

if "bank" in var_data.columns and "bank_id" not in var_data.columns:
    var_data = var_data.rename(columns={"bank": "bank_id"})
elif "bank" in var_data.columns and "bank_id" in var_data.columns:
    var_data = var_data.drop(columns=["bank"])

# If duplicate column names exist (e.g., bank_id twice), keep the first occurrence
if balance.columns.duplicated().any():
    balance = balance.loc[:, ~balance.columns.duplicated()]
if var_data.columns.duplicated().any():
    var_data = var_data.loc[:, ~var_data.columns.duplicated()]

# Harmonize bank names
balance["bank_id"] = balance["bank_id"].astype(str).str.lower().replace(BANK_MAP)
var_data["bank_id"] = var_data["bank_id"].astype(str).str.lower().replace(BANK_MAP)

# Convert dates
balance["period_end_date"] = pd.to_datetime(balance["period_end_date"])
var_data["period_end_date"] = pd.to_datetime(var_data["period_end_date"])

# Create VaR 99% with both methods
# Use existing VaR 99% if available, else scale VaR 95%
var95 = pd.to_numeric(var_data["var_95"], errors="coerce")
var99 = pd.to_numeric(var_data.get("var_99"), errors="coerce")

var_data["var_99_gauss"] = var99.fillna(var95 * GAUSS_SCALE)
var_data["var_99_x2"] = var99.fillna(var95 * SIMPLE_SCALE)

# Merge datasets
df = balance.merge(
    var_data[["bank_id", "period_end_date", "var_99_gauss", "var_99_x2"]],
    on=["bank_id", "period_end_date"],
    how="inner"
)

# Create variables
df["assets"] = pd.to_numeric(df["total_assets_2"], errors="coerce")
df["equity"] = pd.to_numeric(df["common_equity_total"], errors="coerce")
df = df.dropna(subset=["assets", "equity"])
df = df[(df["assets"] > 0) & (df["equity"] > 0)]

df["leverage"] = df["assets"] / df["equity"]
df["log_leverage"] = np.log(df["leverage"])
df["log_assets"] = np.log(df["assets"])
df["log_equity"] = np.log(df["equity"])

print(f"Sample: {len(df)} observations, {df['bank_id'].nunique()} banks")

# ============================================================================
# RUN REGRESSIONS
# ============================================================================

results = []

# Loop over both VaR methods
for method, var_col in [("Gaussian", "var_99_gauss"), ("Simple_x2", "var_99_x2")]:
    
    # Prepare regression data
    reg_df = df[["bank_id", "period_end_date", "log_leverage", 
                 "log_assets", "log_equity", var_col, "assets"]].copy()
    
    reg_df["var99"] = pd.to_numeric(reg_df[var_col], errors="coerce")
    reg_df = reg_df[reg_df["var99"] > 0]
    
    # Create VaR variables
    reg_df["log_var"] = np.log(reg_df["var99"])
    reg_df["unit_var"] = reg_df["var99"] / reg_df["assets"]
    reg_df["log_unit_var"] = np.log(reg_df["unit_var"])
    
    # Clean and set panel index
    reg_df = reg_df.replace([np.inf, -np.inf], np.nan).dropna()
    reg_df = reg_df.set_index(["bank_id", "period_end_date"])
    
    n_banks = reg_df.index.get_level_values(0).nunique()
    
    # Define three models
    models = [
        ("Model 1: Leverage ~ UnitVaR", "log_leverage", "log_unit_var"),
        ("Model 2: Assets ~ VaR", "log_assets", "log_var"),
        ("Model 3: Equity ~ VaR", "log_equity", "log_var")
    ]
    
    # Estimate each model with FE and Pooled
    for model_name, y_var, x_var in models:
        y = reg_df[[y_var]]
        X = reg_df[[x_var]]
        
        # Fixed Effects
        fe_model = PanelOLS(y, X, entity_effects=True)
        fe_res = fe_model.fit(cov_type="clustered", cluster_entity=True)
        
        results.append({
            "var_method": method,
            "model": model_name,
            "estimator": "Bank FE",
            "coefficient": fe_res.params[x_var],
            "std_error": fe_res.std_errors[x_var],
            "t_stat": fe_res.tstats[x_var],
            "p_value": fe_res.pvalues[x_var],
            "r_squared": fe_res.rsquared_within,
            "n_obs": int(fe_res.nobs),
            "n_banks": n_banks
        })
        
        # Pooled OLS
        X_pooled = X.copy()
        X_pooled.insert(0, "const", 1.0)
        pooled_model = PooledOLS(y, X_pooled)
        pooled_res = pooled_model.fit(cov_type="clustered", cluster_entity=True)
        
        results.append({
            "var_method": method,
            "model": model_name,
            "estimator": "Pooled",
            "coefficient": pooled_res.params[x_var],
            "std_error": pooled_res.std_errors[x_var],
            "t_stat": pooled_res.tstats[x_var],
            "p_value": pooled_res.pvalues[x_var],
            "r_squared": pooled_res.rsquared,
            "n_obs": int(pooled_res.nobs),
            "n_banks": n_banks
        })

# ============================================================================
# SAVE RESULTS
# ============================================================================

results_df = pd.DataFrame(results)
OUTPUT_FILE.parent.mkdir(exist_ok=True, parents=True)
results_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nResults saved to {OUTPUT_FILE}")
print(f"Total regressions: {len(results_df)}")
