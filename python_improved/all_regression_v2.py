"""
Panel regression script for bank leverage.

How to modify the script:

1. RAW_VARS
Add new raw variables here using the column name from the CSV.

2. Lagged variables
Lagged variables follow the naming rule:
    variable_lag1
These are created automatically in the code.

3. INTERACTIONS
Interaction terms are defined as:

    "new_name": ("variable_a", "variable_b")

Example:
    "cet1_x_var": ("cet1_lag1", "log_var")

Both variables must already exist in the dataset.

4. MODELS
Add or change regression models here by listing the variables to include.

If a variable does not exist, the script will skip the model instead of crashing.
"""

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from pathlib import Path

# ============================================================================
# SETTINGS
# These are the column names used in the raw CSV files.
# If you add a new variable later, add it here.
# ============================================================================

RAW_VARS = {
    "cet1": "capital_adequacy_core_tier_1",
    "lcr":  "liquidity_coverage_ratio_basel_3",
    "slr":  "leverage_ratio_basel_3",
    "div":  "dividend_payout_ratio",
    "roa":  "return_on_average_total_assets_income_before_discontinued_operations_extraordinary_items_ttm",
    # New variable example: "short_name": "column_name_in_csv"
}

# These are interaction terms.
# Each one is made by multiplying two variables that already exist.
INTERACTIONS = {
    "cet1_x_var": ("cet1_lag1", "log_var"), 
    "lcr_x_var":  ("lcr_lag1",  "log_var"),
    "slr_x_var":  ("slr_lag1",  "log_var"),
    "div_x_var":  ("div_lag1",  "log_var"),
    # New interaction example: "new_name": ("variable_a", "variable_b")
   
}

# Bank groups used in model M6
MARKET_BANKS  = {"goldmansachs", "morganstanley"}
CUSTODY_BANKS = {"wellsfargo", "bny", "statestreet"}
# Reference group (other banks) is implicitly defined as those not in the two groups above

OUTPUT_FILE = Path("output/tables/panel_regression_results.csv")

# ============================================================================
# MODELS
# Each model is listed here.
# To add a new model, copy one block and change the settings.
# The variable names must exist after prepare_variables() has run.
# ============================================================================

MODELS = [
    {
        "name":    "M1_Basis_VaR",
        "vars":    ["log_var", "roa_lag1", "log_assets"],
        "bank_fe": True,
        "time_fe": True,
    },
    {
        "name":    "M2_CET1_Interaction",
        "vars":    ["log_var", "cet1_lag1", "cet1_x_var", "roa_lag1", "log_assets"],
        "bank_fe": True,
        "time_fe": True,
    },
    {
        "name":    "M3_LCR_Interaction",
        "vars":    ["log_var", "lcr_lag1", "lcr_x_var", "roa_lag1", "log_assets"],
        "bank_fe": True,
        "time_fe": True,
    },
    {
        "name":    "M4_SLR_Interaction",
        "vars":    ["log_var", "slr_lag1", "slr_x_var", "roa_lag1", "log_assets"],
        "bank_fe": True,
        "time_fe": True,
    },
    {
        "name":    "M5_Dividend_Interaction",
        "vars":    ["log_var", "div_lag1", "div_x_var", "roa_lag1", "log_assets"],
        "bank_fe": True,
        "time_fe": True,
    },
    {
        "name":    "M6_BankType_Interactions",
        "vars":    ["log_var", "d_market", "d_custody",
                    "market_x_var", "custody_x_var", "roa_lag1", "log_assets"],
        "bank_fe": False,
        "time_fe": True,
    },
    {
        "name":    "M7_All_Interactions",
        "vars":    ["log_var", "cet1_lag1", "lcr_lag1", "slr_lag1", "div_lag1",
                    "cet1_x_var", "lcr_x_var", "slr_x_var", "div_x_var",
                    "roa_lag1", "log_assets"],
        "bank_fe": True,
        "time_fe": True,
    },
]

# ============================================================================
# LOAD DATA
# This function reads the two CSV files and merges them into one dataset.
# ============================================================================

def load_data() -> pd.DataFrame:
    bs  = pd.read_csv("data/processed/balance_sheet_panel_balanced.csv")
    var = pd.read_csv("output/data/var_99.csv")

    # Make sure both datasets use the same column names for bank and date
    for d in [bs, var]:
        id_col   = "bank_id" if "bank_id" in d.columns else "bank"
        date_col = "period_end_date" if "period_end_date" in d.columns else "date"
        d.rename(columns={id_col: "bank_id", date_col: "period_end_date"}, inplace=True)
        d["bank_id"]         = d["bank_id"].str.strip().str.lower()
        d["period_end_date"] = pd.to_datetime(d["period_end_date"])

    # Use the Gaussian VaR as the main VaR measure
    var["var_99_level"] = var["var_99_gaussian"]

    # Merge balance sheet data and VaR data by bank and date
    df = bs.merge(var[["bank_id", "period_end_date", "var_99_level"]],
                  on=["bank_id", "period_end_date"], how="inner")

    print(f"Data lastet: {len(df)} obs., {df['bank_id'].nunique()} banker  "
          f"({df['period_end_date'].min().date()} – {df['period_end_date'].max().date()})")

    # Sort the final dataset by bank and date
    return df.sort_values(["bank_id", "period_end_date"]).reset_index(drop=True)


# ============================================================================
# BUILD VARIABLES
# This section creates the variables used in the regressions.
# ============================================================================

def to_decimal(s: pd.Series) -> pd.Series:
    # Convert text to numbers
    s = pd.to_numeric(s, errors="coerce")

    # If the median is above 1, assume the values are percentages
    # and divide by 100
    return s / 100.0 if s.dropna().median() > 1.0 else s


def prepare_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Build leverage and log variables
    df["assets"]       = pd.to_numeric(df["total_assets"],        errors="coerce")
    df["equity"]       = pd.to_numeric(df["common_equity_total"], errors="coerce")
    valid              = (df["assets"] > 0) & (df["equity"] > 0)
    df["leverage"]     = np.where(valid, df["assets"] / df["equity"], np.nan)
    df["log_leverage"] = np.log(df["leverage"].where(df["leverage"] > 0))
    df["log_var"]      = np.log(df["var_99_level"].where(df["var_99_level"] > 0))
    df["log_assets"]   = np.log(df["assets"].where(df["assets"] > 0))

    # Create the main raw variables and their lagged versions
    # Example: roa -> roa_lag1
    for name, col in RAW_VARS.items():
        if col not in df.columns:
            print(f"Kolonne ikke funnet: {col}")
            continue
        df[name]           = to_decimal(df[col])
        df[f"{name}_lag1"] = df.groupby("bank_id")[name].shift(1)

    # Create interaction terms by multiplying two columns
    for name, (a, b) in INTERACTIONS.items():
        if a in df.columns and b in df.columns:
            df[name] = df[a] * df[b]
        else:
            print(f" Kan ikke lage interaksjon {name}: mangler {a} eller {b}")

    # Create bank-type dummy variables for model M6
    norm                = df["bank_id"].str.replace(r"[^a-z0-9]", "", regex=True)
    df["d_market"]      = norm.isin(MARKET_BANKS).astype(int)
    df["d_custody"]     = norm.isin(CUSTODY_BANKS).astype(int)

    # Create interactions between bank type and log_var
    df["market_x_var"]  = df["d_market"]  * df["log_var"]
    df["custody_x_var"] = df["d_custody"] * df["log_var"]

    # Show how many non-missing observations each key variable has
    check = ["log_leverage", "log_var", "roa_lag1",
             "cet1_lag1", "lcr_lag1", "slr_lag1", "div_lag1"]
    print("\nDatadekning:")
    for c in check:
        if c in df.columns:
            print(f"  {c:25s}: {df[c].notna().sum():4d} / {len(df)}")

    return df


# ============================================================================
# REGRESSION
# This section runs one model at a time.
# ============================================================================

def stars(t: float) -> str:
    # Add significance stars based on the t-statistic
    a = abs(t)
    return "***" if a > 2.576 else "**" if a > 1.960 else "*" if a > 1.645 else ""


def run_model(df: pd.DataFrame, model: dict) -> dict | None:
    # Read the settings for one model
    name    = model["name"]
    indep   = model["vars"]
    bank_fe = model["bank_fe"]
    time_fe = model["time_fe"] 
 
    # Check if all needed columns exist before running the model
    missing = [v for v in indep if v not in df.columns]
    if missing:
        print(f"  [{name}] Mangler kolonner {missing} – hopper over.")
        return None

    # Keep only the columns needed for this model and drop missing rows
    data = df[["bank_id", "period_end_date", "log_leverage"] + indep].dropna()

    # Skip the model if too few observations are left
    if len(data) < 30:
        print(f"\n  [{name}] For få obs. ({len(data)}) – hopper over.")
        return None

    # Set bank and date as the panel index
    panel  = data.set_index(["bank_id", "period_end_date"])

    # Run the panel regression
    result = PanelOLS(panel[["log_leverage"]], panel[indep],
                      entity_effects=bank_fe, time_effects=time_fe,
                      drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)

    # Get the R-squared measure
    r2 = getattr(result, "rsquared_within", result.rsquared_inclusive)

    # Show whether the model uses bank fixed effects and time fixed effects
    fe = f"Bank={'✓' if bank_fe else '✗'}  Tid={'✓' if time_fe else '✗'}"

    # Print the model results
    print(f"\n  [{name}]  N={result.nobs}  R²={r2:.3f}  {fe}")
    for var in result.params.index:
        b = float(result.params[var])
        t = float(result.tstats[var])
        print(f"    {var:35s}  β={b:8.4f}  t={t:7.3f} {stars(t)}")

    # Save the result in a dictionary
    row = {
        "model":      name,
        "bank_fe":    bank_fe,
        "time_fe":    time_fe,
        "n_obs":      int(result.nobs),
        "n_banks":    panel.index.get_level_values(0).nunique(),
        "n_quarters": panel.index.get_level_values(1).nunique(),
        "r2_within":  round(r2, 4),
    }

    # Save coefficient, standard error, and t-stat for each variable
    for var in result.params.index:
        row[f"b_{var}"]  = round(float(result.params[var]),     4)
        row[f"se_{var}"] = round(float(result.std_errors[var]), 4)
        row[f"t_{var}"]  = round(float(result.tstats[var]),     4)

    return row


# ============================================================================
# MAIN
# This is the full workflow:
# 1. Load data
# 2. Build variables
# 3. Run all models
# 4. Save results
# ============================================================================

def main():
    print("=" * 70)
    print("PANEL-REGRESJONER")
    print("=" * 70)

    df      = load_data()
    df      = prepare_variables(df)

    print("\n" + "=" * 70)
    print("KJØRER MODELLER")
    print("=" * 70)

    # Run all models in the list
    results = [r for m in MODELS if (r := run_model(df, m))]

    # Save results to CSV
    out       = pd.DataFrame(results)
    info_cols = [c for c in out.columns if not c.startswith(("b_", "se_", "t_"))]
    coef_cols = sorted(c for c in out.columns if c.startswith(("b_", "se_", "t_")))
    out       = out[info_cols + coef_cols]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'=' * 70}")
    print(f"✓ Lagret: {OUTPUT_FILE}")
    print("\nSUMMER:")
    print("-" * 70)
    for r in results:
        fe = f"Bank={'✓' if r['bank_fe'] else '✗'}  Tid={'✓' if r['time_fe'] else '✗'}"
        print(f"  {r['model']:40s}  N={r['n_obs']:5d}  R²={r['r2_within']:.3f}  {fe}")
    print("\n✓ FERDIG")


if __name__ == "__main__":
    main()