import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from pathlib import Path


# ============================================================
# BASIC SETTINGS
# Change these if you want new specifications or output path
# ============================================================

# File where regression results will be saved
OUTPUT_FILE = Path("output/tables/panel_regression_results.csv")

# Dependent variable: this is what the models try to explain
DEPENDENT_VAR = "d_leverage"

# Risk variable used in the static models
STATIC_RISK = "d_var_lag1"

# Risk variable used in the dynamic models
DYNAMIC_RISK = "var_lag1"

# Empty list = use all banks
# Add bank names here if you only want some banks
BANKS_INCLUDE = []

# Short names used in the code -> real column names in the data
RAW_VARS = {
    "cet1": "capital_adequacy_core_tier_1",
    "lcr":  "liquidity_coverage_ratio_basel_3",
    "slr":  "leverage_ratio_basel_3",
    "roa":  "return_on_average_total_assets_income_before_discontinued_operations_extraordinary_items_ttm",
}

# Bank groups used in the group models
MARKET_BANKS = {"goldmansachs", "morganstanley"}
UNIVERSAL_BANKS = {"jpmorgan", "bankofamerica", "wellsfargo", "citigroup"}


# ============================================================
# MODELS
# This list tells the script which regressions to run
# ============================================================

MODELS = [
    # --------------------------------------------------------
    # STATIC MODELS
    # These use lagged change in risk
    # --------------------------------------------------------
    {
        "name": "M1_Static_Basis",
        "vars": [STATIC_RISK],
        "bank_fe": True, "time_fe": True,
    },
    {
        "name": "M2_Static_CET1",
        "vars": [STATIC_RISK, "cet1_lag1", f"cet1_lag1_x_{STATIC_RISK}", "log_assets_lag1", "roa_lag1"],
        "bank_fe": True, "time_fe": True,
    },
    {
        "name": "M3_Static_LCR",
        "vars": [STATIC_RISK, "lcr_lag1", f"lcr_lag1_x_{STATIC_RISK}", "log_assets_lag1", "roa_lag1"],
        "bank_fe": True, "time_fe": True,
    },
    {
        "name": "M4_Static_SLR",
        "vars": [STATIC_RISK, "slr_lag1", f"slr_lag1_x_{STATIC_RISK}", "log_assets_lag1", "roa_lag1"],
        "bank_fe": True, "time_fe": True,
    },
    {
        "name": "M5_Static_Group",
        "vars": [
            STATIC_RISK,
            "d_market", "d_universal",
            f"market_x_{STATIC_RISK}", f"universals_x_{STATIC_RISK}",
            "log_assets_lag1", "roa_lag1",
        ],
        "bank_fe": False, "time_fe": True,
    },

    # --------------------------------------------------------
    # DYNAMIC MODELS
    # These also include lagged dependent variable
    # --------------------------------------------------------
    {
        "name": "M1_Dynamic_Basis",
        "vars": ["d_leverage_lag1", DYNAMIC_RISK],
        "bank_fe": True, "time_fe": True,
    },
    {
        "name": "M2_Dynamic_CET1",
        "vars": ["d_leverage_lag1", DYNAMIC_RISK, "cet1_lag1", f"cet1_lag1_x_{DYNAMIC_RISK}", "log_assets_lag1", "roa_lag1"],
        "bank_fe": True, "time_fe": True,
    },
    {
        "name": "M3_Dynamic_LCR",
        "vars": ["d_leverage_lag1", DYNAMIC_RISK, "lcr_lag1", f"lcr_lag1_x_{DYNAMIC_RISK}", "log_assets_lag1", "roa_lag1"],
        "bank_fe": True, "time_fe": True,
    },
    {
        "name": "M4_Dynamic_SLR",
        "vars": ["d_leverage_lag1", DYNAMIC_RISK, "slr_lag1", f"slr_lag1_x_{DYNAMIC_RISK}", "log_assets_lag1", "roa_lag1"],
        "bank_fe": True, "time_fe": True,
    },
    {
        "name": "M5_Dynamic_Group",
        "vars": [
            "d_leverage_lag1", DYNAMIC_RISK,
            "d_market", "d_universal",
            f"market_x_{DYNAMIC_RISK}", f"universals_x_{DYNAMIC_RISK}",
            "log_assets_lag1", "roa_lag1",
        ],
        "bank_fe": False, "time_fe": True,
    },
]


# ============================================================
# LOAD DATA
# Reads the two input files and merges them into one dataset
# ============================================================

def load_data() -> pd.DataFrame:
    # Read balance sheet data
    bs = pd.read_csv("data/processed/balance_sheet_panel_balanced.csv")

    # Read VaR data
    var = pd.read_csv("output/data/var_99.csv")

    # Make column names consistent across both files
    for d in [bs, var]:
        id_col = "bank_id" if "bank_id" in d.columns else "bank"
        date_col = "period_end_date" if "period_end_date" in d.columns else "date"

        d.rename(columns={id_col: "bank_id", date_col: "period_end_date"}, inplace=True)

        # Clean bank names so matching works better
        d["bank_id"] = d["bank_id"].astype(str).str.strip().str.lower()

        # Convert date column to real dates
        d["period_end_date"] = pd.to_datetime(d["period_end_date"])

    # Make VaR numeric
    var["var_99_level"] = pd.to_numeric(var["var_99_gaussian"], errors="coerce")

    # Merge the two datasets on bank and date
    df = bs.merge(
        var[["bank_id", "period_end_date", "var_99_level"]],
        on=["bank_id", "period_end_date"],
        how="inner",
    )

    # Keep only selected banks if BANKS_INCLUDE is not empty
    if BANKS_INCLUDE:
        keep = [b.strip().lower() for b in BANKS_INCLUDE]
        df = df[df["bank_id"].isin(keep)]
        print(f"Bank filter active: {keep}")

    # Print a quick data summary
    print(f"Data loaded: {len(df)} obs., {df['bank_id'].nunique()} banks "
          f"({df['period_end_date'].min().date()} – {df['period_end_date'].max().date()})")
    print(f"Banks: {sorted(df['bank_id'].unique())}")

    # Sort by bank and time
    return df.sort_values(["bank_id", "period_end_date"]).reset_index(drop=True)


# ============================================================
# PREPARE VARIABLES
# Creates leverage, changes, lags, dummies, and interactions
# ============================================================

def prepare_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Group by bank so lags and differences are done within each bank
    g = df.groupby("bank_id")

    # --------------------------------------------------------
    # Assets
    # --------------------------------------------------------
    df["assets"] = pd.to_numeric(df["total_assets"], errors="coerce")
    df["log_assets"] = np.log(df["assets"].where(df["assets"] > 0))
    df["log_assets_lag1"] = g["log_assets"].shift(1)

    # --------------------------------------------------------
    # Leverage
    # --------------------------------------------------------
    df["equity"] = pd.to_numeric(df["common_equity_total"], errors="coerce")

    # Only calculate leverage when both assets and equity are positive
    valid = (df["assets"] > 0) & (df["equity"] > 0)
    df["leverage"] = np.where(valid, df["assets"] / df["equity"], np.nan)

    # Change in leverage from previous period
    df["d_leverage"] = g["leverage"].diff()

    # Lagged change in leverage
    df["d_leverage_lag1"] = g["d_leverage"].shift(1)

    # --------------------------------------------------------
    # VaR
    # --------------------------------------------------------
    df["var"] = np.log(df["var_99_level"].where(df["var_99_level"] > 0))
    df["var_lag1"] = g["var"].shift(1)
    df["d_var"] = g["var"].diff()
    df["d_var_lag1"] = g["d_var"].shift(1)

    # --------------------------------------------------------
    # Control and restriction variables
    # --------------------------------------------------------
    for short_name, raw_col in RAW_VARS.items():
        if raw_col not in df.columns:
            print(f"  Warning: column not found – {raw_col}")
            continue

        # IMPORTANT:
        # This divides by 100.
        # That is only correct if the raw data are in percent form, like 12.0.
        # If the raw data are already decimals, like 0.12, remove / 100.0.
        df[short_name] = pd.to_numeric(df[raw_col], errors="coerce") / 100.0
        df[f"{short_name}_lag1"] = g[short_name].shift(1)

    # --------------------------------------------------------
    # Group dummies
    # --------------------------------------------------------
    norm = df["bank_id"].str.replace(r"[^a-z0-9]", "", regex=True)
    df["d_market"] = norm.isin(MARKET_BANKS).astype(int)
    df["d_universal"] = norm.isin(UNIVERSAL_BANKS).astype(int)

    # --------------------------------------------------------
    # Interaction terms: restriction × risk
    # These test whether the risk effect depends on CET1, LCR, or SLR
    # --------------------------------------------------------
    for restriction in ["cet1", "lcr", "slr"]:
        df[f"{restriction}_lag1_x_{STATIC_RISK}"] = df[f"{restriction}_lag1"] * df[STATIC_RISK]
        df[f"{restriction}_lag1_x_{DYNAMIC_RISK}"] = df[f"{restriction}_lag1"] * df[DYNAMIC_RISK]

    # --------------------------------------------------------
    # Interaction terms: group × risk
    # These test whether the risk effect differs across bank groups
    # --------------------------------------------------------
    df[f"market_x_{STATIC_RISK}"] = df["d_market"] * df[STATIC_RISK]
    df[f"universals_x_{STATIC_RISK}"] = df["d_universal"] * df[STATIC_RISK]
    df[f"market_x_{DYNAMIC_RISK}"] = df["d_market"] * df[DYNAMIC_RISK]
    df[f"universals_x_{DYNAMIC_RISK}"] = df["d_universal"] * df[DYNAMIC_RISK]

    # --------------------------------------------------------
    # Show how much usable data exists for key variables
    # --------------------------------------------------------
    check = [
        "d_leverage", "d_leverage_lag1", "var_lag1", "d_var_lag1",
        "cet1_lag1", "lcr_lag1", "slr_lag1", "roa_lag1", "log_assets_lag1"
    ]

    print("\nData coverage:")
    for c in check:
        if c in df.columns:
            print(f"  {c:30s}: {df[c].notna().sum():4d} / {len(df)}")

    return df


# ============================================================
# HELPER FUNCTION FOR SIGNIFICANCE STARS
# ============================================================

def stars(t: float) -> str:
    a = abs(t)
    return "***" if a > 2.576 else "**" if a > 1.960 else "*" if a > 1.645 else ""


# ============================================================
# RUN ONE MODEL
# This takes one model from the MODELS list and estimates it
# ============================================================

def run_model(df: pd.DataFrame, model: dict):
    name = model["name"]
    indep = model["vars"]
    bank_fe = model["bank_fe"]
    time_fe = model["time_fe"]

    # Check that all needed variables exist
    missing = [v for v in indep if v not in df.columns]
    if missing:
        print(f"\n[{name}] Missing columns: {missing} - skipping.")
        return None

    # Keep only needed columns and drop rows with missing values
    data = df[["bank_id", "period_end_date", DEPENDENT_VAR] + indep].dropna()

    # Skip model if too few observations remain
    if len(data) < 30:
        print(f"\n[{name}] Too few observations ({len(data)}) - skipping.")
        return None

    # Set panel structure: bank and date
    panel = data.set_index(["bank_id", "period_end_date"])

    # y = dependent variable
    y = panel[DEPENDENT_VAR]

    # X = explanatory variables
    X = panel[indep].copy()

    # Add constant only if no bank FE and no time FE
    if not bank_fe and not time_fe:
        X["const"] = 1.0

    # Run the panel regression
    result = PanelOLS(
        y, X,
        entity_effects=bank_fe,
        time_effects=time_fe,
        drop_absorbed=True,
    ).fit(cov_type="clustered", cluster_entity=True)

    r2 = float(result.rsquared)
    fe = f"Bank={'✓' if bank_fe else '✗'}  Time={'✓' if time_fe else '✗'}"
    print(f"\n[{name}]  N={result.nobs}  R²={r2:.3f}  {fe}")

    # Print coefficients and t-stats
    for var in result.params.index:
        b = float(result.params[var])
        t = float(result.tstats[var])
        print(f"  {var:40s}  β={b:9.4f}  t={t:7.3f} {stars(t)}")

    # Save results in one row
    row = {
        "model": name,
        "bank_fe": bank_fe,
        "time_fe": time_fe,
        "n_obs": int(result.nobs),
        "n_banks": panel.index.get_level_values(0).nunique(),
        "n_quarters": panel.index.get_level_values(1).nunique(),
        "r2_within": round(r2, 4),
    }

    # Add coefficient, standard error, and t-stat for each variable
    for var in result.params.index:
        row[f"b_{var}"] = round(float(result.params[var]), 4)
        row[f"se_{var}"] = round(float(result.std_errors[var]), 4)
        row[f"t_{var}"] = round(float(result.tstats[var]), 4)

    return row


# ============================================================
# MAIN PROGRAM
# Runs the full process from start to finish
# ============================================================

def main():
    print("=" * 70)
    print("PANEL REGRESSIONS")
    print("=" * 70)

    # 1. Load the raw data
    df = load_data()

    # 2. Create the variables needed for the regressions
    df = prepare_variables(df)

    print("\n" + "=" * 70)
    print("RUNNING MODELS")
    print("=" * 70)

    # 3. Run all models in the MODELS list
    results = [r for m in MODELS if (r := run_model(df, m))]

    # 4. Stop if no model could be estimated
    if not results:
        print("\nNo models were estimated.")
        return

    # 5. Turn results into a DataFrame
    out = pd.DataFrame(results)

    # 6. Put summary info first, then coefficients
    info_cols = [c for c in out.columns if not c.startswith(("b_", "se_", "t_"))]
    coef_cols = sorted(c for c in out.columns if c.startswith(("b_", "se_", "t_")))
    out = out[info_cols + coef_cols]

    # 7. Create output folder if needed
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 8. Save results to CSV
    out.to_csv(OUTPUT_FILE, index=False)

    # 9. Print short summary
    print(f"\n{'=' * 70}")
    print(f"✓ Saved: {OUTPUT_FILE}")
    print("\nSUMMARY:")
    print("-" * 70)
    for r in results:
        fe = f"Bank={'✓' if r['bank_fe'] else '✗'}  Time={'✓' if r['time_fe'] else '✗'}"
        print(f"  {r['model']:35s}  N={r['n_obs']:5d}  R²={r['r2_within']:.3f}  {fe}")

    print("\n✓ DONE")


# Run main() when this file is executed
if __name__ == "__main__":
    main()