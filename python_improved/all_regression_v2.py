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

5. BANKS_INCLUDE
Filter which banks to include in the regressions.
Leave the list empty to include all banks.
"""

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from pathlib import Path

# ============================================================================
# SETTINGS
# ============================================================================

RAW_VARS = {
    "cet1": "capital_adequacy_core_tier_1",
    "lcr":  "liquidity_coverage_ratio_basel_3",
    "slr":  "leverage_ratio_basel_3",
    "div":  "dividend_payout_ratio",
    "roa":  "return_on_average_total_assets_income_before_discontinued_operations_extraordinary_items_ttm",
    # log_assets if you want size control, but be aware of multicollinearity with leverage
     # New variable example: "short_name": "column_name_in_csv"
}

# Interaction terms – product of two already-computed columns
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

# Which banks to include in the regressions.
# Leave empty to include all banks.
# Example: ["jpmorgan", "bankofamerica", "citigroup"]
#BANKS_INCLUDE = ["jpmorgan", "bankofamerica", "citigroup", 
#    "bny", "statestreet", "goldmansachs", "morganstanley"]

BANKS_INCLUDE = [] # for all banks

OUTPUT_FILE = Path("output/tables/panel_regression_results.csv")

# ============================================================================
# MODELS
# ============================================================================

MODELS = [
    {
        "name":    "M1_Basis_VaR",
        "vars":    ["log_var", "log_assets", "roa_lag1"],
        "bank_fe": True,
        "time_fe": True,
    },
    {
        "name":    "M2_CET1_Interaction",
        "vars":    ["log_var", "cet1_lag1", "cet1_x_var", "log_assets", "roa_lag1"],
        "bank_fe": True,
        "time_fe": True,
    },
    {
        "name":    "M3_LCR_Interaction",
        "vars":    ["log_var", "lcr_lag1", "lcr_x_var", "log_assets", "roa_lag1"],
        "bank_fe": True,
        "time_fe": True,
    },
    {
        "name":    "M4_SLR_Interaction",
        "vars":    ["log_var", "slr_lag1", "slr_x_var", "log_assets", "roa_lag1"],
        "bank_fe": True,
        "time_fe": True,
    },
    {
        "name":    "M5_Dividend_Interaction",
        "vars":    ["log_var", "div_lag1", "div_x_var", "log_assets", "roa_lag1"],
        "bank_fe": True,
        "time_fe": True,
    },
    {
        "name":    "M6_BankType_Interactions",
        "vars":    ["log_var", "d_market", "d_custody",
                    "market_x_var", "custody_x_var", "log_assets", "roa_lag1"],
        "bank_fe": True,
        "time_fe": True,
    },
    {
        "name":    "M7_All_Interactions",
        "vars":    ["log_var", "cet1_lag1", "lcr_lag1", "slr_lag1", "div_lag1",
                    "cet1_x_var", "lcr_x_var", "slr_x_var", "div_x_var", "log_assets", "roa_lag1"],
        "bank_fe": True,
        "time_fe": True,
    },
]

# ============================================================================
# LOAD DATA
# ============================================================================

def load_data() -> pd.DataFrame:
    bs  = pd.read_csv("data/processed/balance_sheet_panel_balanced.csv")
    var = pd.read_csv("output/data/var_99.csv")

    for d in [bs, var]:
        id_col   = "bank_id" if "bank_id" in d.columns else "bank"
        date_col = "period_end_date" if "period_end_date" in d.columns else "date"
        d.rename(columns={id_col: "bank_id", date_col: "period_end_date"}, inplace=True)
        d["bank_id"]         = d["bank_id"].str.strip().str.lower()
        d["period_end_date"] = pd.to_datetime(d["period_end_date"])

    var["var_99_level"] = var["var_99_gaussian"]
    df = bs.merge(var[["bank_id", "period_end_date", "var_99_level"]],
                  on=["bank_id", "period_end_date"], how="inner")

    # Filter banks if BANKS_INCLUDE is set
    if BANKS_INCLUDE:
        keep = [b.strip().lower() for b in BANKS_INCLUDE]
        df   = df[df["bank_id"].isin(keep)]
        print(f"Bankfilter aktiv: {keep}")

    print(f"Data lastet: {len(df)} obs., {df['bank_id'].nunique()} banker  "
          f"({df['period_end_date'].min().date()} – {df['period_end_date'].max().date()})")
    print(f"Banker inkludert: {sorted(df['bank_id'].unique())}")

    return df.sort_values(["bank_id", "period_end_date"]).reset_index(drop=True)


# ============================================================================
# BUILD VARIABLES
# ============================================================================

def to_decimal(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s / 100.0 if s.dropna().median() > 1.0 else s


def prepare_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["assets"]       = pd.to_numeric(df["total_assets"],        errors="coerce")
    df["equity"]       = pd.to_numeric(df["common_equity_total"], errors="coerce")
    valid              = (df["assets"] > 0) & (df["equity"] > 0)
    df["leverage"]     = np.where(valid, df["assets"] / df["equity"], np.nan)
    df["log_leverage"] = np.log(df["leverage"].where(df["leverage"] > 0))
    df["log_var"]      = np.log(df["var_99_level"].where(df["var_99_level"] > 0))
    df["log_assets"]   = np.log(df["assets"].where(df["assets"] > 0))

    for name, col in RAW_VARS.items():
        if col not in df.columns:
            print(f"Kolonne ikke funnet: {col}")
            continue
        df[name]           = to_decimal(df[col])
        df[f"{name}_lag1"] = df.groupby("bank_id")[name].shift(1)

    for name, (a, b) in INTERACTIONS.items():
        if a in df.columns and b in df.columns:
            df[name] = df[a] * df[b]
        else:
            print(f"Kan ikke lage interaksjon {name}: mangler {a} eller {b}")

    norm                = df["bank_id"].str.replace(r"[^a-z0-9]", "", regex=True)
    df["d_market"]      = norm.isin(MARKET_BANKS).astype(int)
    df["d_custody"]     = norm.isin(CUSTODY_BANKS).astype(int)
    df["market_x_var"]  = df["d_market"]  * df["log_var"]
    df["custody_x_var"] = df["d_custody"] * df["log_var"]

    check = ["log_leverage", "log_var", "roa_lag1",
             "cet1_lag1", "lcr_lag1", "slr_lag1", "div_lag1"]
    print("\nDatadekning:")
    for c in check:
        if c in df.columns:
            print(f"  {c:25s}: {df[c].notna().sum():4d} / {len(df)}")

    return df


# ============================================================================
# REGRESSION
# ============================================================================

def stars(t: float) -> str:
    a = abs(t)
    return "***" if a > 2.576 else "**" if a > 1.960 else "*" if a > 1.645 else ""


def run_model(df: pd.DataFrame, model: dict) -> dict | None:
    name    = model["name"]
    indep   = model["vars"]
    bank_fe = model["bank_fe"]
    time_fe = model["time_fe"]

    missing = [v for v in indep if v not in df.columns]
    if missing:
        print(f"  [{name}] Mangler kolonner {missing} – hopper over.")
        return None

    data = df[["bank_id", "period_end_date", "log_leverage"] + indep].dropna()

    if len(data) < 30:
        print(f"\n  [{name}] For få obs. ({len(data)}) – hopper over.")
        return None

    panel  = data.set_index(["bank_id", "period_end_date"])
    result = PanelOLS(panel[["log_leverage"]], panel[indep],
                      entity_effects=bank_fe, time_effects=time_fe,
                      drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)

    r2 = getattr(result, "rsquared_within", result.rsquared_inclusive)
    fe = f"Bank={'✓' if bank_fe else '✗'}  Tid={'✓' if time_fe else '✗'}"

    print(f"\n  [{name}]  N={result.nobs}  R²={r2:.3f}  {fe}")
    for var in result.params.index:
        b = float(result.params[var])
        t = float(result.tstats[var])
        print(f"    {var:35s}  β={b:8.4f}  t={t:7.3f} {stars(t)}")

    row = {
        "model":      name,
        "bank_fe":    bank_fe,
        "time_fe":    time_fe,
        "n_obs":      int(result.nobs),
        "n_banks":    panel.index.get_level_values(0).nunique(),
        "n_quarters": panel.index.get_level_values(1).nunique(),
        "r2_within":  round(r2, 4),
    }
    for var in result.params.index:
        row[f"b_{var}"]  = round(float(result.params[var]),     4)
        row[f"se_{var}"] = round(float(result.std_errors[var]), 4)
        row[f"t_{var}"]  = round(float(result.tstats[var]),     4)

    return row


# ============================================================================
# MAIN
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

    results = [r for m in MODELS if (r := run_model(df, m))]

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