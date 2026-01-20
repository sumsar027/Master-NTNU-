"""
Replicate Adrian & Shin-style panel regressions in log levels.

Pipeline:
1) Load quarterly balance-sheet panel and VaR panel.
2) Harmonize bank identifiers, merge on (bank_id, period_end_date), and construct risk measures:
   - VaR level: var_95_level = var_95 if available, else var_95_approx
   - UnitVaR: unit_var_95 = var_95_level / total_assets
   - Logs: log_unit_var_95, log_var_95_level
3) Construct balance-sheet log variables:
   - log_leverage = log(total_assets / common_equity_total)
   - log_assets   = log(total_assets)
   - log_equity   = log(common_equity_total)
4) Use a common estimation sample across all models (comparability).
5) Estimate:
   - FE (main): bank fixed effects (optional time FE toggle), SE clustered by bank
   - Pooled (benchmark): no bank FE, SE clustered by bank
Outputs:
- Prints sanity checks + results table
- Saves results to output/fe_results.csv (and a legacy copy to output/Model_1_result.csv)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, PooledOLS


# -----------------------
# Configuration (results)
# -----------------------

# Main specification: bank fixed effects. Set TIME_FE=True only for robustness checks.
TIME_FE = False

# Risk measures used in the regressions:
# - Leverage: log(UnitVaR) where UnitVaR = VaR / Assets.
# - Assets/Equity: log(VaR level) to avoid using a regressor that contains Assets in its definition.
RISK_REGRESSOR_LEVERAGE = "log_unit_var_95"
RISK_REGRESSOR_LEVEL = "log_var_95_level"

BASE_PATH = Path("output/merged_quarterly_balanced.csv")
VAR_PATH = Path("output/merged_with_var_95_approx.csv")
OUTPUT_PATH = Path("output/fe_results.csv")
LEGACY_OUTPUT_PATH = Path("output/Model_1_result.csv")

# Harmonize bank identifiers across sources to ensure a correct merge on (bank_id, period_end_date).
BANK_ID_MAP: dict[str, str] = {
    "bankofamerica": "bank_of_america",
    "citigroup": "citibank",
    "wellsfargo": "wells_fargo",
}

BASE_REQUIRED_COLS = ["bank_id", "period_end_date", "total_assets", "common_equity_total"]
VAR_REQUIRED_COLS = ["bank_id", "period_end_date"]

RESULT_COLUMNS = [
    "estimator",
    "model_name",
    "dep_var",
    "regressor_name",
    "const",
    "beta",
    "se",
    "t",
    "pvalue",
    "r2",
    "n_obs",
    "n_banks",
    "n_periods",
]


# -----------------------
# Small utilities
# -----------------------

def die(msg: str) -> None:
    print(msg)
    sys.exit(1)


def require_columns(df: pd.DataFrame, cols: list[str], *, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        die(f"{name}: missing required columns: {', '.join(missing)}")


def safe_log(x: pd.Series) -> pd.Series:
    """Log-transform with x>0 guard. Non-positive values become NaN."""
    x_num = pd.to_numeric(x, errors="coerce")
    return pd.Series(np.where(x_num > 0, np.log(x_num), np.nan), index=x.index)


def normalize_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal harmonization of base (balance sheet) file:
    - accept either 'bank_id' or 'bank'
    - allow total_assets to come from total_assets_2 if needed
    """
    df = df.copy()

    if "bank_id" not in df.columns and "bank" in df.columns:
        df = df.rename(columns={"bank": "bank_id"})

    if "total_assets_2" in df.columns:
        if "total_assets" not in df.columns:
            df["total_assets"] = df["total_assets_2"]
        else:
            # Use total_assets_2 only where total_assets is missing
            df["total_assets"] = df["total_assets"].fillna(df["total_assets_2"])

    return df


def normalize_var_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal harmonization of VaR file:
    - accept either 'bank_id' or 'bank'
    """
    df = df.copy()
    if "bank_id" not in df.columns and "bank" in df.columns:
        df = df.rename(columns={"bank": "bank_id"})
    return df


# -----------------------
# Data loading + merge
# -----------------------

def load_data(base_path: Path, var_path: Path) -> pd.DataFrame:
    """Load inputs, harmonize IDs, construct risk measures, and inner-merge to a common bank-quarter sample."""
    if not base_path.exists():
        die(f"Missing data file: {base_path}")
    if not var_path.exists():
        die(f"Missing data file: {var_path}")

    base_df = normalize_base_columns(pd.read_csv(base_path))
    var_df = normalize_var_columns(pd.read_csv(var_path))

    require_columns(base_df, BASE_REQUIRED_COLS, name="base_df")
    require_columns(var_df, VAR_REQUIRED_COLS, name="var_df")

    if "var_95" not in var_df.columns and "var_95_approx" not in var_df.columns:
        die("var_df: need `var_95` or `var_95_approx` to construct VaR series.")

    # Ensure consistent date type for merging and panel indexing
    base_df["period_end_date"] = pd.to_datetime(base_df["period_end_date"])
    var_df["period_end_date"] = pd.to_datetime(var_df["period_end_date"])

    # Sanity check: show bank identifiers before/after harmonization to verify merge keys
    base_banks_before = sorted(base_df["bank_id"].dropna().astype(str).unique().tolist())
    var_banks_before = sorted(var_df["bank_id"].dropna().astype(str).unique().tolist())
    print(f"base banks before map ({len(base_banks_before)}): {base_banks_before}")
    print(f"var banks before map  ({len(var_banks_before)}): {var_banks_before}")

    base_df["bank_id"] = base_df["bank_id"].astype("string").str.strip().replace(BANK_ID_MAP)
    var_df["bank_id"] = var_df["bank_id"].astype("string").str.strip().replace(BANK_ID_MAP)

    base_banks_after = sorted(base_df["bank_id"].dropna().astype(str).unique().tolist())
    var_banks_after = sorted(var_df["bank_id"].dropna().astype(str).unique().tolist())
    print(f"base banks after map  ({len(base_banks_after)}): {base_banks_after}")
    print(f"var banks after map   ({len(var_banks_after)}): {var_banks_after}")

    # Guardrail: duplicates on merge keys would silently duplicate rows after merging
    for name, frame in [("base_df", base_df), ("var_df", var_df)]:
        n_dups = int(frame.duplicated(subset=["bank_id", "period_end_date"]).sum())
        if n_dups:
            die(f"ERROR: {name} has {n_dups} duplicate (bank_id, period_end_date) rows.")

    # Coerce numeric columns used in construction
    base_df["total_assets"] = pd.to_numeric(base_df["total_assets"], errors="coerce")
    base_df["common_equity_total"] = pd.to_numeric(base_df["common_equity_total"], errors="coerce")
    if "var_95" in var_df.columns:
        var_df["var_95"] = pd.to_numeric(var_df["var_95"], errors="coerce")
    if "var_95_approx" in var_df.columns:
        var_df["var_95_approx"] = pd.to_numeric(var_df["var_95_approx"], errors="coerce")

    if base_df["total_assets"].notna().sum() == 0:
        die("base_df: no numeric values found for `total_assets` (after normalization).")

    # Construct VaR level series (prefer true var_95; fall back to proxy var_95_approx)
    if "var_95" in var_df.columns:
        var_df["var_95_level"] = var_df["var_95"].fillna(var_df.get("var_95_approx"))
    else:
        var_df["var_95_level"] = var_df["var_95_approx"]

    var_df = var_df[["bank_id", "period_end_date", "var_95_level"]]

    # Sanity check: confirm the merge keeps the intended bank-quarter sample
    df = base_df.merge(var_df, on=["bank_id", "period_end_date"], how="inner")
    print(f"Rows: base={len(base_df)}, var={len(var_df)}, after_inner_merge={len(df)}")
    if df.empty:
        die("No matched rows after merging on (bank_id, period_end_date).")

    # UnitVaR definition: VaR per dollar of assets (requires both > 0)
    invalid = (
        df["total_assets"].isna() | (df["total_assets"] <= 0) |
        df["var_95_level"].isna() | (df["var_95_level"] <= 0)
    )
    df["unit_var_95"] = np.where(invalid, np.nan, df["var_95_level"] / df["total_assets"])

    # Logs used as regressors
    df["log_var_95_level"] = np.where(df["var_95_level"] > 0, np.log(df["var_95_level"]), np.nan)
    df["log_unit_var_95"] = np.where(df["unit_var_95"] > 0, np.log(df["unit_var_95"]), np.nan)

    # Compact sanity checks for constructed risk measures
    unit_nans = int(df["unit_var_95"].isna().sum())
    log_unit_nans = int(df["log_unit_var_95"].isna().sum())
    print(f"UnitVaR sanity: unit_var_95 NaNs={unit_nans}, log_unit_var_95 NaNs={log_unit_nans}")

    unitvar = df["unit_var_95"].dropna()
    if len(unitvar):
        print(
            "unit_var_95 summary (min/median/max): "
            f"{unitvar.min():.6g} / {unitvar.median():.6g} / {unitvar.max():.6g}"
        )

    return df


# -----------------------
# Variable construction
# -----------------------

def print_nonpositive_log_inputs(df: pd.DataFrame) -> None:
    """Report non-positive values that will turn into NaN under log transforms."""
    leverage = df["total_assets"] / df["common_equity_total"]

    def count_nonpos(x: pd.Series) -> int:
        x_num = pd.to_numeric(x, errors="coerce")
        x_num = x_num[np.isfinite(x_num)]
        return int((x_num <= 0).sum())

    lev_num = pd.to_numeric(leverage, errors="coerce")
    lev_num = lev_num[np.isfinite(lev_num)]
    lev_nonpos = int((lev_num <= 0).sum())

    print(
        "Non-positive (<=0) log inputs: "
        f"leverage={lev_nonpos}, "
        f"total_assets={count_nonpos(df['total_assets'])}, "
        f"common_equity_total={count_nonpos(df['common_equity_total'])}, "
        f"unit_var_95={count_nonpos(df['unit_var_95'])}"
    )


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Construct leverage and log-level dependent variables."""
    df = df.sort_values(["bank_id", "period_end_date"]).copy()

    df["leverage"] = df["total_assets"] / df["common_equity_total"]
    print_nonpositive_log_inputs(df)

    df["log_leverage"] = safe_log(df["leverage"])
    df["log_assets"] = safe_log(df["total_assets"])
    df["log_equity"] = safe_log(df["common_equity_total"])

    return df


def build_common_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use a common estimation sample across all regressions to make coefficients comparable across models.
    """
    required = [
        "bank_id",
        "period_end_date",
        "log_leverage",
        "log_assets",
        "log_equity",
        "log_unit_var_95",
        "log_var_95_level",
    ]
    n_before = len(df)
    sample = df[required].dropna().copy()

    dropped = n_before - len(sample)
    n_banks = int(sample["bank_id"].nunique())
    n_periods = int(sample["period_end_date"].nunique())

    print(
        "Common estimation sample: "
        f"kept={len(sample)} rows, dropped={dropped}, banks={n_banks}, periods={n_periods}"
    )
    return sample


# -----------------------
# Estimation
# -----------------------

def run_fe(df: pd.DataFrame, dep_var: str, regressor: str, model_name: str) -> dict[str, object]:
    """Fixed-effects model with bank FE (and optional time FE), SE clustered at bank level."""
    model_df = df[["bank_id", "period_end_date", dep_var, regressor]].copy()
    # Common sample should already remove missing values; keep an assert-style guard.
    if model_df.isna().any().any():
        die(f"ERROR: {model_name} has missing values after common-sample construction.")

    model_df = model_df.set_index(["bank_id", "period_end_date"])
    y = model_df[[dep_var]]
    X = model_df[[regressor]]

    n_obs = int(len(model_df))
    n_banks = int(model_df.index.get_level_values(0).nunique())
    n_periods = int(model_df.index.get_level_values(1).nunique())

    try:
        model = PanelOLS(y, X, entity_effects=True, time_effects=TIME_FE)
        res = model.fit(cov_type="clustered", cluster_entity=True)
    except ValueError as e:
        # Rare edge case: regressor can be absorbed by fixed effects under certain toggles
        die(f"FE estimation failed for {model_name}: {e}")

    const = np.nan
    if hasattr(res, "estimated_effects") and res.estimated_effects is not None:
        # PanelOLS does not estimate a standalone intercept with fixed effects. We report an
        # implied constant as the average of the estimated fixed effects (entity, and time if enabled).
        try:
            const = float(res.estimated_effects.iloc[:, 0].mean())
        except Exception:
            const = np.nan

    return {
        "estimator": "FE",
        "model_name": model_name,
        "dep_var": dep_var,
        "regressor_name": regressor,
        "const": float(const) if np.isfinite(const) else np.nan,
        "beta": float(res.params[regressor]),
        "se": float(res.std_errors[regressor]),
        "t": float(res.tstats[regressor]),
        "pvalue": float(res.pvalues[regressor]),
        "r2": float(getattr(res, "rsquared_within", res.rsquared)),
        "n_obs": int(res.nobs),
        "n_banks": n_banks,
        "n_periods": n_periods,
    }


def run_pooled(df: pd.DataFrame, dep_var: str, regressor: str, model_name: str) -> dict[str, object]:
    """Pooled benchmark (no bank FE), reported as a descriptive comparison; SE clustered at bank level."""
    model_df = df[["bank_id", "period_end_date", dep_var, regressor]].copy()
    if model_df.isna().any().any():
        die(f"ERROR: pooled {model_name} has missing values after common-sample construction.")

    model_df = model_df.set_index(["bank_id", "period_end_date"])
    y = model_df[[dep_var]]
    X = model_df[[regressor]].copy()
    X.insert(0, "const", 1.0)

    n_obs = int(len(model_df))
    n_banks = int(model_df.index.get_level_values(0).nunique())
    n_periods = int(model_df.index.get_level_values(1).nunique())

    model = PooledOLS(y, X)
    res = model.fit(cov_type="clustered", cluster_entity=True)

    return {
        "estimator": "Pooled",
        "model_name": model_name,
        "dep_var": dep_var,
        "regressor_name": regressor,
        "const": float(res.params["const"]),
        "beta": float(res.params[regressor]),
        "se": float(res.std_errors[regressor]),
        "t": float(res.tstats[regressor]),
        "pvalue": float(res.pvalues[regressor]),
        "r2": float(res.rsquared),
        "n_obs": int(res.nobs),
        "n_banks": n_banks,
        "n_periods": n_periods,
    }


def build_results(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate the three specifications under FE (main) and pooled (benchmark)."""
    specs = [
        ("Model 1", "log_leverage", RISK_REGRESSOR_LEVERAGE),
        ("Model 2", "log_assets", RISK_REGRESSOR_LEVEL),
        ("Model 3", "log_equity", RISK_REGRESSOR_LEVEL),
    ]

    fe_rows = [run_fe(df, dep, reg, name) for name, dep, reg in specs]
    pooled_rows = [run_pooled(df, dep, reg, name) for name, dep, reg in specs]

    return pd.DataFrame(fe_rows + pooled_rows, columns=RESULT_COLUMNS)


# -----------------------
# Main
# -----------------------

def main() -> None:
    df = load_data(BASE_PATH, VAR_PATH)
    df = prepare_data(df)
    df_common = build_common_sample(df)

    results = build_results(df_common)
    print(results.to_string(index=False, float_format="{:.4f}".format))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    results.to_csv(LEGACY_OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
