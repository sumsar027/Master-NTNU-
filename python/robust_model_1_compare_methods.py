"""
Compare VaR 99% estimation methods via panel regressions.
1) Read balance sheet data and VaR 95% data
2) Clean and standardize bank names so the merge works correctly
   (e.g. "bank_of_america" and "bankofamerica" become the same)
3) Convert VaR 95% into VaR 99% using two methods:
   - Gaussian scaling (× 1.41)
   - Simple scaling (× 2.0)
4) Run three regressions for each VaR method:
   - Leverage on VaR
   - Assets on VaR
   - Equity on VaR
   Each regression is estimated both:
   - With bank fixed effects
   - As a pooled regression
5) Save all regression results to:
   output/tables/panel_results_var99_gauss_vs_x2.csv
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, PooledOLS

# -------------------------------------------------------------------
# File locations
# -------------------------------------------------------------------
# Balance sheet data (assets and equity)
BALANCE_FILE = Path("output/data/merged_quarterly_balanced.csv")

# VaR 95% data (used to construct VaR 99%)
VAR95_FILE = Path("output/data/merged_with_var_99_dual_methods.csv")

# Output file with regression results
OUT_FILE = Path("output/tables/panel_results_var99_gauss_vs_x2.csv")

# -------------------------------------------------------------------
# VaR scaling factors
# -------------------------------------------------------------------
# Gaussian conversion from 95% to 99%
GAUSS = 1.41421356237

# Alternative simple scaling
X2 = 2.0

# -------------------------------------------------------------------
# Column names used in the data
# (hardcoded to keep the script simple and transparent)
# -------------------------------------------------------------------
ASSETS_COL = "total_assets_2"
EQUITY_COL = "common_equity_total"
VAR95_COL = "var_95"

# -------------------------------------------------------------------
# Bank name harmonization
# -------------------------------------------------------------------
# Some banks use different names across files.
# This dictionary forces them to match.
BANK_ID_ALIASES = {"citibank": "citigroup"}


def normalize_bank_id(value) -> str:
    """
    Clean bank names so they match across datasets.
    - Lowercase
    - Remove special characters
    - Apply known name fixes
    """
    cleaned = re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())
    return BANK_ID_ALIASES.get(cleaned, cleaned)


def run_reg(y: pd.Series, x: pd.DataFrame, *, fe: bool):
    """
    Run a regression.

    If fe=True:
      - Bank fixed effects regression
    If fe=False:
      - Pooled regression (no fixed effects)

    Returns:
      - Regression results
      - R-squared value
    """
    if fe:
        res = PanelOLS(y, x, entity_effects=True).fit(
            cov_type="clustered", cluster_entity=True
        )
        return res, float(res.rsquared_within)

    # Pooled regression needs an explicit constant
    x_pool = x.copy()
    x_pool.insert(0, "const", 1.0)
    res = PooledOLS(y, x_pool).fit(
        cov_type="clustered", cluster_entity=True
    )
    return res, float(res.rsquared)


def main() -> None:
    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------
    base = pd.read_csv(BALANCE_FILE)
    var = pd.read_csv(VAR95_FILE)

    # ---------------------------------------------------------------
    # Ensure both datasets use the same bank id column name
    # ---------------------------------------------------------------
    if "bank" in base.columns and "bank_id" not in base.columns:
        base = base.rename(columns={"bank": "bank_id"})
    if "bank" in var.columns and "bank_id" not in var.columns:
        var = var.rename(columns={"bank": "bank_id"})

    if "bank_id" not in base.columns or "bank_id" not in var.columns:
        raise ValueError("Missing 'bank_id' in one of the files.")
    if "period_end_date" not in base.columns or "period_end_date" not in var.columns:
        raise ValueError("Missing 'period_end_date' in one of the files.")

    # ---------------------------------------------------------------
    # Normalize bank names and convert dates
    # ---------------------------------------------------------------
    base["bank_id"] = base["bank_id"].map(normalize_bank_id)
    var["bank_id"] = var["bank_id"].map(normalize_bank_id)

    base["period_end_date"] = pd.to_datetime(base["period_end_date"])
    var["period_end_date"] = pd.to_datetime(var["period_end_date"])

    # ---------------------------------------------------------------
    # Check required columns exist
    # ---------------------------------------------------------------
    if ASSETS_COL not in base.columns:
        raise ValueError(f"Balance file is missing '{ASSETS_COL}'.")
    if EQUITY_COL not in base.columns:
        raise ValueError(f"Balance file is missing '{EQUITY_COL}'.")
    if VAR95_COL not in var.columns:
        raise ValueError(f"VaR file is missing '{VAR95_COL}'.")

    # ---------------------------------------------------------------
    # Construct VaR 99% using two methods
    # ---------------------------------------------------------------
    var95_num = pd.to_numeric(var[VAR95_COL], errors="coerce")
    var = var.assign(
        var_99_gauss=var95_num * GAUSS,
        var_99_x2=var95_num * X2
    )

    # ---------------------------------------------------------------
    # Merge balance sheet and VaR data
    # ---------------------------------------------------------------
    df = base.merge(
        var[["bank_id", "period_end_date", "var_99_gauss", "var_99_x2"]],
        on=["bank_id", "period_end_date"],
        how="inner",
    )

    # Convert assets and equity to numeric
    df["assets"] = pd.to_numeric(df[ASSETS_COL], errors="coerce")
    df["equity"] = pd.to_numeric(df[EQUITY_COL], errors="coerce")

    # Keep only valid observations
    df = df.dropna(subset=["assets", "equity"])
    df = df[(df["assets"] > 0) & (df["equity"] > 0)]

    if df.empty:
        raise RuntimeError("No valid observations after filtering.")

    # ---------------------------------------------------------------
    # Create dependent variables
    # ---------------------------------------------------------------
    df["log_assets"] = np.log(df["assets"])
    df["log_equity"] = np.log(df["equity"])
    df["leverage"] = df["assets"] / df["equity"]

    results = []

    # ---------------------------------------------------------------
    # Run regressions for each VaR method
    # ---------------------------------------------------------------
    for method, vcol in [("gauss", "var_99_gauss"), ("x2", "var_99_x2")]:
        d = df[[
            "bank_id",
            "period_end_date",
            "leverage",
            "log_assets",
            "log_equity",
            vcol
        ]].copy()

        d["var99"] = pd.to_numeric(d[vcol], errors="coerce")
        d = d[d["var99"] > 0]
        d["log_var"] = np.log(d["var99"])

        d = d.replace([np.inf, -np.inf], np.nan).dropna()
        d = d.set_index(["bank_id", "period_end_date"]).sort_index()

        print(f"{method.upper()}: rows={len(d)}, banks={d.index.get_level_values(0).nunique()}")

        specs = [
            ("Leverage ~ VaR", "leverage"),
            ("Assets ~ VaR", "log_assets"),
            ("Equity ~ VaR", "log_equity"),
        ]

        for model_name, ycol in specs:
            y = d[ycol]
            X = d[["log_var"]]

            for fe_flag, est_name in [(True, "Bank FE"), (False, "Pooled")]:
                res, r2 = run_reg(y, X, fe=fe_flag)

                results.append({
                    "var_method": method,
                    "model": model_name,
                    "estimator": est_name,
                    "coefficient": float(res.params["log_var"]),
                    "std_error": float(res.std_errors["log_var"]),
                    "t_stat": float(res.tstats["log_var"]),
                    "p_value": float(res.pvalues["log_var"]),
                    "n_obs": int(res.nobs),
                    "n_entities": int(d.index.get_level_values(0).nunique()),
                    "r_squared": r2,
                })

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------
    out = pd.DataFrame(results)
    OUT_FILE.parent.mkdir(exist_ok=True)
    out.to_csv(OUT_FILE, index=False)

    print(f"Saved results to {OUT_FILE}")


if __name__ == "__main__":
    main()
