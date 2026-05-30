"""
Build analysis variables for the panel regressions.

The raw panel contains levels such as assets, equity, VaR, and regulatory ratios.
The regressions use transformed variables: leverage, UnitVaR, log changes, lags,
and interaction terms. Keeping those transformations here makes the regression
script easier to read.
"""

import numpy as np
import pandas as pd


MARKET_BANKS = ["goldmansachs", "morganstanley"]
CUSTODY_BANKS = ["statestreet", "bny"]


def load_regression_panel(data_path: str) -> pd.DataFrame:
    """Load the panel dataset and construct all regression variables."""
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    df = (
        df.sort_values(["bank", "date"])
        .drop_duplicates(subset=["bank", "date"], keep="first")
        .reset_index(drop=True)
    )

    # Convert relevant columns to numeric values. Invalid entries become missing
    # values and are later dropped model by model.
    numeric_cols = [
        "total_assets",
        "total_equity",
        "total_var",
        "roa",
        "lcr_ratio",
        "cet1_ratio",
        "slr_ratio",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Core theoretical variables:
    # leverage = assets / equity
    # unit_var = VaR scaled by total assets
    # size = log total assets
    # Logs are clipped only to avoid undefined log values in missing/zero cases.
    df["leverage"] = df["total_assets"] / df["total_equity"]
    df["ln_leverage"] = np.log(df["leverage"].clip(lower=1e-12))
    df["size"] = np.log(df["total_assets"].clip(lower=1e-12))
    df["unit_var"] = df["total_var"] / df["total_assets"]
    df["ln_unit_var"] = np.log(df["unit_var"].clip(lower=1e-12))

    # Bank-type indicators. Universal/commercial banks are the omitted reference
    # category because they are neither market nor custody banks here.
    df["market_dummy"] = df["bank"].isin(MARKET_BANKS).astype(int)
    df["custody_dummy"] = df["bank"].isin(CUSTODY_BANKS).astype(int)

    grouped = df.groupby("bank", group_keys=False)

    # First differences capture quarterly changes within each bank.
    diff_vars = {
        "ln_leverage": "d_ln_leverage",
        "ln_unit_var": "d_ln_unit_var",
        "size": "d_size",
        "roa": "d_roa",
    }
    for src, new in diff_vars.items():
        df[new] = grouped[src].diff()

    # Lagged variables use the previous quarter. This is how the regressions
    # relate earlier changes in risk to later changes in leverage.
    lag_vars = [
        "roa",
        "size",
        "ln_unit_var",
        "d_ln_unit_var",
        "d_ln_leverage",
        "d_size",
        "d_roa",
        "lcr_ratio",
        "cet1_ratio",
        "slr_ratio",
    ]
    for col in lag_vars:
        df[f"{col}_lag"] = grouped[col].shift(1)

    # Regulatory ratios are mean-centered before constructing interactions.
    # This is an algebraically equivalent reparameterization that reduces
    # mechanical multicollinearity in interaction models.
    regulatory_ratios = ["lcr_ratio_lag", "cet1_ratio_lag", "slr_ratio_lag"]
    for col in regulatory_ratios:
        df[f"{col}_centered"] = df[col] - df[col].mean()

    # Interaction terms test whether the UnitVaR-leverage relationship differs
    # by bank type or regulatory ratio.
    interactions = {
        "d_ln_unit_var_x_market": ("d_ln_unit_var_lag", "market_dummy"),
        "d_ln_unit_var_x_custody": ("d_ln_unit_var_lag", "custody_dummy"),
        "d_ln_unit_var_x_lcr": ("d_ln_unit_var_lag", "lcr_ratio_lag_centered"),
        "d_ln_unit_var_x_cet1": ("d_ln_unit_var_lag", "cet1_ratio_lag_centered"),
        "d_ln_unit_var_x_slr": ("d_ln_unit_var_lag", "slr_ratio_lag_centered"),
    }
    for new, (left, right) in interactions.items():
        df[new] = df[left] * df[right]

    return df.set_index(["bank", "date"])


def quarter_to_dates(quarter: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Convert a quarter label such as '2015-Q1' to start and end dates."""
    year, q = str(quarter).upper().split("-Q")
    q = int(q)

    starts = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
    ends = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}

    return pd.Timestamp(f"{year}-{starts[q]}"), pd.Timestamp(f"{year}-{ends[q]}")
