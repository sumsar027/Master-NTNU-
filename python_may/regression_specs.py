"""
Regression model specifications grouped by thesis table.

This file is intentionally only a list of model definitions. A non-Python reader
can read it as:
- which dependent variable is used
- which explanatory variables are included
- which time period is estimated
- whether bank and time fixed effects are included
"""

LAGGED_LEVERAGE_VAR = "d_ln_leverage_lag"


def spec(x, date_range, bank_fe=True, time_fe=True):
    """Create a standard model specification."""
    return {
        "y": "d_ln_leverage",
        "X": list(x),
        "bank_fe": bank_fe,
        "time_fe": time_fe,
        "date_range": date_range,
    }


# Main H1-H2 tables: baseline risk-leverage models and bank-type interactions.
MAIN_H1_H2 = {
    "M1_baseline": spec(
        ["d_ln_unit_var_lag", "size_lag", "roa_lag"],
        ("2010-Q1", "2025-Q4"),
    ),
    "M1_first_diff": spec(
        ["d_ln_unit_var_lag", "d_size_lag", "d_roa_lag"],
        ("2010-Q1", "2025-Q4"),
        bank_fe=False,
    ),
    "M2_bank_type": spec(
        [
            "d_ln_unit_var_lag",
            "d_ln_unit_var_x_market",
            "d_ln_unit_var_x_custody",
            "size_lag",
            "roa_lag",
        ],
        ("2010-Q1", "2025-Q4"),
    ),
    "M2_bank_type_first_diff": spec(
        [
            "d_ln_unit_var_lag",
            "d_ln_unit_var_x_market",
            "d_ln_unit_var_x_custody",
            "d_size_lag",
            "d_roa_lag",
        ],
        ("2010-Q1", "2025-Q4"),
        bank_fe=False,
    ),
}


# Main H3 table: regulatory interaction models for LCR, CET1, and SLR.
MAIN_H3 = {
    "M3_LCR": spec(
        [
            "d_ln_unit_var_lag",
            "lcr_ratio_lag",
            "d_ln_unit_var_x_lcr",
            "roa_lag",
            "size_lag",
        ],
        ("2015-Q1", "2025-Q4"),
    ),
    "M3_CET1": spec(
        [
            "d_ln_unit_var_lag",
            "cet1_ratio_lag",
            "d_ln_unit_var_x_cet1",
            "roa_lag",
            "size_lag",
        ],
        ("2015-Q1", "2025-Q4"),
    ),
    "M3_SLR": spec(
        [
            "d_ln_unit_var_lag",
            "slr_ratio_lag",
            "d_ln_unit_var_x_slr",
            "roa_lag",
            "size_lag",
        ],
        ("2015-Q1", "2025-Q4"),
    ),
}


# Robustness table: estimate the same relationships before and after COVID.
COVID_SPLIT = {
    "M1_pre_covid": spec(
        ["d_ln_unit_var_lag", "roa_lag", "size_lag"],
        ("2010-Q1", "2019-Q4"),
    ),
    "M1_post_covid": spec(
        ["d_ln_unit_var_lag", "roa_lag", "size_lag"],
        ("2020-Q1", "2025-Q4"),
    ),
    "M2_pre_covid_bank_type": spec(
        [
            "d_ln_unit_var_lag",
            "d_ln_unit_var_x_market",
            "d_ln_unit_var_x_custody",
            "size_lag",
            "roa_lag",
        ],
        ("2010-Q1", "2019-Q4"),
    ),
    "M2_post_covid_bank_type": spec(
        [
            "d_ln_unit_var_lag",
            "d_ln_unit_var_x_market",
            "d_ln_unit_var_x_custody",
            "size_lag",
            "roa_lag",
        ],
        ("2020-Q1", "2025-Q4"),
    ),
    "M3_LCR_pre_covid": spec(
        ["d_ln_unit_var_lag", "lcr_ratio_lag", "d_ln_unit_var_x_lcr", "roa_lag", "size_lag"],
        ("2015-Q1", "2019-Q4"),
    ),
    "M3_LCR_post_covid": spec(
        ["d_ln_unit_var_lag", "lcr_ratio_lag", "d_ln_unit_var_x_lcr", "roa_lag", "size_lag"],
        ("2020-Q1", "2025-Q4"),
    ),
    "M3_CET1_pre_covid": spec(
        ["d_ln_unit_var_lag", "cet1_ratio_lag", "d_ln_unit_var_x_cet1", "roa_lag", "size_lag"],
        ("2015-Q1", "2019-Q4"),
    ),
    "M3_CET1_post_covid": spec(
        ["d_ln_unit_var_lag", "cet1_ratio_lag", "d_ln_unit_var_x_cet1", "roa_lag", "size_lag"],
        ("2020-Q1", "2025-Q4"),
    ),
    "M3_SLR_pre_covid": spec(
        ["d_ln_unit_var_lag", "slr_ratio_lag", "d_ln_unit_var_x_slr", "roa_lag", "size_lag"],
        ("2015-Q1", "2019-Q4"),
    ),
    "M3_SLR_post_covid": spec(
        ["d_ln_unit_var_lag", "slr_ratio_lag", "d_ln_unit_var_x_slr", "roa_lag", "size_lag"],
        ("2020-Q1", "2025-Q4"),
    ),
}


def with_lagged_leverage(model_group):
    """Return a copy of a model group with lagged leverage inserted after UnitVaR."""
    out = {}
    for name, model in model_group.items():
        lagged = model.copy()
        lagged["X"] = list(model["X"])
        if LAGGED_LEVERAGE_VAR not in lagged["X"]:
            lagged["X"].insert(1, LAGGED_LEVERAGE_VAR)
        out[f"{name}_lagged_leverage"] = lagged
    return out


# The runner loops over this dictionary and writes one CSV per group.
TABLES = {
    "main_h1_h2": MAIN_H1_H2,
    "main_h3": MAIN_H3,
    "covid_split": COVID_SPLIT,
    "lagged_leverage_h1_h2": with_lagged_leverage(MAIN_H1_H2),
    "lagged_leverage_h3": with_lagged_leverage(MAIN_H3),
}
