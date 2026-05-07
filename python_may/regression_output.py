"""
Format panel regression results as thesis-ready CSV tables.

The regression package returns rich Python objects. This file extracts only the
information needed for the thesis tables: coefficients, t-statistics, significance
stars, observation counts, fixed effects, sample period, and R-squared.
"""

import pandas as pd

from regression_specs import LAGGED_LEVERAGE_VAR


def significance_stars(p_value: float) -> str:
    """Return conventional significance stars for a p-value."""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def format_r2(result_record: dict) -> str:
    """Use within R-squared for fixed-effects models, overall R-squared otherwise."""
    if "result" not in result_record:
        return "-"

    res = result_record["result"]
    spec = result_record.get("spec", {})
    uses_fe = spec.get("bank_fe") or spec.get("time_fe")

    if uses_fe and hasattr(res, "rsquared_within"):
        return f"{res.rsquared_within:.4f}"
    if hasattr(res, "rsquared"):
        return f"{res.rsquared:.4f}"
    return "-"


def build_results_table(results: list[dict]) -> pd.DataFrame:
    """Build one wide CSV table from a list of model result records."""
    rows = []

    # Different model columns can contain different regressors. Build one common
    # ordered list so the final CSV has a stable row layout.
    all_params = []
    for model in results:
        if "result" not in model:
            continue
        for param in model["result"].params.index:
            if param not in all_params:
                all_params.append(param)

    for param in all_params:
        coef_row = {"variable": param, "stat": "coef"}
        tstat_row = {"variable": param, "stat": "tstat"}

        for model in results:
            name = model["name"]
            if "result" not in model:
                coef_row[name] = ""
                tstat_row[name] = ""
                continue

            res = model["result"]
            # If a variable is not included in a given model, mark it with "-".
            if param not in res.params.index:
                coef_row[name] = "-"
                tstat_row[name] = ""
                continue

            # Report coefficient and t-statistic in adjacent rows, matching the
            # layout commonly used in economics regression tables.
            coef = res.params[param]
            tstat = res.tstats[param]
            stars = significance_stars(res.pvalues[param])
            coef_row[name] = f"{coef:+.4f}{stars}"
            tstat_row[name] = f"({tstat:.2f})"

        rows.extend([coef_row, tstat_row])

    rows.extend(metadata_rows(results))
    return pd.DataFrame(rows)


def metadata_rows(results: list[dict]) -> list[dict]:
    """Create the footer rows with model diagnostics and specification metadata."""
    return [
        metadata_row("Status", results, ["OK" if "result" in r else "ERROR" for r in results]),
        metadata_row("Error", results, [r.get("error", "") for r in results]),
        metadata_row("N (obs)", results, [r.get("n", "-") for r in results]),
        metadata_row("N (banks)", results, [r.get("n_banks", "-") for r in results]),
        metadata_row("R2", results, [format_r2(r) for r in results]),
        metadata_row("Bank FE", results, ["Yes" if r.get("spec", {}).get("bank_fe") else "No" for r in results]),
        metadata_row("Time FE", results, ["Yes" if r.get("spec", {}).get("time_fe") else "No" for r in results]),
        metadata_row("Period", results, [r.get("period", "-") for r in results]),
        metadata_row(
            "Lagged leverage",
            results,
            ["Yes" if LAGGED_LEVERAGE_VAR in r.get("spec", {}).get("X", []) else "No" for r in results],
        ),
    ]


def metadata_row(label: str, results: list[dict], values: list[object]) -> dict:
    """Create one metadata row."""
    row = {"variable": label, "stat": ""}
    for result, value in zip(results, values):
        row[result["name"]] = value
    return row
