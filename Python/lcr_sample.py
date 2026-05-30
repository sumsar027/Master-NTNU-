"""
Check whether the baseline result changes on the LCR regression sample.

This is a small validation script for the LCR data-coverage comment. The
original baseline model and the LCR interaction model are already reported in
the main regression tables. This script only estimates the missing bridge:
the baseline model on the exact observations used by the LCR model.

Outputs:
- output/tables/lcr_matched_sample_regression.csv
- output/tables/lcr_sample_by_bank.csv
- output/tables/lcr_sample_by_quarter.csv
- output/tables/lcr_raw_coverage_by_quarter.csv
"""

import sys
from pathlib import Path

import pandas as pd
from linearmodels.panel import PanelOLS


ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = ROOT / "src" / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from regression_data import load_regression_panel, quarter_to_dates  # noqa: E402
from regression_output import build_results_table  # noqa: E402
from regression_specs import MAIN_H1_H2, MAIN_H3  # noqa: E402
from run_panel_regressions import build_formula  # noqa: E402


DATA_PATH = ROOT / "data" / "processed" / "panel.csv"
OUTPUT_DIR = ROOT / "output" / "tables"


def get_model_sample(panel: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Return the complete-case sample used by one model specification."""
    y = spec["y"]
    x_vars = spec["X"]
    data = panel[[y, *x_vars]].copy()

    start, _ = quarter_to_dates(spec["date_range"][0])
    _, end = quarter_to_dates(spec["date_range"][1])
    dates = data.index.get_level_values("date")

    return data[(dates >= start) & (dates <= end)].dropna()


def estimate(name: str, data: pd.DataFrame, spec: dict) -> dict:
    """Estimate one fixed-effects panel model and return table metadata."""
    formula = build_formula(spec["y"], spec["X"], spec)
    result = PanelOLS.from_formula(formula, data=data).fit(
        cov_type="clustered",
        cluster_entity=True,
    )
    dates = data.index.get_level_values("date")

    return {
        "name": name,
        "spec": spec,
        "result": result,
        "n": len(data),
        "n_banks": data.index.get_level_values("bank").nunique(),
        "period": f"{dates.min().date()} - {dates.max().date()}",
    }


def add_quarter_column(data: pd.DataFrame) -> pd.DataFrame:
    """Move the panel index to columns and add a readable quarter label."""
    out = data.reset_index()
    out["quarter"] = out["date"].dt.to_period("Q").astype(str)
    return out


def save_raw_lcr_coverage() -> None:
    """Save raw LCR availability by quarter before regression lags/dropna."""
    raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    raw = raw[(raw["date"] >= "2010-01-01") & (raw["date"] <= "2025-12-31")].copy()
    raw["quarter"] = raw["date"].dt.to_period("Q").astype(str)
    raw["has_lcr"] = pd.to_numeric(raw["lcr_ratio"], errors="coerce").notna()

    coverage = (
        raw.groupby("quarter")
        .agg(
            total_banks=("bank", "nunique"),
            banks_with_lcr=("has_lcr", "sum"),
            missing_banks=("bank", lambda x: ", ".join(sorted(x[~raw.loc[x.index, "has_lcr"]]))),
        )
        .reset_index()
    )
    coverage.to_csv(OUTPUT_DIR / "lcr_raw_coverage_by_quarter.csv", index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_regression_panel(DATA_PATH)

    baseline_spec = MAIN_H1_H2["M1_baseline"]
    lcr_spec = MAIN_H3["M3_LCR"]

    baseline_full = get_model_sample(panel, baseline_spec)
    lcr_sample = get_model_sample(panel, lcr_spec)

    # Same baseline regression as M1, but restricted to the exact bank-quarters
    # that identify the LCR interaction model.
    baseline_on_lcr_sample = (
        panel[[baseline_spec["y"], *baseline_spec["X"]]]
        .loc[lcr_sample.index]
        .dropna()
    )

    results = [
        estimate("M1_LCR_matched_sample", baseline_on_lcr_sample, baseline_spec),
    ]
    build_results_table(results).to_csv(
        OUTPUT_DIR / "lcr_matched_sample_regression.csv",
        index=False,
    )

    sample_flags = add_quarter_column(baseline_full[[]])
    sample_flags["in_baseline_sample"] = 1
    sample_flags["in_lcr_sample"] = sample_flags.set_index(["bank", "date"]).index.isin(
        lcr_sample.index
    ).astype(int)

    by_bank = (
        sample_flags.groupby("bank")[["in_baseline_sample", "in_lcr_sample"]]
        .sum()
        .reset_index()
    )
    by_bank.to_csv(OUTPUT_DIR / "lcr_sample_by_bank.csv", index=False)

    by_quarter = (
        sample_flags.groupby("quarter")[["in_baseline_sample", "in_lcr_sample"]]
        .sum()
        .reset_index()
    )
    by_quarter.to_csv(OUTPUT_DIR / "lcr_sample_by_quarter.csv", index=False)
    save_raw_lcr_coverage()

    print("Saved output/tables/lcr_matched_sample_regression.csv")
    print("Saved output/tables/lcr_sample_by_bank.csv")
    print("Saved output/tables/lcr_sample_by_quarter.csv")
    print("Saved output/tables/lcr_raw_coverage_by_quarter.csv")


if __name__ == "__main__":
    main()
