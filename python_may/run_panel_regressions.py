"""
Estimate panel regression tables used in the thesis.

This is the main regression runner. It does not define the models itself; it
loads model groups from regression_specs.py, builds the analysis panel from
regression_data.py, estimates each model, and saves formatted CSV tables.
"""

from pathlib import Path
from linearmodels.panel import PanelOLS, PooledOLS
from regression_data import load_regression_panel, quarter_to_dates
from regression_output import build_results_table
from regression_specs import TABLES


DATA_PATH = "data/processed/panel.csv"
OUTPUT_DIR = Path("output/tables/regressions")

COVARIANCE_TYPES = {
    # Clustered standard errors are the main specification in the thesis.
    "clustered": {"cov_type": "clustered", "cluster_entity": True},
    # Robust standard errors are used as a robustness check.
    "robust": {"cov_type": "robust"},
}


def run_model(panel, name: str, spec: dict, fit_kwargs: dict) -> dict:
    """Estimate one model and return result plus table metadata."""
    y = spec["y"]
    x_vars = spec["X"]

    missing = [col for col in [y, *x_vars] if col not in panel.columns]
    if missing:
        return {"name": name, "spec": spec, "error": f"Missing columns: {missing}"}

    # Keep only the dependent variable and regressors needed for this model.
    data = panel[[y, *x_vars]].copy()

    # Apply the sample window specified for the model, for example 2015-Q1 to
    # 2025-Q4 for regulatory-ratio specifications.
    start, _ = quarter_to_dates(spec["date_range"][0])
    _, end = quarter_to_dates(spec["date_range"][1])
    dates = data.index.get_level_values("date")
    data = data[(dates >= start) & (dates <= end)].dropna()

    # Avoid reporting regressions with too few complete observations.
    if len(data) < 30:
        return {"name": name, "spec": spec, "error": f"Too few observations: {len(data)}"}

    try:
        # linearmodels uses formula strings to include regressors and fixed
        # effects. build_formula keeps that syntax in one place.
        formula = build_formula(y, x_vars, spec)
        model = PanelOLS.from_formula(formula, data=data) if uses_fixed_effects(spec) else PooledOLS.from_formula(formula, data=data)
        result = model.fit(**fit_kwargs)
        actual_dates = data.index.get_level_values("date")

        return {
            "name": name,
            "spec": spec,
            "result": result,
            "n": len(data),
            "n_banks": data.index.get_level_values("bank").nunique(),
            "period": f"{actual_dates.min().date()} - {actual_dates.max().date()}",
        }
    except Exception as exc:
        return {"name": name, "spec": spec, "error": str(exc)}


def build_formula(y: str, x_vars: list[str], spec: dict) -> str:
    """Build a linearmodels formula from one model specification."""
    rhs = " + ".join(x_vars)

    if not uses_fixed_effects(spec):
        return f"{y} ~ 1 + {rhs}"

    fixed_effects = []
    if spec.get("bank_fe"):
        fixed_effects.append("EntityEffects")
    if spec.get("time_fe"):
        fixed_effects.append("TimeEffects")

    return f"{y} ~ {rhs} + {' + '.join(fixed_effects)}"


def uses_fixed_effects(spec: dict) -> bool:
    """Return whether a model uses bank or time fixed effects."""
    return bool(spec.get("bank_fe") or spec.get("time_fe"))


def run_table(panel, table_name: str, models: dict, cov_label: str, fit_kwargs: dict) -> Path:
    """Estimate one table under one covariance estimator and save it."""
    results = [
        run_model(panel, model_name, model_spec, fit_kwargs)
        for model_name, model_spec in models.items()
    ]

    output_path = OUTPUT_DIR / f"{table_name}_{cov_label}.csv"
    build_results_table(results).to_csv(output_path, index=False)
    return output_path


def main() -> None:
    """Run all regression tables."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_regression_panel(DATA_PATH)

    # Estimate every thesis table twice: once with clustered standard errors and
    # once with heteroskedasticity-robust standard errors.
    for table_name, models in TABLES.items():
        for cov_label, fit_kwargs in COVARIANCE_TYPES.items():
            output_path = run_table(panel, table_name, models, cov_label, fit_kwargs)
            print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
