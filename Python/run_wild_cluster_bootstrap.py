"""
Wild cluster bootstrap robustness checks for the thesis regressions.

Why we use this:
- The main regressions cluster standard errors at the bank level.
- We only have eight banks.
- With few clusters, ordinary clustered p-values may be unreliable.
- This script therefore computes wild cluster bootstrap p-values for the main
  coefficients used in the thesis interpretation.

Run from the project root:
    python src/validation/run_wild_cluster_bootstrap.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1. Project paths
# ---------------------------------------------------------------------
# This assumes the script is located in:
# src/validation/run_wild_cluster_bootstrap.py
#
# PROJECT_ROOT is then the main thesis/project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The regression helper files are stored in src/pipeline.
PIPELINE_DIR = PROJECT_ROOT / "src" / "pipeline"

# Add the pipeline folder so Python can import your existing regression code.
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


# Import existing project functions/specifications.
from regression_data import load_regression_panel, quarter_to_dates  # noqa: E402
from regression_specs import TABLES  # noqa: E402
from run_panel_regressions import run_model  # noqa: E402


# Input data used in the thesis regressions.
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "panel.csv"

# Output file with bootstrap results.
OUTPUT_PATH = (
    PROJECT_ROOT
    / "output"
    / "tables"
    / "regressions"
    / "wild_cluster_bootstrap_recommended.csv"
)


# ---------------------------------------------------------------------
# 2. Define which coefficients we want to bootstrap
# ---------------------------------------------------------------------
# We do NOT bootstrap every control variable.
# We only bootstrap the coefficients that carry the thesis hypotheses:
# - UnitVaR effect for H1/H2
# - UnitVaR x regulatory ratio interactions for H3
# - Pre/post-COVID UnitVaR effects


@dataclass(frozen=True)
class BootstrapTarget:
    """One coefficient test to run."""

    table: str      # Name of the table group in regression_specs.py
    model: str      # Name of the specific model in that table group
    param: str      # Name of the coefficient to test
    label: str      # Human-readable description


RECOMMENDED_TARGETS = [
    # -------------------------------------------------------------
    # H1-H2: Main UnitVaR effect
    # -------------------------------------------------------------
    BootstrapTarget(
        table="main_h1_h2",
        model="M1_baseline",
        param="d_ln_unit_var_lag",
        label="H1 baseline UnitVaR effect",
    ),
    BootstrapTarget(
        table="main_h1_h2",
        model="M1_first_diff",
        param="d_ln_unit_var_lag",
        label="H1 first-difference UnitVaR effect",
    ),

    # -------------------------------------------------------------
    # H1-H2: Lagged leverage version
    # -------------------------------------------------------------
    BootstrapTarget(
        table="lagged_leverage_h1_h2",
        model="M1_baseline_lagged_leverage",
        param="d_ln_unit_var_lag",
        label="H1 lagged leverage UnitVaR effect",
    ),

    # -------------------------------------------------------------
    # H3: Regulatory interaction models
    # -------------------------------------------------------------
    BootstrapTarget(
        table="main_h3",
        model="M3_LCR",
        param="d_ln_unit_var_x_lcr",
        label="H3 LCR interaction",
    ),
    BootstrapTarget(
        table="main_h3",
        model="M3_CET1",
        param="d_ln_unit_var_x_cet1",
        label="H3 CET1 interaction",
    ),
    BootstrapTarget(
        table="main_h3",
        model="M3_SLR",
        param="d_ln_unit_var_x_slr",
        label="H3 SLR interaction",
    ),

    # -------------------------------------------------------------
    # H3: Lagged leverage versions
    # Added for symmetry, so LCR, CET1 and SLR are all tested.
    # -------------------------------------------------------------
    BootstrapTarget(
        table="lagged_leverage_h3",
        model="M3_LCR_lagged_leverage",
        param="d_ln_unit_var_x_lcr",
        label="H3 LCR interaction with lagged leverage",
    ),
    BootstrapTarget(
        table="lagged_leverage_h3",
        model="M3_CET1_lagged_leverage",
        param="d_ln_unit_var_x_cet1",
        label="H3 CET1 interaction with lagged leverage",
    ),
    BootstrapTarget(
        table="lagged_leverage_h3",
        model="M3_SLR_lagged_leverage",
        param="d_ln_unit_var_x_slr",
        label="H3 SLR interaction with lagged leverage",
    ),

    # -------------------------------------------------------------
    # COVID split
    # -------------------------------------------------------------
    BootstrapTarget(
        table="covid_split",
        model="M1_pre_covid",
        param="d_ln_unit_var_lag",
        label="Pre-COVID UnitVaR effect",
    ),
    BootstrapTarget(
        table="covid_split",
        model="M1_post_covid",
        param="d_ln_unit_var_lag",
        label="Post-COVID UnitVaR effect",
    ),
]


# ---------------------------------------------------------------------
# 3. Prepare the exact data used in each model
# ---------------------------------------------------------------------
def prepare_model_data(panel: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """
    Keep only:
    - the dependent variable,
    - the explanatory variables,
    - the correct date range,
    - complete observations.

    This makes sure the bootstrap uses the same sample as the regression.
    """
    columns = [spec["y"], *spec["X"]]

    # Check that all variables required by the model exist in the data.
    missing = [col for col in columns if col not in panel.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Keep only the variables used in this model.
    data = panel[columns].copy()

    # Apply the date range defined in regression_specs.py.
    start, _ = quarter_to_dates(spec["date_range"][0])
    _, end = quarter_to_dates(spec["date_range"][1])
    dates = data.index.get_level_values("date")
    data = data[(dates >= start) & (dates <= end)].dropna()

    # Avoid running bootstrap on very tiny samples.
    if len(data) < 30:
        raise ValueError(f"Too few observations: {len(data)}")

    # Turn bank/date from index into ordinary columns.
    return data.reset_index()


# ---------------------------------------------------------------------
# 4. Build the regression matrix manually
# ---------------------------------------------------------------------
def build_ols_design(data: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """
    Build the same regression design as the main panel regressions.

    Why this is needed:
    - The bootstrap procedure needs direct access to X and y.
    - Therefore we manually create:
        1. the explanatory variables,
        2. bank fixed effects,
        3. time fixed effects.

    This is just a manual version of what PanelOLS does internally.
    """
    parts = []

    # Add the ordinary explanatory variables.
    parts.append(data[spec["X"]].astype(float).reset_index(drop=True))

    # Add bank fixed effects if the model uses bank FE.
    if spec.get("bank_fe"):
        bank_dummies = pd.get_dummies(
            data["bank"],
            prefix="bank",
            drop_first=False,
            dtype=float,
        ).reset_index(drop=True)
        parts.append(bank_dummies)

    # Add time fixed effects if the model uses time FE.
    if spec.get("time_fe"):
        time_dummies = pd.get_dummies(
            data["date"],
            prefix="date",
            # If bank FE are included, drop one time dummy to avoid perfect collinearity.
            drop_first=bool(spec.get("bank_fe")),
            dtype=float,
        ).reset_index(drop=True)
        parts.append(time_dummies)

    return pd.concat(parts, axis=1)


# ---------------------------------------------------------------------
# 5. Compute coefficient, clustered t-statistic and standard error
# ---------------------------------------------------------------------
def cluster_t_stat(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    param_index: int,
) -> tuple[float, float, float]:
    """
    Estimate OLS and compute a bank-clustered t-statistic.

    This t-statistic is used as the observed test statistic in the bootstrap.
    """
    # OLS beta = (X'X)^(-1) X'y
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y

    # Residuals from the full model.
    residual = y - x @ beta

    # Clustered variance for the tested coefficient.
    row = xtx_inv[param_index]
    projected_x = x @ row

    variance = 0.0
    for cluster in np.unique(clusters):
        idx = clusters == cluster
        score = projected_x[idx] @ residual[idx]
        variance += score**2

    standard_error = float(np.sqrt(max(variance, 0.0)))
    t_stat = float(beta[param_index] / standard_error)

    return float(beta[param_index]), t_stat, standard_error


# ---------------------------------------------------------------------
# 6. Create Rademacher weights
# ---------------------------------------------------------------------
def rademacher_weights(
    clusters: np.ndarray,
    requested_draws: int,
    seed: int,
    exhaustive_limit: int,
) -> tuple[np.ndarray, str]:
    """
    Create cluster-level weights.

    Rademacher weights mean:
    - each bank gets either +1 or -1,
    - all observations for that bank get the same sign.

    With 8 banks, there are only 2^8 = 256 possible sign combinations.
    This script therefore uses all combinations instead of random draws.
    """
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    exact_draws = 2**n_clusters

    # Use every possible sign combination when the number of clusters is small.
    if n_clusters <= exhaustive_limit and exact_draws <= requested_draws:
        weights = np.array(
            list(itertools.product([-1.0, 1.0], repeat=n_clusters)),
            dtype=float,
        )
        return weights, "exhaustive_rademacher"

    # If there are many clusters, draw random sign combinations instead.
    rng = np.random.default_rng(seed)
    weights = rng.choice(
        np.array([-1.0, 1.0]),
        size=(requested_draws, n_clusters),
        replace=True,
    )
    return weights, "random_rademacher"


# ---------------------------------------------------------------------
# 7. Run the wild cluster bootstrap
# ---------------------------------------------------------------------
def bootstrap_t_stats(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    param_index: int,
    cluster_weights: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """
    Compute bootstrap t-statistics.

    Important:
    The bootstrap must impose the null hypothesis.

    How this is done:
    1. Remove the coefficient being tested from the model.
       This creates the restricted model.
    2. Estimate the restricted model.
    3. Save restricted fitted values and restricted residuals.
    4. Flip residual signs at the bank-cluster level.
    5. Create fake y-values.
    6. Re-estimate the full model on each fake dataset.
    7. Save the bootstrap t-statistic.
    """

    # Step 1: remove the tested variable.
    x_restricted = np.delete(x, param_index, axis=1)

    # Step 2: estimate the restricted model.
    beta_restricted = np.linalg.pinv(x_restricted) @ y

    # Step 3: fitted values and residuals under the null.
    fitted_restricted = x_restricted @ beta_restricted
    residual_restricted = y - fitted_restricted

    # Prepare full-model OLS objects for fast repeated estimation.
    xtx_inv = np.linalg.pinv(x.T @ x)
    pinv_x = xtx_inv @ x.T

    # Objects needed for clustered standard errors.
    row = xtx_inv[param_index]
    projected_x = x @ row

    # Map each observation to its bank cluster.
    unique_clusters = np.unique(clusters)
    cluster_to_col = {cluster: i for i, cluster in enumerate(unique_clusters)}
    cluster_cols = np.array([cluster_to_col[cluster] for cluster in clusters])

    out = []

    # Run bootstrap in chunks to avoid memory problems.
    for start in range(0, len(cluster_weights), chunk_size):
        weights_chunk = cluster_weights[start : start + chunk_size]

        # Give every observation the sign of its bank.
        observation_weights = weights_chunk[:, cluster_cols].T

        # Step 4-5: create fake dependent variables.
        y_star = fitted_restricted[:, None] + residual_restricted[:, None] * observation_weights

        # Step 6: estimate full model on each fake dataset.
        beta_star = pinv_x @ y_star
        residual_star = y_star - x @ beta_star

        # Clustered standard error for each fake regression.
        variance = np.zeros(y_star.shape[1])
        for cluster in unique_clusters:
            idx = clusters == cluster
            score = projected_x[idx] @ residual_star[idx, :]
            variance += score**2

        standard_error = np.sqrt(np.maximum(variance, np.finfo(float).eps))

        # Step 7: bootstrap t-statistic for the tested coefficient.
        out.append(beta_star[param_index, :] / standard_error)

    return np.concatenate(out)


# ---------------------------------------------------------------------
# 8. Compute one wild bootstrap p-value
# ---------------------------------------------------------------------
def wild_cluster_bootstrap_pvalue(
    data: pd.DataFrame,
    spec: dict,
    param: str,
    draws: int,
    seed: int,
    exhaustive_limit: int,
    chunk_size: int,
) -> dict:
    """
    Run one wild cluster bootstrap test for one coefficient.
    """
    # Build X matrix.
    design = build_ols_design(data, spec)

    # Make sure the coefficient exists in the model.
    if param not in design.columns:
        raise ValueError(f"Parameter {param!r} is not in the model design.")

    # Convert data to numpy arrays.
    x = design.to_numpy(dtype=float)
    y = data[spec["y"]].to_numpy(dtype=float)
    clusters = data["bank"].to_numpy()

    # Find the position of the coefficient being tested.
    param_index = design.columns.get_loc(param)

    # Observed coefficient and t-statistic.
    coef, observed_t, observed_se = cluster_t_stat(x, y, clusters, param_index)

    # Create cluster-level sign flips.
    weights, draw_type = rademacher_weights(
        clusters=clusters,
        requested_draws=draws,
        seed=seed,
        exhaustive_limit=exhaustive_limit,
    )

    # Generate bootstrap t-statistics.
    boot_t = bootstrap_t_stats(
        x=x,
        y=y,
        clusters=clusters,
        param_index=param_index,
        cluster_weights=weights,
        chunk_size=chunk_size,
    )

    # Two-sided p-value:
    # How often is the bootstrap t-statistic at least as extreme as the real one?
    exceedances = int(np.sum(np.abs(boot_t) >= abs(observed_t) - 1e-12))

    # If using random draws, add-one correction avoids p = 0.
    if draw_type == "random_rademacher":
        p_value = (exceedances + 1) / (len(boot_t) + 1)
    else:
        p_value = exceedances / len(boot_t)

    return {
        "coef": coef,
        "wild_t": observed_t,
        "wild_se": observed_se,
        "wild_p_value": p_value,
        "bootstrap_exceedances": exceedances,
        "bootstrap_draws": len(boot_t),
        "draw_type": draw_type,
    }


# ---------------------------------------------------------------------
# 9. Run one target and collect results
# ---------------------------------------------------------------------
def run_target(
    panel: pd.DataFrame,
    target: BootstrapTarget,
    draws: int,
    seed: int,
    exhaustive_limit: int,
    chunk_size: int,
) -> dict:
    """
    Run one bootstrap test and return one row for the output CSV.
    """
    # Get the model specification from regression_specs.py.
    spec = TABLES[target.table][target.model]

    # Prepare the same sample used in the model.
    data = prepare_model_data(panel, spec)

    # Run wild cluster bootstrap.
    bootstrap = wild_cluster_bootstrap_pvalue(
        data=data,
        spec=spec,
        param=target.param,
        draws=draws,
        seed=seed,
        exhaustive_limit=exhaustive_limit,
        chunk_size=chunk_size,
    )

    # Also run the ordinary clustered regression.
    # This lets us report ordinary clustered p-values next to bootstrap p-values.
    conventional = run_model(
        panel=panel,
        name=target.model,
        spec=spec,
        fit_kwargs={"cov_type": "clustered", "cluster_entity": True},
    )

    if "result" not in conventional:
        raise RuntimeError(conventional.get("error", "PanelOLS model failed."))

    result = conventional["result"]
    dates = data["date"]

    return {
        "label": target.label,
        "table": target.table,
        "model": target.model,
        "parameter": target.param,
        "coef": result.params[target.param],
        "clustered_t": result.tstats[target.param],
        "clustered_p_value": result.pvalues[target.param],
        "wild_t": bootstrap["wild_t"],
        "wild_p_value": bootstrap["wild_p_value"],
        "bootstrap_draws": bootstrap["bootstrap_draws"],
        "bootstrap_exceedances": bootstrap["bootstrap_exceedances"],
        "draw_type": bootstrap["draw_type"],
        "n_obs": len(data),
        "n_banks": data["bank"].nunique(),
        "period": f"{dates.min().date()} - {dates.max().date()}",
    }


# ---------------------------------------------------------------------
# 10. Command-line options
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """
    Allow optional changes from the command line.

    Usually you do not need to change anything.
    """
    parser = argparse.ArgumentParser(
        description="Run Rademacher wild cluster bootstrap checks.",
    )

    parser.add_argument(
        "--draws",
        type=int,
        default=9999,
        help=(
            "Requested bootstrap draws. With eight banks, the script uses all "
            "2^8 exact Rademacher sign patterns instead of random resampling."
        ),
    )

    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument(
        "--exhaustive-limit",
        type=int,
        default=12,
        help="Use exact Rademacher enumeration when clusters are at or below this limit.",
    )

    parser.add_argument("--chunk-size", type=int, default=1000)

    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)

    return parser.parse_args()


# ---------------------------------------------------------------------
# 11. Main script
# ---------------------------------------------------------------------
def main() -> None:
    """
    Main workflow:
    1. Load panel data.
    2. Run bootstrap for all selected targets.
    3. Save results to CSV.
    4. Print short version to terminal.
    """
    args = parse_args()

    # Load regression panel.
    panel = load_regression_panel(str(DATA_PATH))

    # Run all selected bootstrap tests.
    rows = []
    for target in RECOMMENDED_TARGETS:
        row = run_target(
            panel=panel,
            target=target,
            draws=args.draws,
            seed=args.seed,
            exhaustive_limit=args.exhaustive_limit,
            chunk_size=args.chunk_size,
        )
        rows.append(row)

    # Save results.
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    table = pd.DataFrame(rows)
    table.to_csv(output, index=False)

    # Print compact terminal output.
    display_cols = [
        "model",
        "parameter",
        "coef",
        "clustered_p_value",
        "wild_p_value",
        "bootstrap_draws",
        "draw_type",
    ]

    print(table[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()