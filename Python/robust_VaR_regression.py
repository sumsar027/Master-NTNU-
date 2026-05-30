"""
Run two simple VaR-conversion robustness checks.

1) Baseline regression using only the four banks with native 99% VaR.
2) Baseline regression using native 99% VaR where available and a Bank of
   America 99/95 scaling factor for banks that only report 95% VaR.

Output:
- output/tables/regressions/var_conversion_robustness_baseline.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "pipeline"))

from regression_output import build_results_table  # noqa: E402
from regression_specs import MAIN_H1_H2  # noqa: E402
from run_panel_regressions import build_formula  # noqa: E402


RAW_VAR_FILE = ROOT / "data" / "raw" / "VaR_python.xlsx"
PANEL_FILE = ROOT / "data" / "processed" / "panel.csv"
OUT_FILE = ROOT / "output" / "tables" / "regressions" / "var_conversion_robustness_baseline.csv"


def norm_bank(value: object) -> str:
    return "".join(ch for ch in str(value).lower().strip() if ch.isalnum())


def load_var_data() -> pd.DataFrame:
    raw = pd.read_excel(RAW_VAR_FILE, sheet_name="new", header=None)
    banks = raw.iloc[1].ffill()
    levels = raw.iloc[2]
    dates = pd.to_datetime(raw.iloc[3:, 0], errors="coerce")

    rows = []
    for col in range(1, raw.shape[1]):
        level = str(levels.iloc[col]).strip()
        if level not in {"0.95", "0.99", "95", "99"}:
            continue

        rows.append(
            pd.DataFrame(
                {
                    "bank": norm_bank(banks.iloc[col]),
                    "date": dates,
                    f"var_{'95' if '95' in level else '99'}": pd.to_numeric(
                        raw.iloc[3:, col], errors="coerce"
                    ),
                }
            )
        )

    return (
        pd.concat(rows, ignore_index=True)
        .dropna(subset=["date"])
        .groupby(["bank", "date"], as_index=False)
        .first()
    )


def make_panel(total_var: pd.DataFrame) -> pd.DataFrame:
    panel = pd.read_csv(PANEL_FILE, parse_dates=["date"]).drop(columns=["total_var"])
    panel = panel.merge(total_var, on=["bank", "date"], how="left")

    for col in ["total_assets", "total_equity", "total_var", "roa"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

    panel = panel.sort_values(["bank", "date"])
    panel["d_ln_leverage"] = np.log(panel["total_assets"] / panel["total_equity"]).groupby(panel["bank"]).diff()
    panel["ln_unit_var"] = np.log((panel["total_var"] / panel["total_assets"]).clip(lower=1e-12))
    panel["d_ln_unit_var_lag"] = panel.groupby("bank")["ln_unit_var"].diff().groupby(panel["bank"]).shift(1)
    panel["size_lag"] = np.log(panel["total_assets"].clip(lower=1e-12)).groupby(panel["bank"]).shift(1)
    panel["roa_lag"] = panel.groupby("bank")["roa"].shift(1)

    return panel.set_index(["bank", "date"])


def run_baseline(name: str, panel: pd.DataFrame) -> dict:
    spec = MAIN_H1_H2["M1_baseline"]
    data = panel[[spec["y"], *spec["X"]]].copy()
    dates = data.index.get_level_values("date")
    data = data[(dates >= "2010-01-01") & (dates <= "2025-12-31")].dropna()

    result = PanelOLS.from_formula(build_formula(spec["y"], spec["X"], spec), data=data).fit(
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


def main() -> None:
    var = load_var_data()
    boa = var.query("bank == 'bankofamerica'").dropna(subset=["var_95", "var_99"])
    boa_factor = float((boa["var_99"] / boa["var_95"]).mean())

    banks_with_95 = set(var.loc[var["var_95"].notna(), "bank"])
    native_99 = (
        var[~var["bank"].isin(banks_with_95)][["bank", "date", "var_99"]]
        .rename(columns={"var_99": "total_var"})
        .dropna()
    )
    boa_scaled = var[["bank", "date", "var_99", "var_95"]].copy()
    boa_scaled["total_var"] = boa_scaled["var_99"].fillna(boa_scaled["var_95"] * boa_factor)
    boa_scaled = boa_scaled[["bank", "date", "total_var"]].dropna()

    results = [
        run_baseline("M1_native_99_only", make_panel(native_99)),
        run_baseline("M1_BoA_factor", make_panel(boa_scaled)),
    ]

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    build_results_table(results).to_csv(OUT_FILE, index=False)

    print(f"BoA factor: {boa_factor:.6f}")
    print(f"Saved {OUT_FILE}")


if __name__ == "__main__":
    main()
