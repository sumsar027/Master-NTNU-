import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import os
from linearmodels.panel import PanelOLS, PooledOLS


# ══════════════════════════════════════════════════════════════════════════════
#  SETTINGS  –  change these as needed
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH  = "dataframe/dataframe.csv"
OUTPUT_DIR = "output/tables"

# Which banks belong to which group
MARKET_BANKS  = ["goldmansachs", "morganstanley"]
CUSTODY_BANKS = ["statestreet", "bny"]

# Which standard errors to run
#   clustered → main result  (clustered by bank)
#   robust    → robustness check (White)
COV_TYPES = {
    "clustered": "clustered",
    "robust":    "robust",
}

# Filename for output
OUTPUT_FILES = {
    "clustered": "regression_results_clustered_SE.csv",
    "robust":    "regression_results_robust_SE.csv",
}


# ══════════════════════════════════════════════════════════════════════════════
#  MODELS  –  add new models here
# ══════════════════════════════════════════════════════════════════════════════
#
#  Each entry is one model. The fields mean:
#    y          → dependent variable
#    X          → list of explanatory variables
#    bank_fe    → control for bank fixed effects? (True/False)
#    time_fe    → control for time fixed effects? (True/False)
#    banks      → restrict to these banks (None = all banks)
#    date_range → time period on the form ("YYYY-Qn", "YYYY-Qn")

MODELS = {
    "M1_basel_test": {
        "y":          "d_ln_leverage",
        "X":          ["d_ln_unit_var_lag", "size_lag", "roa_lag"],
        "bank_fe":    True,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2010-Q1", "2025-Q4"),
    },
    "M1_FD_baseline": {
        "y":          "d_ln_leverage",
        "X":          ["d_ln_unit_var_lag", "d_size_lag", "d_roa_lag"],
        "bank_fe":    False,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2010-Q1", "2025-Q4"),
    },
    "M1_lagged": {
        "y":          "d_ln_leverage",
        "X":          ["d_ln_unit_var_lag", "d_ln_leverage_lag", "size_lag", "roa_lag"],
        "bank_fe":    True,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2010-Q1", "2025-Q4"),
    },
    "M2_dummy": {
        "y":          "d_ln_leverage",
        "X":          ["d_ln_unit_var_lag", "d_ln_unit_var_x_market", "d_ln_unit_var_x_custody", "size_lag", "roa_lag"],
        "bank_fe":    True,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2010-Q1", "2025-Q4"),
    },
    "M2_FD_dummy": {
        "y":          "d_ln_leverage",
        "X":          ["d_ln_unit_var_lag", "d_ln_unit_var_x_market", "d_ln_unit_var_x_custody", "d_size_lag", "d_roa_lag"],
        "bank_fe":    False,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2010-Q1", "2025-Q4"),
    },
    "M3a_LCR": {
        "y":          "d_ln_leverage",
        "X":          ["d_ln_unit_var_lag", "lcr_ratio_lag", "d_ln_unit_var_x_lcr", "roa_lag", "size_lag"],
        "bank_fe":    True,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2015-Q1", "2025-Q4"),
    },
    "M3b_CET1": {
        "y":          "d_ln_leverage",
        "X":          ["d_ln_unit_var_lag", "cet1_ratio_lag", "d_ln_unit_var_x_cet1", "roa_lag", "size_lag"],
        "bank_fe":    True,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2015-Q1", "2025-Q4"),
    },
    "M3c_SLR": {
        "y":          "d_ln_leverage",
        "X":          ["d_ln_unit_var_lag", "slr_ratio_lag", "d_ln_unit_var_x_slr", "roa_lag", "size_lag"],
        "bank_fe":    True,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2015-Q1", "2025-Q4"),
    },
    "M4_pre_covid": {
        "y":          "d_ln_leverage",
        "X":          ["d_ln_unit_var_lag", "roa_lag", "size_lag"],
        "bank_fe":    True,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2010-Q1", "2019-Q4"),
    },
    "M4_post_covid": {
        "y":          "d_ln_leverage",
        "X":          ["d_ln_unit_var_lag", "roa_lag", "size_lag"],
        "bank_fe":    True,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2020-Q1", "2025-Q4"),
    },
    "M5_pre_covid_SLR": {
        "y":          "d_ln_leverage",
        "X":          ["d_ln_unit_var_lag", "slr_ratio_lag", "d_ln_unit_var_x_slr", "roa_lag", "size_lag"],
        "bank_fe":    True,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2015-Q1", "2019-Q4"),
    },
    "M5_post_covid_SLR": {
        "y":          "d_ln_leverage",
        "X":          ["d_ln_unit_var_lag",  "slr_ratio_lag", "d_ln_unit_var_x_slr", "roa_lag", "size_lag"],
        "bank_fe":    True,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2020-Q1", "2025-Q4"),
    },
    "M5_pre_covid_CET1": {
        "y":          "d_ln_leverage",
        "X":          ["d_ln_unit_var_lag", "cet1_ratio_lag", "d_ln_unit_var_x_cet1", "roa_lag", "size_lag"],
        "bank_fe":    True,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2015-Q1", "2019-Q4"),
    },
    "M5_post_covid_CET1": {
        "y":          "d_ln_leverage",
        "X":          ["d_ln_unit_var_lag",  "cet1_ratio_lag", "d_ln_unit_var_x_cet1", "roa_lag", "size_lag"],
        "bank_fe":    True,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2020-Q1", "2025-Q4"),
    },
    "M5_pre_covid_LCR": {
        "y":          "d_ln_leverage",
        "X":          ["d_ln_unit_var_lag", "lcr_ratio_lag", "d_ln_unit_var_x_lcr", "roa_lag", "size_lag"],
        "bank_fe":    True,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2015-Q1", "2019-Q4"),
    },
    "M5_post_covid_LCR": {
        "y":          "d_ln_leverage",
        "X":          ["d_ln_unit_var_lag",  "lcr_ratio_lag", "d_ln_unit_var_x_lcr", "roa_lag", "size_lag"],
        "bank_fe":    True,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2020-Q1", "2025-Q4"),
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

# Sort and remove duplicates
df = (
    df
    .sort_values(["bank", "date"])
    .drop_duplicates(subset=["bank", "date"], keep="first")
    .reset_index(drop=True)
)


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD VARIABLES
# ══════════════════════════════════════════════════════════════════════════════

# ── Level variables ──
df["leverage"]     = df["total_assets"] / df["total_equity"]
df["ln_leverage"]  = np.log(df["leverage"].clip(lower=1e-12))
df["size"]         = np.log(df["total_assets"].clip(lower=1e-12))
df["unit_var"]     = (df["total_var"]) / df["total_assets"]
df["ln_unit_var"]  = np.log(df["unit_var"].clip(lower=1e-12))

# Dummy variables separating bank types
df["market_dummy"]  = df["bank"].isin(MARKET_BANKS).astype(int)
df["custody_dummy"] = df["bank"].isin(CUSTODY_BANKS).astype(int)
df["post_covid"]    = (df["date"] >= pd.Timestamp("2020-03-01")).astype(int)

# ── First-difference variables (change from previous quarter)  –  add new ones here ──
DIFF_VARS = [
    # (source column,  name of new diff variable)
    ("ln_leverage",  "d_ln_leverage"),
    ("ln_unit_var",  "d_ln_unit_var"),
    ("size",         "d_size"),
    ("roa",          "d_roa"),
]
for src, new in DIFF_VARS:
    df[new] = df.groupby("bank")[src].diff()

# ── Lagged variables (value from previous quarter)  –  add new ones here ──
LAG_VARS = [
    "roa", "size", "ln_unit_var",
    "d_ln_unit_var", "d_ln_leverage", "d_size", "d_roa",
    "lcr_ratio", "cet1_ratio", "slr_ratio",
]
for col in LAG_VARS:
    df[f"{col}_lag"] = df.groupby("bank")[col].shift(1)

# ── Interaction variables (product of two variables)  –  add new ones here ──
INTERACTIONS = [
    # (name of new variable,                    factor 1,               factor 2)
    ("d_ln_unit_var_x_market",        "d_ln_unit_var_lag",    "market_dummy"),
    ("d_ln_unit_var_x_custody",       "d_ln_unit_var_lag",    "custody_dummy"),
    ("d_ln_unit_var_x_lcr",           "d_ln_unit_var_lag",    "lcr_ratio_lag"),
    ("d_ln_unit_var_x_cet1",          "d_ln_unit_var_lag",    "cet1_ratio_lag"),
    ("d_ln_unit_var_x_slr",           "d_ln_unit_var_lag",    "slr_ratio_lag"),
    ("d_ln_unit_var_x_post_covid",    "d_ln_unit_var_lag",    "post_covid"),
    ("slr_x_post_covid",              "slr_ratio_lag",        "post_covid"),
    ("lcr_x_post_covid",              "lcr_ratio_lag",        "post_covid"),
    ("cet1_x_post_covid",             "cet1_ratio_lag",       "post_covid"),
    ("d_ln_unit_var_x_slr_x_post_covid",  "d_ln_unit_var_x_slr",  "post_covid"),
    ("d_ln_unit_var_x_lcr_x_post_covid",  "d_ln_unit_var_x_lcr",  "post_covid"),
    ("d_ln_unit_var_x_cet1_x_post_covid", "d_ln_unit_var_x_cet1", "post_covid"),
]
for new, f1, f2 in INTERACTIONS:
    df[new] = df[f1] * df[f2]

# Set bank and date as index – required by linearmodels
df = df.set_index(["bank", "date"])


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTION: convert quarter string to dates
# ══════════════════════════════════════════════════════════════════════════════

def quarter_to_dates(s):
    # Converts "2015-Q1" to the start and end date of that quarter
    year, q = str(s).upper().split("-Q")
    starts = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
    ends   = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
    q = int(q)
    return pd.Timestamp(f"{year}-{starts[q]}"), pd.Timestamp(f"{year}-{ends[q]}")


# ══════════════════════════════════════════════════════════════════════════════
#  RUN ONE REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

def run_model(name, spec, cov_type):

    y, X       = spec["y"], spec["X"]
    bank_fe    = spec["bank_fe"]
    time_fe    = spec["time_fe"]
    banks      = spec.get("banks")
    date_range = spec.get("date_range")

    # Check that all columns exist
    missing = [c for c in [y] + X if c not in df.columns]
    if missing:
        return {"name": name, "error": f"Missing columns: {missing}"}

    # Select relevant columns and filter by banks and time period
    sub = df[[y] + X].copy()
    if banks is not None:
        sub = sub[sub.index.get_level_values("bank").isin(banks)]
    if date_range is not None:
        start, _ = quarter_to_dates(date_range[0])
        _, end   = quarter_to_dates(date_range[1])
        dates    = sub.index.get_level_values("date")
        sub      = sub[(dates >= start) & (dates <= end)]

    # Drop rows with missing values
    sub = sub.dropna()
    if len(sub) < 30:
        return {"name": name, "error": f"Too few observations: {len(sub)}"}

    # Determine which fixed effects to include
    fe_terms = (["EntityEffects"] if bank_fe else []) + (["TimeEffects"] if time_fe else [])

    # Choose model type and build formula
    if bank_fe or time_fe:
        formula = f"{y} ~ {' + '.join(X)} + {' + '.join(fe_terms)}"
        model   = PanelOLS.from_formula(formula, data=sub)
    else:
        formula = f"{y} ~ 1 + {' + '.join(X)}"
        model   = PooledOLS.from_formula(formula, data=sub)

    # Set standard errors
    fit_kwargs = (
        {"cov_type": "clustered", "cluster_entity": True}
        if cov_type == "clustered"
        else {"cov_type": "robust"}
    )
    res = model.fit(**fit_kwargs)

    actual_dates = sub.index.get_level_values("date")
    return {
        "name":    name,
        "spec":    spec,
        "result":  res,
        "n":       len(sub),
        "n_banks": sub.index.get_level_values("bank").nunique(),
        "period":  f"{actual_dates.min().date()} – {actual_dates.max().date()}",
    }


# ── Run all models for both SE types ──
all_results = {
    label: [run_model(name, spec, cov_type) for name, spec in MODELS.items()]
    for label, cov_type in COV_TYPES.items()
}


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD AND SAVE CSV
# ══════════════════════════════════════════════════════════════════════════════

def build_csv(results):
    rows = []

    # Collect all unique parameters across models
    all_params = []
    for r in results:
        if "result" in r:
            for p in r["result"].params.index:
                if p not in all_params:
                    all_params.append(p)

    # One row for coefficient, one for t-statistic
    for param in all_params:
        coef_row  = {"variable": param, "stat": "coef"}
        tstat_row = {"variable": param, "stat": "tstat"}
        for r in results:
            if "result" not in r:
                coef_row[r["name"]]  = ""
                tstat_row[r["name"]] = ""
                continue
            res = r["result"]
            if param in res.params.index:
                b      = res.params[param]
                t      = res.tstats[param]
                p_val  = res.pvalues[param]
                stars  = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
                coef_row[r["name"]]  = f"{b:+.4f}{stars}"
                tstat_row[r["name"]] = f"({t:.2f})"
            else:
                coef_row[r["name"]]  = "-"
                tstat_row[r["name"]] = ""
        rows.append(coef_row)
        rows.append(tstat_row)

    # Metadata at the bottom of the table
    def meta(label, values):
        row = {"variable": label, "stat": ""}
        for r, v in zip(results, values):
            row[r["name"]] = v
        return row

    rows.append(meta("N (obs)",     [r.get("n",       "–") for r in results]))
    rows.append(meta("N (banker)",  [r.get("n_banks", "–") for r in results]))
    rows.append(meta("R² (within)", [
        f"{r['result'].rsquared_within:.4f}" if "result" in r else "-" for r in results
    ]))
    rows.append(meta("Bank FE",  ["Yes" if r.get("spec", {}).get("bank_fe")  else "No" for r in results]))
    rows.append(meta("Tid FE",   ["Yes" if r.get("spec", {}).get("time_fe")  else "No" for r in results]))
    rows.append(meta("Periode",  [r.get("period", "–") for r in results]))

    return pd.DataFrame(rows)


# Save one file per SE type
os.makedirs(OUTPUT_DIR, exist_ok=True)
for label, results in all_results.items():
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILES[label])
    build_csv(results).to_csv(out_path, index=False)
    print(f"✓ {label:10s} → {out_path}")

print("\nDone!")
print("  clustered_SE → main result")
print("  robust_SE    → robustness check")