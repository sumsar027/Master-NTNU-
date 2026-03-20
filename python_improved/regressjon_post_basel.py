import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import os
from linearmodels.panel import PanelOLS, PooledOLS

# ══════════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH    = "dataframe/dataframe.csv"
OUTPUT_DIR   = "output/tables"

POST_BASEL_YEAR = 2014
MARKET_BANKS    = ["goldmansachs", "morganstanley"]
UNIVERSAL_BANKS = ["jpmorgan", "bankofamerica", "citigroup", "wellsfargo"]
CUSTODY_BANKS   = ["statestreet", "bny"]

# Tre SE-varianter som kjøres for alle modeller:
#   "clustered" → cluster_entity=True  (hovedresultat, standard i panelmodeller)
#   "robust"    → White/heteroskedastisitetsrobust  (robusthetssjekk 1)
#   "kernel"    → Driscoll–Kraay HAC               (robusthetssjekk 2)
COV_TYPES = {
    "clustered": "clustered",
    "robust":    "robust",
    "kernel":    "kernel",
}

# ══════════════════════════════════════════════════════════════════════════════
#  MODELLER
# ══════════════════════════════════════════════════════════════════════════════

MODELS = {
    "M1_basel_test": {
        "y":          "d_ln_leverage",
        "X":          ["d_ln_unit_var_lag", "size_lag", "roa_lag", "d_ln_leverage_lag"],
        "bank_fe":    True,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2010-Q1", "2025-Q4"),
    },
    "M1_FD_baseline": {
        "y":          "d_ln_leverage",
        "X":          ["d_ln_unit_var_lag", "d_size_lag", "d_roa_lag", "d_ln_leverage_lag"],
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
        "X":          ["d_ln_unit_var_lag", "d_ln_unit_var_x_market", "d_ln_unit_var_x_custody", "size_lag", "roa_lag", "d_ln_leverage_lag"],
        "bank_fe":    True,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2010-Q1", "2025-Q4"),
    },
    "M2_FD_dummy": {
        "y":          "d_ln_leverage",
        "X":          ["d_ln_unit_var_lag", "d_ln_unit_var_x_market", "d_ln_unit_var_x_custody", "d_size_lag", "d_roa_lag", "d_ln_leverage_lag"],
        "bank_fe":    False,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2010-Q1", "2025-Q4"),
    },
    "M3a_LCR": {
        "y":          "d_ln_leverage",
        "X":          ["d_ln_unit_var_lag", "d_ln_leverage_lag", "lcr_lag", "d_ln_unit_var_x_lcr", "roa_lag", "size_lag"],
        "bank_fe":    True,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2015-Q1", "2025-Q4"),
    },
    "M3b_CET1": {
        "y":          "d_ln_leverage",
        "X":          ["d_ln_unit_var_lag", "d_ln_leverage_lag", "cet1_lag", "d_ln_unit_var_x_cet1", "roa_lag", "size_lag"],
        "bank_fe":    True,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2015-Q1", "2025-Q4"),
    },
    "M3C_SLR": {
        "y":          "d_ln_leverage",
        "X":          ["d_ln_unit_var_lag", "d_ln_leverage_lag", "slr_lag", "d_ln_unit_var_x_slr", "roa_lag", "size_lag"],
        "bank_fe":    True,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2015-Q1", "2025-Q4"),
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  HJELPEFUNKSJONER
# ══════════════════════════════════════════════════════════════════════════════

def parse_date(s: str):
    s = str(s).strip()
    if "-Q" in s.upper():
        year, q = s.upper().split("-Q")
        quarter_start = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
        quarter_end   = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
        q = int(q)
        return pd.Timestamp(f"{year}-{quarter_start[q]}"), pd.Timestamp(f"{year}-{quarter_end[q]}")
    return pd.Timestamp(s), None


def resolve_date_range(date_range):
    if date_range is None:
        return None, None
    raw_start, raw_end = date_range
    start_ts, _ = parse_date(raw_start)
    end_ts, _   = parse_date(raw_end)
    if "-Q" in str(raw_end).upper():
        _, end_ts = parse_date(raw_end)
    return start_ts, end_ts


# ══════════════════════════════════════════════════════════════════════════════
#  BYGG VARIABLER
# ══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = (
    df
    .sort_values(["bank", "date"])
    .drop_duplicates(subset=["bank", "date"], keep="first")
    .reset_index(drop=True)
)

df["leverage"]         = df["total_assets"] / df["total_equity"]
df["ln_leverage"]      = np.log(df["leverage"].clip(lower=1e-12))
df["d_ln_leverage"]    = df.groupby("bank")["ln_leverage"].diff()

df["size"]             = np.log(df["total_assets"].clip(lower=1e-12))
df["market_dummy"]     = df["bank"].isin(MARKET_BANKS).astype(int)
df["custody_dummy"]    = df["bank"].isin(CUSTODY_BANKS).astype(int)
df["unit_var"]         = (df["total_var"] * 100) / df["total_assets"]
df["ln_unit_var"]      = np.log(df["unit_var"].clip(lower=1e-12))
df["d_ln_unit_var"]    = df.groupby("bank")["ln_unit_var"].diff()

df["lcr_lag"]           = df.groupby("bank")["lcr_ratio"].shift(1)
df["cet1_lag"]          = df.groupby("bank")["cet1_ratio"].shift(1)
df["slr_lag"]           = df.groupby("bank")["slr_ratio"].shift(1)
df["roa_lag"]           = df.groupby("bank")["roa"].shift(1)
df["size_lag"]          = df.groupby("bank")["size"].shift(1)
df["ln_unit_var_lag"]   = df.groupby("bank")["ln_unit_var"].shift(1)
df["d_ln_unit_var_lag"] = df.groupby("bank")["d_ln_unit_var"].shift(1)
df["d_size"]            = df.groupby("bank")["size"].diff()
df["d_roa"]             = df.groupby("bank")["roa"].diff()
df["d_size_lag"]        = df.groupby("bank")["d_size"].shift(1)
df["d_roa_lag"]         = df.groupby("bank")["d_roa"].shift(1)
df["d_ln_leverage_lag"] = df.groupby("bank")["d_ln_leverage"].shift(1)


df["d_ln_unit_var_x_market"]  = df["d_ln_unit_var_lag"] * df["market_dummy"]
df["d_ln_unit_var_x_custody"] = df["d_ln_unit_var_lag"] * df["custody_dummy"]
df["d_ln_unit_var_x_lcr"]     = df["d_ln_unit_var_lag"] * df["lcr_lag"]
df["d_ln_unit_var_x_cet1"]    = df["d_ln_unit_var_lag"] * df["cet1_lag"]
df["d_ln_unit_var_x_slr"]     = df["d_ln_unit_var_lag"] * df["slr_lag"]

df = df.set_index(["bank", "date"])

# ══════════════════════════════════════════════════════════════════════════════
#  KJØR REGRESJON
# ══════════════════════════════════════════════════════════════════════════════

def run_model(name: str, spec: dict, cov_type: str = "clustered") -> dict:
    """
    Kjører én panelregresjon.

    cov_type-valg (alle støttet av linearmodels):
      "clustered" → cluster_entity=True  — standard i panelmodeller med mulig
                    serial correlation innen bank; kan være skjør med få klynger
      "robust"    → White-robust SE       — robusthetssjekk 1
      "kernel"    → Driscoll–Kraay HAC   — robusthetssjekk 2, håndterer
                    både heteroskedastisitet og autokorrelasjon over tid

    Merk: modellen (koeffisientene) er identisk uansett cov_type.
    Kun standardfeil, t-verdier og p-verdier endres.
    """
    if cov_type not in ("clustered", "robust", "kernel"):
        raise ValueError(
            f"Ukjent cov_type: '{cov_type}'. "
            f"Gyldige valg: 'clustered', 'robust', 'kernel'."
        )

    y, X       = spec["y"], spec["X"]
    bank_fe    = spec["bank_fe"]
    time_fe    = spec["time_fe"]
    banks      = spec.get("banks")
    date_range = spec.get("date_range")

    cols    = [y] + X
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return {"name": name, "error": f"Ukjent kolonne(r): {missing}"}

    sub = df[cols].copy()
    if banks is not None:
        sub = sub[sub.index.get_level_values("bank").isin(banks)]

    start_ts, end_ts = resolve_date_range(date_range)
    dates = sub.index.get_level_values("date")
    if start_ts is not None:
        sub = sub[dates >= start_ts]
    if end_ts is not None:
        sub = sub[sub.index.get_level_values("date") <= end_ts]

    sub = sub.dropna()
    if len(sub) < 30:
        return {"name": name, "error": f"For få obs: {len(sub)}"}

    # ── PanelOLS (bank FE og/eller tid FE) ──
    if bank_fe or time_fe:
        fe_terms = (["EntityEffects"] if bank_fe else []) + (["TimeEffects"] if time_fe else [])
        formula  = f"{y} ~ {' + '.join(X)} + {' + '.join(fe_terms)}"

        if cov_type == "clustered":
            res = PanelOLS.from_formula(formula, data=sub).fit(
                cov_type="clustered", cluster_entity=True
            )
        elif cov_type == "robust":
            res = PanelOLS.from_formula(formula, data=sub).fit(
                cov_type="robust"
            )
        elif cov_type == "kernel":
            res = PanelOLS.from_formula(formula, data=sub).fit(
                cov_type="kernel"
            )

    # ── PooledOLS (ingen FE — kun M1_FD og M2_FD i praksis) ──
    else:
        formula = f"{y} ~ 1 + {' + '.join(X)}"

        if cov_type == "clustered":
            res = PooledOLS.from_formula(formula, data=sub).fit(
                cov_type="clustered", cluster_entity=True
            )
        elif cov_type == "robust":
            res = PooledOLS.from_formula(formula, data=sub).fit(
                cov_type="robust"
            )
        elif cov_type == "kernel":
            res = PooledOLS.from_formula(formula, data=sub).fit(
                cov_type="kernel"
            )

    actual_dates = sub.index.get_level_values("date")
    return {
        "name":    name,
        "spec":    spec,
        "result":  res,
        "n":       len(sub),
        "n_banks": sub.index.get_level_values("bank").nunique(),
        "period":  f"{actual_dates.min().date()} - {actual_dates.max().date()}",
    }


# ── Kjør alle modeller for hver SE-type ──
all_results = {
    label: [run_model(name, spec, cov_type=cov_type) for name, spec in MODELS.items()]
    for label, cov_type in COV_TYPES.items()
}

# ══════════════════════════════════════════════════════════════════════════════
#  BYGG OG LAGRE CSV
# ══════════════════════════════════════════════════════════════════════════════

def build_csv(results: list) -> pd.DataFrame:
    rows = []

    # ── Koeffisienter og t-statistikk ──
    all_params = []
    for r in results:
        if "result" in r:
            for p in r["result"].params.index:
                if p not in all_params:
                    all_params.append(p)

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
                b     = res.params[param]
                t     = res.tstats[param]
                p_val = res.pvalues[param]
                stars = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
                coef_row[r["name"]]  = f"{b:+.4f}{stars}"
                tstat_row[r["name"]] = f"({t:.2f})"
            else:
                coef_row[r["name"]]  = "–"
                tstat_row[r["name"]] = ""
        rows.append(coef_row)
        rows.append(tstat_row)

    # ── Metadata ──
    def meta_row(label, values):
        row = {"variable": label, "stat": ""}
        for r, v in zip(results, values):
            row[r["name"]] = v
        return row

    rows.append(meta_row("N (obs)",     [r.get("n",       "–") for r in results]))
    rows.append(meta_row("N (banker)",  [r.get("n_banks", "–") for r in results]))
    rows.append(meta_row("R² (within)", [
        f"{r['result'].rsquared_within:.4f}" if "result" in r else "–" for r in results
    ]))
    rows.append(meta_row("Bank FE",  ["Yes" if r.get("spec", {}).get("bank_fe") else "No" for r in results]))
    rows.append(meta_row("Tid FE",   ["Yes" if r.get("spec", {}).get("time_fe") else "No" for r in results]))
    rows.append(meta_row("Y",        [r.get("spec", {}).get("y", "–") for r in results]))
    rows.append(meta_row("Periode",  [r.get("period", "–") for r in results]))

    return pd.DataFrame(rows)


os.makedirs(OUTPUT_DIR, exist_ok=True)

# Filnavn-mapping: SE-label → filnavn
output_files = {
    "clustered": "regression_results_clustered_SE.csv",   # hovedresultat
    "robust":    "regression_results_robust_SE.csv",       # robusthetssjekk 1
    "kernel":    "regression_results_kernel_SE.csv",       # robusthetssjekk 2
}

for label, results in all_results.items():
    out_df   = build_csv(results)
    out_path = os.path.join(OUTPUT_DIR, output_files[label])
    out_df.to_csv(out_path, index=False)
    print(f"✓ {label:10s} → {out_path}  ({len(out_df)} rader, {len(out_df.columns)} kolonner)")

print("\nFerdig. Tre filer lagret:")
print("  clustered_SE  → hovedresultat  (rapporteres i tabell)")
print("  robust_SE     → robusthetssjekk 1  (White)")
print("  kernel_SE     → robusthetssjekk 2  (Driscoll–Kraay)")