import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS, PooledOLS

# ══════════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH       = "dataframe/dataframe.csv"
POST_BASEL_YEAR = 2015
MARKET_BANKS = ["goldmansachs", "morganstanley"]


# Alle tilgjengelige banker:
# bankofamerica, bny, citigroup, goldmansachs, jpmorgan,
# morganstanley, statestreet, wellsfargo

# ══════════════════════════════════════════════════════════════════════════════
#  MODELLER
# ══════════════════════════════════════════════════════════════════════════════
# "banks": liste med banker som skal inkluderes i akkurat denne modellen.
#          Utelat nøkkelen (eller sett None) for å bruke alle banker.


MODELS = {
    # ─────────────────────────────
    # Baseline
    # ─────────────────────────────
    "M1": {
        "y":       "d_leverage",
        "X":       ["unit_var_lag", "size", "roa_lag"],
        "bank_fe": True,
        "time_fe": True,
        "banks":   None,
    },

    # ─────────────────────────────
    # LCR (hovedmodell)
    # ─────────────────────────────
    "M2_LCR": {
        "y":       "d_leverage",
        "X":       ["unit_var_lag", "lcr_lag", "var_x_lcr", "size", "roa_lag"],
        "bank_fe": True,
        "time_fe": True,
        "banks":   None,
    },

    # ─────────────────────────────
    # CET1 (kapital)
    # ─────────────────────────────
    "M3_CET1": {
        "y":       "d_leverage",
        "X":       ["unit_var_lag", "cet1_lag", "var_x_cet1", "size", "roa_lag"],
        "bank_fe": True,
        "time_fe": True,
        "banks":   None,
    },

    # ─────────────────────────────
    # SLR (leverage constraint)
    # ─────────────────────────────
    "M4_SLR": {
        "y":       "d_leverage",
        "X":       ["unit_var_lag", "slr_lag", "var_x_slr", "size", "roa_lag"],
        "bank_fe": True,
        "time_fe": True,
        "banks":   None,
    },

    # ─────────────────────────────
    # Horse race: LCR vs CET1
    # ─────────────────────────────
    "M5_LCR_CET1": {
        "y":       "d_leverage",
        "X":       ["unit_var_lag", "lcr_lag", "var_x_lcr",
                    "cet1_lag", "var_x_cet1",
                    "size", "roa_lag"],
        "bank_fe": True,
        "time_fe": True,
        "banks":   None,
    },

    # ─────────────────────────────
    # Horse race: LCR vs SLR
    # ─────────────────────────────
    "M6_LCR_SLR": {
        "y":       "d_leverage",
        "X":       ["unit_var_lag", "lcr_lag", "var_x_lcr",
                    "slr_lag", "var_x_slr",
                    "size", "roa_lag"],
        "bank_fe": True,
        "time_fe": True,
        "banks":   None,
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  BYGG VARIABLER
# ══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["bank", "date"])

# Basis
df["leverage"]       = df["total_liabilities"] / df["total_equity"]
df["d_leverage"]     = df.groupby("bank")["leverage"].diff()
df["log_leverage"]   = np.log(df["leverage"].clip(lower=1e-6))
df["d_log_leverage"] = df.groupby("bank")["log_leverage"].diff()
df["size"]           = np.log(df["total_assets"])
df["unit_var"]       = df["total_var"] / df["total_assets"]
df["market_dummy"] = df["bank"].isin(MARKET_BANKS).astype(int)

# Laggede variabler
df["var_lag"]      = df.groupby("bank")["total_var"].shift(1)
df["lcr_lag"]      = df.groupby("bank")["lcr_ratio"].shift(1)
df["cet1_lag"]     = df.groupby("bank")["cet1_ratio"].shift(1)
df["slr_lag"]      = df.groupby("bank")["slr_ratio"].shift(1)
df["roa_lag"]      = df.groupby("bank")["roa"].shift(1)
df["unit_var_lag"] = df.groupby("bank")["unit_var"].shift(1)

# Interaksjonsledd
df["post_basel"] = (df["year"] >= POST_BASEL_YEAR).astype(int)
df["var_x_post"] = df["unit_var_lag"] * df["post_basel"]
df["var_x_lcr"]  = df["unit_var_lag"] * df["lcr_lag"]
df["var_x_cet1"] = df["unit_var_lag"] * df["cet1_lag"]
df["var_x_slr"]  = df["unit_var_lag"] * df["slr_lag"]
df["var_x_market"] = df["unit_var_lag"] * df["market_dummy"]

# Legg til egne variabler her ↓
# df["var_x_roa"] = df["unit_var_lag"] * df["roa"]

df = df.set_index(["bank", "date"])

# ══════════════════════════════════════════════════════════════════════════════
#  KJØR REGRESJON
# ══════════════════════════════════════════════════════════════════════════════

def run_model(name: str, spec: dict) -> dict:
    y       = spec["y"]
    X       = spec["X"]
    bank_fe = spec["bank_fe"]
    time_fe = spec["time_fe"]
    banks   = spec.get("banks")   # None = alle banker

    cols = [y] + X
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return {"name": name, "error": f"Ukjent kolonne(r): {missing}"}

    sub = df[cols].dropna()

    # Filtrer på banker om spesifisert
    if banks is not None:
        sub = sub[sub.index.get_level_values("bank").isin(banks)]

    if len(sub) < 30:
        return {"name": name, "error": f"For få obs: {len(sub)}"}

    if bank_fe or time_fe:
        fe_terms = (["EntityEffects"] if bank_fe else []) + (["TimeEffects"] if time_fe else [])
        formula  = f"{y} ~ {' + '.join(X)} + {' + '.join(fe_terms)}"
        res = PanelOLS.from_formula(formula, data=sub).fit(
            cov_type="clustered", cluster_entity=True
        )
    else:
        formula = f"{y} ~ 1 + {' + '.join(X)}"
        res = PooledOLS.from_formula(formula, data=sub).fit(
            cov_type="clustered", cluster_entity=True
        )

    n_banks = sub.index.get_level_values("bank").nunique()
    return {"name": name, "spec": spec, "result": res, "n": len(sub), "n_banks": n_banks}


results = [run_model(name, spec) for name, spec in MODELS.items()]

# ══════════════════════════════════════════════════════════════════════════════
#  PRINT TABELL
# ══════════════════════════════════════════════════════════════════════════════

col_w  = 18
name_w = 24
total_w = name_w + len(results) * (col_w + 2)

all_params = []
for r in results:
    if "result" in r:
        for p in r["result"].params.index:
            if p not in all_params:
                all_params.append(p)

print(f"\n{'═' * total_w}")
print(f"  Avhengig variabel: se Y per modell  |  Clustered SE (bank)  |  t i parentes")
print(f"{'═' * total_w}\n")

hdr = f"{'':>{name_w}}"
for r in results:
    hdr += f"  {r['name']:>{col_w}}"
print(hdr)

y_hdr = f"  {'Y:':>{name_w - 2}}"
for r in results:
    y_val = r["spec"]["y"] if "spec" in r else "–"
    y_hdr += f"  {y_val:>{col_w}}"
print(y_hdr)

print("─" * total_w)

for param in all_params:
    coef_row  = f"  {param:<{name_w - 2}}"
    tstat_row = f"  {'':>{name_w - 2}}"
    for r in results:
        if "result" not in r:
            coef_row  += f"  {'–':>{col_w}}"
            tstat_row += f"  {'':>{col_w}}"
            continue
        res = r["result"]
        if param in res.params.index:
            b = res.params[param]
            t = res.tstats[param]
            stars = "***" if abs(t) > 3.29 else "**" if abs(t) > 2.58 else "*" if abs(t) > 1.96 else ""
            coef_row  += f"  {f'{b:+.4f}{stars}':>{col_w}}"
            tstat_row += f"  {f'({t:.2f})':>{col_w}}"
        else:
            coef_row  += f"  {'–':>{col_w}}"
            tstat_row += f"  {'':>{col_w}}"
    print(coef_row)
    print(tstat_row)
    print()

print("─" * total_w)

for label, key in [("N (obs)", "n"), ("N (banker)", "n_banks")]:
    row = f"  {label:>{name_w - 2}}"
    for r in results:
        row += f"  {r.get(key, '–'):>{col_w}}"
    print(row)

r2_row = f"  {'R² (within)':>{name_w - 2}}"
for r in results:
    if "result" in r:
        try:    r2_row += f"  {r['result'].rsquared_within:>{col_w}.4f}"
        except: r2_row += f"  {r['result'].rsquared:>{col_w}.4f}"
    else:
        r2_row += f"  {'–':>{col_w}}"
print(r2_row)

for fe_label, fe_key in [("Bank FE", "bank_fe"), ("Tid FE", "time_fe")]:
    fe_row = f"  {fe_label:>{name_w - 2}}"
    for r in results:
        val = "Yes" if ("spec" in r and r["spec"].get(fe_key)) else "No"
        fe_row += f"  {val:>{col_w}}"
    print(fe_row)

# Bankutvalg-rad
banks_row = f"  {'Banker':>{name_w - 2}}"
for r in results:
    if "spec" not in r:
        banks_row += f"  {'–':>{col_w}}"
        continue
    b = r["spec"].get("banks")
    val = "Alle" if b is None else f"{len(b)} valgt"
    banks_row += f"  {val:>{col_w}}"
print(banks_row)

print(f"\n  * p<0.10  ** p<0.05  *** p<0.01  (|t| > 1.96 / 2.58 / 3.29)\n")