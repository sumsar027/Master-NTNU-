import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS, PooledOLS

# ══════════════════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH       = "dataframe/dataframe.csv"
POST_BASEL_YEAR = 2014
MARKET_BANKS    = ["goldmansachs", "morganstanley"]
UNIVERSAL_BANKS = ["jpmorgan", "bankofamerica", "citigroup", "wellsfargo"]
CUSTODY_BANKS   = ["statestreet", "bny"]

# Alle tilgjengelige banker:
# bankofamerica, bny, citigroup, goldmansachs, jpmorgan,
# morganstanley, statestreet, wellsfargo

# ══════════════════════════════════════════════════════════════════════════════
#  MODELLER
# ══════════════════════════════════════════════════════════════════════════════

MODELS = {
    # ─────────────────────────────
    # Baseline
    # ─────────────────────────────
   "M1_basel_test": {
    "y":          "d_ln_leverage",
    "X":          ["d_ln_unit_var_lag", "size_lag", "roa_lag"],
    "bank_fe":    True,
    "time_fe":    True,
    "banks":      None,
    "date_range": ("2010-Q1", "2025-Q4"),
    },
   
   # ─────────────────────────────
   # Baseline ROBUST
    # ─────────────────────────────
    "M1_FD_baseline": {
    "y":          "d_ln_leverage",
    "X":          ["d_ln_unit_var_lag", "d_size_lag", "d_roa_lag"],
    "bank_fe":    False,
    "time_fe":    True,
    "banks":      None,
    "date_range": ("2010-Q1", "2025-Q4"),
},

    # ─────────────────────────────
    # LCR
    # ─────────────────────────────
    "M2_dummy": {
        "y":          "d_ln_leverage",
        "X":          ["d_ln_unit_var_lag", "d_ln_unit_var_x_market","d_ln_unit_var_x_custody", "size_lag", "roa_lag"],
        "bank_fe":    True,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2010-Q1", "2025-Q4"),
    },
    # ─────────────────────────────
    # H2 ROBUST
    "M2_FD_dummy": {
    "y":          "d_ln_leverage",
    "X":          ["d_ln_unit_var_lag", "d_ln_unit_var_x_market", "d_ln_unit_var_x_custody", "d_size_lag", "d_roa_lag"],
    "bank_fe":    False,
    "time_fe":    True,
    "banks":      None,
    "date_range": ("2010-Q1", "2025-Q4"),
},

    # ─────────────────────────────
    # CET1
    # ─────────────────────────────
    "M3a_LCR": {
        "y":          "d_ln_leverage",
        "X":          ["d_ln_unit_var_lag", "lcr_lag", "d_ln_unit_var_x_lcr", "roa_lag", "size_lag"],
        "bank_fe":    True,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2014-Q1", "2025-Q4"),
    },

    # ─────────────────────────────
    # SLR
    # ─────────────────────────────
    "M3b_CET1": {
        "y":          "d_ln_leverage",
        "X":          ["d_ln_unit_var_lag", "cet1_lag", "d_ln_unit_var_x_cet1", "roa_lag", "size_lag"],
        "bank_fe":    True,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2014-Q1", "2025-Q4"),
    },

    # ─────────────────────────────
    # Dummy
    # ─────────────────────────────
    "M3C_SLR": {
        "y":          "d_ln_leverage",
        "X":          ["d_ln_unit_var_lag", "slr_lag", "d_ln_unit_var_x_slr", "roa_lag", "size_lag"],
        "bank_fe":    True,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2014-Q1", "2025-Q4"),
    },
    "M4_Equity": {
        "y":          "d_ln_book_equity",
        "X":          ["payout_lag",  "size_lag", "roa_lag"],
        "bank_fe":    True,
        "time_fe":    True,
        "banks":      None,
        "date_range": ("2014-Q1", "2025-Q4"),
    }
}

# ══════════════════════════════════════════════════════════════════════════════
#  HJELPEFUNKSJONER
# ══════════════════════════════════

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


def make_balanced(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Behold kun datoer der alle gjenværende banker har observasjoner.
    Itererer til panelet er fullt balansert (en bank kan forsvinne etter
    at en dato droppes, og omvendt).
    """
    while True:
        n_banks = sub.index.get_level_values("bank").nunique()

        # Datoer der alle banker er til stede
        date_counts = sub.groupby(level="date").size()
        valid_dates = date_counts[date_counts == n_banks].index
        sub = sub[sub.index.get_level_values("date").isin(valid_dates)]

        # Sjekk om noen banker nå mangler observasjoner (kan skje etter datofiltrering)
        bank_counts = sub.groupby(level="bank").size()
        max_obs     = bank_counts.max()
        valid_banks = bank_counts[bank_counts == max_obs].index
        sub = sub[sub.index.get_level_values("bank").isin(valid_banks)]

        # Ferdig når ingenting endrer seg
        if sub.index.get_level_values("bank").nunique() == n_banks:
            break

    return sub


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

# Basis
df["leverage"] = df["total_assets"] / df["total_equity"]
df["ln_leverage"] = np.log(df["leverage"].clip(lower=1e-12))
df["d_ln_leverage"] = df.groupby("bank")["ln_leverage"].diff()

df["d_ln_book_equity"] = np.log(df["total_equity"].clip(lower=1e-6)).groupby(df["bank"]).diff()

df["size"] = np.log(df["total_assets"].clip(lower=1e-12))
df["market_dummy"] = df["bank"].isin(MARKET_BANKS).astype(int)
df["custody_dummy"] = df["bank"].isin(CUSTODY_BANKS).astype(int)

df["unit_var"] = (df["total_var"] * 100) / df["total_assets"]
df["ln_unit_var"] = np.log(df["unit_var"].clip(lower=1e-12))
df["d_ln_unit_var"] = df.groupby("bank")["ln_unit_var"].diff()

# Laggede nivåvariabler
df["lcr_lag"]   = df.groupby("bank")["lcr_ratio"].shift(1)
df["cet1_lag"]  = df.groupby("bank")["cet1_ratio"].shift(1)
df["slr_lag"]   = df.groupby("bank")["slr_ratio"].shift(1)
df["roa_lag"]   = df.groupby("bank")["roa"].shift(1)
df["roe_lag"]   = df.groupby("bank")["roe"].shift(1)
df["size_lag"]  = df.groupby("bank")["size"].shift(1)

df["ln_unit_var_lag"] = df.groupby("bank")["ln_unit_var"].shift(1)
df["d_ln_unit_var_lag"] = df.groupby("bank")["d_ln_unit_var"].shift(1)

# Differensierte kontroller til FD-robusthet
df["d_size"] = df.groupby("bank")["size"].diff()
df["d_roa"]  = df.groupby("bank")["roa"].diff()

df["d_size_lag"] = df.groupby("bank")["d_size"].shift(1)
df["d_roa_lag"]  = df.groupby("bank")["d_roa"].shift(1)
df["payout_lag"] = df["dividend_payout_ratio"].groupby(df["bank"]).shift(1)

# Interaksjoner
df["d_ln_unit_var_x_market"]  = df["d_ln_unit_var_lag"] * df["market_dummy"]
df["d_ln_unit_var_x_custody"] = df["d_ln_unit_var_lag"] * df["custody_dummy"]
df["d_ln_unit_var_x_lcr"]     = df["d_ln_unit_var_lag"] * df["lcr_lag"]
df["d_ln_unit_var_x_cet1"]    = df["d_ln_unit_var_lag"] * df["cet1_lag"]
df["d_ln_unit_var_x_slr"]     = df["d_ln_unit_var_lag"] * df["slr_lag"]


df = df.set_index(["bank", "date"])

# ══════════════════════════════════════════════════════════════════════════════
#  KJØR REGRESJON
# ══════════════════════════════════════════════════════════════════════════════

def run_model(name: str, spec: dict) -> dict:
    y          = spec["y"]
    X          = spec["X"]
    bank_fe    = spec["bank_fe"]
    time_fe    = spec["time_fe"]
    banks      = spec.get("banks")
    date_range = spec.get("date_range")

    cols = [y] + X
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return {"name": name, "error": f"Ukjent kolonne(r): {missing}"}

    sub = df[cols].copy()

    # Filtrer på banker
    if banks is not None:
        sub = sub[sub.index.get_level_values("bank").isin(banks)]

    # Filtrer på tidsperiode
    start_ts, end_ts = resolve_date_range(date_range)
    dates = sub.index.get_level_values("date")
    if start_ts is not None:
        sub = sub[dates >= start_ts]
    if end_ts is not None:
        dates = sub.index.get_level_values("date")
        sub = sub[dates <= end_ts]

    # Dropp NaN
    sub = sub.dropna()

    if len(sub) < 30:
        return {"name": name, "error": f"For få obs: {len(sub)}"}

    if bank_fe or time_fe:
        fe_terms = (["EntityEffects"] if bank_fe else []) + (["TimeEffects"] if time_fe else [])
        formula = f"{y} ~ {' + '.join(X)} + {' + '.join(fe_terms)}"
        res = PanelOLS.from_formula(formula, data=sub).fit(
            cov_type="clustered", cluster_entity=True
        )
    else:
        formula = f"{y} ~ 1 + {' + '.join(X)}"
        res = PooledOLS.from_formula(formula, data=sub).fit(
            cov_type="clustered", cluster_entity=True
        )

    n_banks    = sub.index.get_level_values("bank").nunique()
    actual_dates = sub.index.get_level_values("date")
    period_str = f"{actual_dates.min().date()} – {actual_dates.max().date()}"

    return {
        "name":    name,
        "spec":    spec,
        "result":  res,
        "n":       len(sub),
        "n_banks": n_banks,
        "period":  period_str,
    }


results = [run_model(name, spec) for name, spec in MODELS.items()]

# ══════════════════════════════════════════════════════════════════════════════
#  PRINT TABELL
# ══════════════════════════════════════════════════════════════════════════════

col_w   = 18
name_w  = 24
total_w = name_w + len(results) * (col_w + 2)

all_params = []
for r in results:
    if "result" in r:
        for p in r["result"].params.index:
            if p not in all_params:
                all_params.append(p)

print(f"\n{'═' * total_w}")
print("  Avhengig variabel: se Y per modell  |  Clustered SE (bank)  |  t i parentes")
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
            p = res.pvalues[param]
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
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
        try:
            r2_row += f"  {r['result'].rsquared_within:>{col_w}.4f}"
        except Exception:
            r2_row += f"  {r['result'].rsquared:>{col_w}.4f}"
    else:
        r2_row += f"  {'–':>{col_w}}"
print(r2_row)

for fe_label, fe_key in [("Bank FE", "bank_fe"), ("Tid FE", "time_fe")]:
    fe_row = f"  {fe_label:>{name_w - 2}}"
    for r in results:
        val = "Yes" if ("spec" in r and r["spec"].get(fe_key)) else "No"
        fe_row += f"  {val:>{col_w}}"
    print(fe_row)

banks_row = f"  {'Banker':>{name_w - 2}}"
for r in results:
    if "spec" not in r:
        banks_row += f"  {'–':>{col_w}}"
        continue
    b = r["spec"].get("banks")
    val = "Alle" if b is None else f"{len(b)} valgt"
    banks_row += f"  {val:>{col_w}}"
print(banks_row)

period_row = f"  {'Periode':>{name_w - 2}}"
for r in results:
    val = r.get("period", "–")
    if len(val) > col_w:
        val = val[:col_w]
    period_row += f"  {val:>{col_w}}"
print(period_row)

print("\n  * p<0.10  ** p<0.05  *** p<0.01\n")