"""Simple VaR approximation for the provided Excel (wide) sheet.

Target: express everything on a 95% VaR scale by approximating 95% from 99%.

Hardcoded for this dataset:
- INPUT_FILE: VaR_python.xlsx
- DATE_COL: year
- CALIB_BANK_ID: bank_of_america (the only bank with both 95% and 99%)
- OUTPUT_FILE: output/merged_with_var_95_approx.csv

Run:
  python var_approx_to95_simple.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Fixed settings for this exact Excel setup (no CLI args; edit here if needed).
INPUT_FILE = Path("VaR_python.xlsx")
DATE_COL = "year"
CALIB_BANK_ID = "bank_of_america"
OUTPUT_FILE = Path("output/merged_with_var_95_approx.csv")


def snake_case(value: object) -> str:
    # Convert a label to a simple ID:
    # "Bank of America" -> "bank_of_america"
    text = "" if value is None else str(value)
    text = text.strip().lower()
    out: list[str] = []
    prev_underscore = False
    for ch in text:
        is_alnum = ("a" <= ch <= "z") or ("0" <= ch <= "9")
        if is_alnum:
            out.append(ch)
            prev_underscore = False
        else:
            if not prev_underscore:
                out.append("_")
                prev_underscore = True
    return "".join(out).strip("_")


def to_quarter_end(value: object) -> pd.Timestamp:
    # Parse the "year" column into dates and normalize to quarter-end dates.
    # Example: any date in Q3 2025 -> 2025-09-30
    if value is None or pd.isna(value):
        return pd.NaT
    dt = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        dt = pd.to_datetime(value, errors="coerce", dayfirst=False)
    if pd.isna(dt):
        return pd.NaT
    return pd.Period(pd.Timestamp(dt), freq="Q").end_time.normalize()


# Matches columns like:
# - goldmansachs_95
# - wells_fargo_99
# - Bank_of_America_99%
# Anything not matching this is ignored (e.g. "Add-on citi").
_VAR_COL_RE = re.compile(r"^(?P<bank>.+?)[_\s-]*(?P<level>95|99)\s*%?$", flags=re.IGNORECASE)


def main() -> int:
    # Plain-English workflow:
    # 1) Read the Excel file (wide format).
    # 2) Keep only VaR columns ending in 95/99 (with or without %).
    # 3) Reshape to a panel: one row per (bank, quarter).
    # 4) Estimate r from Bank of America: r = median(var_95 / var_99).
    # 5) Build var_95_approx for all banks:
    #    - if reported var_95 exists -> use it
    #    - else if only var_99 exists -> approximate: r * var_99
    # 6) Validate on Bank of America and save CSV.
    if not INPUT_FILE.exists():
        raise SystemExit(f"Missing input file: {INPUT_FILE}")

    # 1) Read Excel
    df_wide = pd.read_excel(INPUT_FILE)
    df_wide.columns = [str(c).strip() for c in df_wide.columns]

    if DATE_COL not in df_wide.columns:
        raise SystemExit(f"Missing date column {DATE_COL!r} in {INPUT_FILE}")

    # 2) Identify VaR columns (ignore everything else automatically)
    var_cols: list[str] = []
    for col in df_wide.columns:
        if col == DATE_COL:
            continue
        if _VAR_COL_RE.match(str(col)):
            var_cols.append(col)

    if not var_cols:
        raise SystemExit("Found 0 VaR columns ending in _95/_99 or 95%/99%.")

    # 3) Wide -> long:
    # We create rows like: (period_end_date, bank_id, level, value)
    melted = df_wide.melt(id_vars=[DATE_COL], value_vars=var_cols, var_name="var_col", value_name="var_value")
    extracted = melted["var_col"].astype(str).str.extract(_VAR_COL_RE)
    melted["bank_id"] = extracted["bank"].map(snake_case)
    melted["level"] = extracted["level"].astype("Int64")
    melted["period_end_date"] = melted[DATE_COL].map(to_quarter_end)
    melted = melted.dropna(subset=["bank_id", "level", "period_end_date"]).copy()

    # Then pivot to get one row per (bank_id, period_end_date) with two columns:
    # var_95 and var_99 (when present).
    panel = (
        melted.pivot(index=["bank_id", "period_end_date"], columns="level", values="var_value")
        .rename_axis(None, axis=1)
        .reset_index()
    )
    panel = panel.rename(columns={95: "var_95", 99: "var_99"})
    if "var_95" not in panel.columns:
        panel["var_95"] = np.nan
    if "var_99" not in panel.columns:
        panel["var_99"] = np.nan

    panel["var_95"] = pd.to_numeric(panel["var_95"], errors="coerce")
    panel["var_99"] = pd.to_numeric(panel["var_99"], errors="coerce")
    panel = panel.sort_values(["bank_id", "period_end_date"]).reset_index(drop=True)

    # 4) Estimate r using Bank of America (the calibration bank).
    # r_t = var_95 / var_99 for each quarter where both values exist.
    calib = panel[panel["bank_id"] == CALIB_BANK_ID].copy()
    calib = calib[calib["var_95"].notna() & calib["var_99"].notna()].copy()
    calib = calib[(calib["var_95"] != 0) & (calib["var_99"] != 0)].copy()
    if calib.empty:
        raise SystemExit(f"No calibration rows found for {CALIB_BANK_ID!r} with both var_95 and var_99.")

    calib["r_t"] = calib["var_95"] / calib["var_99"]
    r_series = calib["r_t"].replace([np.inf, -np.inf], np.nan).dropna()
    if r_series.empty:
        raise SystemExit("r_t is empty after dropping non-finite values.")

    r = float(r_series.median())

    print("=" * 72)
    print("Simple VaR approximation (target: 95%)")
    print("=" * 72)
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Calibration bank_id: {CALIB_BANK_ID}")
    print(
        f"Rows (panel): {len(panel):,}  Banks: {panel['bank_id'].nunique():,}  Quarters: {panel['period_end_date'].nunique():,}"
    )
    print("")
    print("[r estimation]  r_t = var_95 / var_99  (Bank of America only)")
    print(
        "n={:,}  median={:.6g}  mean={:.6g}  min={:.6g}  max={:.6g}".format(
            int(r_series.shape[0]),
            float(r_series.median()),
            float(r_series.mean()),
            float(r_series.min()),
            float(r_series.max()),
        )
    )

    out = panel.copy()
    has_95 = out["var_95"].notna()
    has_99 = out["var_99"].notna()

    # 5) Main output variable on a 95% scale:
    # - Prefer reported 95% values.
    # - If missing, approximate 95% from 99% using r.
    out["var_95_approx"] = np.nan
    out.loc[has_95, "var_95_approx"] = out.loc[has_95, "var_95"]
    out.loc[~has_95 & has_99, "var_95_approx"] = r * out.loc[~has_95 & has_99, "var_99"]

    # Tag where var_95_approx comes from (reported vs approximated).
    out["var_95_source"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out.loc[has_95, "var_95_source"] = "reported_95"
    out.loc[~has_95 & has_99, "var_95_source"] = "approx_from_99"

    # Dummy: 1 if reported_95, 0 if approx_from_99 (NaN if missing both).
    out["var_level_dummy_95"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out.loc[has_95, "var_level_dummy_95"] = 1
    out.loc[~has_95 & has_99, "var_level_dummy_95"] = 0

    # Validation (Bank of America only): predict var_95_hat = r * var_99 and compare to reported var_95
    calib_val = out[out["bank_id"] == CALIB_BANK_ID].copy()
    calib_val = calib_val[calib_val["var_95"].notna() & calib_val["var_99"].notna()].copy()
    if calib_val.empty:
        print("\n[validation] No rows with both var_95 and var_99 for Bank of America (after parsing).")
    else:
        calib_val["var_95_hat"] = r * calib_val["var_99"]
        abs_pct_error = ((calib_val["var_95_hat"] - calib_val["var_95"]) / calib_val["var_95"]).abs()
        abs_pct_error = abs_pct_error.replace([np.inf, -np.inf], np.nan).dropna()
        if abs_pct_error.empty:
            print("\n[validation] Absolute percent error is empty after dropping non-finite values.")
        else:
            print("\n[validation] on Bank of America: var_95_hat = r * var_99")
            print(
                "median_abs_pct_error={:.6g}  p95_abs_pct_error={:.6g}".format(
                    float(abs_pct_error.median()),
                    float(abs_pct_error.quantile(0.95)),
                )
            )

    n_approx = int((~has_95 & has_99).sum())
    n_reported = int(has_95.sum())
    n_missing_both = int((~has_95 & ~has_99).sum())
    print("")
    print("[counts]")
    print(f"reported_95: {n_reported:,}")
    print(f"approx_from_99: {n_approx:,}")
    print(f"missing_both: {n_missing_both:,}")

    # 7) Save to CSV (one row per bank per quarter).
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out[
        [
            "bank_id",
            "period_end_date",
            "var_95",
            "var_99",
            "var_95_approx",
            "var_95_source",
            "var_level_dummy_95",
        ]
    ].to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
