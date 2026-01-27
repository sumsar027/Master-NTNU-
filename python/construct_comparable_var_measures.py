"""VaR harmonization using two conversion methods.

Target: express everything on a 99% VaR scale using two different approaches:
1. Gaussian quantile conversion (z_0.99 / z_0.95 ≈ 1.4144)
2. Bank of America empirical factor (2.0)

Hardcoded for this dataset:
- INPUT_FILE: VaR_python.xlsx
- DATE_COL: year
- OUTPUT_FILE: output/merged_with_var_99_harmonized.csv

Conversion methods:
- Gaussian: var_99 = var_95 * (z_0.99 / z_0.95) ≈ var_95 * 1.4144
- BoA Factor: var_99 = var_95 * 2.0

Run:
  python var_approx_to99_gaussian.py
"""

from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

# Fixed settings for this exact Excel setup (no CLI args; edit here if needed).
INPUT_FILE = Path("data/raw/VaR_python.xlsx")
DATE_COL = "year"
OUTPUT_FILE = Path("output/data/merged_with_var_99_dual_methods.csv")

# Standard normal quantiles for 95% and 99% confidence levels
Z_095 = 1.6448536269514722
Z_099 = 2.3263478740408408
GAUSSIAN_RATIO = Z_099 / Z_095  # Approximately 1.4144

# Bank of America empirical factor
BOA_FACTOR = 2.0


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
    # 4) Build TWO harmonized var_99 columns using different conversion methods:
    #    a) Gaussian: var_95 * (z_0.99 / z_0.95)
    #    b) BoA Factor: var_95 * 2.0
    # 5) Validate both methods on banks with both var_95 and var_99 and save CSV.
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

    # 4) Harmonize to 99% using TWO different conversion methods
    print("=" * 72)
    print("VaR harmonization using two conversion methods (target: 99%)")
    print("=" * 72)
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(
        f"Rows (panel): {len(panel):,}  Banks: {panel['bank_id'].nunique():,}  Quarters: {panel['period_end_date'].nunique():,}"
    )
    print("")
    print("[Method 1: Gaussian conversion parameters]")
    print(f"z_0.95 = {Z_095:.16f}")
    print(f"z_0.99 = {Z_099:.16f}")
    print(f"Ratio (z_0.99 / z_0.95) = {GAUSSIAN_RATIO:.16f}")
    print("Conversion: var_99_gaussian = var_95 * (z_0.99 / z_0.95)")
    print("")
    print("[Method 2: Bank of America empirical factor]")
    print(f"BoA Factor = {BOA_FACTOR:.1f}")
    print("Conversion: var_99_boa_factor = var_95 * 2.0")

    out = panel.copy()
    has_95 = out["var_95"].notna()
    has_99 = out["var_99"].notna()

    # 5a) Gaussian harmonization (Method 1)
    out["var_99_gaussian"] = np.nan
    out.loc[has_99, "var_99_gaussian"] = out.loc[has_99, "var_99"]
    out.loc[~has_99 & has_95, "var_99_gaussian"] = GAUSSIAN_RATIO * out.loc[~has_99 & has_95, "var_95"]

    out["var_99_source_gaussian"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out.loc[has_99, "var_99_source_gaussian"] = "reported_99"
    out.loc[~has_99 & has_95, "var_99_source_gaussian"] = "approx_from_95_gaussian"
    out.loc[~has_99 & ~has_95, "var_99_source_gaussian"] = "missing_both"

    # 5b) BoA Factor harmonization (Method 2)
    out["var_99_boa_factor"] = np.nan
    out.loc[has_99, "var_99_boa_factor"] = out.loc[has_99, "var_99"]
    out.loc[~has_99 & has_95, "var_99_boa_factor"] = BOA_FACTOR * out.loc[~has_99 & has_95, "var_95"]

    out["var_99_source_boa"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out.loc[has_99, "var_99_source_boa"] = "reported_99"
    out.loc[~has_99 & has_95, "var_99_source_boa"] = "approx_from_95_boa"
    out.loc[~has_99 & ~has_95, "var_99_source_boa"] = "missing_both"

    # 5c) Backward compatibility: keep var_99_harmonized as Gaussian version
    out["var_99_harmonized"] = out["var_99_gaussian"].copy()
    out["var_99_source"] = out["var_99_source_gaussian"].copy()

    # Dummy: 1 if reported_99, 0 if approx_from_95 (NaN if missing both).
    out["var_level_dummy_99"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out.loc[has_99, "var_level_dummy_99"] = 1
    out.loc[~has_99 & has_95, "var_level_dummy_99"] = 0

    # Validation on banks with both var_95 and var_99:
    val_rows = out[has_95 & has_99].copy()
    
    if val_rows.empty:
        print("\n[validation] No rows with both var_95 and var_99 found.")
    else:
        # Validation for Gaussian method
        val_rows["var_99_hat_gaussian"] = GAUSSIAN_RATIO * val_rows["var_95"]
        abs_pct_error_gaussian = ((val_rows["var_99_hat_gaussian"] - val_rows["var_99"]) / val_rows["var_99"]).abs()
        abs_pct_error_gaussian = abs_pct_error_gaussian.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Validation for BoA Factor method
        val_rows["var_99_hat_boa"] = BOA_FACTOR * val_rows["var_95"]
        abs_pct_error_boa = ((val_rows["var_99_hat_boa"] - val_rows["var_99"]) / val_rows["var_99"]).abs()
        abs_pct_error_boa = abs_pct_error_boa.replace([np.inf, -np.inf], np.nan).dropna()
        
        if abs_pct_error_gaussian.empty:
            print("\n[validation - Gaussian] Absolute percent error is empty after dropping non-finite values.")
        else:
            print("\n[validation - Gaussian method] on banks with both var_95 and var_99:")
            print("var_99_hat = var_95 * (z_0.99 / z_0.95)")
            print(
                "n={:,}  median_abs_pct_error={:.6g}  p95_abs_pct_error={:.6g}".format(
                    int(abs_pct_error_gaussian.shape[0]),
                    float(abs_pct_error_gaussian.median()),
                    float(abs_pct_error_gaussian.quantile(0.95)),
                )
            )
        
        if abs_pct_error_boa.empty:
            print("\n[validation - BoA Factor] Absolute percent error is empty after dropping non-finite values.")
        else:
            print("\n[validation - BoA Factor method] on banks with both var_95 and var_99:")
            print("var_99_hat = var_95 * 2.0")
            print(
                "n={:,}  median_abs_pct_error={:.6g}  p95_abs_pct_error={:.6g}".format(
                    int(abs_pct_error_boa.shape[0]),
                    float(abs_pct_error_boa.median()),
                    float(abs_pct_error_boa.quantile(0.95)),
                )
            )
        
        # Show which banks have both values
        banks_with_both = val_rows["bank_id"].unique()
        print(f"\nBanks with both values: {', '.join(sorted(banks_with_both))}")

    # Count observations by conversion method
    n_reported = int(has_99.sum())
    n_approx_gaussian = int((~has_99 & has_95).sum())
    n_approx_boa = int((~has_99 & has_95).sum())  # Same as Gaussian
    n_missing_both = int((~has_99 & ~has_95).sum())
    
    print("")
    print("[counts by conversion method]")
    print(f"reported_99 (used as-is in both methods): {n_reported:,}")
    print(f"approx_from_95_gaussian (converted using {GAUSSIAN_RATIO:.4f}): {n_approx_gaussian:,}")
    print(f"approx_from_95_boa (converted using {BOA_FACTOR:.1f}): {n_approx_boa:,}")
    print(f"missing_both (no data available): {n_missing_both:,}")

    # 6) Save to CSV - UPDATED COLUMN ORDER
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out[
        [
            "bank_id",
            "period_end_date",
            "var_95",
            "var_99",  # Original 99% values (when reported)
            "var_99_gaussian",  # Harmonized using Gaussian method
            "var_99_boa_factor",  # Harmonized using BoA factor
            "var_99_source_gaussian",
            "var_99_source_boa",
            "var_level_dummy_99",
        ]
    ].to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Output columns: bank_id, period_end_date, var_95, var_99, var_99_gaussian, var_99_boa_factor, var_99_source_gaussian, var_99_source_boa, var_level_dummy_99")
    print("\nColumn descriptions:")
    print("  - var_95: Original 95% VaR (when reported)")
    print("  - var_99: Original 99% VaR (when reported)")
    print("  - var_99_gaussian: Harmonized 99% VaR using Gaussian conversion (×1.4144)")
    print("  - var_99_boa_factor: Harmonized 99% VaR using BoA factor (×2.0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())