"""VaR harmonization using two conversion methods.

Target: express everything on a 99% VaR scale using two different approaches:
1. Gaussian quantile conversion (z_0.99 / z_0.95 ≈ 1.4144)
2. Bank of America empirical factor (2.0)

Run:
  python var_harmonize_balanced.py
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd

# ============ KONFIGURASJON ============
INPUT_FILE = Path("data/raw/VaR_python.xlsx")
DATE_COL = "year"
OUTPUT_FILE = Path("output/data/merged_with_var_99_dual_methods.csv")

# Standard normal quantiles for 95% and 99% confidence levels
Z_095 = 1.6448536269514722
Z_099 = 2.3263478740408408
GAUSSIAN_RATIO = Z_099 / Z_095  # Approximately 1.4144

# Bank of America empirical factor
BOA_FACTOR = 2.0

# Regex for matching VaR columns: goldmansachs_95, wells_fargo_99, Bank_of_America_99%
VAR_COL_PATTERN = re.compile(r"^(?P<bank>.+?)[_\s-]*(?P<level>95|99)\s*%?$", flags=re.IGNORECASE)


def clean_bank_name(bank_name: str) -> str:
    """Convert bank name to clean lowercase with underscores.
    
    Example: 'Bank of America' -> 'bank_of_america'
    """
    text = bank_name.strip().lower()
    # Replace any non-alphanumeric with underscore, then clean up multiple underscores
    cleaned = re.sub(r'[^a-z0-9]+', '_', text)
    return cleaned.strip('_')


def to_quarter_end(date_value: object) -> pd.Timestamp:
    """Parse date and normalize to quarter-end.
    
    Example: any date in Q3 2025 -> 2025-09-30
    """
    if date_value is None or pd.isna(date_value):
        return pd.NaT
    
    # Try parsing with dayfirst=True, then False
    dt = pd.to_datetime(date_value, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        dt = pd.to_datetime(date_value, errors="coerce", dayfirst=False)
    
    if pd.isna(dt):
        return pd.NaT
    
    # Convert to quarter-end
    return pd.Period(pd.Timestamp(dt), freq="Q").end_time.normalize()


def main() -> int:
    """Main workflow for VaR harmonization."""
    
    # 1) Check input file exists
    if not INPUT_FILE.exists():
        raise SystemExit(f"Missing input file: {INPUT_FILE}")

    # 2) Read Excel
    df_wide = pd.read_excel(INPUT_FILE)
    df_wide.columns = [str(c).strip() for c in df_wide.columns]

    if DATE_COL not in df_wide.columns:
        raise SystemExit(f"Missing date column {DATE_COL!r} in {INPUT_FILE}")

    # 3) Identify VaR columns using regex (ignore non-matching columns)
    var_cols: list[str] = []
    for col in df_wide.columns:
        if col == DATE_COL:
            continue
        if VAR_COL_PATTERN.match(str(col)):
            var_cols.append(col)

    if not var_cols:
        raise SystemExit("Found 0 VaR columns ending in _95/_99 or 95%/99%.")

    print("=" * 72)
    print("VaR harmonization using two conversion methods (target: 99%)")
    print("=" * 72)
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"\nFound {len(var_cols)} VaR columns:")
    for col in var_cols:
        match = VAR_COL_PATTERN.match(col)
        if match:
            print(f"  {col:30} -> {match.group('bank')}, level={match.group('level')}")

    # 4) Wide -> long format
    melted = df_wide.melt(
        id_vars=[DATE_COL], 
        value_vars=var_cols, 
        var_name="var_col", 
        value_name="var_value"
    )
    
    # Extract bank and level from column names
    extracted = melted["var_col"].astype(str).str.extract(VAR_COL_PATTERN)
    melted["bank_id"] = extracted["bank"].map(clean_bank_name)
    melted["level"] = extracted["level"].astype("Int64")
    melted["period_end_date"] = melted[DATE_COL].map(to_quarter_end)
    
    # Drop rows with missing data
    melted = melted.dropna(subset=["bank_id", "level", "period_end_date"]).copy()

    # 5) Pivot to panel format: one row per (bank_id, period_end_date)
    panel = (
        melted.pivot(
            index=["bank_id", "period_end_date"], 
            columns="level", 
            values="var_value"
        )
        .rename_axis(None, axis=1)
        .reset_index()
    )
    
    # Ensure var_95 and var_99 columns exist
    panel = panel.rename(columns={95: "var_95", 99: "var_99"})
    if "var_95" not in panel.columns:
        panel["var_95"] = np.nan
    if "var_99" not in panel.columns:
        panel["var_99"] = np.nan

    panel["var_95"] = pd.to_numeric(panel["var_95"], errors="coerce")
    panel["var_99"] = pd.to_numeric(panel["var_99"], errors="coerce")
    panel = panel.sort_values(["bank_id", "period_end_date"]).reset_index(drop=True)

    print(f"\nRows (panel): {len(panel):,}  Banks: {panel['bank_id'].nunique():,}  Quarters: {panel['period_end_date'].nunique():,}")
    
    # 6) Print conversion parameters
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

    # 7) Create harmonized columns
    out = panel.copy()
    has_95 = out["var_95"].notna()
    has_99 = out["var_99"].notna()

    # Method 1: Gaussian harmonization
    out["var_99_gaussian"] = np.nan
    out.loc[has_99, "var_99_gaussian"] = out.loc[has_99, "var_99"]
    out.loc[~has_99 & has_95, "var_99_gaussian"] = GAUSSIAN_RATIO * out.loc[~has_99 & has_95, "var_95"]

    out["var_99_source_gaussian"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out.loc[has_99, "var_99_source_gaussian"] = "reported_99"
    out.loc[~has_99 & has_95, "var_99_source_gaussian"] = "approx_from_95_gaussian"
    out.loc[~has_99 & ~has_95, "var_99_source_gaussian"] = "missing_both"

    # Method 2: BoA Factor harmonization
    out["var_99_boa_factor"] = np.nan
    out.loc[has_99, "var_99_boa_factor"] = out.loc[has_99, "var_99"]
    out.loc[~has_99 & has_95, "var_99_boa_factor"] = BOA_FACTOR * out.loc[~has_99 & has_95, "var_95"]

    out["var_99_source_boa"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out.loc[has_99, "var_99_source_boa"] = "reported_99"
    out.loc[~has_99 & has_95, "var_99_source_boa"] = "approx_from_95_boa"
    out.loc[~has_99 & ~has_95, "var_99_source_boa"] = "missing_both"

    # Backward compatibility: keep var_99_harmonized as Gaussian version
    out["var_99_harmonized"] = out["var_99_gaussian"].copy()
    out["var_99_source"] = out["var_99_source_gaussian"].copy()

    # Dummy: 1 if reported_99, 0 if approx_from_95 (NaN if missing both)
    out["var_level_dummy_99"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out.loc[has_99, "var_level_dummy_99"] = 1
    out.loc[~has_99 & has_95, "var_level_dummy_99"] = 0

    # 8) Validation on banks with both var_95 and var_99
    val_rows = out[has_95 & has_99].copy()
    
    if val_rows.empty:
        print("\n[validation] No rows with both var_95 and var_99 found.")
    else:
        # Gaussian method validation
        val_rows["var_99_hat_gaussian"] = GAUSSIAN_RATIO * val_rows["var_95"]
        abs_pct_error_gaussian = (
            (val_rows["var_99_hat_gaussian"] - val_rows["var_99"]) / val_rows["var_99"]
        ).abs()
        abs_pct_error_gaussian = abs_pct_error_gaussian.replace([np.inf, -np.inf], np.nan).dropna()
        
        # BoA Factor method validation
        val_rows["var_99_hat_boa"] = BOA_FACTOR * val_rows["var_95"]
        abs_pct_error_boa = (
            (val_rows["var_99_hat_boa"] - val_rows["var_99"]) / val_rows["var_99"]
        ).abs()
        abs_pct_error_boa = abs_pct_error_boa.replace([np.inf, -np.inf], np.nan).dropna()
        
        if not abs_pct_error_gaussian.empty:
            print("\n[validation - Gaussian method] on banks with both var_95 and var_99:")
            print("var_99_hat = var_95 * (z_0.99 / z_0.95)")
            print(
                f"n={len(abs_pct_error_gaussian):,}  "
                f"median_abs_pct_error={abs_pct_error_gaussian.median():.6g}  "
                f"p95_abs_pct_error={abs_pct_error_gaussian.quantile(0.95):.6g}"
            )
        
        if not abs_pct_error_boa.empty:
            print("\n[validation - BoA Factor method] on banks with both var_95 and var_99:")
            print("var_99_hat = var_95 * 2.0")
            print(
                f"n={len(abs_pct_error_boa):,}  "
                f"median_abs_pct_error={abs_pct_error_boa.median():.6g}  "
                f"p95_abs_pct_error={abs_pct_error_boa.quantile(0.95):.6g}"
            )
        
        banks_with_both = val_rows["bank_id"].unique()
        print(f"\nBanks with both values: {', '.join(sorted(banks_with_both))}")

    # 9) Count observations by conversion method
    n_reported = int(has_99.sum())
    n_approx = int((~has_99 & has_95).sum())
    n_missing = int((~has_99 & ~has_95).sum())
    
    print("")
    print("[counts by conversion method]")
    print(f"reported_99 (used as-is in both methods): {n_reported:,}")
    print(f"approx_from_95_gaussian (converted using {GAUSSIAN_RATIO:.4f}): {n_approx:,}")
    print(f"approx_from_95_boa (converted using {BOA_FACTOR:.1f}): {n_approx:,}")
    print(f"missing_both (no data available): {n_missing:,}")

    # 10) Save to CSV
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out[
        [
            "bank_id",
            "period_end_date",
            "var_95",
            "var_99",
            "var_99_gaussian",
            "var_99_boa_factor",
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
