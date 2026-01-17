# =============================================================================
# VaR harmonization across banks (95% vs 99%)
# =============================================================================
#
# This script produces one comparable VaR measure at the 99% level, so banks that
# report 95% VaR and banks that report 99% VaR can be compared directly.
#
# HOW IT WORKS:
# 1) (REQUIREMENT 1) Do not mix different VaR definitions:
#    - If the dataset contains "definition" columns (e.g., horizon/reporting/type)
#      and they vary across rows, the script stops and asks you to filter down to
#      a single definition using `--definition-filter col=value`.
#
# 2) (REQUIREMENT 2) Estimate a conversion factor k from a calibration bank:
#    - The calibration bank is a bank that reports both 95% and 99% VaR.
#    - For each quarter we compute k_t = VaR_99 / VaR_95.
#    - We choose k = median(k_t) (robust to outliers).
#
# 3) (REQUIREMENT 3) Create a harmonized 99% VaR:
#    - If VaR_99 is present: use it (reported value).
#    - Else if VaR_95 is present: use k * VaR_95 (converted).
#    - Else: NaN.
#
# 4) (REQUIREMENT 4) Validate on the calibration bank:
#    - Predict VaR_99_hat = k * VaR_95 and compare it to the reported VaR_99.
#
# 5) (REQUIREMENT 5) Minimum robustness options:
#    - Creates `var_level_dummy_99` that you can use in regressions instead of
#      harmonizing (1=reported 99, 0=only 95).
#    - Suggestion: run analyses separately for 95-banks and 99-banks.
#
# INPUT:
# - Default in this folder: `VaR_python.xlsx` (wide format) with columns like
#   `goldmansachs_95`, `citibank_99`, `Bank_of_America_99%`.
# - Alternatively: long/panel (CSV/Parquet) with at least:
#   `bank_id`, `period_end_date`, `var_95`, `var_99` (small naming variations are handled).
#
# OUTPUT:
# - Console report: k statistics + validation + number of converted observations.
# - CSV: `output/merged_with_var_harmonized.csv` (default) with a new column
#   `var_99_harmonized` (+ `var_99_source` + `var_level_dummy_99`).
#
# RUN (simplest, for your Excel):
#   python VaR.py
#
# If you want to be explicit:
#   python VaR.py --calib-bank-id bank_of_america
#
# =============================================================================

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


# =============================================================================
# 1) Small helper functions (names, dates, columns, numbers)
# =============================================================================


# Normalize text to "snake_case" (used to build stable bank IDs).
def snake_case(value: object) -> str:
    s = "" if value is None else str(value)
    s = s.strip().lower()
    out: list[str] = []
    prev_underscore = False
    for ch in s:
        is_alnum = ("a" <= ch <= "z") or ("0" <= ch <= "9")
        if is_alnum:
            out.append(ch)
            prev_underscore = False
        else:
            if not prev_underscore:
                out.append("_")
                prev_underscore = True
    return "".join(out).strip("_")


# Convert a date to quarter-end (Q-end), so all dates align for quarterly comparison.
def parse_period_end_date(value: object) -> pd.Timestamp:
    if value is None or pd.isna(value):
        return pd.NaT
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    return pd.Period(pd.Timestamp(ts), freq="Q").end_time.normalize()


# Robust mapping: return the first column that exists among several candidate names.
def _first_present(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    cols = {snake_case(c): c for c in columns}
    for cand in candidates:
        key = snake_case(cand)
        if key in cols:
            return cols[key]
    return None


# Convert a series to numeric (non-numeric values become NaN).
def _to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return pd.to_numeric(series, errors="coerce")


# =============================================================================
# 2) REQUIREMENT 1: Do not mix VaR definitions
# =============================================================================


# User-supplied filters to select one consistent definition before estimating k.
def apply_definition_filters(df: pd.DataFrame, definition_filters: dict[str, Any] | None) -> pd.DataFrame:
    if not definition_filters:
        return df

    out = df.copy()
    for col, wanted in definition_filters.items():
        if col not in out.columns:
            print(f"[warn] definition filter ignored (missing column): {col}={wanted}")
            continue
        if isinstance(wanted, (list, tuple, set)):
            out = out[out[col].isin(list(wanted))].copy()
        else:
            out = out[out[col] == wanted].copy()
    return out


# Safety check: if definition columns exist and have multiple values, stop.
def assert_single_definition(df: pd.DataFrame, definition_cols: Iterable[str]) -> None:
    mixed: dict[str, list[Any]] = {}
    for col in definition_cols:
        if col not in df.columns:
            continue
        values = pd.Series(df[col]).dropna().unique().tolist()
        if len(values) > 1:
            mixed[col] = values[:10]
    if mixed:
        details = "; ".join([f"{c} has {len(v)} values (e.g. {v})" for c, v in mixed.items()])
        raise ValueError(
            "Mixed VaR definitions detected; filter to a single consistent definition before estimating k: "
            + details
            + ". Use --definition-filter col=value (repeatable) or pre-filter your dataset."
        )


# =============================================================================
# 3) REQUIREMENTS 2–4: Estimate k, harmonize VaR, and validate
# =============================================================================


# Summary statistics for k_t = VaR_99 / VaR_95
@dataclass(frozen=True)
class KSummary:
    n_obs: int
    mean: float
    median: float
    std: float
    min: float
    p25: float
    p75: float
    max: float
    iqr: float

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "n_obs": self.n_obs,
                    "mean_k": self.mean,
                    "median_k": self.median,
                    "std_k": self.std,
                    "min_k": self.min,
                    "p25_k": self.p25,
                    "p75_k": self.p75,
                    "max_k": self.max,
                    "iqr_k": self.iqr,
                }
            ]
        )


# Estimate k (median of VaR_99/VaR_95) from the calibration bank.
def estimate_k(
    df: pd.DataFrame,
    calib_bank_id: str,
    var95_col: str,
    var99_col: str,
    *,
    bank_id_col: str = "bank_id",
    date_col: str = "period_end_date",
    require_positive: bool = True,
) -> tuple[float, pd.DataFrame]:
    # Normalize calib bank id if the bank_id column looks normalized already
    calib_bank_id_norm = str(calib_bank_id)
    if df[bank_id_col].astype(str).str.fullmatch(r"[a-z0-9_]+").all():
        calib_bank_id_norm = snake_case(calib_bank_id_norm)

    # Keep only the calibration bank + rows where both VaR values exist
    calib = df[df[bank_id_col].astype(str) == calib_bank_id_norm].copy()
    calib = calib[[bank_id_col, date_col, var95_col, var99_col]].copy()
    calib[var95_col] = _to_numeric(calib[var95_col])
    calib[var99_col] = _to_numeric(calib[var99_col])
    calib = calib[calib[var95_col].notna() & calib[var99_col].notna()].copy()

    if require_positive:
        calib = calib[(calib[var95_col] > 0) & (calib[var99_col] > 0)].copy()

    if calib.empty:
        raise ValueError(
            "No calibration observations with both VaR_95 and VaR_99 after filtering. "
            f"calib_bank_id={calib_bank_id!r}, var95_col={var95_col!r}, var99_col={var99_col!r}"
        )

    # k_t per quarter and robust choice of k as the median
    calib["k_t"] = calib[var99_col] / calib[var95_col]
    k_series = calib["k_t"].replace([np.inf, -np.inf], np.nan).dropna()
    if k_series.empty:
        raise ValueError("Computed k_t is empty after dropping non-finite values.")

    q25 = float(k_series.quantile(0.25))
    q75 = float(k_series.quantile(0.75))
    summary = KSummary(
        n_obs=int(k_series.shape[0]),
        mean=float(k_series.mean()),
        median=float(k_series.median()),
        std=float(k_series.std(ddof=1)) if k_series.shape[0] > 1 else 0.0,
        min=float(k_series.min()),
        p25=q25,
        p75=q75,
        max=float(k_series.max()),
        iqr=float(q75 - q25),
    )
    k = float(summary.median)
    return k, summary.to_frame()


# Add a new column with harmonized 99% VaR.
def add_harmonized_var(
    df: pd.DataFrame,
    k: float,
    var95_col: str,
    var99_col: str,
    *,
    out_col: str = "var_99_harmonized",
) -> pd.DataFrame:
    out = df.copy()
    out[var95_col] = _to_numeric(out[var95_col]) if var95_col in out.columns else np.nan
    out[var99_col] = _to_numeric(out[var99_col]) if var99_col in out.columns else np.nan

    has_99 = out[var99_col].notna()
    has_95 = out[var95_col].notna()

    # Main variable: 99% (use reported if available, otherwise convert from 95%)
    out[out_col] = np.nan
    out.loc[has_99, out_col] = out.loc[has_99, var99_col]
    out.loc[~has_99 & has_95, out_col] = k * out.loc[~has_99 & has_95, var95_col]

    # Source tag (useful to check what was converted)
    out["var_99_source"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out.loc[has_99, "var_99_source"] = "reported_99"
    out.loc[~has_99 & has_95, "var_99_source"] = "converted_from_95"

    # Robustness option (REQ 5A): dummy for reporting level
    out["var_level_dummy_99"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out.loc[has_99, "var_level_dummy_99"] = 1
    out.loc[~has_99 & has_95, "var_level_dummy_99"] = 0

    return out


# Validate the conversion on the calibration bank (error statistics).
def validate_on_calib(
    df: pd.DataFrame,
    calib_bank_id: str,
    k: float,
    var95_col: str,
    var99_col: str,
    *,
    bank_id_col: str = "bank_id",
    date_col: str = "period_end_date",
) -> pd.DataFrame:
    calib_bank_id_norm = str(calib_bank_id)
    if df[bank_id_col].astype(str).str.fullmatch(r"[a-z0-9_]+").all():
        calib_bank_id_norm = snake_case(calib_bank_id_norm)

    calib = df[df[bank_id_col].astype(str) == calib_bank_id_norm].copy()
    calib[var95_col] = _to_numeric(calib[var95_col])
    calib[var99_col] = _to_numeric(calib[var99_col])
    calib = calib[calib[var95_col].notna() & calib[var99_col].notna()].copy()
    calib = calib[(calib[var95_col] > 0) & (calib[var99_col] > 0)].copy()

    if calib.empty:
        return pd.DataFrame([{"n_obs": 0}])

    calib["var_99_hat"] = k * calib[var95_col]
    calib["pct_error"] = (calib["var_99_hat"] - calib[var99_col]) / calib[var99_col]
    calib["abs_pct_error"] = calib["pct_error"].abs()

    abs_err = calib["abs_pct_error"].replace([np.inf, -np.inf], np.nan).dropna()
    p95 = float(abs_err.quantile(0.95)) if not abs_err.empty else np.nan

    # Simple check for whether error drifts over time (not a full analysis)
    time_corr = np.nan
    if date_col in calib.columns:
        dates = pd.to_datetime(calib[date_col], errors="coerce")
        if dates.notna().any() and abs_err.shape[0] == calib.shape[0]:
            time_corr = float(abs_err.corr(dates.map(pd.Timestamp.toordinal), method="pearson"))

    return pd.DataFrame(
        [
            {
                "n_obs": int(abs_err.shape[0]),
                "median_abs_pct_error": float(abs_err.median()) if not abs_err.empty else np.nan,
                "mean_abs_pct_error": float(abs_err.mean()) if not abs_err.empty else np.nan,
                "p95_abs_pct_error": p95,
                "time_corr_abs_error": time_corr,
            }
        ]
    )


# =============================================================================
# 4) Input: support both "wide" Excel and long/panel
# =============================================================================


# Wide columns are detected by ending in 95 or 99 (optionally with %)
# Example: "Bank_of_America_99%" or "citibank_99" or "goldmansachs_95"
_WIDE_VAR_COL_RE = re.compile(r"^(?P<bank>.+?)[_\s-]*(?P<level>95|99)\s*%?$", flags=re.IGNORECASE)


# If the user does not pass a calibration bank: find a bank that has both 95 and 99 observed.
def infer_calibration_bank_id(
    df: pd.DataFrame,
    *,
    bank_id_col: str,
    var95_col: str,
    var99_col: str,
    min_both_obs: int = 3,
) -> str:
    tmp = df[[bank_id_col, var95_col, var99_col]].copy()
    tmp[var95_col] = _to_numeric(tmp[var95_col])
    tmp[var99_col] = _to_numeric(tmp[var99_col])
    tmp["_both"] = tmp[var95_col].notna() & tmp[var99_col].notna()

    counts = tmp.groupby(bank_id_col, dropna=False)["_both"].sum().sort_values(ascending=False)
    candidates = counts[counts > 0]
    if candidates.empty:
        raise ValueError(
            "Could not infer calibration bank: no bank has both 95% and 99% VaR observed. "
            "Pass --calib-bank-id explicitly."
        )
    if len(candidates) == 1:
        bank_id = str(candidates.index[0])
        if int(candidates.iloc[0]) < min_both_obs:
            raise ValueError(
                f"Only {int(candidates.iloc[0])} overlapping (95,99) observations for inferred calibration bank "
                f"{bank_id!r}; pass --calib-bank-id explicitly or provide more overlap."
            )
        return bank_id

    preview = ", ".join([f"{idx} (n_both={int(val)})" for idx, val in candidates.head(10).items()])
    raise ValueError(
        "Could not infer calibration bank unambiguously: multiple banks have both 95% and 99% VaR. "
        f"Candidates: {preview}. Pass --calib-bank-id explicitly."
    )


# Convert wide format (Excel) to panel format with bank_id + period_end_date + var_95 + var_99
def wide_var_to_long(
    df_wide: pd.DataFrame,
    *,
    date_col: str,
    bank_id_col: str = "bank_id",
    period_end_date_col: str = "period_end_date",
    var95_col: str = "var_95",
    var99_col: str = "var_99",
) -> pd.DataFrame:
    if date_col not in df_wide.columns:
        raise ValueError(f"Missing date column {date_col!r} in wide input.")

    records: list[dict[str, Any]] = []
    for col in df_wide.columns:
        if col == date_col:
            continue

        m = _WIDE_VAR_COL_RE.match(str(col).strip())
        if not m:
            continue

        bank_raw = m.group("bank")
        level = int(m.group("level"))
        bank_id = snake_case(bank_raw)

        series = _to_numeric(df_wide[col])
        for dt, value in zip(df_wide[date_col], series):
            records.append(
                {
                    bank_id_col: bank_id,
                    period_end_date_col: parse_period_end_date(dt),
                    "var_level": level,
                    "var_value": value,
                }
            )

    if not records:
        preview = ", ".join([str(c) for c in df_wide.columns[:10]])
        raise ValueError(
            "Could not detect any wide VaR columns with suffix 95/99 (e.g. 'bank_95', 'bank_99%'). "
            f"Columns preview: {preview}"
        )

    long = pd.DataFrame.from_records(records)
    long = long.dropna(subset=[period_end_date_col]).copy()

    wide = (
        long.pivot(index=[bank_id_col, period_end_date_col], columns="var_level", values="var_value")
        .rename_axis(None, axis=1)
        .reset_index()
    )

    if 95 in wide.columns:
        wide = wide.rename(columns={95: var95_col})
    else:
        wide[var95_col] = np.nan
    if 99 in wide.columns:
        wide = wide.rename(columns={99: var99_col})
    else:
        wide[var99_col] = np.nan

    return wide.sort_values([bank_id_col, period_end_date_col]).reset_index(drop=True)


# Read the input file (Excel/CSV/Parquet).
def load_input(path: Path, sheet_name: str | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        # pandas returnerer dict hvis sheet_name=None → default til første ark
        sheet = 0 if sheet_name is None else sheet_name
        return pd.read_excel(path, sheet_name=sheet)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path}")


# Ensure we end up with a panel with standard columns.
def ensure_long_panel(
    df: pd.DataFrame,
    *,
    bank_id_col: str | None = None,
    date_col: str | None = None,
    var95_col: str | None = None,
    var99_col: str | None = None,
) -> tuple[pd.DataFrame, str, str, str, str]:
    bank_id_col = bank_id_col or _first_present(df.columns, ["bank_id", "bank", "bankid", "id"])
    date_col = date_col or _first_present(df.columns, ["period_end_date", "period", "date", "year", "quarter_end_date"])
    var95_col = var95_col or _first_present(df.columns, ["var_95", "VaR_95", "var95", "VaR95", "var_95_pct", "var95%"])
    var99_col = var99_col or _first_present(df.columns, ["var_99", "VaR_99", "var99", "VaR99", "var_99_pct", "var99%"])

    # Long format (has bank_id + date + at least one of var_95/var_99)
    if bank_id_col and date_col and (var95_col or var99_col):
        out = df.copy()
        out = out.rename(columns={bank_id_col: "bank_id", date_col: "period_end_date"})
        if var95_col and var95_col != "var_95":
            out = out.rename(columns={var95_col: "var_95"})
        if var99_col and var99_col != "var_99":
            out = out.rename(columns={var99_col: "var_99"})
        out["period_end_date"] = out["period_end_date"].map(parse_period_end_date)
        if "bank_id" in out.columns:
            out["bank_id"] = out["bank_id"].astype(str).map(snake_case)
        return out, "bank_id", "period_end_date", "var_95", "var_99"

    # Wide format (like your Excel): we need at least a date column
    if date_col is None:
        raise ValueError(
            "Could not identify a date column. Provide a long panel with `period_end_date` or pass --date-col."
        )

    long = wide_var_to_long(df, date_col=date_col)
    return long, "bank_id", "period_end_date", "var_95", "var_99"


# Method text (2–4 sentences) you can paste into the assignment.
def method_text(k: float) -> str:
    return (
        "We harmonize VaR levels by estimating a conversion factor k from the bank reporting both 95% and 99% VaR, "
        f"using k=median(VaR99/VaR95)={k:.6g}, and converting 95% observations to 99% (VaR99_hat=k·VaR95)."
    )


# =============================================================================
# 5) CLI / run
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Harmonize VaR (95% → 99%) across banks.")
    parser.add_argument("--input", type=Path, default=Path("VaR_python.xlsx"))
    parser.add_argument("--sheet", type=str, default=None, help="Excel sheet name (optional).")
    parser.add_argument("--output", type=Path, default=Path("output/merged_with_var_harmonized.csv"))
    parser.add_argument(
        "--calib-bank-id",
        type=str,
        default=None,
        help="Bank ID used to estimate k (must have both). If omitted, the script tries to infer it.",
    )
    parser.add_argument("--bank-id-col", type=str, default=None)
    parser.add_argument("--date-col", type=str, default=None)
    parser.add_argument("--var95-col", type=str, default=None)
    parser.add_argument("--var99-col", type=str, default=None)
    parser.add_argument(
        "--definition-filter",
        action="append",
        default=[],
        help="Filter for consistent VaR definition: col=value (repeatable). Example: --definition-filter var_horizon=10d",
    )
    args = parser.parse_args(argv)

    # 1) Read input
    df_raw = load_input(args.input, args.sheet)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    # 2) Convert to panel format (bank_id + period_end_date + var_95 + var_99)
    df, bank_id_col, date_col, var95_col, var99_col = ensure_long_panel(
        df_raw,
        bank_id_col=args.bank_id_col,
        date_col=args.date_col,
        var95_col=args.var95_col,
        var99_col=args.var99_col,
    )

    # 3) (Optional) user filter to one VaR definition (REQ 1)
    definition_filters: dict[str, Any] = {}
    for item in args.definition_filter:
        if "=" not in item:
            raise ValueError(f"Bad --definition-filter {item!r}; expected col=value")
        col, value = item.split("=", 1)
        definition_filters[col] = value

    df = apply_definition_filters(df, definition_filters)

    # 4) REQ 1: stop if definition columns vary
    assert_single_definition(
        df,
        [
            "var_horizon",
            "var_reporting",
            "var_type",
            "var_category",
            "var_measure",
        ],
    )

    # 5) Find calibration bank (if not provided)
    calib_bank_id = args.calib_bank_id
    inferred = False
    if calib_bank_id is None:
        calib_bank_id = infer_calibration_bank_id(df, bank_id_col=bank_id_col, var95_col=var95_col, var99_col=var99_col)
        inferred = True

    # 6) Console header
    print("=" * 72)
    print("VaR harmonization (target level: 99%)")
    print("=" * 72)
    print(f"Input: {args.input}")
    if args.sheet:
        print(f"Sheet: {args.sheet}")
    print(f"Rows: {len(df):,}  Banks: {df[bank_id_col].nunique():,}  Quarters: {df[date_col].nunique():,}")
    print(f"Calibration bank_id: {calib_bank_id!r}" + (" (inferred)" if inferred else ""))
    print(f"Definition filters: {definition_filters if definition_filters else '(none provided)'}")

    # 7) REQ 2: estimate k
    k, k_summary = estimate_k(
        df,
        calib_bank_id,
        var95_col,
        var99_col,
        bank_id_col=bank_id_col,
        date_col=date_col,
    )

    # 8) REQ 3: build harmonized VaR
    df_out = add_harmonized_var(df, k, var95_col, var99_col, out_col="var_99_harmonized")

    # 9) REQ 4: validate on the calibration bank
    validation = validate_on_calib(
        df_out, calib_bank_id, k, var95_col, var99_col, bank_id_col=bank_id_col, date_col=date_col
    )

    # 10) Count how much is reported vs converted
    n_converted = int(((df_out[var99_col].isna()) & (df_out[var95_col].notna())).sum())
    n_reported_99 = int(df_out[var99_col].notna().sum())
    n_missing_both = int(((df_out[var99_col].isna()) & (df_out[var95_col].isna())).sum())

    print("\n[k estimation summary]  k_t = VaR_99 / VaR_95  (calibration bank only)")
    print(k_summary.to_string(index=False))
    print(f"Chosen k (median): {k:.6g}")

    print("\n[validation on calibration bank]  VaR_99_hat = k * VaR_95")
    print(validation.to_string(index=False))

    print("\n[harmonized output]")
    print(f"Reported 99% used: {n_reported_99:,}")
    print(f"Converted from 95%: {n_converted:,}")
    print(f"Missing both: {n_missing_both:,}")

    # 11) Save CSV with the harmonized variable
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")

    # 12) Robustness alternatives (REQ 5)
    print("\n[robustness alternatives]")
    print("A) Use `var_level_dummy_99` in regressions instead of harmonizing (1=reported 99, 0=only 95).")
    print("B) Run analyses separately for banks that report only 95% vs only 99%.")

    # 13) Method text (can be pasted into the assignment)
    print("\n[method text]")
    print(method_text(k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
