"""Build a quarterly bank panel from Refinitiv/Workspace Excel exports.

Input (relative to this script):
- Balansesheet/BalanceSheets/*.xlsx        (sheet: "Balance Sheet")
- Income_statement/IncomeStatement/*.xlsx  (sheet: "Income Statement")

Output:
- output/missing_income_banks.csv
- output/date_coverage_report.csv
- output/merged_quarterly_unbalanced.csv
- output/merged_quarterly_balanced.csv

Deterministic: sorted file order, sorted output rows.

Plain-English overview (for non-Python readers):
- This script reads many Excel files (one per bank) for two statement types: Balance Sheet and Income Statement.
- Inside each Excel file it finds the table header row by searching for the text "Field Name" in the first column.
- It treats each table row as a financial line item (e.g., "Total Assets") and each column header as a reporting period.
- It cleans the line item names into consistent machine-friendly labels and also fixes duplicates by adding suffixes (_2, _3, ...).
- It converts each reporting period into a quarter-end date (e.g., Q1 2024 → 2024-03-31) so different date formats match.
- It reshapes the data so you end up with one row per bank per quarter, with many columns for all the line items.
- It then merges Balance Sheet and Income Statement rows on the shared key: (bank, quarter-end date).
- It writes four CSV files:
  1) missing_income_banks.csv: banks that have balance sheets but no income statements.
  2) date_coverage_report.csv: min/max quarter and number of quarters available per bank for each statement type.
  3) merged_quarterly_unbalanced.csv: the merged panel using all bank-quarter rows that exist in both statements.
  4) merged_quarterly_balanced.csv: a stricter panel keeping only the quarters that are present for every bank (intersection).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

BALANCE_DIR_CANDIDATES = [
    ROOT / "Balansesheet" / "BalanceSheets",
    ROOT / "Balansesheet",
]
INCOME_DIR_CANDIDATES = [
    ROOT / "Income_statement" / "IncomeStatement",
    ROOT / "Income_statement",
]
OUTPUT_DIR = ROOT / "output"


def snake_case(value: object) -> str:
    """Convert a label into a simple, consistent identifier.

    Example: "Total Assets (€)" -> "total_assets".
    This helps columns/fields match across files even if the original text differs slightly.
    """
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


def extract_bank_name(stem: str) -> str:
    """Derive a bank identifier from an Excel filename (without the .xlsx).

    It removes common suffix words like "income statement" or "balance sheet"
    and returns a compact bank key used for merging.
    """
    name = snake_case(stem)
    for suffix in ("income", "statement", "balance", "sheet"):
        if name.endswith("_" + suffix):
            name = name[: -(len(suffix) + 1)]
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.replace("_", "")


def parse_period_end_date(value: object) -> pd.Timestamp:
    """Parse many possible "period" formats and return a quarter-end date.

    Input can be: Excel date serials, a year (e.g., 2023), a date string, or "2024Q1".
    Output is normalized to the quarter-end (e.g., 2024-03-31) to make merges consistent.
    """
    if value is None or pd.isna(value):
        return pd.NaT

    def to_quarter_end(dt: object) -> pd.Timestamp:
        ts = pd.to_datetime(dt, errors="coerce")
        if pd.isna(ts):
            return pd.NaT
        return pd.Period(pd.Timestamp(ts), freq="Q").end_time.normalize()

    if isinstance(value, pd.Timestamp):
        return to_quarter_end(value)
    if isinstance(value, np.datetime64):
        return to_quarter_end(value)

    if isinstance(value, (int, np.integer)):
        n = int(value)
        if 1900 <= n <= 2100:
            return to_quarter_end(pd.Timestamp(year=n, month=12, day=31))
        if 20_000 <= n <= 60_000:
            return to_quarter_end(pd.to_datetime(n, unit="D", origin="1899-12-30", errors="coerce"))
        return pd.NaT

    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return parse_period_end_date(int(value))
        if 20_000 <= float(value) <= 60_000:
            return to_quarter_end(pd.to_datetime(value, unit="D", origin="1899-12-30", errors="coerce"))
        return pd.NaT

    s = str(value).strip()
    if not s:
        return pd.NaT

    # Common "2024Q1" / "Q1 2024" variants
    s_low = s.lower().replace("-", " ").replace("/", " ").replace(".", " ").replace("_", " ")
    tokens = [t for t in s_low.split() if t]
    year = next((int(t) for t in tokens if t.isdigit() and len(t) == 4), None)
    quarter = None
    for t in tokens:
        if t in ("q1", "q2", "q3", "q4"):
            quarter = int(t[1])
            break
        if t.startswith("q") and len(t) == 2 and t[1].isdigit():
            quarter = int(t[1])
            break
        if t.endswith("q") and len(t) == 2 and t[0].isdigit():
            quarter = int(t[0])
            break
    if year is not None and quarter in (1, 2, 3, 4):
        return to_quarter_end(pd.Period(f"{year}Q{quarter}").end_time)

    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        dt = pd.to_datetime(s, errors="coerce", dayfirst=False)
    if pd.isna(dt):
        return pd.NaT
    return to_quarter_end(dt)


def find_table_header_row(path: Path, sheet_name: str) -> int:
    """Find the row index where the real table header starts in the Excel sheet.

    Refinitiv exports often have some metadata rows first. We detect the header row by
    scanning the first column for the exact text "Field Name".
    """
    # Fast path: only read first col and first N rows
    col0 = pd.read_excel(path, sheet_name=sheet_name, header=None, usecols=[0], nrows=300)
    series = col0.iloc[:, 0].astype(str).str.strip().str.casefold()
    matches = np.flatnonzero(series.to_numpy() == "field name")
    if len(matches) > 0:
        return int(matches[0])

    # Fallback: scan full first col
    col0_full = pd.read_excel(path, sheet_name=sheet_name, header=None, usecols=[0])
    series_full = col0_full.iloc[:, 0].astype(str).str.strip().str.casefold()
    matches_full = np.flatnonzero(series_full.to_numpy() == "field name")
    if len(matches_full) == 0:
        raise Exception(
            f"{path}: could not find header row where first column equals 'Field Name' in sheet '{sheet_name}'"
        )
    return int(matches_full[0])


def list_xlsx_files(directory: Path) -> list[Path]:
    """List all .xlsx files in a directory, in a deterministic (sorted) order.

    Temporary Excel lock files (~$...) are ignored so they don't break processing.
    """
    if not directory.exists():
        raise Exception(f"Missing input directory: {directory}")
    paths = sorted([p for p in directory.glob("*.xlsx")])
    ignored = [p for p in paths if p.name.startswith("~$")]
    if ignored:
        print(f"  - Ignoring {len(ignored)} temporary Excel files (~$...)")
    return [p for p in paths if not p.name.startswith("~$")]


def resolve_input_dir(candidates: list[Path], label: str) -> Path:
    """Pick the first candidate directory that exists and contains real .xlsx files.

    This allows a couple of common folder layouts (with or without the inner subfolder).
    """
    existing = [p for p in candidates if p.exists() and p.is_dir()]
    for p in existing:
        if any((not f.name.startswith("~$")) for f in p.glob("*.xlsx")):
            return p
    if existing:
        raise Exception(
            f"No .xlsx files found for {label} in any existing candidate dirs: {', '.join(str(p) for p in existing)}"
        )
    raise Exception(f"Missing input directory for {label}: tried {', '.join(str(c) for c in candidates)}")


def read_statement_long(*, path: Path, bank: str, statement_type: str, sheet_name: str) -> pd.DataFrame:
    """Read one Excel statement file and return a "long" table.

    Long format means: one row per (bank, statement type, line item, quarter-end date).
    This is a convenient intermediate format before pivoting to a wide panel.
    """
    header_row = find_table_header_row(path, sheet_name)
    df = pd.read_excel(path, sheet_name=sheet_name, header=header_row)
    n_cols_before = df.shape[1]
    df = df.dropna(axis=1, how="all")
    if df.shape[1] != n_cols_before:
        print(f"    - Dropped {n_cols_before - df.shape[1]} all-empty columns")

    first_col = df.columns[0]
    df = df.rename(columns={first_col: "Field Name"})
    n_rows_before = len(df)
    df = df[df["Field Name"].notna()].copy()
    if len(df) != n_rows_before:
        print(f"    - Dropped {n_rows_before - len(df)} rows with missing Field Name")

    df["field_name_base"] = df["Field Name"].map(snake_case)
    empty_fields = df["field_name_base"] == ""
    if empty_fields.any():
        n = int(empty_fields.sum())
        print(f"    - Dropping {n} rows with empty Field Name")
        df = df.loc[~empty_fields].copy()

    counts = df["field_name_base"].value_counts()
    n_dup_fields = int((counts > 1).sum())
    if n_dup_fields:
        n_rows_dup = int(df["field_name_base"].map(counts).gt(1).sum())
        print(
            f"    - Disambiguating {n_dup_fields} duplicated field names by appending _2, _3, ... "
            f"({n_rows_dup} rows affected)"
        )
    df["_field_occ"] = df.groupby("field_name_base", sort=False).cumcount()
    df["field_name"] = np.where(
        df["_field_occ"].eq(0),
        df["field_name_base"],
        df["field_name_base"] + "_" + (df["_field_occ"] + 1).astype(str),
    )

    helper_cols = {"field_name_base", "_field_occ", "field_name"}
    raw_cols = [c for c in df.columns if c != "Field Name" and c not in helper_cols]
    parsed = {c: parse_period_end_date(c) for c in raw_cols}
    date_cols = [c for c in raw_cols if pd.notna(parsed[c])]
    non_date_cols = [c for c in raw_cols if c not in date_cols]

    if not date_cols:
        raise Exception(f"{path}: found 0 date columns after header row {header_row} in sheet '{sheet_name}'")

    if non_date_cols:
        preview = ", ".join([str(c) for c in non_date_cols[:10]])
        tail = "" if len(non_date_cols) <= 10 else f" (+{len(non_date_cols) - 10} more)"
        print(f"    - Ignoring {len(non_date_cols)} non-date columns: {preview}{tail}")

    parsed_dates = [parsed[c] for c in date_cols]
    if len(parsed_dates) != len(set(parsed_dates)):
        collisions: dict[pd.Timestamp, list[object]] = {}
        for c in date_cols:
            collisions.setdefault(parsed[c], []).append(c)
        bad = {d: cols for d, cols in collisions.items() if len(cols) > 1}
        raise Exception(f"{path}: multiple columns map to same period_end_date: {bad}")

    melted = df.melt(
        id_vars=["field_name"],
        value_vars=date_cols,
        var_name="period_end_date_raw",
        value_name="value",
    )
    melted["period_end_date"] = melted["period_end_date_raw"].map(parsed)
    melted["bank"] = bank
    melted["statement_type"] = statement_type

    long = melted[["bank", "statement_type", "field_name", "period_end_date", "value"]].copy()

    if long["period_end_date"].isna().any():
        raise Exception(f"{path}: produced NaT period_end_date rows after parsing columns")

    dup = long.duplicated(subset=["bank", "statement_type", "field_name", "period_end_date"], keep=False)
    if dup.any():
        bad_rows = long.loc[dup].sort_values(["bank", "field_name", "period_end_date"]).head(50)
        raise Exception(
            f"{path}: duplicate rows on (bank, statement_type, field_name, period_end_date)\n{bad_rows.to_string(index=False)}"
        )

    return long


def read_all_statements_long(*, directory: Path, statement_type: str, sheet_name: str) -> pd.DataFrame:
    """Read all Excel files in a folder and stack them into one long table.

    Each file is treated as one bank. Results are sorted to keep output deterministic.
    """
    paths = list_xlsx_files(directory)
    print(f"\nFound {len(paths)} {statement_type} files in {directory}:")

    frames: list[pd.DataFrame] = []
    for path in paths:
        bank = extract_bank_name(path.stem)
        print(f"  - {path.name} → bank: '{bank}'")
        frames.append(
            read_statement_long(path=path, bank=bank, statement_type=statement_type, sheet_name=sheet_name)
        )

    if not frames:
        raise Exception(f"No {statement_type} files found in {directory}")

    long = pd.concat(frames, ignore_index=True)
    long = long.sort_values(["bank", "period_end_date", "field_name"]).reset_index(drop=True)
    return long


def statement_long_to_wide(long: pd.DataFrame, statement_type: str) -> pd.DataFrame:
    """Convert long-format statement data into a wide bank×quarter panel.

    Output: one row per (bank, quarter-end date) and one column per line item.
    """
    df = long[long["statement_type"] == statement_type].copy()

    dup = df.duplicated(subset=["bank", "period_end_date", "field_name"], keep=False)
    if dup.any():
        bad = df.loc[dup].sort_values(["bank", "period_end_date", "field_name"]).head(50)
        raise Exception(
            f"Duplicate rows in {statement_type} on (bank, period_end_date, field_name)\n{bad.to_string(index=False)}"
        )

    wide = (
        df.pivot(index=["bank", "period_end_date"], columns="field_name", values="value")
        .reset_index()
        .rename_axis(None, axis=1)
    )

    dup_key = wide.duplicated(subset=["bank", "period_end_date"], keep=False)
    if dup_key.any():
        bad = wide.loc[dup_key].sort_values(["bank", "period_end_date"]).head(50)
        raise Exception(
            f"Duplicate rows in {statement_type} on (bank, period_end_date)\n{bad.to_string(index=False)}"
        )

    return wide.sort_values(["bank", "period_end_date"]).reset_index(drop=True)


def date_coverage_report(balance_wide: pd.DataFrame, income_wide: pd.DataFrame) -> pd.DataFrame:
    """Summarize which quarters each bank has for each statement type.

    Produces min/max quarter-end date and number of unique quarters per bank, for balance and income.
    """
    def cov(df: pd.DataFrame, label: str) -> pd.DataFrame:
        return df.groupby("bank", as_index=False)["period_end_date"].agg(
            **{
                f"{label}_min_period_end_date": "min",
                f"{label}_max_period_end_date": "max",
                f"{label}_n_quarters": "nunique",
            }
        )

    return cov(balance_wide, "balance").merge(cov(income_wide, "income"), on="bank", how="outer").sort_values(
        "bank"
    )


def build_balanced_panel(merged_unbalanced: pd.DataFrame) -> pd.DataFrame:
    """Create a strictly balanced panel across banks.

    It keeps only the quarter-end dates that are present for every bank (date intersection).
    This is useful for models that require the same time periods for all banks.
    """
    banks = sorted(merged_unbalanced["bank"].unique())
    if not banks:
        return merged_unbalanced.copy()

    date_sets = [set(merged_unbalanced.loc[merged_unbalanced["bank"] == b, "period_end_date"].unique()) for b in banks]
    common_dates = set.intersection(*date_sets) if date_sets else set()

    balanced = merged_unbalanced[merged_unbalanced["period_end_date"].isin(sorted(common_dates))].copy()
    balanced = balanced.sort_values(["bank", "period_end_date"]).reset_index(drop=True)

    expected = len(common_dates)
    counts = balanced.groupby("bank")["period_end_date"].nunique().reindex(banks).fillna(0).astype(int)
    if (counts != expected).any():
        raise Exception(
            "Balanced panel construction failed; per-bank date counts differ from expected intersection:\n"
            + counts.to_string()
        )

    print(f"\nBalanced panel: {expected} common quarters across {len(banks)} banks")
    return balanced


def main() -> None:
    """Run the end-to-end pipeline: read Excel, clean/reshape, merge, and write CSV outputs."""
    print("=" * 60)
    print("Quarterly Panel Builder (merge key: bank + period_end_date)")
    print("=" * 60)

    balance_dir = resolve_input_dir(BALANCE_DIR_CANDIDATES, "balance")
    income_dir = resolve_input_dir(INCOME_DIR_CANDIDATES, "income")
    print(f"Using balance dir: {balance_dir}")
    print(f"Using income dir: {income_dir}")

    balance_long = read_all_statements_long(directory=balance_dir, statement_type="balance", sheet_name="Balance Sheet")
    income_long = read_all_statements_long(directory=income_dir, statement_type="income", sheet_name="Income Statement")

    balance_wide = statement_long_to_wide(balance_long, "balance")
    income_wide = statement_long_to_wide(income_long, "income")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    balance_banks = set(balance_wide["bank"].unique())
    income_banks = set(income_wide["bank"].unique())
    missing_income_banks = sorted(balance_banks - income_banks)
    pd.DataFrame({"bank": missing_income_banks}).to_csv(OUTPUT_DIR / "missing_income_banks.csv", index=False)
    if missing_income_banks:
        print("\nBanks with balance but no income ({}): {}".format(len(missing_income_banks), missing_income_banks))
    else:
        print("\nAll balance-sheet banks have income statements.")

    date_coverage_report(balance_wide, income_wide).to_csv(OUTPUT_DIR / "date_coverage_report.csv", index=False)

    print("\nMerging on (bank, period_end_date) ...")
    merged_unbalanced = balance_wide.merge(
        income_wide,
        on=["bank", "period_end_date"],
        how="inner",
        suffixes=("_bs", "_is"),
        validate="one_to_one",
    )
    merged_unbalanced = merged_unbalanced.sort_values(["bank", "period_end_date"]).reset_index(drop=True)

    print(f"  Total merged rows: {len(merged_unbalanced)}")
    print(f"  Unique period_end_date: {merged_unbalanced['period_end_date'].nunique()}")
    for bank, n in merged_unbalanced.groupby("bank").size().sort_values().items():
        print(f"  - {bank}: {n} rows")

    merged_unbalanced.to_csv(OUTPUT_DIR / "merged_quarterly_unbalanced.csv", index=False)

    merged_balanced = build_balanced_panel(merged_unbalanced)
    merged_balanced.to_csv(OUTPUT_DIR / "merged_quarterly_balanced.csv", index=False)

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
