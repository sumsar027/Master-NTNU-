"""
Build a bank-level quarterly panel dataset from raw Excel source files.

The source Excel files are not normal flat datasets. Each bank workbook contains
many accounting rows and quarterly dates across columns. This script extracts the
rows needed for the thesis, stacks all banks into one long panel, and merges the
harmonized 99% VaR series.
"""

import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
BALANCE_DIR = PROJECT_ROOT / "data/raw/balance_sheets_final"
VAR_PATH = PROJECT_ROOT / "output/data/var_99.csv"
OUT_PATH = PROJECT_ROOT / "data/processed/panel.csv"

BANK_FILES = {
    "bankofamerica": "bankofamerica.xlsx",
    "bny": "bny.xlsx",
    "citigroup": "citigroup.xlsx",
    "goldmansachs": "goldmansachs.xlsx",
    "jpmorgan": "jpmorgan.xlsx",
    "morganstanley": "morganstanley.xlsx",
    "statestreet": "statestreet.xlsx",
    "wellsfargo": "wellsfargo.xlsx",
}

# Only these workbook sheets are needed for the final thesis variables.
SHEETS = {
    "bs": "Balance Sheet",
    "fs": "Financial Summary",
}

# Mapping from final panel variable names to the row labels in the Excel files.
VARIABLES = {
    "total_assets": {"sheet": "bs", "label": "Total Assets", "exact": True},
    "total_equity": {
        "sheet": "bs",
        "label": "Total Shareholders' Equity - including Minority Interest & Hybrid Debt",
        "exact": False,
    },
    "total_liabilities": {"sheet": "bs", "label": "Total Liabilities", "exact": True},
    "cet1_ratio": {"sheet": "bs", "label": "Capital Adequacy - Core Tier 1 (%)", "exact": True},
    "lcr_ratio": {"sheet": "bs", "label": "Liquidity Coverage Ratio - Basel 3 - %", "exact": True},
    "slr_ratio": {"sheet": "bs", "label": "Leverage Ratio - Basel 3 - %", "exact": True},
    "roa": {"sheet": "fs", "label": "Return on Average Total Assets", "exact": False},
    "roe": {"sheet": "fs", "label": "Return on Average Common Equity - % (Income available to Common excluding Extraordinary Items), TTM", "exact": True},
    "dividend_payout_ratio": {"sheet": "fs", "label": "Dividend Payout Ratio - %", "exact": True},
    "repo": {"sheet": "bs", "label": "Securities Sold Under Repurchase Agreements & Federal Funds Purchased", "exact": False},
}


def read_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """Read one sheet from a source workbook without assuming embedded headers."""
    return pd.read_excel(path, sheet_name=sheet_name, header=None)


def get_dates(df: pd.DataFrame) -> pd.Index:
    """Extract the reporting dates from the standard header row."""
    raw_dates = df.iloc[11, 1:]
    dates = pd.to_datetime(raw_dates, errors="coerce")
    return pd.Index(dates[dates.notna()])


def find_row(df: pd.DataFrame, label: str, exact: bool = False) -> int | None:
    """Locate the row that contains the requested label and non-empty values."""
    for row_number, value in enumerate(df.iloc[:, 0]):
        if not isinstance(value, str):
            continue
        is_match = value == label if exact else label.lower() in value.lower()
        if is_match and df.iloc[row_number, 1:].notna().any():
            return row_number
    return None


def make_empty_series(dates: pd.Index) -> pd.Series:
    """Return an all-missing series aligned to the requested dates."""
    return pd.Series(index=dates, data=pd.NA, dtype="float64")


def extract_row_as_series(df: pd.DataFrame, label: str, dates: pd.Index, exact: bool = False) -> pd.Series:
    """Extract one labelled row from a workbook sheet as a date-indexed series."""
    row_number = find_row(df, label, exact=exact)
    if row_number is None:
        return make_empty_series(dates)

    sheet_dates = get_dates(df)
    values = df.iloc[row_number, 1 : 1 + len(sheet_dates)].values
    series = pd.Series(values, index=sheet_dates, dtype="float64")
    return series.reindex(dates)


def extract_bank_data(bank: str, file_path: Path) -> pd.DataFrame:
    """Extract the configured variables for a single bank workbook."""
    sheets = {key: read_sheet(file_path, sheet_name) for key, sheet_name in SHEETS.items()}
    dates = get_dates(sheets["bs"])

    # Start with dates as the index, then add one extracted variable at a time.
    bank_df = pd.DataFrame(index=dates)

    for variable_name, config in VARIABLES.items():
        bank_df[variable_name] = extract_row_as_series(
            sheets[config["sheet"]],
            config["label"],
            dates,
            exact=config.get("exact", False),
        )

    bank_df["bank"] = bank
    bank_df["date"] = pd.to_datetime(dates).date

    return bank_df.reset_index(drop=True)


def build_long_panel() -> pd.DataFrame:
    """Combine bank-level extracts into one long panel and merge VaR data."""
    bank_frames = []

    for bank, filename in BANK_FILES.items():
        file_path = BALANCE_DIR / filename
        if not file_path.exists():
            print(f"[ADVARSEL] Fant ikke {file_path}, hopper over.")
            continue

        print(f"Laster {filename} ...")
        bank_frames.append(extract_bank_data(bank, file_path))

    if not bank_frames:
        raise ValueError(f"Fant ingen Excel-filer i {BALANCE_DIR}")

    panel = pd.concat(bank_frames, ignore_index=True)

    # Merge harmonized VaR into the accounting panel by bank and quarter.
    var_df = pd.read_csv(VAR_PATH)
    var_df["date"] = pd.to_datetime(var_df["date"]).dt.date
    var_df = var_df.rename(columns={"var_99_gaussian": "total_var"})[["bank", "date", "total_var"]]

    panel = panel.merge(var_df, on=["bank", "date"], how="left")

    # Drop exact duplicate bank-date rows after the merge.
    panel = (
        panel
        .drop_duplicates(subset=["bank", "date"], keep="first")
        .copy()
    )

    panel["year"] = pd.to_datetime(panel["date"]).dt.year
    panel["quarter"] = pd.to_datetime(panel["date"]).dt.quarter

    columns = ["date", "year", "quarter", "bank", *VARIABLES.keys(), "total_var"]

    return (
        panel[columns]
        .sort_values(["date", "bank"], ascending=[False, True])
        .reset_index(drop=True)
    )


def main() -> None:
    """Build and save the final panel dataset."""
    long_panel = build_long_panel()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    long_panel.to_csv(OUT_PATH, index=False)


if __name__ == "__main__":
    main()
