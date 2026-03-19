import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BALANCE_DIR = SCRIPT_DIR / "Balansesheet_v2"
VAR_PATH = PROJECT_ROOT / "output/data/var_99.csv"
OUT_PATH = SCRIPT_DIR / "dataframe.csv"

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

SHEETS = {
    "bs": "Balance Sheet",
    "fs": "Financial Summary",
    "inc": "Income Statement",
}

VARIABLES = {
    "total_assets": {"sheet": "bs", "label": "Total Assets", "exact": True},
    "total_equity": {
        "sheet": "bs",
        "label": "Shareholders' Equity - Attributable to Parent Shareholders - Total",
        "exact": False,
    },
    "total_liabilities": {"sheet": "bs", "label": "Total Liabilities", "exact": True},
    "cet1_ratio": {"sheet": "bs", "label": "Capital Adequacy - Core Tier 1 (%)", "exact": True},
    "lcr_ratio": {"sheet": "bs", "label": "Liquidity Coverage Ratio - Basel 3 - %", "exact": True},
    "slr_ratio": {"sheet": "bs", "label": "Leverage Ratio - Basel 3 - %", "exact": True},
    "roa": {"sheet": "fs", "label": "Return on Average Total Assets", "exact": False},
    "dividend_payout_ratio": {"sheet": "fs", "label": "Dividend Payout Ratio - %", "exact": True},
    "repo":{"sheet": "bs", "label": "Securities Sold Under Repurchase Agreements & Federal Funds Purchased", "exact": False},
}


def read_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name, header=None)


def get_dates(df: pd.DataFrame) -> pd.Index:
    raw_dates = df.iloc[11, 1:]
    dates = pd.to_datetime(raw_dates, errors="coerce")
    return pd.Index(dates[dates.notna()])


def find_row(df: pd.DataFrame, label: str, exact: bool = False) -> int | None:
    for row_number, value in enumerate(df.iloc[:, 0]):
        if not isinstance(value, str):
            continue
        is_match = value == label if exact else label.lower() in value.lower()
        if is_match and df.iloc[row_number, 1:].notna().any():
            return row_number
    return None


def make_empty_series(dates: pd.Index) -> pd.Series:
    return pd.Series(index=dates, data=pd.NA, dtype="float64")


def extract_row_as_series(df: pd.DataFrame, label: str, dates: pd.Index, exact: bool = False) -> pd.Series:
    row_number = find_row(df, label, exact=exact)
    if row_number is None:
        return make_empty_series(dates)

    sheet_dates = get_dates(df)
    values = df.iloc[row_number, 1 : 1 + len(sheet_dates)].values
    series = pd.Series(values, index=sheet_dates, dtype="float64")
    return series.reindex(dates)


def extract_bank_data(bank: str, file_path: Path) -> pd.DataFrame:
    sheets = {key: read_sheet(file_path, sheet_name) for key, sheet_name in SHEETS.items()}
    dates = get_dates(sheets["bs"])

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

    var_df = pd.read_csv(VAR_PATH)
    var_df["date"] = pd.to_datetime(var_df["date"]).dt.date
    var_df = var_df.rename(columns={"var_99_gaussian": "total_var"})[["bank", "date", "total_var"]]

    panel = panel.merge(var_df, on=["bank", "date"], how="left")
    panel["year"] = pd.to_datetime(panel["date"]).dt.year
    panel["quarter"] = pd.to_datetime(panel["date"]).dt.quarter

    columns = ["date", "year", "quarter", "bank", *VARIABLES.keys(), "total_var"]
    return panel[columns].sort_values(["date", "bank"], ascending=[False, True]).reset_index(drop=True)


def main() -> None:
    long_panel = build_long_panel()
    long_panel.to_csv(OUT_PATH, index=False)


if __name__ == "__main__":
    main()
