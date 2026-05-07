"""
Generate a descriptive statistics table for the bank panel dataset.

This script creates the descriptive statistics table used in the data section of
the thesis. It uses the final panel dataset, constructs a few derived variables
such as leverage and growth rates, merges quarterly VIX values, and writes one
CSV table to output/tables/.
"""

from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/processed/panel.csv")
VIX_PATH = Path("data/raw/VIXCLS (1).csv")
OUTPUT_PATH = Path("output/tables/descriptive_statistics.csv")

START_DATE = "2010-01-01"
END_DATE = "2025-12-31"


def summarize(series: pd.Series) -> dict:
    """Return standard summary statistics for a numeric series."""
    series = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "N": int(series.count()),
        "Mean": round(series.mean(), 2),
        "Std. Dev.": round(series.std(), 2),
        "Median": round(series.median(), 2),
        "Min": round(series.min(), 2),
        "Max": round(series.max(), 2),
    }


def load_data() -> pd.DataFrame:
    """Load the panel dataset and construct derived variables for the table."""
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()
    df = df.sort_values(["bank", "date"]).reset_index(drop=True)

    # Convert variables to numeric form before calculating ratios and summary
    # statistics. Non-numeric entries become missing values.
    numeric_columns = [
        "total_assets",
        "total_liabilities",
        "total_equity",
        "repo",
        "total_var",
        "lcr_ratio",
        "cet1_ratio",
        "slr_ratio",
        "dividend_payout_ratio",
        "roa",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Leverage is defined as total assets divided by total equity.
    df["leverage"] = df["total_assets"] / df["total_equity"]

    # Growth variables are percentage changes from the previous quarter within
    # the same bank.
    growth_columns = [
        "leverage",
        "total_assets",
        "total_liabilities",
        "total_equity",
        "repo",
        "total_var",
    ]
    for column in growth_columns:
        df[f"{column}_growth"] = df.groupby("bank")[column].pct_change(fill_method=None) * 100

    # VIX is daily in the raw file, so it is matched to the bank panel by quarter.
    if VIX_PATH.exists():
        vix = pd.read_csv(VIX_PATH)
        vix["observation_date"] = pd.to_datetime(vix["observation_date"])
        vix["quarter"] = vix["observation_date"].dt.to_period("Q")

        df["quarter"] = df["date"].dt.to_period("Q")
        df = df.merge(vix[["quarter", "VIXCLS"]], on="quarter", how="left")
    else:
        df["VIXCLS"] = pd.NA

    # Balance sheet variables are stored in USD millions; convert them to billions.
    for column in ["total_assets", "total_liabilities", "total_equity", "repo"]:
        df[column] = df[column] / 1000

    return df


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the final descriptive statistics table."""
    # Each tuple gives: section heading, table label, and dataframe column.
    variables = [
        ("Balance Sheet Variables", "Leverage", "leverage"),
        ("Balance Sheet Variables", "Total Assets (USD bn)", "total_assets"),
        ("Balance Sheet Variables", "Total Liabilities (USD bn)", "total_liabilities"),
        ("Balance Sheet Variables", "Total Equity (USD bn)", "total_equity"),
        ("Balance Sheet Variables", "Repo Funding (USD bn)", "repo"),
        ("Growth Variables", "Leverage Growth (%)", "leverage_growth"),
        ("Growth Variables", "Total Assets Growth (%)", "total_assets_growth"),
        ("Growth Variables", "Total Liabilities Growth (%)", "total_liabilities_growth"),
        ("Growth Variables", "Total Equity Growth (%)", "total_equity_growth"),
        ("Growth Variables", "Repo Funding Growth (%)", "repo_growth"),
        ("Risk Variables", "Value-at-Risk", "total_var"),
        ("Risk Variables", "Value-at-Risk Growth (%)", "total_var_growth"),
        ("Regulatory Variables", "Liquidity Coverage Ratio", "lcr_ratio"),
        ("Regulatory Variables", "CET1 Ratio", "cet1_ratio"),
        ("Regulatory Variables", "Supplementary Leverage Ratio", "slr_ratio"),
        ("Payout Policy", "Dividend Payout Ratio", "dividend_payout_ratio"),
        ("Control Variables", "Return on Assets", "roa"),
        ("Volatility Index", "VIX", "VIXCLS"),
    ]

    rows = []
    for group, label, column in variables:
        stats = summarize(df[column])
        rows.append({"Group": group, "Variable": label, **stats})

    return pd.DataFrame(rows)


def main() -> None:
    """Generate and save the descriptive statistics table."""
    df = load_data()
    table = build_table(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_PATH, index=False)

    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
