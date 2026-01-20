"""
Asset Growth and Leverage Growth (book-side only).

This script replicates the "book-side" of an Adrian & Shin-style table:
  Y = Asset growth (quarterly log-diff of total assets, in %)
  X = Book leverage growth (quarterly log-diff of Assets/Equity, in %)

Terminal output is intentionally compact:
- Loaded <file> shape=(...)
- ENTITY=<...> TIME=<...>
- Using: assets=<...>, equity=<...>, debt=<...>
- Market data existence (no market regressions are run here)
- Results for (1)-(3) with coef, SE, t-stat, and significance stars

Outputs:
- output/analysis_dataset.csv
- output/table_asset_leverage_growth.tex
- output/table_asset_leverage_growth.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS


# -----------------------------
# Input / output paths
# -----------------------------
ACCOUNTING_PATH = Path("output/merged_quarterly_balanced.csv")
# Market file is *not used* here, but we report whether it exists (for later market-side extension).
MARKET_PATH = Path("output/market_data_quarterly.csv")

OUTPUT_DATASET_PATH = Path("output/analysis_dataset.csv")
OUTPUT_TEX_PATH = Path("output/table_asset_leverage_growth.tex")
OUTPUT_MD_PATH = Path("output/table_asset_leverage_growth.md")


def die(msg: str) -> None:
    """Print an error message and exit immediately (hard stop)."""
    print(msg)
    sys.exit(1)


def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Return the first column name that exists in df from a list of candidates.
    Used to handle different naming conventions across datasets.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric; non-parsable values become NaN."""
    return pd.to_numeric(series, errors="coerce")


def coerce_time(series: pd.Series, *, label: str) -> pd.Series:
    """
    Ensure the TIME column is datetime.
    If parsing fails for more than ~10% of non-missing values, stop (guardrail).
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    parsed = pd.to_datetime(series, errors="coerce")
    nonmissing = int(series.notna().sum())
    parsed_ok = int(parsed.notna().sum())
    if nonmissing == 0:
        return parsed
    if (parsed_ok / nonmissing) >= 0.90:
        return parsed
    die(f"{label}: TIME column could not be parsed to datetime reliably (parsed {parsed_ok}/{nonmissing}).")


def detect_entity_time(df: pd.DataFrame) -> tuple[str, str]:
    """
    Auto-detect panel identifiers:
    - ENTITY: bank identifier (e.g., 'bank', 'ticker', ...)
    - TIME:   quarter end date (e.g., 'period_end_date', ...)
    """
    entity_candidates = ["bank", "bank_id", "ticker", "permno", "entity", "id"]
    time_candidates = ["period_end_date", "date", "quarter", "time", "period"]
    entity = pick_first_existing(df, entity_candidates)
    time = pick_first_existing(df, time_candidates)
    if entity is None:
        die(f"Could not detect ENTITY column. Tried: {entity_candidates}")
    if time is None:
        die(f"Could not detect TIME column. Tried: {time_candidates}")
    return entity, time


def format_stars(pvalue: float) -> str:
    """Significance stars for two-sided p-values."""
    if not np.isfinite(pvalue):
        return ""
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.10:
        return "*"
    return ""


def fmt_num(value: float | None, *, digits: int = 3) -> str:
    """Format numeric output for tables; fallback to N/A for missing/invalid."""
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def log_diff_growth(df: pd.DataFrame, *, entity: str, value_col: str, out_col: str) -> pd.DataFrame:
    """
    Compute within-bank quarterly log-difference growth in percent:
        100 * (log(x_t) - log(x_{t-1}))

    Invalid if:
    - current or lag is missing
    - current or lag is non-positive (log undefined)
    """
    x = coerce_numeric(df[value_col])
    x_lag = x.groupby(df[entity], sort=False).shift(1)
    valid = x.notna() & x_lag.notna() & (x > 0) & (x_lag > 0)
    df[out_col] = np.where(valid, 100.0 * (np.log(x) - np.log(x_lag)), np.nan)
    return df


def run_pooled_ols(df: pd.DataFrame, *, y: str, x: str, entity: str) -> dict[str, object]:
    """
    (1) Pooled OLS:
        y_it = a + b * x_it + e_it
    with standard errors clustered by bank (ENTITY).
    """
    model_df = df[[entity, y, x]].dropna().copy()
    if model_df.empty:
        raise ValueError(f"Empty sample for pooled OLS: {y} ~ {x}")

    y_vec = model_df[y].astype(float).to_numpy()
    x_mat = sm.add_constant(model_df[[x]].astype(float).to_numpy(), has_constant="add")
    res = sm.OLS(y_vec, x_mat).fit(cov_type="cluster", cov_kwds={"groups": model_df[entity]})

    return {
        "coef": float(res.params[1]),
        "se": float(res.bse[1]),
        "pvalue": float(res.pvalues[1]),
        "adj_r2": float(res.rsquared_adj),
        "nobs": int(res.nobs),
    }


def run_panel_fe(
    df: pd.DataFrame,
    *,
    y: str,
    x: str,
    entity: str,
    time: str,
    entity_fe: bool,
    time_fe: bool,
) -> dict[str, object]:
    """
    Fixed effects models using PanelOLS (linearmodels):
    - Time FE: includes time dummies
    - Two-way FE: includes both entity and time dummies
    SE are clustered by entity (bank).
    """
    model_df = df[[entity, time, y, x]].dropna().copy()
    if model_df.empty:
        raise ValueError(f"Empty sample for FE: {y} ~ {x}")

    # PanelOLS expects a MultiIndex [entity, time]
    model_df = model_df.set_index([entity, time])

    mod = PanelOLS(
        model_df[y].astype(float),
        model_df[[x]].astype(float),
        entity_effects=entity_fe,
        time_effects=time_fe,
    )
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    # We compute an "adjusted R2-like" metric (not perfect apples-to-apples vs pooled OLS),
    # and we also keep within R2 if the object provides it.
    denom = float(res.total_ss) / float(res.nobs - 1) if res.nobs and (res.nobs - 1) > 0 else np.nan
    adj_r2 = np.nan
    if np.isfinite(denom) and denom != 0 and np.isfinite(float(res.df_resid)) and float(res.df_resid) > 0:
        adj_r2 = 1.0 - (float(res.resid_ss) / float(res.df_resid)) / denom

    return {
        "coef": float(res.params[x]),
        "se": float(res.std_errors[x]),
        "pvalue": float(res.pvalues[x]),
        "adj_r2": float(adj_r2) if np.isfinite(adj_r2) else np.nan,
        # If available, this is the within R2 (often the most relevant for FE models)
        "r2_within": float(getattr(res, "rsquared_within", np.nan)),
        "nobs": int(res.nobs),
    }


def save_tables_book(results: dict[int, dict[str, object]]) -> None:
    """
    Write a small 3-column table (book-side only) to:
    - LaTeX (booktabs)
    - Markdown
    """
    def coef_cell(k: int) -> str:
        r = results[k]
        return f"{fmt_num(float(r['coef']))}{format_stars(float(r['pvalue']))}"

    def se_cell(k: int) -> str:
        r = results[k]
        return f"({fmt_num(float(r['se']))})"

    def r2_cell(k: int) -> str:
        # For simplicity we keep the same r2 field here as before.
        return fmt_num(float(results[k]["adj_r2"]))

    def n_cell(k: int) -> str:
        return str(int(results[k]["nobs"]))

    tex = "\n".join(
        [
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\caption{Asset Growth and Leverage Growth (Book Leverage)}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r" & (1) & (2) & (3) \\",
            r"\midrule",
            f"Book Leverage Growth & {coef_cell(1)} & {coef_cell(2)} & {coef_cell(3)} \\\\",
            f" & {se_cell(1)} & {se_cell(2)} & {se_cell(3)} \\\\",
            r"\midrule",
            f"Adj. $R^2$ & {r2_cell(1)} & {r2_cell(2)} & {r2_cell(3)} \\\\",
            f"Observations & {n_cell(1)} & {n_cell(2)} & {n_cell(3)} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{flushleft}\footnotesize",
            r"Notes: Robust standard errors clustered at the bank level in parentheses. Significance: *** 1\%, ** 5\%, * 10\%.",
            r"\end{flushleft}",
            r"\end{table}",
            "",
        ]
    )

    md = "\n".join(
        [
            "|  | (1) | (2) | (3) |",
            "| --- | --- | --- | --- |",
            f"| Book Leverage Growth | {coef_cell(1)} | {coef_cell(2)} | {coef_cell(3)} |",
            f"|  | {se_cell(1)} | {se_cell(2)} | {se_cell(3)} |",
            f"| Adj. R2 | {r2_cell(1)} | {r2_cell(2)} | {r2_cell(3)} |",
            f"| N | {n_cell(1)} | {n_cell(2)} | {n_cell(3)} |",
            "",
            "Notes: Robust SEs clustered by bank. Significance: *** 1%, ** 5%, * 10%.",
            "",
        ]
    )

    OUTPUT_TEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEX_PATH.write_text(tex, encoding="utf-8")
    OUTPUT_MD_PATH.write_text(md, encoding="utf-8")


def print_debug_head(df: pd.DataFrame, cols: list[str]) -> None:
    """
    Only used if something fails: print 5 rows for relevant columns.
    Keeps debugging targeted and avoids dumping the full dataset.
    """
    cols_existing = [c for c in cols if c in df.columns]
    if not cols_existing:
        print(df.head(5).to_string(index=False))
        return
    print(df[cols_existing].head(5).to_string(index=False))


def main() -> None:
    # Guardrail: input must exist
    if not ACCOUNTING_PATH.exists():
        die(f"Missing accounting panel file: {ACCOUNTING_PATH}")

    # Load accounting panel
    df = pd.read_csv(ACCOUNTING_PATH)

    # Detect panel keys
    entity, time = detect_entity_time(df)

    # Choose assets/equity/debt columns from available alternatives
    assets_col = "total_assets_2" if "total_assets_2" in df.columns else pick_first_existing(df, ["total_assets", "assets"])
    equity_col = "common_equity_total" if "common_equity_total" in df.columns else pick_first_existing(
        df,
        [
            "common_equity_attributable_to_parent_shareholders",
            "total_shareholders_equity",
            "total_equity",
            "equity_total",
        ],
    )
    debt_col = "debt_total" if "debt_total" in df.columns else pick_first_existing(
        df,
        ["total_debt", "debt_including_finance_and_operating_lease_liabilities", "debt_long_term_total"],
    )

    # Guardrails: assets and equity are required for book leverage
    if assets_col is None:
        die("Missing assets column (tried total_assets_2, total_assets, assets).")
    if equity_col is None:
        die("Missing equity column (expected common_equity_total or similar).")

    # Minimal terminal header
    print(f"Loaded {ACCOUNTING_PATH} shape={df.shape}")
    print(f"ENTITY={entity} TIME={time}")
    print(f"Using: assets={assets_col}, equity={equity_col}, debt={debt_col}")
    if MARKET_PATH.exists():
        print("Market data: found -> not used (book-side only)")
    else:
        print("Market data: missing -> skipping (4)-(6)")

    try:
        # Clean keys and sort panel
        df = df.copy()
        df[entity] = df[entity].astype("string").str.strip()
        df[time] = coerce_time(df[time], label="df")
        df = df.sort_values([entity, time]).reset_index(drop=True)

        # Build core numeric series
        df["assets_used"] = coerce_numeric(df[assets_col])
        df["equity_used"] = coerce_numeric(df[equity_col])
        df["debt_used"] = coerce_numeric(df[debt_col]) if debt_col is not None else np.nan

        # Book leverage = Assets / Equity (only where both are positive)
        df["book_leverage"] = np.where(
            df["assets_used"].notna()
            & df["equity_used"].notna()
            & (df["assets_used"] > 0)
            & (df["equity_used"] > 0),
            df["assets_used"] / df["equity_used"],
            np.nan,
        )

        # Growth rates as quarterly log-differences (in percent)
        df = log_diff_growth(df, entity=entity, value_col="assets_used", out_col="asset_growth")
        df = log_diff_growth(df, entity=entity, value_col="book_leverage", out_col="book_lev_growth")

        # Save the constructed dataset (useful for checking and reuse)
        OUTPUT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_DATASET_PATH, index=False)

        # Regression sample (drop rows without both growth rates)
        base = df[[entity, time, "asset_growth", "book_lev_growth"]].dropna().copy()
        if base.empty:
            raise ValueError("No valid observations after constructing growth rates.")

        # Run the three book-side regressions
        results: dict[int, dict[str, object]] = {}
        results[1] = run_pooled_ols(base, y="asset_growth", x="book_lev_growth", entity=entity)
        results[2] = run_panel_fe(base, y="asset_growth", x="book_lev_growth", entity=entity, time=time, entity_fe=False, time_fe=True)
        results[3] = run_panel_fe(base, y="asset_growth", x="book_lev_growth", entity=entity, time=time, entity_fe=True, time_fe=True)

        # Save LaTeX + Markdown tables
        save_tables_book(results)

        # Compact, explicit terminal output (self-explanatory)
        print("\nDependent variable: Asset growth (Δ log total assets, quarterly, %)")
        print("Independent variable: Book leverage growth (Δ log assets / equity, quarterly, %)")
        print("\nModels:")
        print("(1) Pooled OLS")
        print("(2) Time fixed effects")
        print("(3) Bank + time fixed effects")

        print("\nRESULTS:")
        for k in (1, 2, 3):
            r = results[k]
            coef = float(r["coef"])
            se = float(r["se"])
            tstat = coef / se if np.isfinite(coef) and np.isfinite(se) and se != 0 else np.nan
            stars = format_stars(float(r["pvalue"]))
            print(f"({k}) coef = {coef:.3f}{stars}, se = {se:.3f}, t = {tstat:.2f}")

        # Report R2 consistently: pooled uses adj R2, FE prefers within R2 if available
        r2_1 = float(results[1]["adj_r2"])
        r2_2 = float(results[2].get("r2_within", np.nan))
        r2_3 = float(results[3].get("r2_within", np.nan))
        if not np.isfinite(r2_2):
            r2_2 = float(results[2]["adj_r2"])
        if not np.isfinite(r2_3):
            r2_3 = float(results[3]["adj_r2"])

        print(f"R2 (adj for pooled; within for FE): {r2_1:.3f} {r2_2:.3f} {r2_3:.3f}")
        print(f"N (observations): {int(results[1]['nobs'])} {int(results[2]['nobs'])} {int(results[3]['nobs'])}")

    except Exception as e:
        # If anything fails, print a minimal debug preview (5 rows) and re-raise.
        print(f"ERROR: {type(e).__name__}: {e}")
        print("Debug preview (5 rows, relevant columns):")
        print_debug_head(
            df,
            [
                entity,
                time,
                assets_col,
                equity_col,
                debt_col if debt_col is not None else "",
                "assets_used",
                "equity_used",
                "book_leverage",
                "asset_growth",
                "book_lev_growth",
            ],
        )
        raise


if __name__ == "__main__":
    main()
