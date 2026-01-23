"""
Validate VaR conversion methods specifically on Bank of America.

This script focuses on validating the two conversion methods by:
1. Taking Bank of America's var_95 values
2. Converting them using both methods (Gaussian 1.41 and BoA Factor 2.0)
3. Comparing predictions against actual var_99 values

Creates comprehensive visualizations showing:
- Prediction accuracy for both methods
- Time series comparison
- Error analysis
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# =============================================================================
# Style
# =============================================================================
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

# =============================================================================
# Conversion factors
# =============================================================================
GAUSSIAN_RATIO = 1.4144   # z_0.99 / z_0.95
BOA_FACTOR = 2.0

# =============================================================================
# Paths 
# =============================================================================
INPUT_FILE = Path("output/data/merged_with_var_99_dual_methods.csv")
OUTPUT_DIR = Path("output/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Load data
# =============================================================================
print(f"Loading data from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
df["period_end_date"] = pd.to_datetime(df["period_end_date"])

# Filter to Bank of America with both var_95 and var_99
df_boa = df[df["bank_id"] == "bank_of_america"].copy()
df_boa_validation = df_boa[df_boa["var_95"].notna() & df_boa["var_99"].notna()].copy()

print(f"\nBank of America total rows: {len(df_boa)}")
print(f"Bank of America rows with BOTH var_95 and var_99: {len(df_boa_validation)}")

if df_boa_validation.empty:
    raise RuntimeError("No Bank of America observations with both var_95 and var_99.")

# =============================================================================
# Predictions
# =============================================================================
df_boa_validation["predicted_gaussian"] = df_boa_validation["var_95"] * GAUSSIAN_RATIO
df_boa_validation["predicted_boa"] = df_boa_validation["var_95"] * BOA_FACTOR

# Errors
df_boa_validation["error_gaussian"] = (
    df_boa_validation["predicted_gaussian"] - df_boa_validation["var_99"]
) / df_boa_validation["var_99"]

df_boa_validation["error_boa"] = (
    df_boa_validation["predicted_boa"] - df_boa_validation["var_99"]
) / df_boa_validation["var_99"]

df_boa_validation = df_boa_validation.sort_values("period_end_date")

# =============================================================================
# Figure
# =============================================================================
print("\nCreating comprehensive validation figure...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# -----------------------------------------------------------------------------
# (1) Time series
# -----------------------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, :])

ax1.plot(
    df_boa_validation["period_end_date"],
    df_boa_validation["var_99"],
    marker="o",
    linewidth=3,
    label="Actual VaR 99%",
    color="black",
)

ax1.plot(
    df_boa_validation["period_end_date"],
    df_boa_validation["predicted_gaussian"],
    marker="s",
    linestyle="--",
    linewidth=2,
    label=f"Gaussian (× {GAUSSIAN_RATIO:.4f})",
    color="steelblue",
)

ax1.plot(
    df_boa_validation["period_end_date"],
    df_boa_validation["predicted_boa"],
    marker="^",
    linestyle="--",
    linewidth=2,
    label=f"BoA factor (× {BOA_FACTOR:.1f})",
    color="coral",
)

ax1.set_title("Bank of America: VaR 99% – Actual vs Predicted", fontweight="bold")
ax1.set_ylabel("VaR 99%")
ax1.legend(frameon=False)
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

# -----------------------------------------------------------------------------
# (2) Scatter – Gaussian
# -----------------------------------------------------------------------------
from sklearn.metrics import r2_score, mean_squared_error

ax2 = fig.add_subplot(gs[1, 0])

ax2.scatter(
    df_boa_validation["var_99"],
    df_boa_validation["predicted_gaussian"],
    s=80,
    alpha=0.7,
    color="steelblue",
    edgecolors="navy",
)

min_v = df_boa_validation[["var_99", "predicted_gaussian"]].min().min()
max_v = df_boa_validation[["var_99", "predicted_gaussian"]].max().max()
ax2.plot([min_v, max_v], [min_v, max_v], "k--")

r2_g = r2_score(df_boa_validation["var_99"], df_boa_validation["predicted_gaussian"])
rmse_g = np.sqrt(mean_squared_error(df_boa_validation["var_99"], df_boa_validation["predicted_gaussian"]))

ax2.set_title("Gaussian method")
ax2.set_xlabel("Actual VaR 99%")
ax2.set_ylabel("Predicted VaR 99%")
ax2.text(
    0.05,
    0.95,
    f"R² = {r2_g:.3f}\nRMSE = {rmse_g:.2f}",
    transform=ax2.transAxes,
    va="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

# -----------------------------------------------------------------------------
# (3) Scatter – BoA factor
# -----------------------------------------------------------------------------
ax3 = fig.add_subplot(gs[1, 1])

ax3.scatter(
    df_boa_validation["var_99"],
    df_boa_validation["predicted_boa"],
    s=80,
    alpha=0.7,
    color="coral",
    edgecolors="darkred",
)

min_v = df_boa_validation[["var_99", "predicted_boa"]].min().min()
max_v = df_boa_validation[["var_99", "predicted_boa"]].max().max()
ax3.plot([min_v, max_v], [min_v, max_v], "k--")

r2_b = r2_score(df_boa_validation["var_99"], df_boa_validation["predicted_boa"])
rmse_b = np.sqrt(mean_squared_error(df_boa_validation["var_99"], df_boa_validation["predicted_boa"]))

ax3.set_title("BoA factor method")
ax3.set_xlabel("Actual VaR 99%")
ax3.set_ylabel("Predicted VaR 99%")
ax3.text(
    0.05,
    0.95,
    f"R² = {r2_b:.3f}\nRMSE = {rmse_b:.2f}",
    transform=ax3.transAxes,
    va="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

# -----------------------------------------------------------------------------
# (4) Error distributions
# -----------------------------------------------------------------------------
ax4 = fig.add_subplot(gs[2, :])

ax4.hist(df_boa_validation["error_gaussian"] * 100, bins=20, alpha=0.6, label="Gaussian")
ax4.hist(df_boa_validation["error_boa"] * 100, bins=20, alpha=0.6, label="BoA factor")
ax4.axvline(0, color="black", linestyle="--")

ax4.set_title("Percentage prediction error")
ax4.set_xlabel("(Predicted − Actual) / Actual (%)")
ax4.legend(frameon=False)

# =============================================================================
# Save
# =============================================================================
out_path = OUTPUT_DIR / "boa_validation_comprehensive.png"
plt.tight_layout()
plt.savefig(out_path, bbox_inches="tight")
plt.close()

print(f"Saved: {out_path}")
print("Validation complete.")
