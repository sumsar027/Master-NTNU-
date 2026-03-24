import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize

# ── Load & sort data ──────────────────────────────────────────────────────────
df = pd.read_csv('dataframe/dataframe.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['bank', 'date']).dropna(subset=['total_assets', 'repo', 'total_equity'])

# ── Figure 1: Scatter  Δlog(Assets) vs Δlog(Repo) ────────────────────────────
df['d_log_assets'] = df.groupby('bank')['total_assets'].transform(lambda x: np.log(x).diff())
df['d_log_repo']   = df.groupby('bank')['repo'].transform(lambda x: np.log(x).diff())

scatter_df = df.dropna(subset=['d_log_assets', 'd_log_repo']).copy()

# Winsorize at 1% and 99%
scatter_df['d_log_assets'] = winsorize(scatter_df['d_log_assets'], limits=[0.01, 0.01])
scatter_df['d_log_repo']   = winsorize(scatter_df['d_log_repo'],   limits=[0.01, 0.01])



# ── Plotting ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(figsize=(13, 5))

# --- Left: Scatter ---

x = scatter_df['d_log_assets'].values
y = scatter_df['d_log_repo'].values

axes.scatter(x, y, color="#11131e", alpha=0.5, s=20, edgecolors='none')

m, b = np.polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 200)
axes.plot(x_line, m * x_line + b, color='red', linewidth=2.5,
         linestyle='--', label=f'OLS (β = {m:.2f})')

axes.axhline(0, color='grey', linewidth=0.7, linestyle=':')
axes.axvline(0, color='grey', linewidth=0.7, linestyle=':')
axes.set_xlabel('Δ log(Assets)')
axes.set_ylabel('Δ log(Repo)')
axes.set_title('Δlog(Assets) vs Δlog(Repo)\n(winsorized 1–99%)')
axes.legend()
axes.spines[['top', 'right']].set_visible(False)



plt.tight_layout()
plt.savefig('output/figures/repo_assets_figures.png', dpi=160, bbox_inches='tight')
print("Saved: output/figures/repo_assets_figures.png")