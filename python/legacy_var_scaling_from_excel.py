"""
Bank of America VaR Level Check (99% vs 95%)

Beregner forholdet mellom VaR 99% og VaR 95% for Bank of America
for å validere skaleringsmetoden brukt i masteroppgaven.
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Filstier
DATA_PATH = Path("data/raw/VaR_python.xlsx")
OUTPUT_DIR = Path("output/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Les data
df = pd.read_excel(DATA_PATH, sheet_name="Ark1")

# Velg relevante kolonner for Bank of America
boa_data = df[['year', 'Bank_of_America_95%', 'Bank_of_America_99%']].copy()
boa_data.columns = ['year', 'var_95', 'var_99']

# Fjern manglende verdier
boa_data = boa_data.dropna()

# Beregn ratio
boa_data['ratio'] = boa_data['var_99'] / boa_data['var_95']

# Statistikk
print("=" * 60)
print("Bank of America: VaR 99% / VaR 95% Ratio")
print("=" * 60)
print(f"\nAntall observasjoner: {len(boa_data)}")
print(f"Gjennomsnitt: {boa_data['ratio'].mean():.4f}")
print(f"Median: {boa_data['ratio'].median():.4f}")
print(f"Std.avvik: {boa_data['ratio'].std():.4f}")
print(f"Min: {boa_data['ratio'].min():.4f}")
print(f"Max: {boa_data['ratio'].max():.4f}")

print("\nFørste 10 observasjoner:")
print(boa_data.head(10).to_string(index=False))

# Plot
plt.figure(figsize=(12, 6))
plt.plot(boa_data['year'], boa_data['ratio'], marker='o', linewidth=1.5)
plt.axhline(
    boa_data['ratio'].median(), 
    color='red', 
    linestyle='--', 
    linewidth=2,
    label=f"Median = {boa_data['ratio'].median():.4f}"
)
plt.xlabel("Year")
plt.ylabel("VaR 99% / VaR 95%")
plt.title("Bank of America: VaR 99% / VaR 95% Ratio Over Time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

# Lagre plot
plot_path = OUTPUT_DIR / "bank_of_america_var_ratio.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"\nPlot lagret: {plot_path}")

# Lagre CSV
csv_path = OUTPUT_DIR / "bank_of_america_var_ratio.csv"
boa_data.to_csv(csv_path, index=False)
print(f"CSV lagret: {csv_path}")

print("\n" + "=" * 60)
print("Analyse fullført!")
print("=" * 60)