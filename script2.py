import pandas as pd

df = pd.read_csv("data.csv")

# Infos de base
print(" Shape:", df.shape)
print("\n Types de colonnes:\n", df.dtypes)
print("\n Valeurs manquantes:\n", df.isnull().sum())
print("\n Statistiques numériques:\n", df.describe())

for col in df.columns:
    print(f"\n Colonne '{col}' — {df[col].nunique()} valeurs uniques")

print("\n Matrice de corrélation :\n", df.corr(numeric_only=True))
