import pandas as pd

df = pd.read_csv('data.csv')

print(df.head())
print("\nForme du dataset :", df.shape)
print("\nTypes de données :\n", df.dtypes)

print("\nValeurs manquantes :\n", df.isnull().sum())

print("\nStatistiques descriptives :\n", df.describe())

print("\nCorrélation :\n", df.corr(numeric_only=True))

if 'Categorie' in df.columns:
    print("\nTop catégories :\n", df['Categorie'].value_counts().head())
