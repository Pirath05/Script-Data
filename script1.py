import pandas as pd

df = pd.read_csv('data.csv')

print("Aperçu des données :")
print(df.head())

print("\nInfos générales :")
print(df.info())

df_clean = df.dropna()

print("\nStatistiques descriptives :")
print(df_clean.describe())

if 'Categorie' in df_clean.columns:
    grouped = df_clean.groupby('Categorie').mean()
    print("\nMoyennes par catégorie :")
    print(grouped)
else:
    print("\nColonne 'Categorie' non trouvée.")

# Exporter le DataFrame nettoyé
df_clean.to_csv('data_clean.csv', index=False)
print("\nFichier nettoyé exporté sous 'data_clean.csv'")